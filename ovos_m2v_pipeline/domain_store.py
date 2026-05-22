"""Domain-aware prototype store for parallel intent matching.

Mirrors the API shipped by sibling OVOS intent plugins
(``nebulento.DomainIntentContainer``, ``ovos_padatious.DomainIntentContainer``,
``palavreado.DomainIntentContainer``, ``padacioso.DomainIntentContainer``,
``linha_fina.DomainIntentEngine``, ``ovos_markov_pipeline.DomainMarkovIntentEngine``):
intents are grouped into *domains*, but — following adapt's
``DomainIntentDeterminationEngine`` — there is **no top-level router**.
Every domain's sub-store scores the query in parallel and the global
argmax over the union of per-intent scores wins.

For the prototype paradigm specifically, this gives:

1. **One comparable scale.** Every sub-store is a
   :class:`PrototypeIntentStore` configured with the same
   :class:`PrototypeStrategy`, so per-intent cosines from different
   domains are directly comparable.
2. **Independent domain configuration.** Each per-domain sub-store keeps
   its own anchors + strategy + temperature; no router-level strategy
   needs to be tuned.
3. **Optional ``top_k_domains`` pruning.** When set, a per-domain
   *fingerprint* embedding (concatenated samples → store anchors) is
   scored first and only the top-K domains' sub-stores are evaluated.
   Defaults to ``None`` (score all domains).

No training required — the static encoder produces per-domain intent
embeddings (and, when ``top_k_domains`` is enabled, the domain
fingerprints too).
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy


class DomainPrototypeIntentStore:
    """Parallel-argmax prototype store grouped by domain.

    Intents are grouped into *domains*. At query time every domain's
    sub-store scores the query independently; the caller takes the
    argmax over the flat union of per-intent scores. There is no
    top-level router — routing is implicit in the global argmax.

    Domains can also be selected explicitly via ``scores(query,
    domain=...)`` to restrict scoring to a single sub-store.

    Optional optimisation: set ``top_k_domains`` to score per-domain
    *fingerprint* embeddings first and evaluate only the top-K
    domains' sub-stores. Defaults to ``None`` (score all domains).

    Example::

        from model2vec import StaticModel
        from ovos_m2v_pipeline import DomainPrototypeIntentStore
        from ovos_m2v_pipeline.strategies import PrototypeStrategy

        model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
        store = DomainPrototypeIntentStore(
            intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
            intent_tau=0.1,
        )

        store.add(model, "media", "play",      ["play {song}", "put on {song}"])
        store.add(model, "media", "pause",     ["pause", "pause the music"])
        store.add(model, "home",  "lights_on", ["turn on the lights", "lights on"])

        scores = store.scores(model.encode(["play africa"])[0])
        # scores == {"play": 0.93, "pause": 0.21, "lights_on": 0.05}
    """

    def __init__(
        self,
        *,
        intent_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        intent_top_k: int = 3,
        intent_tau: float = 0.1,
        top_k_domains: Optional[int] = None,
        domain_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        domain_tau: float = 0.1,
    ) -> None:
        self._intent_strategy = PrototypeStrategy(intent_strategy)
        self._intent_top_k = intent_top_k
        self._intent_tau = intent_tau
        #: Optional pruning: if set, score only this many domains by
        #: per-domain fingerprint similarity before flattening intents.
        self._top_k_domains = top_k_domains
        self._domain_strategy = PrototypeStrategy(domain_strategy)
        self._domain_tau = domain_tau

        #: Per-domain intent stores, keyed by domain name.
        self.domains: Dict[str, PrototypeIntentStore] = {}
        #: Raw training samples per (domain, intent) — for inspection,
        #: persistence-bypass paths, and the domain-fingerprint rebuild.
        self._samples: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        #: Optional per-domain fingerprint store, only populated when
        #: ``top_k_domains`` is set. Otherwise stays empty.
        self._domain_fingerprints: Optional[PrototypeIntentStore] = (
            PrototypeIntentStore(
                strategy=self._domain_strategy,
                top_k=intent_top_k,
                tau=self._domain_tau,
            )
            if top_k_domains is not None
            else None
        )

    # ------------------------------------------------------------------
    # Read-only views (parity with PrototypeIntentStore where it makes sense)
    # ------------------------------------------------------------------

    @property
    def intent_strategy(self) -> PrototypeStrategy:
        return self._intent_strategy

    @property
    def top_k_domains(self) -> Optional[int]:
        return self._top_k_domains

    def __len__(self) -> int:
        """Total number of prototypes across every domain's sub-store."""
        return sum(len(s) for s in self.domains.values())

    @property
    def unique_labels(self) -> np.ndarray:
        """All intent labels across every domain (sorted, deduplicated)."""
        labels: set = set()
        for store in self.domains.values():
            labels.update(str(l) for l in store.unique_labels)
        return np.asarray(sorted(labels), dtype=object)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, model, domain: str, label: str, sentences: List[str],
            k: int = 5, random_state: int = 42) -> int:
        """Register a domain-scoped intent ``label`` with example sentences.

        Args:
            model: model2vec encoder.
            domain: Domain name (created on first use).
            label: Intent label, unique within the domain.
            sentences: Training utterances.
            k: Max prototypes per intent (per the parent store's contract).
            random_state: Forwarded to the parent store for sample choice.

        Returns:
            Number of prototypes added.
        """
        if domain not in self.domains:
            self.domains[domain] = PrototypeIntentStore(
                strategy=self._intent_strategy,
                top_k=self._intent_top_k,
                tau=self._intent_tau,
            )
        n = self.domains[domain].add(model, label, sentences,
                                      k=k, random_state=random_state)
        self._samples[domain][label] = list(sentences)
        if self._domain_fingerprints is not None:
            self._rebuild_domain_fingerprint(model, domain,
                                             k=k, random_state=random_state)
        return n

    def remove(self, domain: str, label: str) -> None:
        """Remove an intent from a domain."""
        if domain in self.domains:
            self.domains[domain].remove(label)
        self._samples[domain].pop(label, None)

    def remove_domain(self, domain: str) -> None:
        """Remove a domain and all its intents."""
        self.domains.pop(domain, None)
        self._samples.pop(domain, None)
        if self._domain_fingerprints is not None:
            self._domain_fingerprints.remove(domain)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def scores(self, query_embedding: np.ndarray,
               domain: Optional[str] = None) -> Dict[str, float]:
        """Return ``{intent_label: score}`` across (selected) domains.

        Args:
            query_embedding: Raw (unnormalised) query embedding vector.
            domain: If given, only score inside this domain. Otherwise
                every domain's sub-store is scored in parallel and the
                results are flattened into one ``{label: score}`` dict —
                the caller takes argmax.
        """
        if domain is not None:
            if domain not in self.domains:
                return {}
            return self.domains[domain].scores(query_embedding)

        if not self.domains:
            return {}

        candidate_domains = self._candidate_domains(query_embedding)

        flat: Dict[str, float] = {}
        for dname in candidate_domains:
            sub = self.domains.get(dname)
            if sub is None:
                continue
            for label, score in sub.scores(query_embedding).items():
                # Same label in two domains shouldn't happen with the
                # ``skill_id:intent`` convention, but if it does keep the
                # higher score so the argmax stays well-defined.
                prev = flat.get(label)
                if prev is None or score > prev:
                    flat[label] = score
        return flat

    def calc_intent(self, query_embedding: np.ndarray,
                    domain: Optional[str] = None) -> Optional[str]:
        """Convenience: argmax over :meth:`scores`. Returns ``None`` if empty."""
        scored = self.scores(query_embedding, domain=domain)
        if not scored:
            return None
        return max(scored, key=scored.get)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _candidate_domains(self, query_embedding: np.ndarray) -> List[str]:
        """Return the domains to evaluate for a given query.

        Without ``top_k_domains`` this is every known domain. With
        ``top_k_domains`` set, the per-domain fingerprint store is
        scored first and only the top-K names are returned.
        """
        if self._domain_fingerprints is None or self._top_k_domains is None:
            return list(self.domains.keys())
        fp_scores = self._domain_fingerprints.scores(query_embedding)
        if not fp_scores:
            return list(self.domains.keys())
        ranked = sorted(fp_scores.items(), key=lambda kv: kv[1], reverse=True)
        return [name for name, _ in ranked[: self._top_k_domains]]

    def _rebuild_domain_fingerprint(self, model, domain: str, k: int,
                                     random_state: int) -> None:
        """Rebuild the *domain*'s fingerprint embedding from its samples.

        Only invoked when ``top_k_domains`` is set. Concatenates every
        intent's samples in the domain; the fingerprint store's own
        strategy decides which subset to keep as anchors.
        """
        if self._domain_fingerprints is None:
            return
        all_samples: List[str] = []
        for sents in self._samples.get(domain, {}).values():
            all_samples.extend(sents)
        if not all_samples:
            return
        self._domain_fingerprints.remove(domain)
        self._domain_fingerprints.add(model, domain, all_samples,
                                       k=k, random_state=random_state)


class HierarchicalPrototypeIntentStore(DomainPrototypeIntentStore):
    """Two-stage (hierarchical) prototype store grouped by domain.

    Intents are grouped into *domains*, but — unlike
    :class:`DomainPrototypeIntentStore` — there **is** a top-level
    router. At query time a single best domain is selected first, then
    intent resolution happens only inside that domain's sub-store.

    The router is the per-domain *fingerprint* store
    (:attr:`_domain_fingerprints`): a concatenated-sample embedding per
    domain. :meth:`calc_domain` picks the single highest-scoring domain
    by fingerprint cosine similarity. Unlike the optional
    ``top_k_domains`` prune on :class:`DomainPrototypeIntentStore`, the
    fingerprint store here is **mandatory** — it is always built.

    A ``domain_threshold`` gate rejects off-topic queries: when the best
    domain's fingerprint score is below the threshold, scoring returns
    no match. ``0.0`` (default) disables the gate.

    Example::

        from model2vec import StaticModel
        from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore
        from ovos_m2v_pipeline.strategies import PrototypeStrategy

        model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
        store = HierarchicalPrototypeIntentStore(
            intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
            intent_tau=0.1,
            domain_threshold=0.2,
        )

        store.add(model, "media", "play",      ["play {song}", "put on {song}"])
        store.add(model, "media", "pause",     ["pause", "pause the music"])
        store.add(model, "home",  "lights_on", ["turn on the lights", "lights on"])

        scores = store.scores(model.encode(["play africa"])[0])
        # only the winning domain's intents are scored
    """

    def __init__(
        self,
        *,
        intent_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        intent_top_k: int = 3,
        intent_tau: float = 0.1,
        domain_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        domain_tau: float = 0.1,
        domain_threshold: float = 0.0,
    ) -> None:
        # The fingerprint store is mandatory in the hierarchical variant —
        # force top_k_domains=1 so the parent always builds it.
        super().__init__(
            intent_strategy=intent_strategy,
            intent_top_k=intent_top_k,
            intent_tau=intent_tau,
            top_k_domains=1,
            domain_strategy=domain_strategy,
            domain_tau=domain_tau,
        )
        #: Minimum fingerprint score the winning domain must reach for a
        #: query to be routed at all. Below it, scoring returns no match.
        self._domain_threshold = domain_threshold

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def domain_threshold(self) -> float:
        return self._domain_threshold

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calc_domain(self, query_embedding: np.ndarray) -> Optional[str]:
        """Return the single best-matching domain name for a query.

        The per-domain fingerprint store is scored and the highest
        scoring domain is returned. When the best score is below
        :attr:`domain_threshold`, or no domains are registered,
        ``None`` is returned.
        """
        if self._domain_fingerprints is None:
            return None
        fp_scores = self._domain_fingerprints.scores(query_embedding)
        if not fp_scores:
            return None
        domain, score = max(fp_scores.items(), key=lambda kv: kv[1])
        if score < self._domain_threshold:
            return None
        return domain

    def scores(self, query_embedding: np.ndarray,
               domain: Optional[str] = None) -> Dict[str, float]:
        """Return ``{intent_label: score}`` for the routed domain only.

        Args:
            query_embedding: Raw (unnormalised) query embedding vector.
            domain: If given, skip the top-level router and score only
                this domain (bypasses the ``domain_threshold`` gate).
                Otherwise :meth:`calc_domain` picks one domain and only
                that domain's sub-store is scored.
        """
        if domain is not None:
            if domain not in self.domains:
                return {}
            return self.domains[domain].scores(query_embedding)

        if not self.domains:
            return {}

        routed = self.calc_domain(query_embedding)
        if routed is None:
            return {}
        sub = self.domains.get(routed)
        if sub is None:
            return {}
        return sub.scores(query_embedding)
