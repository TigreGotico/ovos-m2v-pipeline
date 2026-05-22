"""Two-stage (hierarchical) prototype store for domain-routed intent matching.

Intents are grouped into *domains*. At query time a top-level router
selects the single best-matching domain, then intent resolution happens
only inside that domain's sub-store. This mirrors nebulento's standalone
:class:`~nebulento.HierarchicalIntentContainer`: the store owns its own
``domains`` dict of per-domain sub-stores plus a domain-fingerprint
classifier used as the router.

No training required — the static encoder produces both the per-domain
intent embeddings and the domain fingerprints.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy


class HierarchicalPrototypeIntentStore:
    """Two-stage (hierarchical) prototype store grouped by domain.

    Intents are grouped into *domains*. At query time a top-level router
    selects a single best domain first, then intent resolution happens
    only inside that domain's sub-store.

    The router is a per-domain *fingerprint* store
    (:attr:`_domain_fingerprints`): a concatenated-sample embedding per
    domain. :meth:`calc_domain` picks the single highest-scoring domain
    by fingerprint cosine similarity.

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
        self._intent_strategy = PrototypeStrategy(intent_strategy)
        self._intent_top_k = intent_top_k
        self._intent_tau = intent_tau
        self._domain_strategy = PrototypeStrategy(domain_strategy)
        self._domain_tau = domain_tau
        #: Minimum fingerprint score the winning domain must reach for a
        #: query to be routed at all. Below it, scoring returns no match.
        self._domain_threshold = domain_threshold

        #: Per-domain intent stores, keyed by domain name.
        self.domains: Dict[str, PrototypeIntentStore] = {}
        #: Raw training samples per (domain, intent) — for inspection and
        #: the domain-fingerprint rebuild.
        self._samples: Dict[str, Dict[str, List[str]]] = defaultdict(dict)
        #: Top-level router: a per-domain fingerprint store mapping
        #: concatenated-sample embeddings to domain names.
        self._domain_fingerprints: PrototypeIntentStore = PrototypeIntentStore(
            strategy=self._domain_strategy,
            top_k=intent_top_k,
            tau=self._domain_tau,
        )

    # ------------------------------------------------------------------
    # Read-only views
    # ------------------------------------------------------------------

    @property
    def intent_strategy(self) -> PrototypeStrategy:
        return self._intent_strategy

    @property
    def domain_threshold(self) -> float:
        return self._domain_threshold

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
            k: Max prototypes per intent (per the sub-store's contract).
            random_state: Forwarded to the sub-store for sample choice.

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
        self._domain_fingerprints.remove(domain)

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

    def _rebuild_domain_fingerprint(self, model, domain: str, k: int,
                                    random_state: int) -> None:
        """Rebuild the *domain*'s fingerprint embedding from its samples.

        Concatenates every intent's samples in the domain; the
        fingerprint store's own strategy decides which subset to keep
        as anchors.
        """
        all_samples: List[str] = []
        for sents in self._samples.get(domain, {}).values():
            all_samples.extend(sents)
        if not all_samples:
            return
        self._domain_fingerprints.remove(domain)
        self._domain_fingerprints.add(model, domain, all_samples,
                                      k=k, random_state=random_state)
