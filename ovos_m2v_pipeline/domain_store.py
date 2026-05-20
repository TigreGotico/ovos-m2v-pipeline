"""Domain-aware prototype store for hierarchical intent matching.

Mirrors the API shipped by sibling OVOS intent plugins
(``nebulento.DomainIntentContainer``, ``ovos_padatious.DomainIntentContainer``,
``palavreado.DomainIntentContainer``, ``padacioso.DomainIntentContainer``,
``linha_fina.DomainIntentEngine``, ``ovos_markov_pipeline.DomainMarkovIntentEngine``):
intents are grouped into *domains*, a top-level prototype store first
picks the domain, and the domain's sub-store resolves the intent.

For the prototype paradigm specifically, hierarchical matching gives two
benefits:

1. **Tighter local cosine distributions.** A domain's prototypes share a
   subspace (lights/thermostat/door all live near "smarthome"), so the
   score across just that domain's prototypes is a sharper signal than
   the global score over all intents.
2. **Lower far-OOD false-positive rate.** The top-level classifier
   rejects chitchat that doesn't strongly match any domain *before* any
   sub-store sees it.

Strategy support
----------------
Each level — the top-level domain router and every per-domain intent
store — uses a :class:`PrototypeIntentStore` configured with a
:class:`PrototypeStrategy`. The strategy can be set independently for
the two levels (``domain_strategy`` vs ``intent_strategy``) because the
two have different shapes: the router compares against many concatenated
in-domain samples (max-cosine usually wins), while a per-domain store
sees a small handful of per-intent samples (centroid / top-k_mean / softmax
often help).

No training required — the static encoder produces both the top-level
domain embeddings and the per-domain intent embeddings.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy


class DomainPrototypeIntentStore:
    """Two-level prototype store: domain classification then intent matching.

    Intents are grouped into *domains*. At query time the engine first
    selects the most likely domain via :attr:`domain_store`, then runs
    the domain-specific :class:`PrototypeIntentStore` to find the best
    intent within that domain.

    Domains can also be selected explicitly, bypassing the top-level
    classifier.

    Strategy / temperature / top_k can be set independently for the two
    levels: ``domain_*`` controls the router, ``intent_*`` controls every
    per-domain sub-store. Both default to :class:`PrototypeStrategy.MAX_OVER_ALL`
    to preserve the pre-strategy hierarchical scoring.

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
        # scores == {"play": 0.93, ...}  (best label within resolved domain)
    """

    def __init__(
        self,
        *,
        domain_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        domain_top_k: int = 3,
        domain_tau: float = 0.1,
        intent_strategy: PrototypeStrategy = PrototypeStrategy.MAX_OVER_ALL,
        intent_top_k: int = 3,
        intent_tau: float = 0.1,
    ) -> None:
        self._domain_strategy = PrototypeStrategy(domain_strategy)
        self._domain_top_k = domain_top_k
        self._domain_tau = domain_tau
        self._intent_strategy = PrototypeStrategy(intent_strategy)
        self._intent_top_k = intent_top_k
        self._intent_tau = intent_tau

        #: Top-level prototype store mapping query → domain.
        self.domain_store: PrototypeIntentStore = PrototypeIntentStore(
            strategy=self._domain_strategy,
            top_k=self._domain_top_k,
            tau=self._domain_tau,
        )
        #: Per-domain intent stores, keyed by domain name.
        self.domains: Dict[str, PrototypeIntentStore] = {}
        #: Raw training samples per (domain, intent) — for inspection,
        #: persistence-bypass paths, and the auto domain-classifier seed.
        self._samples: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Read-only views (parity with PrototypeIntentStore where it makes sense)
    # ------------------------------------------------------------------

    @property
    def domain_strategy(self) -> PrototypeStrategy:
        return self._domain_strategy

    @property
    def intent_strategy(self) -> PrototypeStrategy:
        return self._intent_strategy

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

        Adds the prototypes to the domain's sub-store AND mirrors them
        under the domain name into :attr:`domain_store` so the top-level
        classifier learns the domain's surface forms incrementally.

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
        self._rebuild_domain_proto(model, domain, k=k, random_state=random_state)
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
        self.domain_store.remove(domain)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def calc_domain(self, query_embedding: np.ndarray) -> Optional[str]:
        """Return the best matching domain name, or ``None`` if empty."""
        scores = self.domain_store.scores(query_embedding)
        if not scores:
            return None
        return max(scores, key=scores.get)

    def scores(self, query_embedding: np.ndarray,
               domain: Optional[str] = None) -> Dict[str, float]:
        """Return label → score inside the resolved (or specified) domain.

        Args:
            query_embedding: Raw (unnormalised) query embedding vector.
            domain: If given, skip the top-level classifier and score
                inside this domain directly.
        """
        resolved = domain if domain is not None else self.calc_domain(query_embedding)
        if resolved is None or resolved not in self.domains:
            return {}
        return self.domains[resolved].scores(query_embedding)

    # ------------------------------------------------------------------
    # Internal: keep domain_store synced with sub-store contents
    # ------------------------------------------------------------------

    def _rebuild_domain_proto(self, model, domain: str, k: int,
                                random_state: int) -> None:
        """Rebuild the *domain*'s entry in :attr:`domain_store` from the
        current set of in-domain samples.

        Concatenates every intent's samples; the router's own strategy
        decides which subset to keep as the domain's anchors.
        """
        all_samples: List[str] = []
        for sents in self._samples.get(domain, {}).values():
            all_samples.extend(sents)
        if not all_samples:
            return
        self.domain_store.remove(domain)
        self.domain_store.add(model, domain, all_samples,
                              k=k, random_state=random_state)
