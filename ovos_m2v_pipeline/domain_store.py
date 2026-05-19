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
   max-cosine score across just that domain's prototypes is a sharper
   signal than the global max-cosine over 50 intents.
2. **Lower far-OOD false-positive rate.** The top-level classifier
   rejects chitchat that doesn't strongly match any domain *before* any
   sub-store sees it.

No training required — the existing static encoder produces both the
top-level domain embeddings and the per-domain intent embeddings.
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore


class DomainPrototypeIntentStore:
    """Two-level prototype store: domain classification then intent matching.

    Intents are grouped into *domains*. At query time the engine first
    selects the most likely domain via :attr:`domain_store`, then runs
    the domain-specific :class:`PrototypeIntentStore` to find the best
    intent within that domain.

    Domains can also be selected explicitly, bypassing the top-level
    classifier.

    Example::

        from model2vec import StaticModel
        from ovos_m2v_pipeline import DomainPrototypeIntentStore

        model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
        store = DomainPrototypeIntentStore()

        store.add(model, "media", "play",       ["play {song}", "put on {song}"])
        store.add(model, "media", "pause",      ["pause", "pause the music"])
        store.add(model, "home",  "lights_on",  ["turn on the lights", "lights on"])

        scores = store.scores(model.encode(["play africa"])[0])
        # scores == {"play": 0.93, ...}  (best label within resolved domain)
    """

    def __init__(self) -> None:
        #: Top-level prototype store mapping query → domain.
        self.domain_store: PrototypeIntentStore = PrototypeIntentStore()
        #: Per-domain intent stores, keyed by domain name.
        self.domains: Dict[str, PrototypeIntentStore] = {}
        #: Raw training samples per (domain, intent) — for inspection,
        #: persistence-bypass paths, and the auto domain-classifier seed.
        self._samples: Dict[str, Dict[str, List[str]]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, model, domain: str, label: str, sentences: List[str],
            k: int = 5, random_state: int = 42) -> int:
        """Register a domain-scoped intent ``label`` with example sentences.

        Adds the prototypes to the domain's sub-store AND adds the same
        prototypes under the domain name to :attr:`domain_store` so the
        top-level classifier learns the domain's surface forms
        incrementally.

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
            self.domains[domain] = PrototypeIntentStore()
        n = self.domains[domain].add(model, label, sentences,
                                      k=k, random_state=random_state)
        # Mirror the same prototypes into the domain_store under the
        # domain name. Re-uses self.add()-as-is via _accumulate_domain_proto
        # below, which preserves all samples per domain so subsequent
        # add() calls extend rather than replace.
        self._samples[domain][label] = list(sentences)
        self._rebuild_domain_proto(model, domain, k=k, random_state=random_state)
        return n

    def remove(self, domain: str, label: str) -> None:
        """Remove an intent from a domain."""
        if domain in self.domains:
            self.domains[domain].remove(label)
        self._samples[domain].pop(label, None)
        # If the domain still has intents, rebuild its top-level prototype.
        if self._samples[domain]:
            # We can't easily re-encode without `model`; the next add() call
            # will refresh. Most callers wipe the entire domain instead.
            pass

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
        """Return label → cosine inside the resolved (or specified) domain.

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

        Strategy: concatenate every intent's samples, then let
        :meth:`PrototypeIntentStore.add` choose ``k`` representatives.
        """
        all_samples: List[str] = []
        for sents in self._samples.get(domain, {}).values():
            all_samples.extend(sents)
        if not all_samples:
            return
        self.domain_store.remove(domain)
        self.domain_store.add(model, domain, all_samples,
                              k=k, random_state=random_state)
