"""Unit tests for ``HierarchicalPrototypeIntentStore`` (two-stage routing)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore, PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

ALL_STRATEGIES = list(PrototypeStrategy)


class _FakeModel:
    """Deterministic encoder. Sentences that share an obvious *anchor word*
    map to embeddings that cluster together — enough to drive two-stage
    routing across multiple domains."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._anchors = {
            "media":  self._direction(0),
            "home":   self._direction(1),
            "play":   self._direction(0),
            "pause":  self._direction(0),
            "lights": self._direction(1),
            "thermostat": self._direction(1),
        }

    def _direction(self, axis: int) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        v[axis] = 1.0
        return v

    def encode(self, sentences):
        out = []
        for s in sentences:
            sl = s.lower()
            chosen = None
            for kw, vec in self._anchors.items():
                if kw in sl:
                    chosen = vec
                    break
            if chosen is None:
                seed = int(hashlib.md5(s.encode()).hexdigest(), 16) % (2**32)
                chosen = np.random.default_rng(seed).standard_normal(self.dim).astype(np.float32)
                chosen = chosen / (np.linalg.norm(chosen) + 1e-12)
            out.append(chosen)
        return np.asarray(out)


# ---------------------------------------------------------------------------
# Construction & defaults
# ---------------------------------------------------------------------------

def test_default_is_max_over_all():
    s = HierarchicalPrototypeIntentStore()
    assert s.intent_strategy is PrototypeStrategy.MAX_OVER_ALL
    assert s.domain_threshold == 0.0
    assert s.domains == {}


def test_fingerprint_store_is_always_built():
    s = HierarchicalPrototypeIntentStore()
    # Unlike the optional top_k_domains prune, the router fingerprint
    # store is mandatory here.
    assert s._domain_fingerprints is not None


def test_custom_intent_strategy_and_threshold():
    s = HierarchicalPrototypeIntentStore(
        intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
        intent_tau=0.05,
        intent_top_k=2,
        domain_threshold=0.3,
    )
    assert s.intent_strategy is PrototypeStrategy.SOFTMAX_WEIGHTED
    assert s.domain_threshold == 0.3


# ---------------------------------------------------------------------------
# add() / remove() lifecycle
# ---------------------------------------------------------------------------

def test_add_creates_domain_and_intent():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    n = s.add(m, "media", "play", ["play song", "put on song"])
    assert n > 0
    assert "media" in s.domains
    assert "play" in list(s.domains["media"].labels)


def test_add_existing_domain_reuses_sub_store():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    assert {"play", "pause"} <= set(s.domains["media"].labels)


def test_remove_intent_keeps_domain():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    s.remove("media", "play")
    assert "play" not in list(s.domains["media"].labels)
    assert "pause" in list(s.domains["media"].labels)


def test_remove_domain_drops_everything():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play", ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    s.remove_domain("media")
    assert "media" not in s.domains
    assert "home" in s.domains


# ---------------------------------------------------------------------------
# Stage 1 — domain routing
# ---------------------------------------------------------------------------

def test_calc_domain_picks_single_best_domain():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    assert s.calc_domain(m.encode(["lights on please"])[0]) == "home"
    assert s.calc_domain(m.encode(["play africa"])[0]) == "media"


def test_calc_domain_empty_store_returns_none():
    s = HierarchicalPrototypeIntentStore()
    assert s.calc_domain(np.zeros(16, dtype=np.float32)) is None


def test_calc_domain_below_threshold_returns_none():
    m = _FakeModel()
    # A threshold above any achievable fingerprint score rejects everything.
    s = HierarchicalPrototypeIntentStore(domain_threshold=2.0)
    s.add(m, "media", "play", ["play song"])
    assert s.calc_domain(m.encode(["play africa"])[0]) is None


# ---------------------------------------------------------------------------
# Stage 2 — two-stage scoring
# ---------------------------------------------------------------------------

def test_scores_route_to_single_domain():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    out = s.scores(m.encode(["lights on please"])[0])
    # Only the routed domain's intents are scored.
    assert "lights" in out
    assert "play" not in out


def test_calc_intent_two_stage_argmax():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    s.add(m, "home",  "lights", ["lights on"])
    assert s.calc_intent(m.encode(["play africa"])[0]) == "play"


def test_scores_below_threshold_returns_empty():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore(domain_threshold=2.0)
    s.add(m, "media", "play", ["play song"])
    assert s.scores(m.encode(["play africa"])[0]) == {}
    assert s.calc_intent(m.encode(["play africa"])[0]) is None


def test_scores_with_explicit_domain_bypasses_router():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["lights on please"])[0]
    # Force the wrong domain — only that domain's labels are scored.
    out = s.scores(q, domain="media")
    assert set(out) == {"play"}


def test_scores_explicit_domain_bypasses_threshold():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore(domain_threshold=2.0)
    s.add(m, "media", "play", ["play song"])
    # Explicit domain skips the rejection gate entirely.
    assert s.scores(m.encode(["play"])[0], domain="media") != {}


def test_scores_explicit_unknown_domain_returns_empty():
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore()
    s.add(m, "media", "play", ["play song"])
    assert s.scores(m.encode(["play"])[0], domain="nope") == {}


def test_scores_empty_store_returns_empty_dict():
    s = HierarchicalPrototypeIntentStore()
    q = np.zeros(16, dtype=np.float32)
    assert s.scores(q) == {}
    assert s.calc_intent(q) is None


# ---------------------------------------------------------------------------
# Strategy round-trip — every PrototypeStrategy survives add+scores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strat", ALL_STRATEGIES)
def test_intent_strategy_round_trip(strat):
    m = _FakeModel()
    s = HierarchicalPrototypeIntentStore(intent_strategy=strat,
                                         intent_top_k=2, intent_tau=0.1)
    s.add(m, "media", "play",  ["play one", "play two", "play three"])
    s.add(m, "media", "pause", ["pause one", "pause two", "pause three"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    q = m.encode(["play four"])[0]
    out = s.scores(q)
    assert out, f"{strat.value} produced no scores"
    assert max(out, key=out.get) == "play", f"{strat.value} mispredicted"
