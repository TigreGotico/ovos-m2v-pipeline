"""Unit tests for ``DomainPrototypeIntentStore`` (parallel-argmax)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ovos_m2v_pipeline import DomainPrototypeIntentStore, PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

ALL_STRATEGIES = list(PrototypeStrategy)


class _FakeModel:
    """Deterministic encoder. Sentences that share an obvious *anchor word*
    map to embeddings that cluster together — enough to drive parallel
    scoring across multiple domains."""

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

def test_default_is_max_over_all_no_router():
    s = DomainPrototypeIntentStore()
    assert s.intent_strategy is PrototypeStrategy.MAX_OVER_ALL
    assert s.top_k_domains is None
    assert s.domains == {}
    # No top-level router attribute exists.
    assert not hasattr(s, "domain_store")


def test_custom_intent_strategy():
    s = DomainPrototypeIntentStore(
        intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
        intent_tau=0.05,
        intent_top_k=2,
    )
    assert s.intent_strategy is PrototypeStrategy.SOFTMAX_WEIGHTED


# ---------------------------------------------------------------------------
# add() / remove() lifecycle
# ---------------------------------------------------------------------------

def test_add_creates_domain_and_intent():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    n = s.add(m, "media", "play", ["play song", "put on song"])
    assert n > 0
    assert "media" in s.domains
    assert "play" in list(s.domains["media"].labels)


def test_add_propagates_intent_strategy_to_new_sub_store():
    m = _FakeModel()
    s = DomainPrototypeIntentStore(
        intent_strategy=PrototypeStrategy.MEAN_CENTROID,
    )
    s.add(m, "media", "play", ["play song one", "play song two", "put on song"])
    assert s.domains["media"].strategy is PrototypeStrategy.MEAN_CENTROID
    # MEAN_CENTROID → exactly one anchor per label.
    assert list(s.domains["media"].labels).count("play") == 1


def test_add_existing_domain_reuses_sub_store():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    assert {"play", "pause"} <= set(s.domains["media"].labels)


def test_remove_intent_keeps_domain():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    s.remove("media", "play")
    assert "play" not in list(s.domains["media"].labels)
    assert "pause" in list(s.domains["media"].labels)


def test_remove_domain_drops_everything():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play", ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    s.remove_domain("media")
    assert "media" not in s.domains
    assert "home" in s.domains


# ---------------------------------------------------------------------------
# Inference — flat parallel scoring
# ---------------------------------------------------------------------------

def test_scores_flatten_across_all_domains():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["lights on please"])[0]
    out = s.scores(q)
    # Every intent across every domain is represented.
    assert set(out) == {"play", "lights"}
    # Argmax picks the right one.
    assert max(out, key=out.get) == "lights"


def test_calc_intent_argmax_across_domains():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["play africa"])[0]
    assert s.calc_intent(q) == "play"


def test_scores_with_explicit_domain_restricts():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["lights on please"])[0]
    # Force the wrong domain — only that domain's labels are scored.
    out = s.scores(q, domain="media")
    assert set(out) == {"play"}


def test_scores_explicit_unknown_domain_returns_empty():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play", ["play song"])
    q = m.encode(["play"])[0]
    assert s.scores(q, domain="nope") == {}


def test_scores_empty_store_returns_empty_dict():
    s = DomainPrototypeIntentStore()
    q = np.zeros(16, dtype=np.float32)
    assert s.scores(q) == {}
    assert s.calc_intent(q) is None


# ---------------------------------------------------------------------------
# Optional top_k_domains pruning
# ---------------------------------------------------------------------------

def test_top_k_domains_prunes_to_best_domains():
    m = _FakeModel()
    s = DomainPrototypeIntentStore(top_k_domains=1)
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    q = m.encode(["lights on please"])[0]
    out = s.scores(q)
    # Only the winning domain's intents are returned.
    assert "lights" in out
    assert "play" not in out


def test_top_k_domains_none_evaluates_all():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()  # default: None
    s.add(m, "media", "play", ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["lights on please"])[0]
    out = s.scores(q)
    assert set(out) == {"play", "lights"}


# ---------------------------------------------------------------------------
# Strategy round-trip — every PrototypeStrategy survives add+scores
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strat", ALL_STRATEGIES)
def test_intent_strategy_round_trip(strat):
    m = _FakeModel()
    s = DomainPrototypeIntentStore(intent_strategy=strat,
                                    intent_top_k=2, intent_tau=0.1)
    s.add(m, "media", "play",  ["play one", "play two", "play three"])
    s.add(m, "media", "pause", ["pause one", "pause two", "pause three"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    q = m.encode(["play four"])[0]
    out = s.scores(q)
    assert out, f"{strat.value} produced no scores"
    assert max(out, key=out.get) == "play", f"{strat.value} mispredicted"
