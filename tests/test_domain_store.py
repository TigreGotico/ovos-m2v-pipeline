"""Unit tests for ``DomainPrototypeIntentStore``."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ovos_m2v_pipeline import DomainPrototypeIntentStore, PrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

ALL_STRATEGIES = list(PrototypeStrategy)


class _FakeModel:
    """Deterministic encoder. Sentences that share an obvious *anchor word*
    map to embeddings that cluster together — enough to drive both levels
    of the hierarchical store."""

    def __init__(self, dim: int = 16):
        self.dim = dim
        self._anchors = {
            "media":  self._direction("media",  0),
            "home":   self._direction("home",   1),
            "play":   self._direction("media",  0),
            "pause":  self._direction("media",  0),
            "lights": self._direction("home",   1),
            "thermostat": self._direction("home", 1),
        }

    def _direction(self, name: str, axis: int) -> np.ndarray:
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

def test_default_strategies_are_max_over_all_backcompat():
    s = DomainPrototypeIntentStore()
    assert s.domain_strategy is PrototypeStrategy.MAX_OVER_ALL
    assert s.intent_strategy is PrototypeStrategy.MAX_OVER_ALL
    assert isinstance(s.domain_store, PrototypeIntentStore)
    assert s.domain_store.strategy is PrototypeStrategy.MAX_OVER_ALL
    assert s.domains == {}


def test_custom_strategies_per_level():
    s = DomainPrototypeIntentStore(
        domain_strategy=PrototypeStrategy.MEAN_CENTROID,
        intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
        intent_tau=0.05,
        intent_top_k=2,
    )
    assert s.domain_store.strategy is PrototypeStrategy.MEAN_CENTROID
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
    # Domain router mirrors the same samples under the domain name.
    assert "media" in list(s.domain_store.labels)


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
    assert "media" not in list(s.domain_store.labels)
    assert "home" in s.domains


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def test_calc_domain_picks_correct_top_level():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "media", "pause", ["pause"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    q = m.encode(["lights on please"])[0]
    assert s.calc_domain(q) == "home"


def test_scores_returns_only_in_resolved_domain():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["lights on please"])[0]
    out = s.scores(q)
    assert "lights" in out
    assert "play" not in out


def test_scores_with_explicit_domain_bypasses_router():
    m = _FakeModel()
    s = DomainPrototypeIntentStore()
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on"])
    q = m.encode(["lights on please"])[0]
    # Force the wrong domain — confirms the override path is honoured.
    out = s.scores(q, domain="media")
    assert set(out) == {"play"}


def test_scores_empty_store_returns_empty_dict():
    s = DomainPrototypeIntentStore()
    q = np.zeros(16, dtype=np.float32)
    assert s.scores(q) == {}
    assert s.calc_domain(q) is None


# ---------------------------------------------------------------------------
# Strategy round-trip — each PrototypeStrategy survives an add+scores cycle
# at both levels.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("strat", ALL_STRATEGIES)
def test_intent_strategy_round_trip(strat):
    m = _FakeModel()
    s = DomainPrototypeIntentStore(intent_strategy=strat, intent_top_k=2, intent_tau=0.1)
    s.add(m, "media", "play",  ["play one", "play two", "play three"])
    s.add(m, "media", "pause", ["pause one", "pause two", "pause three"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    q = m.encode(["play four"])[0]
    out = s.scores(q)
    assert out, f"{strat.value} produced no scores"
    assert max(out, key=out.get) == "play", f"{strat.value} mispredicted"


@pytest.mark.parametrize("strat", ALL_STRATEGIES)
def test_domain_strategy_round_trip(strat):
    m = _FakeModel()
    s = DomainPrototypeIntentStore(domain_strategy=strat, domain_top_k=2, domain_tau=0.1)
    s.add(m, "media", "play",  ["play song"])
    s.add(m, "home",  "lights", ["lights on", "turn on lights"])
    q = m.encode(["lights on now"])[0]
    assert s.calc_domain(q) == "home", f"{strat.value} routed wrong domain"
