"""Embedding must stay in-process: model2vec spawns os.cpu_count() loky
workers for large batches unless use_multiprocessing=False — observed as 24
subprocesses inside a 2G service cgroup and an OOM-kill at skill load."""
from unittest import mock

import numpy as np

from ovos_m2v_pipeline import PrototypeIntentStore


def test_store_add_disables_multiprocessing():
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.return_value = np.ones((2, 4), dtype=np.float32)
    store.add(model, "label", ["a b", "c d"])
    assert model.encode.called
    _, kwargs = model.encode.call_args
    assert kwargs.get("use_multiprocessing") is False


def test_entity_expansion_is_bounded():
    """Two large entities in one template must not materialize the full
    cartesian product (observed: ~4.8M strings, 23G swap, OOM-kill)."""
    from ovos_m2v_pipeline import MAX_ENTITY_EXPANSIONS, Model2VecIntentPipeline
    p = Model2VecIntentPipeline.__new__(Model2VecIntentPipeline)
    p.entities = {"mon": [f"mon{i}" for i in range(2200)],
                  "move": [f"move{i}" for i in range(2200)]}
    out = p._expand_entities(["can {mon} learn {move}"])
    assert len(out) == MAX_ENTITY_EXPANSIONS
    assert len(set(out)) == len(out)
    # deterministic: same input, same sample
    assert out == p._expand_entities(["can {mon} learn {move}"])
    # endpoints of the combination space are kept
    assert "can mon0 learn move0" in out
    assert "can mon2199 learn move2199" in out


def test_small_expansion_untouched():
    from ovos_m2v_pipeline import Model2VecIntentPipeline
    p = Model2VecIntentPipeline.__new__(Model2VecIntentPipeline)
    p.entities = {"city": ["porto", "lisbon"]}
    out = p._expand_entities(["weather in {city}"])
    assert sorted(out) == ["weather in lisbon", "weather in porto"]


def test_store_ingest_is_bounded_regardless_of_source():
    """The cap must hold at the store, not one expansion path: real
    deployments feed pre-expanded padatious samples straight to add()."""
    import numpy as np
    from ovos_m2v_pipeline import (MAX_ENTITY_EXPANSIONS,
                                   PrototypeIntentStore)
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    n = store.add(model, "big", [f"sentence {i}" for i in range(50000)])
    assert n <= MAX_ENTITY_EXPANSIONS
    sents_passed = model.encode.call_args[0][0]
    assert len(sents_passed) == MAX_ENTITY_EXPANSIONS
    assert sents_passed[0] == "sentence 0"
    assert sents_passed[-1] == "sentence 49999"


def test_padatious_labels_dealias_to_canonical():
    """Dual-contract registration must collapse to ONE label, or the store
    holds every prototype twice (observed live: 83 -> 116 labels, ~1.1M
    prototypes)."""
    import numpy as np
    from ovos_utils.fakebus import FakeBus
    from ovos_bus_client.message import Message
    from ovos_m2v_pipeline import Model2VecPrototypePipeline
    p = Model2VecPrototypePipeline.__new__(Model2VecPrototypePipeline)
    p.ignore_labels = set()
    p.intents = set()
    p.model = mock.Mock()
    p.model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    from ovos_m2v_pipeline import PrototypeIntentStore
    p.prototype_store = PrototypeIntentStore()
    p._prototype_k = None
    p._handle_register_padatious(Message(
        "padatious:register_intent",
        {"name": "skill.test:go.intent", "samples": ["go to work"]}))
    assert "skill.test:go" in p.intents
    assert "skill.test:go.intent" not in p.intents
    # detach with the suffixed name removes the canonical entry
    p._handle_detach_intent(Message(
        "detach_intent", {"intent_name": "skill.test:go.intent"}))
    assert len(p.prototype_store) == 0


def test_store_add_is_amortized_no_per_add_vstack():
    """Each add must not copy the whole store (O(n^2) build churn: a 1.2GB
    array reallocated per registration on real installs). Adds buffer into
    pending chunks; consolidation happens once on read."""
    import numpy as np
    from ovos_m2v_pipeline import PrototypeIntentStore
    store = PrototypeIntentStore()
    model = mock.Mock()
    model.encode.side_effect = \
        lambda sents, **kw: np.ones((len(sents), 4), dtype=np.float32)
    for i in range(50):
        store.add(model, f"label{i}", [f"s{i}a", f"s{i}b"])
        assert len(store._pending) == i + 1  # no consolidation during adds
    assert len(store) == 100
    labels = store.labels  # first read consolidates, once
    assert len(labels) == 100
    assert len(store._pending) == 0
    # re-registration replaces, through the consolidation path
    store.add(model, "label0", ["new"])
    assert (store.labels == "label0").sum() == 1
    assert len(store) == 99
