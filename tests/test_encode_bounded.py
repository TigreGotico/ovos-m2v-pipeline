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
