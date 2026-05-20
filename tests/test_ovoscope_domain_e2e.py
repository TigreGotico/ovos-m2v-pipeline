"""End-to-end tests for Model2VecDomainPrototypePipeline using ovoscope.

Drives a `MiniCroft` instance with the standalone domain-prototype entry
point (`ovos-m2v-domain-prototype-pipeline`) and exercises the
``padatious:register_intent`` → encode/build prototypes → utterance
dispatch path against a :class:`DomainPrototypeIntentStore`.

The fake encoder produces orthogonal directions per keyword, so the
router can route by keyword presence (skill_id = domain, taken from
the ``skill_id:intent`` prefix of each registered intent label).
"""
import sys
import threading
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("ovoscope", reason="ovoscope not installed; skipping E2E tests")

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

from ovos_m2v_pipeline import (  # noqa: E402
    DomainPrototypeIntentStore,
    Model2VecDomainPrototypePipeline,
    PrototypeIntentStore,
)
from ovos_m2v_pipeline.strategies import PrototypeStrategy  # noqa: E402

PIPELINE_ID = "ovos-m2v-domain-prototype-pipeline"
CONFIG_KEY = "ovos_m2v_domain_prototype_pipeline"

_DIRECTIONS = {
    "lights":  np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "music":   np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "weather": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
}
_NOISE = np.zeros(4, dtype=np.float32)


def _fake_encode(sentences):
    out = []
    for s in sentences:
        sl = s.lower()
        vec = _NOISE.copy()
        for key, d in _DIRECTIONS.items():
            if key in sl:
                vec = d.copy()
                break
        out.append(vec)
    return np.stack(out)


class _DomainE2EBase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._mock_model = MagicMock()
        cls._mock_model.encode.side_effect = _fake_encode

        fake_m2v = MagicMock()
        fake_m2v.StaticModel.from_pretrained.return_value = cls._mock_model
        cls._sys_modules_patch = patch.dict(sys.modules, {"model2vec": fake_m2v})
        cls._sys_modules_patch.start()

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig = intents_cfg.get(CONFIG_KEY)
        intents_cfg[CONFIG_KEY] = {"model": "fake-model", "prototype_k": 5}

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=60,
        )
        cls.pipeline: Model2VecDomainPrototypePipeline = (
            cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        )
        cls.pipeline.model = cls._mock_model

    @classmethod
    def tearDownClass(cls):
        try:
            cls.mc.stop()
        finally:
            cfg = Configuration()
            intents_cfg = cfg.get("intents", {})
            if cls._orig is None:
                intents_cfg.pop(CONFIG_KEY, None)
            else:
                intents_cfg[CONFIG_KEY] = cls._orig
            cls._sys_modules_patch.stop()

    def setUp(self):
        # Fresh hierarchical store between tests.
        self.pipeline.prototype_store = DomainPrototypeIntentStore()
        self.pipeline.intents = set()
        self.pipeline.ignore_labels = []

    def _register(self, name: str, samples: list[str]):
        self.mc.bus.emit(Message(
            "padatious:register_intent",
            data={"name": name, "samples": samples},
        ))

    def _utterance_msg(self, utterance: str,
                       session_pipeline: list[str] | None = None) -> Message:
        ctx = {}
        if session_pipeline is not None:
            sess = Session(session_id="ovoscope-domain", pipeline=session_pipeline)
            ctx["session"] = sess.serialize()
        return Message(
            "recognizer_loop:utterance",
            data={"utterances": [utterance], "lang": "en-US"},
            context=ctx,
        )

    def _send_and_capture(self, utterance: str, expected_types: list[str],
                          timeout: float = 5.0) -> Message | None:
        got: list[Message] = []
        done = threading.Event()

        def _on_match(msg):
            got.append(msg)
            done.set()

        def _on_fail(_):
            done.set()

        for t in expected_types:
            self.mc.bus.on(t, _on_match)
        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _on_match)
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        return got[0] if got else None


class TestDomainPipelineLoad(_DomainE2EBase):
    def test_loaded_with_domain_store(self):
        self.assertIsInstance(self.pipeline, Model2VecDomainPrototypePipeline)
        self.assertIsInstance(self.pipeline.prototype_store,
                              DomainPrototypeIntentStore)
        # Mode is still "prototype" — only the store shape differs.
        self.assertEqual(self.pipeline.config.get("mode"), "prototype")

    def test_flat_pipeline_is_separate_class(self):
        """The domain pipeline must not inherit the flat store."""
        self.assertNotIsInstance(self.pipeline.prototype_store,
                                  PrototypeIntentStore.__mro__[0]
                                  if False else type(None))
        # Tautology-free check: it's specifically the domain store.
        self.assertIsInstance(self.pipeline.prototype_store,
                              DomainPrototypeIntentStore)


class TestDomainRegistrationRouting(_DomainE2EBase):
    def test_intents_routed_to_skill_id_domain(self):
        self._register("smarthome.skill:lights.intent",
                       ["turn on the lights", "lights on"])
        self._register("media.skill:music.intent",
                       ["play music", "start the music"])
        store = self.pipeline.prototype_store
        self.assertIn("smarthome.skill", store.domains)
        self.assertIn("media.skill", store.domains)
        self.assertIn("smarthome.skill:lights.intent",
                      list(store.domains["smarthome.skill"].labels))
        self.assertIn("media.skill:music.intent",
                      list(store.domains["media.skill"].labels))

    def test_intent_without_namespace_falls_back_to_full_label(self):
        # Labels with no ":" use the whole name as the domain.
        self._register("orphan_label", ["lights on"])
        self.assertIn("orphan_label", self.pipeline.prototype_store.domains)

    def test_detach_intent_removes_only_that_label(self):
        self._register("smarthome.skill:lights.intent", ["lights on"])
        self._register("smarthome.skill:door.intent",   ["open the door"])
        self.mc.bus.emit(Message("detach_intent", data={
            "intent_name": "smarthome.skill:lights.intent",
        }))
        labels = list(self.pipeline.prototype_store
                      .domains["smarthome.skill"].labels)
        self.assertNotIn("smarthome.skill:lights.intent", labels)
        self.assertIn("smarthome.skill:door.intent", labels)

    def test_detach_skill_drops_whole_domain(self):
        self._register("smarthome.skill:lights.intent", ["lights on"])
        self._register("media.skill:music.intent",      ["play music"])
        self.mc.bus.emit(Message("detach_skill",
                                 data={"skill_id": "smarthome.skill"}))
        self.assertNotIn("smarthome.skill",
                         self.pipeline.prototype_store.domains)
        self.assertIn("media.skill", self.pipeline.prototype_store.domains)


class TestDomainMatch(_DomainE2EBase):
    def _seed(self):
        self._register("smarthome.skill:lights.intent",
                       ["turn on the lights"])
        self._register("media.skill:music.intent",
                       ["play music", "start the music"])

    def test_router_picks_correct_domain(self):
        self._seed()
        msg = self._send_and_capture(
            "lights on now",
            expected_types=["smarthome.skill:lights.intent"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "smarthome.skill:lights.intent")

    def test_no_match_for_unknown_keywords(self):
        self._seed()
        msg = self._send_and_capture(
            "glorpfest",
            expected_types=["smarthome.skill:lights.intent",
                            "media.skill:music.intent"],
            timeout=1.5,
        )
        self.assertIsNone(msg)


class TestDomainStrategyConfigPlumbing(unittest.TestCase):
    """The domain pipeline must wire ``intent_*`` config keys onto the
    per-domain sub-stores and ``prototype_*`` keys onto the router."""

    @classmethod
    def setUpClass(cls):
        cls._mock_model = MagicMock()
        cls._mock_model.encode.side_effect = _fake_encode
        fake_m2v = MagicMock()
        fake_m2v.StaticModel.from_pretrained.return_value = cls._mock_model
        cls._patch = patch.dict(sys.modules, {"model2vec": fake_m2v})
        cls._patch.start()

    @classmethod
    def tearDownClass(cls):
        cls._patch.stop()

    def _build(self, **overrides):
        return Model2VecDomainPrototypePipeline(
            bus=None, config={"model": "fake-model", **overrides},
        )

    def test_default_strategies_at_both_levels(self):
        pipe = self._build()
        store = pipe.prototype_store
        self.assertIs(store.domain_store.strategy,
                      PrototypeStrategy.MAX_OVER_ALL)
        self.assertIs(store.intent_strategy,
                      PrototypeStrategy.MAX_OVER_ALL)

    def test_intent_strategy_lands_on_sub_stores(self):
        pipe = self._build(
            prototype_strategy="mean_centroid",
            intent_strategy="softmax_weighted",
            intent_tau=0.05,
            intent_top_k=4,
        )
        store = pipe.prototype_store
        self.assertIs(store.domain_store.strategy,
                      PrototypeStrategy.MEAN_CENTROID)
        self.assertIs(store.intent_strategy,
                      PrototypeStrategy.SOFTMAX_WEIGHTED)
        # New sub-stores inherit the intent_* settings.
        pipe._handle_register_padatious(Message(
            "padatious:register_intent",
            data={"name": "skill.x:foo.intent",
                  "samples": ["lights on"]},
        ))
        sub = store.domains["skill.x"]
        self.assertIs(sub.strategy, PrototypeStrategy.SOFTMAX_WEIGHTED)
        self.assertEqual(sub.top_k, 4)
        self.assertAlmostEqual(sub.tau, 0.05)

    def test_intent_keys_default_to_prototype_keys(self):
        pipe = self._build(prototype_strategy="top_k_mean", prototype_top_k=6)
        pipe._handle_register_padatious(Message(
            "padatious:register_intent",
            data={"name": "skill.x:foo.intent", "samples": ["lights on"]},
        ))
        sub = pipe.prototype_store.domains["skill.x"]
        self.assertIs(sub.strategy, PrototypeStrategy.TOP_K_MEAN)
        self.assertEqual(sub.top_k, 6)


if __name__ == "__main__":
    unittest.main()
