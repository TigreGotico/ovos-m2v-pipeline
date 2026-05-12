"""End-to-end tests for Model2VecPrototypePipeline using ovoscope.

Drives a `MiniCroft` instance with the standalone prototype entry point
(`ovos-m2v-prototype-pipeline`) and exercises the full
``padatious:register_intent`` → encode/build prototypes → utterance
dispatch path on the bus.

`model2vec.StaticModel` is patched at sys.modules level so no model is
actually downloaded. The mock model's `encode()` returns deterministic,
linearly separable embeddings keyed on whether the input contains specific
substrings — enough to drive cosine similarity scoring through the real
PrototypeIntentStore math.
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

from ovos_m2v_pipeline import Model2VecPrototypePipeline  # noqa: E402

PIPELINE_ID = "ovos-m2v-prototype-pipeline"
CONFIG_KEY = "ovos_m2v_prototype_pipeline"

# Deterministic 4-D directions for the mock encoder. Cosine similarity
# between any two of these is 0 (orthogonal); identical inputs score 1.0.
_DIRECTIONS = {
    "lights": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    "music":  np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32),
    "weather": np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32),
    "stop":   np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
}
_NOISE = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def _fake_encode(sentences):
    """Return deterministic embeddings: pick the first matching keyword."""
    out = []
    for s in sentences:
        sl = s.lower()
        vec = _NOISE.copy()
        for key, direction in _DIRECTIONS.items():
            if key in sl:
                vec = direction.copy()
                break
        out.append(vec)
    return np.stack(out)


class _PrototypeE2EBase(unittest.TestCase):
    """Shared setup: MiniCroft + prototype pipeline + mocked StaticModel."""

    @classmethod
    def setUpClass(cls):
        # Fake `model2vec` module so prototype mode's `from model2vec import
        # StaticModel` succeeds without any network or heavy deps.
        cls._mock_model = MagicMock()
        cls._mock_model.encode.side_effect = _fake_encode

        fake_m2v = MagicMock()
        fake_m2v.StaticModel.from_pretrained.return_value = cls._mock_model
        cls._sys_modules_patch = patch.dict(sys.modules, {"model2vec": fake_m2v})
        cls._sys_modules_patch.start()

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig = intents_cfg.get(CONFIG_KEY)
        intents_cfg[CONFIG_KEY] = {
            "model": "fake-model",
            "prototype_k": 5,
        }

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=60,
        )
        cls.pipeline: Model2VecPrototypePipeline = (
            cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        )
        # Replace whatever was loaded by the entry point with our deterministic mock.
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
        # Reset the prototype store between tests.
        from ovos_m2v_pipeline import PrototypeIntentStore
        self.pipeline.prototype_store = PrototypeIntentStore()
        self.pipeline.intents = set()
        self.pipeline.ignore_labels = []

    def _register_padatious(self, name: str, samples: list[str]):
        """Synchronously seed the prototype store via the bus event the
        real Padatious skill would fire."""
        self.mc.bus.emit(Message(
            "padatious:register_intent",
            data={"name": name, "samples": samples},
        ))

    def _utterance_msg(self, utterance: str,
                       session_pipeline: list[str] | None = None) -> Message:
        ctx = {}
        if session_pipeline is not None:
            sess = Session(session_id="ovoscope-proto",
                           pipeline=session_pipeline)
            ctx["session"] = sess.serialize()
        return Message(
            "recognizer_loop:utterance",
            data={"utterances": [utterance], "lang": "en-US"},
            context=ctx,
        )

    def _send_and_capture(self, utterance: str, expected_types: list[str],
                          timeout: float = 5.0,
                          session_pipeline: list[str] | None = None) -> Message | None:
        got: list[Message] = []
        done = threading.Event()
        failed = threading.Event()

        def _on_match(msg):
            got.append(msg)
            done.set()

        def _on_fail(_msg):
            failed.set()
            done.set()

        for t in expected_types:
            self.mc.bus.on(t, _on_match)
        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            done.wait(timeout=timeout)
        finally:
            for t in expected_types:
                self.mc.bus.remove(t, _on_match)
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        if failed.is_set() and not got:
            return None
        return got[0] if got else None

    def _expect_no_match(self, utterance: str, timeout: float = 2.0,
                         session_pipeline: list[str] | None = None):
        failed = threading.Event()

        def _on_fail(_msg):
            failed.set()

        self.mc.bus.on("complete_intent_failure", _on_fail)
        try:
            self.mc.bus.emit(self._utterance_msg(utterance, session_pipeline))
            failed.wait(timeout=timeout)
        finally:
            self.mc.bus.remove("complete_intent_failure", _on_fail)
        self.assertTrue(
            failed.is_set(),
            f"Expected no match for {utterance!r}, but got no intent_failure.",
        )


class TestPrototypePipelineLoad(_PrototypeE2EBase):
    def test_pipeline_loaded_in_prototype_mode(self):
        self.assertIsInstance(self.pipeline, Model2VecPrototypePipeline)
        self.assertIsNotNone(self.pipeline.prototype_store)
        # Mode is forced by Model2VecPrototypePipeline.__init__.
        self.assertEqual(self.pipeline.config.get("mode"), "prototype")


class TestPrototypeRegistration(_PrototypeE2EBase):
    def test_register_padatious_inline_samples_builds_prototypes(self):
        self._register_padatious(
            "skill_a:lights.intent",
            ["turn on the lights", "switch on the lights", "lights on"],
        )
        # Prototype store should have ≥1 prototype for the new label.
        labels = list(self.pipeline.prototype_store.labels)
        self.assertIn("skill_a:lights.intent", labels)
        self.assertIn("skill_a:lights.intent", self.pipeline.intents)

    def test_register_padatious_ignored_label_does_not_build(self):
        self.pipeline.ignore_labels = ["skill_z:blocked.intent"]
        self._register_padatious(
            "skill_z:blocked.intent",
            ["this should never register"],
        )
        labels = list(self.pipeline.prototype_store.labels)
        self.assertNotIn("skill_z:blocked.intent", labels)


class TestPrototypeMatch(_PrototypeE2EBase):
    def _seed_two_skills(self):
        self._register_padatious(
            "skill_a:lights.intent",
            ["turn on the lights", "switch on the lights"],
        )
        self._register_padatious(
            "skill_b:music.intent",
            ["play some music", "start the music"],
        )

    def test_high_confidence_dispatch(self):
        self._seed_two_skills()
        msg = self._send_and_capture(
            "lights on now",  # contains 'lights' → matches skill_a direction
            expected_types=["skill_a:lights.intent"],
        )
        self.assertIsNotNone(msg, "expected match message on bus")
        self.assertEqual(msg.msg_type, "skill_a:lights.intent")
        # Cosine of identical orthogonal direction is 1.0.
        self.assertAlmostEqual(msg.data.get("confidence"), 1.0, places=5)

    def test_unrelated_utterance_no_match(self):
        # No prototypes registered → scores dict is empty → no match.
        self._expect_no_match("anything at all")

    def test_orthogonal_utterance_below_threshold(self):
        # Register only skill_a (lights direction), then send an utterance
        # whose embedding is the zero vector (no keyword match).
        self._register_padatious(
            "skill_a:lights.intent",
            ["turn on the lights"],
        )
        # 'glorpfest' triggers the noise (zero) vector → cosine 0 → below
        # every threshold including conf_low=0.15.
        self._expect_no_match("glorpfest")


class TestPrototypeSpecialLabelGating(_PrototypeE2EBase):
    """Prototype mode must apply the same session.pipeline gating as
    classifier mode for ocp/stop/common_query special labels."""

    def test_ocp_special_label_passes_when_ocp_in_session(self):
        # Build a prototype for the raw special label.
        self._register_padatious("ocp:play", ["play some music"])
        msg = self._send_and_capture(
            "play some music",
            expected_types=["ovos.common_play.play_search"],
            session_pipeline=["ovos-ocp-pipeline-plugin-high",
                              "ovos-m2v-prototype-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "ovos.common_play.play_search")

    def test_stop_special_label_dropped_when_session_excludes_it(self):
        self._register_padatious("stop:stop", ["stop now"])
        self._expect_no_match(
            "stop now",
            session_pipeline=["ovos-m2v-prototype-pipeline"],
        )

    def test_common_query_routed_when_pipeline_present(self):
        self._register_padatious(
            "common_query:common_query",
            ["what is the weather like"],
        )
        msg = self._send_and_capture(
            "what is the weather like",
            expected_types=["common_query.question"],
            session_pipeline=["ovos-common-query-pipeline-plugin",
                              "ovos-m2v-prototype-pipeline"],
        )
        self.assertIsNotNone(msg)
        self.assertEqual(msg.msg_type, "common_query.question")


if __name__ == "__main__":
    unittest.main()
