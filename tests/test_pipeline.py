import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from ovos_bus_client.message import Message
from ovos_plugin_manager.templates.pipeline import IntentHandlerMatch


def _make_pipeline(config=None, intents=None, renormalize=True):
    """Helper: create a pipeline with a mocked model and FakeBus."""
    config = config or {}
    config.setdefault("model", "fake-model")
    config["renormalize"] = renormalize

    mock_model = MagicMock()
    mock_model.classes_ = np.array([])
    mock_model.predict_proba.return_value = np.array([[]])

    with patch("ovos_m2v_pipeline.StaticModelPipeline") as MockSMP, \
         patch("ovos_m2v_pipeline.Configuration", return_value={}):
        MockSMP.from_pretrained.return_value = mock_model
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)

    pipeline.model = mock_model
    if intents is not None:
        pipeline.intents = set(intents)
    return pipeline


def _setup_model(pipeline, labels, probs):
    """Set numpy-typed classes and a single-row probability matrix."""
    pipeline.model.classes_ = np.array(labels)
    pipeline.model.predict_proba.return_value = np.array([probs])


class TestInit(unittest.TestCase):
    def test_default_conf_thresholds(self):
        p = _make_pipeline()
        self.assertEqual(p.config.get("conf_high", 0.7), 0.7)
        self.assertEqual(p.config.get("conf_medium", 0.5), 0.5)
        self.assertEqual(p.config.get("conf_low", 0.15), 0.15)

    def test_ignore_labels_default_empty(self):
        p = _make_pipeline()
        self.assertEqual(p.ignore_labels, [])

    def test_ignore_labels_from_config(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill:bad.intent"]})
        self.assertIn("skill:bad.intent", p.ignore_labels)

    def test_missing_model_raises(self):
        with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
             patch("ovos_m2v_pipeline.Configuration", return_value={}):
            from ovos_m2v_pipeline import Model2VecIntentPipeline
            from ovos_utils.fakebus import FakeBus
            with self.assertRaises(FileNotFoundError):
                Model2VecIntentPipeline(bus=FakeBus(), config={"model": ""})


class TestGetAdaptIntents(unittest.TestCase):
    def test_returns_intent_names(self):
        p = _make_pipeline()
        fake_intents = [{"name": "skill_a:intent_one"}, {"name": "skill_b:intent_two"}]
        mock_response = Message("intent.service.adapt.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_adapt_intents()
        self.assertEqual(result, ["skill_a:intent_one", "skill_b:intent_two"])

    def test_filters_ignore_labels(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:intent_one"]})
        fake_intents = [{"name": "skill_a:intent_one"}, {"name": "skill_b:intent_two"}]
        mock_response = Message("intent.service.adapt.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_adapt_intents()
        self.assertNotIn("skill_a:intent_one", result)
        self.assertIn("skill_b:intent_two", result)

    def test_raises_on_no_response(self):
        p = _make_pipeline()
        p.bus.wait_for_response = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            p._get_adapt_intents()


class TestGetPadatiousIntents(unittest.TestCase):
    def test_returns_intent_names(self):
        p = _make_pipeline()
        fake_intents = ["skill_a:one.intent", "skill_b:two.intent"]
        mock_response = Message("intent.service.padatious.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_padatious_intents()
        self.assertEqual(result, fake_intents)

    def test_filters_ignore_labels(self):
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:one.intent"]})
        fake_intents = ["skill_a:one.intent", "skill_b:two.intent"]
        mock_response = Message("intent.service.padatious.manifest",
                                data={"intents": fake_intents})
        p.bus.wait_for_response = MagicMock(return_value=mock_response)
        result = p._get_padatious_intents()
        self.assertNotIn("skill_a:one.intent", result)
        self.assertIn("skill_b:two.intent", result)

    def test_raises_on_no_response(self):
        p = _make_pipeline()
        p.bus.wait_for_response = MagicMock(return_value=None)
        with self.assertRaises(RuntimeError):
            p._get_padatious_intents()


class TestHandleSyncIntents(unittest.TestCase):
    def test_debounce_while_syncing(self):
        p = _make_pipeline()
        p._syncing = True
        p._get_adapt_intents = MagicMock()
        p.handle_sync_intents(Message("test"))
        p._get_adapt_intents.assert_not_called()

    def test_syncs_intents(self):
        p = _make_pipeline()
        p._get_adapt_intents = MagicMock(return_value=["skill:adapt_intent"])
        p._get_padatious_intents = MagicMock(return_value=["skill:pad_intent"])
        with patch("ovos_m2v_pipeline.time") as mock_time:
            mock_time.sleep = MagicMock()
            p.handle_sync_intents(Message("test"))
        self.assertIn("skill:adapt_intent", p.intents)
        self.assertIn("skill:pad_intent", p.intents)
        self.assertFalse(p._syncing)

    def test_handles_runtime_error_gracefully(self):
        p = _make_pipeline()
        p._get_adapt_intents = MagicMock(side_effect=RuntimeError("bus timeout"))
        with patch("ovos_m2v_pipeline.time") as mock_time:
            mock_time.sleep = MagicMock()
            p.handle_sync_intents(Message("test"))  # must not raise
        self.assertFalse(p._syncing)


class TestMatch(unittest.TestCase):
    def test_registered_intent_yielded(self):
        p = _make_pipeline(intents=["skill_a:my.intent"], renormalize=False)
        _setup_model(p, ["skill_a:my.intent"], [0.9])
        results = list(p._match("turn on the lights"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(label, "skill_a:my.intent")
        self.assertEqual(skill_id, "skill_a")
        self.assertAlmostEqual(prob, 0.9)

    def test_unregistered_intent_discarded(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, ["skill_a:my.intent"], [0.9])
        # no intents registered and not a special label → mask is all-False → empty
        results = list(p._match("something"))
        self.assertEqual(results, [])

    def test_ocp_special_case_bypasses_intents_check(self):
        # ocp:play must match even with no Adapt/Padatious intents registered
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, ["ocp:play"], [0.95])
        results = list(p._match("play some music"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(skill_id, "ovos.common_play")
        self.assertEqual(label, "ovos.common_play.play_search")

    def test_common_query_special_case_bypasses_intents_check(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, ["common_query:common_query"], [0.8])
        results = list(p._match("what is the capital of France"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(skill_id, "common_query.openvoiceos")
        self.assertEqual(label, "common_query.question")

    def test_stop_special_case_bypasses_intents_check(self):
        p = _make_pipeline(intents=[], renormalize=False)
        _setup_model(p, ["stop:stop"], [0.85])
        results = list(p._match("stop"))
        self.assertEqual(len(results), 1)
        skill_id, label, prob = results[0]
        self.assertEqual(skill_id, "stop.openvoiceos")
        self.assertEqual(label, "mycroft.stop")

    def test_multiple_candidates_sorted_by_prob(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent"], [0.3, 0.6])
        results = list(p._match("test"))
        self.assertEqual(results[0][2], 0.6)  # highest prob first
        self.assertEqual(results[1][2], 0.3)

    def test_ignore_labels_not_yielded(self):
        # ignore_labels are excluded at sync time; they won't be in self.intents
        p = _make_pipeline(config={"model": "fake", "ignore_intents": ["skill_a:my.intent"]},
                           renormalize=False)
        p.intents = set()
        _setup_model(p, ["skill_a:my.intent"], [0.9])
        results = list(p._match("test"))
        self.assertEqual(results, [])


class TestNormalization(unittest.TestCase):
    def test_renormalize_true_sums_to_one(self):
        # Two registered intents with raw probs that don't sum to 1 after masking
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=True)
        # Model has 3 classes; only 2 are registered. Raw probs: [0.6, 0.2, 0.2]
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent", "skill_c:c.intent"],
                     [0.6, 0.2, 0.2])
        results = list(p._match("test"))
        total = sum(prob for _, _, prob in results)
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_renormalize_false_preserves_raw_probs(self):
        p = _make_pipeline(intents=["skill_a:a.intent", "skill_b:b.intent"], renormalize=False)
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent", "skill_c:c.intent"],
                     [0.6, 0.2, 0.2])
        results = list(p._match("test"))
        probs = [prob for _, _, prob in results]
        # Only the 2 registered intents survive; probs are raw (0.6 and 0.2)
        self.assertAlmostEqual(sorted(probs, reverse=True)[0], 0.6, places=6)
        self.assertAlmostEqual(sorted(probs, reverse=True)[1], 0.2, places=6)

    def test_renormalize_redistributes_masked_probability(self):
        # One registered intent gets 0.3 raw; 0.7 is in unregistered classes.
        # After renormalization it should become 1.0.
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, ["skill_a:a.intent", "skill_b:b.intent"], [0.3, 0.7])
        results = list(p._match("test"))
        _, _, prob = results[0]
        self.assertAlmostEqual(prob, 1.0, places=6)

    def test_renormalize_no_division_by_zero(self):
        # All masked probs are 0 → should not raise, probs stay 0
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, ["skill_a:a.intent"], [0.0])
        results = list(p._match("test"))
        _, _, prob = results[0]
        self.assertFalse(np.isnan(prob))
        self.assertAlmostEqual(prob, 0.0)

    def test_special_labels_included_in_normalization(self):
        # ocp:play + one registered intent: both survive masking
        p = _make_pipeline(intents=["skill_a:a.intent"], renormalize=True)
        _setup_model(p, ["skill_a:a.intent", "ocp:play", "skill_b:b.intent"],
                     [0.3, 0.4, 0.3])
        results = list(p._match("test"))
        total = sum(prob for _, _, prob in results)
        self.assertAlmostEqual(total, 1.0, places=6)
        # ocp:play (0.4) + skill_a (0.3) → renorm sum = 0.7
        # ocp:play renorm = 0.4/0.7 ≈ 0.571
        labels = {label for _, label, _ in results}
        self.assertIn("ovos.common_play.play_search", labels)
        self.assertIn("skill_a:a.intent", labels)


class TestMatchConfidence(unittest.TestCase):
    """Test threshold logic with renormalize=False to isolate confidence from normalization."""

    def _setup(self, prob, intents=None):
        p = _make_pipeline(intents=["skill_a:my.intent"] if intents is None else intents,
                           renormalize=False)
        _setup_model(p, ["skill_a:my.intent"], [prob])
        msg = Message("recognizer_loop:utterance")
        return p, msg

    def test_match_high_above_threshold(self):
        p, msg = self._setup(0.8)
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)
        self.assertAlmostEqual(result.match_data["confidence"], 0.8)

    def test_match_high_below_threshold_returns_none(self):
        p, msg = self._setup(0.6)
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_medium_above_threshold(self):
        p, msg = self._setup(0.55)
        result = p.match_medium(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)

    def test_match_medium_below_threshold_returns_none(self):
        p, msg = self._setup(0.4)
        result = p.match_medium(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_low_above_threshold(self):
        p, msg = self._setup(0.2)
        result = p.match_low(["turn on lights"], "en", msg)
        self.assertIsInstance(result, IntentHandlerMatch)

    def test_match_low_below_threshold_returns_none(self):
        p, msg = self._setup(0.05)
        result = p.match_low(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_returns_none_when_no_intents(self):
        p, msg = self._setup(0.99, intents=[])
        result = p.match_high(["turn on lights"], "en", msg)
        self.assertIsNone(result)

    def test_match_data_contains_utterance(self):
        p, msg = self._setup(0.9)
        result = p.match_high(["hello world"], "en", msg)
        self.assertEqual(result.match_data["utterance"], "hello world")
        self.assertEqual(result.utterance, "hello world")

    def test_custom_conf_high_from_config(self):
        p = _make_pipeline(config={"model": "fake", "conf_high": 0.95},
                           intents=["skill_a:my.intent"], renormalize=False)
        _setup_model(p, ["skill_a:my.intent"], [0.9])
        msg = Message("recognizer_loop:utterance")
        result = p.match_high(["test"], "en", msg)
        self.assertIsNone(result)  # 0.9 < 0.95


if __name__ == "__main__":
    unittest.main()
