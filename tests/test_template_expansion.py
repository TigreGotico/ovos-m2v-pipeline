"""Malformed Padatious templates must never abort intent registration.

Real-world skill locale files contain malformed template lines
(translated slot names such as ``{Medien}``, truncated braces such as
``{location``, adjacent slots such as ``{a}{b}``, and bracketed markers
such as ``[UNUSED]``).  One bad line must be logged and skipped while
every valid line keeps registering.
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message

from ovos_m2v_pipeline import _parse_intent_file


def _make_prototype_pipeline():
    config = {"model": "fake-embed-model", "mode": "prototype"}

    mock_embed_model = MagicMock()
    mock_embed_model.encode.return_value = np.zeros((1, 4), dtype=np.float32)

    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)

    pipeline.model = mock_embed_model
    pipeline.prototype_store = MagicMock()
    pipeline.prototype_store.add.return_value = 1
    return pipeline


class TestParseIntentFile(unittest.TestCase):
    def _write(self, tmp, lines):
        import os
        path = os.path.join(tmp, "demo.intent")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return path

    def test_malformed_lines_skipped_valid_kept(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, [
                "play {Medien}",          # invalid (uppercase) slot name
                "weather in {location",   # truncated brace
                "{first}{second}",        # adjacent slots
                "[UNUSED] noise",         # bracketed marker (valid optional)
                "play {media}",
                "(start|begin) {media}",
            ])
            sentences = _parse_intent_file(path)
        self.assertIn("play {media}", sentences)
        self.assertIn("start {media}", sentences)
        self.assertIn("begin {media}", sentences)
        self.assertIn("noise", sentences)
        self.assertNotIn("play {Medien}", sentences)
        self.assertNotIn("weather in {location", sentences)
        self.assertNotIn("{first}{second}", sentences)

    def test_all_malformed_yields_empty(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            path = self._write(tmp, ["{Medien}", "{oops", "{a}{b}"])
            self.assertEqual(_parse_intent_file(path), [])


class TestRegisterPadatiousInlineSamples(unittest.TestCase):
    def test_mixed_samples_register_valid_ones(self):
        pipeline = _make_prototype_pipeline()
        msg = Message("padatious:register_intent", data={
            "name": "test_skill:demo.intent",
            "samples": ["play {Medien}", "tune to {station",
                        "{a}{b}", "play {media}"],
        })
        pipeline._handle_register_padatious(msg)
        self.assertIn("test_skill:demo.intent", pipeline.intents)
        args = pipeline.prototype_store.add.call_args
        self.assertEqual(args[0][2], ["play {media}"])

    def test_malformed_only_does_not_raise_or_register(self):
        pipeline = _make_prototype_pipeline()
        msg = Message("padatious:register_intent", data={
            "name": "test_skill:demo.intent",
            "samples": ["{Medien}", "{loc", "{a}{b}"],
        })
        pipeline._handle_register_padatious(msg)
        self.assertNotIn("test_skill:demo.intent", pipeline.intents)
        pipeline.prototype_store.add.assert_not_called()


if __name__ == "__main__":
    unittest.main()
