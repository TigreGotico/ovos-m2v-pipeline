"""Regression tests for prototype-mode exact-sample recall."""

import hashlib
import unittest

import numpy as np
from ovos_bus_client.message import Message

from tests.test_pipeline import _make_prototype_pipeline


def _hash_encode(sentences):
    """Deterministic bag-of-words hashing encoder.

    Identical sentences map to identical vectors (cosine 1.0); sentences
    sharing few words score low. Stands in for a real embedding model so the
    store's anchor bookkeeping can be tested exactly.
    """
    dim = 64
    out = np.zeros((len(sentences), dim), dtype=np.float32)
    for i, sent in enumerate(sentences):
        for tok in sent.lower().split():
            bucket = int(hashlib.md5(tok.encode()).hexdigest(), 16) % dim
            out[i, bucket] += 1.0
    return out


class TestExactSampleHighTierMatch(unittest.TestCase):
    """A registered training sample must match at the high tier.

    Guards against per-label anchor subsampling dropping the very samples a
    skill registered: with hundreds of expanded templates per intent, a
    capped anchor set almost never contains a given sample, so an exact
    training utterance scores below ``conf_high`` and the pipeline silently
    declines a match its store should hit.
    """

    def _register_and_match(self, pipeline, samples, utterance):
        pipeline.model.encode.side_effect = _hash_encode
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "ovos-skill-date-time.openvoiceos:what.time.is.it.intent",
             "samples": samples, "lang": "en-US"},
            {"skill_id": "ovos-skill-date-time.openvoiceos"}))
        msg = Message("recognizer_loop:utterance",
                      {"utterances": [utterance], "lang": "en-US"})
        return pipeline.match_high([utterance], "en-US", msg)

    def test_exact_sample_matches_high_tier(self):
        pipeline = _make_prototype_pipeline()
        # A large intent file: many distinct templates, like the expanded
        # what.time.is.it.intent shipped by ovos-skill-date-time.
        samples = [f"placeholder sample number {i} variant" for i in range(400)]
        samples.append("what time is it")
        match = self._register_and_match(pipeline, samples, "what time is it")
        self.assertIsNotNone(match)
        self.assertEqual(
            match.match_type,
            "ovos-skill-date-time.openvoiceos:what.time.is.it.intent")
        self.assertGreaterEqual(match.match_data["confidence"], 0.99)

    def test_every_sample_kept_by_default(self):
        pipeline = _make_prototype_pipeline()
        pipeline.model.encode.side_effect = _hash_encode
        samples = [f"sample number {i}" for i in range(50)]
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "skill:intent", "samples": samples, "lang": "en-US"},
            {"skill_id": "skill"}))
        self.assertEqual(len(pipeline.prototype_store), len(samples))

    def test_prototype_k_still_caps_anchor_count(self):
        pipeline = _make_prototype_pipeline(config={"prototype_k": 5})
        pipeline.model.encode.side_effect = _hash_encode
        samples = [f"sample number {i}" for i in range(50)]
        pipeline.bus.emit(Message(
            "padatious:register_intent",
            {"name": "skill:intent", "samples": samples, "lang": "en-US"},
            {"skill_id": "skill"}))
        self.assertEqual(len(pipeline.prototype_store), 5)


if __name__ == "__main__":
    unittest.main()
