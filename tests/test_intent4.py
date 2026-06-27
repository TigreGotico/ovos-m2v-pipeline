"""Unit tests for OVOS-INTENT-4 template/entity registration adoption.

m2v is a TEMPLATE-style engine (it matches on example utterances), so it
consumes `ovos.intent.register.template` (§6) and `ovos.entity.register`
(§7) in addition to the legacy `padatious:register_intent` topics. It does
NOT consume `ovos.intent.register.keyword` (§11).
"""
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from ovos_bus_client.message import Message
from ovos_spec_tools import SpecMessage


def _make_prototype_pipeline(config=None):
    """Prototype-mode pipeline with a mocked StaticModel + FakeBus."""
    config = config or {}
    config.setdefault("model", "fake-embed-model")
    config["mode"] = "prototype"

    mock_embed_model = MagicMock()
    # identity rows so each sample becomes a distinct unit prototype
    mock_embed_model.encode.side_effect = lambda sents: np.eye(len(sents), 4, dtype=np.float32)

    fake_m2v = MagicMock()
    fake_m2v.StaticModel.from_pretrained.return_value = mock_embed_model

    with patch("ovos_m2v_pipeline.StaticModelPipeline"), \
         patch("ovos_m2v_pipeline.Configuration", return_value={}), \
         patch.dict(sys.modules, {"model2vec": fake_m2v}):
        from ovos_m2v_pipeline import Model2VecIntentPipeline
        from ovos_utils.fakebus import FakeBus
        pipeline = Model2VecIntentPipeline(bus=FakeBus(), config=config)
    pipeline.model = mock_embed_model
    return pipeline


def _make_classifier_pipeline(config=None):
    """Classifier-mode pipeline with a mocked StaticModelPipeline + FakeBus."""
    config = config or {}
    config.setdefault("model", "fake-model")

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
    return pipeline


class TestIntent4Subscriptions(unittest.TestCase):
    """The INTENT-4 topics are subscribed (and the keyword topic is NOT)."""

    def _registered_topics(self, pipeline):
        # FakeBus stores handlers in .ee (EventEmitter); fall back to events dict
        try:
            return set(pipeline.bus.ee._events.keys())
        except AttributeError:
            return set(getattr(pipeline.bus, "events", {}).keys())

    def test_prototype_mode_subscribes_template_not_keyword(self):
        p = _make_prototype_pipeline()
        topics = self._registered_topics(p)
        self.assertIn(SpecMessage.INTENT_REGISTER_TEMPLATE.value, topics)
        self.assertIn(SpecMessage.ENTITY_REGISTER.value, topics)
        self.assertIn(SpecMessage.INTENT_DEREGISTER.value, topics)
        self.assertIn(SpecMessage.SKILL_DEREGISTER.value, topics)
        self.assertNotIn(SpecMessage.INTENT_REGISTER_KEYWORD.value, topics)

    def test_classifier_mode_subscribes_template_not_keyword(self):
        p = _make_classifier_pipeline()
        topics = self._registered_topics(p)
        self.assertIn(SpecMessage.INTENT_REGISTER_TEMPLATE.value, topics)
        self.assertNotIn(SpecMessage.INTENT_REGISTER_KEYWORD.value, topics)

    def test_legacy_topics_still_subscribed(self):
        p = _make_prototype_pipeline()
        topics = self._registered_topics(p)
        self.assertIn("padatious:register_intent", topics)
        self.assertIn("detach_intent", topics)
        self.assertIn("detach_skill", topics)


class TestIntent4TemplateRegistration(unittest.TestCase):
    def _register(self, p, skill_id="music.skill", intent_name="play_music",
                  samples=None, lang="en-US"):
        msg = Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={
                "skill_id": skill_id,
                "intent_name": intent_name,
                "lang": lang,
                "samples": samples if samples is not None else ["play music", "put on a song"],
            },
            context={"skill_id": skill_id},
        )
        p._handle_intent4_register_template(msg)

    def test_register_accepted_and_label_tracked(self):
        p = _make_prototype_pipeline()
        self._register(p)
        self.assertIn("music.skill:play_music", p.intents)
        self.assertIn("music.skill:play_music", p.prototype_store.unique_labels)

    def test_registered_intent_matches_utterance(self):
        """Register via the template topic, then match an utterance.

        The mock encoder maps the Nth registered sample to basis vector e_N.
        With max_over_all the query that equals one prototype yields cosine 1.0.
        """
        p = _make_prototype_pipeline()
        # encode returns deterministic basis vectors; register a single sample
        self._register(p, samples=["play music"])
        # query embedding identical to the stored prototype -> cosine 1.0
        p.model.encode.side_effect = None
        p.model.encode.return_value = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        results = list(p._match("play music"))
        self.assertEqual(len(results), 1)
        skill_id, label, score = results[0]
        self.assertEqual(label, "music.skill:play_music")
        self.assertEqual(skill_id, "music.skill")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_bracket_templates_expanded(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["(play|put on) the music"])
        # two expanded variants -> two prototypes for the label
        n = (p.prototype_store.labels == "music.skill:play_music").sum()
        self.assertEqual(n, 2)

    def test_replacement_on_re_register(self):
        p = _make_prototype_pipeline()
        self._register(p, samples=["one"])
        self._register(p, samples=["two"])  # same triple -> replaces (§8.1)
        self.assertEqual((p.prototype_store.labels == "music.skill:play_music").sum(), 1)

    def test_missing_samples_rejected(self):
        p = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            self._register(p, samples=[])
        self.assertNotIn("music.skill:play_music", p.intents)
        warn.assert_called()

    def test_missing_identity_rejected(self):
        p = _make_prototype_pipeline()
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                      data={"intent_name": "x", "samples": ["hi"]})
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            p._handle_intent4_register_template(msg)
        warn.assert_called()
        self.assertEqual(len(p.prototype_store), 0)

    def test_ignored_label_skipped(self):
        p = _make_prototype_pipeline()
        p.ignore_labels = ["music.skill:play_music"]
        self._register(p)
        self.assertEqual(len(p.prototype_store), 0)

    def test_classifier_mode_tracks_label_only(self):
        p = _make_classifier_pipeline()
        msg = Message(SpecMessage.INTENT_REGISTER_TEMPLATE.value,
                      data={"skill_id": "music.skill", "intent_name": "play_music",
                            "lang": "en-US", "samples": ["play music"]})
        p._handle_intent4_register_template(msg)
        self.assertIn("music.skill:play_music", p.intents)
        self.assertIsNone(p.prototype_store)


class TestIntent4EntityRegistration(unittest.TestCase):
    def test_entity_fills_template_slots(self):
        p = _make_prototype_pipeline()
        p._handle_intent4_register_entity(Message(
            SpecMessage.ENTITY_REGISTER.value,
            data={"skill_id": "music.skill", "entity_name": "engine",
                  "lang": "en-US", "samples": ["spotify", "youtube"]},
        ))
        self.assertIn("engine", p.entities)
        # register a template that references {engine}
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": "music.skill", "intent_name": "play_on",
                  "lang": "en-US", "samples": ["play on {engine}"]},
        ))
        n = (p.prototype_store.labels == "music.skill:play_on").sum()
        # two entity values -> two filled prototypes
        self.assertEqual(n, 2)

    def test_entity_missing_samples_rejected(self):
        p = _make_prototype_pipeline()
        with patch("ovos_m2v_pipeline.LOG.warning") as warn:
            p._handle_intent4_register_entity(Message(
                SpecMessage.ENTITY_REGISTER.value,
                data={"skill_id": "s", "entity_name": "engine", "samples": []},
            ))
        self.assertNotIn("engine", p.entities)
        warn.assert_called()

    def test_unregistered_slot_left_literal(self):
        p = _make_prototype_pipeline()
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": "music.skill", "intent_name": "play_on",
                  "lang": "en-US", "samples": ["play on {engine}"]},
        ))
        # no entity registered -> single literal prototype, still accepted
        self.assertIn("music.skill:play_on", p.intents)


class TestIntent4Deregistration(unittest.TestCase):
    def _register(self, p, skill_id="music.skill", intent_name="play_music"):
        p._handle_intent4_register_template(Message(
            SpecMessage.INTENT_REGISTER_TEMPLATE.value,
            data={"skill_id": skill_id, "intent_name": intent_name,
                  "lang": "en-US", "samples": ["play music"]},
        ))

    def test_deregister_intent(self):
        p = _make_prototype_pipeline()
        self._register(p)
        p._handle_intent4_deregister_intent(Message(
            SpecMessage.INTENT_DEREGISTER.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"},
        ))
        self.assertNotIn("music.skill:play_music", p.intents)
        self.assertNotIn("music.skill:play_music", p.prototype_store.unique_labels)

    def test_deregister_skill_removes_all(self):
        p = _make_prototype_pipeline()
        self._register(p, intent_name="a")
        self._register(p, intent_name="b")
        self._register(p, skill_id="other.skill", intent_name="c")
        p._handle_intent4_deregister_skill(Message(
            SpecMessage.SKILL_DEREGISTER.value, data={"skill_id": "music.skill"},
        ))
        self.assertNotIn("music.skill:a", p.intents)
        self.assertNotIn("music.skill:b", p.intents)
        self.assertIn("other.skill:c", p.intents)

    def test_deregister_entity(self):
        p = _make_prototype_pipeline()
        p.entities["engine"] = ["spotify"]
        p._handle_intent4_deregister_entity(Message(
            SpecMessage.ENTITY_DEREGISTER.value,
            data={"skill_id": "music.skill", "entity_name": "engine", "lang": "en-US"},
        ))
        self.assertNotIn("engine", p.entities)

    def test_disable_then_enable(self):
        p = _make_prototype_pipeline()
        self._register(p)
        p._handle_intent4_disable(Message(
            SpecMessage.INTENT_DISABLE.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"},
        ))
        self.assertNotIn("music.skill:play_music", p.intents)
        p._handle_intent4_enable(Message(
            SpecMessage.INTENT_ENABLE.value,
            data={"skill_id": "music.skill", "intent_name": "play_music", "lang": "en-US"},
        ))
        self.assertIn("music.skill:play_music", p.intents)


if __name__ == "__main__":
    unittest.main()
