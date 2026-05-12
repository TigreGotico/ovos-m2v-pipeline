"""Live E2E test: run the en-US intent fixture (drawn from
`ovos-localize/data/datasets/classification/en-US.jsonl`, official OVOS
skills only) through the real model2vec pipeline on a MiniCroft instance.

Gated behind `OVOSCOPE_LIVE=1` because it downloads the model from
HuggingFace (default
`Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2`) and
takes a minute to initialise. CI opts in by setting the env var.

For each fixture line (`{"label": "<skill:intent>", "utterance": "..."}`)
the test sets `pipeline.intents` to the union of fixture labels and asserts
that the intent dispatched by `IntentService._emit_match_message` matches
the expected label. Allows up to 20% drift since the model is not pinned.
"""
import json
import os
import threading
import unittest
from pathlib import Path

import pytest

if os.environ.get("OVOSCOPE_LIVE") != "1":
    pytest.skip(
        "Live model test skipped; set OVOSCOPE_LIVE=1 to enable.",
        allow_module_level=True,
    )

pytest.importorskip("ovoscope", reason="ovoscope not installed")

from ovos_bus_client.message import Message  # noqa: E402
from ovos_bus_client.session import Session  # noqa: E402
from ovos_config.config import Configuration  # noqa: E402
from ovoscope import get_minicroft  # noqa: E402

FIXTURE = Path(__file__).parent / "fixtures" / "en_us_intents.jsonl"
PIPELINE_ID = "ovos-m2v-pipeline"
CONFIG_KEY = "ovos_m2v_pipeline"


def _load_fixture():
    cases = []
    with FIXTURE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cases.append(json.loads(line))
    return cases


class TestLiveFixture(unittest.TestCase):
    """Real model, real bus, real dispatch."""

    @classmethod
    def setUpClass(cls):
        cls.cases = _load_fixture()
        cls.all_labels = sorted({c["label"] for c in cls.cases})

        cfg = Configuration()
        intents_cfg = cfg.setdefault("intents", {})
        cls._orig = intents_cfg.get(CONFIG_KEY)
        intents_cfg[CONFIG_KEY] = {"renormalize": False, "conf_low": 0.0}

        cls.mc = get_minicroft(
            skill_ids=[],
            lang="en-US",
            default_pipeline=[PIPELINE_ID],
            max_wait=300,  # model download can be slow on first run
        )
        cls.pipeline = cls.mc.intents.pipeline_plugins[PIPELINE_ID]
        cls.pipeline.intents = list(cls.all_labels)

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

    def _emit(self, utterance: str, expected_label: str, timeout: float = 15.0):
        got: list[Message] = []
        done = threading.Event()

        def _capture(msg):
            got.append(msg)
            done.set()

        def _fail(_msg):
            done.set()

        self.mc.bus.on(expected_label, _capture)
        self.mc.bus.on("complete_intent_failure", _fail)
        sess = Session(session_id="live", pipeline=[PIPELINE_ID])
        try:
            self.mc.bus.emit(Message(
                "recognizer_loop:utterance",
                data={"utterances": [utterance], "lang": "en-US"},
                context={"session": sess.serialize()},
            ))
            done.wait(timeout=timeout)
        finally:
            self.mc.bus.remove(expected_label, _capture)
            self.mc.bus.remove("complete_intent_failure", _fail)
        return got[0] if got else None

    def test_fixture_top_intent_matches_label(self):
        misses = []
        for case in self.cases:
            utt, expected = case["utterance"], case["label"]
            msg = self._emit(utt, expected)
            if msg is None:
                misses.append((utt, expected, "no match"))
                continue
            if msg.msg_type != expected:
                misses.append((utt, expected, msg.msg_type))

        total = len(self.cases)
        passed = total - len(misses)
        accuracy = passed / total if total else 0.0
        max_misses = max(1, total // 5)
        ok = len(misses) <= max_misses

        report = _build_report(
            pipeline_id=PIPELINE_ID,
            total=total,
            passed=passed,
            accuracy=accuracy,
            tolerance_pct=20,
            misses=misses,
            ok=ok,
        )
        out_path = os.environ.get("LIVE_REPORT_PATH", "live_test_report.md")
        try:
            Path(out_path).write_text(report)
        except OSError:
            pass
        print("\n" + report)

        self.assertTrue(
            ok,
            f"{len(misses)}/{total} fixture cases misclassified (max allowed "
            f"{max_misses}):\n"
            + "\n".join(f"  {u!r} expected={e} got={g}" for u, e, g in misses),
        )


def _build_report(pipeline_id, total, passed, accuracy, tolerance_pct,
                  misses, ok):
    status = "✅ PASS" if ok else "❌ FAIL"
    lines = [
        f"## Live fixture test — `{pipeline_id}`",
        "",
        f"**Status:** {status}",
        f"**Accuracy:** {passed}/{total} ({accuracy:.1%}) — tolerance ≤ {tolerance_pct}% drift",
        "",
    ]
    if misses:
        lines.append("### Misclassifications")
        lines.append("")
        lines.append("| utterance | expected | got |")
        lines.append("|---|---|---|")
        for u, e, g in misses[:30]:
            lines.append(f"| `{u}` | `{e}` | `{g}` |")
        if len(misses) > 30:
            lines.append(f"\n_…and {len(misses) - 30} more_")
    else:
        lines.append("No misclassifications.")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    unittest.main()
