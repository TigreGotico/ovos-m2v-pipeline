"""
Comparative accuracy + speed benchmark across intent engines.

Engines
-------
padaos      – regex-based matcher
padatious   – neural-network matcher (requires training pass)
nebulento   – fuzzy string matching engine (flat, damerau-levenshtein)
m2v         – this repo's model2vec embedding-based prototype engine
              (flat / hierarchical two-stage)

Every engine here is a template / sample matcher: it trains on example
sentences, not keyword vocabularies. They are evaluated on two OpenVoiceOS
datasets — ``intents-for-eval`` and ``massive`` — each engine training on the
``<lang>-templates`` config and evaluating on ``<lang>-test``. See
``benchmark/dataset.py``.

The padaos / padatious / nebulento rows are the fixed baselines shared by
every OVOS intent engine benchmark, so results are comparable across the
whole engine family. The ``m2v`` rows are this repo's own engine.

Usage
-----
    python benchmark/compare.py
    python benchmark/compare.py intents-for-eval
    python benchmark/compare.py massive
"""
import sys
import time
import tempfile
import statistics
import logging
from collections import defaultdict

# Ensure this repo's ``benchmark/`` resolves before any sibling repo's editable
# install — without this, ``from benchmark.dataset`` can shadow into another
# pipeline plugin's package when several are installed in the same venv.
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))

from benchmark.dataset import DATASETS, load  # noqa: E402

logging.disable(logging.CRITICAL)

_CI_MODE = "--ci" in sys.argv

#: model2vec encoder used for the m2v rows. A bare ``StaticModel`` — no
#: trained classifier head — embeddings only, exactly the prototype mode.
#: ``potion-retrieval-32M`` is the English-tuned retrieval/similarity model,
#: which is the regime intent matching lives in (cosine over prototypes /
#: linear classifier over the same embedding space).
M2V_MODEL = "minishlab/potion-base-32M"


def normalize_utterance(utt):
    """Lowercase + collapse whitespace — engine-agnostic light normalisation."""
    return " ".join(str(utt).lower().split())


# ── shared helpers ─────────────────────────────────────────────────────────

def all_cases(bundle):
    """Flatten a :class:`~benchmark.dataset.Bundle` into ``(utterance, expected)``."""
    cases = []
    for name, data in bundle.intents.items():
        for utt in data["test_match"]:
            cases.append((utt, name))
    for utt in bundle.no_match:
        cases.append((utt, None))
    return cases


def compute_metrics(results, cases):
    total     = len(cases)
    match_n   = sum(1 for _, e in cases if e is not None)
    nomatch_n = total - match_n
    tp = fp = fn = tn = 0
    per_tp = defaultdict(int)
    per_fn = defaultdict(int)
    per_fp = defaultdict(int)
    wrong  = []
    for (predicted, conf), (utt, expected) in zip(results, cases):
        if expected is not None:
            if predicted == expected:
                tp += 1
                per_tp[expected] += 1
            else:
                fn += 1
                per_fn[expected] += 1
                wrong.append((utt, expected, predicted, conf))
        else:
            if predicted is not None:
                fp += 1
                per_fp[predicted] += 1
                wrong.append((utt, expected, predicted, conf))
            else:
                tn += 1
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / match_n   if match_n   else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return dict(
        accuracy=(tp + tn) / total if total else 0.0,
        precision=precision, recall=recall, f1=f1,
        fp=fp, fn=fn, match_n=match_n, nomatch_n=nomatch_n,
        per_tp=per_tp, per_fn=per_fn, per_fp=per_fp, wrong=wrong,
    )


def _stats_lines(label, metrics, latencies, intents, train_ms=None):
    s = sorted(latencies)
    total = metrics['match_n'] + metrics['nomatch_n']
    nomatch_n = metrics['nomatch_n']
    match_n = metrics['match_n']
    fp_pct = f"  ({metrics['fp']/nomatch_n:.0%} of no-match)" if nomatch_n else ""
    fn_pct = f"  ({metrics['fn']/match_n:.0%} of match)" if match_n else ""
    lines = [
        f"{'='*64}",
        f"  {label}",
        f"{'='*64}",
    ]
    if train_ms is not None:
        lines.append(f"  Train time: {train_ms:.0f} ms")
    lines += [
        f"  Accuracy  : {metrics['accuracy']:.1%}  ({int(metrics['accuracy']*total)}/{total})",
        f"  Precision : {metrics['precision']:.1%}",
        f"  Recall    : {metrics['recall']:.1%}",
        f"  F1        : {metrics['f1']:.3f}",
        f"  FP        : {metrics['fp']} / {nomatch_n}{fp_pct}",
        f"  FN        : {metrics['fn']} / {match_n}{fn_pct}",
        f"  Latency   : median={statistics.median(latencies):.2f}ms  "
        f"p95={s[int(len(s)*.95)]:.2f}ms  max={s[-1]:.2f}ms",
    ]
    issues = sorted(set(metrics['per_fn']) | set(metrics['per_fp']))
    if issues:
        lines.append("")
        lines.append("  Per-intent (issues only):")
        for i in sorted(intents):
            fn = metrics['per_fn'].get(i, 0)
            fp = metrics['per_fp'].get(i, 0)
            tp = metrics['per_tp'].get(i, 0)
            if fn or fp:
                rec = tp / (tp + fn) if (tp + fn) else 0
                lines.append(f"    {i:<28}  recall={rec:.0%}  fn={fn}  fp={fp}")
    return lines


def print_report(label, metrics, latencies, intents, train_ms=None):
    lines = _stats_lines(label, metrics, latencies, intents, train_ms)
    if _CI_MODE:
        acc = metrics['accuracy']
        fp  = metrics['fp']
        med = statistics.median(latencies)
        print("<details>")
        print(f"<summary><b>{label}</b> &mdash; acc {acc:.1%} &middot; "
              f"FP {fp} &middot; median {med:.2f}ms</summary>")
        print()
        print("```text")
        for line in lines:
            print(line)
        print("```")
        print()
        print("</details>")
        print()
    else:
        for line in lines:
            print(line)


# ── engine runners ─────────────────────────────────────────────────────────

def run_padaos(bundle, cases):
    import padaos
    c = padaos.IntentContainer()
    for entity_name, samples in bundle.entities.items():
        c.add_entity(entity_name, samples)
    for name, data in bundle.intents.items():
        c.add_intent(name, data["train"])
    t0 = time.perf_counter()
    c.compile()
    train_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for utt, _ in cases:
        q = normalize_utterance(utt)
        t0 = time.perf_counter()
        r  = c.calc_intent(q)
        latencies.append((time.perf_counter() - t0) * 1000)
        results.append((r.get("name"), 1.0 if r.get("name") else 0.0))

    m = compute_metrics(results, cases)
    print_report("padaos  (regex, no fuzz)", m, latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_padatious(bundle, cases, threshold=0.5):
    from padatious import IntentContainer as PC
    with tempfile.TemporaryDirectory() as d:
        c = PC(cache_dir=d)
        for entity_name, samples in bundle.entities.items():
            c.add_entity(entity_name, samples)
        for name, data in bundle.intents.items():
            c.add_intent(name, data["train"])
        t0 = time.perf_counter()
        c.train(single_thread=True, debug=False)
        train_ms = (time.perf_counter() - t0) * 1000

        results, latencies = [], []
        for utt, _ in cases:
            t0 = time.perf_counter()
            r  = c.calc_intent(normalize_utterance(utt))
            latencies.append((time.perf_counter() - t0) * 1000)
            predicted = r.name if (r and r.conf >= threshold) else None
            results.append((predicted, r.conf if r else 0.0))

    m = compute_metrics(results, cases)
    print_report(f"padatious  (neural, threshold={threshold})", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_nebulento(bundle, cases, strategy_name="DAMERAU_LEVENSHTEIN_SIMILARITY",
                  threshold=0.5):
    try:
        from nebulento import IntentContainer
        from nebulento.fuzz import MatchStrategy
    except ImportError:
        print("  [SKIP] nebulento — package unavailable")
        return None
    strategy = getattr(MatchStrategy, strategy_name)
    c = IntentContainer(fuzzy_strategy=strategy)
    try:
        for entity_name, samples in bundle.entities.items():
            c.add_entity(entity_name, samples)
        for name, data in bundle.intents.items():
            c.add_intent(name, data["train"])
    except Exception as exc:
        print(f"  [SKIP] nebulento — registration failed: {exc}")
        return None

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        r  = c.calc_intent(utt)
        latencies.append((time.perf_counter() - t0) * 1000)
        predicted = r.get("name") if (r and r.get("conf", 0) >= threshold) else None
        results.append((predicted, r.get("conf", 0.0) if r else 0.0))

    m = compute_metrics(results, cases)
    label = f"nebulento  {strategy_name.lower().replace('_', '-')}"
    print_report(label, m, latencies, bundle.intents)
    return m, statistics.median(latencies), statistics.mean(latencies), None


# ── m2v engine runners ─────────────────────────────────────────────────────

def _load_m2v_model():
    """Load the bare model2vec encoder; ``None`` if it cannot be fetched.

    Mirrors how ``run_padatious`` degrades — if the environment blocks the
    model download the m2v rows skip with a clear message rather than
    crashing, and the shared baselines still run.
    """
    try:
        from model2vec import StaticModel
        return StaticModel.from_pretrained(M2V_MODEL)
    except Exception as exc:  # network / missing dep / cache miss
        print(f"  [SKIP] m2v — could not load model {M2V_MODEL!r}: {exc}")
        return None


_SLOT_RE = __import__("re").compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _fill_slots(template, entities, max_fills=4, rng=None):
    """Expand a template's ``{slot}`` placeholders with concrete values.

    Trained classifiers learn precise decision boundaries on token embeddings,
    so a literal ``{song}`` token in training never matches a real song name at
    eval time. Substituting concrete values from ``entities`` (the dataset's
    own slot examples) gives the classifier realistic embeddings to fit on.
    A small per-template cap keeps the training set bounded.
    """
    import random
    rng = rng or random.Random(0)
    slots = _SLOT_RE.findall(template)
    if not slots:
        return [template]
    # one fill per chosen sample value combination, capped
    out, seen = [], set()
    for _ in range(max_fills):
        s = template
        for slot in slots:
            values = entities.get(slot) or [slot]
            s = s.replace("{" + slot + "}", rng.choice(values), 1)
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _build_training_set(bundle):
    """Yield ``(sentence, label)`` pairs with slots filled from bundle entities."""
    import random
    rng = random.Random(0)
    for name, data in bundle.intents.items():
        for template in data["train"]:
            for s in _fill_slots(template, bundle.entities, max_fills=4, rng=rng):
                yield s, name


def _domain_of(label):
    """Domain prefix from a ``<domain>.<intent>`` or ``<skill_id>:<intent>`` label.

    The benchmark datasets use different separators: ``intents-for-eval`` ships
    ``media.play_song``-style labels (``.`` separator) and ``massive`` ships
    ``alarm:alarm_set``-style (``:`` separator). The benchmark needs to extract
    the leading domain in both cases.
    """
    for sep in (".", ":"):
        if sep in label:
            return label.split(sep, 1)[0]
    return label


def run_m2v(bundle, cases, model=None, threshold=0.5, strategy=None):
    """Flat ``PrototypeIntentStore`` — one row per :class:`PrototypeStrategy`."""
    if model is None:
        print("  [SKIP] m2v (flat prototype) — model unavailable")
        return None
    from ovos_m2v_pipeline import PrototypeIntentStore
    from ovos_m2v_pipeline.strategies import PrototypeStrategy
    strategy = PrototypeStrategy(strategy) if strategy else PrototypeStrategy.MAX_OVER_ALL

    sentences, labels = [], []
    for name, data in bundle.intents.items():
        for utt in data["train"]:
            sentences.append(utt)
            labels.append(name)

    t0 = time.perf_counter()
    store = PrototypeIntentStore.build(model, sentences, labels, strategy=strategy)
    train_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        emb = model.encode([utt])[0]
        scored = store.scores(emb)
        latencies.append((time.perf_counter() - t0) * 1000)
        if scored:
            label = max(scored, key=scored.get)
            conf = scored[label]
        else:
            label, conf = None, 0.0
        predicted = label if conf >= threshold else None
        results.append((predicted, conf))

    m = compute_metrics(results, cases)
    print_report(f"m2v  flat-prototype  {strategy.value}", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_m2v_hierarchical(bundle, cases, model=None, threshold=0.5,
                         domain_threshold=0.0, intent_strategy=None):
    """``HierarchicalPrototypeIntentStore`` — one row per :class:`PrototypeStrategy`."""
    if model is None:
        print("  [SKIP] m2v (hierarchical prototype) — model unavailable")
        return None
    from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore
    from ovos_m2v_pipeline.strategies import PrototypeStrategy
    intent_strategy = (PrototypeStrategy(intent_strategy)
                       if intent_strategy
                       else PrototypeStrategy.MAX_OVER_ALL)

    store = HierarchicalPrototypeIntentStore(
        intent_strategy=intent_strategy,
        domain_threshold=domain_threshold,
    )
    t0 = time.perf_counter()
    for name, data in bundle.intents.items():
        store.add(model, _domain_of(name), name, data["train"])
    train_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for utt, _ in cases:
        t0 = time.perf_counter()
        emb = model.encode([utt])[0]
        scored = store.scores(emb)
        latencies.append((time.perf_counter() - t0) * 1000)
        if scored:
            label = max(scored, key=scored.get)
            conf = scored[label]
        else:
            label, conf = None, 0.0
        predicted = label if conf >= threshold else None
        results.append((predicted, conf))

    m = compute_metrics(results, cases)
    print_report(
        f"m2v  hierarchical-prototype  {intent_strategy.value}",
        m, latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


# ── m2v trained-classifier runners (scikit-learn) ──────────────────────────

def _have_sklearn():
    try:
        import sklearn  # noqa: F401
        return True
    except Exception:
        return False


def _train_static_classifier(sentences, labels, base_model=M2V_MODEL):
    """Fine-tune a ``StaticModelForClassification`` on (sentence, label) pairs.

    This is the same recipe ``train/train_en.py`` uses for the shipped models —
    model2vec's purpose-built end-to-end trainer (fine-tunes the static embedding
    AND the linear head), not a sklearn LogisticRegression on frozen embeddings.
    """
    from model2vec.train import StaticModelForClassification
    clf = StaticModelForClassification.from_pretrained(model_name=base_model)
    clf.fit(sentences, labels, max_epochs=25)
    return clf


def run_m2v_trained_flat(bundle, cases, model=None, threshold=0.0):
    """Flat trained classifier — fine-tune model2vec end-to-end on every intent.

    Training sentences have ``{slot}`` placeholders filled from ``bundle.entities``
    so the classifier learns embeddings of real phrases, not literal braces.
    ``threshold=0.0`` returns top-1 argmax (no rejection); over many classes the
    softmax max is small even when correct.
    """
    if not _have_sklearn():
        print("  [SKIP] m2v (trained flat) — scikit-learn unavailable")
        return None
    try:
        from model2vec.train import StaticModelForClassification  # noqa: F401
    except ImportError:
        print("  [SKIP] m2v (trained flat) — model2vec[train] unavailable")
        return None
    import numpy as np

    sentences, labels = [], []
    for s, lbl in _build_training_set(bundle):
        sentences.append(s)
        labels.append(lbl)
    if len(set(labels)) < 2:
        print("  [SKIP] m2v (trained flat) — need >=2 intents to train")
        return None

    t0 = time.perf_counter()
    clf = _train_static_classifier(sentences, labels)
    train_ms = (time.perf_counter() - t0) * 1000

    test_utts = [utt for utt, _ in cases]
    t0 = time.perf_counter()
    probs = clf.predict_proba(test_utts)
    total_ms = (time.perf_counter() - t0) * 1000
    classes = list(clf.classes)

    results, latencies = [], []
    for row in probs:
        idx = int(row.argmax())
        conf = float(row[idx])
        label = str(classes[idx])
        latencies.append(total_ms / len(test_utts))
        predicted = label if conf >= threshold else None
        results.append((predicted, conf))

    m = compute_metrics(results, cases)
    print_report("m2v  trained  flat", m, latencies, bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_m2v_trained_domain(bundle, cases, model=None, threshold=0.0):
    """Domain (parallel-argmax) trained classifier — one classifier per domain,
    each fine-tuned end-to-end with ``StaticModelForClassification``. No
    top-level router: every head scores every query and a global argmax over
    their softmax outputs picks the winner.

    Same recipe as ``run_m2v_trained_flat`` (fills slot placeholders, top-1
    argmax), but trains one intent classifier per domain on its own subset.
    """
    if not _have_sklearn():
        print("  [SKIP] m2v (trained domain) — scikit-learn unavailable")
        return None
    try:
        from model2vec.train import StaticModelForClassification  # noqa: F401
    except ImportError:
        print("  [SKIP] m2v (trained domain) — model2vec[train] unavailable")
        return None

    sentences, labels, domains = [], [], []
    for s, lbl in _build_training_set(bundle):
        sentences.append(s)
        labels.append(lbl)
        domains.append(_domain_of(lbl))
    if len(set(labels)) < 2:
        print("  [SKIP] m2v (trained domain) — need >=2 intents to train")
        return None

    t0 = time.perf_counter()
    by_domain = defaultdict(list)
    for s, lbl, dom in zip(sentences, labels, domains):
        by_domain[dom].append((s, lbl))
    intent_clfs = {}
    for dom, pairs in by_domain.items():
        intent_labels = sorted({lbl for _, lbl in pairs})
        if len(intent_labels) < 2:
            intent_clfs[dom] = ("single", intent_labels[0])
            continue
        intent_clfs[dom] = ("clf", _train_static_classifier(
            [s for s, _ in pairs], [lbl for _, lbl in pairs]))
    train_ms = (time.perf_counter() - t0) * 1000

    test_utts = [utt for utt, _ in cases]
    t0 = time.perf_counter()
    # Score every utterance with every per-domain head; global argmax wins.
    # Per-classifier softmax outputs are NOT directly comparable — a 2-class
    # head's max sits near 0.5 even on random inputs, while a 10-class head's
    # max sits near 0.1. We rank by margin above the random baseline
    # (max_prob - 1/num_classes), so off-topic uniform predictions across
    # heads of any size score ~0 and don't fight for the argmax.
    best_label = [None] * len(test_utts)
    best_conf = [0.0] * len(test_utts)
    best_score = [-1.0] * len(test_utts)
    for dom, (kind, payload) in intent_clfs.items():
        if kind == "single":
            # A single-label head has random baseline 1.0, so its margin is
            # always 0 — it only wins where no multi-label head has scored,
            # which is handled in the fill-in pass below.
            continue
        iprobs = payload.predict_proba(test_utts)
        iclasses = list(payload.classes)
        baseline = 1.0 / len(iclasses)
        for i, row in enumerate(iprobs):
            idx = int(row.argmax())
            conf = float(row[idx])
            score = conf - baseline
            if score > best_score[i]:
                best_score[i] = score
                best_conf[i] = conf
                best_label[i] = str(iclasses[idx])
    # Single-label-domain fill-in: only when nothing multi-class scored.
    for dom, (kind, payload) in intent_clfs.items():
        if kind != "single":
            continue
        for i in range(len(test_utts)):
            if best_label[i] is None:
                best_label[i] = payload
                best_conf[i] = 1.0
    total_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for label, conf in zip(best_label, best_conf):
        latencies.append(total_ms / len(test_utts))
        predicted = label if (label is not None and conf >= threshold) else None
        results.append((predicted, conf))

    m = compute_metrics(results, cases)
    print_report("m2v  trained  domain", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


def run_m2v_trained_hierarchical(bundle, cases, model=None, threshold=0.0,
                                 domain_threshold=0.0):
    """Two-stage trained classifier — domain + per-domain intent classifiers,
    each fine-tuned end-to-end with ``StaticModelForClassification``.

    Same recipe as ``run_m2v_trained_flat`` (fills slot placeholders and uses
    top-1 argmax), but trains one domain classifier and one intent classifier
    per domain. At inference: classify domain, then resolve intent only inside
    that domain.
    """
    if not _have_sklearn():
        print("  [SKIP] m2v (trained hierarchical) — scikit-learn unavailable")
        return None
    try:
        from model2vec.train import StaticModelForClassification  # noqa: F401
    except ImportError:
        print("  [SKIP] m2v (trained hierarchical) — model2vec[train] unavailable")
        return None

    sentences, labels, domains = [], [], []
    for s, lbl in _build_training_set(bundle):
        sentences.append(s)
        labels.append(lbl)
        domains.append(_domain_of(lbl))
    if len(set(labels)) < 2:
        print("  [SKIP] m2v (trained hierarchical) — need >=2 intents to train")
        return None

    t0 = time.perf_counter()
    # top-level domain classifier
    domain_clf = _train_static_classifier(sentences, domains)
    # one intent classifier per domain (skip domains with a single intent —
    # nothing to learn there, the domain prediction alone resolves it).
    by_domain = defaultdict(list)
    for s, lbl, dom in zip(sentences, labels, domains):
        by_domain[dom].append((s, lbl))
    intent_clfs = {}
    for dom, pairs in by_domain.items():
        intent_labels = sorted({lbl for _, lbl in pairs})
        if len(intent_labels) < 2:
            intent_clfs[dom] = ("single", intent_labels[0])
            continue
        intent_clfs[dom] = ("clf", _train_static_classifier(
            [s for s, _ in pairs], [lbl for _, lbl in pairs]))
    train_ms = (time.perf_counter() - t0) * 1000

    test_utts = [utt for utt, _ in cases]
    t0 = time.perf_counter()
    domain_probs = domain_clf.predict_proba(test_utts)
    domain_classes = list(domain_clf.classes)
    # for each predicted domain, batch the queries that route there
    per_domain_idxs = defaultdict(list)
    chosen_domain, domain_conf = [], []
    for i, row in enumerate(domain_probs):
        idx = int(row.argmax())
        dconf = float(row[idx])
        dname = str(domain_classes[idx])
        if dconf < domain_threshold:
            chosen_domain.append(None)
        else:
            chosen_domain.append(dname)
            per_domain_idxs[dname].append(i)
        domain_conf.append(dconf)
    predicted_label = [None] * len(test_utts)
    predicted_conf = [0.0] * len(test_utts)
    for dom, idxs in per_domain_idxs.items():
        kind, payload = intent_clfs.get(dom, ("missing", None))
        if kind == "single":
            for i in idxs:
                predicted_label[i] = payload
                predicted_conf[i] = domain_conf[i]
        elif kind == "clf":
            iprobs = payload.predict_proba([test_utts[i] for i in idxs])
            iclasses = list(payload.classes)
            for k, i in enumerate(idxs):
                idx = int(iprobs[k].argmax())
                predicted_label[i] = str(iclasses[idx])
                predicted_conf[i] = float(iprobs[k][idx])
    total_ms = (time.perf_counter() - t0) * 1000

    results, latencies = [], []
    for label, conf in zip(predicted_label, predicted_conf):
        latencies.append(total_ms / len(test_utts))
        predicted = label if (label is not None and conf >= threshold) else None
        results.append((predicted, conf))

    m = compute_metrics(results, cases)
    print_report("m2v  trained  hierarchical", m, latencies,
                 bundle.intents, train_ms)
    return m, statistics.median(latencies), statistics.mean(latencies), train_ms


# ── summary table ──────────────────────────────────────────────────────────

def summary(title, rows):
    """rows: list of (label, metrics, median_lat_ms, mean_lat_ms, train_ms_or_None)"""
    if _CI_MODE:
        print(f"## {title}\n")
        print("| Engine | Acc | Prec | Recall | F1 | FP | Median |")
        print("|---|---|---|---|---|---|---|")
        for label, m, median_lat, mean_lat, _ in rows:
            print(f"| {label} | {m['accuracy']:.1%} | {m['precision']:.1%} | "
                  f"{m['recall']:.1%} | {m['f1']:.3f} | {m['fp']} | {median_lat:.2f}ms |")
        print()
        print("_FP = false positives on no-match_")
    else:
        print(f"\n\n{'─'*84}")
        print(f"  {title}")
        print(f"  {'Engine':<36} {'Acc':>6} {'Prec':>6} {'Recall':>7} {'F1':>6}  "
              f"{'FP':>4}  {'Median':>8}  {'Mean':>8}")
        print(f"{'─'*84}")
        for label, m, median_lat, mean_lat, train_ms in rows:
            print(f"  {label:<36} {m['accuracy']:>5.1%} {m['precision']:>5.1%} "
                  f"{m['recall']:>6.1%} {m['f1']:>5.3f}  {m['fp']:>4}  "
                  f"{median_lat:>6.2f}ms  {mean_lat:>6.2f}ms")
        print(f"{'─'*84}")
        print("  FP = false positives on no-match | Median/Mean = query latency in ms")


# ── main ───────────────────────────────────────────────────────────────────

def run_dataset(name):
    bundle = load(name)
    cases = all_cases(bundle)
    match_n = sum(1 for _, e in cases if e is not None)
    print(f"\nDataset : {bundle.repo}  ({bundle.lang})")
    print(f"Cases   : {len(cases)}  ({match_n} match, {len(cases)-match_n} no-match)")
    print(f"Intents : {len(bundle.intents)}  across {len(bundle.domains)} domains")
    print("Splits  : " + ", ".join(f"{k}={len(v)}" for k, v in bundle.splits.items()))

    rows = []

    # ── fixed baselines (shared across the OVOS intent-engine family) ──
    m, lat, mean_lat, tr = run_padaos(bundle, cases)
    rows.append(("padaos  (regex)", m, lat, mean_lat, tr))

    m, lat, mean_lat, tr = run_padatious(bundle, cases, threshold=0.5)
    rows.append(("padatious  neural  threshold=0.5", m, lat, mean_lat, tr))

    res = run_nebulento(bundle, cases, threshold=0.5)
    if res is not None:
        m, lat, mean_lat, tr = res
        rows.append(("nebulento  damerau-levenshtein", m, lat, mean_lat, tr))

    # ── subject — this repo's model2vec embedding engine, every variant ──
    # one row per PrototypeStrategy, for both flat and hierarchical
    model = _load_m2v_model()
    from ovos_m2v_pipeline.strategies import PrototypeStrategy

    for strat in PrototypeStrategy:
        res = run_m2v(bundle, cases, model=model, threshold=0.5, strategy=strat)
        if res is not None:
            m, lat, mean_lat, tr = res
            rows.append((f"m2v  flat  {strat.value}", m, lat, mean_lat, tr))

    for strat in PrototypeStrategy:
        res = run_m2v_hierarchical(bundle, cases, model=model, threshold=0.5,
                                   domain_threshold=0.0, intent_strategy=strat)
        if res is not None:
            m, lat, mean_lat, tr = res
            rows.append((f"m2v  hierarchical  {strat.value}",
                         m, lat, mean_lat, tr))

    # ── trained classifier rows (flat + hierarchical) ──
    res = run_m2v_trained_flat(bundle, cases, model=model, threshold=0.5)
    if res is not None:
        m, lat, mean_lat, tr = res
        rows.append(("m2v  trained  flat", m, lat, mean_lat, tr))

    res = run_m2v_trained_domain(bundle, cases, model=model, threshold=0.5)
    if res is not None:
        m, lat, mean_lat, tr = res
        rows.append(("m2v  trained  domain", m, lat, mean_lat, tr))

    res = run_m2v_trained_hierarchical(bundle, cases, model=model,
                                       threshold=0.5, domain_threshold=0.0)
    if res is not None:
        m, lat, mean_lat, tr = res
        rows.append(("m2v  trained  hierarchical", m, lat, mean_lat, tr))

    summary(f"{name}  —  {bundle.repo}", rows)


if __name__ == "__main__":
    selected = [a for a in sys.argv[1:] if not a.startswith("-")]
    targets = selected or list(DATASETS)
    for dataset_name in targets:
        if dataset_name not in DATASETS:
            print(f"unknown dataset {dataset_name!r}; choose from {list(DATASETS)}")
            continue
        run_dataset(dataset_name)
