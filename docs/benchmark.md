# Benchmark

`ovos-m2v-pipeline` ships a comparative accuracy and speed benchmark in
`benchmark/compare.py`. It runs on two OpenVoiceOS evaluation datasets and
reports the model2vec engine side by side with the fixed baselines shared by
every OVOS intent engine, so results are comparable across the whole engine
family.

---

## Datasets

Both datasets are loaded from the Hugging Face Hub by `benchmark/dataset.py`.
Each has a `<lang>-templates` config (training templates) and a `<lang>-test`
config (labelled evaluation utterances). Every engine in this benchmark is a
template / sample matcher, so it trains on `-templates` and is evaluated on
`-test`.

| Name | Repo | Intents | Test cases | Notes |
|---|---|---|---|---|
| `intents-for-eval` | [`OpenVoiceOS/intents-for-eval`](https://huggingface.co/datasets/OpenVoiceOS/intents-for-eval) | 50 | 1750 | Six test splits, including a `far_ood` no-match set |
| `massive` | [`OpenVoiceOS/massive-templates`](https://huggingface.co/datasets/OpenVoiceOS/massive-templates) | 60 | 2974 | OVOS-templated rebuild of the MASSIVE corpus; one labelled split, no no-match cases |

`intents-for-eval` test splits:

| Split | Cases | What it tests |
|---|---|---|
| `template` | 500 | Utterances that fill a training template directly |
| `paraphrase` | 700 | Natural rephrasings — different words, same intent |
| `near_ood` | 400 | Boundary utterances close to another intent |
| `far_ood` | 50 | Genuinely off-topic — should match **nothing** |
| `asr_noise` | 50 | Speech-recognition artefacts |
| `typos` | 50 | Spelling errors |

`massive` has a single labelled `test` split and **no no-match cases** — so on
`massive` every engine has zero false positives by construction, and accuracy
equals recall.

### Entities

Each `{slot}` placeholder ships with example values. `benchmark/dataset.py`
collects them into a `Bundle.entities` map. The padaos / padatious / nebulento
baselines register them (the equivalent of a padatious `.entity` file) before
matching. The model2vec engine has no entity-registration API — it is an
embedding matcher, so the `{slot}` placeholders are embedded directly into the
intent prototypes.

---

## Engines Compared

| Engine | Description |
|---|---|
| `padaos` | Regex-based exact matcher (no fuzzy) — fixed baseline |
| `padatious` | Neural network matcher (requires a training pass) — fixed baseline |
| `nebulento damerau-levenshtein` | Flat fuzzy `IntentContainer`, default strategy — fixed baseline |
| `m2v flat-prototype` | `PrototypeIntentStore` — cosine nearest-neighbour over all intent prototypes |
| `m2v domain-prototype` | `DomainPrototypeIntentStore` — parallel-argmax, intents grouped by domain |
| `m2v hierarchical-prototype` | `HierarchicalPrototypeIntentStore` — two-stage, route to a domain then resolve the intent within it |

The first three rows are the **fixed baselines** shared by every OVOS intent
engine benchmark. The `m2v` rows are this repo's subject engine — one row per
prototype store variant it ships.

The model2vec encoder used for the `m2v` rows is
`minishlab/potion-multilingual-128M`, loaded as a bare `StaticModel` (embeddings
only, no trained classifier head — prototype mode). The first run downloads it
from the Hugging Face Hub. If the model cannot be fetched, the `m2v` rows skip
with a `[SKIP]` message and the baselines still run.

---

## Results — `intents-for-eval`

1750 cases (1700 match, 50 no-match), 50 intents across 10 domains.

Run `python benchmark/compare.py intents-for-eval` to generate the table. The
summary has these columns: Engine, Accuracy, Precision, Recall, F1, FP / 50,
Median latency.

---

## Results — `massive`

2974 cases, 60 intents across 18 domains. The corpus has no no-match cases, so
false positives are zero for every engine and accuracy equals recall — this
dataset measures recall on a broad, diverse intent set.

Run `python benchmark/compare.py massive` to generate the table.

---

## How to Run

Install benchmark dependencies:

```bash
pip install ovos-m2v-pipeline[benchmark]
# installs: padaos, padatious, fann2==1.0.7, nebulento, datasets
```

Run both datasets:

```bash
python benchmark/compare.py
```

Or one at a time:

```bash
python benchmark/compare.py intents-for-eval
python benchmark/compare.py massive
```

The first run downloads each dataset from the Hugging Face Hub and the
model2vec encoder (all cached afterwards). Padatious requires a training pass;
the other engines start immediately.

---

## How Metrics Are Calculated

Source: `compute_metrics` in `benchmark/compare.py`.

- **Accuracy** = (TP + TN) / total
- **Precision** = TP / (TP + FP)
- **Recall** = TP / total_match_cases
- **F1** = 2 × precision × recall / (precision + recall)
- **FP** = no-match utterances incorrectly assigned an intent

A prediction is a TP when the predicted intent name exactly matches the
expected intent and `conf >= threshold` (0.5). A no-match case is correct only
when the engine returns `None` or a confidence below threshold.

---

## The m2v variants

All three `m2v` rows train on the dataset's templates and score test
utterances by cosine similarity against intent prototype embeddings.

- **`m2v flat-prototype`** — every intent's prototypes live in one
  `PrototypeIntentStore`; the global argmax cosine wins.
- **`m2v domain-prototype`** — intents are grouped into domains (domain ==
  skill_id). Every domain's sub-store scores in parallel and the global argmax
  over the flat union of per-intent scores wins. There is no separate routing
  stage.
- **`m2v hierarchical-prototype`** — a two-stage store: a top-level router
  picks the single best domain by per-domain fingerprint similarity, then only
  that domain's sub-store resolves the intent. The `domain_threshold` gate
  (default `0.0`, no gate) rejects queries that match no domain.
