# Training

The `train/` directory contains all scripts needed to gather data, distill base models, train classifiers, and benchmark results.

> Training is only needed if you want to produce a custom model. The [pre-trained models](models.md) cover the standard OVOS skill corpus.

---

## Pipeline Overview

Run the scripts in order:

```
1. gather_dataset.py              — Download & normalize multilingual intent examples
2. distill.py                     — (Optional) Distill a Sentence Transformer → Model2Vec base
3. train_multilingual.py          — Train one StaticModelPipeline per base model
4. train_monolingual.py           — (Optional) Train per-language models; compare vs multilingual
5. create_diverse_subset.py       — Select a diverse validation subset (greedy farthest-point)
6. benchmarks.py                  — Evaluate all trained models; produce benchmark report

Optional LLM augmentation loop (run between steps 1 and 3):
  1a. create_augmentation_targets.py — Analyse dataset gaps; produce JSON + Markdown targets
  1b. augment_dataset.py            — Call a local LLM (llama.cpp / Ollama) to fill gaps
  1c. gather_dataset.py             — Re-run to merge augmented sentences into the dataset
```

---

## Directory Layout

```
train/
├── gather_dataset.py            # Step 1
├── distill.py                   # Step 2 (optional)
├── train_multilingual.py        # Step 3
├── create_diverse_subset.py     # Step 4
├── benchmarks.py                # Step 5
├── predict.py                   # Quick smoke-test / inference demo
│
└── output/                      # All generated artefacts live here
    ├── dataset_cache/               # Downloaded source CSVs (MD5-hashed filenames)
    ├── dataset_plots/               # Exploratory plots produced by Step 1
    ├── by_lang/                     # Per-language CSV subsets from Step 1
    │
    ├── merged_intents_dataset.csv       # Compact output of Step 1 (label + sentence)
    ├── merged_intents_dataset_full.csv  # Expanded output of Step 1 (domain + intent + sentence)
    │
    ├── distilled/               # Distilled base models from Step 2
    │
    ├── model_mul_<name>/        # Saved StaticModelPipeline from Step 3
    ├── plots_<name>/            # Per-model evaluation plots from Step 3
    ├── metrics_<name>.md        # Per-model eval metrics from Step 3
    ├── model_comparison.md      # Cross-model comparison table from Step 3
    ├── model_sizes.png          # Size vs F1 comparison from Step 3
    │
    ├── diverse_subset.csv       # Output of Step 4 (compact)
    ├── diverse_subset_full.csv  # Output of Step 4 (expanded)
    ├── diverse_subset_stats.csv # Per-label selection counts from Step 4
    ├── diverse_plots/           # Coverage plots from Step 4
    │
    ├── benchmark_results.csv    # Raw benchmark numbers from Step 5
    ├── benchmark_report.md      # Formatted benchmark report from Step 5
    └── benchmark_plots/         # Cross-model heatmaps and scatter plots from Step 5
```

---

## Step 1 — Gather the Dataset

```bash
cd train
python gather_dataset.py
```

Downloads CSV files from HuggingFace and GitHub, normalises columns into a common schema, deduplicates, and writes output files.

**Sources:**

| Source | Languages |
|--------|-----------|
| `OpenVoiceOS/ovos-llm-augmented-intents` | en |
| `OpenVoiceOS/ovos-common-query-intents` | en |
| `OpenVoiceOS/ovos-intents-massive-subset` | en |
| `Jarbas/music_queries_templates` | en |
| `OpenVoiceOS/ovos-weather-intents` | en |
| Per-language CSVs from `OpenVoiceOS/lang-support-tracker` | pt, eu, es, gl, nl, fr, de, ca, it, da |

Downloaded CSVs are cached in `dataset_cache/` by MD5-hashed filename. Delete a file there to force a re-download.

**Outputs:**

| File | Columns | Description |
|------|---------|-------------|
| `merged_intents_dataset.csv` | `lang`, `label`, `sentence` | Compact format used by `train_multilingual.py` |
| `merged_intents_dataset_full.csv` | `lang`, `domain`, `intent`, `sentence` | Expanded format with separate skill/intent columns |
| `by_lang/intents_<lang>.csv` | `lang`, `label`, `sentence` | Per-language compact subset |
| `by_lang/intents_<lang>_full.csv` | `lang`, `domain`, `intent`, `sentence` | Per-language expanded subset |
| `dataset_plots/*.png` | — | 5 exploratory plots (see below) |

**Normalisation applied:**

- **`sentence`**: lowercased, commas removed, multi-word separators collapsed, leading/trailing quotes stripped. Rows that normalise to empty or bare `"nan"` are dropped.
- **`domain`** (skill ID): `skill-ovos-` prefix rewritten to `ovos-skill-`.
- **`intent`**: configurable alias map merges near-duplicate intent names (e.g. `is_rain` → `do-i-need-an-umbrella.intent`).
- **`label`**: `domain + ":" + intent` composite string.

Blacklists (`BLACKLIST_SKILLS`, `BLACKLIST_INTENTS`) in the script allow excluding specific skills or intents.

**Exploratory plots** saved to `dataset_plots/`:

| File | Description |
|------|-------------|
| `lang_distribution.png` | Examples per language (log scale) |
| `label_distribution.png` | Top-30 labels by example count (log scale) |
| `domain_distribution.png` | Examples per skill domain (log scale) |
| `label_size_histogram.png` | Histogram of per-label counts (reveals imbalance) |
| `lang_domain_heatmap.png` | Language × domain coverage — raw counts + row-normalised |

---

## Step 2 — (Optional) Distill a New Base Model

If you want to start from a Sentence Transformer that is not yet available as a Model2Vec distillate:

```bash
python distill.py
```

Edit the model list at the top of `distill.py` to add your Sentence Transformer, then run the script. It calls `model2vec.distill.distill()` and saves the result to `distilled/<model-name>/`. The distilled directory can then be used as the `path` in `train_multilingual.py`'s `base_models` list.

---

## Step 3 — Train the Classifier

```bash
python train_multilingual.py
```

For each entry in `base_models` (defined at the top of the script) this step:

1. Loads `merged_intents_dataset_full.csv` and synthesises a `label` column (`domain + ":" + intent`).
2. Optionally filters to a language subset (`langs` key in `base_models`; `None` = all languages).
3. Applies `balance_dataset()` — drops labels below `MIN_SAMPLES` and caps labels above `MAX_SAMPLES`.
4. Performs a balanced stratified 80/20 train/test split (`balanced_split()`).
5. Trains a `StaticModelPipeline` (25 epochs) on the training split.
6. Evaluates on the test split and saves metrics + plots.

**`base_models` format:**

```python
base_models = [
    {"path": "minishlab/potion-multilingual-128M", "langs": None},   # all languages
    {"path": "sentence-transformers/LaBSE",         "langs": None},
    {"path": _d("my-distilled-model"),              "langs": ["en", "pt"]},
]
```

`_d(name)` is a helper that expands to `train/distilled/<name>`.

**Outputs per model:**

| Path | Description |
|------|-------------|
| `model_mul_<name>/` | Saved `StaticModelPipeline` (ready for `from_pretrained`) |
| `metrics_<name>.md` | Accuracy, F1, per-class breakdown |
| `plots_<name>/per_class_f1.png` | Per-class F1 bar chart |
| `plots_<name>/per_language_metrics.png` | Accuracy / F1 by language |
| `model_comparison.md` | Ranked comparison table across all models |

---

## Step 4 — Create a Diverse Validation Subset

```bash
python create_diverse_subset.py
```

Uses greedy maximin farthest-point selection to pick a representative, spread-out subset from `merged_intents_dataset_full.csv`. This subset is used by `benchmarks.py` as an additional evaluation split.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_PER_LABEL` | 20 | Maximum examples per label to select |
| `MIN_PER_LABEL` | 1 | Minimum examples per label required to include it |

**Outputs:**

| File | Description |
|------|-------------|
| `diverse_subset.csv` | Compact format (`label`, `sentence`) |
| `diverse_subset_full.csv` | Expanded format (`domain`, `intent`, `sentence`) |
| `diverse_subset_stats.csv` | Per-label selection counts |
| `diverse_plots/` | Coverage and distribution plots |

---

## Step 5 — Benchmark All Models

```bash
python benchmarks.py
```

Evaluates every `model_mul_*/` directory on two datasets:

1. The full test split derived from `merged_intents_dataset_full.csv`.
2. The diverse subset from `diverse_subset.csv`.

**Outputs:**

| File | Description |
|------|-------------|
| `benchmark_results.csv` | Raw metrics for every model × dataset combination |
| `benchmark_report.md` | Formatted markdown summary table |
| `benchmark_plots/` | Cross-model accuracy and F1 heatmaps |

---

## Quick Inference Check

Use `predict.py` as a smoke test after training:

```bash
python predict.py
```

Edit the model path at the top of the file, then run. Expected output:

```
took 0.137 seconds to load model
took 0.003 seconds to predict
['ovos-skill-date-time.openvoiceos:what.time.is.it.intent']

Input: do you know the time
  ovos-skill-date-time.openvoiceos:what.time.is.it.intent: 0.9135
  ovos-skill-naptime.openvoiceos:naptime.intent: 0.0205
  ...
```

---

## Publishing to Hugging Face

Once satisfied with the model, push it to the Hub:

```bash
huggingface-cli login
python -c "
from model2vec.inference import StaticModelPipeline
m = StaticModelPipeline.from_pretrained('train/model_mul_LaBSE')
m.push_to_hub('YourOrg/your-model-name')
"
```

Then update the `model` key in your OVOS configuration to point at the new repo.

See [Models](models.md) for the available pre-trained models and [Configuration](configuration.md) to wire the model into OVOS.
