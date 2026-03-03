# Training

The `train/` directory contains all scripts needed to gather data and train new models.

> Training is only needed if you want to produce a custom model. The [pre-trained models](models.md) cover the standard OVOS skill corpus.

---

## Overview

```
train/
├── gather_dataset.py          # Download & merge multilingual intent examples
├── gather_dataset_en.py       # Download & merge English-only intent examples
├── train_multilingual.py      # Train a multilingual classifier
├── train_en.py                # Train English classifiers (multiple base models)
├── distill.py                 # Distill a Sentence Transformer into a Model2Vec base
├── predict.py                 # Quick smoke-test / inference demo
├── merged_intents_dataset.csv     # Output of gather_dataset.py
└── merged_intents_dataset_en.csv  # Output of gather_dataset_en.py
```

---

## Step 1 — Gather the Dataset

### Multilingual

```bash
cd train
python gather_dataset.py
```

This pulls from:
- `Jarbas/ovos_intent_examples` on Hugging Face (English examples)
- `Jarbas/music_queries_templates` on Hugging Face (OCP / music playback templates)
- Per-language intent CSVs from the [OpenVoiceOS lang-support-tracker](https://github.com/OpenVoiceOS/lang-support-tracker)

Languages included: `en`, `pt`, `eu`, `es`, `gl`, `nl`, `fr`, `de`, `ca`, `it`, `da`.

Output: `merged_intents_dataset.csv` with columns `lang`, `label`, `sentence`.

### English-only

```bash
python gather_dataset_en.py
```

Same sources but filtered to `lang=en`.

Output: `merged_intents_dataset_en.csv`.

---

## Dataset Schema

Each row in the CSV represents one training example:

| Column | Example |
|--------|---------|
| `lang` | `en` |
| `label` | `ovos-skill-date-time.openvoiceos:what.time.is.it.intent` |
| `sentence` | `what time is it` |

Labels follow the format `<skill_id>:<intent_name>`.

### Normalisation

`gather_dataset.py` applies the following normalisation before saving:

- **`sentence`**: lowercased, commas removed, multi-word separators collapsed, leading/trailing quotes stripped.
- **`domain`** (skill ID): `skill-ovos` prefix replaced with `ovos-skill`.
- **`intent`**: configurable replacements to merge near-duplicate intent names (e.g. `is_rain` → `do-i-need-an-umbrella.intent`).

Blacklists (`BLACKLIST_SKILLS`, `BLACKLIST_INTENTS`) allow excluding specific skills or intents.

---

## Step 2 — Train the Model

### Multilingual

```bash
python train_multilingual.py
```

- Base model: `minishlab/M2V_multilingual_output`
- Trains for 25 epochs with an 90/10 train/test split.
- Saves each trained pipeline to `model_mul_<base_model_name>/`.
- Writes evaluation metrics to `metrics_mul_<base_model_name>.md`.
- Writes a comparison table to `model_comparison.md`.

### English

```bash
python train_en.py
```

Trains five separate classifiers, one per Potion base model:

- `minishlab/potion-base-2M`
- `minishlab/potion-base-4M`
- `minishlab/potion-base-8M`
- `minishlab/potion-base-32M`
- `minishlab/potion-retrieval-32M`

Each is trained for 25 epochs. Outputs:
- `m2v_intents_<base_model_name>/` — saved pipeline
- `metrics_en_<base_model_name>.md` — per-model metrics
- `model_comparison_en.md` — comparison table across all English models

---

## Step 3 — (Optional) Distill a New Base Model

If you want to start from a Sentence Transformer that is not yet available as a Model2Vec distillate:

```bash
python distill.py
```

Edit the model list at the top of `distill.py`, then run the script. It calls `model2vec.distill.distill()` and saves the result locally. The distilled model can then be used as the `base_model` in the training scripts.

---

## Step 4 — Test the Trained Model

Use `predict.py` as a quick smoke test:

```python
# predict.py (edit the model path first)
model = StaticModelPipeline.from_pretrained("/path/to/m2v_intents_potion-base-2M")

inputs = ["do you know the time"]
predicted = model.predict(inputs)
probs = model.predict_proba(inputs)
```

Expected output:

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
m = StaticModelPipeline.from_pretrained('m2v_intents_potion-base-32M')
m.push_to_hub('YourOrg/your-model-name')
"
```

Then update the `model` key in your OVOS configuration to point at the new repo.
