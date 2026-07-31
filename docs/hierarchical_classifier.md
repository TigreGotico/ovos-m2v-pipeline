# Hierarchical Trained Classifier Pipeline

This page documents two layers that ship together:

* **`Model2VecHierarchicalIntentPipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-m2v-hierarchical-intent-pipeline`. Subclasses the flat classifier pipeline; the only difference is that matching is delegated to a two-stage trained classifier (domain router + per-domain intent heads).
* **`HierarchicalIntentClassifier`** — the supervised two-stage classifier used internally by that pipeline. Mirrors the architecture of [`HierarchicalPrototypeIntentStore`](hierarchical_store.md) but with scikit-learn `LogisticRegression` heads trained on top of the model2vec embedding instead of cosine matching over centroids.

A separate entry point keeps the trained-hierarchical pipeline independently selectable in `default_pipeline` ordering, with its own `intents.<key>` config block.

## Enabling

```json
{
  "intents": {
    "ovos-m2v-hierarchical-intent-pipeline": {
      "model": "minishlab/potion-multilingual-128M",
      "model_path": "/path/to/m2v_hier_intents_bundle",
      "domain_threshold": 0.2
    }
  }
}
```

* `model` — the bare `StaticModel` encoder used at training time.
* `model_path` — directory (or HF repo path) containing the saved `HierarchicalIntentClassifier` bundle.
* `domain_threshold` — optional override of the bundle's domain rejection gate.

## Architecture

```
              query embedding
                    │
                    ▼
        ┌───────────────────────┐
        │ domain LR classifier  │  stage 1: pick ONE domain
        │ (softmax over domains)│  argmax probability
        └───────────────────────┘
                    │
            domain_threshold gate
                    │
                    ▼
        ┌───────────────────────┐
        │ per-domain intent LR  │  stage 2: intent argmax
        │ (only routed domain)  │  conf = p(domain) * p(intent)
        └───────────────────────┘
```

## Bundle layout

`HierarchicalIntentClassifier.save(path)` produces:

```
<bundle>/
    manifest.json
    domain/
        classifier.joblib
    intent/
        <domain_a>/classifier.joblib
        <domain_b>/classifier.joblib
        ...
```

`manifest.json` records the saved `domain_threshold` and the list of domains. Each `classifier.joblib` is a single scikit-learn `LogisticRegression` fitted on raw model2vec embeddings.

## Programmatic use

```python
import numpy as np
from model2vec import StaticModel
from ovos_m2v_pipeline.hierarchical_classifier import HierarchicalIntentClassifier

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

sentences = ["play africa", "pause music", "lights on", "set thermostat to 21"]
labels    = ["media:play", "media:pause", "home:lights", "home:thermo"]

X = np.asarray(model.encode(sentences))
clf = HierarchicalIntentClassifier.train(X, labels, domain_threshold=0.2)
clf.save("my_bundle")

clf2 = HierarchicalIntentClassifier.load("my_bundle")
intent, conf = clf2.predict(model.encode(["put on africa"])[0])
```

## Training from a CSV

See [`train/train_hierarchical.py`](training.md#hierarchical-trained-classifier) for the CLI that consumes the same `merged_intents_dataset*.csv` as the flat trainer.

## See also

- [Training](training.md) — full training pipeline including the flat trainer.
- [Hierarchical Prototype Store](hierarchical_store.md) — unsupervised counterpart.
- [Pipeline Internals](pipeline.md) — how pipelines plug into the OVOS bus.
