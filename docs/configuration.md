# Configuration

The pipeline is configured inside `mycroft.conf` under the `intents` section.

## Minimal Configuration — Classifier Mode

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE"
    }
  }
}
```

## Minimal Configuration — Prototype Mode

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "mode": "prototype",
      "model": "minishlab/M2V_multilingual_output"
    }
  }
}
```

Any bare `StaticModel` on Hugging Face (or a local path) can be used as the embedding backbone for prototype mode — no classifier head is needed.

## Full Configuration Reference

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
      "mode": "classifier",
      "prototype_k": 5,
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": [],
      "timeout": 1
    }
  }
}
```

## Options

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `model` | `str` | `"Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2"` | Hugging Face repo ID or local path. In classifier mode this must be a `StaticModelPipeline`; in prototype mode any bare `StaticModel` works. |
| `mode` | `str` | `"classifier"` | Operating mode: `"classifier"` or `"prototype"`. |
| `prototype_k` | `int` | `5` | Maximum number of prototype embeddings stored per intent label (prototype mode only). |
| `conf_high` | `float` | `0.7` | Minimum score for a `match_high` result. |
| `conf_medium` | `float` | `0.5` | Minimum score for a `match_medium` result. |
| `conf_low` | `float` | `0.15` | Minimum score for a `match_low` result. |
| `ignore_intents` | `list[str]` | `[]` | Intent labels to always discard, regardless of confidence. |
| `timeout` | `int` | `1` | Seconds to wait for Adapt / Padatious manifest responses (classifier mode only). |

## Confidence Thresholds

Each `conf_*` key sets the minimum score required for the corresponding tier method to return a match:

- **`conf_high`** — threshold for `match_high()`, called when `ovos-m2v-pipeline-high` appears in the pipeline list.
- **`conf_medium`** — threshold for `match_medium()`, called when `ovos-m2v-pipeline-medium` appears.
- **`conf_low`** — threshold for `match_low()`, called when `ovos-m2v-pipeline-low` appears.

OVOS evaluates the pipeline list top-to-bottom and stops at the first match. Only the tiers you add to the pipeline list are ever invoked — unused tiers consume no resources.

In classifier mode the scores are softmax probabilities (sum to 1 across all labels). In prototype mode the scores are cosine similarities (typically 0–1); you may need to tune thresholds downward.

Only the top-ranked valid intent is returned at each tier. If its score is below the threshold, `None` is returned and the OVOS pipeline moves to the next entry.

## Ignoring Intents

Use `ignore_intents` to suppress specific labels that cause repeated false matches:

```json
{
  "ignore_intents": [
    "ovos-skill-hello-world.openvoiceos:Greetings.intent"
  ]
}
```

Labels in this list are filtered out before any confidence check.

## Using a Local Model

Set `model` to an absolute path to use a locally saved model:

```json
{
  "model": "/opt/models/m2v_intents_LaBSE"
}
```

See [Models](models.md) for available pre-trained models and [Training](training.md) to produce your own.
