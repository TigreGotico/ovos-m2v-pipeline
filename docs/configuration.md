# Configuration

The pipeline is configured inside `mycroft.conf` under the `intents` section.

## Minimal Configuration

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE"
    }
  }
}
```

## Full Configuration Reference

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
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
| `model` | `str` | `"Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2"` | Hugging Face repo ID or local path to a Model2Vec `StaticModelPipeline`. |
| `conf_high` | `float` | `0.7` | Minimum probability for a `match_high` result. |
| `conf_medium` | `float` | `0.5` | Minimum probability for a `match_medium` result. |
| `conf_low` | `float` | `0.15` | Minimum probability for a `match_low` result. |
| `ignore_intents` | `list[str]` | `[]` | Intent labels to always discard, regardless of confidence. |
| `timeout` | `int` | `1` | Seconds to wait for Adapt / Padatious manifest responses from the bus. |

## Confidence Thresholds

The OVOS pipeline system calls matchers at three priority levels. The `conf_*` settings control how strict each level is:

- **`conf_high`** — used by `match_high()`. Raise this to reduce false positives at the highest priority tier.
- **`conf_medium`** — used by `match_medium()`. A balanced middle ground.
- **`conf_low`** — used by `match_low()`. The last-resort fallback; keep this low enough to catch weak but valid matches.

Only the top-ranked valid intent is returned at each level. If its probability is below the threshold, `None` is returned and the OVOS pipeline moves on.

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
  "model": "/opt/models/m2v_intents_potion-base-32M"
}
```

See [Models](models.md) for available pre-trained models and [Training](training.md) to produce your own.
