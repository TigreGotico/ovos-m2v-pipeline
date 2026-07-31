# Installation

## Install from PyPI

```bash
pip install ovos-m2v-pipeline
```

This installs the plugin and registers it as an `opm.pipeline` entry point under the name `ovos-m2v-pipeline`.

## Dependencies

| Package | Purpose |
|---------|---------|
| `model2vec[inference]` | Static embedding model and classification pipeline |
| `ovos-plugin-manager` | OVOS plugin infrastructure and base pipeline class |
| `ovos-bus-client` | Message bus communication |
| `ovos-config` | Reading `mycroft.conf` configuration |
| `ovos-utils` | Logging and bus utilities |
| `ovos-workshop` | OVOS skill framework compatibility |

## Model Download

The model is **not bundled** with the package. It is downloaded on first use from Hugging Face Hub. The default model is:

```
Jarbas/ovos-model2vec-intents-distiluse-base-multilingual-cased-v2
```

To pre-download or switch models, set the `model` key in your configuration (see [Configuration](configuration.md)).

> **Note:** The default multilingual model is ~500 MB. English-only models range from 8 MB to 150 MB.

## Verifying the Installation

After installing, confirm the entry point is registered:

```bash
python -c "from ovos_m2v_pipeline import Model2VecIntentPipeline; print('OK')"
```

---
[Home](README.md) · [OVOS Pipeline Plugin →](ovos_pipeline.md)
