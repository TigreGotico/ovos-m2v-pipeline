[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenVoiceOS/ovos-m2v-pipeline)

# OVOS Model2Vec Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org). It uses the [Model2Vec](https://github.com/MinishLab/model2vec) model for intent classification.

This plugin uses a pretrained Model2Vec model to classify natural language utterances into intent labels registered with the system (Adapt, Padatious, and plugin-specific labels). It only considers intents from loaded skills and ignores labels from unregistered intents. Use this pipeline when deterministic engines fail to give a high-confidence match.

---

## Features

* Model2Vec drives intent classification.
* The plugin integrates directly with OVOS pipelines.
* The Model2Vec models train on [GitLocalize](https://gitlocalize.com/users/OpenVoiceOS) exports.
* English models come in several sizes, distilled from [Potion](https://huggingface.co/collections/minishlab/potion-6721e0abd4ea41881417f062).
* The multilingual model is distilled from [LaBSE](https://huggingface.co/minishlab/M2V_multilingual_output).
* The plugin syncs Adapt and Padatious intents dynamically at runtime.
* The plugin considers only intents from loaded skills and ignores unregistered labels.

> English models range from 8 MB to 150 MB. The multilingual model (the default) is over 500 MB.

---

## Installation

Install the plugin with `pip`:

```bash
pip install ovos-m2v-pipeline
```

---

## Configuration

In your `mycroft.conf`:

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "Jarbas/ovos-model2vec-intents-LaBSE",
      "conf_high": 0.7,
      "conf_medium": 0.5,
      "conf_low": 0.15,
      "ignore_intents": []
    }
  }
}
```

* `model`: Path to your pretrained Model2Vec model or huggingface repo.
* `conf_xxx`: Minimum confidence threshold for intent matching.
* `ignore_intents`: List of canonical labels to exclude from matching (deny-list, applied after `label_map`).
* `valid_labels`: List of canonical labels eligible to match (allow-list, applied after `label_map`). When unset, every mapped label is eligible.
* `label_map`: Maps a raw model label to its canonical `skill_id:intent` label. Merges over (and can override) the built-in OCP/common-query/stop remaps and any labels the model itself declares in `labels.json`; see [Trained models document their labels](#trained-models-document-their-labels).
* `prototype_strategy`: Scoring strategy for prototype mode (default `"max_over_all"`, back-compatible). See [docs/strategies.md](docs/strategies.md).
* `prototype_top_k`: Top-k cosines averaged by the `top_k_mean` strategy (default `3`).
* `prototype_tau`: Softmax temperature for the `softmax_weighted` strategy (default `0.1`).

> The Model2Vec model is pretrained on GitLocalize exports. It **cannot learn new skills** dynamically.

---

## Which entrypoint do I want?

This plugin ships two `opm.pipeline` entrypoints. Both use the same
`Model2VecIntentPipeline` class, running in different modes:

* **`ovos-m2v-pipeline`** (`Model2VecIntentPipeline`, `mode: "classifier"`, the
  default) loads a pretrained, **frozen** classification head with a fixed
  label set baked in at training time. It is fast and needs no runtime
  fitting, but it can only ever return the labels it was trained on. It still
  tracks OVOS-INTENT-4 `ovos.intent.register.template` registrations from
  skills so it can gate/allowlist a trained label, but registering a new
  intent that was not part of training does **not** teach it to that skill.
  That intent will never be matched.
* **`ovos-m2v-prototype-pipeline`** (`Model2VecPrototypePipeline`, `mode:
  "prototype"`) loads a bare embedding model with no classification head and
  builds its label set entirely at runtime, from the example utterances
  supplied by Adapt/Padatious registrations and OVOS-INTENT-4 template
  registrations. Use this entrypoint whenever skills need to register new
  intents (including custom/dynamically-created skills) that must actually be
  matched.

You can enable both entrypoints together. Configure each independently under
its own `intents.<entrypoint-name>` key (see the `Model2VecPrototypePipeline`
docstring for an example), so a deployment can keep the fast frozen
classifier for its core trained intents while the prototype matcher picks up
everything else.

---

## Trained models document their labels

In classifier mode the label head is frozen at training time, so which bus
intent each label denotes (`ocp:play` -> `ovos.common_play:ovos.common_play.play_search`,
for example) is a property of that particular trained model, not of the
plugin code. A model can ship this mapping alongside its weights as a
`labels.json` file in its repo/directory:

```json
{
  "my_domain:book_flight": "travel_skill:book.flight.intent",
  "my_domain:cancel_flight": "travel_skill:cancel.flight.intent",
  "valid_labels": ["travel_skill:book.flight.intent", "travel_skill:cancel.flight.intent"]
}
```

`labels.json` has the same shape as the `label_map` config option, plus an
optional `valid_labels` list. When present, list the labels a model was
trained on on its model card too, so users know what to expect without
downloading it first.

Three layers combine, each overriding the previous on a per-key basis:

1. Built-in defaults (the OCP / common-query / stop remaps that predate this
   mechanism).
2. The loaded model's own `labels.json`, if it ships one.
3. The deployment's `label_map` / `valid_labels` config.

For a Hugging Face hub model id, `labels.json` rides the same local cache
as the model's own weights: the plugin never fetches it over the network at
construction time, and only ever consults the cache entry already populated
by whatever downloaded the model. If the model has not been cached yet, or
the cached copy has no `labels.json`, the manifest layer is treated as empty
- it is never a reason for plugin construction to touch the network or block
on one.

A missing or corrupt `labels.json` is logged and ignored; the plugin falls
back to the layers below it rather than failing to load. A `label_map` target
that is not a `skill_id:intent` string (no colon) is logged as a warning
(once per label) and used as-is - the plugin never invents a bus topic
from it.

---

## Usage

The `Model2VecIntentPipeline` class integrates with the OVOS intent system. It:

1. Receives an utterance (text).
2. Predicts intent labels using the pretrained Model2Vec model.
3. Filters out intents that are not part of the loaded skills.
4. Returns a match for the highest-confidence intent from the list of valid intents.

---

## Tips

* Tune `min_conf` to control the confidence threshold for intent matching.
* Use the `ignore_intents` list to filter out specific problematic intents from predictions.
* The plugin syncs Adapt and Padatious intents automatically at runtime, over the OVOS message bus.

> Pre-trained models are available in the [ovos-model2vec-intents](https://huggingface.co/collections/Jarbas/ovos-model2vec-intents-681c478aecb9979e659b17f8) Hugging Face collection.

---

## Related projects

* [OpenVoiceOS](https://github.com/OpenVoiceOS): the OVOS org, and the intent-pipeline system this plugin extends.
* [ovos-plugin-manager](https://github.com/OpenVoiceOS/ovos-plugin-manager): the base pipeline class and plugin infrastructure.
* [ovos-workshop](https://github.com/OpenVoiceOS/ovos-workshop): the skill framework this plugin integrates with.

---

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Credits

The model2vec intent pipeline was first prototyped by
[TigreGótico](https://tigregotico.pt) under the [ILENIA](https://proyectoilenia.es)
project for [OpenVoiceOS](https://openvoiceos.org) and later extended
with an embeddings-only mode and new models, through the NGI0 Commons Fund.

<img src="./ilenia.png" width="128"/>

> This project was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU, NextGenerationEU within the framework of the project [ILENIA](https://proyectoilenia.es) with reference 2022/TL22/00215337

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
