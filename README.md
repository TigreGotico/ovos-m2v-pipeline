[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/OpenVoiceOS/ovos-m2v-pipeline)

# OVOS Model2Vec Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org), powered by the Model2Vec model for intent classification.

This plugin uses a pretrained [Model2Vec](https://github.com/MinishLab/model2vec) model to classify natural language utterances into intent labels registered with the system (Adapt, Padatious, and plugin-specific labels). It only considers intents from loaded skills and ignores any labels from unregistered intents. This pipeline is ideal for use cases where other deterministic engines fail to provide a high-confidence match.

---

## ✨ Features

* ✅ Powered by Model2Vec for high-quality intent classification
* ✅ Plug-and-play integration with OVOS pipelines
* ✅ Model2Vec trained on [GitLocalize](https://gitlocalize.com/users/OpenVoiceOS) exports
* ✅ English models in various sizes, distilled from [Potion](https://huggingface.co/collections/minishlab/potion-6721e0abd4ea41881417f062)
* ✅ Multilingual model, distilled from [LaBSE](https://huggingface.co/minishlab/M2V_multilingual_output)
* ✅ Syncs Adapt and Padatious intents dynamically at runtime
* ✅ Only considers intents from loaded skills, ignoring unregistered labels

> 💡 english models size ranges from 8MB to 150MB, the multilingual model (default) is over 500MB

---

## 📦 Installation

You can install the plugin via `pip`:

```bash
pip install ovos-m2v-pipeline
```

---

## ⚙️ Configuration

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
* `ignore_intents`: List of intents to ignore during matching.
* `prototype_strategy`: Scoring strategy for prototype mode (`"max_over_all"` default — back-compatible). See [docs/strategies.md](docs/strategies.md).
* `prototype_top_k`: Top-k cosines averaged by the `top_k_mean` strategy (default `3`).
* `prototype_tau`: Softmax temperature for the `softmax_weighted` strategy (default `0.1`).

> ⚠️  The Model2Vec model is pretrained based on GitLocalize exports and **cannot learn new skills** dynamically.

---

## 🧠 Usage

The `Model2VecIntentPipeline` class integrates with the OVOS intent system. It:

1. Receives an utterance (text).
2. Predicts intent labels using the pretrained Model2Vec model.
3. Filters out intents that are not part of the loaded skills.
4. Returns a match for the highest-confidence intent from the list of valid intents.


---

## 🧪 Tips

* Tune `min_conf` to control the confidence threshold for intent matching.
* Use the `ignore_intents` list to filter out specific problematic intent from predictions.
* Syncing of Adapt and Padatious intents is done automatically at runtime via the OVOS message bus.

> 💡 pre-trained models available in this huggingface collection [ovos-model2vec-intents](https://huggingface.co/collections/Jarbas/ovos-model2vec-intents-681c478aecb9979e659b17f8)

---

## 🛡 License

This project is licensed under the [Apache 2.0 License](LICENSE).

---

## Credits

The model2vec intent pipeline was first prototyped by
[TigreGótico](https://tigregotico.pt) under the [ILENIA](https://proyectoilenia.es)
project for [OpenVoiceOS](https://openvoiceos.org) and substantially extended  —
an embeddings-only mode and new models — through the NGI0 Commons Fund.

<img src="./ilenia.png" width="128"/>

> This project was funded by the Ministerio para la Transformación Digital y de la Función Pública and Plan de Recuperación, Transformación y Resiliencia - Funded by EU – NextGenerationEU within the framework of the project [ILENIA](https://proyectoilenia.es) with reference 2022/TL22/00215337

[![NGI0 Commons Fund](./ngi.png)](https://nlnet.nl/project/OpenVoiceOS)

This project was funded through the [NGI0 Commons Fund](https://nlnet.nl/commonsfund),
a fund established by [NLnet](https://nlnet.nl) with financial support from the
European Commission's [Next Generation Internet](https://ngi.eu) programme, under
the aegis of [DG Communications Networks, Content and Technology](https://commission.europa.eu/about-european-commission/departments-and-executive-agencies/communications-networks-content-and-technology_en)
under grant agreement No [101135429](https://cordis.europa.eu/project/id/101135429).
