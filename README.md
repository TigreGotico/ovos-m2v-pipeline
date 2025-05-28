[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TigreGotico/ovos-m2v-pipeline)

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
