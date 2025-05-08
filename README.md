# 🧠 OVOS Model2Vec Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org), powered by the Model2Vec model for intent classification.

This plugin uses a pretrained [Model2Vec](https://github.com/MinishLab/model2vec) model to classify natural language utterances into intent labels registered with the system (Adapt, Padatious, and plugin-specific labels). It only considers intents from loaded skills and ignores any labels from unregistered intents. This pipeline is ideal for use cases where other deterministic engines fail to provide a high-confidence match.

---

## ✨ Features

* ✅ Powered by Model2Vec for high-quality intent classification
* ✅ Syncs Adapt and Padatious intents dynamically at runtime
* ✅ Only considers intents from loaded skills, ignoring unregistered labels
* ✅ Plug-and-play integration with OVOS pipelines
* ✅ Pretrained Model2Vec model based on GitLocalize exports, distilled from [LaBSE](https://huggingface.co/minishlab/M2V_multilingual_output) (multilingual)

---

## 📦 Installation

You can install the plugin via `pip`:

```bash
pip install ovos-m2v-pipeline
```

---

## ⚙️ Configuration

In your `mycroft.conf` or OVOS config file:

```json
{
  "intents": {
    "ovos-m2v-pipeline": {
      "model": "/path/to/your/model",
      "timeout": 1,
      "min_conf": 0.5,
      "ignore_intents": []
    }
  }
}
```

* `model`: Path to your pretrained Model2Vec model.
* `timeout`: Timeout for intent syncing, in seconds (default: `1`).
* `min_conf`: Minimum confidence threshold for intent matching (default: `0.5`).
* `ignore_intents`: List of intents to ignore during matching.

---

## 🧠 Usage

The `Model2VecIntentPipeline` class integrates with the OVOS intent system. It:

1. Receives an utterance (text).
2. Predicts intent labels using the pretrained Model2Vec model.
3. Filters out intents that are not part of the loaded skills.
4. Returns a match for the highest-confidence intent from the list of valid intents.

> Note: The Model2Vec model is pretrained based on GitLocalize exports and **cannot learn new skills** dynamically.

---

## 🧪 Tips

* Tune `min_conf` to control the confidence threshold for intent matching.
* Use the `ignore_intents` list to filter out specific intent labels from predictions.
* Syncing of Adapt and Padatious intents is done automatically at runtime via the OVOS message bus.
* You can adjust the `timeout` setting to control the time for retrieving intent names from the bus.

---

## 🛡 License

This project is licensed under the [Apache 2.0 License](LICENSE).
