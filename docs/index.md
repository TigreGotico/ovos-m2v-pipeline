# OVOS Model2Vec Intent Pipeline

An intent matching pipeline for [OpenVoiceOS (OVOS)](https://openvoiceos.org), powered by [Model2Vec](https://github.com/MinishLab/model2vec) for fast, lightweight intent classification.

## Overview

This plugin integrates a Model2Vec static embedding model with the OVOS intent pipeline system. It classifies natural language utterances into intent labels that have been registered at runtime by loaded skills.

Two operating modes are available:

- **Classifier mode** (default) — a `StaticModelPipeline` with a trained linear head. Inference returns softmax probabilities over a fixed label set baked into the model. Best when a pre-trained model covers all target skills.
- **Prototype mode** — a bare `StaticModel` (embeddings only, no trained head). Each Padatious intent's example utterances are embedded at boot into a `PrototypeIntentStore`; matching uses cosine similarity. Best when skills are not known ahead of time or when you want zero-shot generalisation to new skills.

The pipeline is designed as a **fallback or confidence-based matcher** — it is most useful when deterministic engines (Adapt, Padatious) fail to produce a high-confidence match.

## How It Works — Classifier Mode

1. An utterance arrives from the user.
2. The `StaticModelPipeline` produces a probability distribution over all training labels.
3. The pipeline filters out any labels not currently registered by loaded skills.
4. The highest-probability valid intent is returned if it meets the configured confidence threshold.

## How It Works — Prototype Mode

1. At boot, each `padatious:register_intent` bus event carries the path to a `.intent` file with example utterances (or an inline `samples` list).
2. The examples are template-expanded (e.g. `(turn on|switch on) the lights`) and embedded with the bare `StaticModel`.
3. `select_anchors()` reduces the embeddings to the subset or aggregation dictated by `prototype_strategy`; up to `prototype_k` anchors per label are stored in `PrototypeIntentStore`.
4. At inference time, `score_labels()` turns the query embedding into one score per label according to the active `PrototypeStrategy`.

## Key Properties

- **Classifier mode: pretrained, not adaptive** — the model was trained on a fixed corpus of OVOS skill intent examples. It cannot learn new skills at runtime.
- **Prototype mode: zero-shot, adaptive** — any skill whose Padatious intent file contains example utterances can be matched, no retraining required.
- **Runtime filtering** — only intents from currently loaded skills are ever returned.
- **Tiered confidence** — three match levels (`high`, `medium`, `low`) with separate thresholds, integrated into the OVOS pipeline priority system.
- **Multilingual by default** — the default model covers 11+ languages.

## Documentation

| Page | Description |
|------|-------------|
| [Installation](installation.md) | How to install the plugin |
| [OVOS Pipeline Plugin](ovos_pipeline.md) | Entry points, bus events, mixing plugins, confidence tiers |
| [Configuration](configuration.md) | All configuration options |
| [Pipeline Internals](pipeline.md) | Architecture and runtime behaviour |
| [Prototype Strategies](strategies.md) | Scoring strategy reference for prototype mode |
| [Domain Prototype Store](domain_store.md) | Domain-grouped parallel-argmax prototype matching |
| [Hierarchical Prototype Store](hierarchical_store.md) | Two-stage domain-routed prototype matching |
| [Models](models.md) | Available pre-trained models and benchmark results |
| [Training](training.md) | How to gather data and train/retrain models |
