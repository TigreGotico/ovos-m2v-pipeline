# Pipeline Internals

## Class: `Model2VecIntentPipeline`

Defined in `ovos_m2v_pipeline/__init__.py`.

Extends `ConfidenceMatcherPipeline` from `ovos-plugin-manager`, which provides the three-tier match interface (`match_high`, `match_medium`, `match_low`) and integrates with the OVOS pipeline priority system.

---

## Initialisation

```python
Model2VecIntentPipeline(bus=None, config=None)
```

On startup the pipeline:

1. Reads configuration from `mycroft.conf` under `intents.ovos_m2v_pipeline` (or uses the `config` kwarg).
2. Loads the `StaticModelPipeline` from the configured `model` path / Hugging Face repo.
3. Registers message bus event handlers to keep the active intent list in sync.

### Bus Events Subscribed

| Event | Handler |
|-------|---------|
| `mycroft.ready` | `handle_sync_intents` |
| `padatious:register_intent` | `handle_sync_intents` |
| `register_intent` | `handle_sync_intents` |
| `detach_intent` | `handle_sync_intents` |
| `detach_skill` | `handle_sync_intents` |

---

## Intent Synchronisation

`handle_sync_intents` is called whenever skills are loaded or unloaded. It:

1. Applies a 3-second debounce (`_syncing` flag) to avoid redundant requests during bulk skill loading.
2. Requests the Adapt manifest via `intent.service.adapt.manifest.get`.
3. Requests the Padatious manifest via `intent.service.padatious.manifest.get`.
4. Merges the two lists into `self.intents` (a `set`), excluding any labels in `ignore_intents`.

The result is a set of labels like:

```
ovos-skill-date-time.openvoiceos:what.time.is.it.intent
ovos-skill-naptime.openvoiceos:naptime.intent
...
```

---

## Inference (`_match`)

`_match(utterance)` is the core inference method. It:

1. Calls `model.predict_proba([utterance])` to get a probability over all training labels.
2. Sorts results by probability descending.
3. Iterates candidates and applies three rules:
   - **OCP special-case**: `ocp:play` → remapped to `ovos.common_play.play_search` under skill `ovos.common_play`.
   - **Common query special-case**: `common_query:common_query` → remapped to `common_query.question`.
   - **Stop special-case**: `stop:stop` → remapped to `mycroft.stop`.
   - **Runtime filter**: any label not in `self.intents` is skipped.
4. Yields `(skill_id, label, probability)` tuples for all surviving candidates.

---

## Match Methods

All three methods share the same logic: call `_match`, take the first result, and check it against the relevant threshold.

```
match_high   → conf_high  (default 0.70)
match_medium → conf_medium (default 0.50)
match_low    → conf_low   (default 0.15)
```

If the top result's probability is below the threshold, `None` is returned. Otherwise an `IntentHandlerMatch` is returned with:

| Field | Value |
|-------|-------|
| `match_type` | The intent label (e.g. `ovos-skill-date-time.openvoiceos:what.time.is.it.intent`) |
| `match_data` | `{"utterance": ..., "confidence": ...}` |
| `skill_id` | Extracted from the label prefix (before `:`) |
| `utterance` | The original utterance string |

---

## Limitations

- The model is **pretrained on a fixed dataset**. Skills that were not present in the training corpus will not be recognised, even if their intents are registered at runtime.
- Intent sync uses a fixed 3-second sleep debounce. This may cause a brief window after startup where `self.intents` is empty and all matches are discarded.
- Only the **first utterance** in the input list is evaluated (`utterances[0]`).
