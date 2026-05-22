# Hierarchical Prototype Pipeline & Store

This page documents two layers that ship together:

* **`Model2VecHierarchicalPrototypePipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-m2v-hierarchical-prototype-pipeline`. Subclasses the flat prototype pipeline; the only difference is the store shape (below) — intents are grouped into a domain == skill_id at registration time and matched in two stages.
* **`HierarchicalPrototypeIntentStore`** — the two-stage, domain-routed prototype store used internally by that pipeline.

A separate entry point keeps the two prototype pipelines (flat, hierarchical) independently selectable in `default_pipeline` ordering, each with its own `intents.<key>` config block.

## Enabling

Add it to your OVOS config and place it in your pipeline order alongside (or in place of) the other prototype pipelines:

```json
{
  "intents": {
    "ovos-m2v-hierarchical-prototype-pipeline": {
      "model": "minishlab/potion-multilingual-128M",
      "intent_strategy": "softmax_weighted",
      "intent_tau": 0.1,
      "domain_threshold": 0.2
    }
  }
}
```

Configuration keys are read from `intents.ovos_m2v_hierarchical_prototype_pipeline`. The pipeline accepts every key the flat plugin does, plus `intent_strategy` / `intent_top_k` / `intent_tau` for the per-domain sub-stores (defaults inherit from `prototype_*`) and `domain_threshold` — the minimum router score required to route a query.

## Architecture

`HierarchicalPrototypeIntentStore` groups intents into *domains* and routes queries in **two stages**. Stage one is a top-level router: a per-domain *fingerprint* embedding (the centroid of the domain's concatenated samples) is scored and the single best-scoring domain is selected. Stage two resolves the intent only inside that domain's sub-store — exactly one sub-store runs per query.

```
              query embedding
                    │
                    ▼
        ┌───────────────────────┐
        │  domain fingerprints  │  stage 1: pick ONE domain
        │  {media: .., home: ..}│  argmax cosine similarity
        └───────────────────────┘
                    │
            domain_threshold gate
                    │
                    ▼
            ┌───────────────┐
            │ routed domain │       stage 2: PrototypeIntentStore
            │   {scores}    │       strategy = intent_strategy
            └───────────────┘
                    │
                    ▼
              global argmax
```

## Why two-stage routing

The router decides the domain before any per-intent scoring happens. Two consequences:

1. **Off-topic rejection.** When the best domain's fingerprint score is below `domain_threshold`, the query is rejected outright and no intent is returned. `0.0` (default) disables the gate.
2. **Cheaper inference with many domains.** Only one sub-store is scored per query rather than every intent across every domain.

No training step required — the static encoder produces both the per-domain intent embeddings and the domain fingerprints.

## Configuration

`HierarchicalPrototypeIntentStore` exposes the [`PrototypeStrategy`](strategies.md) for the per-domain sub-stores via the `intent_*` keys, the router via the `domain_*` keys, and the rejection gate via `domain_threshold`.

```python
from model2vec import StaticModel
from ovos_m2v_pipeline import HierarchicalPrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
store = HierarchicalPrototypeIntentStore(
    intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
    intent_tau=0.1,
    domain_threshold=0.2,
)
```

## Usage

```python
store.add(model, "media", "play",      ["play {song}", "put on {song}"])
store.add(model, "media", "pause",     ["pause", "pause the music"])
store.add(model, "home",  "lights_on", ["turn on the lights", "lights on"])

store.calc_domain(model.encode(["play africa"])[0])   # → "media"
scores = store.scores(model.encode(["play africa"])[0])
# scores → {"play": 0.93, "pause": 0.21}  — only the routed domain
best = max(scores, key=scores.get)  # "play"
```

### Restricting to one domain

Pass `domain=...` to `scores()` (or `calc_intent()`) to skip the top-level router and score only inside a specific domain. This also bypasses the `domain_threshold` gate:

```python
scores = store.scores(query_embedding, domain="home")
```

### Convenience argmax

```python
store.calc_intent(query_embedding)                 # → "play" / None
store.calc_intent(query_embedding, domain="home")  # restricted to home
```

### Lifecycle

```python
store.remove("media", "pause")   # drop an intent
store.remove_domain("media")     # drop a whole domain (intents + fingerprint)
```

## Strategy round-trip

Every `PrototypeStrategy` works inside every per-domain sub-store: they're all `PrototypeIntentStore` instances, so they inherit the full strategy machinery described in [Prototype Strategies](strategies.md). Tests in `tests/test_hierarchical_store.py` round-trip all seven strategies.

## See also

- [Prototype Strategies](strategies.md) — the seven scoring strategies the underlying stores expose.
- [Pipeline Internals](pipeline.md) — how `PrototypeIntentStore` is wired into the bus.
- [Configuration](configuration.md) — top-level OVOS config keys.
