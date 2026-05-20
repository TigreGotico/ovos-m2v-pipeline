# Domain Prototype Pipeline & Store

This page documents two layers that ship together:

* **`Model2VecDomainPrototypePipeline`** — the OPM-discoverable pipeline class. Entry point: `ovos-m2v-domain-prototype-pipeline`. Subclasses the flat prototype pipeline; the only difference is the store shape (below) — intents are grouped into a domain == skill_id at registration time.
* **`DomainPrototypeIntentStore`** — the domain-grouped variant of [`PrototypeIntentStore`](pipeline.md#prototypeintentstore) used internally by that pipeline.

A separate entry point (rather than a `domain_engine: true` config flag on the flat pipeline) keeps the two pipelines independently selectable in `default_pipeline` ordering and lets each have its own `intents.<key>` config block.

## Enabling

Add it to your OVOS config and place it in your pipeline order alongside (or in place of) the flat prototype pipeline:

```json
{
  "intents": {
    "ovos-m2v-domain-prototype-pipeline": {
      "model": "minishlab/potion-multilingual-128M",
      "intent_strategy": "softmax_weighted",
      "intent_tau": 0.1
    }
  }
}
```

Configuration keys are read from `intents.ovos_m2v_domain_prototype_pipeline`. The pipeline accepts every key the flat plugin does, plus `intent_strategy` / `intent_top_k` / `intent_tau` for the per-domain sub-stores (defaults inherit from `prototype_*`) and an optional `top_k_domains` pruning knob.

## Architecture

`DomainPrototypeIntentStore` groups intents into *domains*, but — following adapt's `DomainIntentDeterminationEngine` — there is **no top-level router**. At inference time every domain's sub-store scores the query in parallel and the global argmax over the flat union of per-intent scores wins. Routing is implicit in the argmax. This mirrors the API shipped by sibling OVOS intent plugins (`nebulento.DomainIntentContainer`, `ovos_padatious.DomainIntentContainer`, `palavreado.DomainIntentContainer`, `padacioso.DomainIntentContainer`, `linha_fina.DomainIntentEngine`, `ovos_markov_pipeline.DomainMarkovIntentEngine`).

```
              query embedding
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼
   ┌────────┐  ┌────────┐  ┌────────┐
   │media   │  │home    │  │…       │   PrototypeIntentStore per domain
   │{scores}│  │{scores}│  │{scores}│   strategy = intent_strategy
   └────────┘  └────────┘  └────────┘
        │           │           │
        └───────────┼───────────┘
                    ▼
        flat {label: score} union
                    │
                    ▼
              global argmax
```

## Why parallel-argmax

Every per-domain sub-store is a `PrototypeIntentStore` configured with the same `PrototypeStrategy`, so cosines from different domains are on a comparable scale and can be merged into one flat dict. The argmax then selects both the domain (implicitly) and the intent in a single step. Two concrete benefits:

1. **No router to tune.** Earlier two-stage variants needed a separate strategy/temperature for the top-level router; here there's nothing to tune above the per-domain stores.
2. **Independent per-domain configuration.** Each sub-store keeps its own anchors + strategy + temperature.

No training step required — the static encoder produces the per-domain intent embeddings (and, when `top_k_domains` is enabled, the optional domain fingerprints too).

Every `add(model, domain, label, sentences)` simply registers the embedded sentences as anchors for `label` inside `domains[domain]` (a `PrototypeIntentStore` configured with `intent_strategy`). When `top_k_domains` is set, a per-domain *fingerprint* (concatenated samples → store anchors) is also rebuilt so the optimisation has something to score against.

## Configuration

`DomainPrototypeIntentStore` exposes the [`PrototypeStrategy`](strategies.md) for the per-domain sub-stores via the `intent_*` keys. There is no separate router strategy — routing happens at argmax time.

```python
from model2vec import StaticModel
from ovos_m2v_pipeline import DomainPrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
store = DomainPrototypeIntentStore(
    intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,
    intent_tau=0.1,
)
```

### Optional: `top_k_domains` pruning

For workspaces with many domains, set `top_k_domains` to score only the best-matching K domains' sub-stores per query. The store maintains a small per-domain fingerprint (centroid of concatenated samples) used purely as a coarse filter; the final argmax still happens on per-intent cosines.

```python
store = DomainPrototypeIntentStore(top_k_domains=3)
```

Defaults to `None` (every domain scores in parallel).

## Usage

```python
store.add(model, "media", "play",      ["play {song}", "put on {song}"])
store.add(model, "media", "pause",     ["pause", "pause the music"])
store.add(model, "home",  "lights_on", ["turn on the lights", "lights on"])

scores = store.scores(model.encode(["play africa"])[0])
# scores → {"play": 0.93, "pause": 0.21, "lights_on": 0.05}
best = max(scores, key=scores.get)  # "play"
```

### Restricting to one domain

Pass `domain=...` to `scores()` (or `calc_intent()`) to score only inside a specific domain — useful for session/context-driven scoping where the caller already knows the active domain:

```python
scores = store.scores(query_embedding, domain="home")
```

### Convenience argmax

```python
store.calc_intent(query_embedding)              # → "play" / None
store.calc_intent(query_embedding, domain="home")  # restricted to home
```

### Lifecycle

```python
store.remove("media", "pause")   # drop an intent
store.remove_domain("media")     # drop a whole domain (intents + fingerprint)
```

## Strategy round-trip

Every `PrototypeStrategy` works inside every per-domain sub-store: they're all `PrototypeIntentStore` instances, so they inherit the full strategy machinery described in [Prototype Strategies](strategies.md). Tests in `tests/test_domain_store.py` round-trip all seven strategies.

## See also

- [Prototype Strategies](strategies.md) — the seven scoring strategies the underlying stores expose.
- [Pipeline Internals](pipeline.md) — how `PrototypeIntentStore` is wired into the bus.
- [Configuration](configuration.md) — top-level OVOS config keys.
