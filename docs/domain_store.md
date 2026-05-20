# DomainPrototypeIntentStore

Hierarchical, two-level variant of [`PrototypeIntentStore`](pipeline.md#prototypeintentstore). Intents are grouped into *domains*, and at inference time the engine first picks the most likely domain, then scores intents only within that domain. This mirrors the API shipped by sibling OVOS intent plugins (`nebulento.DomainIntentContainer`, `ovos_padatious.DomainIntentContainer`, `palavreado.DomainIntentContainer`, `padacioso.DomainIntentContainer`, `linha_fina.DomainIntentEngine`, `ovos_markov_pipeline.DomainMarkovIntentEngine`).

## Why hierarchical

Two-level matching gives the prototype paradigm two concrete benefits:

1. **Tighter local cosine distributions.** A domain's prototypes share a subspace (lights / thermostat / door all live near "smarthome"), so the score across just that domain's prototypes is a sharper signal than the global score over all intents.
2. **Lower far-OOD false-positive rate.** The top-level router rejects chitchat that doesn't strongly match any domain *before* any sub-store sees it.

No training step required — the static encoder produces both the top-level domain embeddings and the per-domain intent embeddings (see `ovos_m2v_pipeline/domain_store.py:33`).

## Architecture

```
              query embedding
                    │
                    ▼
        ┌───────────────────────┐
        │   domain_store        │   PrototypeIntentStore
        │   (router)            │   strategy = domain_strategy
        └───────────────────────┘
                    │
              best domain
                    │
                    ▼
        ┌───────────────────────┐
        │   domains[<name>]     │   PrototypeIntentStore
        │   (intent matcher)    │   strategy = intent_strategy
        └───────────────────────┘
                    │
            {label: score} dict
```

Every `add(model, domain, label, sentences)` does two things:

* Adds the embedded sentences as anchors for `label` inside `domains[domain]` (a `PrototypeIntentStore` configured with `intent_strategy`).
* Mirrors the same sentences under the domain name in the top-level `domain_store`; the router learns the domain's surface forms incrementally.

See `ovos_m2v_pipeline/domain_store.py:120` for the `add()` implementation.

## Configuration

`DomainPrototypeIntentStore` lets you set the [`PrototypeStrategy`](strategies.md) independently for each level, because the two have different shapes:

| level | typical shape | good defaults |
|---|---|---|
| router (`domain_*`) | many concatenated in-domain samples per domain | `max_over_all` |
| per-domain (`intent_*`) | a small handful of per-intent samples | `max_over_all` or `softmax_weighted` |

Both default to `MAX_OVER_ALL` (the pre-strategy behaviour).

```python
from model2vec import StaticModel
from ovos_m2v_pipeline import DomainPrototypeIntentStore
from ovos_m2v_pipeline.strategies import PrototypeStrategy

model = StaticModel.from_pretrained("minishlab/potion-multilingual-128M")
store = DomainPrototypeIntentStore(
    domain_strategy=PrototypeStrategy.MEAN_CENTROID,   # router uses centroid per domain
    intent_strategy=PrototypeStrategy.SOFTMAX_WEIGHTED,  # per-domain stores use softmax
    intent_tau=0.1,
)
```

## Usage

```python
store.add(model, "media", "play",      ["play {song}", "put on {song}"])
store.add(model, "media", "pause",     ["pause", "pause the music"])
store.add(model, "home",  "lights_on", ["turn on the lights", "lights on"])

scores = store.scores(model.encode(["play africa"])[0])
# scores → {"play": 0.93, ...}  (only labels inside the resolved domain)
```

### Bypassing the router

Pass `domain=...` to `scores()` to skip the top-level classifier and score directly inside a specific domain — useful for session/context-driven scoping where the caller already knows the active domain:

```python
scores = store.scores(query_embedding, domain="home")
```

### Inspecting the resolved domain

```python
store.calc_domain(query_embedding)  # → "media" / "home" / None
```

### Lifecycle

```python
store.remove("media", "pause")   # drop an intent
store.remove_domain("media")     # drop a whole domain (intents + router entry)
```

## Strategy round-trip

Every `PrototypeStrategy` works at both levels: the router and every per-domain sub-store are themselves `PrototypeIntentStore` instances, so they inherit the full strategy machinery described in [Prototype Strategies](strategies.md). Tests in `tests/test_domain_store.py` round-trip all seven strategies at both levels.

## See also

- [Prototype Strategies](strategies.md) — the seven scoring strategies the underlying stores expose.
- [Pipeline Internals](pipeline.md) — how `PrototypeIntentStore` is wired into the bus.
- [Configuration](configuration.md) — top-level OVOS config keys.
