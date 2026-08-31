# Pre-release quirks

Behavior changes since the last stable release, newest first. This file is
reset at each stable release.

## 0.5.8a2

- Declared template slots fill from live intent context (OVOS-CONTEXT-1
  §7, `ovos-spec-tools` `context_slot_candidates`): a registered template's
  `{slot}` placeholders are recorded at registration time, and any slot with
  a live non-null context entry fills `match_data[slot]` when the utterance
  itself never produces a value for it, independently of any
  `requires_context` declaration on the same intent. m2v never extracts a
  slot value from the utterance (it is a label classifier), so context is
  the only source of a filled value and the OVOS-INTENT-4 per-slot entity
  `.blacklist` remains a no-op here — there is nothing utterance-supplied to
  exclude.

## 0.5.8a1

- Template expansion in every registration path is lazy and bounded:
  `islice(iter_expand(...))` (ovos-spec-tools 1.8.0a1) replaces
  materializing `expand` calls, so a combinatorial template costs what the
  bound takes — registration-time expansion previously materialized full
  cartesian products transiently (~3GB and ~20 minutes measured on a real
  deployment) even though the store then kept at most 2000 samples.

## 0.5.7a1

- Prototype-store growth is amortized: registrations buffer into pending
  chunks and consolidate once on first read (match, save, or property
  access). Previously every registration re-stacked the entire embedding
  array — quadratic build cost, measured as a 1.2GB array reallocated per
  skill on a real install, which is why store construction took tens of
  minutes with one core pegged. Each registration also logs an INFO
  progress line (prototypes added, running totals) so long builds are
  visible instead of looking like an idle hang.

## 0.5.6a1

- The sample bound applies at store ingest, whatever path materialized the
  samples: the 2000-cap previously applied only to entity slot-filling, so
  pre-expanded padatious registrations flowed in unbounded — a real
  deployment still built a ~1.1M-prototype store on 0.5.5a1 and swapped its
  cgroup to death. `PrototypeIntentStore.add` now evenly samples any batch
  over the bound.
- Padatious-contract labels are dealiased: the legacy `.intent`-suffixed
  name folds onto the canonical suffixless label (registration, detach and
  ignore_labels accept both forms), so the dual-emit from ovos-workshop no
  longer stores every prototype twice, and matches dispatch on the same
  canonical id padatious itself uses.

## 0.5.5a1

- Embedding runs strictly in-process: every `model.encode` call passes
  `use_multiprocessing=False`. model2vec otherwise spawns `os.cpu_count()`
  loky worker subprocesses for sentence batches over its threshold, which
  inside a memory-capped service cgroup meant dozens of workers, deep swap
  and an OOM-kill during skill loading on a real deployment. Large template
  batches now encode sequentially — slightly slower for huge batches, never
  a process fan-out.
- Entity-filled template expansion is bounded: the cartesian product over a
  template's registered entity values caps at 2000 combinations, sampled
  deterministically and evenly across the space (endpoints kept). Two
  ~2200-value entities in one template previously materialized ~4.8M
  strings and embedded them all on every registration — tens of GB of swap
  and an OOM-kill on a real deployment.
