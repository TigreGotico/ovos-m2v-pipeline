# Pre-release quirks

Behavior changes since the last stable release, newest first. This file is
reset at each stable release.

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
