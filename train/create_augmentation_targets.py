"""Create LLM augmentation targets from the current intent dataset.

For every (lang, domain, intent) combination that has fewer than
``TARGET_PER_INTENT`` sentences, this script records:

- how many new sentences are needed
- 5 diverse example sentences to seed the LLM prompt

Outputs
-------
``output/augmentation_targets.json``
    Machine-readable target list consumed by ``augment_dataset.py``.
``output/augmentation_targets.md``
    Human-readable summary for review before running augmentation.

Usage::

    python create_augmentation_targets.py

Tuning::

    TARGET_PER_INTENT   = 200   # desired sentences per (lang, intent)
    MIN_EXAMPLES        = 3     # skip intents with fewer existing examples
    NUM_SEED_EXAMPLES   = 5     # examples shown to the LLM per intent
"""

import json
import logging
import os
import random

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(TRAIN_DIR, "output")

CSV_FULL: str = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")

OUT_JSON: str = os.path.join(OUTPUT_DIR, "augmentation_targets.json")
OUT_MD:   str = os.path.join(OUTPUT_DIR, "augmentation_targets.md")

# Target sentence count per (lang, intent) combination.
# Intents already at or above this count are skipped.
TARGET_PER_INTENT: int = 200

# Intents with fewer than this many existing sentences are skipped entirely
# (too few examples to give the LLM useful guidance).
MIN_EXAMPLES: int = 1

# Number of seed examples included in each augmentation request.
NUM_SEED_EXAMPLES: int = 6

RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["sentence"])
    df["sentence"] = df["sentence"].astype(str).str.strip()
    df = df[df["sentence"] != ""]
    return df


def pick_seed_examples(sentences: list[str], n: int, rng: random.Random) -> list[str]:
    """Pick ``n`` diverse seed examples from ``sentences``.

    Diversity heuristic: sort by length, then sample evenly across the
    length spectrum so the LLM sees both short and long phrasings.
    """
    if len(sentences) <= n:
        return sentences[:]
    sorted_sents = sorted(set(sentences), key=len)
    # Sample evenly across the sorted list
    indices = [int(i * (len(sorted_sents) - 1) / (n - 1)) for i in range(n)]
    return [sorted_sents[i] for i in indices]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not os.path.exists(CSV_FULL):
        raise FileNotFoundError(
            f"Dataset not found: {CSV_FULL}\nRun gather_dataset.py first."
        )

    df = load_dataset(CSV_FULL)

    # Reconstruct label if needed (full CSV has domain + intent columns)
    if "label" not in df.columns:
        df["label"] = df["domain"] + ":" + df["intent"]
    if "domain" not in df.columns or "intent" not in df.columns:
        df[["domain", "intent"]] = df["label"].str.split(":", n=1, expand=True)
    if "lang" not in df.columns:
        df["lang"] = "en"

    rng = random.Random(RANDOM_STATE)

    total_langs   = df["lang"].nunique()
    total_intents = df["label"].nunique()
    logger.info(
        f"Loaded {len(df):,} rows  |  {total_intents} intents  |  {total_langs} languages"
    )

    targets = []
    skipped_low  = 0
    skipped_done = 0

    for (lang, domain, intent), grp in df.groupby(["lang", "domain", "intent"]):
        sentences = grp["sentence"].tolist()
        current   = len(sentences)

        if current < MIN_EXAMPLES:
            skipped_low += 1
            continue

        needed = max(0, TARGET_PER_INTENT - current)
        if needed == 0:
            skipped_done += 1
            continue

        examples = pick_seed_examples(sentences, NUM_SEED_EXAMPLES, rng)

        targets.append({
            "lang":          lang,
            "domain":        domain,
            "intent":        intent,
            "label":         f"{domain}:{intent}",
            "current_count": current,
            "target_count":  TARGET_PER_INTENT,
            "needed":        needed,
            "examples":      examples,
        })

    targets.sort(key=lambda r: (r["lang"], r["domain"], r["intent"]))

    total_needed = sum(r["needed"] for r in targets)
    logger.info(f"Intents needing augmentation : {len(targets):,}")
    logger.info(f"Skipped (< {MIN_EXAMPLES} examples) : {skipped_low:,}")
    logger.info(f"Skipped (already at target)  : {skipped_done:,}")
    logger.info(f"Total sentences to generate  : {total_needed:,}")

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    payload = {
        "config": {
            "target_per_intent":  TARGET_PER_INTENT,
            "min_examples":       MIN_EXAMPLES,
            "num_seed_examples":  NUM_SEED_EXAMPLES,
            "source_csv":         CSV_FULL,
        },
        "summary": {
            "total_intents_needing_augmentation": len(targets),
            "total_sentences_to_generate":        total_needed,
        },
        "intents": targets,
    }

    with open(OUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    logger.info(f"JSON saved → {OUT_JSON}")

    # ------------------------------------------------------------------
    # Markdown output
    # ------------------------------------------------------------------
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write("# Dataset Augmentation Targets\n\n")
        fh.write(f"**Target sentences per intent per language:** {TARGET_PER_INTENT}  \n")
        fh.write(f"**Intents to augment:** {len(targets):,}  \n")
        fh.write(f"**Total sentences to generate:** {total_needed:,}  \n\n")
        fh.write("---\n\n")

        current_lang = None
        current_domain = None

        for r in targets:
            if r["lang"] != current_lang:
                current_lang = r["lang"]
                fh.write(f"## Language: `{current_lang}`\n\n")
                current_domain = None

            if r["domain"] != current_domain:
                current_domain = r["domain"]
                fh.write(f"### {current_domain}\n\n")

            fh.write(f"#### `{r['intent']}`\n\n")
            fh.write(f"- Current: **{r['current_count']}** sentences\n")
            fh.write(f"- Target:  **{r['target_count']}** sentences\n")
            fh.write(f"- **Generate: {r['needed']} new sentences**\n\n")
            fh.write("**Seed examples:**\n\n")
            for ex in r["examples"]:
                fh.write(f"> {ex}\n")
            fh.write("\n---\n\n")

    logger.info(f"Markdown saved → {OUT_MD}")


if __name__ == "__main__":
    main()
