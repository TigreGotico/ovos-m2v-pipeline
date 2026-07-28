"""Generate new intent utterances via an OpenAI-compatible LLM API.

Reads ``output/augmentation_targets.json`` (produced by
``create_augmentation_targets.py``), calls the LLM for each intent+lang
that needs more sentences, and appends the generated rows to
``output/augmented_sentences.csv``.

The script is **resume-safe**: already-generated (lang, label) pairs are
loaded from the output CSV on startup and skipped automatically.

Requirements
------------
Only the Python standard library + ``requests`` are used.  No OpenAI SDK.

Usage::

    # Start llama.cpp server first:
    # llama-server -m model.gguf --port 8080

    python augment_dataset.py
    python augment_dataset.py --url http://192.168.1.10:8080
    python augment_dataset.py --url http://localhost:11434 --model llama3
    python augment_dataset.py --dry-run      # print prompts, no API calls

Tuning::

    API_URL          – base URL of the OpenAI-compatible server
    MODEL            – model name passed in the request body
    TEMPERATURE_MIN/MAX – temperature is sampled uniformly from this range per request
    MAX_TOKENS       – upper bound on tokens per response
    BATCH_SIZE       – sentences requested per API call (keep ≤ 30)
    REQUEST_TIMEOUT  – seconds before a single HTTP request times out
    MAX_RETRIES      – retries on transient errors before skipping
"""
import random
import argparse
import csv
import json
import logging
import os
import time

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration  (override via CLI flags or edit here)
# ---------------------------------------------------------------------------

TRAIN_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(TRAIN_DIR, "output")

IN_JSON:   str = os.path.join(OUTPUT_DIR, "augmentation_targets.json")
CSV_FULL:  str = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")
OUT_CSV:   str = os.path.join(OUTPUT_DIR, "augmented_sentences.csv")

NUM_SEED_EXAMPLES:  int = 8    # fresh examples sampled from the full pool per API call
SYNTHETIC_TARGET:   int = 200  # sentences per intent when --synthetic is used

API_URL:    str = "http://192.168.1.200:8000"
MODEL:      str = "gemma-3n-E4B-it-GGUF"           # passed as-is to the API; llama.cpp ignores it
TEMPERATURE_MIN: float = 0.7   # random temperature is sampled from [MIN, MAX] per request
TEMPERATURE_MAX: float = 1.0
MAX_TOKENS:  int   = 2024
BATCH_SIZE:  int   = 10            # sentences requested per API call
REQUEST_TIMEOUT: int = 120         # seconds
MAX_RETRIES: int = 3

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a multilingual dataset augmentation assistant. "
    "Your task is to generate new, diverse, natural-sounding utterances "
    "for voice assistant intent recognition. "
    "Write ONLY the utterances — one per line, no numbering, no explanations, "
    "no punctuation at the end of lines. "
    "Use the same language as the provided examples. "
    "Vary sentence structure, wording, and length across the generated utterances.\n\n"
    "IMPORTANT — template variables:\n"
    "Some example utterances contain placeholders in curly braces, e.g. {artist}, {song}, "
    "{city}, {number}, {name}. These are slot variables that the voice assistant fills at "
    "runtime. (e.g. 'play {song} by {artist}').\n"
    "When you see them ALWAYS replace the placeholder with a diverse, realistic example "
    "value (e.g. 'play bohemian rhapsody by queen', 'play hotel california by eagles'). "
    "Use varied, realistic values across all generated utterances — "
    "different names, places, numbers, genres, etc. — never the same value twice."
)


def build_user_prompt(lang: str, domain: str, intent: str,
                      examples: list[str], n: int) -> str:
    """Build the user-turn prompt for a single augmentation request."""
    intent_display = intent.replace(".", " ").replace("_", " ").rstrip(" intent")

    # Detect whether examples contain template variables
    has_vars = any("{" in ex for ex in examples)
    var_note = (
        "\nNote: some examples use {variable} placeholders. "
        "Mix utterances that keep the placeholder syntax with utterances that "
        "substitute the placeholder with diverse, realistic values "
        "(different people, places, numbers, genres, etc.)."
        if has_vars else ""
    )

    lines = [
        f"Generate {n} new utterances for the voice assistant intent "
        f'"{intent_display}" ({domain}) in language code "{lang}".',
        "",
        "Existing example utterances (do NOT repeat these exactly):",
    ]
    for ex in examples:
        lines.append(f"  - {ex}")
    lines += [
        "",
        f"Write exactly {n} new utterances, one per line:{var_note}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# API call
# ---------------------------------------------------------------------------

def call_api(url: str, model: str, system: str, user: str,
             temperature: float, max_tokens: int, timeout: int) -> str | None:
    """POST to /v1/chat/completions and return the assistant message text."""
    endpoint = url.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": temperature,
        "max_tokens":  max_tokens,
    }
    try:
        resp = requests.post(endpoint, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        logger.warning("  Request timed out.")
        return None
    except requests.exceptions.ConnectionError as exc:
        logger.error(f"  Connection error: {exc}")
        return None
    except (KeyError, IndexError, ValueError) as exc:
        logger.warning(f"  Unexpected response format: {exc}")
        return None
    except requests.exceptions.HTTPError as exc:
        logger.warning(f"  HTTP error: {exc}")
        return None


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------

def parse_sentences(text: str) -> list[str]:
    """Extract one sentence per line; strip bullets, numbers, quotes."""
    sentences = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        # Strip leading list markers: "1.", "-", "*", "•"
        for prefix in ("- ", "* ", "• "):
            if line.startswith(prefix):
                line = line[len(prefix):]
                break
        # Strip leading "N." numbering
        if len(line) > 2 and line[0].isdigit() and line[1] in (".", ")"):
            line = line[2:].lstrip()
        elif len(line) > 3 and line[:2].isdigit() and line[2] in (".", ")"):
            line = line[3:].lstrip()
        # Strip surrounding quotes
        line = line.strip('"\'').strip()
        if line:
            sentences.append(line.lower())
    return sentences


# ---------------------------------------------------------------------------
# Dataset sentence index (for per-request random seed sampling)
# ---------------------------------------------------------------------------

def load_sentence_index(csv_path: str) -> dict[tuple[str, str], list[str]]:
    """Load the full dataset and return {(lang, label): [sentence, ...]}."""
    index: dict[tuple[str, str], list[str]] = {}
    if not os.path.exists(csv_path):
        return index
    import csv as _csv
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in _csv.DictReader(fh):
            lang  = row.get("lang", "en")
            domain = row.get("domain", "")
            intent = row.get("intent", "")
            label  = row.get("label") or f"{domain}:{intent}"
            sent   = row.get("sentence", "").strip()
            if sent:
                index.setdefault((lang, label), []).append(sent)
    return index


def sample_seed_examples(index: dict[tuple[str, str], list[str]],
                         lang: str, label: str,
                         fallback: list[str],
                         n: int, rng: random.Random) -> list[str]:
    """Return n randomly sampled seed examples from the full dataset pool."""
    pool = index.get((lang, label), fallback)
    if len(pool) <= n:
        return list(pool)
    return rng.sample(pool, n)


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def load_already_generated(csv_path: str) -> dict[tuple[str, str], int]:
    """Return {(lang, label): sentence_count} already present in the output CSV."""
    counts: dict[tuple[str, str], int] = {}
    if not os.path.exists(csv_path):
        return counts
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (row.get("lang", ""), row.get("label", ""))
            counts[key] = counts.get(key, 0) + 1
    return counts


def append_rows(csv_path: str, rows: list[dict], write_header: bool) -> None:
    """Append generated rows to the output CSV."""
    fieldnames = ["lang", "domain", "intent", "label", "sentence", "model"]
    mode = "w" if write_header else "a"
    with open(csv_path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def load_all_intents(csv_path: str) -> list[dict]:
    """Return one entry per (lang, domain, intent) from the full dataset CSV.

    Used by --synthetic to build a work queue covering every label in the
    dataset, regardless of whether it already has enough samples.
    """
    if not os.path.exists(csv_path):
        return []
    groups: dict[tuple[str, str, str], dict] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            lang   = row.get("lang", "en")
            domain = row.get("domain", "")
            intent = row.get("intent", "")
            if not domain or not intent:
                continue
            key = (lang, domain, intent)
            if key not in groups:
                groups[key] = {
                    "lang":    lang,
                    "domain":  domain,
                    "intent":  intent,
                    "label":   f"{domain}:{intent}",
                    "needed":  SYNTHETIC_TARGET,
                    "examples": [],
                }
            sent = row.get("sentence", "").strip()
            if sent:
                groups[key]["examples"].append(sent)
    return list(groups.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(api_url: str, model: str, dry_run: bool, synthetic: bool = False,
         temp_min: float = TEMPERATURE_MIN,
         temp_max: float = TEMPERATURE_MAX) -> None:
    global TEMPERATURE_MIN, TEMPERATURE_MAX
    TEMPERATURE_MIN, TEMPERATURE_MAX = temp_min, temp_max

    if synthetic:
        intents = load_all_intents(CSV_FULL)
        if not intents:
            raise FileNotFoundError(
                f"Full dataset not found: {CSV_FULL}\n"
                "Run gather_dataset.py first."
            )
        logger.info(
            f"--synthetic mode: {len(intents):,} intents loaded from {CSV_FULL}  "
            f"(target={SYNTHETIC_TARGET} per intent, originals ignored)"
        )
    else:
        if not os.path.exists(IN_JSON):
            raise FileNotFoundError(
                f"Targets file not found: {IN_JSON}\n"
                "Run create_augmentation_targets.py first."
            )
        with open(IN_JSON, encoding="utf-8") as fh:
            data = json.load(fh)
        intents = data["intents"]
        logger.info(f"Loaded {len(intents):,} augmentation targets from {IN_JSON}")

    sentence_index = load_sentence_index(CSV_FULL)
    logger.info(
        f"Sentence index: {len(sentence_index):,} (lang, label) pools "
        f"({'from ' + CSV_FULL if sentence_index else 'empty — fallback to JSON examples'})"
    )
    rng = random.Random()  # unseeded for true per-run randomness

    # Resume: count sentences already written per (lang, label)
    already_counts = load_already_generated(OUT_CSV)
    total_already = sum(already_counts.values())
    if total_already:
        logger.info(
            f"Resuming: {total_already:,} sentences already written for "
            f"{len(already_counts)} (lang, label) pairs"
        )

    # Build work queue — include partially-done intents with reduced remaining count
    work = []
    total_skipped = 0
    for entry in intents:
        lang  = entry["lang"]
        label = entry["label"]
        already = already_counts.get((lang, label), 0)
        remaining = max(0, entry["needed"] - already)
        if remaining == 0:
            total_skipped += 1
            continue
        work.append({
            **entry,
            "remaining":          remaining,
            "generated_this_run": set(),   # dedup within this session
            "failures":           0,
        })

    logger.info(
        f"Work queue: {len(work)} intents to augment  "
        f"({total_skipped} already complete)"
    )

    need_header = not os.path.exists(OUT_CSV)
    total_written = 0
    total_failed  = 0
    round_num     = 0

    rng.shuffle(work)

    # Round-robin: one batch per intent per round, write immediately after each batch.
    # This spreads coverage evenly and survives restarts at any point.
    while work:
        round_num += 1
        logger.info(f"--- Round {round_num}: {len(work)} intents in queue ---")
        still_working = []

        for item in work:
            lang   = item["lang"]
            domain = item["domain"]
            intent = item["intent"]
            label  = item["label"]
            batch  = min(item["remaining"], BATCH_SIZE)

            # Fresh random seed examples per batch for variety
            examples = sample_seed_examples(
                sentence_index, lang, label, item["examples"],
                NUM_SEED_EXAMPLES, rng
            )
            prompt      = build_user_prompt(lang, domain, intent, examples, batch)
            temperature = rng.uniform(TEMPERATURE_MIN, TEMPERATURE_MAX)

            logger.info(
                f"  [{lang}] {label}  "
                f"batch={batch}  remaining={item['remaining']}  "
                f"temp={temperature:.3f}"
            )

            if dry_run:
                logger.info(f"  [dry-run] Prompt:\n{prompt}\n")
                fake = [f"<dry-run {label} {j+1}>" for j in range(batch)]
                rows = [{"lang": lang, "domain": domain, "intent": intent,
                         "label": label, "sentence": s, "model": model} for s in fake]
                append_rows(OUT_CSV, rows, write_header=need_header)
                need_header = False
                total_written += len(rows)
                item["remaining"] -= batch
                if item["remaining"] > 0:
                    still_working.append(item)
                continue

            response_text = call_api(
                url=api_url, model=model,
                system=SYSTEM_PROMPT, user=prompt,
                temperature=temperature, max_tokens=MAX_TOKENS,
                timeout=REQUEST_TIMEOUT,
            )

            if response_text is None:
                item["failures"] += 1
                if item["failures"] >= MAX_RETRIES:
                    logger.warning(
                        f"  Giving up on [{lang}] {label} "
                        f"after {MAX_RETRIES} consecutive failures"
                    )
                    total_failed += 1
                else:
                    still_working.append(item)
                continue

            batch_sentences = parse_sentences(response_text)
            existing_set = (
                {ex.lower() for ex in examples} | item["generated_this_run"]
            )
            new_sents = [s for s in batch_sentences if s not in existing_set]

            if not new_sents:
                logger.warning("  No usable sentences returned; will retry next round")
                item["failures"] += 1
                if item["failures"] < MAX_RETRIES:
                    still_working.append(item)
                else:
                    logger.warning(
                        f"  Giving up on [{lang}] {label} "
                        f"after {MAX_RETRIES} consecutive failures"
                    )
                    total_failed += 1
                continue

            item["failures"] = 0  # reset on success
            item["generated_this_run"].update(new_sents)

            rows = [{"lang": lang, "domain": domain, "intent": intent,
                     "label": label, "sentence": s, "model": model} for s in new_sents]
            append_rows(OUT_CSV, rows, write_header=need_header)
            need_header = False
            total_written += len(rows)
            item["remaining"] = max(0, item["remaining"] - len(new_sents))
            logger.info(
                f"  Wrote {len(new_sents)} rows  "
                f"(remaining: {item['remaining']}  total written: {total_written:,})"
            )
            print(new_sents[5:])

            if item["remaining"] > 0:
                still_working.append(item)

        work = still_working
        rng.shuffle(work)  # re-randomize order for next round

    logger.info("=" * 60)
    logger.info(f"Done.  Written: {total_written:,}  "
                f"Skipped (already complete): {total_skipped:,}  "
                f"Failed: {total_failed:,}")
    logger.info(f"Output → {OUT_CSV}")
    if total_written > 0:
        logger.info(
            "Next step: re-run gather_dataset.py after adding "
            f"{OUT_CSV} as a source, or merge manually."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Augment the intent dataset via an OpenAI-compatible LLM API."
    )
    parser.add_argument(
        "--url", default=API_URL,
        help=f"Base URL of the OpenAI-compatible server (default: {API_URL})",
    )
    parser.add_argument(
        "--model", default=MODEL,
        help=f"Model name to pass in the request body (default: {MODEL})",
    )
    parser.add_argument(
        "--temp-min", type=float, default=TEMPERATURE_MIN,
        help=f"Lower bound of the random temperature range (default: {TEMPERATURE_MIN})",
    )
    parser.add_argument(
        "--temp-max", type=float, default=TEMPERATURE_MAX,
        help=f"Upper bound of the random temperature range (default: {TEMPERATURE_MAX})",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print prompts without making API calls",
    )
    parser.add_argument(
        "--synthetic", action="store_true",
        help=(
            f"Generate {SYNTHETIC_TARGET} sentences for every intent in the dataset "
            "(including those already at target). Produces a fully synthetic dataset "
            "where no original sentences are used as training data."
        ),
    )
    args = parser.parse_args()
    main(api_url=args.url, model=args.model, dry_run=args.dry_run,
         synthetic=args.synthetic,
         temp_min=args.temp_min, temp_max=args.temp_max)
