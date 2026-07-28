"""Download, normalize, and merge multilingual OVOS intent datasets.

This script fetches CSV files from HuggingFace and GitHub, normalizes the
columns into a common schema, and writes two output files:

* ``merged_intents_dataset.csv`` – compact training format used by
  ``train_multilingual.py``:

  ========  =================================================
  lang      ISO-639-1 language code (e.g. "en", "pt", "eu")
  label     ``<domain>:<intent>`` composite label
  sentence  normalized utterance text
  ========  =================================================

* ``merged_intents_dataset_full.csv`` – expanded format with separate
  ``domain`` and ``intent`` columns, useful for domain-only or intent-only
  classifiers and for inspection:

  ========  =================================================
  lang      ISO-639-1 language code
  domain    skill identifier (e.g. "ovos-skill-date-time.openvoiceos")
  intent    intent name (e.g. "what.time.is.it.intent")
  sentence  normalized utterance text
  ========  =================================================

* ``by_lang/intents_<lang>.csv`` – per-language compact subsets.
* ``by_lang/intents_<lang>_full.csv`` – per-language expanded subsets.

Usage::

    python gather_dataset.py
"""

import hashlib
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns

matplotlib.use("Agg")

# Root directory for all generated artefacts.
TRAIN_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(TRAIN_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Directory where downloaded source CSVs are cached between runs.
# Delete a file here to force a fresh download for that source.
CACHE_DIR = os.path.join(OUTPUT_DIR, "dataset_cache")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cached_path(url: str) -> str:
    """Return the local cache path for a remote URL.

    The filename is ``<last-url-segment>_<8-char-hash>.csv`` so collisions
    between identically-named files from different sources are avoided while
    the name stays human-readable.
    """
    stem = url.rstrip("/").split("/")[-1].split("?")[0]
    digest = hashlib.md5(url.encode()).hexdigest()[:8]
    return os.path.join(CACHE_DIR, f"{stem}_{digest}")


def _fetch(url: str) -> str:
    """Return the local path to the CSV, downloading it if not cached.

    Parameters
    ----------
    url:
        Remote CSV URL.

    Returns
    -------
    str
        Absolute path to the local (possibly freshly downloaded) copy.
    """
    local = _cached_path(url)
    if os.path.exists(local):
        return local
    print(f"  Downloading {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(local, "wb") as fh:
        fh.write(response.content)
    return local

# ---------------------------------------------------------------------------
# Dataset sources
# ---------------------------------------------------------------------------

# HuggingFace-hosted CSV files (all treated as English unless path says otherwise)
csv_sources = [
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-llm-augmented-intents/resolve/main/augmented.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-common-query-intents/resolve/main/common_query.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-massive-subset/resolve/main/ovos_massive_subset.csv",
]
en_csv = [
    "https://huggingface.co/datasets/Jarbas/music_queries_templates/resolve/main/music_templates.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-weather-intents/resolve/main/weather_intents_en.csv",
    #"https://huggingface.co/datasets/OpenVoiceOS/OCP_templates/resolve/main/ocp_media_templates_en.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_psytrance_tracks/resolve/main/mq_psy_tracks.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_metal_tracks/resolve/main/mq_ma_tracks.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_classical/resolve/main/mq_classical.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_prog/resolve/main/mq_prog.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_jazz/resolve/main/mq_jazz.csv",
    "https://huggingface.co/datasets/Jarbas/music_queries_metal_bands/resolve/main/mq_ma_bands.csv"
]
pt_csv = [
    "https://huggingface.co/datasets/OpenVoiceOS/MT-intents-dataset-pt-PT/resolve/main/train_pt-PT.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/MT-intents-dataset-pt-PT/resolve/main/test_pt-PT.csv",
    "https://huggingface.co/datasets/OpenVoiceOS/MT-intents-dataset-pt-PT/resolve/main/validation_pt-PT.csv",
]
ca_csv = [
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-ilenia-testset-ca/resolve/main/test.csv",
]
es_csv = [
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-ilenia-testset-es/resolve/main/test.csv",
]
nl_csv = [
    "https://huggingface.co/datasets/OpenVoiceOS/ovos-intents-ilenia-testset-nl/resolve/main/test.csv",
]


# Per-language skill-intent CSV files from the OVOS language-tracker repo
langs = ["en", "pt", "eu", "es", "gl", "nl", "fr", "de", "ca", "it", "da"]
github_sources = [
    f"https://raw.githubusercontent.com/OpenVoiceOS/lang-support-tracker/refs/heads/dev/skills/intents_{lang}.csv"
    for lang in langs
]
csv_sources += github_sources + en_csv + pt_csv + ca_csv + es_csv + nl_csv

# ---------------------------------------------------------------------------
# Filtering and normalisation rules
# ---------------------------------------------------------------------------

# Skills whose examples should be excluded from training
BLACKLIST_SKILLS = [
    "ovos-skill-local-media.openvoiceos",
    "ovos-skill-spotify.openvoiceos",
]

# Intent names that should be merged into a canonical name
INTENT_REPLACEMENTS = {
    "is_rain": "do-i-need-an-umbrella.intent",
    "do-i-need-an-umbrella.intent": "daily_forecast.intent",
    "howto.intent": "wikihow.intent",
    "HowAreYou.intent": "Greetings.intent",
    "handle_show_time": "what.time.is.it.intent",
    "handle_query_time": "what.time.is.it.intent",
    "how_hot_or_cold": "is_hot_cold",
    "current_wind": "is_wind",
}

BLACKLIST_INTENTS: list[str] = []
SKILL_REPLACEMENTS: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """Normalise a raw utterance string.

    Lowercases the text, strips surrounding quotes and extra whitespace, and
    removes commas.  Path-like tokens (``a/b``) are reduced to the last segment.

    Parameters
    ----------
    text:
        Raw utterance value from the source CSV.
    """
    return str(text).lower().replace(",", "").split("/")[-1].replace("  ", " ").strip().strip('"')


def normalize_domain(text: str) -> str:
    """Normalise a skill/domain identifier.

    Applies ``SKILL_REPLACEMENTS`` and fixes the common ``skill-ovos`` →
    ``ovos-skill`` prefix inversion.

    Parameters
    ----------
    text:
        Raw domain value from the source CSV.
    """
    n = str(text).strip().strip('"')
    for k, v in SKILL_REPLACEMENTS.items():
        n = n.replace(k, v)
    return n.replace("skill-ovos", "ovos-skill").split(":", 1)[0]


def normalize_intent(text: str) -> str:
    """Normalise an intent name by applying ``INTENT_REPLACEMENTS``.

    Parameters
    ----------
    text:
        Raw intent value from the source CSV.
    """
    n = str(text).strip().strip('"')
    for k, v in INTENT_REPLACEMENTS.items():
        n = n.replace(k, v)
    return n.split(":", 1)[-1]


def extract_domain(text: str) -> str:
    """Extract the domain portion from a ``domain:intent`` composite string.

    Parameters
    ----------
    text:
        Full label string, optionally in ``domain:intent`` format.
    """
    return str(text).strip().strip('"').split(":")[0]


# ---------------------------------------------------------------------------
# Per-source loader
# ---------------------------------------------------------------------------

def load_and_format_csv(url: str) -> pd.DataFrame:
    """Download and normalise a single source CSV into the common schema.

    The function detects the source type from the URL and maps its columns to
    the canonical ``(lang, label, sentence)`` schema.  Any errors during
    download or parsing result in an empty DataFrame being returned so that
    the overall merge can continue.

    Parameters
    ----------
    url:
        Fully-qualified URL to a CSV file.

    Returns
    -------
    pd.DataFrame
        Columns: ``lang``, ``domain``, ``intent``, ``label``, ``sentence``.
        Empty on failure.
    """
    try:
        df = pd.read_csv(_fetch(url))

        lang = None
        # Determine language from URL structure
        if "github" in url:
            lang = url.split("_")[-1].split(".csv")[0]
        elif url in en_csv:
            lang = "en"
        elif url in pt_csv:
            lang = "pt"
        elif url in es_csv:
            lang = "es"
        elif url in ca_csv:
            lang = "ca"
        elif url in nl_csv:
            lang = "nl"

        # Map source-specific column names to canonical schema
        if "music_templates" in url or "ocp_media" in url  or "music_queries" in url:
            df["domain"] = "ocp"
            df["intent"] = "play"
            df = df.rename(columns={"template": "sentence"})
        elif "weather" in url:
            df["domain"] = "ovos-skill-weather.openvoiceos"

        if "synthetic_query" in df.columns:
            df = df.rename(columns={"synthetic_query": "sentence"})
        if "example" in df.columns:
            df = df.rename(columns={"example": "sentence"})
        if "utterance" in df.columns:
            df = df.rename(columns={"utterance": "sentence"})
        if "label" in df.columns:
            df = df.rename(columns={"label": "intent"})
        if "skill" in df.columns:
            df = df.rename(columns={"skill": "domain"})
        if "domain" not in df.columns:
            df["domain"] = df["intent"].apply(extract_domain)

        # Preserve per-row lang codes before the column-selection slice drops them.
        # Some sources (e.g. MASSIVE multilingual subset) already carry a lang column.
        existing_lang = df["lang"].copy() if "lang" in df.columns else None

        df = df[["domain", "intent", "sentence"]]

        # Drop rows with missing sentences before normalising so that
        # normalize(NaN) → "nan" never reaches the output CSV.
        # (pd.read_csv silently converts the string "nan" back to float NaN.)
        df = df.dropna(subset=["sentence"])

        # Apply normalisation to all columns
        df["domain"] = df["domain"].apply(normalize_domain)
        df["intent"] = df["intent"].apply(normalize_intent)
        df["sentence"] = df["sentence"].apply(normalize)

        # Drop sentences that normalised to empty or the bare string "nan"
        df = df[df["sentence"].str.strip().astype(bool)]
        df = df[df["sentence"] != "nan"]

        # Remove blacklisted skills and intents
        df = df[~df["domain"].isin(BLACKLIST_SKILLS)]
        df = df[~df["intent"].isin(BLACKLIST_INTENTS)]

        df["label"] = df["domain"] + ":" + df["intent"]

        # Priority: URL-derived lang > per-row column from source CSV > "en" fallback
        if lang:
            df["lang"] = lang
        elif existing_lang is not None:
            df["lang"] = existing_lang
        else:
            df["lang"] = "en"

        return df[["lang", "domain", "intent", "label", "sentence"]]

    except Exception as e:
        print(f"Failed to load {url}: {e}")
        return pd.DataFrame(columns=["lang", "domain", "intent", "label", "sentence"])


# ---------------------------------------------------------------------------
# Main – download, merge, deduplicate, report
# ---------------------------------------------------------------------------

def plot_dataset(df: pd.DataFrame, plots_dir: str = "dataset_plots") -> None:
    """Generate and save exploratory plots for the merged intent dataset.

    Saves the following PNGs to ``plots_dir``:

    1. ``lang_distribution.png``      – examples per language (bar chart)
    2. ``label_distribution.png``     – top-30 labels by example count (log scale)
    3. ``domain_distribution.png``    – examples per skill domain (log scale)
    4. ``label_size_histogram.png``   – histogram of per-label example counts
                                        (reveals overall class imbalance)
    5. ``lang_domain_heatmap.png``    – examples per (language × domain) for
                                        the top-20 domains (shows multilingual
                                        coverage gaps)

    Parameters
    ----------
    df:
        Merged DataFrame with columns ``lang``, ``domain``, ``intent``,
        ``label``, ``sentence``.
    plots_dir:
        Directory where PNGs are written (created if absent).
    """
    os.makedirs(plots_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Language distribution
    # ------------------------------------------------------------------
    lang_counts = df["lang"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(lang_counts) * 0.4)))
    bars = ax.barh(lang_counts.index, lang_counts.values, color="steelblue")
    for bar, val in zip(bars, lang_counts.values):
        ax.text(bar.get_width() + lang_counts.max() * 0.005, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Examples")
    ax.set_title(f"Examples per language  (total {len(df):,})")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_distribution.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Top-30 label distribution  (log scale so dominant labels don't
    #    swamp smaller ones)
    # ------------------------------------------------------------------
    top_n = 30
    label_counts = df["label"].value_counts()
    top_labels = label_counts.head(top_n).sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(14, max(6, top_n // 3)))
    ax.barh(top_labels.index, top_labels.values, color="steelblue")
    ax.set_xscale("log")
    ax.set_xlabel("Examples (log scale)")
    ax.set_title(f"Top {top_n} labels by example count  ({label_counts.nunique()} total labels)")
    ax.tick_params(axis="y", labelsize=7)
    for bar, val in zip(ax.patches, top_labels.values):
        ax.text(bar.get_width() * 1.03, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "label_distribution.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Domain distribution  (log scale for the same reason)
    # ------------------------------------------------------------------
    domain_counts = df["domain"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(14, max(6, len(domain_counts) * 0.3)))
    ax.barh(domain_counts.index, domain_counts.values, color="darkorange")
    ax.set_xscale("log")
    ax.set_xlabel("Examples (log scale)")
    ax.set_title(f"Examples per skill domain  ({len(domain_counts)} domains)")
    ax.tick_params(axis="y", labelsize=7)
    for bar, val in zip(ax.patches, domain_counts.values):
        ax.text(bar.get_width() * 1.03, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=7)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "domain_distribution.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Per-label size histogram (class imbalance view)
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(label_counts.values, bins=50, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(label_counts.mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean = {label_counts.mean():.0f}")
    ax.axvline(label_counts.median(), color="darkorange", linestyle="--", linewidth=1.2,
               label=f"Median = {label_counts.median():.0f}")
    ax.set_xlabel("Examples per label")
    ax.set_ylabel("Number of labels")
    ax.set_title("Class size distribution (imbalance)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "label_size_histogram.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Language × domain heatmap (top-20 domains)
    # ------------------------------------------------------------------
    top_domains = df["domain"].value_counts().head(20).index.tolist()
    heat_df = (
        df[df["domain"].isin(top_domains)]
        .groupby(["lang", "domain"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=top_domains)  # keep domain order by frequency
    )
    # Row-normalise so each language's coverage is comparable regardless of size
    heat_norm = heat_df.div(heat_df.sum(axis=1), axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(max(16, len(top_domains) * 0.9),
                                            max(4, len(heat_df) * 0.5)))

    sns.heatmap(heat_df, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                linewidths=0.3, cbar_kws={"label": "Example count"},
                annot_kws={"size": 7})
    axes[0].set_title("Examples per language × domain (raw counts)")
    axes[0].tick_params(axis="x", rotation=40, labelsize=7)
    axes[0].tick_params(axis="y", labelsize=8)

    sns.heatmap(heat_norm, annot=True, fmt=".2f", cmap="YlOrRd", ax=axes[1],
                linewidths=0.3, vmin=0, vmax=heat_norm.values.max(),
                cbar_kws={"label": "Share of language examples"},
                annot_kws={"size": 7})
    axes[1].set_title("Examples per language × domain (row-normalised)")
    axes[1].tick_params(axis="x", rotation=40, labelsize=7)
    axes[1].tick_params(axis="y", labelsize=8)

    plt.suptitle("Language × domain coverage (top 20 domains)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_domain_heatmap.png"), dpi=120)
    plt.close(fig)

    print(f"Dataset plots saved → {plots_dir}/  (5 files)")


def main() -> None:
    """Entry point: download all sources, merge, deduplicate, and save."""
    print(f"Fetching {len(csv_sources)} dataset sources ...")
    frames = [load_and_format_csv(url) for url in csv_sources]
    merged_df = pd.concat(frames, ignore_index=True)

    # Merge LLM-augmented sentences if they exist.
    # augmented_sentences.csv is produced by augment_dataset.py.
    augmented_path = os.path.join(OUTPUT_DIR, "augmented_sentences.csv")
    if os.path.exists(augmented_path):
        aug = pd.read_csv(augmented_path)
        aug = aug.dropna(subset=["sentence"])
        aug["sentence"] = aug["sentence"].astype(str).str.strip()
        aug = aug[aug["sentence"] != ""]
        # Ensure it has domain + intent columns
        if "label" in aug.columns and "domain" not in aug.columns:
            aug[["domain", "intent"]] = aug["label"].str.split(":", n=1, expand=True)
        if "lang" not in aug.columns:
            aug["lang"] = "en"
        aug["label"] = aug["domain"].apply(normalize_domain) + ":" + aug["intent"].apply(normalize_intent)
        aug["sentence"] = aug["sentence"].apply(normalize)
        aug = aug.dropna(subset=["sentence"])
        aug = aug[aug["sentence"].str.strip().astype(bool)]
        aug = aug[aug["sentence"] != "nan"]
        print(f"Merging {len(aug):,} LLM-augmented sentences from {augmented_path}")
        merged_df = pd.concat([merged_df, aug[merged_df.columns.intersection(aug.columns)]], ignore_index=True)


    # Deduplicate identical (lang, label, sentence) rows
    before = len(merged_df)
    merged_df.drop_duplicates(inplace=True)
    after = len(merged_df)
    print(f"Deduplicated {before - after} duplicate rows  ({after} remain)")

    # Save compact format (lang, label, sentence) – used by train_multilingual.py
    out_path = os.path.join(OUTPUT_DIR, "merged_intents_dataset.csv")
    merged_df[["lang", "label", "sentence"]].to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({after} examples)")

    # Save expanded format (lang, domain, intent, sentence) – separate columns
    out_path_full = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")
    merged_df[["lang", "domain", "intent", "sentence"]].to_csv(out_path_full, index=False)
    print(f"Saved → {out_path_full}  ({after} examples)")

    # Save per-language subset files (both compact and full formats)
    lang_dir = os.path.join(OUTPUT_DIR, "by_lang")
    os.makedirs(lang_dir, exist_ok=True)
    for lang, lang_df in merged_df.groupby("lang"):
        lang_df[["lang", "label", "sentence"]].to_csv(
            os.path.join(lang_dir, f"intents_{lang}.csv"), index=False
        )
        lang_df[["lang", "domain", "intent", "sentence"]].to_csv(
            os.path.join(lang_dir, f"intents_{lang}_full.csv"), index=False
        )
        print(f"  [{lang}] {len(lang_df):>6} examples → {lang_dir}/intents_{lang}{{,_full}}.csv")

    # ------------------------------------------------------------------
    # Dataset balance report
    # ------------------------------------------------------------------
    print("\n--- Language distribution ---")
    lang_counts = merged_df["lang"].value_counts()
    for lang, count in lang_counts.items():
        print(f"  {lang:<6} {count:>7} examples")

    print(f"\n--- Label distribution (top 20 of {merged_df['label'].nunique()} total) ---")
    label_counts = merged_df["label"].value_counts()
    for label, count in label_counts.head(20).items():
        print(f"  {count:>6}  {label}")

    print(f"\n--- Class balance ---")
    print(f"  Min examples per label : {label_counts.min()}")
    print(f"  Max examples per label : {label_counts.max()}")
    print(f"  Mean examples per label: {label_counts.mean():.1f}")
    print(f"  Labels with < 10 examples: {(label_counts < 10).sum()}")

    plot_dataset(merged_df, plots_dir=os.path.join(OUTPUT_DIR, "dataset_plots"))


if __name__ == "__main__":
    main()
