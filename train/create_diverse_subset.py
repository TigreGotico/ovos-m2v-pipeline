"""Create a diverse, balanced subset of the merged intent dataset.

For each label, sentences that are minor rewrites of each other (e.g.
"what time is it" / "what's the time") waste training budget without adding
new information.  This script selects the **most linguistically diverse**
K sentences per label using a greedy farthest-point (maximin) algorithm:

1. **Word-bag deduplication** – sentences whose word sets are identical are
   collapsed to the longest representative.
2. **Greedy maximin selection** – starting from the sentence with the highest
   average distance to all others, each subsequent pick is the sentence that
   is *farthest from its nearest already-selected neighbour* (maximises the
   minimum pairwise distance in the selected set).  This provably covers
   vocabulary variations as broadly as possible.

The distance metric is **TF-IDF cosine distance**, which weights rare /
domain-specific words more heavily than stop-words, making it more
discriminating than plain word-Jaccard for intent sentences.

Outputs
-------
``diverse_subset.csv``      – compact format (lang, label, sentence)
``diverse_subset_full.csv`` – expanded format (lang, domain, intent, sentence)
                              written only when ``merged_intents_dataset_full.csv``
                              is present alongside the input.

A per-label stats CSV (``diverse_subset_stats.csv``) reports how many
sentences were removed and how average pairwise diversity changed.

Usage::

    python create_diverse_subset.py

Tuning::

    MAX_PER_LABEL   = 100   # target samples per label
    MIN_PER_LABEL   = 5     # labels with ≤ this many samples are kept intact
    MAX_CANDIDATES  = 2000  # pre-sample pool size for very large labels
                            # (avoids huge distance matrices; 2000×2000 ≈ 16 MB)
"""

import logging
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_distances
from tqdm import tqdm

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "merged_intents_dataset.csv")
FULL_CSV_PATH = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")  # optional; written if present

OUT_PATH = os.path.join(OUTPUT_DIR, "diverse_subset.csv")
OUT_FULL_PATH = os.path.join(OUTPUT_DIR, "diverse_subset_full.csv")
OUT_STATS_PATH = os.path.join(OUTPUT_DIR, "diverse_subset_stats.csv")
OUT_PLOTS_DIR = os.path.join(OUTPUT_DIR, "diverse_plots")

# Target number of samples kept per label after diversification.
# Labels smaller than this are kept entirely (no selection applied).
MAX_PER_LABEL: int = 25

# Labels with this many samples or fewer are never filtered.
MIN_PER_LABEL: int = 1

# Maximum pool size fed into the distance matrix for very large labels.
# After word-bag deduplication, if more than this many candidates remain,
# a random pre-sample is drawn before greedy selection.
MAX_CANDIDATES: int = 2000

# Number of random sentence pairs sampled to estimate average pairwise
# word-Jaccard diversity (for the stats report).
DIVERSITY_SAMPLE_N: int = 500

RANDOM_STATE: int = 42


# ---------------------------------------------------------------------------
# Distance / diversity helpers
# ---------------------------------------------------------------------------

def word_jaccard_distance(a: str, b: str) -> float:
    """Word-level Jaccard distance between two sentences.

    Returns 0.0 for identical word sets and 1.0 for completely disjoint ones.

    Parameters
    ----------
    a, b:
        Normalised sentence strings.
    """
    sa, sb = set(a.split()), set(b.split())
    union = sa | sb
    if not union:
        return 0.0
    return 1.0 - len(sa & sb) / len(union)


def avg_pairwise_word_jaccard(sentences: list[str], sample_n: int = DIVERSITY_SAMPLE_N) -> float:
    """Estimate mean pairwise word-Jaccard distance on a random sample.

    Higher values indicate greater vocabulary diversity across samples.

    Parameters
    ----------
    sentences:
        List of sentence strings to measure.
    sample_n:
        Maximum number of sentences to include in the sampled pairs.
    """
    n = len(sentences)
    if n < 2:
        return 0.0
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(n, min(sample_n, n), replace=False)
    sampled = [sentences[i] for i in idx]
    total, count = 0.0, 0
    for i in range(len(sampled)):
        for j in range(i + 1, len(sampled)):
            total += word_jaccard_distance(sampled[i], sampled[j])
            count += 1
    return total / count if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Core selection algorithm
# ---------------------------------------------------------------------------

def greedy_maximin(sentences: list[str], k: int) -> list[int]:
    """Select k maximally diverse sentences using greedy farthest-point selection.

    Algorithm
    ---------
    1. Represent all sentences as TF-IDF word vectors.
    2. Seed: pick the sentence with the highest *mean* cosine distance to all
       others — the most "peripheral" sample, good for diversity coverage.
    3. Iteratively add the candidate whose *minimum* distance to the already-
       selected set is largest (maximises worst-case coverage gap).

    This runs in O(n·k) distance lookups after the initial O(n²) matrix
    computation.

    Parameters
    ----------
    sentences:
        Pool of candidate strings (after deduplication / pre-sampling).
    k:
        Number of sentences to select.

    Returns
    -------
    list[int]
        Indices into ``sentences`` of the selected items.
    """
    n = len(sentences)
    if n <= k:
        return list(range(n))

    # Vectorise with TF-IDF (sublinear TF reduces the dominance of repeated words)
    vec = TfidfVectorizer(analyzer="word", min_df=1, sublinear_tf=True)
    try:
        X = vec.fit_transform(sentences)
    except ValueError:
        # Degenerate input (e.g. all-empty strings after normalisation)
        return list(range(min(k, n)))

    dist = cosine_distances(X)  # (n, n) float32-ish

    # Seed: sentence with the highest average distance to all others
    seed = int(dist.mean(axis=1).argmax())
    selected = [seed]

    remaining = np.ones(n, dtype=bool)
    remaining[seed] = False

    # min_dist[i] = distance from candidate i to its nearest selected neighbour
    min_dist = dist[seed].copy()

    for _ in range(k - 1):
        # Candidate with the largest minimum distance to selected set
        scores = np.where(remaining, min_dist, -1.0)
        next_idx = int(scores.argmax())
        selected.append(next_idx)
        remaining[next_idx] = False
        # Update running minimum distances
        np.minimum(min_dist, dist[next_idx], out=min_dist)

    return selected


def diverse_select(sentences: list[str], k: int) -> list[int]:
    """Full diverse-selection pipeline returning indices into ``sentences``.

    Steps
    -----
    1. Collapse sentences with identical word bags to their longest
       representative (exact-duplicate removal).
    2. If the deduplicated pool exceeds ``MAX_CANDIDATES``, draw a random
       pre-sample to keep the distance matrix tractable.
    3. Apply :func:`greedy_maximin` to select ``k`` diverse sentences.

    Parameters
    ----------
    sentences:
        Original sentence list for a single label.
    k:
        Target number of selections.

    Returns
    -------
    list[int]
        Indices into the *original* ``sentences`` list.
    """
    # -- Step 1: word-bag deduplication -------------------------------------
    # For each unique word set, keep the longest sentence.
    bag_to_best: dict[frozenset, tuple[int, int]] = {}  # bag → (length, index)
    for i, s in enumerate(sentences):
        bag = frozenset(s.split())
        length = len(s)
        if bag not in bag_to_best or length > bag_to_best[bag][0]:
            bag_to_best[bag] = (length, i)

    dedup_indices = [idx for _, idx in bag_to_best.values()]

    if len(dedup_indices) <= k:
        return dedup_indices

    # -- Step 2: pre-sample if pool is still very large ---------------------
    rng = np.random.default_rng(RANDOM_STATE)
    if len(dedup_indices) > MAX_CANDIDATES:
        pre = rng.choice(len(dedup_indices), MAX_CANDIDATES, replace=False)
        pool_indices = [dedup_indices[i] for i in pre]
    else:
        pool_indices = dedup_indices

    # -- Step 3: greedy maximin on the pool --------------------------------
    pool_sentences = [sentences[i] for i in pool_indices]
    picked_in_pool = greedy_maximin(pool_sentences, k)
    return [pool_indices[i] for i in picked_in_pool]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_diversification(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    stats_df: pd.DataFrame,
    plots_dir: str = OUT_PLOTS_DIR,
) -> None:
    """Generate and save diagnostic plots for the diversification run.

    Saves six PNGs to ``plots_dir``:

    1. ``label_size_distribution.png``  – overlaid log-scale histograms of
       per-label example counts before and after selection.
    2. ``kept_pct_distribution.png``    – histogram of the fraction of
       examples kept per label; reveals how aggressively each label was trimmed.
    3. ``diversity_scatter.png``        – scatter of div_before vs div_after
       coloured by kept_pct; points above the diagonal improved in diversity.
    4. ``diversity_gain_hist.png``      – histogram of (div_after − div_before)
       per label; shows the distribution of diversity improvement.
    5. ``lang_balance.png``             – grouped bar chart comparing per-language
       example counts before and after; checks that language balance is preserved.
    6. ``dedup_pipeline.png``           – horizontal bar chart of the top-30
       most-reduced labels showing three stages: original → deduped → final.

    Parameters
    ----------
    df_before:
        Original full DataFrame (``lang``, ``label``, ``sentence``).
    df_after:
        Diverse subset DataFrame with the same columns.
    stats_df:
        Per-label stats produced during selection (from ``diverse_subset_stats.csv``).
    plots_dir:
        Output directory (created if absent).
    """
    os.makedirs(plots_dir, exist_ok=True)

    # Pre-compute per-label counts for convenience
    counts_before = df_before["label"].value_counts()
    counts_after  = df_after["label"].value_counts()

    # ------------------------------------------------------------------
    # 1. Label size distribution (before vs after) – log-scale histogram
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.logspace(
        np.log10(max(1, min(counts_before.min(), counts_after.min()))),
        np.log10(counts_before.max()),
        40,
    )
    ax.hist(counts_before.values, bins=bins, alpha=0.6, label="Before", color="steelblue")
    ax.hist(counts_after.values,  bins=bins, alpha=0.6, label="After",  color="darkorange")
    ax.set_xscale("log")
    ax.set_xlabel("Examples per label (log scale)")
    ax.set_ylabel("Number of labels")
    ax.set_title(
        f"Label size distribution  "
        f"(before: {len(df_before):,}  after: {len(df_after):,} examples)"
    )
    ax.axvline(MAX_PER_LABEL, color="tomato", linestyle="--", linewidth=1,
               label=f"MAX_PER_LABEL = {MAX_PER_LABEL}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "label_size_distribution.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 2. Kept-% distribution – how aggressively each label was trimmed
    # ------------------------------------------------------------------
    trimmed = stats_df[stats_df["n_before"] > MIN_PER_LABEL]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(trimmed["kept_pct"], bins=25, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(trimmed["kept_pct"].mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean = {trimmed['kept_pct'].mean():.1f}%")
    ax.axvline(100, color="grey", linestyle=":", linewidth=0.8, label="100% (no trim)")
    ax.set_xlabel("% of examples kept per label")
    ax.set_ylabel("Number of labels")
    ax.set_title("Distribution of kept fraction across labels")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "kept_pct_distribution.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 3. Diversity scatter: div_before vs div_after, coloured by kept_pct
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        stats_df["div_before"], stats_df["div_after"],
        c=stats_df["kept_pct"], cmap="RdYlGn", alpha=0.6,
        s=18, vmin=0, vmax=100,
    )
    # Diagonal = no diversity change
    lim_max = max(stats_df["div_before"].max(), stats_df["div_after"].max()) * 1.05
    ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=0.8, label="No change")
    plt.colorbar(sc, ax=ax, label="% examples kept")
    ax.set_xlabel("Word-Jaccard diversity  before")
    ax.set_ylabel("Word-Jaccard diversity  after")
    ax.set_title("Diversity change per label\n(above diagonal = improved)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "diversity_scatter.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 4. Diversity gain histogram: distribution of (div_after - div_before)
    # ------------------------------------------------------------------
    gain = stats_df["div_after"] - stats_df["div_before"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.hist(gain, bins=30, color="steelblue", edgecolor="white", linewidth=0.4)
    ax.axvline(0,          color="grey",  linestyle=":",  linewidth=0.8, label="No change")
    ax.axvline(gain.mean(), color="tomato", linestyle="--", linewidth=1.2,
               label=f"Mean gain = {gain.mean():+.4f}")
    ax.set_xlabel("Diversity gain  (div_after − div_before)")
    ax.set_ylabel("Number of labels")
    ax.set_title("Distribution of diversity improvement across labels")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "diversity_gain_hist.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 5. Language balance before vs after
    # ------------------------------------------------------------------
    lang_before = df_before["lang"].value_counts().rename("Before")
    lang_after  = df_after["lang"].value_counts().rename("After")
    lang_cmp = pd.concat([lang_before, lang_after], axis=1).fillna(0).astype(int)
    lang_cmp = lang_cmp.sort_values("Before", ascending=False)

    x = np.arange(len(lang_cmp))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(8, len(lang_cmp) * 1.1), 5))
    bars_b = ax.bar(x - width / 2, lang_cmp["Before"], width, label="Before", color="steelblue")
    bars_a = ax.bar(x + width / 2, lang_cmp["After"],  width, label="After",  color="darkorange")
    # Annotate kept % per language
    for xpos, (_, row) in zip(x, lang_cmp.iterrows()):
        pct = 100 * row["After"] / row["Before"] if row["Before"] > 0 else 0
        ax.text(xpos, max(row["Before"], row["After"]) * 1.01,
                f"{pct:.0f}%", ha="center", va="bottom", fontsize=7, color="dimgrey")
    ax.set_xticks(x)
    ax.set_xticklabels(lang_cmp.index, fontsize=9)
    ax.set_ylabel("Examples")
    ax.set_title("Language distribution before vs after diversification")
    ax.legend()
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "lang_balance.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # 6. Deduplication pipeline: top-30 reduced labels (3 stages)
    # ------------------------------------------------------------------
    top_n = 30
    reduced = (
        stats_df[stats_df["n_before"] > stats_df["n_after"]]
        .nlargest(top_n, "n_before")
        .sort_values("n_before", ascending=True)
    )
    fig, ax = plt.subplots(figsize=(14, max(6, len(reduced) * 0.42)))
    y = np.arange(len(reduced))
    ax.barh(y, reduced["n_before"],  height=0.7, label="Original",   color="steelblue",  alpha=0.9)
    ax.barh(y, reduced["n_deduped"], height=0.7, label="After dedup", color="darkorange", alpha=0.85)
    ax.barh(y, reduced["n_after"],   height=0.7, label="Final",       color="seagreen",   alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(reduced["label"], fontsize=7)
    ax.set_xlabel("Example count")
    ax.set_title(f"Top {len(reduced)} most-reduced labels: three-stage pipeline")
    ax.legend()
    ax.axvline(MAX_PER_LABEL, color="tomato", linestyle="--", linewidth=0.8,
               label=f"MAX_PER_LABEL={MAX_PER_LABEL}")
    plt.tight_layout()
    fig.savefig(os.path.join(plots_dir, "dedup_pipeline.png"), dpi=120)
    plt.close(fig)

    logger.info(f"Saved diversification plots → {plots_dir}/  (6 files)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Load the dataset, apply per-label diverse selection, and save outputs."""
    logger.info(f"Loading {CSV_PATH} ...")
    df = pd.read_csv(CSV_PATH)
    before = len(df)
    df.dropna(subset=["sentence"], inplace=True)
    df["sentence"] = df["sentence"].astype(str)
    if len(df) < before:
        logger.warning(f"  Dropped {before - len(df):,} rows with missing sentences")
    logger.info(f"  {len(df):,} rows  |  {df['label'].nunique()} labels")

    # Load full dataset if available (to write the expanded output)
    full_df: pd.DataFrame | None = None
    if os.path.exists(FULL_CSV_PATH):
        full_df = pd.read_csv(FULL_CSV_PATH)
        logger.info(f"  Full dataset loaded ({len(full_df):,} rows)")

    selected_rows: list[pd.DataFrame] = []
    stats_rows: list[dict] = []

    labels = df["label"].unique()
    logger.info(
        f"Selecting up to {MAX_PER_LABEL} diverse sentences per label "
        f"(min kept intact: {MIN_PER_LABEL}) ..."
    )

    for label in tqdm(labels, desc="Labels", unit="label"):
        group = df[df["label"] == label]
        sentences = group["sentence"].tolist()
        n_before = len(sentences)

        if n_before <= MIN_PER_LABEL:
            # Too few samples – keep all without filtering
            selected_rows.append(group)
            stats_rows.append({
                "label": label,
                "n_before": n_before,
                "n_after": n_before,
                "n_deduped": n_before,
                "div_before": round(avg_pairwise_word_jaccard(sentences), 4),
                "div_after": round(avg_pairwise_word_jaccard(sentences), 4),
                "kept_pct": 100.0,
            })
            continue

        k = min(MAX_PER_LABEL, n_before)

        # Diversity before selection (sampled estimate)
        div_before = avg_pairwise_word_jaccard(sentences)

        # Select diverse subset
        chosen_indices = diverse_select(sentences, k)
        chosen_df = group.iloc[chosen_indices]
        chosen_sentences = chosen_df["sentence"].tolist()

        # Diversity after selection
        div_after = avg_pairwise_word_jaccard(chosen_sentences)

        # Word-bag deduplicated count (before greedy step)
        n_deduped = len({frozenset(s.split()) for s in sentences})

        stats_rows.append({
            "label": label,
            "n_before": n_before,
            "n_deduped": n_deduped,
            "n_after": len(chosen_df),
            "div_before": round(div_before, 4),
            "div_after": round(div_after, 4),
            "kept_pct": round(100 * len(chosen_df) / n_before, 1),
        })

        selected_rows.append(chosen_df)

    # ------------------------------------------------------------------
    # Assemble and save outputs
    # ------------------------------------------------------------------
    subset_df = pd.concat(selected_rows, ignore_index=True)

    subset_df[["lang", "label", "sentence"]].to_csv(OUT_PATH, index=False)
    logger.info(f"Saved compact subset → {OUT_PATH}  ({len(subset_df):,} rows)")

    # Expanded format: join back on (lang, label, sentence) to recover domain/intent
    if full_df is not None:
        merge_keys = ["lang", "label", "sentence"]
        full_subset = subset_df[merge_keys].merge(
            full_df[["lang", "domain", "intent", "sentence"]],
            on=["lang", "sentence"],
            how="left",
        ).drop_duplicates(subset=merge_keys)
        full_subset[["lang", "domain", "intent", "sentence"]].to_csv(OUT_FULL_PATH, index=False)
        logger.info(f"Saved full subset    → {OUT_FULL_PATH}  ({len(full_subset):,} rows)")

    # Stats report
    stats_df = pd.DataFrame(stats_rows).sort_values("n_before", ascending=False)
    stats_df.to_csv(OUT_STATS_PATH, index=False)
    logger.info(f"Saved stats          → {OUT_STATS_PATH}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_orig = len(df)
    n_new = len(subset_df)
    logger.info("=" * 60)
    logger.info(f"Total examples : {n_orig:>8,}  →  {n_new:>8,}  ({100*n_new/n_orig:.1f}%)")
    logger.info(f"Labels         : {len(labels):>8,}")
    logger.info(
        f"Avg diversity  : {stats_df['div_before'].mean():.4f}  →  "
        f"{stats_df['div_after'].mean():.4f}  (word-Jaccard)"
    )

    # Top-10 most trimmed labels
    trimmed = stats_df[stats_df["n_before"] > stats_df["n_after"]].head(10)
    if not trimmed.empty:
        logger.info("\nTop 10 most trimmed labels:")
        for _, row in trimmed.iterrows():
            logger.info(
                f"  {row['label']:<60}  "
                f"{row['n_before']:>5} → {row['n_after']:>3}  "
                f"div {row['div_before']:.3f} → {row['div_after']:.3f}"
            )

    plot_diversification(df, subset_df, stats_df)


if __name__ == "__main__":
    main()
