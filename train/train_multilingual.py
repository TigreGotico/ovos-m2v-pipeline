"""Train and evaluate multilingual intent classifiers using Model2Vec.

For each entry in ``base_models``, this script:

1. Loads the full merged multilingual intent dataset.
2. **Balances** the dataset: drops rare classes (< ``MIN_SAMPLES_PER_CLASS``)
   and caps over-represented ones (> ``MAX_SAMPLES_PER_CLASS``).
3. Performs a **stratified train/test split** (``TEST_SIZE`` fraction held out)
   so that both splits share the same class proportions.
4. Optionally **filters** the training and test data to a specific set of
   languages per model — enabling language-specific classifiers evaluated on
   the same held-out diverse test set.
5. Re-balances the language-filtered training slice (so min/max constraints
   still hold within the target language subset).
6. Trains a ``StaticModelForClassification`` on the (filtered) training pool.
7. Measures wall-clock time for training and inference (benchmark).
8. Evaluates **overall** accuracy/F1 and **per-language** accuracy/F1/throughput.
9. Saves the trained model pipeline to disk.
10. Generates diagnostic plots:
    - Class-distribution bar chart (top N labels by count)
    - Per-class F1 score bar chart (sorted, worst→best)
    - Confusion matrix heatmap (capped at ``CONF_MATRIX_MAX_CLASSES`` classes)
    - Per-language accuracy & F1 grouped bar chart
11. Writes a Markdown metrics summary for each model.
12. After all models finish, writes a cross-model comparison table and a
    per-language × per-model heatmap.

Configuring per-model language filters::

    base_models = [
        # Multilingual model – trains on all languages
        {"path": ".../LaBSE",    "langs": None},

        # Language-specific – trains only on Spanish, Catalan, and English data
        {"path": ".../MrBERT",   "langs": ["es", "ca", "en"]},

        # Monolingual – Portuguese only
        {"path": ".../bert-base-portuguese-cased", "langs": ["pt"]},
    ]

    # langs=None  → use all languages in the dataset (multilingual model)
    # langs=[...] → restrict training AND evaluation to those languages only

Tuning dataset balance::

    MIN_SAMPLES_PER_CLASS = 10   # drop labels with fewer samples
    MAX_SAMPLES_PER_CLASS = 800  # cap labels above this count (None = no cap)
"""

import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from model2vec.train import StaticModelForClassification
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Use non-interactive backend so plots are saved to files without a display
matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_PATH = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")
RANDOM_STATE = 42
MAX_EPOCHS = 30
TEST_SIZE: float = 0.2

# --- Dataset balance controls ----------------------------------------------
# Labels with fewer than MIN_SAMPLES_PER_CLASS examples are removed entirely.
# Set to 1 to keep all classes.
MIN_SAMPLES_PER_CLASS: int = 10

# Labels with more than MAX_SAMPLES_PER_CLASS examples are randomly down-
# sampled to this cap.  Set to None to disable capping.
MAX_SAMPLES_PER_CLASS: int | None = 800

# --- Plot controls ---------------------------------------------------------
# Confusion matrices with more classes than this are truncated to the N most
# frequent classes so the chart stays legible.
CONF_MATRIX_MAX_CLASSES: int = 40

# Number of best/worst classes shown in per-class F1 bar chart
F1_PLOT_TOP_N: int = 30

# ---------------------------------------------------------------------------
# Per-model training configuration
#
# Each entry is a dict with two keys:
#   path  (str)             – local path or HuggingFace ID of the distilled model
#   langs (list[str]|None)  – language filter applied before training.
#                             None → use all languages (multilingual).
#                             A list of ISO-639-1 codes → restrict to those langs.
#
# The same base split is used for all models so results are comparable.
# When langs is set, both train and test slices are filtered to those languages
# and the training slice is re-balanced independently.
#
# The same base model can appear multiple times with different lang filters to
# produce several language-specific variants in one run.
#
# Models mirror distill.py – each is trained on the languages its underlying
# pre-trained checkpoint was designed for.
# ---------------------------------------------------------------------------

# Root directory containing all distilled Model2Vec checkpoints (distill.py output)
DISTILLED_DIR: str = os.path.join(OUTPUT_DIR, "distilled")


def _d(name: str) -> str:
    """Return the full local path for a distilled model checkpoint by short name."""
    return os.path.join(DISTILLED_DIR, name)


base_models: list[dict] = [
    # -----------------------------------------------------------------------
    # Truly multilingual models – trained on all available languages
    # -----------------------------------------------------------------------
    #
    {"path": "minishlab/potion-multilingual-128M", "langs": None},
    # LaBSE: 109-language sentence-BERT
    {"path": _d("LaBSE"), "langs": None},
    # paraphrase-multilingual-MiniLM-L12-v2: 50+ language model, compact
    {"path": _d("paraphrase-multilingual-MiniLM-L12-v2"), "langs": None},
    # multilingual-e5-small: Microsoft, 100+ languages
    {"path": _d("multilingual-e5-small"), "langs": None},
    # bert-base-multilingual-cased: Google mBERT, 104 languages
    {"path": _d("bert-base-multilingual-cased"), "langs": None},
    # MrBERT: multilingual
    {"path": _d("MrBERT"), "langs": None},
]

lang_specific_models: list[dict] = [
    # MrBERT-es: Spanish / English variant
    {"path": _d("MrBERT-es"),                                "langs": ["es", "en"]},
    # MrBERT-ca: Catalan / English variant
    {"path": _d("MrBERT-ca"),                                "langs": ["ca", "en"]},
    # MrBERT-legal: Spanish / English, legal domain
    {"path": _d("MrBERT-legal"),                             "langs": ["es", "en"]},
    # MrBERT-biomed: Spanish / English, biomedical domain
    {"path": _d("MrBERT-biomed"),                            "langs": ["es", "en"]},
    # MrBERT-science: Spanish / English, scientific domain
    {"path": _d("MrBERT-science"),                           "langs": ["es", "en"]},

    # -----------------------------------------------------------------------
    # English-only model – restricted to English training data
    # -----------------------------------------------------------------------
    # bge-base-en-v1.5: strong English-only embedding model
    {"path": _d("bge-base-en-v1.5"),                         "langs": ["en"]},
    {"path": "minishlab/potion-base-32M",                    "langs": ["en"]},
    {"path": "minishlab/potion-retrieval-32M",               "langs": ["en"]},
    {"path": "minishlab/potion-base-8M",                     "langs": ["en"]},
    {"path": "minishlab/potion-base-4M",                     "langs": ["en"]},
    {"path": "minishlab/potion-base-2M",                     "langs": ["en"]},

    # -----------------------------------------------------------------------
    # Iberian-family models (Spanish, Catalan, Galician, Portuguese, Basque)
    # BSC-LT MrBERT family – trained on Iberian languages + English
    # -----------------------------------------------------------------------

    # -----------------------------------------------------------------------
    # Galician
    # -----------------------------------------------------------------------
    {"path": _d("bertinho-gl-small-cased"),                  "langs": ["gl"]},

    # -----------------------------------------------------------------------
    # Portuguese (Brazilian) models
    # -----------------------------------------------------------------------
    {"path": _d("bertha-portuguese-small"),                  "langs": ["pt"]},
    {"path": _d("bert-base-portuguese-cased"),               "langs": ["pt"]},
    {"path": _d("bert-large-portuguese-cased"),              "langs": ["pt"]},

    # -----------------------------------------------------------------------
    # Catalan models (projecte-aina)
    # -----------------------------------------------------------------------
    {"path": _d("roberta-base-ca-v2"),                       "langs": ["ca"]},
    {"path": _d("roberta-large-ca-v2"),                      "langs": ["ca"]},
    {"path": _d("distilroberta-base-ca-v2"),                 "langs": ["ca"]},

    # -----------------------------------------------------------------------
    # Basque models (HiTZ)
    # -----------------------------------------------------------------------
    {"path": _d("BERnaT-base"),                              "langs": ["eu"]},
    {"path": _d("BERnaT-medium"),                            "langs": ["eu"]},
    {"path": _d("BERnaT-large"),                             "langs": ["eu"]},
    {"path": _d("EriBERTa-base"),                            "langs": ["eu"]},
]

# Accumulated metrics for the final cross-model comparison
metrics_summary: list[dict] = []


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the merged intent dataset from a CSV file.

    Parameters
    ----------
    csv_path:
        Path to the CSV produced by ``gather_dataset.py``.
        Expected columns: ``lang``, ``label``, ``sentence``.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame ready for balancing and splitting.
    """
    logger.info(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    # Support both merged formats:
    #   merged_intents_dataset.csv      → has a single "label" column
    #   merged_intents_dataset_full.csv → has separate "domain" and "intent" columns
    if "label" not in df.columns and {"domain", "intent"}.issubset(df.columns):
        df["label"] = df["domain"] + ":" + df["intent"]
        df.drop(columns=["domain", "intent"], inplace=True)
    before = len(df)
    df.drop_duplicates(inplace=True)
    logger.info(
        f"Loaded {len(df)} rows  ({before - len(df)} duplicates removed)  "
        f"| {df['label'].nunique()} unique labels"
    )
    lang_dist = df["lang"].value_counts().to_dict()
    for lang, cnt in sorted(lang_dist.items()):
        logger.info(f"  {lang:<6} {cnt:>7} examples")
    return df


def balance_dataset(
    df: pd.DataFrame,
    min_samples: int = MIN_SAMPLES_PER_CLASS,
    max_samples: int | None = MAX_SAMPLES_PER_CLASS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """Filter rare classes and cap over-represented ones.

    Two independent passes are applied:

    1. **Minimum filter** – any label with fewer than ``min_samples`` rows is
       dropped entirely.  This removes noise-prone, barely-trained classes that
       would otherwise degrade generalisation.

    2. **Maximum cap** – any label with more than ``max_samples`` rows is
       randomly down-sampled to exactly ``max_samples`` rows.  This prevents
       high-frequency intents from dominating training and biasing the model.

    Parameters
    ----------
    df:
        Input DataFrame with at least ``label`` and ``sentence`` columns.
    min_samples:
        Minimum number of examples a label must have to be retained.
        Set to 1 to keep all classes.
    max_samples:
        Maximum number of examples kept per label.  ``None`` disables capping.
    random_state:
        Seed used for the down-sampling step.

    Returns
    -------
    pd.DataFrame
        Balanced DataFrame, reset index.
    """
    counts_before = df["label"].value_counts()
    n_labels_before = len(counts_before)
    n_rows_before = len(df)

    # -- Step 1: drop rare labels -------------------------------------------
    keep_labels = counts_before[counts_before >= min_samples].index
    dropped_labels = n_labels_before - len(keep_labels)
    df = df[df["label"].isin(keep_labels)].copy()
    rows_after_min = len(df)
    logger.info(
        f"Min-filter (< {min_samples} samples): "
        f"dropped {dropped_labels} labels, "
        f"{n_rows_before - rows_after_min} rows removed  "
        f"→ {df['label'].nunique()} labels / {rows_after_min} rows remain"
    )

    # -- Step 2: cap over-represented labels --------------------------------
    if max_samples is not None:
        capped_labels = (df["label"].value_counts() > max_samples).sum()
        df = pd.concat(
            [g.sample(min(len(g), max_samples), random_state=random_state)
             for _, g in df.groupby("label")],
            ignore_index=True,
        )
        rows_after_cap = len(df)
        logger.info(
            f"Max-cap  (> {max_samples} samples): "
            f"capped {capped_labels} labels, "
            f"{rows_after_min - rows_after_cap} rows removed  "
            f"→ {rows_after_cap} rows remain"
        )
    else:
        df = df.reset_index(drop=True)
        logger.info("Max-cap disabled (MAX_SAMPLES_PER_CLASS = None)")

    # Summary statistics after balancing
    counts_after = df["label"].value_counts()
    logger.info(
        f"Post-balance class stats: "
        f"min={counts_after.min()}  max={counts_after.max()}  "
        f"mean={counts_after.mean():.1f}  median={counts_after.median():.0f}"
    )
    return df


def balanced_split(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into stratified train/test sets.

    Stratification is performed on the ``label`` column so that each class
    keeps the same proportion in both splits.  Falls back to a random shuffle
    split if stratification is not possible (e.g. singletons still present
    after balancing).

    Parameters
    ----------
    df:
        Balanced DataFrame with ``lang``, ``label``, ``sentence`` columns.
    test_size:
        Fraction of samples to reserve for testing.
    random_state:
        Seed for reproducibility.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(train_df, test_df)`` — both retain all original columns.
    """
    idx = df.index.values
    y = df["label"].values

    # Warn about singletons that cannot be stratified
    counts = pd.Series(y).value_counts()
    singletons = (counts == 1).sum()
    if singletons:
        logger.warning(
            f"{singletons} label(s) still have exactly 1 sample after balancing "
            "– stratification will fall back to random split."
        )

    try:
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=random_state, stratify=y
        )
    except ValueError:
        logger.warning("Stratified split failed; using random shuffle split.")
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=random_state, shuffle=True
        )

    train_df = df.loc[idx_train].reset_index(drop=True)
    test_df = df.loc[idx_test].reset_index(drop=True)
    logger.info(
        f"Split: {len(train_df)} train / {len(test_df)} test  "
        f"(test_size={test_size}, stratified by label)"
    )
    return train_df, test_df


# ---------------------------------------------------------------------------
# Per-language evaluation
# ---------------------------------------------------------------------------

def evaluate_per_language(
    classifier: StaticModelForClassification,
    test_df: pd.DataFrame,
) -> dict[str, dict]:
    """Evaluate the classifier separately for each language in the test set.

    Parameters
    ----------
    classifier:
        A fitted ``StaticModelForClassification`` instance.
    test_df:
        Test DataFrame with columns ``lang``, ``label``, ``sentence``.

    Returns
    -------
    dict[str, dict]
        Keyed by language code.  Each value is a dict with keys:
        ``n_samples``, ``accuracy``, ``f1``, ``throughput_sps``.
    """
    results: dict[str, dict] = {}
    for lang in sorted(test_df["lang"].unique()):
        sub = test_df[test_df["lang"] == lang]
        X_sub = sub["sentence"].values
        y_sub = sub["label"].values

        t0 = time.monotonic()
        y_pred = classifier.predict(X_sub)
        elapsed = time.monotonic() - t0

        acc = accuracy_score(y_sub, y_pred)
        f1 = f1_score(y_sub, y_pred, average="weighted", zero_division=0)
        throughput = len(X_sub) / elapsed if elapsed > 0 else float("inf")

        results[lang] = {
            "n_samples": int(len(X_sub)),
            "accuracy": round(acc, 4),
            "f1": round(f1, 4),
            "throughput_sps": round(throughput, 0),
        }
        logger.info(
            f"  [{lang}] n={len(X_sub):>6}  "
            f"acc={acc:.4f}  f1={f1:.4f}  "
            f"throughput={throughput:.0f} sps"
        )
    return results


# ---------------------------------------------------------------------------
# Model size helpers
# ---------------------------------------------------------------------------

def measure_model_size(model_dir: str) -> dict:
    """Measure the on-disk size and parameter count of a saved Model2Vec pipeline.

    Parameters
    ----------
    model_dir:
        Path to the directory produced by ``pipeline.save_pretrained()``.

    Returns
    -------
    dict
        - ``size_mb`` (float): total directory size in megabytes.
        - ``n_params`` (int): total scalar parameters summed across all
          ``.safetensors`` weight files.  ``-1`` if ``safetensors`` is not
          installed.
    """
    # -- On-disk size --------------------------------------------------------
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, fname))
        for dp, _, files in os.walk(model_dir)
        for fname in files
    )
    size_mb = total_bytes / (1024 * 1024)

    # -- Parameter count from all safetensors weight files -------------------
    n_params = 0
    try:
        from safetensors import safe_open  # type: ignore[import]
        for dp, _, files in os.walk(model_dir):
            for fname in files:
                if fname.endswith(".safetensors"):
                    with safe_open(os.path.join(dp, fname), framework="np") as st:
                        for key in st.keys():
                            n_params += int(np.prod(st.get_tensor(key).shape))
    except ImportError:
        logger.warning("safetensors not installed; parameter count will be reported as -1.")
        n_params = -1

    return {"size_mb": round(size_mb, 2), "n_params": n_params}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_class_distribution(
    y: np.ndarray,
    title: str,
    out_path: str,
    top_n: int = 30,
) -> None:
    """Save a horizontal bar chart of the most frequent classes.

    Parameters
    ----------
    y:
        Full label array (train + test combined).
    title:
        Chart title.
    out_path:
        File path where the PNG will be saved.
    top_n:
        How many of the most frequent labels to show.
    """
    counts = pd.Series(y).value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=(14, max(6, top_n // 3)))
    counts[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of examples")
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved class-distribution plot → {out_path}")


def plot_per_class_f1(
    report_dict: dict,
    title: str,
    out_path: str,
    top_n: int = F1_PLOT_TOP_N,
) -> None:
    """Save a sorted bar chart of per-class F1 scores.

    Shows the ``top_n`` worst and ``top_n`` best performing classes.  When
    fewer than ``2 * top_n`` classes exist, all are shown.  Bars below 0.5
    are coloured red to highlight underperforming intents.

    Parameters
    ----------
    report_dict:
        Output of ``sklearn.metrics.classification_report(output_dict=True)``.
    title:
        Chart title.
    out_path:
        File path where the PNG will be saved.
    top_n:
        Number of worst/best classes to display.
    """
    skip_keys = {"accuracy", "macro avg", "weighted avg"}
    f1_scores = {
        label: vals["f1-score"]
        for label, vals in report_dict.items()
        if label not in skip_keys
    }
    if not f1_scores:
        return

    series = pd.Series(f1_scores).sort_values(ascending=True)
    if len(series) > 2 * top_n:
        # Show extremes only; add a visual gap between worst and best
        series = pd.concat([series.head(top_n), series.tail(top_n)])

    fig, ax = plt.subplots(figsize=(14, max(6, len(series) // 3)))
    colors = ["tomato" if v < 0.5 else "steelblue" for v in series.values]
    series.plot(kind="barh", ax=ax, color=colors)
    ax.axvline(0.5, color="grey", linestyle="--", linewidth=0.8, label="F1 = 0.5")
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.tick_params(axis="y", labelsize=7)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved per-class F1 plot → {out_path}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: str,
    max_classes: int = CONF_MATRIX_MAX_CLASSES,
) -> None:
    """Save a row-normalised confusion matrix heatmap.

    When the number of unique classes exceeds ``max_classes``, the matrix is
    restricted to the ``max_classes`` most frequent labels in ``y_true``.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    labels:
        Ordered list of class names.
    title:
        Chart title.
    out_path:
        File path where the PNG will be saved.
    max_classes:
        Maximum number of classes to include in the plot.
    """
    if len(labels) > max_classes:
        logger.info(
            f"Confusion matrix: {len(labels)} classes exceed limit ({max_classes}). "
            f"Restricting to {max_classes} most frequent."
        )
        top_labels = pd.Series(y_true).value_counts().head(max_classes).index.tolist()
        mask = np.isin(y_true, top_labels) & np.isin(y_pred, top_labels)
        y_true, y_pred = y_true[mask], y_pred[mask]
        labels = top_labels

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    # Row-normalise → each cell shows recall for that true class
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    cell_size = max(0.3, 12 / len(labels))
    fig_size = max(8, len(labels) * cell_size)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    sns.heatmap(
        cm_norm,
        annot=len(labels) <= 20,
        fmt=".2f",
        xticklabels=labels,
        yticklabels=labels,
        cmap="Blues",
        ax=ax,
        linewidths=0.3 if len(labels) <= 20 else 0,
        cbar_kws={"label": "Recall (row-normalised)"},
    )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    ax.tick_params(axis="both", labelsize=max(4, 8 - len(labels) // 10))
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved confusion matrix → {out_path}")


def plot_per_language_metrics(
    per_lang: dict[str, dict],
    title: str,
    out_path: str,
) -> None:
    """Save a grouped bar chart of accuracy and F1 score per language.

    Each language gets two adjacent bars (accuracy in blue, F1 in orange).
    The sample count is annotated above each group.

    Parameters
    ----------
    per_lang:
        Output of :func:`evaluate_per_language`.
    title:
        Chart title.
    out_path:
        File path where the PNG will be saved.
    """
    if not per_lang:
        return

    langs = sorted(per_lang.keys())
    accs = [per_lang[l]["accuracy"] for l in langs]
    f1s = [per_lang[l]["f1"] for l in langs]
    ns = [per_lang[l]["n_samples"] for l in langs]

    x = np.arange(len(langs))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(8, len(langs) * 1.2), 5))
    bars_acc = ax.bar(x - width / 2, accs, width, label="Accuracy", color="steelblue")
    bars_f1 = ax.bar(x + width / 2, f1s, width, label="Weighted F1", color="darkorange")

    # Annotate sample counts above each language group
    for i, (bar_a, bar_f, n) in enumerate(zip(bars_acc, bars_f1, ns)):
        ax.text(
            x[i], max(bar_a.get_height(), bar_f.get_height()) + 0.01,
            f"n={n}", ha="center", va="bottom", fontsize=7, color="dimgrey"
        )

    ax.set_xticks(x)
    ax.set_xticklabels(langs, fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    ax.axhline(0.8, color="grey", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved per-language metrics plot → {out_path}")


def plot_cross_model_language_heatmap(
    metrics_summary: list[dict],
    metric: str,
    out_path: str,
) -> None:
    """Save a heatmap of a metric across all models and languages.

    Rows are languages, columns are model short names.  Useful for identifying
    which model generalises best to each language.

    Parameters
    ----------
    metrics_summary:
        List of result dicts returned by :func:`train_and_evaluate`.
    metric:
        Key inside each ``per_lang_metrics[lang]`` dict, e.g. ``"accuracy"``
        or ``"f1"``.
    out_path:
        File path where the PNG will be saved.
    """
    # Collect all language codes
    all_langs: set[str] = set()
    for s in metrics_summary:
        all_langs.update(s["per_lang_metrics"].keys())
    langs = sorted(all_langs)

    model_names = [s["run_id"] for s in metrics_summary]

    # Build matrix: rows=langs, cols=models
    matrix = np.full((len(langs), len(model_names)), np.nan)
    for col_idx, s in enumerate(metrics_summary):
        for row_idx, lang in enumerate(langs):
            if lang in s["per_lang_metrics"]:
                matrix[row_idx, col_idx] = s["per_lang_metrics"][lang][metric]

    fig, ax = plt.subplots(figsize=(max(6, len(model_names) * 1.8), max(4, len(langs) * 0.6)))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".3f",
        xticklabels=model_names,
        yticklabels=langs,
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        ax=ax,
        cbar_kws={"label": metric.replace("_", " ").title()},
        linewidths=0.5,
    )
    ax.set_xlabel("Model")
    ax.set_ylabel("Language")
    ax.set_title(f"Cross-model {metric} by language")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved cross-model heatmap ({metric}) → {out_path}")


def plot_model_sizes(
    metrics_summary: list[dict],
    out_path: str,
) -> None:
    """Save a side-by-side bar chart comparing model disk size and parameter count.

    Left panel shows on-disk size in megabytes; right panel shows the total
    parameter count in millions.  Both panels are sorted by descending disk
    size for easy visual ranking.  Parameter counts are annotated as text
    labels; ``N/A`` is shown when the ``safetensors`` library is unavailable.

    Parameters
    ----------
    metrics_summary:
        List of result dicts from :func:`train_and_evaluate`.
        Each must contain ``run_id``, ``size_mb``, and ``n_params``.
    out_path:
        File path where the PNG will be saved.
    """
    run_ids = [s["run_id"] for s in metrics_summary]
    size_mbs = [s["size_mb"] for s in metrics_summary]
    n_params_raw = [s["n_params"] for s in metrics_summary]

    # Sort by disk size descending so the largest model is at the top
    order = sorted(range(len(size_mbs)), key=lambda i: size_mbs[i], reverse=True)
    run_ids   = [run_ids[i]      for i in order]
    size_mbs  = [size_mbs[i]     for i in order]
    n_params_raw = [n_params_raw[i] for i in order]
    params_m  = [p / 1e6 if p >= 0 else 0.0 for p in n_params_raw]

    n = len(run_ids)
    fig, (ax_mb, ax_p) = plt.subplots(
        1, 2,
        figsize=(max(14, n * 0.8), max(5, n * 0.45)),
        sharey=True,
    )

    # -- Left panel: disk size (MB) ------------------------------------------
    bars_mb = ax_mb.barh(run_ids, size_mbs, color="steelblue")
    for bar, val in zip(bars_mb, size_mbs):
        ax_mb.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f} MB", va="center", fontsize=7,
        )
    ax_mb.set_xlabel("Disk size (MB)")
    ax_mb.set_title("On-disk size")
    ax_mb.tick_params(axis="y", labelsize=7)

    # -- Right panel: parameter count (millions) -----------------------------
    bars_p = ax_p.barh(run_ids, params_m, color="darkorange")
    for bar, val_m, raw in zip(bars_p, params_m, n_params_raw):
        label = f"{val_m:.2f} M" if raw >= 0 else "N/A"
        ax_p.text(
            bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
            label, va="center", fontsize=7,
        )
    ax_p.set_xlabel("Parameters (millions)")
    ax_p.set_title("Parameter count")
    ax_p.tick_params(axis="y", labelsize=7)

    fig.suptitle("Model size comparison", fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved model size comparison plot → {out_path}")


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_and_evaluate(
    base_model: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    langs: list[str] | None = None,
) -> dict:
    """Train a classifier on top of a distilled base model and evaluate it.

    Overall and per-language metrics are computed, plots are saved, and a
    Markdown report is written.

    Parameters
    ----------
    base_model:
        Path to (or HuggingFace ID of) the distilled Model2Vec checkpoint.
    train_df:
        Full training DataFrame with ``lang``, ``label``, ``sentence`` columns.
    test_df:
        Full test DataFrame with the same columns.
    langs:
        Optional list of ISO-639-1 language codes to restrict training and
        evaluation to.  When provided:

        - Both ``train_df`` and ``test_df`` are filtered to rows whose
          ``lang`` is in this list.
        - The filtered ``train_df`` is re-balanced independently (the global
          min/max constraints are re-applied within the language subset).
        - Output files are suffixed with the language codes so that multiple
          variants of the same base model do not overwrite each other.

        ``None`` uses all languages (multilingual mode).

    Returns
    -------
    dict
        Keys: ``run_id``, ``model``, ``langs``, ``accuracy``, ``f1_score``,
        ``report``, ``report_dict``, ``train_time_s``, ``predict_time_s``,
        ``throughput_sps``, ``per_lang_metrics``.
    """
    model_name = base_model.split("/")[-1]

    # -- Derive a unique run identifier --------------------------------------
    # When langs is set, append the sorted language codes so that different
    # language variants of the same base model produce distinct output files.
    if langs is not None:
        lang_suffix = "_" + "-".join(sorted(langs))
        run_id = model_name + lang_suffix
        display_langs = ", ".join(sorted(langs))
    else:
        lang_suffix = ""
        run_id = model_name
        display_langs = "all"

    logger.info(f"  Language filter : {display_langs}")
    logger.info(f"  Run identifier  : {run_id}")

    # -- Apply language filter -----------------------------------------------
    if langs is not None:
        train_df = train_df[train_df["lang"].isin(langs)].reset_index(drop=True)
        test_df = test_df[test_df["lang"].isin(langs)].reset_index(drop=True)
        logger.info(
            f"  After lang filter: {len(train_df)} train / {len(test_df)} test rows"
        )
        if len(train_df) == 0:
            raise ValueError(
                f"Language filter {langs} produced an empty training set. "
                "Check that the requested languages exist in the dataset."
            )

        # Re-balance the filtered training slice: label frequencies change
        # after removing other languages, so re-apply min/max constraints.
        logger.info("  Re-balancing training slice after language filter ...")
        train_df = balance_dataset(train_df)

    X_train = train_df["sentence"].values
    y_train = train_df["label"].values
    X_test = test_df["sentence"].values
    y_test = test_df["label"].values

    # -- Training ------------------------------------------------------------
    logger.info(f"Training classifier on top of {base_model} ...")
    classifier = StaticModelForClassification.from_pretrained(model_name=base_model)

    t0 = time.monotonic()
    classifier.fit(X_train, y_train, max_epochs=MAX_EPOCHS)
    train_time = time.monotonic() - t0
    logger.info(f"  Training completed in {train_time:.1f}s")

    # -- Overall inference benchmark -----------------------------------------
    logger.info("Running overall predictions ...")
    t0 = time.monotonic()
    y_pred = classifier.predict(X_test)
    predict_time = time.monotonic() - t0
    throughput = len(X_test) / predict_time if predict_time > 0 else float("inf")
    logger.info(
        f"  Predicted {len(X_test)} samples in {predict_time:.3f}s "
        f"({throughput:.0f} sps)"
    )

    # -- Overall metrics -----------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    logger.info(f"  Overall accuracy     : {accuracy:.4f}")
    logger.info(f"  Overall weighted F1  : {f1:.4f}")

    # -- Per-language evaluation ---------------------------------------------
    logger.info("Evaluating per language ...")
    per_lang = evaluate_per_language(classifier, test_df)

    # -- Save model ----------------------------------------------------------
    # Use run_id so language-specific variants don't overwrite each other
    output_dir = os.path.join(OUTPUT_DIR, f"model_mul_{run_id}")
    classifier.to_pipeline().save_pretrained(output_dir)
    logger.info(f"  Model saved → {output_dir}")

    # -- Model size ----------------------------------------------------------
    size_info = measure_model_size(output_dir)
    params_label = (
        f"{size_info['n_params'] / 1e6:.2f} M"
        if size_info["n_params"] >= 0
        else "N/A"
    )
    logger.info(
        f"  Model size: {size_info['size_mb']:.2f} MB  |  {params_label} parameters"
    )

    # -- Plots ---------------------------------------------------------------
    plots_dir = os.path.join(OUTPUT_DIR, f"plots_{run_id}")
    os.makedirs(plots_dir, exist_ok=True)

    all_labels = sorted(set(y_test) | set(y_pred))

    plot_class_distribution(
        np.concatenate([y_train, y_test]),
        title=f"Class distribution – {run_id}",
        out_path=os.path.join(plots_dir, "class_distribution.png"),
    )
    plot_per_class_f1(
        report_dict,
        title=f"Per-class F1 – {run_id}",
        out_path=os.path.join(plots_dir, "per_class_f1.png"),
    )
    plot_confusion_matrix(
        y_test, y_pred,
        labels=all_labels,
        title=f"Confusion matrix – {run_id}",
        out_path=os.path.join(plots_dir, "confusion_matrix.png"),
    )
    plot_per_language_metrics(
        per_lang,
        title=f"Per-language accuracy & F1 – {run_id}",
        out_path=os.path.join(plots_dir, "per_language_metrics.png"),
    )

    # -- Markdown report -----------------------------------------------------
    md_path = os.path.join(OUTPUT_DIR, f"metrics_{run_id}.md")
    with open(md_path, "w") as fh:
        fh.write(f"# Model Evaluation – {run_id}\n\n")
        fh.write(f"**Base model:** `{base_model}`  \n")
        fh.write(f"**Language filter:** {display_langs}\n\n")

        fh.write("## Dataset balance settings\n\n")
        fh.write("| Setting | Value |\n|---|---|\n")
        fh.write(f"| Min samples per class | {MIN_SAMPLES_PER_CLASS} |\n")
        fh.write(f"| Max samples per class | {MAX_SAMPLES_PER_CLASS} |\n")
        fh.write(f"| Language filter | {display_langs} |\n")
        fh.write(f"| Training examples | {len(X_train)} |\n")
        fh.write(f"| Test examples | {len(X_test)} |\n\n")

        fh.write("## Model size\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        fh.write(f"| Disk size | {size_info['size_mb']:.2f} MB |\n")
        fh.write(f"| Parameters | {params_label} |\n\n")

        fh.write("## Benchmark\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        fh.write(f"| Training time | {train_time:.1f}s |\n")
        fh.write(f"| Inference time ({len(X_test)} samples) | {predict_time:.3f}s |\n")
        fh.write(f"| Throughput | {throughput:.0f} sps |\n\n")

        fh.write("## Overall evaluation\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        fh.write(f"| Accuracy | {accuracy:.4f} |\n")
        fh.write(f"| Weighted F1 | {f1:.4f} |\n\n")
        fh.write(f"### Classification report\n\n```\n{report_text}\n```\n\n")

        fh.write("## Per-language evaluation\n\n")
        fh.write("| Language | Samples | Accuracy | Weighted F1 | Throughput (sps) |\n")
        fh.write("|---|---|---|---|---|\n")
        for lang in sorted(per_lang.keys()):
            m = per_lang[lang]
            fh.write(
                f"| {lang} | {m['n_samples']} | {m['accuracy']:.4f} "
                f"| {m['f1']:.4f} | {m['throughput_sps']:.0f} |\n"
            )
        fh.write("\n")
    logger.info(f"  Metrics report saved → {md_path}")

    return {
        "run_id": run_id,
        "model": base_model,
        "langs": langs,
        "accuracy": accuracy,
        "f1_score": f1,
        "report": report_text,
        "report_dict": report_dict,
        "train_time_s": round(train_time, 2),
        "predict_time_s": round(predict_time, 3),
        "throughput_sps": round(throughput, 0),
        "per_lang_metrics": per_lang,
        "size_mb": size_info["size_mb"],
        "n_params": size_info["n_params"],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Orchestrate dataset loading, balancing, splitting, training, and reporting."""
    # -- Load full dataset ---------------------------------------------------
    df = load_dataset(CSV_PATH)

    # -- Balance before splitting so both splits share the same label set ----
    logger.info(
        f"Balancing dataset  "
        f"(min_samples={MIN_SAMPLES_PER_CLASS}, max_samples={MAX_SAMPLES_PER_CLASS}) ..."
    )
    df = balance_dataset(df)
    logger.info(f"Balanced dataset: {len(df):,} rows")

    # -- Stratified train/test split -----------------------------------------
    train_df, test_df = balanced_split(df)
    logger.info(f"Split: {len(train_df):,} train / {len(test_df):,} test")

    # -- Train & evaluate each model -----------------------------------------
    for cfg in base_models:
        path = cfg["path"]
        langs = cfg.get("langs")  # None = all languages
        logger.info("=" * 70)
        logger.info(f"Model : {path}")
        logger.info(f"Langs : {langs if langs is not None else 'all'}")
        logger.info("=" * 70)
        try:
            result = train_and_evaluate(path, train_df, test_df, langs=langs)
            metrics_summary.append(result)
        except Exception as exc:
            logger.error(f"Failed to train {path} (langs={langs}): {exc}")

    if not metrics_summary:
        logger.warning("No models were trained.")
        return

    # -- Cross-model comparison table ----------------------------------------
    rows = []
    for s in tqdm(metrics_summary, desc="Building comparison table", unit="model"):
        lang_label = ", ".join(sorted(s["langs"])) if s["langs"] else "all"
        params_m = (
            f"{s['n_params'] / 1e6:.2f} M" if s["n_params"] >= 0 else "N/A"
        )
        base_row = {
            "Run": s["run_id"],
            "Base model": s["model"].split("/")[-1],
            "Languages": lang_label,
            "Size (MB)": s["size_mb"],
            "Params": params_m,
            "Accuracy": s["accuracy"],
            "F1 Score": s["f1_score"],
            "Train time (s)": s["train_time_s"],
            "Throughput (sps)": s["throughput_sps"],
        }
        # Flatten per-language F1 into the row for the comparison table
        for lang, m in sorted(s["per_lang_metrics"].items()):
            base_row[f"F1 [{lang}]"] = m["f1"]
        rows.append(base_row)

    comparison_df = pd.DataFrame(rows)

    with open(os.path.join(OUTPUT_DIR, "model_comparison.md"), "w") as fh:
        fh.write("# Model Comparison\n\n")
        fh.write(f"**Balance settings:** min_samples={MIN_SAMPLES_PER_CLASS}, "
                 f"max_samples={MAX_SAMPLES_PER_CLASS}\n\n")
        fh.write("## Overall metrics\n\n")
        overall_cols = ["Run", "Base model", "Languages", "Size (MB)", "Params", "Accuracy", "F1 Score", "Train time (s)", "Throughput (sps)"]
        fh.write(comparison_df[overall_cols].to_markdown(index=False))
        fh.write("\n\n## Per-language F1 scores\n\n")
        lang_cols = ["Run"] + [c for c in comparison_df.columns if c.startswith("F1 [")]
        fh.write(comparison_df[lang_cols].to_markdown(index=False))
        fh.write("\n")
    logger.info(f"Model comparison table saved → {os.path.join(OUTPUT_DIR, 'model_comparison.md')}")

    # -- Model size comparison plot -------------------------------------------
    plot_model_sizes(metrics_summary, out_path=os.path.join(OUTPUT_DIR, "model_sizes.png"))

    # -- Cross-model heatmaps -------------------------------------------------
    if len(metrics_summary) > 1:
        plot_cross_model_language_heatmap(
            metrics_summary, metric="accuracy",
            out_path=os.path.join(OUTPUT_DIR, "cross_model_accuracy_heatmap.png")
        )
        plot_cross_model_language_heatmap(
            metrics_summary, metric="f1",
            out_path=os.path.join(OUTPUT_DIR, "cross_model_f1_heatmap.png")
        )

    # -- Console summary ------------------------------------------------------
    logger.info("\nFinal summary:")
    for s in metrics_summary:
        lang_f1 = "  ".join(
            f"{l}={m['f1']:.3f}" for l, m in sorted(s["per_lang_metrics"].items())
        )
        params_label = (
            f"{s['n_params'] / 1e6:.2f}M params"
            if s["n_params"] >= 0 else "N/A params"
        )
        logger.info(
            f"  {s['run_id']:<48} "
            f"acc={s['accuracy']:.4f}  f1={s['f1_score']:.4f}  "
            f"{s['size_mb']:.2f} MB  {params_label}  "
            f"train={s['train_time_s']}s | {lang_f1}"
        )


if __name__ == "__main__":
    main()
