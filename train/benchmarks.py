"""Benchmark trained Model2Vec intent classifiers across dataset variants.

For every trained model discovered in the working directory, this script
evaluates performance on each configured dataset variant using a fresh
stratified hold-out split.  Results are broken down by language so you can
see where each model excels or degrades.

Discovered models
-----------------
Any directory matching ``model_mul_*/`` in the working directory is treated
as a trained model.  Override by editing ``MODEL_DIRS`` below.

Dataset variants
----------------
``DATASETS`` maps a short label to a CSV path.  Add or remove entries freely.
Each CSV must have at least ``lang``, ``label``, ``sentence`` columns.

Outputs
-------
``benchmark_results.csv``      – raw (model × dataset × lang) metrics table
``benchmark_report.md``        – human-readable markdown summary
``benchmark_plots/``
    f1_heatmap.png             – weighted F1: models × dataset variants
    accuracy_heatmap.png       – accuracy:    models × dataset variants
    per_lang_f1_<ds>.png       – weighted F1: models × languages, one per dataset
    throughput.png             – inference throughput (samples/s) per model
    size_vs_f1.png             – scatter: on-disk MB vs weighted F1, labelled

Usage::

    python benchmarks.py

Tuning::

    TEST_SIZE      = 0.2    # fraction of each dataset held out for evaluation
    MAX_TEST_ROWS  = 5000   # cap on test-set size per dataset (keep benchmarks fast)
    RANDOM_STATE   = 42
"""

import glob
import logging
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from model2vec.inference import StaticModelPipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset variants to evaluate on.  Key = short name used in plots/tables.
DATASETS: dict[str, str] = {
    "full":    os.path.join(OUTPUT_DIR, "merged_intents_dataset.csv"),
    "diverse": os.path.join(OUTPUT_DIR, "diverse_subset.csv"),
}

# Per-language CSVs produced by gather_dataset.py → output/by_lang/intents_<lang>.csv
# Key = language code, value = CSV path.  Auto-discovered; override if needed.
LANG_DIR: str = os.path.join(OUTPUT_DIR, "by_lang")
LANG_DATASETS: dict[str, str] = {
    os.path.basename(p).replace("intents_", "").replace(".csv", ""): p
    for p in sorted(glob.glob(os.path.join(LANG_DIR, "intents_*.csv")))
    if not p.endswith("_full.csv")
}

# Auto-discover trained models (multilingual and monolingual); override if needed.
MODEL_DIRS: list[str] = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "model_mul_*/")) +
    glob.glob(os.path.join(OUTPUT_DIR, "model_mono_*/"))
)

TEST_SIZE: float = 0.2
MAX_TEST_ROWS: int = 5000   # cap test-set rows so large datasets run quickly
MIN_LABEL_SAMPLES: int = 2  # labels with fewer samples are dropped (can't stratify)
RANDOM_STATE: int = 42

OUT_RESULTS  = os.path.join(OUTPUT_DIR, "benchmark_results.csv")
OUT_REPORT   = os.path.join(OUTPUT_DIR, "benchmark_report.md")
OUT_PLOTS    = os.path.join(OUTPUT_DIR, "benchmark_plots")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def model_size_mb(model_dir: str) -> float:
    """Return the total on-disk size of a model directory in megabytes."""
    total = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, files in os.walk(model_dir)
        for f in files
    )
    return total / (1024 * 1024)


def model_n_params(model_dir: str) -> int:
    """Count scalar parameters from all .safetensors files; -1 if unavailable."""
    n = 0
    try:
        from safetensors import safe_open  # type: ignore[import]
        for dp, _, files in os.walk(model_dir):
            for fname in files:
                if fname.endswith(".safetensors"):
                    with safe_open(os.path.join(dp, fname), framework="np") as st:
                        for key in st.keys():
                            n += int(np.prod(st.get_tensor(key).shape))
    except ImportError:
        return -1
    return n


def load_dataset(csv_path: str) -> pd.DataFrame | None:
    """Load a CSV dataset, returning None if the file does not exist."""
    if not os.path.exists(csv_path):
        logger.warning(f"Dataset not found, skipping: {csv_path}")
        return None
    df = pd.read_csv(csv_path)
    df.drop_duplicates(inplace=True)
    return df


def stratified_test_split(
    df: pd.DataFrame,
    known_labels: set[str],
    test_size: float = TEST_SIZE,
    max_rows: int = MAX_TEST_ROWS,
) -> pd.DataFrame:
    """Return a stratified test split restricted to labels the model knows.

    Parameters
    ----------
    df:
        Full dataset with ``label`` column.
    known_labels:
        Set of label strings the model was trained on (``model.classes_``).
    test_size:
        Fraction to hold out.
    max_rows:
        Hard cap on the returned test set size; random-sampled if exceeded.

    Returns
    -------
    pd.DataFrame or empty DataFrame if insufficient data.
    """
    # Keep only labels the model knows and that have enough samples to split
    sub = df[df["label"].isin(known_labels)].copy()
    counts = sub["label"].value_counts()
    keep = counts[counts >= MIN_LABEL_SAMPLES].index
    sub = sub[sub["label"].isin(keep)]

    if len(sub) < 10:
        return pd.DataFrame()

    try:
        _, test_df = train_test_split(
            sub, test_size=test_size, stratify=sub["label"], random_state=RANDOM_STATE
        )
    except ValueError:
        _, test_df = train_test_split(
            sub, test_size=test_size, random_state=RANDOM_STATE, shuffle=True
        )

    if len(test_df) > max_rows:
        test_df = test_df.sample(max_rows, random_state=RANDOM_STATE)

    return test_df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_overall(
    model: StaticModelPipeline,
    test_df: pd.DataFrame,
) -> dict:
    """Compute overall accuracy, weighted F1, macro F1, and throughput.

    Parameters
    ----------
    model:
        Loaded ``StaticModelPipeline`` instance.
    test_df:
        Test DataFrame with ``sentence`` and ``label`` columns.

    Returns
    -------
    dict with keys: accuracy, f1_weighted, f1_macro, throughput_sps, n_samples.
    """
    X = test_df["sentence"].values
    y = test_df["label"].values

    t0 = time.monotonic()
    y_pred = model.predict(X)
    elapsed = time.monotonic() - t0

    return {
        "n_samples":      len(X),
        "accuracy":       round(accuracy_score(y, y_pred), 4),
        "f1_weighted":    round(f1_score(y, y_pred, average="weighted", zero_division=0), 4),
        "f1_macro":       round(f1_score(y, y_pred, average="macro",    zero_division=0), 4),
        "throughput_sps": round(len(X) / elapsed if elapsed > 0 else float("inf"), 0),
    }


def evaluate_per_language(
    model: StaticModelPipeline,
    test_df: pd.DataFrame,
) -> dict[str, dict]:
    """Evaluate separately for each language present in ``test_df``.

    Parameters
    ----------
    model:
        Loaded ``StaticModelPipeline`` instance.
    test_df:
        Test DataFrame with ``lang``, ``sentence``, ``label`` columns.

    Returns
    -------
    dict keyed by language code.  Each value has accuracy, f1_weighted,
    f1_macro, throughput_sps, n_samples.
    """
    results = {}
    for lang in sorted(test_df["lang"].unique()):
        sub = test_df[test_df["lang"] == lang]
        if len(sub) < 2:
            continue
        X_sub = sub["sentence"].values
        y_sub = sub["label"].values

        t0 = time.monotonic()
        y_pred = model.predict(X_sub)
        elapsed = time.monotonic() - t0

        results[lang] = {
            "n_samples":      len(X_sub),
            "accuracy":       round(accuracy_score(y_sub, y_pred), 4),
            "f1_weighted":    round(f1_score(y_sub, y_pred, average="weighted", zero_division=0), 4),
            "f1_macro":       round(f1_score(y_sub, y_pred, average="macro",    zero_division=0), 4),
            "throughput_sps": round(len(X_sub) / elapsed if elapsed > 0 else float("inf"), 0),
        }
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def _heatmap(matrix: pd.DataFrame, title: str, cbar_label: str, out_path: str) -> None:
    """Save a labelled seaborn heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(matrix.columns) * 1.4),
                                    max(4, len(matrix) * 0.55)))
    sns.heatmap(
        matrix.astype(float), annot=True, fmt=".3f",
        cmap="RdYlGn", vmin=0, vmax=1, ax=ax,
        cbar_kws={"label": cbar_label}, linewidths=0.4,
    )
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved → {out_path}")


def plot_overall_heatmaps(
    records: list[dict],
    plots_dir: str,
) -> None:
    """F1 and accuracy heatmaps: rows = models, columns = dataset variants."""
    df = pd.DataFrame(records)
    if df.empty:
        return

    for metric, label in [("f1_weighted", "Weighted F1"), ("accuracy", "Accuracy")]:
        pivot = df.pivot(index="model", columns="dataset", values=metric)
        fname = f"{'f1' if 'f1' in metric else 'accuracy'}_heatmap.png"
        _heatmap(
            pivot,
            title=f"{label} – models × datasets",
            cbar_label=label,
            out_path=os.path.join(plots_dir, fname),
        )


def plot_per_lang_heatmaps(
    lang_records: list[dict],
    plots_dir: str,
) -> None:
    """Per-language F1 heatmap for each dataset variant: rows = models, cols = langs."""
    df = pd.DataFrame(lang_records)
    if df.empty:
        return

    for ds in df["dataset"].unique():
        sub = df[df["dataset"] == ds]
        pivot = sub.pivot(index="model", columns="lang", values="f1_weighted")
        _heatmap(
            pivot,
            title=f"Weighted F1 per language – {ds}",
            cbar_label="Weighted F1",
            out_path=os.path.join(plots_dir, f"per_lang_f1_{ds}.png"),
        )


def plot_throughput(records: list[dict], plots_dir: str) -> None:
    """Horizontal bar chart of inference throughput per model (average across datasets)."""
    df = pd.DataFrame(records)
    if df.empty:
        return

    avg_tp = (
        df.groupby("model")["throughput_sps"]
        .mean()
        .sort_values(ascending=True)
    )
    fig, ax = plt.subplots(figsize=(10, max(4, len(avg_tp) * 0.45)))
    bars = ax.barh(avg_tp.index, avg_tp.values, color="steelblue")
    for bar, val in zip(bars, avg_tp.values):
        ax.text(bar.get_width() * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} sps", va="center", fontsize=7)
    ax.set_xlabel("Samples / second (avg across datasets)")
    ax.set_title("Inference throughput comparison")
    ax.tick_params(axis="y", labelsize=7)
    plt.tight_layout()
    path = os.path.join(plots_dir, "throughput.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_size_vs_f1(records: list[dict], size_map: dict[str, float], plots_dir: str) -> None:
    """Scatter plot of on-disk model size (MB) vs average weighted F1."""
    df = pd.DataFrame(records)
    if df.empty:
        return

    agg = df.groupby("model")["f1_weighted"].mean().reset_index()
    agg["size_mb"] = agg["model"].map(size_map)
    agg = agg.dropna(subset=["size_mb"])

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(agg["size_mb"], agg["f1_weighted"], color="steelblue", s=60, zorder=3)
    for _, row in agg.iterrows():
        ax.annotate(
            row["model"], (row["size_mb"], row["f1_weighted"]),
            fontsize=6.5, textcoords="offset points", xytext=(4, 3),
        )
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Avg weighted F1  (across datasets)")
    ax.set_title("Model size vs performance")
    ax.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    path = os.path.join(plots_dir, "size_vs_f1.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_dataset_comparison(records: list[dict], plots_dir: str) -> None:
    """Grouped bar chart: for each model, F1 side-by-side per dataset variant."""
    df = pd.DataFrame(records)
    if df.empty or df["dataset"].nunique() < 2:
        return

    models   = df["model"].unique()
    datasets = sorted(df["dataset"].unique())
    x = np.arange(len(models))
    width = 0.8 / len(datasets)

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 1.2), 5))
    palette = plt.cm.tab10.colors
    for i, ds in enumerate(datasets):
        sub = df[df["dataset"] == ds].set_index("model")["f1_weighted"].reindex(models)
        bars = ax.bar(x + i * width - 0.4 + width / 2, sub.values, width,
                      label=ds, color=palette[i % len(palette)], alpha=0.85)
        for bar, val in zip(bars, sub.values):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.3f}", ha="center", va="bottom", fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Weighted F1")
    ax.set_title("Weighted F1 per model and dataset variant")
    ax.legend(title="Dataset")
    ax.axhline(0.8, color="grey", linestyle=":", linewidth=0.8)
    plt.tight_layout()
    path = os.path.join(plots_dir, "dataset_comparison.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    logger.info(f"Saved → {path}")


def plot_per_language(bylang_records: list[dict], size_map: dict[str, float],
                      plots_dir: str) -> None:
    """One plot per language: F1 bar chart + size-vs-F1 scatter.

    Monolingual models are coloured blue, multilingual baselines orange.
    Bars are sorted ascending by F1 so the best model is always at the top.
    """
    df = pd.DataFrame(bylang_records)
    if df.empty:
        return

    df["size_mb"] = df["model"].map(size_map)
    # Short display name: strip the model_mul_ / model_mono_<lang>_ prefix
    df["display"] = df["model"].str.replace(r"^model_mul_", "", regex=True)
    df["display"] = df["display"].str.replace(r"^model_mono_[a-z]+_", "", regex=True)

    for lang, grp in df.groupby("lang"):
        grp = grp.sort_values("f1_weighted", ascending=True)
        colors = ["steelblue" if t == "monolingual" else "darkorange"
                  for t in grp["model_type"]]

        fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(grp) * 0.45 + 1)))

        # --- F1 bar chart ---------------------------------------------------
        bars = axes[0].barh(grp["display"], grp["f1_weighted"], color=colors)
        for bar, val in zip(bars, grp["f1_weighted"]):
            axes[0].text(bar.get_width() + 0.003,
                         bar.get_y() + bar.get_height() / 2,
                         f"{val:.3f}", va="center", fontsize=8)
        axes[0].set_xlabel("Weighted F1")
        axes[0].set_xlim(0, 1.08)
        axes[0].set_title(f"[{lang.upper()}] Weighted F1 – all models")
        from matplotlib.patches import Patch
        axes[0].legend(handles=[Patch(color="steelblue", label="monolingual"),
                                 Patch(color="darkorange", label="multilingual")],
                       fontsize=8, loc="lower right")

        # --- Size vs F1 scatter ---------------------------------------------
        for mtype, color, marker in [("monolingual", "steelblue", "o"),
                                      ("multilingual", "darkorange", "^")]:
            sub = grp[grp["model_type"] == mtype]
            axes[1].scatter(sub["size_mb"], sub["f1_weighted"],
                            label=mtype, color=color, marker=marker, s=70, zorder=3)
        for _, row in grp.iterrows():
            axes[1].annotate(row["display"], (row["size_mb"], row["f1_weighted"]),
                             textcoords="offset points", xytext=(4, 2), fontsize=7)
        axes[1].set_xlabel("Model size (MB)")
        axes[1].set_ylabel("Weighted F1")
        axes[1].set_title(f"[{lang.upper()}] Size vs F1")
        axes[1].legend(fontsize=8)

        fig.tight_layout()
        out = os.path.join(plots_dir, f"per_lang_{lang}.png")
        fig.savefig(out, dpi=120)
        plt.close(fig)
        logger.info(f"Saved → {out}")


def plot_bylang_heatmap(bylang_records: list[dict], plots_dir: str) -> None:
    """Heatmap of weighted F1: rows = models, columns = languages.

    Each cell uses the dedicated per-language CSV (not a filtered slice of the
    full dataset), giving a cleaner picture of per-language coverage.
    Rows are sorted so monolingual models appear first, multilingual after.
    """
    df = pd.DataFrame(bylang_records)
    if df.empty:
        return
    pivot = df.pivot(index="model", columns="lang", values="f1_weighted")
    # Sort rows: monolingual first (they have "model_mono_" prefix), then multilingual
    mono_rows = sorted(r for r in pivot.index if r.startswith("model_mono_"))
    mul_rows  = sorted(r for r in pivot.index if not r.startswith("model_mono_"))
    pivot = pivot.reindex(mono_rows + mul_rows)
    _heatmap(
        pivot,
        title="Weighted F1 per language (dedicated lang CSVs)\n"
              "— monolingual models top, multilingual baselines bottom —",
        cbar_label="Weighted F1",
        out_path=os.path.join(plots_dir, "bylang_f1_heatmap.png"),
    )


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_markdown_report(
    records: list[dict],
    lang_records: list[dict],
    bylang_records: list[dict],
    size_map: dict[str, float],
    params_map: dict[str, int],
    out_path: str,
) -> None:
    """Write a human-readable Markdown benchmark report.

    Parameters
    ----------
    records:
        List of overall-metric dicts (model, dataset, accuracy, f1_weighted, …).
    lang_records:
        Per-language metric dicts from filtering the full/diverse test sets.
    bylang_records:
        Per-language metric dicts from dedicated by_lang/ CSV files.
    size_map:
        model name → on-disk MB.
    params_map:
        model name → parameter count (-1 if unknown).
    out_path:
        Output path for the .md file.
    """
    df = pd.DataFrame(records)
    lang_df = pd.DataFrame(lang_records)
    bylang_df = pd.DataFrame(bylang_records)

    with open(out_path, "w") as fh:
        fh.write("# Intent Classifier Benchmark Report\n\n")
        fh.write(f"Datasets evaluated: {', '.join(sorted(df['dataset'].unique()))}\n\n")

        # -- Model size table ------------------------------------------------
        fh.write("## Model sizes\n\n")
        size_rows = [
            {
                "Model": m,
                "Size (MB)": size_map.get(m, "?"),
                "Params": (
                    f"{params_map[m] / 1e6:.2f} M" if params_map.get(m, -1) >= 0 else "N/A"
                ),
            }
            for m in sorted(size_map)
        ]
        fh.write(pd.DataFrame(size_rows).to_markdown(index=False))
        fh.write("\n\n")

        # -- Overall metrics table -------------------------------------------
        fh.write("## Overall metrics\n\n")
        overall_cols = ["model", "dataset", "n_samples", "accuracy",
                        "f1_weighted", "f1_macro", "throughput_sps"]
        fh.write(df[overall_cols].sort_values(["dataset", "f1_weighted"], ascending=[True, False])
                 .to_markdown(index=False))
        fh.write("\n\n")

        # -- Per-language tables (one per dataset) ---------------------------
        if not lang_df.empty:
            fh.write("## Per-language weighted F1 (filtered from full/diverse)\n\n")
            for ds in sorted(lang_df["dataset"].unique()):
                fh.write(f"### Dataset: {ds}\n\n")
                sub = lang_df[lang_df["dataset"] == ds]
                pivot = (
                    sub.pivot(index="model", columns="lang", values="f1_weighted")
                    .round(4)
                    .reset_index()
                )
                fh.write(pivot.to_markdown(index=False))
                fh.write("\n\n")

        # -- Dedicated per-language CSV benchmarks ---------------------------
        if not bylang_df.empty:
            fh.write("## Per-language weighted F1 (dedicated by_lang/ CSVs)\n\n")
            pivot = (
                bylang_df.pivot(index="model", columns="lang", values="f1_weighted")
                .round(4)
                .reset_index()
            )
            fh.write(pivot.to_markdown(index=False))
            fh.write("\n\n")

            fh.write("### Samples evaluated per language\n\n")
            pivot_n = (
                bylang_df.pivot(index="model", columns="lang", values="n_samples")
                .fillna(0).astype(int)
                .reset_index()
            )
            fh.write(pivot_n.to_markdown(index=False))
            fh.write("\n\n")

    logger.info(f"Saved report → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Discover models and datasets, run evaluation, save results and plots."""
    if not MODEL_DIRS:
        logger.error(
            "No trained models found.  Run train_multilingual.py / "
            "train_monolingual.py first, or set MODEL_DIRS manually."
        )
        return

    logger.info(f"Found {len(MODEL_DIRS)} model(s): {[os.path.basename(m.rstrip('/')) for m in MODEL_DIRS]}")
    logger.info(f"Datasets: {list(DATASETS.keys())}")
    logger.info(f"Lang datasets: {list(LANG_DATASETS.keys()) or '(none found)'}")

    os.makedirs(OUT_PLOTS, exist_ok=True)

    # Pre-load datasets once
    datasets: dict[str, pd.DataFrame] = {}
    for ds_name, ds_path in DATASETS.items():
        df = load_dataset(ds_path)
        if df is not None:
            datasets[ds_name] = df
            logger.info(f"  [{ds_name}] {len(df):,} rows  |  {df['label'].nunique()} labels")

    # Pre-load per-language datasets from by_lang/
    lang_datasets: dict[str, pd.DataFrame] = {}
    for lang_code, lang_path in LANG_DATASETS.items():
        df = load_dataset(lang_path)
        if df is not None:
            lang_datasets[lang_code] = df
            logger.info(f"  [lang:{lang_code}] {len(df):,} rows  |  {df['label'].nunique()} labels")

    if not datasets:
        logger.error("No datasets available.")
        return

    # Collect results
    overall_records: list[dict] = []
    lang_records:    list[dict] = []
    bylang_records:  list[dict] = []
    size_map:  dict[str, float] = {}
    params_map: dict[str, int]  = {}

    for model_dir in tqdm(MODEL_DIRS, desc="Models"):
        model_name = os.path.basename(model_dir.rstrip("/"))
        model_type = "monolingual" if model_name.startswith("model_mono_") else "multilingual"

        # Model size
        size_map[model_name]   = round(model_size_mb(model_dir), 2)
        params_map[model_name] = model_n_params(model_dir)

        # Load model
        try:
            model = StaticModelPipeline.from_pretrained(model_dir)
        except Exception as exc:
            logger.error(f"  Failed to load {model_dir}: {exc}")
            continue

        known_labels = set(model.classes_)
        logger.info(
            f"  {model_name}  |  {len(known_labels)} classes  "
            f"|  {size_map[model_name]:.2f} MB"
        )

        for ds_name, df in datasets.items():
            test_df = stratified_test_split(df, known_labels)
            if test_df.empty:
                logger.warning(f"    [{ds_name}] No usable test rows – skipping.")
                continue

            logger.info(f"    [{ds_name}] {len(test_df):,} test rows ...")

            # Overall metrics
            overall = evaluate_overall(model, test_df)
            overall_records.append({
                "model":      model_name,
                "model_type": model_type,
                "dataset":    ds_name,
                **overall,
            })
            logger.info(
                f"      acc={overall['accuracy']:.4f}  "
                f"f1={overall['f1_weighted']:.4f}  "
                f"{overall['throughput_sps']:.0f} sps"
            )

            # Per-language metrics
            per_lang = evaluate_per_language(model, test_df)
            for lang, metrics in per_lang.items():
                lang_records.append({
                    "model":      model_name,
                    "model_type": model_type,
                    "dataset":    ds_name,
                    "lang":       lang,
                    **metrics,
                })

        # Evaluate on dedicated per-language CSVs
        for lang_code, lang_df in lang_datasets.items():
            test_df = stratified_test_split(lang_df, known_labels)
            if test_df.empty:
                logger.warning(f"    [bylang:{lang_code}] No usable test rows – skipping.")
                continue
            logger.info(f"    [bylang:{lang_code}] {len(test_df):,} test rows ...")
            overall = evaluate_overall(model, test_df)
            bylang_records.append({
                "model":      model_name,
                "model_type": model_type,
                "lang":       lang_code,
                **overall,
            })
            logger.info(
                f"      acc={overall['accuracy']:.4f}  "
                f"f1={overall['f1_weighted']:.4f}  "
                f"{overall['throughput_sps']:.0f} sps"
            )

    if not overall_records:
        logger.error("No results collected.")
        return

    # ------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(overall_records)
    results_df.to_csv(OUT_RESULTS, index=False)
    logger.info(f"Saved raw results → {OUT_RESULTS}")

    # ------------------------------------------------------------------
    # Markdown report
    # ------------------------------------------------------------------
    write_markdown_report(overall_records, lang_records, bylang_records, size_map, params_map, OUT_REPORT)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_overall_heatmaps(overall_records, OUT_PLOTS)
    plot_per_lang_heatmaps(lang_records, OUT_PLOTS)
    plot_bylang_heatmap(bylang_records, OUT_PLOTS)
    plot_per_language(bylang_records, size_map, OUT_PLOTS)
    plot_throughput(overall_records, OUT_PLOTS)
    plot_size_vs_f1(overall_records, size_map, OUT_PLOTS)
    plot_dataset_comparison(overall_records, OUT_PLOTS)

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    logger.info("=" * 70)
    logger.info("Benchmark summary (sorted by weighted F1)")
    logger.info("=" * 70)
    summary = (
        results_df.groupby("model")[["accuracy", "f1_weighted", "f1_macro", "throughput_sps"]]
        .mean()
        .sort_values("f1_weighted", ascending=False)
    )
    for model_name, row in summary.iterrows():
        params = params_map.get(model_name, -1)
        params_str = f"{params / 1e6:.2f}M" if params >= 0 else "N/A"
        logger.info(
            f"  {model_name:<48}  "
            f"acc={row['accuracy']:.4f}  f1={row['f1_weighted']:.4f}  "
            f"macro={row['f1_macro']:.4f}  "
            f"{row['throughput_sps']:.0f} sps  "
            f"{size_map.get(model_name, '?'):.2f} MB  {params_str}"
        )


if __name__ == "__main__":
    main()
