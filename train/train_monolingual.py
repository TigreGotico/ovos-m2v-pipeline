"""Train and benchmark monolingual intent classifiers per language.

Languages covered: ca, da, de, es, gl, pt
(languages with sufficient training data for reliable monolingual classifiers;
eu, fr, nl, it have too few examples and too many untranslated labels)

Workflow
--------
1. For each target language, train language-specific models using distilled
   regional checkpoints (where available).
2. Load the pre-trained multilingual models from ``output/model_mul_*/`` and
   evaluate them on each language's test split — no retraining needed.
3. Produce per-language comparison plots and an overall F1 heatmap.
4. Write a Markdown report with a recommendation table:
   - **Best (unconstrained)** — highest F1 regardless of size
   - **Best for Raspberry Pi** — smallest model within ``RASPI_F1_TOLERANCE``
     of the best F1 (and under ``RASPI_SIZE_MB``)

Outputs (all under ``output/``)
--------------------------------
``model_mono_<lang>_<name>/``        – saved monolingual pipeline
``metrics_mono_<lang>_<name>.md``    – per-model metrics report
``monolingual_comparison.md``        – cross-model table + recommendations
``mono_plots/<lang>_comparison.png`` – F1 bar chart + size vs F1 scatter per language
``mono_plots/overall_heatmap.png``   – languages × models F1 heatmap

Usage::

    python train_monolingual.py
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
from model2vec.train import StaticModelForClassification
from safetensors import safe_open
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

matplotlib.use("Agg")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TRAIN_DIR: str = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR: str = os.path.join(TRAIN_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MONO_PLOTS_DIR: str = os.path.join(OUTPUT_DIR, "mono_plots")
os.makedirs(MONO_PLOTS_DIR, exist_ok=True)

DISTILLED_DIR: str = os.path.join(OUTPUT_DIR, "distilled")
CSV_FULL: str = os.path.join(OUTPUT_DIR, "merged_intents_dataset_full.csv")
COMPARISON_MD: str = os.path.join(OUTPUT_DIR, "monolingual_comparison.md")

# ---------------------------------------------------------------------------
# Training configuration
# ---------------------------------------------------------------------------

RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2
MAX_EPOCHS: int = 30
MIN_SAMPLES_PER_CLASS: int = 10
MAX_SAMPLES_PER_CLASS: int | None = 800

# ---------------------------------------------------------------------------
# Recommendation thresholds
# ---------------------------------------------------------------------------

# Raspberry Pi size budget in megabytes.
RASPI_SIZE_MB: float = 100.0

# The RPi candidate must achieve at least (best_f1 - RASPI_F1_TOLERANCE).
RASPI_F1_TOLERANCE: float = 0.05

# ---------------------------------------------------------------------------
# Languages and model configs
# ---------------------------------------------------------------------------

# Languages with enough data for reliable monolingual classifiers.
# eu / fr / nl / it are omitted: too few examples and many untranslated labels.
TARGET_LANGS: list[str] = ["ca", "da", "de", "es", "gl", "pt"]


def _d(name: str) -> str:
    """Return the full path to a distilled checkpoint by short name."""
    return os.path.join(DISTILLED_DIR, name)


# For each target language: list of distilled checkpoints to train.
# Languages without language-specific distilled models (da, de) get an empty
# list — they are still covered by the multilingual baseline evaluation.
MODELS_BY_LANG: dict[str, list[dict]] = {
    "ca": [
        {"path": _d("distilroberta-base-ca-v2"), "name": "distilroberta-ca"},
        {"path": _d("roberta-base-ca-v2"),        "name": "roberta-base-ca"},
        {"path": _d("roberta-large-ca-v2"),       "name": "roberta-large-ca"},
        {"path": _d("MrBERT-ca"),                 "name": "MrBERT-ca"},
    ],
    "da": [],  # No language-specific distilled models; multilingual baselines only
    "de": [],  # No language-specific distilled models; multilingual baselines only
    "es": [
        {"path": _d("MrBERT-es"),      "name": "MrBERT-es"},
        {"path": _d("MrBERT-legal"),   "name": "MrBERT-legal"},
        {"path": _d("MrBERT-biomed"),  "name": "MrBERT-biomed"},
        {"path": _d("MrBERT-science"), "name": "MrBERT-science"},
    ],
    "gl": [
        {"path": _d("bertinho-gl-small-cased"), "name": "bertinho-gl"},
    ],
    "pt": [
        {"path": _d("bertha-portuguese-small"),      "name": "bertha-pt-small"},
        {"path": _d("bert-base-portuguese-cased"),   "name": "bert-base-pt"},
        {"path": _d("bert-large-portuguese-cased"),  "name": "bert-large-pt"},
    ],
}

# Pre-trained multilingual models are discovered automatically.
MULTILINGUAL_MODEL_DIRS: list[str] = sorted(
    glob.glob(os.path.join(OUTPUT_DIR, "model_mul_*/"))
)

# ---------------------------------------------------------------------------
# Shared dataset helpers
# ---------------------------------------------------------------------------


def load_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["sentence"])
    df = df.drop_duplicates()
    if "label" not in df.columns and {"domain", "intent"}.issubset(df.columns):
        df["label"] = df["domain"] + ":" + df["intent"]
        df.drop(columns=["domain", "intent"], inplace=True)
    if "lang" not in df.columns:
        df["lang"] = "en"
    return df


def balance_dataset(df: pd.DataFrame,
                    min_samples: int = MIN_SAMPLES_PER_CLASS,
                    max_samples: int | None = MAX_SAMPLES_PER_CLASS) -> pd.DataFrame:
    counts = df["label"].value_counts()
    keep = counts[counts >= min_samples].index
    df = df[df["label"].isin(keep)].copy()
    if max_samples is not None:
        df = pd.concat(
            [g.sample(min(len(g), max_samples), random_state=RANDOM_STATE)
             for _, g in df.groupby("label")],
            ignore_index=True,
        )
    return df


def balanced_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    counts = df["label"].value_counts()
    stratifiable = counts[counts >= 2].index
    df = df[df["label"].isin(stratifiable)].copy()
    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def measure_model_size(model_dir: str) -> dict:
    total_bytes = sum(
        os.path.getsize(os.path.join(dp, fname))
        for dp, _, files in os.walk(model_dir)
        for fname in files
    )
    n_params = -1
    for dp, _, files in os.walk(model_dir):
        for fname in files:
            if fname.endswith(".safetensors"):
                try:
                    with safe_open(os.path.join(dp, fname), framework="np") as st:
                        n_params = sum(
                            int(np.prod(st.get_slice(k)[:].shape)) for k in st.keys()
                        )
                except Exception:
                    pass
    return {"size_mb": total_bytes / 1024**2, "n_params": n_params}


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------


def _eval_pipeline(model: StaticModelPipeline,
                   lang_df: pd.DataFrame) -> dict:
    """Evaluate a loaded pipeline on a balanced split of ``lang_df``."""
    df = balance_dataset(lang_df)
    _, test_df = balanced_split(df)
    X_test = test_df["sentence"].tolist()
    y_test = test_df["label"].tolist()

    t0 = time.monotonic()
    y_pred = model.predict(X_test)
    infer_time = time.monotonic() - t0

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "throughput_sps": len(X_test) / infer_time if infer_time > 0 else 0,
    }


def train_monolingual(lang: str, base_model_path: str,
                      model_name: str, lang_df: pd.DataFrame) -> dict | None:
    """Train a monolingual classifier and return a metrics dict.

    If a saved model already exists at the expected output path, it is loaded
    and evaluated without retraining.
    """
    run_id = f"{lang}_{model_name}"
    output_dir = os.path.join(OUTPUT_DIR, f"model_mono_{run_id}")

    if os.path.exists(output_dir):
        logger.info(f"  [{run_id}] Already trained — loading for evaluation.")
        model = StaticModelPipeline.from_pretrained(output_dir)
        metrics = _eval_pipeline(model, lang_df)
        size_info = measure_model_size(output_dir)
        result = {"run_id": run_id, "lang": lang, "model_name": model_name,
                  "model_type": "monolingual", "train_time_s": 0,
                  "output_dir": output_dir, **metrics, **size_info}
        logger.info(f"  [{run_id}]  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}")
        return result

    df = balance_dataset(lang_df)
    if df["label"].nunique() < 2:
        logger.warning(f"  [{run_id}] Too few classes ({df['label'].nunique()}); skipping.")
        return None

    train_df, test_df = balanced_split(df)
    X_train = train_df["sentence"].tolist()
    y_train = train_df["label"].tolist()
    X_test = test_df["sentence"].tolist()
    y_test = test_df["label"].tolist()

    logger.info(f"  [{run_id}] Training  ({len(X_train)} train / {len(X_test)} test)")

    try:
        classifier = StaticModelForClassification.from_pretrained(model_name=base_model_path)
    except Exception as exc:
        logger.error(f"  [{run_id}] Failed to load {base_model_path}: {exc}")
        return None

    t0 = time.monotonic()
    classifier.fit(X_train, y_train, max_epochs=MAX_EPOCHS)
    train_time = time.monotonic() - t0

    t0 = time.monotonic()
    y_pred = classifier.predict(X_test)
    infer_time = time.monotonic() - t0

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    throughput = len(X_test) / infer_time if infer_time > 0 else 0

    classifier.to_pipeline().save_pretrained(output_dir)
    size_info = measure_model_size(output_dir)

    # Write markdown metrics
    md_path = os.path.join(OUTPUT_DIR, f"metrics_mono_{run_id}.md")
    with open(md_path, "w") as fh:
        fh.write(f"# Monolingual Model – {run_id}\n\n")
        fh.write(f"**Language:** `{lang}`  \n**Base model:** `{base_model_path}`\n\n")
        fh.write("| Metric | Value |\n|---|---|\n")
        fh.write(f"| Accuracy | {accuracy:.4f} |\n")
        fh.write(f"| Weighted F1 | {f1:.4f} |\n")
        fh.write(f"| Train time | {train_time:.1f}s |\n")
        fh.write(f"| Throughput | {throughput:.0f} sps |\n")
        fh.write(f"| Disk size | {size_info['size_mb']:.2f} MB |\n")
        n_label = f"{size_info['n_params'] / 1e6:.2f} M" if size_info['n_params'] >= 0 else "N/A"
        fh.write(f"| Parameters | {n_label} |\n")

    logger.info(
        f"  [{run_id}]  acc={accuracy:.4f}  f1={f1:.4f}  "
        f"{size_info['size_mb']:.1f} MB  {throughput:.0f} sps"
    )
    return {
        "run_id": run_id, "lang": lang, "model_name": model_name,
        "model_type": "monolingual", "accuracy": accuracy, "f1": f1,
        "throughput_sps": throughput, "train_time_s": train_time,
        "output_dir": output_dir, **size_info,
    }


def eval_multilingual_on_lang(model_dir: str, lang: str,
                               lang_df: pd.DataFrame) -> dict | None:
    """Load a pre-trained multilingual model and evaluate it on one language."""
    model_name = os.path.basename(model_dir.rstrip("/")).replace("model_mul_", "")
    df = balance_dataset(lang_df)
    if df["label"].nunique() < 2:
        return None

    try:
        model = StaticModelPipeline.from_pretrained(model_dir)
    except Exception as exc:
        logger.error(f"  Failed to load {model_dir}: {exc}")
        return None

    metrics = _eval_pipeline(model, lang_df)
    size_info = measure_model_size(model_dir)
    display_name = f"{model_name} (multilingual)"

    logger.info(
        f"  [{lang}|{model_name}]  acc={metrics['accuracy']:.4f}  f1={metrics['f1']:.4f}  "
        f"{size_info['size_mb']:.1f} MB"
    )
    return {
        "run_id": f"{lang}_{model_name}_mul",
        "lang": lang,
        "model_name": display_name,
        "model_type": "multilingual",
        "train_time_s": 0,
        "output_dir": model_dir,
        **metrics,
        **size_info,
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_lang_comparison(lang: str, rows: list[dict]) -> None:
    """Bar chart + size-vs-F1 scatter for a single language."""
    df = pd.DataFrame(rows).sort_values("f1", ascending=True)
    colors = ["steelblue" if t == "monolingual" else "darkorange"
              for t in df["model_type"]]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(4, len(df) * 0.5 + 1)))

    # F1 bar chart
    bars = axes[0].barh(df["model_name"], df["f1"], color=colors)
    best_f1 = df["f1"].max()
    axes[0].axvline(best_f1 - RASPI_F1_TOLERANCE, color="tomato", linestyle="--",
                    linewidth=1.2,
                    label=f"RPi threshold (best − {RASPI_F1_TOLERANCE:.0%})")
    if RASPI_SIZE_MB < df["size_mb"].max():
        # Mark RPi-eligible bars with a star
        for bar, (_, row) in zip(bars, df.iterrows()):
            if row["size_mb"] <= RASPI_SIZE_MB and row["f1"] >= best_f1 - RASPI_F1_TOLERANCE:
                axes[0].text(bar.get_width() - 0.003,
                             bar.get_y() + bar.get_height() / 2,
                             "★", va="center", ha="right", fontsize=10,
                             color="gold")
    for bar, val in zip(bars, df["f1"]):
        axes[0].text(bar.get_width() + 0.003, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va="center", fontsize=8)
    axes[0].set_xlabel("Weighted F1")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_title(f"[{lang.upper()}] F1 by model  (★ = RPi-eligible)")
    axes[0].legend(fontsize=8)
    # Legend patches for model types
    from matplotlib.patches import Patch
    handles = [Patch(color="steelblue", label="monolingual"),
               Patch(color="darkorange", label="multilingual")]
    axes[0].legend(handles=handles + axes[0].get_legend_handles_labels()[0][:1],
                   fontsize=8)

    # Size vs F1 scatter
    for model_type, color, marker in [("monolingual", "steelblue", "o"),
                                       ("multilingual", "darkorange", "^")]:
        sub = df[df["model_type"] == model_type]
        axes[1].scatter(sub["size_mb"], sub["f1"], label=model_type,
                        color=color, marker=marker, s=80, zorder=3)
    for _, row in df.iterrows():
        axes[1].annotate(row["model_name"],
                         (row["size_mb"], row["f1"]),
                         textcoords="offset points", xytext=(4, 2), fontsize=7)
    if RASPI_SIZE_MB < df["size_mb"].max() * 1.2:
        axes[1].axvline(RASPI_SIZE_MB, color="tomato", linestyle="--",
                        linewidth=1.2, label=f"RPi budget ({RASPI_SIZE_MB:.0f} MB)")
    axes[1].axhline(best_f1 - RASPI_F1_TOLERANCE, color="tomato", linestyle=":",
                    linewidth=1, label=f"RPi F1 threshold")
    axes[1].set_xlabel("Model size (MB)")
    axes[1].set_ylabel("Weighted F1")
    axes[1].set_title(f"[{lang.upper()}] Size vs F1")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(MONO_PLOTS_DIR, f"{lang}_comparison.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_overall_heatmap(all_rows: list[dict]) -> None:
    """Heatmap: languages (rows) × model names (columns) coloured by F1."""
    df = pd.DataFrame(all_rows)
    pivot = df.pivot_table(index="model_name", columns="lang",
                           values="f1", aggfunc="mean")
    # Sort rows: monolingual first (by mean F1), then multilingual
    order = (df.groupby("model_name")
             .agg(mean_f1=("f1", "mean"), model_type=("model_type", "first"))
             .sort_values(["model_type", "mean_f1"], ascending=[True, False])
             .index)
    pivot = pivot.reindex(order)

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.3),
                                    max(5, len(pivot) * 0.5 + 1)))
    sns.heatmap(pivot, annot=True, fmt=".3f", cmap="YlGn", ax=ax,
                linewidths=0.3, vmin=0.5, vmax=1.0,
                cbar_kws={"label": "Weighted F1"})
    ax.set_title("Weighted F1 – all models × languages\n"
                 "(monolingual models at top, multilingual baselines at bottom)",
                 fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=0, labelsize=9)
    ax.tick_params(axis="y", rotation=0, labelsize=8)
    fig.tight_layout()
    out = os.path.join(MONO_PLOTS_DIR, "overall_heatmap.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"  Saved → {out}")


def plot_size_vs_f1_all(all_rows: list[dict]) -> None:
    """Single scatter of all models coloured by language."""
    df = pd.DataFrame(all_rows)
    langs = sorted(df["lang"].unique())
    palette = plt.cm.tab10(np.linspace(0, 1, len(langs)))
    lang_color = dict(zip(langs, palette))

    fig, ax = plt.subplots(figsize=(14, 7))
    for model_type, marker in [("monolingual", "o"), ("multilingual", "^")]:
        sub = df[df["model_type"] == model_type]
        for lang in langs:
            pts = sub[sub["lang"] == lang]
            ax.scatter(pts["size_mb"], pts["f1"],
                       color=lang_color[lang], marker=marker,
                       s=80, zorder=3, label=f"{lang} ({model_type})" if marker == "o" else None)
    for _, row in df.drop_duplicates(subset=["model_name"]).iterrows():
        ax.annotate(row["model_name"], (row["size_mb"], row["f1"]),
                    textcoords="offset points", xytext=(4, 2), fontsize=6)
    ax.axvline(RASPI_SIZE_MB, color="tomato", linestyle="--",
               linewidth=1.2, label=f"RPi budget ({RASPI_SIZE_MB:.0f} MB)")
    ax.set_xlabel("Model size (MB)")
    ax.set_ylabel("Weighted F1")
    ax.set_title("All models: size vs F1  (○ monolingual  △ multilingual)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    out = os.path.join(MONO_PLOTS_DIR, "size_vs_f1_all.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    logger.info(f"  Saved → {out}")


# ---------------------------------------------------------------------------
# Recommendation logic
# ---------------------------------------------------------------------------


def recommend(all_rows: list[dict]) -> pd.DataFrame:
    """For each language, recommend the best unconstrained and best RPi model."""
    df = pd.DataFrame(all_rows)
    recs = []
    for lang, grp in df.groupby("lang"):
        best_f1 = grp["f1"].max()
        threshold = best_f1 - RASPI_F1_TOLERANCE

        best_row = grp.sort_values("f1", ascending=False).iloc[0]

        raspi_cands = grp[(grp["f1"] >= threshold) & (grp["size_mb"] <= RASPI_SIZE_MB)]
        if raspi_cands.empty:
            # Relax size constraint: just pick smallest that meets F1 threshold
            raspi_cands = grp[grp["f1"] >= threshold]
        if raspi_cands.empty:
            raspi_cands = grp
        raspi_row = raspi_cands.sort_values("size_mb").iloc[0]

        recs.append({
            "Language": lang,
            "Best (unconstrained)": best_row["model_name"],
            "Best F1": f"{best_row['f1']:.4f}",
            "Best size (MB)": f"{best_row['size_mb']:.1f}",
            "Best for RPi": raspi_row["model_name"],
            "RPi F1": f"{raspi_row['f1']:.4f}",
            "RPi size (MB)": f"{raspi_row['size_mb']:.1f}",
        })
    return pd.DataFrame(recs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    if not os.path.exists(CSV_FULL):
        raise FileNotFoundError(
            f"Dataset not found: {CSV_FULL}\nRun gather_dataset.py first."
        )

    full_df = load_dataset(CSV_FULL)
    logger.info(f"Loaded {len(full_df):,} rows, {full_df['label'].nunique()} labels, "
                f"{full_df['lang'].nunique()} languages")
    logger.info(f"Found {len(MULTILINGUAL_MODEL_DIRS)} pre-trained multilingual models")

    all_rows: list[dict] = []

    for lang in tqdm(TARGET_LANGS, desc="Languages"):
        logger.info(f"\n{'='*70}")
        logger.info(f"Language: {lang.upper()}")
        logger.info(f"{'='*70}")

        lang_df = full_df[full_df["lang"] == lang].copy()
        logger.info(f"  {len(lang_df):,} rows for [{lang}]")
        if len(lang_df) < 50:
            logger.warning(f"  Too few rows for [{lang}]; skipping.")
            continue

        lang_rows: list[dict] = []

        # --- Train language-specific monolingual models -----------------------
        for model_cfg in MODELS_BY_LANG.get(lang, []):
            if not os.path.exists(model_cfg["path"]):
                logger.warning(
                    f"  Distilled model not found: {model_cfg['path']}; skipping."
                )
                continue
            result = train_monolingual(lang, model_cfg["path"],
                                       model_cfg["name"], lang_df)
            if result:
                lang_rows.append(result)

        # --- Evaluate pre-trained multilingual models on this language --------
        logger.info(f"  Evaluating {len(MULTILINGUAL_MODEL_DIRS)} multilingual models ...")
        for mul_dir in MULTILINGUAL_MODEL_DIRS:
            result = eval_multilingual_on_lang(mul_dir, lang, lang_df)
            if result:
                lang_rows.append(result)

        all_rows.extend(lang_rows)

        if lang_rows:
            plot_lang_comparison(lang, lang_rows)

    if not all_rows:
        logger.warning("No results collected. Check that multilingual models exist "
                       "under output/model_mul_*/ and distilled models under output/distilled/.")
        return

    # --- Overall plots --------------------------------------------------------
    logger.info("\nGenerating overall plots ...")
    plot_overall_heatmap(all_rows)
    plot_size_vs_f1_all(all_rows)

    # --- Recommendation table -------------------------------------------------
    rec_df = recommend(all_rows)
    logger.info("\n" + "=" * 70)
    logger.info("Recommendations")
    logger.info("=" * 70)
    for _, row in rec_df.iterrows():
        logger.info(
            f"  [{row['Language']}]  "
            f"Best: {row['Best (unconstrained)']} (F1={row['Best F1']}, {row['Best size (MB)']} MB)  |  "
            f"RPi: {row['Best for RPi']} (F1={row['RPi F1']}, {row['RPi size (MB)']} MB)"
        )

    # --- Write markdown report ------------------------------------------------
    all_df = pd.DataFrame(all_rows).sort_values(
        ["lang", "f1"], ascending=[True, False]
    )

    with open(COMPARISON_MD, "w") as fh:
        fh.write("# Monolingual vs Multilingual – Intent Classification\n\n")
        fh.write(f"**Languages:** {', '.join(TARGET_LANGS)}\n\n")
        fh.write(
            f"**Dataset balance:** min_samples={MIN_SAMPLES_PER_CLASS}, "
            f"max_samples={MAX_SAMPLES_PER_CLASS}\n\n"
        )
        fh.write(
            f"**Raspberry Pi budget:** ≤ {RASPI_SIZE_MB} MB, "
            f"within {RASPI_F1_TOLERANCE:.0%} of best F1\n\n"
        )

        fh.write("## Recommendations\n\n")
        fh.write(rec_df.to_markdown(index=False))
        fh.write("\n\n")

        fh.write("## Full results by language\n\n")
        for lang in TARGET_LANGS:
            lang_rows_df = all_df[all_df["lang"] == lang]
            if lang_rows_df.empty:
                continue
            fh.write(f"### [{lang.upper()}]\n\n")
            display = lang_rows_df[
                ["model_name", "model_type", "accuracy", "f1",
                 "size_mb", "n_params", "throughput_sps"]
            ].copy()
            display["n_params"] = display["n_params"].apply(
                lambda x: f"{x / 1e6:.2f}M" if x >= 0 else "N/A"
            )
            display.columns = [
                "Model", "Type", "Accuracy", "F1",
                "Size (MB)", "Params", "Throughput (sps)"
            ]
            fh.write(display.to_markdown(index=False))
            fh.write("\n\n")

    logger.info(f"\nComparison report → {COMPARISON_MD}")
    logger.info(f"Plots → {MONO_PLOTS_DIR}/")


if __name__ == "__main__":
    main()
