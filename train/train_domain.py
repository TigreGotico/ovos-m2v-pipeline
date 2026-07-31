"""Train a Domain (parallel-argmax) intent classifier bundle.

Consumes the same ``merged_intents_dataset*.csv`` produced by
``gather_dataset.py`` / ``gather_dataset_en.py`` (columns ``lang``, ``label``,
``sentence``). Splits each ``label`` on the first ``.`` (or ``:``) to derive a
``<domain>.<intent>`` taxonomy, then trains one intent classifier per domain
(no top-level domain router).

The trained bundle is written via :meth:`DomainIntentClassifier.save` and can
be loaded by ``Model2VecDomainIntentPipeline`` at runtime.

Usage
-----

    python train/train_domain.py \
        --csv train/merged_intents_dataset_en.csv \
        --output m2v_domain_intents_potion-base-8M \
        --base-model minishlab/potion-base-8M
"""
from __future__ import annotations

import argparse
import logging
import os
from collections import Counter

import numpy as np
import pandas as pd
from model2vec import StaticModel
from sklearn.metrics import accuracy_score, classification_report

from ovos_m2v_pipeline.domain_classifier import DomainIntentClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_domain")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a domain (parallel-argmax) Model2Vec intent classifier.")
    p.add_argument("--csv", default="merged_intents_dataset_en.csv",
                   help="Path to the merged intents CSV (columns lang,label,sentence).")
    p.add_argument("--output", default="m2v_domain_intents",
                   help="Output bundle directory.")
    p.add_argument("--base-model", default="minishlab/potion-base-8M",
                   help="Model2Vec encoder to use for embeddings.")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="LogisticRegression max_iter.")
    p.add_argument("--test-size", type=float, default=0.1,
                   help="Held-out fraction for evaluation (0 disables eval).")
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    logger.info(f"Loading dataset {args.csv}")
    df = pd.read_csv(args.csv).dropna(subset=["sentence", "label"])
    sentences = df["sentence"].astype(str).tolist()
    labels = df["label"].astype(str).tolist()

    counts = Counter(labels)
    keep = [(s, l) for s, l in zip(sentences, labels) if counts[l] >= 2]
    sentences = [s for s, _ in keep]
    labels = [l for _, l in keep]
    logger.info(f"Kept {len(sentences)} rows across {len(set(labels))} intent labels")

    logger.info(f"Loading encoder {args.base_model}")
    model = StaticModel.from_pretrained(args.base_model)

    logger.info("Embedding training corpus")
    embeddings = np.asarray(model.encode(sentences))

    eval_X = eval_y = None
    if args.test_size and args.test_size > 0:
        from sklearn.model_selection import train_test_split

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                embeddings, labels, test_size=args.test_size,
                random_state=args.random_state, stratify=labels,
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                embeddings, labels, test_size=args.test_size,
                random_state=args.random_state,
            )
        embeddings, labels = X_tr, y_tr
        eval_X, eval_y = X_te, y_te

    logger.info("Training domain (parallel-argmax) classifier")
    clf = DomainIntentClassifier.train(
        embeddings, labels,
        max_iter=args.max_iter,
        random_state=args.random_state,
    )
    logger.info(
        f"Trained {len(clf.intent_classifiers)} per-domain intent classifiers "
        f"({len(clf)} total intent labels)."
    )

    if eval_X is not None:
        preds = [clf.predict(x)[0] for x in eval_X]
        scored = [(p, y) for p, y in zip(preds, eval_y) if p is not None]
        if scored:
            acc = accuracy_score([y for _, y in scored], [p for p, _ in scored])
            logger.info(f"Eval: acc={acc:.1%}")
            logger.info("\n" + classification_report(
                [y for _, y in scored], [p for p, _ in scored], zero_division=0,
            ))

    os.makedirs(args.output, exist_ok=True)
    clf.save(args.output)
    logger.info(f"Saved domain bundle to {args.output}/")


if __name__ == "__main__":
    main()
