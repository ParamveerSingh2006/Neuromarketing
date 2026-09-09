#!/usr/bin/env python3
"""
Stage 4 — Evaluation of Stage-3 out-of-fold predictions.

Input:
    stage3_cv_results.pkl

Outputs:
    stage4_metrics.pkl
    stage4_subject_metrics.csv
    stage4_overall_metrics.csv
    stage4_confusion_matrices.csv

Metrics:
    Accuracy, Precision, Recall, F1, Cohen's Kappa.

This stage performs NO model fitting or feature processing. It evaluates
only the OOF predictions produced by Stage 3.

Subject-level macro summaries give every subject equal weight.
Pooled summaries concatenate all OOF predictions and therefore weight
subjects by their number of epochs.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


METRICS = ["accuracy", "precision", "recall", "f1", "kappa"]


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def evaluate_subject(subject_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    rows = sorted(rows, key=lambda r: int(r["test_row_index"]))
    y_true = np.asarray([int(r["y_true"]) for r in rows])
    y_pred = np.asarray([int(r["y_pred"]) for r in rows])

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    metrics = calculate_metrics(y_true, y_pred)

    return {
        "subject_id": subject_id,
        "n_oof_predictions": len(rows),
        "n_true_nobuy": int(np.sum(y_true == 0)),
        "n_true_buy": int(np.sum(y_true == 1)),
        "n_pred_nobuy": int(np.sum(y_pred == 0)),
        "n_pred_buy": int(np.sum(y_pred == 1)),
        **metrics,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Stage-3 OOF predictions."
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage3 = load_pickle(args.input)

    if stage3.get("stage") != "3_subject_specific_loocv":
        raise ValueError("Input is not the expected Stage-3 artifact.")

    predictions = stage3.get("all_predictions", [])
    if not predictions:
        raise ValueError("No OOF predictions found in Stage-3 artifact.")

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in predictions:
        grouped.setdefault(str(row["subject_id"]), []).append(row)

    subject_metrics = []
    all_true: list[int] = []
    all_pred: list[int] = []

    for subject_id in sorted(grouped):
        result = evaluate_subject(subject_id, grouped[subject_id])
        subject_metrics.append(result)
        all_true.extend(int(r["y_true"]) for r in grouped[subject_id])
        all_pred.extend(int(r["y_pred"]) for r in grouped[subject_id])

    y_true = np.asarray(all_true)
    y_pred = np.asarray(all_pred)

    pooled_metrics = calculate_metrics(y_true, y_pred)
    pooled_cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    macro_mean = {
        m: float(np.mean([r[m] for r in subject_metrics]))
        for m in METRICS
    }
    macro_std = {
        m: float(np.std([r[m] for r in subject_metrics], ddof=1))
        if len(subject_metrics) > 1 else 0.0
        for m in METRICS
    }
    macro_median = {
        m: float(np.median([r[m] for r in subject_metrics]))
        for m in METRICS
    }

    result = {
        "stage": "4_oof_evaluation",
        "input_stage": stage3.get("stage"),
        "cv_config": stage3.get("config", {}),
        "n_subjects": len(subject_metrics),
        "n_total_oof_predictions": len(predictions),
        "subject_metrics": subject_metrics,
        "macro_mean_subject_metrics": macro_mean,
        "macro_std_subject_metrics": macro_std,
        "macro_median_subject_metrics": macro_median,
        "pooled_metrics": pooled_metrics,
        "pooled_confusion_matrix_labels": ["NoBuy", "Buy"],
        "pooled_confusion_matrix": pooled_cm.tolist(),
    }

    pkl = args.output_dir / "stage4_metrics.pkl"
    subjects_csv = args.output_dir / "stage4_subject_metrics.csv"
    overall_csv = args.output_dir / "stage4_overall_metrics.csv"
    cm_csv = args.output_dir / "stage4_confusion_matrices.csv"

    save_pickle(pkl, result)

    fields = [
        "subject_id", "n_oof_predictions", "n_true_nobuy", "n_true_buy",
        "n_pred_nobuy", "n_pred_buy", "accuracy", "precision", "recall",
        "f1", "kappa", "tn", "fp", "fn", "tp"
    ]
    with subjects_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subject_metrics)

    with overall_csv.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "metric", "macro_mean_subject", "macro_std_subject",
            "macro_median_subject", "pooled"
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in METRICS:
            writer.writerow({
                "metric": m,
                "macro_mean_subject": macro_mean[m],
                "macro_std_subject": macro_std[m],
                "macro_median_subject": macro_median[m],
                "pooled": pooled_metrics[m],
            })

    with cm_csv.open("w", newline="", encoding="utf-8") as f:
        fields = ["subject_id", "tn", "fp", "fn", "tp"]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in subject_metrics:
            writer.writerow({k: row[k] for k in fields})
        writer.writerow({
            "subject_id": "POOLED",
            "tn": int(pooled_cm[0, 0]),
            "fp": int(pooled_cm[0, 1]),
            "fn": int(pooled_cm[1, 0]),
            "tp": int(pooled_cm[1, 1]),
        })

    print("Stage 4 complete.")
    print(f"Metrics object: {pkl}")
    print(f"Subject metrics: {subjects_csv}")
    print(f"Overall metrics: {overall_csv}")
    print(f"Confusion matrices: {cm_csv}")
    print("\nMacro mean across subjects:")
    for m in METRICS:
        print(f"  {m:9s}: {macro_mean[m]:.6f} (SD={macro_std[m]:.6f})")
    print("\nPooled OOF metrics:")
    for m in METRICS:
        print(f"  {m:9s}: {pooled_metrics[m]:.6f}")


if __name__ == "__main__":
    main()
