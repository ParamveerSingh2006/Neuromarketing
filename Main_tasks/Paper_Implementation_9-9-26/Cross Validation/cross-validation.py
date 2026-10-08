#!/usr/bin/env python3
"""
03_NeuMa_CV_Evaluation.py

Stage 3 — subject-specific LOEO classification + evaluation.

For each subject:
  - Leave one product/epoch out
  - Wilcoxon rank-sum feature selection on TRAIN only, alpha=0.05
  - train-only median imputation
  - optional train-only StandardScaler (default ON)
  - Linear SVM (LinearSVC), C=1.0
  - predict held-out product
  - aggregate OOF predictions
  - compute subject-level and pooled metrics

Primary paper metrics:
  - Cohen's kappa
  - F1

Also reported:
  - accuracy
  - precision
  - recall

The graph/GFT basis is inherited from Stage 2 and is not rebuilt per fold.
That is deliberately retained for closer replication of the paper's
personalized graph description, but means the full pipeline is not strictly
test-isolated. Feature selection, imputation, scaling and SVM fitting are
train-only.
"""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import ranksums
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DEFAULT_ALPHA = 0.05
DEFAULT_C = 1.0
DEFAULT_MAX_ITER = 10000


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def wilcoxon_select(
    X: np.ndarray,
    y: np.ndarray,
    alpha: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Two-sided Wilcoxon rank-sum test per feature.

    The paper says Wilcoxon feature selection but does not provide additional
    multiple-testing correction details; therefore alpha=.05 is used directly.
    """
    classes = np.unique(y)
    if len(classes) != 2:
        raise ValueError(
            f"Training data must contain both classes; got {classes.tolist()}"
        )

    p_values = np.ones(X.shape[1], dtype=float)
    statistics = np.zeros(X.shape[1], dtype=float)

    x0 = X[y == classes[0]]
    x1 = X[y == classes[1]]

    for j in range(X.shape[1]):
        try:
            result = ranksums(x0[:, j], x1[:, j])
            statistics[j] = float(result.statistic)
            p_values[j] = (
                float(result.pvalue)
                if np.isfinite(result.pvalue) else 1.0
            )
        except Exception:
            p_values[j] = 1.0

    selected = np.flatnonzero(p_values < alpha)
    fallback = False

    if len(selected) == 0:
        selected = np.asarray([int(np.argmin(p_values))], dtype=int)
        fallback = True

    return selected, p_values, statistics, fallback


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }


def run_subject(
    subject: Dict[str, Any],
    feature_names: List[str],
    alpha: float,
    C: float,
    max_iter: int,
    scaling: bool,
) -> Dict[str, Any]:
    rows = subject["rows"]
    X = np.vstack([
        np.asarray(row["features"], dtype=float)
        for row in rows
    ])
    y = np.asarray([int(row["label"]) for row in rows], dtype=int)

    predictions = []
    folds = []

    for test_idx in range(len(rows)):
        train_idx = np.asarray(
            [i for i in range(len(rows)) if i != test_idx],
            dtype=int
        )

        X_train = X[train_idx]
        y_train = y[train_idx]
        X_test = X[[test_idx]]

        fold = {
            "fold": test_idx,
            "test_page": int(rows[test_idx]["page"]),
            "test_product": int(rows[test_idx]["product"]),
            "valid_training_split": False,
        }

        if len(np.unique(y_train)) != 2:
            fold["reason_skipped"] = "training fold contains one class"
            folds.append(fold)
            continue

        # ---------------- TRAIN-ONLY OPERATIONS ----------------
        imputer = SimpleImputer(
            strategy="median",
            keep_empty_features=True
        )
        Xtr = imputer.fit_transform(X_train)
        Xte = imputer.transform(X_test)

        selected, p_values, statistics, fallback = wilcoxon_select(
            Xtr, y_train, alpha
        )

        Xtr = Xtr[:, selected]
        Xte = Xte[:, selected]

        if scaling:
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(Xtr)
            Xte = scaler.transform(Xte)

        clf = LinearSVC(
            C=C,
            class_weight=None,
            max_iter=max_iter
        )
        clf.fit(Xtr, y_train)

        pred = int(clf.predict(Xte)[0])
        decision = float(clf.decision_function(Xte)[0])
        # -------------------------------------------------------

        fold.update({
            "valid_training_split": True,
            "n_train": int(len(train_idx)),
            "n_train_class_0": int(np.sum(y_train == 0)),
            "n_train_class_1": int(np.sum(y_train == 1)),
            "n_features_total": int(X.shape[1]),
            "n_features_selected": int(len(selected)),
            "selected_feature_indices": selected.tolist(),
            "selected_feature_names": [
                feature_names[i] for i in selected
            ],
            "selected_feature_pvalues": p_values[selected].tolist(),
            "selected_feature_statistics": statistics[selected].tolist(),
            "selection_fallback_used": bool(fallback),
            "all_feature_pvalues": p_values.tolist(),
            "all_feature_statistics": statistics.tolist(),
            "scaling_train_only": bool(scaling),
            "svm_C": float(C),
            "y_true": int(y[test_idx]),
            "prediction": pred,
            "decision_function": decision,
        })
        folds.append(fold)

        predictions.append({
            "subject_id": subject["subject_id"],
            "fold": test_idx,
            "test_page": int(rows[test_idx]["page"]),
            "test_product": int(rows[test_idx]["product"]),
            "y_true": int(y[test_idx]),
            "y_pred": pred,
            "decision_function": decision,
        })

    if predictions:
        yt = np.asarray([r["y_true"] for r in predictions])
        yp = np.asarray([r["y_pred"] for r in predictions])
        subject_metrics = metrics(yt, yp)
        cm = confusion_matrix(yt, yp, labels=[0, 1]).tolist()
    else:
        subject_metrics = {
            key: np.nan
            for key in ["accuracy", "precision", "recall", "f1", "kappa"]
        }
        cm = [[0, 0], [0, 0]]

    return {
        "subject_id": subject["subject_id"],
        "n_products": len(rows),
        "n_valid_predictions": len(predictions),
        "metrics": subject_metrics,
        "confusion_matrix_labels_0_1": cm,
        "predictions": predictions,
        "folds": folds,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--C", type=float, default=DEFAULT_C)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        help="Disable train-fold-only StandardScaler."
    )
    args = parser.parse_args()

    data = load_pickle(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    all_predictions = []

    for subject in data["subjects"]:
        print(f"Running LOEO for {subject['subject_id']}")
        result = run_subject(
            subject=subject,
            feature_names=data["feature_names"],
            alpha=args.alpha,
            C=args.C,
            max_iter=args.max_iter,
            scaling=not args.no_scaling,
        )
        results.append(result)
        all_predictions.extend(result["predictions"])

    metric_names = [
        "accuracy", "precision", "recall", "f1", "kappa"
    ]

    macro = {}
    for metric_name in metric_names:
        values = np.asarray(
            [r["metrics"][metric_name] for r in results],
            dtype=float
        )
        finite = np.isfinite(values)
        macro[metric_name] = {
            "mean": float(np.mean(values[finite]))
            if finite.any() else np.nan,
            "std": float(np.std(values[finite], ddof=1))
            if np.sum(finite) > 1 else np.nan,
            "median": float(np.median(values[finite]))
            if finite.any() else np.nan,
        }

    if all_predictions:
        yt = np.asarray([r["y_true"] for r in all_predictions])
        yp = np.asarray([r["y_pred"] for r in all_predictions])
        pooled = metrics(yt, yp)
    else:
        pooled = {m: np.nan for m in metric_names}

    artifact = {
        "stage": 3,
        "subjects": results,
        "all_predictions": all_predictions,
        "macro_subject_metrics": macro,
        "pooled_metrics": pooled,
        "configuration": {
            "cv": "subject-specific Leave-One-Epoch-Out",
            "wilcoxon_alpha": args.alpha,
            "svm": "LinearSVC",
            "svm_C": args.C,
            "class_weight": None,
            "max_iter": args.max_iter,
            "scaling": (
                "train-fold-only StandardScaler"
                if not args.no_scaling else "disabled"
            ),
            "primary_metrics": ["kappa", "f1"],
            "supplementary_metrics": [
                "accuracy", "precision", "recall"
            ],
            "graph_scope": (
                "subject-level all retained EEG segments, inherited from Stage 2"
            ),
            "strict_end_to_end_test_isolation": False,
            "blink_rate": "omitted",
        },
    }

    pkl = args.output_dir / "stage3_cv_evaluation.pkl"
    save_pickle(pkl, artifact)

    pred_csv = args.output_dir / "stage3_oof_predictions.csv"
    with pred_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id", "fold", "test_page", "test_product",
                "y_true", "y_pred", "decision_function"
            ]
        )
        writer.writeheader()
        writer.writerows(all_predictions)

    subject_csv = args.output_dir / "stage3_subject_metrics.csv"
    with subject_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subject_id", "n_products",
                        "n_valid_predictions"] + metric_names
        )
        writer.writeheader()
        for result in results:
            writer.writerow({
                "subject_id": result["subject_id"],
                "n_products": result["n_products"],
                "n_valid_predictions": result["n_valid_predictions"],
                **result["metrics"],
            })

    macro_csv = args.output_dir / "stage3_macro_metrics.csv"
    with macro_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "mean", "std", "median"])
        for metric_name in metric_names:
            writer.writerow([
                metric_name,
                macro[metric_name]["mean"],
                macro[metric_name]["std"],
                macro[metric_name]["median"],
            ])

    selection_csv = args.output_dir / "stage3_feature_selection.csv"
    with selection_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subject_id", "fold", "test_page", "test_product",
                "feature_index", "feature", "p_value",
                "wilcoxon_statistic", "selected", "fallback_used"
            ]
        )
        writer.writeheader()

        for result in results:
            for fold in result["folds"]:
                if not fold.get("valid_training_split"):
                    continue
                selected = set(
                    fold["selected_feature_indices"]
                )
                for j, (p, stat) in enumerate(zip(
                    fold["all_feature_pvalues"],
                    fold["all_feature_statistics"]
                )):
                    writer.writerow({
                        "subject_id": result["subject_id"],
                        "fold": fold["fold"],
                        "test_page": fold["test_page"],
                        "test_product": fold["test_product"],
                        "feature_index": j,
                        "feature": data["feature_names"][j],
                        "p_value": p,
                        "wilcoxon_statistic": stat,
                        "selected": j in selected,
                        "fallback_used": fold["selection_fallback_used"],
                    })

    print(f"\nWrote {pkl}")
    print(f"Wrote {pred_csv}")
    print(f"Wrote {subject_csv}")
    print(f"Wrote {macro_csv}")
    print(f"Wrote {selection_csv}")


if __name__ == "__main__":
    main()
