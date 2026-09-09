#!/usr/bin/env python3
"""
Stage 3 — Subject-specific Leave-One-Epoch-Out CV
for the Kalaganis et al. (2025) GFT + eye-tracking pipeline.

Consumes:
    stage2_features.pkl

Produces:
    stage3_cv_results.pkl
    stage3_oof_predictions.csv
    stage3_feature_selection.csv
    stage3_summary.csv

Paper-supported structure:
    1. Each subject is evaluated independently.
    2. Leave-One-Epoch-Out Cross Validation is used.
    3. Wilcoxon-statistic feature selection is performed on TRAIN only.
    4. A Linear SVM is trained on the selected TRAIN features.
    5. The held-out epoch is predicted only after all TRAIN operations.

Important reproducibility notes:
    - The paper does not disclose the exact Wilcoxon threshold, exact
      normalization formula, or Linear SVM C value.
    - This implementation therefore exposes those choices as arguments.
    - Default feature selection uses the Wilcoxon rank-sum p-value with
      alpha=0.05 and, if no feature passes the threshold, retains the
      single lowest-p feature so that the fold remains executable. Such
      fallback folds are explicitly recorded.
    - Scaling is TRAIN-FOLD ONLY and defaults to standardization because
      Linear SVM is scale-sensitive. It can be disabled with --no-scaling.
    - The supplied Stage 2 artifact constructs the EEG graph at subject
      level using all retained EEG segments before CV. Stage 3 does not
      reconstruct that graph. This is an important methodological caveat:
      the graph is label-independent but is not strictly test-isolated.
      It is recorded in the output metadata and must not be described as
      completely leakage-free until Stage 2 graph construction is revised.
    - Blink rate is absent from Stage 2 because the documented preprocessed
      ET structure does not expose an explicit blink-event channel.

No classification metrics are calculated here. Stage 4 should calculate
metrics from the complete out-of-fold predictions.
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
from sklearn.pipeline import Pipeline
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


def wilcoxon_rank_sum_feature_selection(
    X_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    min_features: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Perform feature-wise Wilcoxon rank-sum tests on TRAIN only.

    For this binary classification problem, Buy and NoBuy are independent
    groups of epochs within the training fold. scipy.stats.ranksums is the
    Wilcoxon rank-sum test.

    Returns:
        selected_indices
        p_values
        statistics
    """
    X_train = np.asarray(X_train, dtype=float)
    y_train = np.asarray(y_train)

    classes = np.unique(y_train)
    if len(classes) != 2:
        raise ValueError(
            "Wilcoxon feature selection requires both classes in the "
            f"training fold; found {classes.tolist()}."
        )

    c0, c1 = classes[0], classes[1]
    group0 = y_train == c0
    group1 = y_train == c1

    p_values = np.ones(X_train.shape[1], dtype=float)
    statistics = np.zeros(X_train.shape[1], dtype=float)

    for j in range(X_train.shape[1]):
        x0 = X_train[group0, j]
        x1 = X_train[group1, j]

        # A constant feature cannot discriminate through a rank difference.
        if np.allclose(x0, x0[0]) and np.allclose(x1, x1[0]):
            p_values[j] = 1.0
            statistics[j] = 0.0
            continue

        result = ranksums(x0, x1)
        statistics[j] = float(result.statistic)
        p_values[j] = float(result.pvalue)

    selected = np.flatnonzero(p_values < alpha)
    fallback_used = False

    if selected.size < min_features:
        order = np.argsort(p_values, kind="stable")
        selected = order[:min_features]
        fallback_used = True

    return (
        selected.astype(int),
        p_values,
        statistics,
        fallback_used,
    )


def extract_subject_matrix(subject: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    rows = subject.get("features", [])
    if not rows:
        raise ValueError(f"{subject.get('subject_id')}: no feature rows.")

    X = []
    y = []
    metadata = []

    for row in rows:
        vector = np.asarray(row["feature_vector"], dtype=float).ravel()
        if vector.size == 0:
            raise ValueError(
                f"{subject['subject_id']} page={row.get('page')} "
                f"product={row.get('product')}: empty feature vector."
            )

        X.append(vector)
        y.append(int(row["label"]))
        metadata.append(
            {
                "page": int(row["page"]),
                "product": int(row["product"]),
                "label": int(row["label"]),
                "label_name": row.get("label_name"),
            }
        )

    lengths = {len(v) for v in X}
    if len(lengths) != 1:
        raise ValueError(
            f"{subject['subject_id']}: inconsistent feature-vector lengths: "
            f"{sorted(lengths)}"
        )

    return np.vstack(X), np.asarray(y, dtype=int), metadata


def make_feature_names(
    n_features: int,
    stage2_config: Dict[str, Any],
) -> List[str]:
    """
    Infer names from the Stage 2 structure.

    The EEG graph-mode count is taken from the actual feature dimension;
    it is NOT hard-coded to 19 because Stage 2 currently does not
    explicitly store channel/reference selection in the feature artifact.
    """
    et_beh = stage2_config.get("et_behavioral", [])
    et_mark = stage2_config.get("et_marketing", [])

    # Stage 2 currently concatenates:
    #   EEG graph modes + 3 Hjorth + 4 marketing features.
    n_et = len(et_beh) + len(et_mark)
    n_gft = n_features - n_et

    if n_gft < 1:
        raise ValueError(
            f"Could not infer EEG feature count from total={n_features} "
            f"and ET feature count={n_et}."
        )

    names = [f"gft_{i+1:02d}" for i in range(n_gft)]
    names.extend(
        [
            "hjorth_activity",
            "hjorth_mobility",
            "hjorth_complexity",
        ]
    )
    names.extend(
        [
            "fixation_count",
            "mean_fixation_duration",
            "time_to_first_fixation",
            "dwell_time",
        ]
    )

    if len(names) != n_features:
        raise ValueError(
            f"Feature-name count {len(names)} does not match "
            f"feature-vector count {n_features}."
        )

    return names


def run_subject_loocv(
    subject: Dict[str, Any],
    alpha: float,
    C: float,
    max_iter: int,
    scaling: bool,
    min_features: int,
) -> Dict[str, Any]:
    subject_id = subject["subject_id"]
    X, y, row_meta = extract_subject_matrix(subject)

    n_epochs, n_features = X.shape

    if n_epochs < 3:
        raise ValueError(
            f"{subject_id}: at least 3 epochs are required for a useful "
            "binary LOOCV experiment."
        )

    if np.unique(y).size != 2:
        raise ValueError(
            f"{subject_id}: only one class is present; LOOCV classification "
            "cannot be performed."
        )

    predictions = []
    fold_records = []

    for test_idx in range(n_epochs):
        train_mask = np.ones(n_epochs, dtype=bool)
        train_mask[test_idx] = False

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_idx : test_idx + 1]
        y_test = y[test_idx]

        if np.unique(y_train).size != 2:
            raise ValueError(
                f"{subject_id}, held-out epoch {test_idx}: training fold "
                "does not contain both classes."
            )

        # -----------------------------------------------------------
        # TRAIN-ONLY imputation
        # -----------------------------------------------------------
        imputer = SimpleImputer(strategy="median", keep_empty_features=True)
        X_train_imp = imputer.fit_transform(X_train)
        X_test_imp = imputer.transform(X_test)

        # -----------------------------------------------------------
        # TRAIN-ONLY Wilcoxon feature selection
        # -----------------------------------------------------------
        (
            selected_idx,
            p_values,
            statistics,
            fallback_used,
        ) = wilcoxon_rank_sum_feature_selection(
            X_train=X_train_imp,
            y_train=y_train,
            alpha=alpha,
            min_features=min_features,
        )

        X_train_sel = X_train_imp[:, selected_idx]
        X_test_sel = X_test_imp[:, selected_idx]

        # -----------------------------------------------------------
        # TRAIN-ONLY scaling
        # -----------------------------------------------------------
        if scaling:
            scaler = StandardScaler()
            X_train_model = scaler.fit_transform(X_train_sel)
            X_test_model = scaler.transform(X_test_sel)
        else:
            scaler = None
            X_train_model = X_train_sel
            X_test_model = X_test_sel

        # -----------------------------------------------------------
        # Linear SVM
        # -----------------------------------------------------------
        model = LinearSVC(
            C=C,
            class_weight=None,
            max_iter=max_iter,
        )
        model.fit(X_train_model, y_train)

        pred = int(model.predict(X_test_model)[0])
        decision = float(model.decision_function(X_test_model)[0])

        predictions.append(
            {
                "subject_id": subject_id,
                "fold": test_idx + 1,
                "test_row_index": test_idx,
                "page": row_meta[test_idx]["page"],
                "product": row_meta[test_idx]["product"],
                "y_true": int(y_test),
                "y_pred": pred,
                "decision_function": decision,
            }
        )

        fold_records.append(
            {
                "subject_id": subject_id,
                "fold": test_idx + 1,
                "test_row_index": test_idx,
                "test_page": row_meta[test_idx]["page"],
                "test_product": row_meta[test_idx]["product"],
                "test_label": int(y_test),
                "n_train": int(len(y_train)),
                "n_train_buy": int(np.sum(y_train == 1)),
                "n_train_nobuy": int(np.sum(y_train == 0)),
                "n_features_total": int(n_features),
                "n_features_selected": int(len(selected_idx)),
                "selected_feature_indices": selected_idx.tolist(),
                "selected_feature_pvalues": p_values[selected_idx].tolist(),
                "selected_feature_statistics": statistics[selected_idx].tolist(),
                "all_feature_pvalues": p_values.tolist(),
                "all_feature_statistics": statistics.tolist(),
                "selection_alpha": float(alpha),
                "selection_fallback_used": bool(fallback_used),
                "scaling": bool(scaling),
                "svm_C": float(C),
                "prediction": pred,
                "decision_function": decision,
            }
        )

    return {
        "subject_id": subject_id,
        "n_epochs": int(n_epochs),
        "n_features": int(n_features),
        "predictions": predictions,
        "folds": fold_records,
    }


def write_prediction_csv(path: Path, predictions: List[Dict[str, Any]]) -> None:
    fields = [
        "subject_id",
        "fold",
        "test_row_index",
        "page",
        "product",
        "y_true",
        "y_pred",
        "decision_function",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(predictions)


def write_selection_csv(path: Path, fold_records: List[Dict[str, Any]], feature_names: List[str]) -> None:
    fields = [
        "subject_id",
        "fold",
        "test_page",
        "test_product",
        "feature",
        "feature_index",
        "p_value",
        "wilcoxon_statistic",
        "selected",
        "selection_fallback_used",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for fold in fold_records:
            selected = set(fold["selected_feature_indices"])
            selected_p = {
                idx: p
                for idx, p in zip(
                    fold["selected_feature_indices"],
                    fold["selected_feature_pvalues"],
                )
            }
            selected_stat = {
                idx: s
                for idx, s in zip(
                    fold["selected_feature_indices"],
                    fold["selected_feature_statistics"],
                )
            }

            for idx, (p_value, stat) in enumerate(
                zip(
                    fold["all_feature_pvalues"],
                    fold["all_feature_statistics"],
                )
            ):
                writer.writerow(
                    {
                        "subject_id": fold["subject_id"],
                        "fold": fold["fold"],
                        "test_page": fold["test_page"],
                        "test_product": fold["test_product"],
                        "feature": feature_names[idx],
                        "feature_index": idx,
                        "p_value": p_value,
                        "wilcoxon_statistic": stat,
                        "selected": idx in selected,
                        "selection_fallback_used": fold["selection_fallback_used"],
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Stage 3: subject-specific Leave-One-Epoch-Out CV with "
            "TRAIN-only Wilcoxon feature selection and Linear SVM."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to stage2_features.pkl",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for Stage 3 outputs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Wilcoxon rank-sum selection threshold. Default: 0.05.",
    )
    parser.add_argument(
        "--min-features",
        type=int,
        default=1,
        help=(
            "Minimum number of features retained per fold. If fewer than "
            "this pass alpha, the lowest-p features are retained and the "
            "fallback is explicitly recorded."
        ),
    )
    parser.add_argument(
        "--C",
        type=float,
        default=DEFAULT_C,
        help="Linear SVM C. The paper does not disclose this value.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=DEFAULT_MAX_ITER,
        help="Maximum LinearSVC iterations.",
    )
    parser.add_argument(
        "--no-scaling",
        action="store_true",
        help=(
            "Disable TRAIN-fold-only standardization. By default scaling "
            "is applied because Linear SVM is scale-sensitive."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    if not (0 < args.alpha <= 1):
        raise ValueError("--alpha must be in (0, 1].")

    if args.min_features < 1:
        raise ValueError("--min-features must be >= 1.")

    if args.C <= 0:
        raise ValueError("--C must be > 0.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_pickle(args.input)

    if dataset.get("stage") != "2_signal_processing_feature_construction":
        raise ValueError(
            "Input does not appear to be the expected Stage 2 feature artifact."
        )

    subjects = dataset.get("subjects", [])
    if not subjects:
        raise ValueError("Stage 2 artifact contains no subjects.")

    first_X, _, _ = extract_subject_matrix(subjects[0])
    feature_names = make_feature_names(
        first_X.shape[1],
        dataset.get("config", {}),
    )

    subject_results = []
    all_predictions = []
    all_folds = []
    subject_summary = []

    for idx, subject in enumerate(subjects, start=1):
        print(
            f"[{idx}/{len(subjects)}] LOOCV for {subject['subject_id']}..."
        )

        result = run_subject_loocv(
            subject=subject,
            alpha=args.alpha,
            C=args.C,
            max_iter=args.max_iter,
            scaling=not args.no_scaling,
            min_features=args.min_features,
        )

        subject_results.append(result)
        all_predictions.extend(result["predictions"])
        all_folds.extend(result["folds"])

        subject_summary.append(
            {
                "subject_id": result["subject_id"],
                "n_epochs": result["n_epochs"],
                "n_features": result["n_features"],
                "n_predictions": len(result["predictions"]),
                "n_fallback_folds": int(
                    sum(
                        f["selection_fallback_used"]
                        for f in result["folds"]
                    )
                ),
            }
        )

    output = {
        "stage": "3_subject_specific_loocv",
        "paper": dataset.get("paper"),
        "config": {
            "cv": "Leave-One-Epoch-Out Cross Validation",
            "cv_scope": "independent within each subject",
            "feature_selection": "Wilcoxon rank-sum on training fold only",
            "wilcoxon_alpha": args.alpha,
            "minimum_features": args.min_features,
            "feature_selection_fallback": (
                "lowest-p features when fewer than minimum pass alpha"
            ),
            "normalization": (
                "training-fold-only StandardScaler"
                if not args.no_scaling
                else "disabled"
            ),
            "classifier": "LinearSVC",
            "svm_C": args.C,
            "svm_class_weight": None,
            "max_iter": args.max_iter,
            "metrics": "not calculated; reserved for Stage 4",
            "graph_scope_in_stage2": (
                "subject-level graph constructed from all retained EEG "
                "segments before CV; not reconstructed per fold"
            ),
            "graph_test_isolation_status": (
                "NOT STRICTLY TEST-ISOLATED because Stage 2 graph "
                "construction precedes CV. The graph is label-independent "
                "but uses held-out EEG signals."
            ),
            "blink_rate": (
                "not available in Stage 2 because no explicit blink-event "
                "channel is exposed by the documented preprocessed ET data"
            ),
        },
        "feature_names": feature_names,
        "subjects": subject_results,
        "all_predictions": all_predictions,
        "all_folds": all_folds,
        "subject_summary": subject_summary,
    }

    pkl_path = args.output_dir / "stage3_cv_results.pkl"
    pred_path = args.output_dir / "stage3_oof_predictions.csv"
    selection_path = args.output_dir / "stage3_feature_selection.csv"
    summary_path = args.output_dir / "stage3_summary.csv"

    save_pickle(pkl_path, output)
    write_prediction_csv(pred_path, all_predictions)
    write_selection_csv(selection_path, all_folds, feature_names)

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        fields = [
            "subject_id",
            "n_epochs",
            "n_features",
            "n_predictions",
            "n_fallback_folds",
        ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(subject_summary)

    print("\nStage 3 complete.")
    print(f"CV results          : {pkl_path}")
    print(f"OOF predictions     : {pred_path}")
    print(f"Feature selection   : {selection_path}")
    print(f"Subject summary     : {summary_path}")
    print(
        "\nImportant: Stage 3 keeps the Stage 2 subject-level graph. "
        "Therefore the CV predictions are not yet a strictly "
        "test-isolated end-to-end pipeline. The graph-construction "
        "scope must be addressed before making a strong 'no leakage' claim."
    )


if __name__ == "__main__":
    main()
