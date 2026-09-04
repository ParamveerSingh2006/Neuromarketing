#!/usr/bin/env python3
"""
PT_1 — Leakage-safe subject-wise 10-fold EEG/ET classification pipeline.

Experimental design
-------------------
42 subjects
    -> independent Stratified 10-fold CV per subject
    -> fold-specific XGBoost + SHAP EEG feature selection
    -> common Top-50 / Top-75 EEG features for all downstream classifiers
    -> fixed 10 ET features
    -> 5 modality configurations
    -> XGBoost/SVM: class weighting OFF and ON
    -> LDA: class weighting is intentionally NOT applied
    -> Accuracy, F1, Cohen's Kappa

Total configurations:
    XGBoost: 5 x 2 = 10
    SVM:     5 x 2 = 10
    LDA:     5 x 1 =  5
    Total = 25

Leakage rule
------------
For every subject/fold, all supervised training-dependent operations are
performed using the training portion only. In particular, SHAP feature
selection is fitted/calculated only from the training fold. Validation data
are used only for prediction/evaluation.

Expected input directory
------------------------
<dataset_dir>/
    subjects.txt
    feature_names_eeg.json
    feature_names_et.json
    extraction_meta.json              # optional
    S01_eeg_X.npy
    S01_et_X.npy
    S01_y.npy
    ...
    S44_eeg_X.npy
    S44_et_X.npy
    S44_y.npy

The script discovers subjects from subjects.txt when available; otherwise
it discovers subjects from *_eeg_X.npy files.

Required packages
-----------------
numpy, pandas, scikit-learn, xgboost, shap, openpyxl

Example
-------
python PT1_10fold_SHAP_25config.py ^
    --data-dir "D:\\PT_1\\Dataset_Extracted\\Separate" ^
    --output-dir "D:\\PT_1\\Results\\10Fold_25Config"
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier
except ImportError as exc:
    raise ImportError(
        "xgboost is required. Install it with: pip install xgboost"
    ) from exc

try:
    import shap
except ImportError as exc:
    raise ImportError(
        "shap is required. Install it with: pip install shap"
    ) from exc


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_STATE = 42
N_SPLITS = 10

TOP_K_VALUES = (50, 75)

EXPECTED_N_EEG_FEATURES = 19 * 7 * 10
EXPECTED_N_IMFS = 7
EXPECTED_N_EEG_CHANNELS = 19
EXPECTED_N_EEG_FEATURES_PER_IMF = 10
EXPECTED_N_ET_FEATURES = 10

ET_FEATURES = [
    "Fixation Count",
    "Mean Fixation Duration",
    "Total Fixation Duration",
    "Time to First Fixation",
    "Mean Saccade Amplitude",
    "Mean Saccade Velocity",
    "Scanpath Length",
    "Mean Pupil Diameter",
    "Pupil Diameter Variability",
    "Dwell Time",
]

EEG_FEATURES = [
    "Relative Energy",
    "Differential Entropy",
    "Sample Entropy",
    "Higuchi Fractal Dimension",
    "Katz Fractal Dimension",
    "Hjorth Activity",
    "Hjorth Mobility",
    "Hjorth Complexity",
    "Hurst Exponent",
    "Largest Lyapunov Exponent",
]

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2",
    "F7", "F8", "T7", "T8", "P7", "P8", "Fz", "Cz", "Pz",
]

CLASSIFIERS = ("XGBoost", "SVM", "LDA")

# Five modality configurations.
MODALITY_CONFIGS = (
    ("ET-10", "ET", 0, 10),
    ("EEG-50", "EEG", 50, 0),
    ("EEG-75", "EEG", 75, 0),
    ("EEG-50+ET-10", "HYBRID", 50, 10),
    ("EEG-75+ET-10", "HYBRID", 75, 10),
)


@dataclass(frozen=True)
class Config:
    name: str
    classifier: str
    weighting: str
    modality: str
    eeg_k: int
    et_n: int


def build_configurations() -> List[Config]:
    configs: List[Config] = []

    for modality_name, modality, eeg_k, et_n in MODALITY_CONFIGS:
        for clf in ("XGBoost", "SVM"):
            for weighting in ("OFF", "ON"):
                configs.append(
                    Config(
                        name=f"{modality_name}__{clf}__Weight-{weighting}",
                        classifier=clf,
                        weighting=weighting,
                        modality=modality,
                        eeg_k=eeg_k,
                        et_n=et_n,
                    )
                )

        # LDA deliberately has no weighting branch.
        configs.append(
            Config(
                name=f"{modality_name}__LDA__Weight-OFF",
                classifier="LDA",
                weighting="OFF",
                modality=modality,
                eeg_k=eeg_k,
                et_n=et_n,
            )
        )

    assert len(configs) == 25, f"Expected 25 configurations, got {len(configs)}"
    return configs


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def configure_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "pipeline.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def normalise_feature_name(value) -> str:
    return str(value).strip()


def extract_feature_list(metadata, kind: str) -> List[str]:
    """
    Read the existing feature_names_*.json structure.

    In the current PT_1 dataset the JSON stores the ten base feature names
    under the "features" key. EEG column-level names are constructed from
    channel × IMF × base-feature order, matching the existing notebook.
    """
    if not isinstance(metadata, dict) or "features" not in metadata:
        raise ValueError(
            f"{kind} feature metadata must contain a 'features' list."
        )

    names = [normalise_feature_name(x) for x in metadata["features"]]

    if len(names) != 10:
        raise ValueError(
            f"{kind} metadata must contain exactly 10 base features; "
            f"found {len(names)}."
        )

    if len(set(names)) != len(names):
        raise ValueError(f"{kind} base feature names are not unique.")

    return names


def build_eeg_feature_names() -> List[str]:
    """Build the 1330 EEG column names in the existing channel-major order."""
    names = []
    for channel in EEG_CHANNELS:
        for imf in range(1, EXPECTED_N_IMFS + 1):
            for feature in EEG_FEATURES:
                names.append(f"EEG|{channel}|IMF{imf}|{feature}")
    return names


def build_et_feature_names() -> List[str]:
    return [f"ET|{feature}" for feature in ET_FEATURES]



def load_subject_ids(data_dir: Path) -> List[str]:
    subjects_file = data_dir / "subjects.txt"

    if subjects_file.exists():
        subjects = []
        for line in subjects_file.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if s:
                subjects.append(s)

        if subjects:
            return subjects

    subjects = sorted(
        p.name[: -len("_eeg_X.npy")]
        for p in data_dir.glob("*_eeg_X.npy")
    )

    if not subjects:
        raise FileNotFoundError(
            f"No *_eeg_X.npy files were found in {data_dir}"
        )

    return subjects


def load_subject_arrays(
    data_dir: Path,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_path = data_dir / f"{subject}_eeg_X.npy"
    et_path = data_dir / f"{subject}_et_X.npy"
    y_path = data_dir / f"{subject}_y.npy"

    missing = [str(p) for p in (eeg_path, et_path, y_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"{subject}: missing required files: {missing}"
        )

    eeg = np.load(eeg_path, allow_pickle=False)
    et = np.load(et_path, allow_pickle=False)
    y = np.load(y_path, allow_pickle=False)

    eeg = np.asarray(eeg)
    et = np.asarray(et)
    y = np.asarray(y).reshape(-1)

    if eeg.ndim != 2:
        raise ValueError(f"{subject}: EEG array must be 2-D; got {eeg.shape}")
    if et.ndim != 2:
        raise ValueError(f"{subject}: ET array must be 2-D; got {et.shape}")
    if y.ndim != 1:
        raise ValueError(f"{subject}: y must be 1-D; got {y.shape}")

    if not (len(eeg) == len(et) == len(y)):
        raise ValueError(
            f"{subject}: row mismatch: EEG={len(eeg)}, ET={len(et)}, y={len(y)}"
        )

    if not np.isfinite(eeg.astype(float, copy=False)).all():
        raise ValueError(f"{subject}: EEG contains NaN/Inf.")
    if not np.isfinite(et.astype(float, copy=False)).all():
        raise ValueError(f"{subject}: ET contains NaN/Inf.")

    return eeg, et, y


def resolve_et_columns(
    et_feature_names: Sequence[str],
    expected_names: Sequence[str],
) -> List[int]:
    normalised = {
        normalise_feature_name(name): i
        for i, name in enumerate(et_feature_names)
    }

    missing = [name for name in expected_names if name not in normalised]
    if missing:
        raise ValueError(
            "The required 10 ET features were not found in feature_names_et.json: "
            + ", ".join(missing)
        )

    indices = [normalised[name] for name in expected_names]

    if len(indices) != 10:
        raise AssertionError("ET feature selection must contain exactly 10 features.")

    return indices


def validate_eeg_feature_structure(
    eeg_feature_names: Sequence[str],
    n_columns: int,
) -> None:
    if n_columns != EXPECTED_N_EEG_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_N_EEG_FEATURES} EEG columns "
            f"(19 channels × 7 IMFs × 10 features), but found {n_columns}."
        )

    if len(eeg_feature_names) != n_columns:
        raise ValueError(
            f"EEG metadata has {len(eeg_feature_names)} names but EEG matrix has "
            f"{n_columns} columns."
        )

    # We deliberately validate the fixed representation rather than silently
    # guessing a different IMF count.
    logging.info(
        "EEG representation validated: %d channels × %d IMFs × %d features = %d columns.",
        EXPECTED_N_EEG_CHANNELS,
        EXPECTED_N_IMFS,
        EXPECTED_N_EEG_FEATURES_PER_IMF,
        EXPECTED_N_EEG_FEATURES,
    )


def validate_labels(y: np.ndarray, subject: str) -> None:
    unique = np.unique(y)

    if len(unique) != 2:
        raise ValueError(
            f"{subject}: binary classification requires exactly 2 classes; "
            f"found {unique.tolist()}"
        )

    counts = {str(v): int(np.sum(y == v)) for v in unique}
    logging.info("%s class distribution: %s", subject, counts)


def make_binary_labels(y: np.ndarray) -> Tuple[np.ndarray, Dict[object, int]]:
    """
    Convert arbitrary two-class labels to 0/1.

    The mapping is deterministic using sorted unique labels.
    """
    unique = np.unique(y)
    if len(unique) != 2:
        raise ValueError(
            f"Expected exactly two classes; found {unique.tolist()}"
        )

    mapping = {unique[0]: 0, unique[1]: 1}
    y_binary = np.asarray([mapping[v] for v in y], dtype=np.int8)

    return y_binary, mapping


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

def make_shap_selector() -> XGBClassifier:
    """
    Dedicated XGBoost model used ONLY for fold-specific SHAP feature selection.

    It is separate from the downstream XGBoost evaluator. The selector uses
    the XGBoost settings already present in the supplied PT_1 notebook.
    """
    return XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )


def make_xgb_classifier(weighting: str, y_train: np.ndarray) -> XGBClassifier:
    """
    Downstream XGBoost evaluator.

    Weighting ON uses scale_pos_weight computed from the CURRENT TRAINING
    FOLD ONLY. The validation fold is never used to determine the weight.
    """
    model = XGBClassifier(
        n_estimators=150,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
    )

    if weighting == "ON":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))

        if n_neg == 0 or n_pos == 0:
            raise ValueError(
                "Cannot calculate XGBoost class weighting with a one-class "
                "training fold."
            )

        model.set_params(scale_pos_weight=n_neg / n_pos)

    return model


def make_svm(weighting: str) -> Pipeline:
    """
    Preserve the existing SVM configuration:
    linear kernel, C=1.0.

    StandardScaler is fitted by this pipeline on the training fold only.
    """
    class_weight = "balanced" if weighting == "ON" else None

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                SVC(
                    kernel="linear",
                    C=1.0,
                    class_weight=class_weight,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_lda() -> Pipeline:
    """
    Preserve the existing LDA configuration from the supplied notebook:
    LSQR solver with automatic shrinkage.

    LDA deliberately has no class-weighting branch.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            ),
        ]
    )


def fit_downstream_model(
    config: Config,
    X_train: np.ndarray,
    y_train: np.ndarray,
):
    if config.classifier == "XGBoost":
        model = make_xgb_classifier(config.weighting, y_train)
        model.fit(X_train, y_train)
        return model

    if config.classifier == "SVM":
        model = make_svm(config.weighting)
        model.fit(X_train, y_train)
        return model

    if config.classifier == "LDA":
        model = make_lda()
        model.fit(X_train, y_train)
        return model

    raise ValueError(f"Unknown classifier: {config.classifier}")


# ---------------------------------------------------------------------------
# SHAP feature selection
# ---------------------------------------------------------------------------

def _coerce_shap_values(shap_values) -> np.ndarray:
    """
    Normalize SHAP's version-dependent output for binary XGBoost.
    """
    if isinstance(shap_values, list):
        if len(shap_values) == 1:
            arr = np.asarray(shap_values[0])
        elif len(shap_values) == 2:
            arr = np.asarray(shap_values[1])
        else:
            raise ValueError(
                f"Unexpected SHAP list length: {len(shap_values)}"
            )
    else:
        arr = np.asarray(shap_values)

    if arr.ndim == 3:
        # Possible shape: samples x features x classes.
        if arr.shape[-1] == 2:
            arr = arr[:, :, 1]
        else:
            arr = np.squeeze(arr)

    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP array shape after normalization: {arr.shape}"
        )

    return arr


def select_top_eeg_features_with_shap(
    X_train_eeg: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
    top_k_values: Sequence[int] = TOP_K_VALUES,
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Fit the dedicated selector ONLY on X_train_eeg/y_train.

    Returns
    -------
    selected_indices:
        {50: indices, 75: indices}
    mean_abs_shap:
        Mean absolute SHAP importance for every EEG feature.
    """
    selector = make_shap_selector()
    selector.fit(X_train_eeg, y_train)

    explainer = shap.TreeExplainer(selector)

    # No validation samples are passed to the explainer.
    shap_raw = explainer.shap_values(X_train_eeg)
    shap_values = _coerce_shap_values(shap_raw)

    if shap_values.shape != X_train_eeg.shape:
        raise ValueError(
            "SHAP output shape does not match training EEG shape: "
            f"SHAP={shap_values.shape}, X_train={X_train_eeg.shape}"
        )

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

    # Stable deterministic ordering:
    # first by descending SHAP importance, then by original feature index.
    feature_indices = np.arange(X_train_eeg.shape[1])
    ranking = np.lexsort((feature_indices, -mean_abs_shap))

    selected: Dict[int, np.ndarray] = {}
    for k in top_k_values:
        if k > len(ranking):
            raise ValueError(
                f"Requested Top-{k}, but only {len(ranking)} EEG features exist."
            )
        selected[k] = ranking[:k].copy()

    return selected, mean_abs_shap


# ---------------------------------------------------------------------------
# Configuration matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    config: Config,
    X_eeg: np.ndarray,
    X_et: np.ndarray,
    eeg_indices: Dict[int, np.ndarray],
    et_indices: Sequence[int],
) -> np.ndarray:
    """
    Construct a configuration matrix from already-selected columns.

    The caller is responsible for passing either training or validation data.
    The selected EEG indices originate exclusively from the corresponding
    training fold.
    """
    if config.modality == "ET":
        return X_et[:, et_indices]

    if config.modality == "EEG":
        return X_eeg[:, eeg_indices[config.eeg_k]]

    if config.modality == "HYBRID":
        eeg_part = X_eeg[:, eeg_indices[config.eeg_k]]
        et_part = X_et[:, et_indices]
        return np.concatenate([eeg_part, et_part], axis=1)

    raise ValueError(f"Unknown modality: {config.modality}")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_subject(
    subject: str,
    X_eeg: np.ndarray,
    X_et: np.ndarray,
    y: np.ndarray,
    eeg_feature_names: Sequence[str],
    et_indices: Sequence[int],
    configurations: Sequence[Config],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run independent stratified 10-fold CV for one subject.

    Returns
    -------
    subject_metrics:
        One row per configuration for this subject.
    fold_predictions:
        One row per validation observation/configuration/fold.
    """
    validate_labels(y, subject)
    y_binary, label_mapping = make_binary_labels(y)

    min_class_count = int(np.min(np.bincount(y_binary)))
    if min_class_count < N_SPLITS:
        raise ValueError(
            f"{subject}: smallest class has only {min_class_count} samples, "
            f"but 10-fold stratification requires at least {N_SPLITS}."
        )

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    # Store predictions from all folds so that final subject metrics are
    # calculated from out-of-fold predictions, not by averaging fold metrics.
    # Each configuration stores a Python list of fold-level NumPy arrays.
    # Using List[np.ndarray] here prevents Pylance from inferring [] as a
    # NumPy array and incorrectly flagging the later .append() calls.
    prediction_store: Dict[str, Dict[str, List[np.ndarray]]] = {
        cfg.name: {
            "y_true": [],
            "y_pred": [],
        }
        for cfg in configurations
    }

    fold_rows: List[dict] = []
    feature_rows: List[dict] = []

    subject_start = time.perf_counter()

    for fold_idx, (train_idx, val_idx) in enumerate(
        cv.split(X_eeg, y_binary),
        start=1,
    ):
        fold_start = time.perf_counter()

        X_train_eeg = X_eeg[train_idx]
        X_val_eeg = X_eeg[val_idx]

        X_train_et = X_et[train_idx]
        X_val_et = X_et[val_idx]

        y_train = y_binary[train_idx]
        y_val = y_binary[val_idx]

        logging.info(
            "%s | Fold %02d/%02d | train=%d validation=%d",
            subject,
            fold_idx,
            N_SPLITS,
            len(train_idx),
            len(val_idx),
        )

        # ------------------------------------------------------------------
        # CRITICAL LEAKAGE BOUNDARY
        #
        # SHAP selector is fitted ONLY on the current training fold.
        # X_val_eeg/y_val are not passed anywhere in this operation.
        # ------------------------------------------------------------------
        selected_eeg, mean_abs_shap = select_top_eeg_features_with_shap(
            X_train_eeg=X_train_eeg,
            y_train=y_train,
            feature_names=eeg_feature_names,
        )

        # Record the fold-specific feature ranking for reproducibility.
        for rank, feature_idx in enumerate(
            np.lexsort(
                (
                    np.arange(len(mean_abs_shap)),
                    -mean_abs_shap,
                )
            ),
            start=1,
        ):
            feature_rows.append(
                {
                    "subject": subject,
                    "fold": fold_idx,
                    "feature_rank": rank,
                    "feature_index": int(feature_idx),
                    "feature_name": eeg_feature_names[feature_idx],
                    "mean_abs_shap": float(mean_abs_shap[feature_idx]),
                }
            )

        for config in configurations:
            X_train = build_feature_matrix(
                config,
                X_train_eeg,
                X_train_et,
                selected_eeg,
                et_indices,
            )
            X_val = build_feature_matrix(
                config,
                X_val_eeg,
                X_val_et,
                selected_eeg,
                et_indices,
            )

            # ----------------------------------------------------------------
            # Another leakage boundary:
            # SVM/LDA scalers are fitted by the Pipeline on X_train only.
            # X_val is transformed only after fitting on X_train.
            #
            # XGBoost does not require scaling.
            # ----------------------------------------------------------------
            model = fit_downstream_model(
                config=config,
                X_train=X_train,
                y_train=y_train,
            )

            y_pred = np.asarray(model.predict(X_val)).reshape(-1).astype(int)

            prediction_store[config.name]["y_true"].append(y_val.copy())
            prediction_store[config.name]["y_pred"].append(y_pred.copy())

            for local_i, global_i in enumerate(val_idx):
                fold_rows.append(
                    {
                        "subject": subject,
                        "fold": fold_idx,
                        "global_sample_index": int(global_i),
                        "configuration": config.name,
                        "classifier": config.classifier,
                        "weighting": config.weighting,
                        "modality": config.modality,
                        "eeg_features": config.eeg_k,
                        "et_features": config.et_n,
                        "y_true": int(y_val[local_i]),
                        "y_pred": int(y_pred[local_i]),
                    }
                )

        logging.info(
            "%s | Fold %02d completed in %.2f min",
            subject,
            fold_idx,
            (time.perf_counter() - fold_start) / 60.0,
        )

    subject_rows: List[dict] = []

    for config in configurations:
        y_true = np.concatenate(prediction_store[config.name]["y_true"])
        y_pred = np.concatenate(prediction_store[config.name]["y_pred"])

        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        kappa = cohen_kappa_score(y_true, y_pred)

        subject_rows.append(
            {
                "subject": subject,
                "configuration": config.name,
                "classifier": config.classifier,
                "weighting": config.weighting,
                "modality": config.modality,
                "eeg_features": config.eeg_k,
                "et_features": config.et_n,
                "accuracy": float(accuracy),
                "f1": float(f1),
                "kappa": float(kappa),
                "n_samples": int(len(y_true)),
                "label_mapping": json.dumps(
                    {str(k): int(v) for k, v in label_mapping.items()}
                ),
            }
        )

    subject_metrics = pd.DataFrame(subject_rows)
    fold_predictions = pd.DataFrame(fold_rows)
    feature_importance = pd.DataFrame(feature_rows)

    # Per-subject files make long runs recoverable and auditable.
    safe_subject = subject.replace("/", "_").replace("\\", "_")

    subject_metrics.to_csv(
        output_dir / f"{safe_subject}_metrics.csv",
        index=False,
    )

    fold_predictions.to_csv(
        output_dir / f"{safe_subject}_fold_predictions.csv",
        index=False,
    )

    feature_importance.to_csv(
        output_dir / f"{safe_subject}_fold_shap_ranking.csv",
        index=False,
    )

    logging.info(
        "%s completed in %.2f min",
        subject,
        (time.perf_counter() - subject_start) / 60.0,
    )

    return subject_metrics, fold_predictions


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="PT_1 leakage-safe subject-wise 10-fold, 25-configuration pipeline."
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing *_eeg_X.npy, *_et_X.npy, *_y.npy and feature JSON files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which all results will be written.",
    )

    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional explicit subject IDs, e.g. S01 S02 S03. Default: subjects.txt/discovery.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    configure_logging(output_dir)

    logging.info("=" * 80)
    logging.info("PT_1 — leakage-safe subject-wise 10-fold / 25-configuration pipeline")
    logging.info("=" * 80)

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    # ----------------------------------------------------------------------
    # Load feature metadata
    # ----------------------------------------------------------------------
    eeg_json = data_dir / "feature_names_eeg.json"
    et_json = data_dir / "feature_names_et.json"

    if not eeg_json.exists():
        raise FileNotFoundError(f"Missing EEG feature metadata: {eeg_json}")
    if not et_json.exists():
        raise FileNotFoundError(f"Missing ET feature metadata: {et_json}")

    eeg_base_features = extract_feature_list(
        load_json(eeg_json),
        "EEG",
    )
    et_base_features = extract_feature_list(
        load_json(et_json),
        "ET",
    )

    # Validate the supplied metadata against the locked PT_1 feature lists.
    # The JSON files use snake_case names; compare normalized keys so naming
    # style does not incorrectly reject an otherwise correct dataset.
    expected_eeg_keys = [
        "relative_energy",
        "differential_entropy",
        "sample_entropy",
        "higuchi_fd",
        "katz_fd",
        "hjorth_activity",
        "hjorth_mobility",
        "hjorth_complexity",
        "hurst_exponent",
        "lyapunov_exponent",
    ]
    # The current ET metadata uses unit-qualified snake_case names.
    # Map those exact metadata keys to the locked canonical 10-feature order.
    expected_et_keys = [
        "fixation_count",
        "mean_fixation_duration_ms",
        "total_fixation_duration_ms",
        "time_to_first_fixation_ms",
        "mean_saccade_amplitude_px",
        "mean_saccade_velocity_px_s",
        "scanpath_length_px",
        "mean_pupil_diameter_px",
        "pupil_diameter_variability_px",
        "dwell_time_ms",
    ]

    def feature_key(name: str) -> str:
        return name.strip().lower().replace(" ", "_").replace("-", "_")

    actual_eeg_keys = [feature_key(x) for x in eeg_base_features]
    actual_et_keys = [feature_key(x) for x in et_base_features]

    if actual_eeg_keys != expected_eeg_keys:
        raise ValueError(
            "feature_names_eeg.json does not match the locked 10 EEG features "
            "in the expected order.\n"
            f"Expected keys: {expected_eeg_keys}\n"
            f"Found keys:    {actual_eeg_keys}"
        )

    if actual_et_keys != expected_et_keys:
        raise ValueError(
            "feature_names_et.json does not match the locked 10 ET features "
            "in the expected order.\n"
            f"Expected keys: {expected_et_keys}\n"
            f"Found keys:    {actual_et_keys}"
        )

    # The JSON files store the 10 base EEG/ET features. The EEG matrix stores
    # those features for 19 channels × 7 IMFs, so build the 1330 column-level
    # names in the same order as the existing notebook.
    eeg_feature_names = build_eeg_feature_names()
    et_feature_names = build_et_feature_names()
    et_indices = list(range(EXPECTED_N_ET_FEATURES))

    logging.info("Feature metadata validated successfully.")
    logging.info("Using the following 10 ET features:")
    for i, name in enumerate(ET_FEATURES, start=1):
        logging.info("  ET-%02d: %s", i, name)

    # ----------------------------------------------------------------------
    # Subject list
    # ----------------------------------------------------------------------
    subjects = args.subjects if args.subjects else load_subject_ids(data_dir)

    if len(subjects) != 42:
        logging.warning(
            "Expected 42 subjects from the current PT_1 dataset, but found %d.",
            len(subjects),
        )

    logging.info("Subjects to process (%d): %s", len(subjects), ", ".join(subjects))

    configurations = build_configurations()

    config_table = pd.DataFrame(
        [
            {
                "configuration": c.name,
                "classifier": c.classifier,
                "weighting": c.weighting,
                "modality": c.modality,
                "eeg_features": c.eeg_k,
                "et_features": c.et_n,
            }
            for c in configurations
        ]
    )

    config_table.to_csv(
        output_dir / "configuration_25_definition.csv",
        index=False,
    )

    logging.info("Configuration count: %d", len(configurations))
    logging.info("\n%s", config_table.to_string(index=False))

    # ----------------------------------------------------------------------
    # Save a machine-readable run specification.
    # ----------------------------------------------------------------------
    run_spec = {
        "random_state": RANDOM_STATE,
        "n_splits": N_SPLITS,
        "n_subjects": len(subjects),
        "subjects": subjects,
        "n_eeg_features": EXPECTED_N_EEG_FEATURES,
        "eeg_channels": EXPECTED_N_EEG_CHANNELS,
        "eeg_imfs": EXPECTED_N_IMFS,
        "eeg_features_per_imf": EXPECTED_N_EEG_FEATURES_PER_IMF,
        "n_et_features": EXPECTED_N_ET_FEATURES,
        "et_features": ET_FEATURES,
        "eeg_features": EEG_FEATURES,
        "classifiers": list(CLASSIFIERS),
        "threshold_optimization": False,
        "shap_selector": "dedicated XGBoost fitted separately inside every training fold",
        "shap_shared_across_downstream_classifiers": True,
        "svm_kernel": "linear",
        "svm_C": 1.0,
        "lda_solver": "lsqr",
        "lda_shrinkage": "auto",
        "xgb_n_estimators": 150,
        "xgb_max_depth": 3,
        "lda_class_weighting": False,
        "configuration_count": len(configurations),
        "data_leakage_policy": {
            "shap": "training fold only",
            "svm_scaler": "training fold only",
            "lda_scaler": "training fold only",
            "validation_use": "prediction/evaluation only",
        },
    }

    with (output_dir / "run_specification.json").open("w", encoding="utf-8") as f:
        json.dump(run_spec, f, indent=2)

    # ----------------------------------------------------------------------
    # Process subjects
    # ----------------------------------------------------------------------
    all_subject_metrics: List[pd.DataFrame] = []
    all_fold_predictions: List[pd.DataFrame] = []
    failed_subjects: List[Dict[str, str]] = []

    overall_start = time.perf_counter()

    for subject_number, subject in enumerate(subjects, start=1):
        logging.info("")
        logging.info("=" * 80)
        logging.info("SUBJECT %d/%d: %s", subject_number, len(subjects), subject)
        logging.info("=" * 80)

        try:
            eeg, et, y = load_subject_arrays(data_dir, subject)

            validate_eeg_feature_structure(
                eeg_feature_names,
                eeg.shape[1],
            )

            if et.shape[1] != EXPECTED_N_ET_FEATURES:
                raise ValueError(
                    f"{subject}: expected {EXPECTED_N_ET_FEATURES} ET columns, "
                    f"found {et.shape[1]}."
                )

            logging.info(
                "%s shapes: EEG=%s, ET=%s, y=%s",
                subject,
                eeg.shape,
                et.shape,
                y.shape,
            )

            metrics, predictions = evaluate_subject(
                subject=subject,
                X_eeg=eeg.astype(np.float32, copy=False),
                X_et=et.astype(np.float32, copy=False),
                y=y,
                eeg_feature_names=eeg_feature_names,
                et_indices=et_indices,
                configurations=configurations,
                output_dir=output_dir,
            )

            all_subject_metrics.append(metrics)
            all_fold_predictions.append(predictions)

        except Exception as exc:
            failed_subjects.append(
                {
                    "subject": subject,
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
            )

            logging.exception(
                "FAILED subject %s. Continuing with remaining subjects.",
                subject,
            )

    # ----------------------------------------------------------------------
    # Aggregate only successfully completed subjects
    # ----------------------------------------------------------------------
    if all_subject_metrics:
        subject_metrics_all = pd.concat(
            all_subject_metrics,
            ignore_index=True,
        )

        subject_metrics_all.to_csv(
            output_dir / "all_subject_metrics.csv",
            index=False,
        )

        config_summary = (
            subject_metrics_all
            .groupby(
                [
                    "configuration",
                    "classifier",
                    "weighting",
                    "modality",
                    "eeg_features",
                    "et_features",
                ],
                dropna=False,
            )
            .agg(
                subjects_evaluated=("subject", "nunique"),
                mean_accuracy=("accuracy", "mean"),
                std_accuracy=("accuracy", "std"),
                mean_f1=("f1", "mean"),
                std_f1=("f1", "std"),
                mean_kappa=("kappa", "mean"),
                std_kappa=("kappa", "std"),
            )
            .reset_index()
        )

        config_summary.to_csv(
            output_dir / "configuration_summary_25.csv",
            index=False,
        )

    else:
        subject_metrics_all = pd.DataFrame()
        logging.error("No subjects completed successfully.")

    if all_fold_predictions:
        fold_predictions_all = pd.concat(
            all_fold_predictions,
            ignore_index=True,
        )

        fold_predictions_all.to_csv(
            output_dir / "all_fold_predictions.csv",
            index=False,
        )
    else:
        fold_predictions_all = pd.DataFrame()

    failed_df = pd.DataFrame(failed_subjects)

    if not failed_df.empty:
        failed_df.to_csv(
            output_dir / "failed_subjects.csv",
            index=False,
        )

    # ----------------------------------------------------------------------
    # Excel workbook for convenient inspection
    # ----------------------------------------------------------------------
    try:
        excel_path = output_dir / "PT1_25_configuration_results.xlsx"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            config_table.to_excel(
                writer,
                sheet_name="Config_25",
                index=False,
            )

            if not subject_metrics_all.empty:
                subject_metrics_all.to_excel(
                    writer,
                    sheet_name="Subject_Metrics",
                    index=False,
                )

                config_summary.to_excel(
                    writer,
                    sheet_name="Config_Summary",
                    index=False,
                )

            if not fold_predictions_all.empty:
                fold_predictions_all.to_excel(
                    writer,
                    sheet_name="Fold_Predictions",
                    index=False,
                )

            if not failed_df.empty:
                failed_df.to_excel(
                    writer,
                    sheet_name="Failed_Subjects",
                    index=False,
                )

        logging.info("Excel workbook written: %s", excel_path)

    except Exception:
        logging.exception(
            "Could not write Excel workbook. CSV outputs remain available."
        )

    elapsed_minutes = (time.perf_counter() - overall_start) / 60.0

    logging.info("")
    logging.info("=" * 80)
    logging.info("RUN COMPLETE")
    logging.info("Elapsed time: %.2f minutes", elapsed_minutes)
    logging.info("Successful subjects: %d/%d", len(all_subject_metrics), len(subjects))
    logging.info("Failed subjects: %d", len(failed_subjects))
    logging.info("Output directory: %s", output_dir)
    logging.info("=" * 80)

    if failed_subjects:
        logging.warning(
            "Some subjects failed. See failed_subjects.csv and pipeline.log."
        )
        return 2

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
