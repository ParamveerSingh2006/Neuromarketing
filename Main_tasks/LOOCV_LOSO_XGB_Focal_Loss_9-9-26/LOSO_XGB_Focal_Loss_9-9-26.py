#!/usr/bin/env python3
"""
PT_1 — Pipeline 2
Leakage-safe XGBoost experiment with configurable LOOCV or LOSO:
    - fold-specific SHAP EEG feature selection
    - Top-50 / Top-75 EEG features
    - fixed 10 ET features
    - ET / EEG / Hybrid feature configurations
    - class weighting OFF / ON
    - focal loss OFF / ON
    - fixed decision threshold = 0.50
    - Accuracy, F1, Cohen's Kappa, Balanced Accuracy,
      Precision, Recall/Sensitivity, Specificity, ROC-AUC

EXPERIMENTAL DESIGN
-------------------
Two cross-validation modes are supported:

    LOOCV:
        Leave-One-Observation-Out within each subject.
        One observation is held out at a time.

    LOSO:
        Leave-One-Subject-Out across subjects.
        One complete subject is held out at a time.

For every outer fold:

    TRAINING EEG ONLY
        |
        +--> dedicated standard XGBoost selector
        |        |
        |        +--> SHAP on training EEG only
        |        +--> complete ranking of 1330 EEG features
        |        +--> Top-50 and Top-75 indices
        |
        +--> construct 5 feature configurations
        |        ET-10
        |        EEG-50
        |        EEG-75
        |        EEG-50+ET-10
        |        EEG-75+ET-10
        |
        +--> class weighting OFF / ON
        |
        +--> focal loss OFF / ON
        |
        +--> XGBoost
        |
        +--> validation prediction
                 |
                 +--> fixed threshold 0.50

TOTAL:
    5 feature configurations
    x 2 weighting states
    x 2 focal-loss states
    = 20 configurations per evaluated unit

CRITICAL LEAKAGE RULE
---------------------
All supervised training-dependent operations are performed using the
outer training fold only.

Validation observations are NEVER used for:
    - SHAP fitting
    - SHAP calculation
    - feature selection
    - class-weight calculation
    - model fitting
    - threshold selection

Threshold optimization is intentionally NOT included in Pipeline 2.
The decision threshold is fixed at 0.50.

FOCAL LOSS
----------
Pipeline 2 uses:
    alpha = 0.5
    gamma = 2.0

alpha=0.5 keeps focal loss symmetric between classes, while the separate
class-weighting experiment retains the Pipeline-1 imbalance correction.

When class weighting is ON, training sample weights are:
    positive class = n_negative / n_positive
    negative class = 1

This is mathematically equivalent to Pipeline-1's binary
scale_pos_weight definition. For the custom focal objective, explicit
training sample weights are used so that the same weighting is applied
reliably to the custom loss.

Because focal loss can have a locally negative second derivative,
the custom XGBoost Hessian is clipped to a small positive floor.
This is a numerical safeguard required by tree boosting; the script
records how often Hessian clipping occurs.

XGBOOST
-------
Pipeline-1 structural parameters are retained:

    n_estimators = 150
    max_depth = 3
    learning_rate = 0.05
    subsample = 0.8
    colsample_bytree = 0.8
    random_state = 42
    n_jobs = -1

INPUT
-----
<dataset_dir>/
    subjects.txt
    feature_names_eeg.json
    feature_names_et.json
    S01_eeg_X.npy
    S01_et_X.npy
    S01_y.npy
    ...

OUTPUT
------
<output_dir>/
    subject_results/
        SXX_metrics.csv
    predictions/
        SXX_fold_predictions.csv
    feature_selection/
        SXX_fold_shap_ranking.csv
    summaries/
        all_subject_metrics.csv
        configuration_summary_20.csv
        failed_subjects.csv
    run_specification.json
    pipeline_2.log
    PT1_Pipeline_CV_results.xlsx

Paths can be changed either in the USER SETTINGS section below or by
using --data-dir and --output-dir.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple, Optional, Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    import xgboost as xgb
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


# ============================================================
# USER SETTINGS
# ============================================================

DATA_DIR = Path(
    r"C:\Users\tiet\Desktop\Paramveer\Neuromarketing\Dataset_Extracted_CEEMDAN\Separate"
)

OUTPUT_DIR = Path(
    r"C:\Users\tiet\Desktop\Paramveer\Neuromarketing\Results\LOSO_XGB_Focal_Loss_9-9-26"
)

# ============================================================
# FIXED EXPERIMENT SETTINGS
# ============================================================

RANDOM_STATE = 42

# Cross-validation mode:
#   "LOOCV" = Leave-One-Observation-Out within each subject
#   "LOSO"  = Leave-One-Subject-Out across subjects
CV_MODE = "LOSO"

TOP_K_VALUES = (50, 75)

FOCAL_ALPHA = 0.5
FOCAL_GAMMA = 2.0

DECISION_THRESHOLD = 0.50

EXPECTED_SUBJECT_COUNT = 42

N_EEG_CHANNELS = 19
N_IMFS = 7
N_BASE_EEG_FEATURES = 10
EXPECTED_N_EEG_FEATURES = (
    N_EEG_CHANNELS * N_IMFS * N_BASE_EEG_FEATURES
)
EXPECTED_N_ET_FEATURES = 10

EEG_CHANNELS = [
    "Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4",
    "O1", "O2", "F7", "F8", "T7", "T8", "P7", "P8",
    "Fz", "Cz", "Pz",
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

# Actual metadata keys in the current dataset.
EXPECTED_EEG_METADATA_KEYS = [
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

EXPECTED_ET_METADATA_KEYS = [
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

XGB_COMMON_PARAMS = {
    "n_estimators": 150,
    "max_depth": 3,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

# Small positive Hessian floor for XGBoost's custom objective.
HESSIAN_FLOOR = 1e-8


# ============================================================
# DATA CLASSES
# ============================================================

@dataclass(frozen=True)
class Config:
    name: str
    feature_config: str
    modality: str
    eeg_k: int
    et_n: int
    weighting: str
    focal_loss: str


# ============================================================
# LOGGING
# ============================================================

def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / "pipeline_2.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )


# ============================================================
# CONFIGURATION GENERATION
# ============================================================

def build_configurations() -> List[Config]:
    feature_configs = [
        ("ET-10", "ET", 0, 10),
        ("EEG-50", "EEG", 50, 0),
        ("EEG-75", "EEG", 75, 0),
        ("EEG-50+ET-10", "HYBRID", 50, 10),
        ("EEG-75+ET-10", "HYBRID", 75, 10),
    ]

    configurations: List[Config] = []

    for feature_name, modality, eeg_k, et_n in feature_configs:
        for weighting in ("OFF", "ON"):
            for focal_loss in ("OFF", "ON"):
                configurations.append(
                    Config(
                        name=(
                            f"{feature_name}"
                            f"__Weight-{weighting}"
                            f"__Focal-{focal_loss}"
                        ),
                        feature_config=feature_name,
                        modality=modality,
                        eeg_k=eeg_k,
                        et_n=et_n,
                        weighting=weighting,
                        focal_loss=focal_loss,
                    )
                )

    if len(configurations) != 20:
        raise RuntimeError(
            f"Expected 20 configurations, got {len(configurations)}"
        )

    return configurations


# ============================================================
# METADATA / FEATURE NAMES
# ============================================================

def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_feature_list(obj: Any, source_name: str) -> List[str]:
    if isinstance(obj, dict):
        if "features" in obj and isinstance(obj["features"], list):
            return [str(x) for x in obj["features"]]

        for key in ("feature_names", "names"):
            if key in obj and isinstance(obj[key], list):
                return [str(x) for x in obj[key]]

    if isinstance(obj, list):
        return [str(x) for x in obj]

    raise ValueError(
        f"{source_name}: could not extract a feature list from metadata."
    )


def normalise_feature_name(name: str) -> str:
    return (
        str(name)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
    )


def build_eeg_feature_names() -> List[str]:
    names: List[str] = []

    for channel in EEG_CHANNELS:
        for imf in range(1, N_IMFS + 1):
            for feature in EEG_FEATURES:
                names.append(
                    f"EEG|{channel}|IMF{imf}|{feature}"
                )

    if len(names) != EXPECTED_N_EEG_FEATURES:
        raise RuntimeError(
            f"Constructed {len(names)} EEG feature names; "
            f"expected {EXPECTED_N_EEG_FEATURES}."
        )

    return names


def load_feature_metadata(
    data_dir: Path,
) -> Tuple[List[str], List[str]]:
    eeg_json = data_dir / "feature_names_eeg.json"
    et_json = data_dir / "feature_names_et.json"

    if not eeg_json.exists():
        raise FileNotFoundError(
            f"Missing EEG feature metadata: {eeg_json}"
        )

    if not et_json.exists():
        raise FileNotFoundError(
            f"Missing ET feature metadata: {et_json}"
        )

    eeg_base = extract_feature_list(
        load_json(eeg_json),
        "feature_names_eeg.json",
    )
    et_base = extract_feature_list(
        load_json(et_json),
        "feature_names_et.json",
    )

    actual_eeg = [
        normalise_feature_name(x)
        for x in eeg_base
    ]
    actual_et = [
        normalise_feature_name(x)
        for x in et_base
    ]

    if actual_eeg != EXPECTED_EEG_METADATA_KEYS:
        raise ValueError(
            "feature_names_eeg.json does not match the locked 10 EEG "
            "features in the expected order.\n"
            f"Expected: {EXPECTED_EEG_METADATA_KEYS}\n"
            f"Found:    {actual_eeg}"
        )

    if actual_et != EXPECTED_ET_METADATA_KEYS:
        raise ValueError(
            "feature_names_et.json does not match the locked 10 ET "
            "features in the expected order.\n"
            f"Expected: {EXPECTED_ET_METADATA_KEYS}\n"
            f"Found:    {actual_et}"
        )

    eeg_names = build_eeg_feature_names()
    et_names = [
        f"ET|{feature}"
        for feature in ET_FEATURES
    ]

    return eeg_names, et_names


# ============================================================
# SUBJECT DISCOVERY / LOADING
# ============================================================

def load_subject_ids(data_dir: Path) -> List[str]:
    subjects_file = data_dir / "subjects.txt"

    if subjects_file.exists():
        subjects = [
            line.strip()
            for line in subjects_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]

        if subjects:
            return subjects

    subjects = sorted(
        p.name[:-len("_eeg_X.npy")]
        for p in data_dir.glob("*_eeg_X.npy")
    )

    if not subjects:
        raise FileNotFoundError(
            f"No *_eeg_X.npy files found in {data_dir}"
        )

    return subjects


def load_subject_arrays(
    data_dir: Path,
    subject: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    eeg_path = data_dir / f"{subject}_eeg_X.npy"
    et_path = data_dir / f"{subject}_et_X.npy"
    y_path = data_dir / f"{subject}_y.npy"

    missing = [
        str(p)
        for p in (eeg_path, et_path, y_path)
        if not p.exists()
    ]

    if missing:
        raise FileNotFoundError(
            f"{subject}: missing required files: {missing}"
        )

    eeg = np.asarray(
        np.load(eeg_path, allow_pickle=False)
    )
    et = np.asarray(
        np.load(et_path, allow_pickle=False)
    )
    y = np.asarray(
        np.load(y_path, allow_pickle=False)
    ).reshape(-1)

    if eeg.ndim != 2:
        raise ValueError(
            f"{subject}: EEG must be 2-D, got {eeg.shape}"
        )

    if et.ndim != 2:
        raise ValueError(
            f"{subject}: ET must be 2-D, got {et.shape}"
        )

    if y.ndim != 1:
        raise ValueError(
            f"{subject}: y must be 1-D, got {y.shape}"
        )

    if not (len(eeg) == len(et) == len(y)):
        raise ValueError(
            f"{subject}: row mismatch: "
            f"EEG={len(eeg)}, ET={len(et)}, y={len(y)}"
        )

    eeg_float = eeg.astype(np.float64, copy=False)
    et_float = et.astype(np.float64, copy=False)

    if not np.isfinite(eeg_float).all():
        raise ValueError(
            f"{subject}: EEG contains NaN/Inf."
        )

    if not np.isfinite(et_float).all():
        raise ValueError(
            f"{subject}: ET contains NaN/Inf."
        )

    if eeg.shape[1] != EXPECTED_N_EEG_FEATURES:
        raise ValueError(
            f"{subject}: expected {EXPECTED_N_EEG_FEATURES} EEG "
            f"features, got {eeg.shape[1]}"
        )

    if et.shape[1] != EXPECTED_N_ET_FEATURES:
        raise ValueError(
            f"{subject}: expected {EXPECTED_N_ET_FEATURES} ET "
            f"features, got {et.shape[1]}"
        )

    return (
        eeg_float.astype(np.float32),
        et_float.astype(np.float32),
        y,
    )


# ============================================================
# LABEL HANDLING
# ============================================================

def make_binary_labels(
    y: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, int]]:
    unique = np.unique(y)

    if len(unique) != 2:
        raise ValueError(
            f"Expected exactly two classes, found {unique.tolist()}"
        )

    # Deterministic mapping.
    try:
        ordered = sorted(unique.tolist())
    except TypeError:
        ordered = list(unique)

    mapping = {
        str(ordered[0]): 0,
        str(ordered[1]): 1,
    }

    y_binary = np.array(
        [mapping[str(value)] for value in y],
        dtype=np.int64,
    )

    return y_binary, mapping


# ============================================================
# XGBOOST / FOCAL LOSS
# ============================================================

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    positive = z >= 0
    out[positive] = 1.0 / (
        1.0 + np.exp(-np.clip(z[positive], -40.0, 40.0))
    )

    exp_z = np.exp(np.clip(z[~positive], -40.0, 40.0))
    out[~positive] = exp_z / (1.0 + exp_z)

    return out


def focal_loss_gradient_hessian(
    predt: np.ndarray,
    y: np.ndarray,
    alpha: float = FOCAL_ALPHA,
    gamma: float = FOCAL_GAMMA,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Binary focal loss:

        L = -alpha * (1 - q)^gamma * log(q)

    where:
        q = p        if y=1
        q = 1-p      if y=0

    Derivatives are with respect to the raw XGBoost margin z.

    XGBoost tree boosting requires a usable Hessian. Focal loss can have
    a negative local second derivative, so negative/small Hessians are
    clipped to HESSIAN_FLOOR. The number of clipped entries is returned
    for auditability.
    """
    predt = np.asarray(predt, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    p = sigmoid(predt)
    eps = 1e-12
    p = np.clip(p, eps, 1.0 - eps)

    q = np.where(y == 1.0, p, 1.0 - p)
    q = np.clip(q, eps, 1.0 - eps)

    one_minus_q = np.clip(1.0 - q, eps, 1.0)

    # dL/dq
    d1_dq = alpha * (
        gamma
        * one_minus_q ** (gamma - 1.0)
        * np.log(q)
        - one_minus_q ** gamma / q
    )

    # d2L/dq2
    d2_dq2 = alpha * (
        -gamma
        * (gamma - 1.0)
        * one_minus_q ** (gamma - 2.0)
        * np.log(q)
        + 2.0
        * gamma
        * one_minus_q ** (gamma - 1.0)
        / q
        + one_minus_q ** gamma / (q ** 2)
    )

    sign = 2.0 * y - 1.0

    dq_dz = sign * p * (1.0 - p)

    d2q_dz2 = (
        sign
        * p
        * (1.0 - p)
        * (1.0 - 2.0 * p)
    )

    grad = d1_dq * dq_dz

    hess_raw = (
        d2_dq2 * (dq_dz ** 2)
        + d1_dq * d2q_dz2
    )

    hess = np.maximum(
        hess_raw,
        HESSIAN_FLOOR,
    )

    clipped = int(np.sum(hess_raw < HESSIAN_FLOOR))

    return grad.astype(np.float64), hess.astype(np.float64), clipped


def focal_loss_objective(
    y_true: np.ndarray,
    predt: np.ndarray,
    sample_weight: Optional[np.ndarray] = None,
):
    """
    Scikit-learn compatible XGBoost custom objective.

    XGBoost 3.x wraps a custom sklearn objective using the signature:
        objective(y_true, y_pred, sample_weight=None)

    Explicit sample weights are used for the class-weighting branch so
    that Pipeline-1's binary scale_pos_weight definition is preserved
    for the custom focal-loss objective.
    """
    grad, hess, clipped = focal_loss_gradient_hessian(
        predt,
        y_true,
        alpha=FOCAL_ALPHA,
        gamma=FOCAL_GAMMA,
    )

    if sample_weight is not None:
        weights = np.asarray(
            sample_weight,
            dtype=np.float64,
        )

        if weights.shape != grad.shape:
            raise ValueError(
                "Focal objective sample_weight shape does not match "
                f"gradient shape: weights={weights.shape}, "
                f"grad={grad.shape}"
            )

        grad *= weights
        hess *= weights

    focal_loss_objective.last_hessian_clipped = clipped

    return grad, hess


focal_loss_objective.last_hessian_clipped = 0


def make_shap_selector() -> XGBClassifier:
    """
    Standard XGBoost selector used only for fold-specific SHAP ranking.

    It is intentionally independent of the focal-loss evaluation models.
    """
    return XGBClassifier(
        **XGB_COMMON_PARAMS,
        objective="binary:logistic",
        eval_metric="logloss",
    )


def make_xgb_classifier(
    weighting: str,
    focal_loss: str,
    y_train: np.ndarray,
) -> Tuple[XGBClassifier, Optional[np.ndarray]]:
    """
    Build one downstream XGBoost model.

    Weighting is calculated from the CURRENT OUTER TRAINING FOLD ONLY.

    For custom focal loss, explicit sample weights are used because they
    reliably propagate into the custom objective. For ordinary logistic
    loss, the same sample weights are used; they are mathematically
    equivalent to Pipeline-1's scale_pos_weight for binary classification.
    """
    sample_weight: Optional[np.ndarray] = None

    if weighting == "ON":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))

        if n_neg == 0 or n_pos == 0:
            raise ValueError(
                "Cannot calculate class weighting with a one-class "
                "training fold."
            )

        positive_weight = n_neg / n_pos

        sample_weight = np.where(
            y_train == 1,
            positive_weight,
            1.0,
        ).astype(np.float64)

    if focal_loss == "ON":
        model = XGBClassifier(
            **XGB_COMMON_PARAMS,
            objective=focal_loss_objective,
            eval_metric="logloss",
        )
    else:
        model = XGBClassifier(
            **XGB_COMMON_PARAMS,
            objective="binary:logistic",
            eval_metric="logloss",
        )

    return model, sample_weight


# ============================================================
# SHAP FEATURE SELECTION
# ============================================================

def coerce_shap_values(shap_values: Any) -> np.ndarray:
    """
    Normalize SHAP output across SHAP versions for binary XGBoost.
    """
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            arr = np.asarray(shap_values[1])
        elif len(shap_values) == 1:
            arr = np.asarray(shap_values[0])
        else:
            raise ValueError(
                f"Unexpected SHAP list length: {len(shap_values)}"
            )
    else:
        arr = np.asarray(shap_values)

    # Newer SHAP versions may return:
    #   (n_samples, n_features, 2)
    if arr.ndim == 3:
        if arr.shape[2] == 2:
            arr = arr[:, :, 1]
        else:
            raise ValueError(
                f"Unexpected 3-D SHAP shape: {arr.shape}"
            )

    if arr.ndim != 2:
        raise ValueError(
            f"Unexpected SHAP output shape: {arr.shape}"
        )

    return arr


def select_top_eeg_features_with_shap(
    X_train_eeg: np.ndarray,
    y_train: np.ndarray,
    feature_names: Sequence[str],
) -> Tuple[Dict[int, np.ndarray], np.ndarray]:
    """
    Leakage boundary:
        ONLY X_train_eeg/y_train enter this function.

    Returns:
        selected indices for Top-50 and Top-75
        mean absolute SHAP importance for all 1330 features
    """
    selector = make_shap_selector()
    selector.fit(X_train_eeg, y_train)

    explainer = shap.TreeExplainer(selector)

    # Validation data are intentionally absent.
    shap_raw = explainer.shap_values(X_train_eeg)
    shap_values = coerce_shap_values(shap_raw)

    if shap_values.shape != X_train_eeg.shape:
        raise ValueError(
            "SHAP shape mismatch: "
            f"SHAP={shap_values.shape}, "
            f"X_train={X_train_eeg.shape}"
        )

    mean_abs_shap = np.mean(
        np.abs(shap_values),
        axis=0,
    )

    feature_indices = np.arange(
        X_train_eeg.shape[1]
    )

    ranking = np.lexsort(
        (
            feature_indices,
            -mean_abs_shap,
        )
    )

    selected: Dict[int, np.ndarray] = {}

    for k in TOP_K_VALUES:
        if k > len(ranking):
            raise ValueError(
                f"Top-{k} requested but only "
                f"{len(ranking)} EEG features exist."
            )

        selected[k] = ranking[:k].copy()

    return selected, mean_abs_shap


# ============================================================
# FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    feature_config: Config,
    X_eeg: np.ndarray,
    X_et: np.ndarray,
    selected_eeg: Dict[int, np.ndarray],
) -> np.ndarray:
    if feature_config.modality == "ET":
        return X_et[:, :EXPECTED_N_ET_FEATURES]

    if feature_config.modality == "EEG":
        return X_eeg[:, selected_eeg[feature_config.eeg_k]]

    if feature_config.modality == "HYBRID":
        eeg_part = X_eeg[
            :, selected_eeg[feature_config.eeg_k]
        ]
        et_part = X_et[
            :, :EXPECTED_N_ET_FEATURES
        ]

        return np.concatenate(
            [eeg_part, et_part],
            axis=1,
        )

    raise ValueError(
        f"Unknown modality: {feature_config.modality}"
    )


# ============================================================
# METRICS
# ============================================================

def calculate_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    tn = int(np.sum(
        (y_true == 0) & (y_pred == 0)
    ))
    fp = int(np.sum(
        (y_true == 0) & (y_pred == 1)
    ))

    denominator = tn + fp

    if denominator == 0:
        return float("nan")

    return tn / denominator


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_probability: np.ndarray,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "f1": float(
            f1_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "kappa": float(
            cohen_kappa_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            recall_score(
                y_true,
                y_pred,
                zero_division=0,
            )
        ),
        "specificity": calculate_specificity(
            y_true,
            y_pred,
        ),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                y_probability,
            )
        )
    else:
        metrics["roc_auc"] = float("nan")

    return metrics


# ============================================================
# SUBJECT EVALUATION
# ============================================================

def evaluate_subject(
    subject: str,
    X_eeg: np.ndarray,
    X_et: np.ndarray,
    y: np.ndarray,
    eeg_feature_names: Sequence[str],
    et_feature_names: Sequence[str],
    configurations: Sequence[Config],
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_binary, label_mapping = make_binary_labels(y)

    class_counts = np.bincount(y_binary)

    if len(class_counts) != 2:
        raise ValueError(
            f"{subject}: expected two binary classes."
        )

    if len(y_binary) < 2:
        raise ValueError(
            f"{subject}: LOOCV requires at least 2 observations."
        )

    if CV_MODE != "LOOCV":
        raise ValueError(
            "evaluate_subject() is intended for CV_MODE='LOOCV'."
        )

    def loocv_splits(n_samples):
        if n_samples < 2:
            raise ValueError(
                f"{subject}: LOOCV requires at least 2 observations."
            )
        all_idx = np.arange(n_samples)
        for held_out in range(n_samples):
            val_idx = np.array([held_out], dtype=np.int64)
            train_idx = np.delete(all_idx, held_out)
            yield train_idx, val_idx

    prediction_store: Dict[
        str,
        Dict[str, List[np.ndarray]]
    ] = {
        config.name: {
            "y_true": [],
            "y_pred": [],
            "y_probability": [],
        }
        for config in configurations
    }

    fold_prediction_rows: List[dict] = []
    shap_rows: List[dict] = []

    fold_audit_rows: List[dict] = []

    for fold_idx, (train_idx, val_idx) in enumerate(
        loocv_splits(len(y_binary)),
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
            "%s | LOOCV fold %04d/%04d | train=%d | validation=%d",
            subject,
            fold_idx,
            len(y_binary),
            len(train_idx),
            len(val_idx),
        )

        # ========================================================
        # LEAKAGE BOUNDARY:
        # SHAP sees ONLY outer-training EEG and labels.
        # ========================================================
        selected_eeg, mean_abs_shap = (
            select_top_eeg_features_with_shap(
                X_train_eeg=X_train_eeg,
                y_train=y_train,
                feature_names=eeg_feature_names,
            )
        )

        feature_indices = np.arange(
            X_train_eeg.shape[1]
        )

        ranking = np.lexsort(
            (
                feature_indices,
                -mean_abs_shap,
            )
        )

        for rank, feature_idx in enumerate(
            ranking,
            start=1,
        ):
            shap_rows.append(
                {
                    "subject": subject,
                    "fold": fold_idx,
                    "feature_rank": rank,
                    "feature_index": int(feature_idx),
                    "feature_name": eeg_feature_names[
                        feature_idx
                    ],
                    "mean_abs_shap": float(
                        mean_abs_shap[feature_idx]
                    ),
                    "selected_top50": bool(rank <= 50),
                    "selected_top75": bool(rank <= 75),
                }
            )

        # ========================================================
        # Fit/evaluate all 20 configurations.
        # ========================================================
        for config in configurations:
            X_train = build_feature_matrix(
                config,
                X_train_eeg,
                X_train_et,
                selected_eeg,
            )

            X_val = build_feature_matrix(
                config,
                X_val_eeg,
                X_val_et,
                selected_eeg,
            )

            model, sample_weight = make_xgb_classifier(
                weighting=config.weighting,
                focal_loss=config.focal_loss,
                y_train=y_train,
            )

            # Training data only.
            if sample_weight is None:
                model.fit(
                    X_train,
                    y_train,
                )
                weighting_ratio = 1.0
            else:
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weight,
                )

                weighting_ratio = (
                    float(
                        np.sum(y_train == 0)
                    )
                    /
                    float(
                        np.sum(y_train == 1)
                    )
                )

            y_probability = np.asarray(
                model.predict_proba(X_val)[:, 1],
                dtype=np.float64,
            )

            # Threshold optimization intentionally removed.
            y_pred = (
                y_probability >= DECISION_THRESHOLD
            ).astype(np.int64)

            prediction_store[
                config.name
            ]["y_true"].append(
                y_val.copy()
            )
            prediction_store[
                config.name
            ]["y_pred"].append(
                y_pred.copy()
            )
            prediction_store[
                config.name
            ]["y_probability"].append(
                y_probability.copy()
            )

            for local_i, global_i in enumerate(val_idx):
                fold_prediction_rows.append(
                    {
                        "cv_mode": CV_MODE,
                        "subject": subject,
                        "fold": fold_idx,
                        "global_sample_index": int(
                            global_i
                        ),
                        "configuration": config.name,
                        "feature_config": config.feature_config,
                        "modality": config.modality,
                        "eeg_features": config.eeg_k,
                        "et_features": config.et_n,
                        "weighting": config.weighting,
                        "focal_loss": config.focal_loss,
                        "focal_alpha": (
                            FOCAL_ALPHA
                            if config.focal_loss == "ON"
                            else np.nan
                        ),
                        "focal_gamma": (
                            FOCAL_GAMMA
                            if config.focal_loss == "ON"
                            else np.nan
                        ),
                        "threshold_mode": "FIXED",
                        "decision_threshold": (
                            DECISION_THRESHOLD
                        ),
                        "y_true": int(y_val[local_i]),
                        "y_probability": float(
                            y_probability[local_i]
                        ),
                        "y_pred": int(y_pred[local_i]),
                    }
                )

            hessian_clipped = 0

            if config.focal_loss == "ON":
                hessian_clipped = int(
                    getattr(
                        focal_loss_objective,
                        "last_hessian_clipped",
                        0,
                    )
                )

            fold_audit_rows.append(
                {
                    "cv_mode": CV_MODE,
                    "subject": subject,
                    "fold": fold_idx,
                    "configuration": config.name,
                    "feature_config": config.feature_config,
                    "weighting": config.weighting,
                    "focal_loss": config.focal_loss,
                    "focal_alpha": (
                        FOCAL_ALPHA
                        if config.focal_loss == "ON"
                        else np.nan
                    ),
                    "focal_gamma": (
                        FOCAL_GAMMA
                        if config.focal_loss == "ON"
                        else np.nan
                    ),
                    "threshold_mode": "FIXED",
                    "decision_threshold": (
                        DECISION_THRESHOLD
                    ),
                    "train_n": len(train_idx),
                    "validation_n": len(val_idx),
                    "train_n_negative": int(
                        np.sum(y_train == 0)
                    ),
                    "train_n_positive": int(
                        np.sum(y_train == 1)
                    ),
                    "scale_pos_weight_equivalent": (
                        weighting_ratio
                        if config.weighting == "ON"
                        else 1.0
                    ),
                    "n_hessian_clipped": hessian_clipped,
                }
            )

        elapsed = time.perf_counter() - fold_start

        logging.info(
            "%s | LOOCV fold %04d complete | %.2f seconds",
            subject,
            fold_idx,
            elapsed,
        )

    # ========================================================
    # Subject-level metrics from concatenated OOF predictions.
    # NOT an average of fold metrics.
    # ========================================================
    subject_metric_rows: List[dict] = []

    for config in configurations:
        y_true = np.concatenate(
            prediction_store[
                config.name
            ]["y_true"]
        )
        y_pred = np.concatenate(
            prediction_store[
                config.name
            ]["y_pred"]
        )
        y_probability = np.concatenate(
            prediction_store[
                config.name
            ]["y_probability"]
        )

        metrics = calculate_metrics(
            y_true,
            y_pred,
            y_probability,
        )

        subject_metric_rows.append(
            {
                "subject": subject,
                "configuration": config.name,
                "feature_config": config.feature_config,
                "modality": config.modality,
                "eeg_features": config.eeg_k,
                "et_features": config.et_n,
                "weighting": config.weighting,
                "focal_loss": config.focal_loss,
                "focal_alpha": (
                    FOCAL_ALPHA
                    if config.focal_loss == "ON"
                    else np.nan
                ),
                "focal_gamma": (
                    FOCAL_GAMMA
                    if config.focal_loss == "ON"
                    else np.nan
                ),
                "threshold_mode": "FIXED",
                "decision_threshold": DECISION_THRESHOLD,
                **metrics,
                "n_samples": len(y_true),
                "label_mapping": json.dumps(
                    label_mapping,
                    sort_keys=True,
                ),
                "class_0_count": int(
                    np.sum(y_binary == 0)
                ),
                "class_1_count": int(
                    np.sum(y_binary == 1)
                ),
            }
        )

    subject_metrics = pd.DataFrame(
        subject_metric_rows
    )

    fold_predictions = pd.DataFrame(
        fold_prediction_rows
    )

    shap_ranking = pd.DataFrame(
        shap_rows
    )

    fold_audit = pd.DataFrame(
        fold_audit_rows
    )

    # Per-subject exports.
    subject_dir = output_dir / "subject_results"
    prediction_dir = output_dir / "predictions"
    feature_dir = output_dir / "feature_selection"
    audit_dir = output_dir / "fold_audits"

    for directory in (
        subject_dir,
        prediction_dir,
        feature_dir,
        audit_dir,
    ):
        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    subject_metrics.to_csv(
        subject_dir / f"{subject}_metrics.csv",
        index=False,
    )

    fold_predictions.to_csv(
        prediction_dir
        / f"{subject}_fold_predictions.csv",
        index=False,
    )

    shap_ranking.to_csv(
        feature_dir
        / f"{subject}_fold_shap_ranking.csv",
        index=False,
    )

    fold_audit.to_csv(
        audit_dir
        / f"{subject}_fold_audit.csv",
        index=False,
    )

    return (
        subject_metrics,
        fold_predictions,
        shap_ranking,
    )


def evaluate_loso(
    subjects,
    loaded_subjects,
    eeg_feature_names,
    et_feature_names,
    configurations,
    output_dir,
):
    """
    Leave-One-Subject-Out evaluation.

    One complete subject is the validation set per outer fold.
    All SHAP selection, class weighting, and model fitting use only
    the remaining training subjects.
    """
    if CV_MODE != "LOSO":
        raise ValueError(
            "evaluate_loso() is intended for CV_MODE='LOSO'."
        )

    available = [s for s in subjects if s in loaded_subjects]
    if len(available) < 2:
        raise ValueError("LOSO requires at least two subjects.")

    prediction_rows = []
    shap_rows = []
    audit_rows = []
    subject_metric_rows = []

    for fold_idx, heldout_subject in enumerate(available, start=1):
        fold_start = time.perf_counter()

        train_subjects = [
            s for s in available if s != heldout_subject
        ]

        X_train_eeg = np.concatenate(
            [loaded_subjects[s][0] for s in train_subjects], axis=0
        )
        X_train_et = np.concatenate(
            [loaded_subjects[s][1] for s in train_subjects], axis=0
        )

        y_train_raw = np.concatenate(
            [loaded_subjects[s][2] for s in train_subjects], axis=0
        )

        X_val_eeg, X_val_et, y_val_raw = loaded_subjects[heldout_subject]

        # Establish ONE deterministic class mapping for the complete
        # available dataset, then apply that mapping to both training
        # and held-out subjects. This prevents train/validation class
        # semantics from being independently reversed.
        combined_y_raw = np.concatenate(
            [loaded_subjects[s][2] for s in available],
            axis=0,
        )
        _, global_label_mapping = make_binary_labels(combined_y_raw)

        y_train = np.array(
            [global_label_mapping[str(value)] for value in y_train_raw],
            dtype=np.int64,
        )
        y_val = np.array(
            [global_label_mapping[str(value)] for value in y_val_raw],
            dtype=np.int64,
        )
        label_mapping = global_label_mapping

        if len(np.unique(y_train)) != 2:
            raise ValueError(
                f"LOSO fold {fold_idx} ({heldout_subject}) has "
                "only one class in training data."
            )

        logging.info(
            "LOSO fold %02d/%02d | held-out=%s | train_subjects=%d | "
            "train_samples=%d | validation_samples=%d",
            fold_idx,
            len(available),
            heldout_subject,
            len(train_subjects),
            len(y_train),
            len(y_val),
        )

        # Training-only SHAP selection.
        selected_eeg, mean_abs_shap = (
            select_top_eeg_features_with_shap(
                X_train_eeg=X_train_eeg,
                y_train=y_train,
                feature_names=eeg_feature_names,
            )
        )

        ranking = np.lexsort(
            (
                np.arange(X_train_eeg.shape[1]),
                -mean_abs_shap,
            )
        )

        for rank, feature_idx in enumerate(ranking, start=1):
            shap_rows.append(
                {
                    "subject": heldout_subject,
                    "fold": fold_idx,
                    "feature_rank": rank,
                    "feature_index": int(feature_idx),
                    "feature_name": eeg_feature_names[feature_idx],
                    "mean_abs_shap": float(mean_abs_shap[feature_idx]),
                    "selected_top50": bool(rank <= 50),
                    "selected_top75": bool(rank <= 75),
                }
            )

        # Each held-out subject is predicted once for every configuration.
        for config in configurations:
            X_train = build_feature_matrix(
                config, X_train_eeg, X_train_et, selected_eeg
            )
            X_val = build_feature_matrix(
                config, X_val_eeg, X_val_et, selected_eeg
            )

            model, sample_weight = make_xgb_classifier(
                weighting=config.weighting,
                focal_loss=config.focal_loss,
                y_train=y_train,
            )

            if sample_weight is None:
                model.fit(X_train, y_train)
                weighting_ratio = 1.0
            else:
                model.fit(
                    X_train,
                    y_train,
                    sample_weight=sample_weight,
                )
                weighting_ratio = (
                    float(np.sum(y_train == 0))
                    / float(np.sum(y_train == 1))
                )

            y_probability = np.asarray(
                model.predict_proba(X_val)[:, 1],
                dtype=np.float64,
            )
            y_pred = (
                y_probability >= DECISION_THRESHOLD
            ).astype(np.int64)

            for local_i in range(len(y_val)):
                prediction_rows.append(
                    {
                        "cv_mode": CV_MODE,
                        "subject": heldout_subject,
                        "fold": fold_idx,
                        "global_sample_index": int(local_i),
                        "configuration": config.name,
                        "feature_config": config.feature_config,
                        "modality": config.modality,
                        "eeg_features": config.eeg_k,
                        "et_features": config.et_n,
                        "weighting": config.weighting,
                        "focal_loss": config.focal_loss,
                        "focal_alpha": (
                            FOCAL_ALPHA
                            if config.focal_loss == "ON"
                            else np.nan
                        ),
                        "focal_gamma": (
                            FOCAL_GAMMA
                            if config.focal_loss == "ON"
                            else np.nan
                        ),
                        "threshold_mode": "FIXED",
                        "decision_threshold": DECISION_THRESHOLD,
                        "y_true": int(y_val[local_i]),
                        "y_probability": float(y_probability[local_i]),
                        "y_pred": int(y_pred[local_i]),
                    }
                )

            hessian_clipped = 0
            if config.focal_loss == "ON":
                hessian_clipped = int(
                    getattr(
                        focal_loss_objective,
                        "last_hessian_clipped",
                        0,
                    )
                )

            audit_rows.append(
                {
                    "cv_mode": CV_MODE,
                    "subject": heldout_subject,
                    "fold": fold_idx,
                    "configuration": config.name,
                    "feature_config": config.feature_config,
                    "weighting": config.weighting,
                    "focal_loss": config.focal_loss,
                    "focal_alpha": (
                        FOCAL_ALPHA
                        if config.focal_loss == "ON"
                        else np.nan
                    ),
                    "focal_gamma": (
                        FOCAL_GAMMA
                        if config.focal_loss == "ON"
                        else np.nan
                    ),
                    "threshold_mode": "FIXED",
                    "decision_threshold": DECISION_THRESHOLD,
                    "train_n": len(y_train),
                    "validation_n": len(y_val),
                    "train_n_negative": int(np.sum(y_train == 0)),
                    "train_n_positive": int(np.sum(y_train == 1)),
                    "scale_pos_weight_equivalent": (
                        weighting_ratio
                        if config.weighting == "ON"
                        else 1.0
                    ),
                    "n_hessian_clipped": hessian_clipped,
                }
            )

        # Calculate subject-level metrics from this held-out subject's
        # complete OOF predictions, exactly as the original subject-level
        # aggregation principle, but now across one LOSO fold.
        fold_predictions = pd.DataFrame(
            [
                r for r in prediction_rows
                if r["subject"] == heldout_subject
            ]
        )

        for config in configurations:
            d = fold_predictions[
                fold_predictions["configuration"] == config.name
            ]

            y_true = d["y_true"].to_numpy(dtype=np.int64)
            y_pred = d["y_pred"].to_numpy(dtype=np.int64)
            y_probability = d["y_probability"].to_numpy(dtype=np.float64)

            metrics = calculate_metrics(
                y_true,
                y_pred,
                y_probability,
            )

            subject_metric_rows.append(
                {
                    "subject": heldout_subject,
                    "configuration": config.name,
                    "feature_config": config.feature_config,
                    "modality": config.modality,
                    "eeg_features": config.eeg_k,
                    "et_features": config.et_n,
                    "weighting": config.weighting,
                    "focal_loss": config.focal_loss,
                    "focal_alpha": (
                        FOCAL_ALPHA
                        if config.focal_loss == "ON"
                        else np.nan
                    ),
                    "focal_gamma": (
                        FOCAL_GAMMA
                        if config.focal_loss == "ON"
                        else np.nan
                    ),
                    "threshold_mode": "FIXED",
                    "decision_threshold": DECISION_THRESHOLD,
                    **metrics,
                    "n_samples": len(y_true),
                    "label_mapping": json.dumps(
                        label_mapping,
                        sort_keys=True,
                    ),
                    "class_0_count": int(np.sum(y_val == 0)),
                    "class_1_count": int(np.sum(y_val == 1)),
                }
            )

        logging.info(
            "LOSO fold %02d complete | held-out=%s | %.2f minutes",
            fold_idx,
            heldout_subject,
            (time.perf_counter() - fold_start) / 60.0,
        )

    subject_metrics = pd.DataFrame(subject_metric_rows)
    predictions_df = pd.DataFrame(prediction_rows)
    shap_ranking = pd.DataFrame(shap_rows)
    fold_audit = pd.DataFrame(audit_rows)

    subject_dir = output_dir / "subject_results"
    prediction_dir = output_dir / "predictions"
    feature_dir = output_dir / "feature_selection"
    audit_dir = output_dir / "fold_audits"

    for directory in (
        subject_dir,
        prediction_dir,
        feature_dir,
        audit_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    subject_metrics.to_csv(
        subject_dir / "all_subject_loso_metrics.csv",
        index=False,
    )
    predictions_df.to_csv(
        prediction_dir / "all_loso_predictions.csv",
        index=False,
    )
    shap_ranking.to_csv(
        feature_dir / "all_loso_shap_ranking.csv",
        index=False,
    )
    fold_audit.to_csv(
        audit_dir / "all_loso_fold_audit.csv",
        index=False,
    )

    return (
        subject_metrics,
        predictions_df,
        shap_ranking,
        fold_audit,
    )


# ============================================================
# SUMMARY
# ============================================================

def build_configuration_summary(
    all_subject_metrics: pd.DataFrame,
) -> pd.DataFrame:
    group_cols = [
        "configuration",
        "feature_config",
        "modality",
        "eeg_features",
        "et_features",
        "weighting",
        "focal_loss",
        "focal_alpha",
        "focal_gamma",
        "threshold_mode",
        "decision_threshold",
    ]

    summary_rows: List[dict] = []

    for keys, group in all_subject_metrics.groupby(
        group_cols,
        dropna=False,
    ):
        row = dict(
            zip(group_cols, keys)
        )

        row["subjects_evaluated"] = int(
            group["subject"].nunique()
        )

        for metric in (
            "accuracy",
            "f1",
            "kappa",
            "balanced_accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "roc_auc",
        ):
            row[f"mean_{metric}"] = float(
                group[metric].mean()
            )
            row[f"std_{metric}"] = float(
                group[metric].std(
                    ddof=1
                )
            )

            row[f"median_{metric}"] = float(
                group[metric].median()
            )

        summary_rows.append(row)

    summary = pd.DataFrame(
        summary_rows
    )

    if not summary.empty:
        summary = summary.sort_values(
            "mean_f1",
            ascending=False,
        ).reset_index(drop=True)

        summary.insert(
            0,
            "f1_rank",
            np.arange(
                1,
                len(summary) + 1,
            ),
        )

    return summary


# ============================================================
# RUN SPECIFICATION
# ============================================================

def write_run_specification(
    output_dir: Path,
    data_dir: Path,
    subjects: Sequence[str],
    configurations: Sequence[Config],
) -> None:
    spec = {
        "pipeline": "PT_1 Pipeline 2",
        "timestamp": time.strftime(
            "%Y-%m-%d %H:%M:%S"
        ),
        "python_version": sys.version,
        "platform": platform.platform(),
        "xgboost_version": getattr(
            xgb,
            "__version__",
            "unknown",
        ),
        "shap_version": getattr(
            shap,
            "__version__",
            "unknown",
        ),
        "data_dir": str(data_dir),
        "output_dir": str(output_dir),
        "expected_subject_count": EXPECTED_SUBJECT_COUNT,
        "subjects_discovered": list(subjects),
        "subjects_discovered_count": len(subjects),
        "cv": {
            "mode": CV_MODE,
            "random_state": RANDOM_STATE,
            "LOOCV": {
                "description": "Leave-One-Observation-Out within each subject",
                "fold_count": "number_of_observations_per_subject",
            },
            "LOSO": {
                "description": "Leave-One-Subject-Out across subjects",
                "fold_count": "number_of_available_subjects",
            },
        },
        "eeg": {
            "channels": EEG_CHANNELS,
            "n_channels": N_EEG_CHANNELS,
            "n_imfs": N_IMFS,
            "base_features": EEG_FEATURES,
            "total_features": EXPECTED_N_EEG_FEATURES,
            "shap_selector": "XGBoost",
            "shap_data": "outer training fold only",
            "top_k": list(TOP_K_VALUES),
        },
        "et": {
            "features": ET_FEATURES,
            "total_features": EXPECTED_N_ET_FEATURES,
            "shap_selection": False,
        },
        "xgboost": XGB_COMMON_PARAMS,
        "class_weighting": {
            "states": ["OFF", "ON"],
            "on_definition": (
                "n_negative / n_positive from current outer "
                "training fold"
            ),
            "implementation": (
                "training sample weights; mathematically "
                "equivalent to binary scale_pos_weight"
            ),
        },
        "focal_loss": {
            "states": ["OFF", "ON"],
            "alpha": FOCAL_ALPHA,
            "gamma": FOCAL_GAMMA,
            "hessian_floor": HESSIAN_FLOOR,
        },
        "threshold": {
            "optimization": False,
            "mode": "FIXED",
            "value": DECISION_THRESHOLD,
        },
        "configurations": [
            asdict(config)
            for config in configurations
        ],
        "configuration_count": len(
            configurations
        ),
        "metrics": [
            "accuracy",
            "f1",
            "kappa",
            "balanced_accuracy",
            "precision",
            "recall_sensitivity",
            "specificity",
            "roc_auc",
        ],
        "leakage_controls": [
            "requested LOOCV/LOSO outer CV",
            "fold-specific SHAP selector",
            "SHAP calculated on training fold only",
            "class weights calculated from training fold only",
            "validation data used only for prediction",
            "fixed threshold 0.50",
            "subject metrics calculated from concatenated OOF predictions",
        ],
    }

    with (
        output_dir
        / "run_specification.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            spec,
            f,
            indent=2,
        )


# ============================================================
# MAIN
# ============================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "PT_1 Pipeline 2: leakage-safe XGBoost experiment "
            "with configurable LOOCV or LOSO."
        )
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing EEG/ET/y NPY files and metadata.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where Pipeline-2 results are written.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = args.data_dir
    output_dir = args.output_dir

    setup_logging(output_dir)

    if CV_MODE not in {"LOOCV", "LOSO"}:
        raise ValueError(
            f"Unsupported CV_MODE={CV_MODE!r}. "
            "Use 'LOOCV' or 'LOSO'."
        )

    logging.info("=" * 72)
    logging.info("PT_1 XGBOOST CROSS-VALIDATION START")
    logging.info("=" * 72)
    logging.info("CV mode: %s", CV_MODE)

    configurations = build_configurations()

    eeg_feature_names, et_feature_names = (
        load_feature_metadata(data_dir)
    )

    subjects = load_subject_ids(data_dir)

    logging.info("Subjects discovered: %d", len(subjects))

    if len(subjects) != EXPECTED_SUBJECT_COUNT:
        logging.warning(
            "Expected %d subjects, but discovered %d. "
            "The pipeline will process the subjects actually found.",
            EXPECTED_SUBJECT_COUNT,
            len(subjects),
        )

    write_run_specification(
        output_dir=output_dir,
        data_dir=data_dir,
        subjects=subjects,
        configurations=configurations,
    )

    loaded_subjects = {}
    failed_rows = []

    for subject in subjects:
        try:
            loaded_subjects[subject] = load_subject_arrays(
                data_dir,
                subject,
            )
        except Exception as exc:
            logging.error(
                "%s | LOAD FAILED | %s",
                subject,
                str(exc),
            )
            failed_rows.append(
                {
                    "subject": subject,
                    "stage": "loading",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )

    if not loaded_subjects:
        raise RuntimeError(
            "No subjects could be loaded successfully."
        )

    if CV_MODE == "LOSO":
        (
            all_metrics_df,
            all_predictions_df,
            all_shap_df,
            all_audit_df,
        ) = evaluate_loso(
            subjects=subjects,
            loaded_subjects=loaded_subjects,
            eeg_feature_names=eeg_feature_names,
            et_feature_names=et_feature_names,
            configurations=configurations,
            output_dir=output_dir,
        )
    else:
        all_subject_metrics = []
        all_predictions = []
        all_shap = []
        all_audits = []

        for subject in subjects:
            if subject not in loaded_subjects:
                continue

            try:
                X_eeg, X_et, y = loaded_subjects[subject]

                (
                    subject_metrics,
                    predictions,
                    shap_ranking,
                ) = evaluate_subject(
                    subject=subject,
                    X_eeg=X_eeg,
                    X_et=X_et,
                    y=y,
                    eeg_feature_names=eeg_feature_names,
                    et_feature_names=et_feature_names,
                    configurations=configurations,
                    output_dir=output_dir,
                )

                all_subject_metrics.append(subject_metrics)
                all_predictions.append(predictions)
                all_shap.append(shap_ranking)

                audit_path = (
                    output_dir
                    / "fold_audits"
                    / f"{subject}_fold_audit.csv"
                )
                if audit_path.exists():
                    all_audits.append(
                        pd.read_csv(audit_path)
                    )

            except Exception as exc:
                logging.error(
                    "%s | LOOCV FAILED | %s",
                    subject,
                    str(exc),
                )
                failed_rows.append(
                    {
                        "subject": subject,
                        "stage": "LOOCV_evaluation",
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    }
                )

        if not all_subject_metrics:
            raise RuntimeError(
                "LOOCV produced no successful subject results."
            )

        all_metrics_df = pd.concat(
            all_subject_metrics,
            ignore_index=True,
        )
        all_predictions_df = pd.concat(
            all_predictions,
            ignore_index=True,
        )
        all_shap_df = pd.concat(
            all_shap,
            ignore_index=True,
        )
        all_audit_df = (
            pd.concat(all_audits, ignore_index=True)
            if all_audits
            else pd.DataFrame()
        )

    summaries_dir = output_dir / "summaries"
    summaries_dir.mkdir(parents=True, exist_ok=True)

    config_summary = build_configuration_summary(
        all_metrics_df
    )

    all_metrics_df.to_csv(
        summaries_dir / "all_subject_metrics.csv",
        index=False,
    )
    config_summary.to_csv(
        summaries_dir / "configuration_summary_20.csv",
        index=False,
    )
    all_predictions_df.to_csv(
        summaries_dir / "all_fold_predictions.csv",
        index=False,
    )
    all_shap_df.to_csv(
        summaries_dir / "all_fold_shap_ranking.csv",
        index=False,
    )

    if not all_audit_df.empty:
        all_audit_df.to_csv(
            summaries_dir / "all_fold_audit.csv",
            index=False,
        )

    failed_df = pd.DataFrame(failed_rows)
    if failed_df.empty:
        failed_df = pd.DataFrame(
            columns=[
                "subject",
                "stage",
                "error_type",
                "error_message",
            ]
        )

    failed_df.to_csv(
        summaries_dir / "failed_subjects.csv",
        index=False,
    )

    excel_path = (
        output_dir / f"PT1_Pipeline_{CV_MODE}_results.xlsx"
    )

    with pd.ExcelWriter(
        excel_path,
        engine="openpyxl",
    ) as writer:
        config_summary.to_excel(
            writer,
            sheet_name="Config_Summary",
            index=False,
        )
        all_metrics_df.to_excel(
            writer,
            sheet_name="Subject_Metrics",
            index=False,
        )
        all_predictions_df.to_excel(
            writer,
            sheet_name="Fold_Predictions",
            index=False,
        )
        all_shap_df.to_excel(
            writer,
            sheet_name="SHAP_Ranking",
            index=False,
        )
        if not all_audit_df.empty:
            all_audit_df.to_excel(
                writer,
                sheet_name="Fold_Audits",
                index=False,
            )
        failed_df.to_excel(
            writer,
            sheet_name="Failed_Subjects",
            index=False,
        )

    logging.info("=" * 72)
    logging.info("PT_1 XGBOOST CROSS-VALIDATION COMPLETE")
    logging.info("=" * 72)
    logging.info(
        "CV mode: %s | subjects evaluated: %d | configurations: %d",
        CV_MODE,
        all_metrics_df["subject"].nunique(),
        len(configurations),
    )

    if not config_summary.empty:
        best_f1 = config_summary.loc[
            config_summary["mean_f1"].idxmax()
        ]
        best_accuracy = config_summary.loc[
            config_summary["mean_accuracy"].idxmax()
        ]
        best_kappa = config_summary.loc[
            config_summary["mean_kappa"].idxmax()
        ]

        logging.info(
            "Best mean F1: %s | %.4f",
            best_f1["configuration"],
            best_f1["mean_f1"],
        )
        logging.info(
            "Best mean Accuracy: %s | %.4f",
            best_accuracy["configuration"],
            best_accuracy["mean_accuracy"],
        )
        logging.info(
            "Best mean Kappa: %s | %.4f",
            best_kappa["configuration"],
            best_kappa["mean_kappa"],
        )

    logging.info("Results directory: %s", output_dir)
    logging.info("Excel export: %s", excel_path)


if __name__ == "__main__":
    main()
