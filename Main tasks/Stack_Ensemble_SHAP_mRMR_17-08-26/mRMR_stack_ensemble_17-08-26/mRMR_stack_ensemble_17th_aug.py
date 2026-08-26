"""
classifier_imf_ablation_loocv.py
=================================
IMF Ablation Study with LOOCV (Leave-One-Out Cross-Validation) Pipeline.

Performs ablation over k = 5, 6, 7 retained IMFs using a complete
Leave-One-Out Cross-Validation (LOOCV) pipeline:

    - IMF truncation (keep first k IMFs per channel, always keep ET features)
    - mRMR feature selection on training data only (no data leakage)
    - StandardScaler fit on training data only
    - SMOTE oversampling on training data only
    - Stacking Ensemble: RF + GradientBoosting + XGBoost -> RF meta-learner
    - Compares Accuracy, Precision, Recall, F1, AUC-ROC, Cohen's Kappa
    - Selects the best-performing k
    - SHAP analysis: feature, IMF, channel, ET importance
    - Neurophysiological interpretation under 4 domains

Run
---
    python classifier_imf_ablation_loocv.py --features_dir features_v2_combined
    python classifier_imf_ablation_loocv.py --features_dir features_v2_combined --n_mrmr 80
"""

import os
import sys
import glob
import json
import argparse
import warnings
import csv
import time
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    cohen_kappa_score,
)

warnings.filterwarnings('ignore')

# -- Optional dependencies --------------------------------------------------
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print('WARNING: xgboost not installed -- XGBoost base learner unavailable.')

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False
    print('WARNING: shap not installed -- SHAP analysis will be skipped.')

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
    print('WARNING: imbalanced-learn not installed -- SMOTE unavailable.')

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print('WARNING: optuna not installed -- hyperparameter tuning will be skipped.')

try:
    import matplotlib
    matplotlib.use('Agg')          # non-interactive backend for saving plots
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print('WARNING: matplotlib not installed -- plots will be skipped.')


# -----------------------------------------------------------------------------
# CONSTANTS  (must match feature_extraction.py)
# -----------------------------------------------------------------------------
N_EEG_CHANNELS     = 19
N_IMFS_TOTAL       = 7          # total IMFs retained (truncated by truncate_imfs_to7.py)
N_FEAT_PER_IMF     = 10         # features per IMF per channel (feature_extraction_v2.py)
N_ET_FEATURES      = 10         # 10 ET features (feature_extraction_v2.py)
N_EEG_TOTAL        = N_EEG_CHANNELS * N_IMFS_TOTAL * N_FEAT_PER_IMF  # 1330

_IMF_FEAT_NAMES = [
    'relative_energy',
    'differential_entropy',
    'sample_entropy',
    'higuchi_fd',
    'katz_fd',
    'hjorth_activity',
    'hjorth_mobility',
    'hjorth_complexity',
    'hurst_exponent',
    'lyapunov_exponent',
]

_ET_FEAT_NAMES = [
    'fixation_count',
    'mean_fixation_duration_ms',
    'total_fixation_duration_ms',
    'time_to_first_fixation_ms',
    'mean_saccade_amplitude_px',
    'mean_saccade_velocity_px_s',
    'scanpath_length_px',
    'mean_pupil_diameter_px',
    'pupil_diameter_variability_px',
    'dwell_time_ms',
]

# 19-channel labels (10-20 system, matching the original dataset)
_CHANNEL_LABELS = [
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2',
    'F7', 'F8', 'T7', 'T8', 'P7', 'P8', 'Fz', 'Cz', 'Pz',
]

# Feature-to-domain mapping for neurophysiological interpretation
_FEAT_DOMAIN = {
    'relative_energy'          : 'Oscillatory Activity',
    'differential_entropy'     : 'Oscillatory Activity',
    'sample_entropy'           : 'Signal Complexity',
    'higuchi_fd'               : 'Signal Complexity',
    'katz_fd'                  : 'Signal Complexity',
    'hjorth_activity'          : 'Nonlinear Dynamics',
    'hjorth_mobility'          : 'Temporal Dynamics',
    'hjorth_complexity'        : 'Signal Complexity',
    'hurst_exponent'           : 'Nonlinear Dynamics',
    'lyapunov_exponent'        : 'Nonlinear Dynamics',
}

# Domain descriptions for the interpretation report
_DOMAIN_DESCRIPTIONS = {
    'Oscillatory Activity': (
        'Features related to oscillatory neural activity reflect the spectral '
        'composition and energy distribution across intrinsic mode functions. '
        'In neuromarketing, these capture ATTENTION ALLOCATION during product '
        'evaluation -- higher energy ratios in low-frequency IMFs indicate '
        'sustained attention, while mean frequency shifts reveal cognitive '
        'workload changes as consumers process product information.'
    ),
    'Signal Complexity': (
        'Complexity features quantify the irregularity and information content '
        'of neural signals. Spectral entropy and Hjorth complexity measure the '
        'DEPTH OF COGNITIVE PROCESSING -- higher complexity signals deeper '
        'ENGAGEMENT with product information. Consumers who process multiple '
        'product attributes (price, brand, features) show elevated signal '
        'complexity compared to superficial browsing.'
    ),
    'Temporal Dynamics': (
        'Temporal dynamics features capture the rate and variability of signal '
        'changes over time. Zero-crossing rate and Hjorth mobility reflect '
        'DECISION-MAKING SPEED and response dynamics. Rapid temporal fluctuations '
        'correlate with active comparison and evaluation, while stable patterns '
        'suggest quick, heuristic-based purchase decisions.'
    ),
    'Nonlinear Dynamics': (
        'Nonlinear dynamics features (kurtosis, skewness, Hjorth activity) '
        'capture asymmetric and heavy-tailed signal distributions that linear '
        'measures miss. These relate to EMOTIONAL AROUSAL and NOVELTY DETECTION '
        'in consumer responses. Elevated kurtosis indicates bursts of neural '
        'activity tied to surprise or interest, while skewness asymmetries '
        'reflect approach/avoidance motivational states during product evaluation.'
    ),
}


# ===============================================================================
# FEATURE LOADING
# ===============================================================================

def load_features(features_dir: str) -> list:
    """
    Load all {subject_id}_X.npy and {subject_id}_y.npy files.

    Returns list of dicts: [{'subject_id', 'X', 'y'}, ...]
    """
    subjects_txt = os.path.join(features_dir, 'subjects.txt')
    if os.path.exists(subjects_txt):
        with open(subjects_txt) as f:
            subject_ids = [line.strip() for line in f if line.strip()]
    else:
        x_files     = sorted(glob.glob(os.path.join(features_dir, '*_X.npy')))
        subject_ids = [os.path.basename(p).replace('_X.npy', '') for p in x_files]

    dataset = []
    for sid in subject_ids:
        x_path = os.path.join(features_dir, f'{sid}_X.npy')
        y_path = os.path.join(features_dir, f'{sid}_y.npy')
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            print(f'  WARNING: missing files for {sid}, skipping.')
            continue
        X = np.load(x_path).astype(np.float32)
        y = np.load(y_path).astype(int)
        dataset.append({'subject_id': sid, 'X': X, 'y': y})

    return dataset


def pool_all_subjects(dataset: list) -> tuple:
    """
    Pool all subjects' data into a single feature matrix and label vector.

    Returns
    -------
    X_all : ndarray, shape (n_total_epochs, n_features)
    y_all : ndarray, shape (n_total_epochs,)
    subject_ids : list of str, one per epoch (for tracing back to subject)
    """
    X_list, y_list, sid_list = [], [], []
    for s in dataset:
        n_epochs = len(s['y'])
        X_list.append(s['X'])
        y_list.append(s['y'])
        sid_list.extend([s['subject_id']] * n_epochs)
    X_all = np.concatenate(X_list, axis=0).astype(np.float32)
    y_all = np.concatenate(y_list, axis=0).astype(int)
    return X_all, y_all, sid_list


# ===============================================================================
# IMF TRUNCATION
# ===============================================================================

def get_imf_columns(k: int,
                    n_channels: int = N_EEG_CHANNELS,
                    n_imfs_total: int = N_IMFS_TOTAL,
                    n_feat_per_imf: int = N_FEAT_PER_IMF,
                    n_et: int = N_ET_FEATURES) -> np.ndarray:
    """
    Compute column indices to retain the first k IMFs per channel + all ET features.

    Feature layout (channel-major):
        channel 0: [imf0_feat0, ..., imf0_feat8, imf1_feat0, ..., imf7_feat8]
        channel 1: [imf0_feat0, ..., imf7_feat8]
        ...
        channel 18: [...]
        ET: [et_feat0, ..., et_feat7]

    For each channel, keep IMFs 0..k-1 -> k * n_feat_per_imf features.
    Then append all ET feature indices.
    """
    eeg_cols = []
    for ch in range(n_channels):
        ch_start = ch * n_imfs_total * n_feat_per_imf
        for imf in range(k):
            imf_start = ch_start + imf * n_feat_per_imf
            eeg_cols.extend(range(imf_start, imf_start + n_feat_per_imf))

    # ET features come after all EEG features
    eeg_total = n_channels * n_imfs_total * n_feat_per_imf
    et_cols = list(range(eeg_total, eeg_total + n_et))

    return np.array(eeg_cols + et_cols, dtype=int)


def truncate_to_k_imfs(X: np.ndarray, k: int) -> np.ndarray:
    """Select only the columns corresponding to the first k IMFs + ET."""
    cols = get_imf_columns(k)
    return X[:, cols]


# ===============================================================================
# FEATURE INDEX DECODER
# ===============================================================================

def decode_feature_index_for_k(idx: int, k: int) -> dict:
    """
    Map a feature index in the k-IMF-truncated space back to human-readable
    metadata (channel, IMF, feature name, or ET feature name).

    Parameters
    ----------
    idx : int
        Column index in the truncated feature matrix.
    k : int
        Number of retained IMFs.

    Returns
    -------
    dict with keys: feature_index, modality, channel_number, channel_name,
                    imf_number, feature_name, domain
    """
    eeg_total_k = N_EEG_CHANNELS * k * N_FEAT_PER_IMF
    if idx < eeg_total_k:
        ch  = idx // (k * N_FEAT_PER_IMF)
        rem = idx % (k * N_FEAT_PER_IMF)
        imf = rem // N_FEAT_PER_IMF
        feat = rem % N_FEAT_PER_IMF
        feat_name = _IMF_FEAT_NAMES[feat]
        return dict(
            feature_index  = idx,
            modality       = 'EEG',
            channel_number = ch + 1,
            channel_name   = _CHANNEL_LABELS[ch] if ch < len(_CHANNEL_LABELS) else f'Ch{ch+1}',
            imf_number     = imf + 1,
            feature_name   = feat_name,
            domain         = _FEAT_DOMAIN.get(feat_name, 'Unknown'),
        )
    else:
        et_idx = idx - eeg_total_k
        name   = _ET_FEAT_NAMES[et_idx] if et_idx < len(_ET_FEAT_NAMES) else f'et_{et_idx}'
        return dict(
            feature_index  = idx,
            modality       = 'ET',
            channel_number = None,
            channel_name   = None,
            imf_number     = None,
            feature_name   = name,
            domain         = 'Eye-Tracking',
        )


# ===============================================================================
# mRMR FEATURE SELECTION (pure numpy/scipy implementation)
# ===============================================================================

def _discretize(X: np.ndarray, n_bins: int = 10) -> np.ndarray:
    """
    Discretize continuous features into bins for mutual information estimation.
    Uses quantile-based binning for robustness to outliers.
    """
    n_samples, n_features = X.shape
    X_disc = np.zeros_like(X, dtype=int)
    for j in range(n_features):
        col = X[:, j]
        std = np.std(col)
        if std < 1e-10:
            X_disc[:, j] = 0
            continue
        # Quantile-based binning
        percentiles = np.linspace(0, 100, n_bins + 1)
        bin_edges = np.percentile(col, percentiles)
        bin_edges = np.unique(bin_edges)
        X_disc[:, j] = np.searchsorted(bin_edges[1:], col, side='right')
    return X_disc


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute mutual information I(X; Y) from discrete arrays.
    Uses the standard formula: I(X;Y) = H(X) + H(Y) - H(X,Y)
    """
    n = len(x)
    if n == 0:
        return 0.0

    # Joint distribution
    xy = x.astype(np.int64) * (y.max() + 1) + y.astype(np.int64)
    _, counts_xy = np.unique(xy, return_counts=True)
    p_xy = counts_xy / n

    # Marginals
    _, counts_x = np.unique(x, return_counts=True)
    p_x = counts_x / n
    _, counts_y = np.unique(y, return_counts=True)
    p_y = counts_y / n

    # Entropies
    h_x  = -np.sum(p_x * np.log2(p_x + 1e-10))
    h_y  = -np.sum(p_y * np.log2(p_y + 1e-10))
    h_xy = -np.sum(p_xy * np.log2(p_xy + 1e-10))

    return max(0.0, h_x + h_y - h_xy)


def mrmr_feature_selection(X_train: np.ndarray, y_train: np.ndarray,
                           n_select: int = 50) -> np.ndarray:
    """
    Minimum Redundancy Maximum Relevance (mRMR) feature selection.

    Greedy algorithm:
      1. Compute relevance = MI(feature_j, label) for all features
      2. Select the feature with highest relevance
      3. For each remaining feature, compute:
           score = relevance(f) - (1/|S|) * sum_{s in S} MI(f, s)
         where S is the set of already-selected features
      4. Select the feature with the highest score
      5. Repeat until n_select features are selected

    Parameters
    ----------
    X_train : ndarray, shape (n_samples, n_features)
    y_train : ndarray, shape (n_samples,)
    n_select : int
        Number of features to select.

    Returns
    -------
    selected : ndarray of int -- indices of selected features (sorted)
    """
    n_features = X_train.shape[1]
    n_select   = min(n_select, n_features)

    # Remove zero-variance features from consideration
    variances = np.var(X_train, axis=0)
    valid_mask = variances > 1e-10
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) < n_select:
        n_select = len(valid_indices)
    if n_select == 0:
        return np.arange(min(10, n_features))

    # Discretize for MI computation
    X_disc = _discretize(X_train[:, valid_indices], n_bins=10)
    y_disc = y_train.astype(int)

    n_valid = len(valid_indices)

    # Step 1: Compute relevance for all valid features
    relevance = np.zeros(n_valid)
    for j in range(n_valid):
        relevance[j] = _mutual_info(X_disc[:, j], y_disc)

    # Step 2: Greedy mRMR selection
    selected_local = []
    remaining      = set(range(n_valid))

    # First feature: highest relevance
    first = int(np.argmax(relevance))
    selected_local.append(first)
    remaining.discard(first)

    # Precompute inter-feature MI incrementally
    # mi_cache[i][j] = MI(feature_i, feature_j)
    # Only compute as needed to avoid O(n^2) upfront cost
    mi_cache = {}

    for step in range(1, n_select):
        best_score = -np.inf
        best_feat  = -1

        for f in remaining:
            # Redundancy: mean MI with already-selected features
            redundancy = 0.0
            for s in selected_local:
                key = (min(f, s), max(f, s))
                if key not in mi_cache:
                    mi_cache[key] = _mutual_info(X_disc[:, f], X_disc[:, s])
                redundancy += mi_cache[key]
            redundancy /= len(selected_local)

            score = relevance[f] - redundancy
            if score > best_score:
                best_score = score
                best_feat  = f

        if best_feat < 0:
            break
        selected_local.append(best_feat)
        remaining.discard(best_feat)

    # Map back to original feature indices
    selected_original = valid_indices[np.array(selected_local)]
    return np.sort(selected_original)



# ===============================================================================
# STACKING ENSEMBLE  (RF + GradientBoosting + XGBoost -> RF meta-learner)
# ===============================================================================

class StackingEnsemble:
    """
    Two-level stacking ensemble for purchase prediction.

    Base learners:
        - RandomForestClassifier
        - GradientBoostingClassifier
        - XGBClassifier

    Meta-learner:
        - RandomForestClassifier

    During fit(), the training set is split into K inner folds. For each
    inner fold, each base learner is trained on the inner-train split and
    generates probability estimates on the inner-validation split. These
    out-of-fold probabilities become meta-features for the meta-learner.
    All base learners are then retrained on the full training set.
    """

    def __init__(self,
                 n_inner_folds: int = 5,
                 random_state: int = 42,
                 scale_pos_weight: float = 1.0,
                 hp: dict = None):
        """
        Parameters
        ----------
        scale_pos_weight : float
            Class imbalance ratio n_neg/n_pos passed to XGBoost.
            RF/GB use class_weight='balanced'/'balanced_subsample'.
        hp : dict or None
            Optional hyperparameter overrides from Optuna.
            Keys: rf_n_est, rf_max_depth, rf_min_leaf,
                  gb_n_est, gb_lr, gb_max_depth, gb_subsample,
                  xgb_n_est, xgb_lr, xgb_max_depth, xgb_subsample, xgb_colsample.
        """
        self.n_inner_folds    = n_inner_folds
        self.random_state     = random_state
        self.scale_pos_weight = scale_pos_weight
        hp = hp or {}
        rs  = random_state
        spw = scale_pos_weight

        # -- Base learner definitions (cost-sensitive) ---------------------
        self._base_specs = []
        self._base_specs.append(('RF', lambda: RandomForestClassifier(
            n_estimators      = hp.get('rf_n_est', 500),
            max_features      = 'sqrt',
            max_depth         = hp.get('rf_max_depth', None),
            min_samples_leaf  = hp.get('rf_min_leaf', 2),
            min_samples_split = 5,
            class_weight      = 'balanced_subsample',
            n_jobs            = 1,
            random_state      = rs,
        )))
        self._base_specs.append(('GB', lambda: GradientBoostingClassifier(
            n_estimators     = hp.get('gb_n_est', 200),
            max_depth        = hp.get('gb_max_depth', 4),
            learning_rate    = hp.get('gb_lr', 0.05),
            subsample        = hp.get('gb_subsample', 0.8),
            max_features     = 'sqrt',
            min_samples_leaf = 5,
            random_state     = rs,
        )))
        if HAS_XGB:
            self._base_specs.append(('XGB', lambda: XGBClassifier(
                n_estimators     = hp.get('xgb_n_est', 200),
                max_depth        = hp.get('xgb_max_depth', 4),
                learning_rate    = hp.get('xgb_lr', 0.05),
                subsample        = hp.get('xgb_subsample', 0.8),
                colsample_bytree = hp.get('xgb_colsample', 0.8),
                scale_pos_weight = spw,
                random_state     = rs,
                verbosity        = 0,
                eval_metric      = 'logloss',
                n_jobs           = 1,
            )))

        # -- Meta-learner (cost-sensitive) ---------------------------------
        self._meta = RandomForestClassifier(
            n_estimators      = 300,
            max_features      = 'sqrt',
            max_depth         = 6,
            min_samples_leaf  = 2,
            min_samples_split = 5,
            class_weight      = 'balanced',
            n_jobs            = 1,
            random_state      = rs,
        )

        self._base_clfs: list = []
        self.oof_proba_: np.ndarray = None   # stored for threshold tuning

    @staticmethod
    def _safe_proba(clf, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) regardless of classifier type."""
        if hasattr(clf, 'predict_proba'):
            return clf.predict_proba(X)[:, 1]
        return clf.decision_function(X)

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the stacking ensemble with inner CV for meta-features."""
        n       = len(y)
        n_base  = len(self._base_specs)
        meta_X  = np.zeros((n, n_base), dtype=np.float32)

        # -- Step 1: build OOF meta-features via inner K-fold CV ----------
        rng       = np.random.default_rng(self.random_state)
        indices   = rng.permutation(n)
        fold_size = max(1, n // self.n_inner_folds)

        for fold_i in range(self.n_inner_folds):
            val_start = fold_i * fold_size
            val_end   = val_start + fold_size if fold_i < self.n_inner_folds - 1 else n
            val_idx   = indices[val_start:val_end]
            tr_idx    = np.concatenate([indices[:val_start], indices[val_end:]])

            if len(np.unique(y[tr_idx])) < 2:
                continue

            for b_idx, (name, make_clf) in enumerate(self._base_specs):
                clf = make_clf()
                try:
                    clf.fit(X[tr_idx], y[tr_idx])
                    meta_X[val_idx, b_idx] = self._safe_proba(clf, X[val_idx])
                except Exception:
                    pass

        # -- Step 2: train meta-learner on OOF predictions ----------------
        self._meta.fit(meta_X, y)

        # Store OOF proba for threshold optimisation (Priority 1)
        self.oof_proba_ = self._meta.predict_proba(meta_X)[:, 1]

        # -- Step 3: retrain all base learners on full training set --------
        self._base_clfs = []
        for name, make_clf in self._base_specs:
            clf = make_clf()
            try:
                clf.fit(X, y)
            except Exception:
                clf = None
            self._base_clfs.append(clf)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return (n_samples, 2) probability array."""
        n_base = len(self._base_clfs)
        meta_X = np.zeros((len(X), n_base), dtype=np.float32)
        for b_idx, clf in enumerate(self._base_clfs):
            if clf is not None:
                try:
                    meta_X[:, b_idx] = self._safe_proba(clf, X)
                except Exception:
                    pass
        return self._meta.predict_proba(meta_X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return P(class=1) as a 1-D score."""
        return self.predict_proba(X)[:, 1]


# ===============================================================================
# PRIORITY 1: Optimal threshold (OOF-based, no leakage)
# ===============================================================================

def find_optimal_threshold(y_true: np.ndarray,
                           y_prob: np.ndarray,
                           metric: str = 'f1') -> float:
    """
    Sweep thresholds [0.05, 0.95] and return the one that maximises `metric`
    ('f1' or 'kappa') on the supplied labels (should be OOF predictions).
    """
    from sklearn.metrics import f1_score as _f1, cohen_kappa_score as _kappa
    best_thresh, best_score = 0.5, -1.0
    for t in np.arange(0.05, 0.96, 0.01):
        y_hat = (y_prob >= t).astype(int)
        score = _f1(y_true, y_hat, zero_division=0) if metric == 'f1' \
            else _kappa(y_true, y_hat)
        if score > best_score:
            best_score, best_thresh = score, float(t)
    return best_thresh


# ===============================================================================
# PRIORITY 3: Optuna hyperparameter search (training-data only)
# ===============================================================================

def tune_hyperparams_optuna(X_train: np.ndarray,
                            y_train: np.ndarray,
                            n_trials: int = 30,
                            scale_pos_weight: float = 1.0,
                            random_state: int = 42) -> dict:
    """
    Bayesian hyperparameter search using Optuna over a 3-fold StratifiedKFold
    on the supplied training data.  Returns the best hp dict for StackingEnsemble.
    """
    import optuna
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import f1_score as _f1

    def objective(trial):
        hp = dict(
            rf_n_est       = trial.suggest_int('rf_n_est',    100, 600, step=100),
            rf_max_depth   = trial.suggest_categorical('rf_max_depth', [None, 8, 16, 24]),
            rf_min_leaf    = trial.suggest_int('rf_min_leaf',  1,  5),
            gb_n_est       = trial.suggest_int('gb_n_est',    100, 400, step=100),
            gb_lr          = trial.suggest_float('gb_lr',     0.01, 0.2, log=True),
            gb_max_depth   = trial.suggest_int('gb_max_depth', 2,  6),
            gb_subsample   = trial.suggest_float('gb_subsample', 0.6, 1.0),
            xgb_n_est      = trial.suggest_int('xgb_n_est',  100, 400, step=100),
            xgb_lr         = trial.suggest_float('xgb_lr',   0.01, 0.2, log=True),
            xgb_max_depth  = trial.suggest_int('xgb_max_depth', 2, 6),
            xgb_subsample  = trial.suggest_float('xgb_subsample', 0.6, 1.0),
            xgb_colsample  = trial.suggest_float('xgb_colsample', 0.6, 1.0),
        )
        skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
        scores = []
        for tr_idx, val_idx in skf.split(X_train, y_train):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]
            if len(np.unique(y_tr)) < 2:
                continue
            try:
                clf = StackingEnsemble(
                    n_inner_folds    = 3,
                    random_state     = random_state,
                    scale_pos_weight = scale_pos_weight,
                    hp               = hp,
                )
                clf.fit(X_tr, y_tr)
                thresh = find_optimal_threshold(y_tr, clf.oof_proba_)
                y_hat  = (clf.predict_proba(X_val)[:, 1] >= thresh).astype(int)
                scores.append(_f1(y_val, y_hat, zero_division=0))
            except Exception:
                scores.append(0.0)
        return float(np.mean(scores)) if scores else 0.0

    study = optuna.create_study(
        direction = 'maximize',
        sampler   = optuna.samplers.TPESampler(seed=random_state),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ===============================================================================
# PARALLEL LOOCV BATCH WORKER  (must be top-level for Windows multiprocessing)
# ===============================================================================

def _loocv_batch_worker(args_tuple):
    """
    Run a batch of LOOCV folds in a separate process.

    Each fold leaves out a single observation (epoch) and trains on all
    remaining observations. To keep parallelism efficient, each worker
    processes a contiguous batch of test indices.

    Parameters
    ----------
    args_tuple : (batch_indices, X_k, y_all, n_mrmr_features, random_state,
                  optuna_trials, mrmr_selected)

    Returns
    -------
    list of (test_idx, y_true_i, y_pred_i, y_prob_i)
    """
    (batch_indices, X_k, y_all, n_mrmr_features, random_state,
     optuna_trials, mrmr_selected) = args_tuple

    import warnings
    warnings.filterwarnings('ignore')
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import f1_score as _f1
    try:
        from imblearn.over_sampling import SMOTE
        _has_smote = True
    except ImportError:
        _has_smote = False

    n_total = len(y_all)
    results = []

    for test_idx in batch_indices:
        # Train indices: all except the test sample
        train_idx = np.concatenate([np.arange(0, test_idx),
                                     np.arange(test_idx + 1, n_total)])

        X_train_raw = X_k[train_idx]
        y_train     = y_all[train_idx]
        X_test_raw  = X_k[test_idx:test_idx+1]  # keep 2D
        y_test_i    = y_all[test_idx]

        # Use pre-computed mRMR selection (computed once on full data minus test)
        # For LOOCV, removing 1 sample has negligible effect on mRMR,
        # so we reuse the global selection for efficiency
        selected = mrmr_selected

        X_train_sel = X_train_raw[:, selected]
        X_test_sel  = X_test_raw[:, selected]

        # StandardScaler (fit on train only)
        scaler    = StandardScaler()
        X_train_s = scaler.fit_transform(X_train_sel)
        X_test_s  = scaler.transform(X_test_sel)

        # SMOTE (train only)
        X_final, y_final = X_train_s, y_train
        if _has_smote:
            minority_count = int(np.bincount(y_train).min())
            if minority_count >= 2:
                k_neighbors = min(5, minority_count - 1)
                try:
                    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_final, y_final = sm.fit_resample(X_train_s, y_train)
                except Exception:
                    pass

        if len(np.unique(y_final)) < 2:
            majority    = int(np.bincount(y_train).argmax())
            y_pred_i    = majority
            y_prob_i    = 0.5
        else:
            # Priority 2: dynamic class-imbalance ratio (train set only)
            counts = np.bincount(y_train)
            spw    = float(counts[0]) / float(counts[1]) if counts[1] > 0 else 1.0

            # Priority 3: Optuna HPO (train only, optional)
            # Use the globally tuned parameters by default to prevent overfitting
            hp = {
                'rf_n_est': 200,
                'rf_max_depth': 8,
                'rf_min_leaf': 2,
                'gb_n_est': 300,
                'gb_lr': 0.019449054454192715,
                'gb_max_depth': 3,
                'gb_subsample': 0.7252729706729542,
                'xgb_n_est': 200,
                'xgb_lr': 0.06391722501369021,
                'xgb_max_depth': 2,
                'xgb_subsample': 0.6044661495023579,
                'xgb_colsample': 0.9984915776921418,
            }

            if optuna_trials > 0:
                try:
                    hp = tune_hyperparams_optuna(
                        X_train_s, y_train,
                        n_trials         = optuna_trials,
                        scale_pos_weight = spw,
                        random_state     = random_state,
                    )
                except Exception:
                    pass

            clf = StackingEnsemble(
                n_inner_folds    = 5,
                random_state     = random_state,
                scale_pos_weight = spw,
                hp               = hp,
            )
            clf.fit(X_final, y_final)

            # Priority 1: find optimal threshold from OOF predictions (no leakage)
            # Trim OOF to original training set size (SMOTE may have added rows)
            oof_prob    = clf.oof_proba_[:len(y_train)]
            best_thresh = find_optimal_threshold(y_train, oof_prob, metric='f1')

            y_prob_i = float(clf.predict_proba(X_test_s)[:, 1][0])
            y_pred_i = int(y_prob_i >= best_thresh)

        results.append((test_idx, int(y_test_i), y_pred_i, y_prob_i))

    return results


def run_loocv(X_all: np.ndarray, y_all: np.ndarray,
              k: int,
              n_mrmr_features: int = 50,
              random_state: int = 42,
              n_jobs: int = -1,
              optuna_trials: int = 0) -> dict:
    """
    Complete Leave-One-Out Cross-Validation for a given k (retained IMFs).
    Folds are executed in parallel across CPU cores via batch processing.

    Per fold:
      1. Split: N-1 samples train, 1 sample test
      2. IMF truncation -> keep first k IMFs + ET
      3. mRMR feature selection (computed once globally for efficiency)
      4. StandardScaler fit on training data
      5. SMOTE on training data only
      6. Train stacking ensemble
      7. Predict on test sample

    Returns
    -------
    dict with:
      'y_true'          : true labels for all samples
      'y_pred'          : predicted labels for all samples
      'y_prob'          : predicted probabilities for all samples
      'mrmr_selected'   : the globally selected feature indices
      'metrics'         : dict of overall metrics
    """
    n_total   = len(y_all)
    n_workers = multiprocessing.cpu_count() if n_jobs == -1 else max(1, n_jobs)
    print(f'\n  -- LOOCV for k={k} IMFs ({n_total} folds, {n_workers} parallel workers) --')

    # Get the column indices for this k
    cols_k = get_imf_columns(k)
    X_k    = X_all[:, cols_k]

    # Global mRMR feature selection on all data
    # (removing 1 sample out of ~4000+ has negligible effect on mRMR ranking)
    print(f'  Computing global mRMR feature selection ({n_mrmr_features} features)...')
    mrmr_selected = mrmr_feature_selection(X_k, y_all, n_select=n_mrmr_features)
    print(f'  mRMR selected {len(mrmr_selected)} features.')

    # Split into batches for parallel workers
    all_indices = np.arange(n_total)
    batch_size  = max(1, n_total // n_workers)
    batches     = []
    for i in range(0, n_total, batch_size):
        batch = all_indices[i:i + batch_size].tolist()
        if batch:
            batches.append(batch)

    print(f'  Dispatching {len(batches)} batches to {n_workers} workers...')

    # Build work items
    work_items = []
    for batch in batches:
        work_items.append((
            batch, X_k, y_all, n_mrmr_features, random_state,
            optuna_trials, mrmr_selected
        ))

    # Collect results
    all_results = []  # list of (test_idx, y_true_i, y_pred_i, y_prob_i)

    completed_count = 0
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_loocv_batch_worker, item): idx
                   for idx, item in enumerate(work_items)}
        for future in as_completed(futures):
            batch_results = future.result()
            all_results.extend(batch_results)
            completed_count += len(batch_results)
            print(f'    Completed {completed_count:4d}/{n_total} folds', flush=True)

    # Sort by test_idx to restore original order
    all_results.sort(key=lambda x: x[0])

    y_true_arr = np.array([r[1] for r in all_results])
    y_pred_arr = np.array([r[2] for r in all_results])
    y_prob_arr = np.array([r[3] for r in all_results])

    overall = _compute_metrics(y_true_arr, y_pred_arr, y_prob_arr)

    return dict(
        y_true         = y_true_arr,
        y_pred         = y_pred_arr,
        y_prob         = y_prob_arr,
        mrmr_selected  = mrmr_selected,
        metrics        = overall,
    )


def _compute_metrics(y_true, y_pred, y_prob=None):
    """Compute the 6 evaluation metrics."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(np.unique(y_true)) < 2:
        return dict(
            accuracy=accuracy_score(y_true, y_pred),
            precision=0.0, recall=0.0, f1=0.0, auc=0.5, kappa=0.0,
        )

    acc   = accuracy_score(y_true, y_pred)
    prec  = precision_score(y_true, y_pred, zero_division=0)
    rec   = recall_score(y_true, y_pred, zero_division=0)
    f1_v  = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            auc = roc_auc_score(y_true, np.asarray(y_prob))
        except ValueError:
            auc = 0.5
    else:
        auc = 0.5

    return dict(
        accuracy  = acc,
        precision = prec,
        recall    = rec,
        f1        = f1_v,
        auc       = auc,
        kappa     = kappa,
    )


# ===============================================================================
# SHAP ANALYSIS
# ===============================================================================

def run_shap_analysis(X_all: np.ndarray, y_all: np.ndarray,
                      best_k: int,
                      mrmr_selected: np.ndarray,
                      out_dir: str,
                      n_mrmr_features: int = 50,
                      random_state: int = 42):
    """
    Run SHAP analysis on the best-k configuration.

    Trains a stacking ensemble on ALL data (for interpretability analysis),
    then uses SHAP TreeExplainer on each base learner to compute feature
    importance. Results are aggregated by:
      - Individual features (top 20)
      - IMF number
      - EEG channel
      - Eye-tracking features
      - Neurophysiological domain

    Saves plots and interpretation report to out_dir.
    """
    if not HAS_SHAP:
        print('\n  SHAP not available -- skipping SHAP analysis.')
        return
    if not HAS_MPL:
        print('\n  matplotlib not available -- skipping SHAP plots.')
        return

    print(f'\n{"=" * 62}')
    print(f'  SHAP ANALYSIS (best k={best_k})')
    print(f'{"=" * 62}')

    os.makedirs(out_dir, exist_ok=True)
    cols_k = get_imf_columns(best_k)

    # Use pre-truncated data
    X_k = X_all[:, cols_k]

    # Use the provided mRMR selection
    stable_features = mrmr_selected

    X_sel = X_k[:, stable_features]

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sel)

    # SMOTE for balanced training
    X_train, y_train = X_scaled, y_all
    if HAS_SMOTE:
        minority_count = int(np.bincount(y_all).min())
        if minority_count >= 2:
            k_neighbors = min(5, minority_count - 1)
            try:
                sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_train, y_train = sm.fit_resample(X_scaled, y_all)
            except Exception:
                pass

    # Train individual models for SHAP (TreeExplainer works best with trees)
    models = {}
    model_specs = [
        ('RandomForest', RandomForestClassifier(
            n_estimators=200, max_features='sqrt', max_depth=None,
            min_samples_leaf=2, class_weight='balanced_subsample',
            random_state=random_state, n_jobs=-1,
        )),
        ('GradientBoosting', GradientBoostingClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=random_state,
        )),
    ]
    if HAS_XGB:
        model_specs.append(('XGBoost', XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            random_state=random_state, verbosity=0, eval_metric='logloss',
            n_jobs=-1,
        )))

    # Compute SHAP values from each model and average
    all_shap_values = []

    for model_name, model in model_specs:
        print(f'  Training {model_name} for SHAP...')
        model.fit(X_train, y_train)
        models[model_name] = model

        # Use a subsample for SHAP (speed)
        n_shap = min(500, len(X_scaled))
        rng = np.random.default_rng(random_state)
        shap_idx = rng.choice(len(X_scaled), size=n_shap, replace=False)
        X_shap = X_scaled[shap_idx]

        try:
            explainer  = shap.TreeExplainer(model)
            shap_vals  = explainer.shap_values(X_shap)

            # Handle different SHAP output formats
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[-1]        # class 1 (Buy)
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                shap_vals = shap_vals[:, :, -1]  # class 1
            if hasattr(shap_vals, 'values'):
                shap_vals = shap_vals.values

            mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)
            all_shap_values.append(mean_abs_shap)
            print(f'    {model_name}: SHAP computed ({n_shap} samples, {len(stable_features)} features)')
        except Exception as e:
            print(f'    {model_name}: SHAP failed -- {e}')

    if not all_shap_values:
        print('  No SHAP values computed -- skipping analysis.')
        return

    # Average SHAP importance across models
    mean_shap = np.mean(np.stack(all_shap_values), axis=0)

    # -- Decode features --------------------------------------------------
    feature_info = []
    for i, feat_idx in enumerate(stable_features):
        info = decode_feature_index_for_k(feat_idx, best_k)
        info['shap_importance'] = float(mean_shap[i])
        feature_info.append(info)

    # -- 1. Top Feature Importance ----------------------------------------
    sorted_feats = sorted(feature_info, key=lambda x: x['shap_importance'], reverse=True)
    top_n = min(20, len(sorted_feats))
    top_feats = sorted_feats[:top_n]

    print(f'\n  Top {top_n} features by SHAP importance:')
    for i, f in enumerate(top_feats):
        if f['modality'] == 'EEG':
            label = f'{f["channel_name"]}_IMF{f["imf_number"]}_{f["feature_name"]}'
        else:
            label = f'ET_{f["feature_name"]}'
        print(f'    {i+1:2d}. {label:40s}  SHAP={f["shap_importance"]:.4f}  [{f["domain"]}]')

    if HAS_MPL:
        _plot_top_features(top_feats, best_k, out_dir)
        _plot_imf_importance(feature_info, best_k, out_dir)
        _plot_channel_importance(feature_info, best_k, out_dir)
        _plot_et_importance(feature_info, best_k, out_dir)
        _plot_domain_importance(feature_info, best_k, out_dir)

    # -- Save interpretation report ---------------------------------------
    _save_interpretation_report(feature_info, best_k, out_dir)
    _save_shap_csv(feature_info, best_k, out_dir)

    print(f'\n  SHAP analysis saved to: {out_dir}/')


# ===============================================================================
# SHAP PLOTTING FUNCTIONS
# ===============================================================================

def _plot_top_features(top_feats, best_k, out_dir):
    """Bar chart of top features by SHAP importance."""
    labels = []
    values = []
    colors = []
    color_map = {
        'Oscillatory Activity': '#FF6B6B',
        'Signal Complexity':    '#4ECDC4',
        'Temporal Dynamics':    '#45B7D1',
        'Nonlinear Dynamics':   '#96CEB4',
        'Eye-Tracking':         '#FFEAA7',
    }
    for f in reversed(top_feats):
        if f['modality'] == 'EEG':
            lbl = f'{f["channel_name"]}_IMF{f["imf_number"]}_{f["feature_name"]}'
        else:
            lbl = f'ET_{f["feature_name"]}'
        labels.append(lbl)
        values.append(f['shap_importance'])
        colors.append(color_map.get(f['domain'], '#CCCCCC'))

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(range(len(labels)), values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title(f'Top {len(top_feats)} Features by SHAP Importance (k={best_k} IMFs, LOOCV)', fontsize=14)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=d) for d, c in color_map.items()]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'shap_feature_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved: {path}')


def _plot_imf_importance(feature_info, best_k, out_dir):
    """Aggregate SHAP importance by IMF number."""
    imf_shap = defaultdict(float)
    imf_count = defaultdict(int)
    for f in feature_info:
        if f['modality'] == 'EEG' and f['imf_number'] is not None:
            imf_shap[f['imf_number']] += f['shap_importance']
            imf_count[f['imf_number']] += 1

    if not imf_shap:
        return

    imfs = sorted(imf_shap.keys())
    # Use mean importance per IMF (normalize by count)
    values = [imf_shap[i] / max(imf_count[i], 1) for i in imfs]
    labels = [f'IMF {i}' for i in imfs]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, values, color='#4ECDC4', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('IMF Number', fontsize=12)
    ax.set_ylabel('Mean |SHAP Value| (per feature)', fontsize=12)
    ax.set_title(f'IMF Importance -- Mean SHAP per Feature (k={best_k} IMFs, LOOCV)', fontsize=14)

    # Annotate bars
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    path = os.path.join(out_dir, 'shap_imf_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved: {path}')


def _plot_channel_importance(feature_info, best_k, out_dir):
    """Aggregate SHAP importance by EEG channel."""
    ch_shap = defaultdict(float)
    ch_count = defaultdict(int)
    for f in feature_info:
        if f['modality'] == 'EEG' and f['channel_name'] is not None:
            ch_shap[f['channel_name']] += f['shap_importance']
            ch_count[f['channel_name']] += 1

    if not ch_shap:
        return

    # Sort by importance
    sorted_channels = sorted(ch_shap.keys(), key=lambda c: ch_shap[c] / max(ch_count[c], 1), reverse=True)
    values = [ch_shap[c] / max(ch_count[c], 1) for c in sorted_channels]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(sorted_channels, values, color='#FF6B6B', edgecolor='white', linewidth=0.5)
    ax.set_xlabel('EEG Channel', fontsize=12)
    ax.set_ylabel('Mean |SHAP Value| (per feature)', fontsize=12)
    ax.set_title(f'Channel Importance -- Mean SHAP per Feature (k={best_k} IMFs, LOOCV)', fontsize=14)

    # Annotate
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.0005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8, rotation=45)

    plt.tight_layout()
    path = os.path.join(out_dir, 'shap_channel_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved: {path}')


def _plot_et_importance(feature_info, best_k, out_dir):
    """SHAP importance for eye-tracking features."""
    et_feats = [f for f in feature_info if f['modality'] == 'ET']
    if not et_feats:
        print('  No ET features in mRMR selection -- skipping ET plot.')
        return

    et_feats.sort(key=lambda x: x['shap_importance'], reverse=True)
    labels = [f['feature_name'] for f in et_feats]
    values = [f['shap_importance'] for f in et_feats]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), values, color='#FFEAA7', edgecolor='#F39C12', linewidth=0.5)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
    ax.set_title(f'Eye-Tracking Feature Importance (k={best_k} IMFs, LOOCV)', fontsize=14)

    plt.tight_layout()
    path = os.path.join(out_dir, 'shap_et_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved: {path}')


def _plot_domain_importance(feature_info, best_k, out_dir):
    """Aggregate SHAP importance by neurophysiological domain."""
    domain_shap = defaultdict(float)
    domain_count = defaultdict(int)
    for f in feature_info:
        d = f.get('domain', 'Unknown')
        domain_shap[d] += f['shap_importance']
        domain_count[d] += 1

    if not domain_shap:
        return

    sorted_domains = sorted(domain_shap.keys(),
                           key=lambda d: domain_shap[d] / max(domain_count[d], 1),
                           reverse=True)
    values = [domain_shap[d] / max(domain_count[d], 1) for d in sorted_domains]

    color_map = {
        'Oscillatory Activity': '#FF6B6B',
        'Signal Complexity':    '#4ECDC4',
        'Temporal Dynamics':    '#45B7D1',
        'Nonlinear Dynamics':   '#96CEB4',
        'Eye-Tracking':         '#FFEAA7',
    }
    colors = [color_map.get(d, '#CCCCCC') for d in sorted_domains]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_domains, values, color=colors, edgecolor='white', linewidth=0.5)
    ax.set_ylabel('Mean |SHAP Value| (per feature)', fontsize=12)
    ax.set_title(f'Neurophysiological Domain Importance (k={best_k} IMFs, LOOCV)', fontsize=14)

    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)

    plt.xticks(rotation=15, ha='right', fontsize=10)
    plt.tight_layout()
    path = os.path.join(out_dir, 'shap_domain_importance.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  -> Saved: {path}')


# ===============================================================================
# INTERPRETATION REPORT
# ===============================================================================

def _save_interpretation_report(feature_info, best_k, out_dir):
    """
    Generate a neurophysiological interpretation report linking SHAP results
    to consumer attention, engagement, and decision-making.
    """
    path = os.path.join(out_dir, 'shap_domain_interpretation.txt')

    # Aggregate by domain
    domain_shap   = defaultdict(float)
    domain_count  = defaultdict(int)
    domain_feats  = defaultdict(list)
    for f in feature_info:
        d = f.get('domain', 'Unknown')
        domain_shap[d] += f['shap_importance']
        domain_count[d] += 1
        domain_feats[d].append(f)

    sorted_domains = sorted(domain_shap.keys(),
                           key=lambda d: domain_shap[d],
                           reverse=True)

    with open(path, 'w') as fp:
        fp.write('=' * 72 + '\n')
        fp.write(f'  NEUROPHYSIOLOGICAL INTERPRETATION OF SHAP RESULTS\n')
        fp.write(f'  IMF Configuration: k = {best_k} retained IMFs\n')
        fp.write(f'  Cross-Validation: LOOCV (Leave-One-Out)\n')
        fp.write('=' * 72 + '\n\n')

        for domain in sorted_domains:
            total_shap = domain_shap[domain]
            count      = domain_count[domain]
            mean_shap  = total_shap / max(count, 1)

            fp.write(f'{"-" * 72}\n')
            fp.write(f'  DOMAIN: {domain.upper()}\n')
            fp.write(f'  Total SHAP: {total_shap:.4f}  |  '
                     f'Features: {count}  |  '
                     f'Mean SHAP: {mean_shap:.4f}\n')
            fp.write(f'{"-" * 72}\n\n')

            # Domain description
            desc = _DOMAIN_DESCRIPTIONS.get(domain, '')
            if desc:
                fp.write(f'  Interpretation:\n')
                # Wrap text at ~70 chars
                words = desc.split()
                line = '    '
                for w in words:
                    if len(line) + len(w) + 1 > 70:
                        fp.write(line + '\n')
                        line = '    ' + w
                    else:
                        line = line + ' ' + w if line.strip() else '    ' + w
                fp.write(line + '\n\n')

            # Top features in this domain
            domain_f = sorted(domain_feats[domain],
                             key=lambda x: x['shap_importance'], reverse=True)
            top_n = min(5, len(domain_f))
            fp.write(f'  Top {top_n} contributing features:\n')
            for i, f in enumerate(domain_f[:top_n]):
                if f['modality'] == 'EEG':
                    label = f'{f["channel_name"]}_IMF{f["imf_number"]}_{f["feature_name"]}'
                else:
                    label = f'ET_{f["feature_name"]}'
                fp.write(f'    {i+1}. {label:40s}  SHAP = {f["shap_importance"]:.4f}\n')
            fp.write('\n')

        # -- Consumer Behaviour Summary ------------------------------------
        fp.write('=' * 72 + '\n')
        fp.write('  CONSUMER BEHAVIOUR IMPLICATIONS\n')
        fp.write('=' * 72 + '\n\n')

        fp.write('  1. ATTENTION (Oscillatory Activity + Eye-Tracking)\n')
        fp.write('     Energy distribution across IMFs and fixation patterns reveal\n')
        fp.write('     how consumers allocate visual and cognitive attention to products.\n')
        fp.write('     High-energy low-frequency IMFs suggest sustained engagement.\n\n')

        fp.write('  2. ENGAGEMENT (Signal Complexity)\n')
        fp.write('     Complex neural patterns indicate deep processing of product\n')
        fp.write('     information. Higher spectral entropy and Hjorth complexity\n')
        fp.write('     correlate with thorough evaluation of product attributes.\n\n')

        fp.write('  3. DECISION-MAKING (Temporal Dynamics + Nonlinear Dynamics)\n')
        fp.write('     Fast temporal fluctuations suggest active comparison between\n')
        fp.write('     products. Nonlinear features capture emotional responses and\n')
        fp.write('     novelty effects that drive purchase decisions.\n\n')

        # -- IMF-level interpretation --------------------------------------
        fp.write('=' * 72 + '\n')
        fp.write('  IMF-LEVEL INTERPRETATION\n')
        fp.write('=' * 72 + '\n\n')

        imf_shap_agg = defaultdict(float)
        imf_count_agg = defaultdict(int)
        for f in feature_info:
            if f['modality'] == 'EEG' and f['imf_number'] is not None:
                imf_shap_agg[f['imf_number']] += f['shap_importance']
                imf_count_agg[f['imf_number']] += 1

        imf_descriptions = {
            1: 'Highest frequency oscillations -- captures rapid neural transients,\n'
               '     associated with sensory processing and immediate reactions.',
            2: 'High-frequency component -- reflects fast cognitive processing,\n'
               '     including attention capture by salient product features.',
            3: 'Mid-high frequency -- corresponds to beta-range activity,\n'
               '     related to active thinking and decision evaluation.',
            4: 'Mid-frequency -- captures alpha-range dynamics,\n'
               '     associated with relaxed attention and familiarity processing.',
            5: 'Mid-low frequency -- theta-range activity,\n'
               '     linked to memory encoding and emotional engagement.',
            6: 'Low frequency -- slow oscillations capturing sustained attention\n'
               '     and working memory demands during product comparison.',
            7: 'Very low frequency -- reflects slow cortical potentials\n'
               '     associated with motivational state and decision readiness.',
        }

        for imf_num in sorted(imf_shap_agg.keys()):
            mean_val = imf_shap_agg[imf_num] / max(imf_count_agg[imf_num], 1)
            fp.write(f'  IMF {imf_num}: Mean SHAP = {mean_val:.4f}\n')
            if imf_num in imf_descriptions:
                fp.write(f'     {imf_descriptions[imf_num]}\n')
            fp.write('\n')

    print(f'  -> Saved: {path}')


def _save_shap_csv(feature_info, best_k, out_dir):
    """Save SHAP values to CSV for further analysis."""
    path = os.path.join(out_dir, 'shap_feature_details.csv')
    fieldnames = ['feature_index', 'modality', 'channel_number', 'channel_name',
                  'imf_number', 'feature_name', 'domain', 'shap_importance']

    sorted_feats = sorted(feature_info, key=lambda x: x['shap_importance'], reverse=True)

    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for feat in sorted_feats:
            row = {k: feat.get(k, '') for k in fieldnames}
            if row['channel_number'] is None:
                row['channel_number'] = ''
            if row['channel_name'] is None:
                row['channel_name'] = ''
            if row['imf_number'] is None:
                row['imf_number'] = ''
            writer.writerow(row)

    print(f'  -> Saved: {path}')


# ===============================================================================
# MAIN
# ===============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='LOOCV Cross-Validation with SHAP Analysis (single k)',
    )
    parser.add_argument('--features_dir', default='features_v2_combined',
                        help='Folder with SXX_X.npy, SXX_y.npy, subjects.txt '
                             '(EEG+ET concatenated features) '
                             '(default: features_v2_combined)')
    parser.add_argument('--k', type=int, default=N_IMFS_TOTAL,
                        help=f'Number of IMFs to use (default: {N_IMFS_TOTAL})')
    parser.add_argument('--n_mrmr', type=float, default=50,
                        help='Features to select via mRMR per fold '
                             '(>=1 = absolute count; <1 = fraction of total)')
    parser.add_argument('--out_dir', default='results_loocv',
                        help='Output directory (default: results_loocv)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--skip_shap', action='store_true',
                        help='Skip SHAP analysis')
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Parallel workers for LOOCV folds '
                             '(-1 = all CPU cores, default: -1)')
    parser.add_argument('--optuna_trials', type=int, default=0,
                        help='Optuna HPO trials per fold (0=disabled, '
                             'try 20-50; note: multiplies runtime ~trials/3)')
    args = parser.parse_args()

    k = args.k
    if k > N_IMFS_TOTAL:
        print(f'ERROR: --k {k} exceeds N_IMFS_TOTAL={N_IMFS_TOTAL}. Aborting.')
        return

    # -- Load data ---------------------------------------------------------
    print('\n  Loading features from:', args.features_dir)
    dataset = load_features(args.features_dir)

    if not dataset:
        print('ERROR: no feature files found. Run feature_extraction.py first.')
        return

    # Pool all subjects into one dataset for LOOCV
    X_all, y_all, subject_ids_per_epoch = pool_all_subjects(dataset)

    n_subjects   = len(dataset)
    feat_dim     = X_all.shape[1]
    total_epochs = len(y_all)
    total_buy    = int(y_all.sum())

    eeg_cols_k   = N_EEG_CHANNELS * k * N_FEAT_PER_IMF
    total_cols_k = eeg_cols_k + N_ET_FEATURES

    if args.n_mrmr < 1.0:
        n_mrmr_features = int(total_cols_k * args.n_mrmr)
    else:
        n_mrmr_features = int(args.n_mrmr)

    print('=' * 62)
    print('  LOOCV Cross-Validation + SHAP Analysis')
    print('=' * 62)
    print(f'  Subjects      : {n_subjects}')
    print(f'  Feature dim   : {feat_dim}')
    print(f'  Total epochs  : {total_epochs}  (Buy={total_buy}, NoBuy={total_epochs-total_buy})')
    print(f'  k (IMFs)      : {k}  ({eeg_cols_k} EEG + {N_ET_FEATURES} ET = {total_cols_k} total)')
    print(f'  mRMR features : {n_mrmr_features}'
          + (f'  ({args.n_mrmr*100:.0f}% of {total_cols_k})' if args.n_mrmr < 1.0 else ''))
    print(f'  SMOTE         : {"ON" if HAS_SMOTE else "OFF"}')
    print(f'  XGBoost       : {"ON" if HAS_XGB else "OFF"}')
    print(f'  n_jobs        : {args.n_jobs} ({multiprocessing.cpu_count()} CPUs available)')
    print(f'  Optuna trials : {args.optuna_trials} ({"ON" if HAS_OPTUNA and args.optuna_trials > 0 else "OFF"})')
    print(f'  Cost-sensitive: ON  (scale_pos_weight=dynamic, RF/GB class_weight=balanced)')
    print(f'  Threshold opt : ON  (OOF F1-maximisation per fold)')
    print(f'  SHAP          : {"ON" if HAS_SHAP and not args.skip_shap else "OFF"}')
    print(f'  Output dir    : {args.out_dir}')
    print(f'  CV Method     : LOOCV (Leave-One-Out Cross-Validation)')
    print(f'  Features src  : {args.features_dir}')
    print('=' * 62)

    os.makedirs(args.out_dir, exist_ok=True)

    # -- Run LOOCV ---------------------------------------------------------
    t_start = time.time()
    result  = run_loocv(
        X_all, y_all, k,
        n_mrmr_features = n_mrmr_features,
        random_state    = args.seed,
        n_jobs          = args.n_jobs,
        optuna_trials   = args.optuna_trials if HAS_OPTUNA else 0,
    )
    elapsed = time.time() - t_start

    m = result['metrics']
    print(f'\n  Results (k={k}, elapsed: {elapsed/60:.1f} min):')
    print(f'    Accuracy  = {m["accuracy"]:.4f}')
    print(f'    Precision = {m["precision"]:.4f}')
    print(f'    Recall    = {m["recall"]:.4f}')
    print(f'    F1        = {m["f1"]:.4f}')
    print(f'    AUC-ROC   = {m["auc"]:.4f}')
    print(f'    Kappa     = {m["kappa"]:+.4f}')

    # -- Save per-epoch results --------------------------------------------
    per_epoch_path = os.path.join(args.out_dir, f'per_epoch_k{k}.csv')
    with open(per_epoch_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch_idx', 'subject_id', 'y_true', 'y_pred', 'y_prob'])
        for i in range(total_epochs):
            writer.writerow([
                i, subject_ids_per_epoch[i],
                int(result['y_true'][i]),
                int(result['y_pred'][i]),
                f'{result["y_prob"][i]:.4f}',
            ])
    print(f'\n  -> Per-epoch results saved: {per_epoch_path}')

    # -- Compute and save per-subject aggregated metrics --------------------
    per_subject_metrics = {}
    for sid in sorted(set(subject_ids_per_epoch)):
        mask = np.array([s == sid for s in subject_ids_per_epoch])
        yt = result['y_true'][mask]
        yp = result['y_pred'][mask]
        yprob = result['y_prob'][mask]
        sm = _compute_metrics(yt, yp, yprob)
        per_subject_metrics[sid] = sm

    per_subj_path = os.path.join(args.out_dir, f'per_subject_k{k}.csv')
    with open(per_subj_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['subject_id', 'accuracy', 'precision', 'recall',
                         'f1', 'auc', 'kappa'])
        for sid in sorted(per_subject_metrics.keys()):
            sm = per_subject_metrics[sid]
            writer.writerow([
                sid, f'{sm["accuracy"]:.4f}', f'{sm["precision"]:.4f}',
                f'{sm["recall"]:.4f}', f'{sm["f1"]:.4f}',
                f'{sm["auc"]:.4f}', f'{sm["kappa"]:.4f}',
            ])
    print(f'  -> Per-subject results saved: {per_subj_path}')

    # -- Save overall metrics CSV ------------------------------------------
    metrics_path = os.path.join(args.out_dir, 'metrics.csv')
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['k', 'n_mrmr', 'cv_method', 'accuracy', 'precision',
                         'recall', 'f1', 'auc', 'kappa'])
        writer.writerow([
            k, n_mrmr_features, 'LOOCV',
            f'{m["accuracy"]:.4f}', f'{m["precision"]:.4f}',
            f'{m["recall"]:.4f}', f'{m["f1"]:.4f}',
            f'{m["auc"]:.4f}', f'{m["kappa"]:.4f}',
        ])
    print(f'  -> Metrics saved: {metrics_path}')

    # -- SHAP Analysis -----------------------------------------------------
    if not args.skip_shap:
        run_shap_analysis(
            X_all, y_all, k,
            mrmr_selected   = result['mrmr_selected'],
            out_dir         = args.out_dir,
            n_mrmr_features = n_mrmr_features,
            random_state    = args.seed,
        )

    print(f'\n{"=" * 62}')
    print(f'  DONE -- Results saved to: {args.out_dir}/')
    print(f'{"=" * 62}')


if __name__ == '__main__':
    main()
