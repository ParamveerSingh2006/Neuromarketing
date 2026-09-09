#!/usr/bin/env python3
"""
Stage 2 — NeuMa signal processing and feature construction
for Kalaganis et al. (2025), points 7–27.

Consumes:
    stage1_dataset.pkl

Produces:
    stage2_features.pkl
    stage2_features.csv
    stage2_graphs.pkl

Implemented:
    EEG:
      - 8–30 Hz zero-phase Butterworth filtering
      - Hilbert-envelope functional connectivity
      - absolute Pearson envelope correlation
      - subject-specific weighted graph W
      - unnormalised Laplacian L = D - W
      - eigendecomposition
      - Graph Fourier Transform X~ = U.T @ X
      - mean graph-mode power
      - mean aggregation across valid product segments

    ET:
      - gaze trajectory / accumulated distance
      - Hjorth Activity, Mobility, Complexity
      - EyeMMV-style two-step fixation detection for:
          fixation count
          mean fixation duration
          time to first fixation
          dwell time
      - blink rate is NOT inferred from interpolated ET data by default.
        The paper specifies blink rate but the supplied preprocessed
        dataset documentation does not expose a blink indicator. A
        pupil-based blink detector would therefore be an additional
        operationalization, not a faithful reproduction.

    Fusion:
      - concatenate EEG and ET features at product level
      - retain Buy/NoBuy labels

IMPORTANT:
    The paper does not explicitly disclose whether the functional graph
    is reconstructed inside each CV fold or once per subject. Because
    this stage precedes CV, this implementation uses a subject-level graph
    built from all retained EEG segments. This is recorded in metadata and
    MUST be revisited when implementing the leakage-controlled CV stage.

    Likewise, exact normalization and the precise blink-rate implementation
    are not specified sufficiently by the paper. This file does not silently
    invent them.

This file does NOT implement:
    - normalization for the classifier
    - Wilcoxon feature selection
    - LOOCV
    - SVM
    - metrics
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert


DEFAULT_LOWER_HZ = 8.0
DEFAULT_UPPER_HZ = 30.0
DEFAULT_FILTER_ORDER = 3

# EyeMMV reference operationalization.
# The Kalaganis paper cites EyeMMV [43] for fixation identification but
# does not state these parameters explicitly.
DEFAULT_T1 = 0.250
DEFAULT_T2 = 0.100
DEFAULT_MIN_FIXATION_SECONDS = 0.150


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def bandpass_zero_phase(
    x: np.ndarray,
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int,
) -> np.ndarray:
    nyquist = fs / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Invalid band [{low_hz}, {high_hz}] Hz for Fs={fs} Hz."
        )

    b, a = butter(
        order,
        [low_hz / nyquist, high_hz / nyquist],
        btype="bandpass",
    )

    # filtfilt can fail on extremely short segments. Such a segment should
    # already have been screened by Stage 1, but we retain a defensive check.
    padlen = 3 * max(len(a), len(b))
    if x.shape[-1] <= padlen:
        raise ValueError(
            f"Segment has only {x.shape[-1]} samples; "
            f"zero-phase filter requires more than padlen={padlen}."
        )

    return filtfilt(b, a, x, axis=-1)


def envelope_connectivity(concatenated_eeg: np.ndarray) -> np.ndarray:
    """
    W[m,n] = |corr(HilbertEnvelope(x_m), HilbertEnvelope(x_n))|

    Input:
        [channels x samples]
    Output:
        [channels x channels]
    """
    analytic = hilbert(concatenated_eeg, axis=1)
    envelopes = np.abs(analytic)

    # np.corrcoef treats rows as variables.
    W = np.corrcoef(envelopes)
    W = np.abs(W)

    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    W = (W + W.T) / 2.0
    np.fill_diagonal(W, 0.0)

    return W


def graph_fourier_basis(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    degree = np.sum(W, axis=1)
    D = np.diag(degree)
    L = D - W

    eigenvalues, eigenvectors = np.linalg.eigh(L)

    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    return eigenvalues, eigenvectors


def gft_power(
    eeg_segment: np.ndarray,
    eigenvectors: np.ndarray,
) -> np.ndarray:
    """
    X~ = U.T X
    p(m) = mean_t |X~(m,t)|^2
    """
    transformed = eigenvectors.T @ eeg_segment
    power = np.mean(np.abs(transformed) ** 2, axis=1)
    return np.asarray(power, dtype=np.float64)


def aggregate_segment_features(
    segment_features: List[np.ndarray],
) -> np.ndarray:
    if not segment_features:
        raise ValueError("Cannot aggregate an empty feature list.")
    return np.mean(np.vstack(segment_features), axis=0)


def gaze_xy_from_et(
    et_segment: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    NeuMa ET_clean.Data rows:
      0 = left X
      1 = left Y
      2 = left pupil
      3 = right X
      4 = right Y
      5 = right pupil

    Coordinates are normalized to [0,1].
    We average left/right gaze coordinates where both are finite.
    If only one eye is finite, that eye is used.
    """
    if et_segment.shape[0] < 5:
        raise ValueError(
            f"Expected at least 5 ET rows, got {et_segment.shape[0]}"
        )

    lx, ly = et_segment[0], et_segment[1]
    rx, ry = et_segment[3], et_segment[4]

    valid_l = np.isfinite(lx) & np.isfinite(ly)
    valid_r = np.isfinite(rx) & np.isfinite(ry)

    x = np.full(lx.shape, np.nan, dtype=float)
    y = np.full(ly.shape, np.nan, dtype=float)

    both = valid_l & valid_r
    only_l = valid_l & ~valid_r
    only_r = ~valid_l & valid_r

    x[both] = (lx[both] + rx[both]) / 2.0
    y[both] = (ly[both] + ry[both]) / 2.0

    x[only_l] = lx[only_l]
    y[only_l] = ly[only_l]

    x[only_r] = rx[only_r]
    y[only_r] = ry[only_r]

    return x, y


def interpolate_internal_nans(
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Defensive interpolation for any residual NaNs.

    The NeuMa preprocessed dataset states that missing ET values caused by
    spontaneous blinking were linearly interpolated already. This function
    only handles residual NaNs if present in the supplied file.
    """
    idx = np.arange(len(x))
    valid = np.isfinite(x) & np.isfinite(y)

    if valid.sum() < 2:
        raise ValueError(
            "Insufficient valid gaze samples after preprocessing."
        )

    x_out = np.interp(idx, idx[valid], x[valid])
    y_out = np.interp(idx, idx[valid], y[valid])

    return x_out, y_out


def accumulated_distance(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    dx = np.diff(x, prepend=x[0])
    dy = np.diff(y, prepend=y[0])
    distance = np.sqrt(dx * dx + dy * dy)
    return distance


def hjorth_features(x: np.ndarray) -> np.ndarray:
    """
    Hjorth:
      Activity   = var(x)
      Mobility   = sqrt(var(dx) / var(x))
      Complexity = Mobility(dx) / Mobility(x)
    """
    x = np.asarray(x, dtype=float)

    activity = float(np.var(x))

    if activity <= np.finfo(float).eps:
        return np.array([activity, 0.0, 0.0], dtype=float)

    dx = np.diff(x)

    if len(dx) < 2:
        return np.array([activity, np.nan, np.nan], dtype=float)

    dx_var = float(np.var(dx))
    mobility = np.sqrt(dx_var / activity) if dx_var > 0 else 0.0

    ddx = np.diff(dx)

    if len(ddx) < 2 or mobility <= np.finfo(float).eps:
        complexity = 0.0
    else:
        ddx_var = float(np.var(ddx))
        mobility_dx = np.sqrt(ddx_var / dx_var) if dx_var > 0 else 0.0
        complexity = mobility_dx / mobility

    return np.array(
        [activity, mobility, complexity],
        dtype=float,
    )


def euclidean_distance(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> float:
    return float(np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2))


def detect_fixations_eyemmv_style(
    x: np.ndarray,
    y: np.ndarray,
    fs: float,
    t1: float,
    t2: float,
    min_duration_seconds: float,
) -> List[Dict[str, float]]:
    """
    Python implementation of the two-stage spatial-dispersion concept
    described by EyeMMV.

    Stage 1:
      build a cluster while samples remain within t1 of the current mean.

    Stage 2:
      remove samples farther than t2 from the cluster mean and recompute
      the fixation center/duration.

    This is an operationalization based on the cited EyeMMV method.
    The Kalaganis paper does not explicitly publish these parameter values.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 2:
        return []

    x = x[valid]
    y = y[valid]

    fixations: List[Dict[str, float]] = []
    n = len(x)
    i = 0

    while i < n:
        start = i
        mean_x = x[i]
        mean_y = y[i]
        i += 1

        while i < n:
            d = euclidean_distance(x[i], y[i], mean_x, mean_y)
            if d > t1:
                break

            cluster_x = x[start:i + 1]
            cluster_y = y[start:i + 1]
            mean_x = float(np.mean(cluster_x))
            mean_y = float(np.mean(cluster_y))
            i += 1

        end = i
        cluster_x = x[start:end]
        cluster_y = y[start:end]

        if len(cluster_x) < 2:
            continue

        center_x = float(np.mean(cluster_x))
        center_y = float(np.mean(cluster_y))

        distances = np.sqrt(
            (cluster_x - center_x) ** 2
            + (cluster_y - center_y) ** 2
        )

        keep = distances <= t2

        if keep.sum() < 2:
            continue

        kept_x = cluster_x[keep]
        kept_y = cluster_y[keep]

        duration = len(kept_x) / fs

        if duration < min_duration_seconds:
            continue

        # `start` is the onset of the first-stage fixation cluster.
        # Retain it so time-to-first-fixation can be calculated relative
        # to the beginning of the product epoch.
        fixations.append({
            "center_x": float(np.mean(kept_x)),
            "center_y": float(np.mean(kept_y)),
            "duration_seconds": float(duration),
            "start_sample": int(start),
        })

    return fixations


def et_marketing_features(
    et_segment: np.ndarray,
    fs: float,
    t1: float,
    t2: float,
    min_fixation_seconds: float,
) -> np.ndarray:
    """
    Returns four source-supported/operationalized features:
      1. fixation count
      2. mean fixation duration
      3. time to first fixation
      4. dwell time

    Blink rate is intentionally not inferred here because the supplied
    preprocessed ET representation contains interpolated gaze/pupil signals
    but no explicit blink event channel.
    """
    x, y = gaze_xy_from_et(et_segment)
    x, y = interpolate_internal_nans(x, y)

    fixations = detect_fixations_eyemmv_style(
        x=x,
        y=y,
        fs=fs,
        t1=t1,
        t2=t2,
        min_duration_seconds=min_fixation_seconds,
    )

    if not fixations:
        return np.array([0.0, 0.0, np.nan, 0.0], dtype=float)

    durations = np.array(
        [f["duration_seconds"] for f in fixations],
        dtype=float,
    )

    first_fixation_start = fixations[0]["start_sample"]
    time_to_first_fixation = first_fixation_start / fs

    return np.array(
        [
            float(len(fixations)),
            float(np.mean(durations)),
            float(time_to_first_fixation),
            float(np.sum(durations)),
        ],
        dtype=float,
    )


def process_subject(
    subject: Dict,
    lower_hz: float,
    upper_hz: float,
    filter_order: int,
    t1: float,
    t2: float,
    min_fixation_seconds: float,
) -> Dict:

    subject_id = subject["subject_id"]
    eeg_fs = float(subject["sampling_rates"]["eeg_hz"])
    et_fs = float(subject["sampling_rates"]["et_hz"])

    products = subject["products"]

    # ---------------------------------------------------------------
    # EEG graph construction
    # ---------------------------------------------------------------
    filtered_segments = []

    for product in products:
        for record in product["eeg_segments"]:
            eeg = record["data"]
            if eeg is None:
                continue

            eeg = np.asarray(eeg, dtype=float)

            filtered = bandpass_zero_phase(
                eeg,
                fs=eeg_fs,
                low_hz=lower_hz,
                high_hz=upper_hz,
                order=filter_order,
            )

            filtered_segments.append(filtered)

    if not filtered_segments:
        raise RuntimeError(
            f"{subject_id}: no valid EEG segments available."
        )

    concatenated = np.concatenate(
        filtered_segments,
        axis=1,
    )

    W = envelope_connectivity(concatenated)
    eigenvalues, U = graph_fourier_basis(W)

    # ---------------------------------------------------------------
    # Product feature extraction
    # ---------------------------------------------------------------
    feature_rows = []

    for product in products:
        eeg_segment_features = []

        for record in product["eeg_segments"]:
            eeg = record["data"]
            if eeg is None:
                continue

            eeg = bandpass_zero_phase(
                np.asarray(eeg, dtype=float),
                fs=eeg_fs,
                low_hz=lower_hz,
                high_hz=upper_hz,
                order=filter_order,
            )

            eeg_segment_features.append(
                gft_power(eeg, U)
            )

        if eeg_segment_features:
            eeg_features = aggregate_segment_features(
                eeg_segment_features
            )
        else:
            eeg_features = np.full(U.shape[1], np.nan)

        et_segment_features = []
        et_marketing_segment_features = []

        for record in product["et_segments"]:
            et = record["data"]
            if et is None:
                continue

            et = np.asarray(et, dtype=float)

            x, y = gaze_xy_from_et(et)
            x, y = interpolate_internal_nans(x, y)

            distance = accumulated_distance(x, y)

            et_segment_features.append(
                hjorth_features(distance)
            )

            et_marketing_segment_features.append(
                et_marketing_features(
                    et_segment=et,
                    fs=et_fs,
                    t1=t1,
                    t2=t2,
                    min_fixation_seconds=min_fixation_seconds,
                )
            )

        if et_segment_features:
            et_behavioral = aggregate_segment_features(
                et_segment_features
            )
            et_marketing = aggregate_segment_features(
                et_marketing_segment_features
            )
        else:
            et_behavioral = np.full(3, np.nan)
            et_marketing = np.full(4, np.nan)

        feature_vector = np.concatenate(
            [
                eeg_features,
                et_behavioral,
                et_marketing,
            ]
        )

        feature_rows.append({
            "subject_id": subject_id,
            "page": product["page"],
            "product": product["product"],
            "label": product["label"],
            "label_name": product["label_name"],
            "eeg_gft": eeg_features,
            "et_behavioral": et_behavioral,
            "et_marketing": et_marketing,
            "feature_vector": feature_vector,
        })

    return {
        "subject_id": subject_id,
        "sampling_rates": {
            "eeg_hz": eeg_fs,
            "et_hz": et_fs,
        },
        "graph": {
            "W": W,
            "eigenvalues": eigenvalues,
            "U": U,
            "construction_scope": (
                "subject-level; all retained EEG segments"
            ),
        },
        "features": feature_rows,
    }


def build_csv_rows(subject_results: List[Dict]) -> Tuple[List[str], List[List]]:
    columns = (
        ["subject_id", "page", "product", "label", "label_name"]
        + [f"gft_{i+1:02d}" for i in range(19)]
        + ["hjorth_activity", "hjorth_mobility", "hjorth_complexity"]
        + [
            "fixation_count",
            "mean_fixation_duration",
            "time_to_first_fixation",
            "dwell_time",
        ]
    )

    rows = []

    for subject in subject_results:
        for item in subject["features"]:
            eeg = item["eeg_gft"]
            beh = item["et_behavioral"]
            marketing = item["et_marketing"]

            row = [
                item["subject_id"],
                item["page"],
                item["product"],
                item["label"],
                item["label_name"],
            ]

            row.extend(eeg.tolist())
            row.extend(beh.tolist())
            row.extend(marketing.tolist())

            rows.append(row)

    return columns, rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Stage 2: GFT EEG + gaze-dynamics/ET feature extraction "
            "for the Kalaganis et al. pipeline."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="stage1_dataset.pkl",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory.",
    )

    parser.add_argument(
        "--lower-hz",
        type=float,
        default=DEFAULT_LOWER_HZ,
    )

    parser.add_argument(
        "--upper-hz",
        type=float,
        default=DEFAULT_UPPER_HZ,
    )

    parser.add_argument(
        "--filter-order",
        type=int,
        default=DEFAULT_FILTER_ORDER,
    )

    parser.add_argument(
        "--t1",
        type=float,
        default=DEFAULT_T1,
        help=(
            "EyeMMV first spatial threshold in normalized tracker "
            "coordinates. Operationalization; not explicitly stated "
            "by the Kalaganis paper."
        ),
    )

    parser.add_argument(
        "--t2",
        type=float,
        default=DEFAULT_T2,
        help=(
            "EyeMMV second spatial threshold in normalized tracker "
            "coordinates. Operationalization; not explicitly stated "
            "by the Kalaganis paper."
        ),
    )

    parser.add_argument(
        "--min-fixation-seconds",
        type=float,
        default=DEFAULT_MIN_FIXATION_SECONDS,
        help=(
            "Minimum fixation duration. EyeMMV reference operationalization "
            "uses 150 ms; exact value is not explicitly stated by "
            "the Kalaganis paper."
        ),
    )

    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    stage1 = load_pickle(args.input)

    if stage1.get("stage") != "1_dataset_epoch_preparation":
        raise ValueError(
            "Input does not appear to be the expected Stage 1 output."
        )

    subject_results = []

    for idx, subject in enumerate(stage1["subjects"], start=1):
        print(
            f"[{idx}/{len(stage1['subjects'])}] "
            f"Extracting features for {subject['subject_id']}..."
        )

        result = process_subject(
            subject=subject,
            lower_hz=args.lower_hz,
            upper_hz=args.upper_hz,
            filter_order=args.filter_order,
            t1=args.t1,
            t2=args.t2,
            min_fixation_seconds=args.min_fixation_seconds,
        )

        subject_results.append(result)

    feature_dataset = {
        "stage": "2_signal_processing_feature_construction",
        "paper": stage1["paper"],
        "config": {
            "eeg_bandpass_hz": [
                args.lower_hz,
                args.upper_hz,
            ],
            "eeg_filter_order": args.filter_order,
            "eeg_filter_phase": "zero_phase",
            "graph_connectivity": (
                "absolute Pearson correlation of Hilbert envelopes"
            ),
            "graph_laplacian": "D-W",
            "gft_power": "mean(abs(U.T @ X)**2, axis=time)",
            "et_behavioral": [
                "Hjorth Activity",
                "Hjorth Mobility",
                "Hjorth Complexity",
            ],
            "et_fixation_method": (
                "EyeMMV-style two-step spatial-dispersion "
                "operationalization"
            ),
            "eyemmv_t1": args.t1,
            "eyemmv_t2": args.t2,
            "eyemmv_min_fixation_seconds":
                args.min_fixation_seconds,
            "blink_rate": (
                "not computed: no explicit blink event channel "
                "is supplied by the documented preprocessed ET structure"
            ),
            "normalization": (
                "not performed; exact normalization is not disclosed "
                "sufficiently by the paper and belongs before CV training"
            ),
            "graph_scope": (
                "subject-level using all retained EEG segments; "
                "must be revisited for leakage-safe CV"
            ),
        },
        "subjects": subject_results,
    }

    pkl_path = args.output_dir / "stage2_features.pkl"
    graph_path = args.output_dir / "stage2_graphs.pkl"
    csv_path = args.output_dir / "stage2_features.csv"

    save_pickle(pkl_path, feature_dataset)

    graphs = {
        s["subject_id"]: s["graph"]
        for s in subject_results
    }
    save_pickle(graph_path, graphs)

    columns, rows = build_csv_rows(subject_results)

    import csv
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(rows)

    print("\nStage 2 complete.")
    print(f"Feature dataset : {pkl_path}")
    print(f"Graphs          : {graph_path}")
    print(f"CSV             : {csv_path}")
    print(
        "\nNOTE: This is a feature-construction stage. "
        "No CV, feature selection, normalization, SVM, or evaluation "
        "has been performed."
    )


if __name__ == "__main__":
    main()
