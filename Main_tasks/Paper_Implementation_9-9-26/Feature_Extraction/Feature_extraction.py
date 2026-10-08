#!/usr/bin/env python3
"""
02_NeuMa_Feature_Extraction.py

Stage 2 — Kalaganis et al. feature extraction from the verified NeuMa
subject-level representation.

EEG:
  - 8–30 Hz, 3rd-order zero-phase Butterworth
  - 19 EEG channels exactly as present in the subject files
  - Hilbert envelope
  - W_mn = |Pearson correlation(envelope_m, envelope_n)|
  - L = D - W
  - eigendecomposition of L
  - graph Fourier transform X_tilde = U.T @ X
  - feature = mean time-averaged power of each graph mode
  - arithmetic mean across segments for each product

ET:
  - accumulated gaze trajectory distance
  - Hjorth Activity, Mobility, Complexity
  - EyeMMV-style fixation detection
  - fixation count
  - mean fixation duration
  - TTFF
  - dwell time
  - blink rate intentionally omitted

Replication caveats:
  - The paper does not disclose exact EyeMMV thresholds. The values used here
    are a documented EyeMMV operationalization, not a claim about hidden
    authors' parameters.
  - The paper states that ET features were normalized and averaged, but does
    not publish the exact normalization formula. This implementation therefore
    makes the normalization choice explicit and configurable.
  - The subject graph is built from all retained EEG segments before CV.
    This matches the personalized-graph interpretation more closely than
    rebuilding an unspecified graph per fold, but it is not strict
    fold-isolated preprocessing.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.signal import butter, filtfilt, hilbert

LOW_HZ = 8.0
HIGH_HZ = 30.0
FILTER_ORDER = 3

# EyeMMV-style operationalization; not claimed as a hidden Kalaganis setting.
EYE_T1 = 0.250
EYE_T2 = 0.100
MIN_FIXATION_SECONDS = 0.150

EXPECTED_EEG_CHANNELS = 19


def load_pickle(path: Path) -> Any:
    with path.open("rb") as f:
        return pickle.load(f)


def save_pickle(path: Path, obj: Any) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def bandpass(x: np.ndarray, fs: float) -> np.ndarray:
    nyq = fs / 2.0
    if HIGH_HZ >= nyq:
        raise ValueError(f"EEG Fs={fs} is too low for a 30 Hz upper cutoff.")
    b, a = butter(
        FILTER_ORDER,
        [LOW_HZ / nyq, HIGH_HZ / nyq],
        btype="bandpass"
    )
    padlen = min(3 * max(len(a), len(b)), x.shape[-1] - 1)
    if padlen < 1:
        raise ValueError("Segment is too short for zero-phase filtering.")
    return filtfilt(b, a, x, axis=-1, padlen=padlen)


def build_subject_graph(
    eeg_segments: List[np.ndarray],
    fs: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct the personalized subject graph from all retained EEG segments.

    The graph is label-independent, but because all subject segments are used
    before CV, it is not strict test-fold-isolated preprocessing.
    """
    if not eeg_segments:
        raise ValueError("No EEG segments available for graph construction.")

    envelopes = []
    for x in eeg_segments:
        x8_30 = bandpass(x, fs)
        envelopes.append(np.abs(hilbert(x8_30, axis=1)))

    pooled = np.concatenate(envelopes, axis=1)
    W = np.abs(np.corrcoef(pooled))
    W = np.nan_to_num(W, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(W, 0.0)
    W = (W + W.T) / 2.0

    degree = W.sum(axis=1)
    L = np.diag(degree) - W
    eigenvalues, U = np.linalg.eigh(L)
    return W, eigenvalues, U


def gft_feature(x: np.ndarray, U: np.ndarray, fs: float) -> np.ndarray:
    x8_30 = bandpass(x, fs)
    gft = U.T @ x8_30
    return np.mean(np.abs(gft) ** 2, axis=1)


def gaze_xy(et: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # ET rows: left X/Y/pupil, right X/Y/pupil.
    x = np.nanmean(et[[0, 3], :], axis=0)
    y = np.nanmean(et[[1, 4], :], axis=0)

    # NeuMa already interpolates blink-related missing samples. These checks
    # only provide numerical robustness if residual NaNs remain.
    for arr in (x, y):
        if np.any(~np.isfinite(arr)):
            good = np.isfinite(arr)
            if good.sum() < 2:
                return np.zeros(len(x)), np.zeros(len(y))

    if np.any(~np.isfinite(x)):
        idx = np.arange(len(x))
        good = np.isfinite(x)
        x = np.interp(idx, idx[good], x[good])
    if np.any(~np.isfinite(y)):
        idx = np.arange(len(y))
        good = np.isfinite(y)
        y = np.interp(idx, idx[good], y[good])

    return x, y


def trajectory_distance(et: np.ndarray) -> np.ndarray:
    x, y = gaze_xy(et)
    d = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
    return np.concatenate([[0.0], d])


def hjorth(x: np.ndarray) -> Tuple[float, float, float]:
    var_x = float(np.var(x))
    if var_x <= 0:
        return 0.0, 0.0, 0.0

    dx = np.diff(x)
    var_dx = float(np.var(dx)) if len(dx) else 0.0
    mobility = np.sqrt(var_dx / var_x)

    ddx = np.diff(dx)
    var_ddx = float(np.var(ddx)) if len(ddx) else 0.0
    mobility_dx = (
        np.sqrt(var_ddx / var_dx) if var_dx > 0 else 0.0
    )
    complexity = (
        mobility_dx / mobility if mobility > 0 else 0.0
    )
    return var_x, float(mobility), float(complexity)


def fixation_intervals(
    et: np.ndarray,
    fs: float,
    t1: float = EYE_T1,
    t2: float = EYE_T2,
    minimum_seconds: float = MIN_FIXATION_SECONDS
) -> List[Tuple[int, int]]:
    """
    Dispersion-based EyeMMV-style fixation detector.

    The paper cites EyeMMV but does not disclose exact detector parameters.
    """
    x, y = gaze_xy(et)
    n = len(x)
    minimum_samples = max(1, int(round(minimum_seconds * fs)))
    out = []

    i = 0
    while i + minimum_samples <= n:
        j = i + minimum_samples - 1
        accepted = False

        while j < n:
            wx = x[i:j + 1]
            wy = y[i:j + 1]
            if np.ptp(wx) <= t1 and np.ptp(wy) <= t2:
                accepted = True
                j += 1
            else:
                break

        if accepted:
            end = j - 1
            out.append((i, end))
            i = end + 1
        else:
            i += 1

    return out


def et_features(
    et: np.ndarray,
    fs: float,
    normalization: str
) -> Dict[str, float]:
    d = trajectory_distance(et)
    activity, mobility, complexity = hjorth(d)

    fixations = fixation_intervals(et, fs)
    durations = np.asarray(
        [(b - a + 1) / fs for a, b in fixations],
        dtype=float
    )

    duration = len(et[0]) / fs
    fixation_count = float(len(fixations))
    mean_fixation_duration = (
        float(np.mean(durations)) if len(durations) else 0.0
    )
    ttff = (
        float(fixations[0][0] / fs) if fixations else np.nan
    )
    dwell = float(np.sum(durations))

    if normalization == "duration":
        # Transparent operationalization of the paper's statement that ET
        # measures were normalized to handle varying epoch duration.
        ttff_value = (
            ttff / duration
            if np.isfinite(ttff) and duration > 0 else np.nan
        )
        dwell_value = dwell / duration if duration > 0 else 0.0
    else:
        ttff_value = ttff
        dwell_value = dwell

    return {
        "ET_activity": activity,
        "ET_mobility": mobility,
        "ET_complexity": complexity,
        "ET_fixation_count": fixation_count,
        "ET_mean_fixation_duration": mean_fixation_duration,
        "ET_TTFF": ttff_value,
        "ET_dwell_time": dwell_value,
    }


def finite_mean(values: List[float], default: float = 0.0) -> float:
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if len(a) else default


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--et-normalization",
        choices=["duration", "none"],
        default="duration"
    )
    args = parser.parse_args()

    data = load_pickle(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    feature_names = [
        *(f"EEG_GFT_mode_{i:02d}" for i in range(EXPECTED_EEG_CHANNELS)),
        "ET_activity",
        "ET_mobility",
        "ET_complexity",
        "ET_fixation_count",
        "ET_mean_fixation_duration",
        "ET_TTFF",
        "ET_dwell_time",
    ]

    subjects_out = []

    for subject in data["subjects"]:
        sid = subject["subject_id"]
        if subject["eeg_n_channels"] != EXPECTED_EEG_CHANNELS:
            raise ValueError(
                f"{sid}: expected exactly 19 EEG channels."
            )

        eeg_fs = float(subject["eeg_fs"])
        et_fs = float(subject["et_fs"])

        all_eeg_segments = [
            np.asarray(seg["data"], dtype=float)
            for product in subject["products"]
            for seg in product["eeg_segments"]
        ]

        W, eigenvalues, U = build_subject_graph(
            all_eeg_segments, eeg_fs
        )

        product_rows = []
        for product in subject["products"]:
            eeg_features = []
            for seg in product["eeg_segments"]:
                x = np.asarray(seg["data"], dtype=float)
                eeg_features.append(gft_feature(x, U, eeg_fs))

            if not eeg_features:
                continue

            eeg_vector = np.mean(np.vstack(eeg_features), axis=0)

            et_features_list = [
                et_features(
                    np.asarray(seg["data"], dtype=float),
                    et_fs,
                    args.et_normalization
                )
                for seg in product["et_segments"]
            ]

            if et_features_list:
                et_vector = [
                    finite_mean([f["ET_activity"] for f in et_features_list]),
                    finite_mean([f["ET_mobility"] for f in et_features_list]),
                    finite_mean([f["ET_complexity"] for f in et_features_list]),
                    finite_mean([f["ET_fixation_count"] for f in et_features_list]),
                    finite_mean([f["ET_mean_fixation_duration"] for f in et_features_list]),
                    finite_mean([f["ET_TTFF"] for f in et_features_list]),
                    finite_mean([f["ET_dwell_time"] for f in et_features_list]),
                ]
            else:
                et_vector = [0.0] * 7

            product_rows.append({
                "subject_id": sid,
                "page": int(product["page"]),
                "product": int(product["product"]),
                "label": int(product["label"]),
                "features": np.concatenate(
                    [eeg_vector, np.asarray(et_vector, dtype=float)]
                ),
            })

        subjects_out.append({
            "subject_id": sid,
            "eeg_channel_names_used": subject["eeg_channel_names"],
            "graph": {
                "scope": "subject_all_retained_EEG_segments",
                "adjacency": W,
                "laplacian_eigenvalues": eigenvalues,
                "eigenvectors": U,
            },
            "rows": product_rows,
        })

    artifact = {
        "stage": 2,
        "feature_names": feature_names,
        "subjects": subjects_out,
        "methodology": {
            "EEG_filter": "8-30 Hz, 3rd-order zero-phase Butterworth",
            "EEG_nodes": 19,
            "A1_A2_handling": (
                "not present as additional EEG channels in the verified "
                "subject files; no channel removal is performed"
            ),
            "graph": "W=absolute Pearson correlation of Hilbert envelopes",
            "laplacian": "L=D-W",
            "GFT": "U.T @ X",
            "EEG_feature": "mean time-averaged GFT power",
            "EEG_segment_aggregation": "arithmetic mean per product",
            "ET_features": [
                "trajectory distance",
                "Hjorth Activity",
                "Hjorth Mobility",
                "Hjorth Complexity",
                "fixation count",
                "mean fixation duration",
                "TTFF",
                "dwell time",
            ],
            "fixation_parameters": {
                "t1": EYE_T1,
                "t2": EYE_T2,
                "minimum_fixation_seconds": MIN_FIXATION_SECONDS,
                "status": "operationalization; exact paper settings undisclosed",
            },
            "et_normalization": args.et_normalization,
            "blink_rate": "omitted for now",
            "graph_cv_isolation": (
                "not strict: graph uses all retained subject EEG segments "
                "before CV"
            ),
        },
    }

    output = args.output_dir / "stage2_features.pkl"
    save_pickle(output, artifact)

    csv_path = args.output_dir / "stage2_features.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(
            "subject_id,page,product,label,"
            + ",".join(feature_names) + "\n"
        )
        for subject in subjects_out:
            for row in subject["rows"]:
                values = ",".join(
                    f"{float(v):.12g}" for v in row["features"]
                )
                f.write(
                    f'{row["subject_id"]},{row["page"]},'
                    f'{row["product"]},{row["label"]},{values}\n'
                )

    graph_path = args.output_dir / "stage2_graphs.pkl"
    save_pickle(
        graph_path,
        {s["subject_id"]: s["graph"] for s in subjects_out}
    )

    print(f"Wrote {output}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {graph_path}")


if __name__ == "__main__":
    main()
