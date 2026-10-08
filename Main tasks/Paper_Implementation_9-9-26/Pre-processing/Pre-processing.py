#!/usr/bin/env python3
"""
01_NeuMa_Preprocessing.py

Stage 1 — prepare the actual NeuMa subject .mat files for the paper pipeline.

The supplied subject files (verified on S13.mat and S19.mat) contain:
  - EEG_clean.Data: 19 EEG channels, Fs=300 Hz
  - ET_clean.Data: 6 ET channels, Fs=120 Hz
  - Page1..Page6 / Product1..Product24
  - ProductInfo.Bought
  - EEG_segments / ET_segments
  - EEG channel metadata in EEG_clean.chans

The NeuMa dataset documentation states that EEG has already undergone:
  0.5–45 Hz 3rd-order zero-phase Butterworth filtering, ASR and FORCe.
ET blink-related missing samples have already been linearly interpolated.

Therefore this script DOES NOT repeat those operations.

It:
  1. discovers S*.mat files;
  2. loads the actual subject structures;
  3. verifies the expected 19-channel EEG structure;
  4. converts MATLAB inclusive [start,end] indices to Python [start-1:end];
  5. discards segments shorter than 0.5 s;
  6. preserves signal arrays, channel names and metadata;
  7. writes one portable stage-1 pickle.

No feature selection, normalization, graph construction, filtering, or ML
takes place here.
"""

from __future__ import annotations

import argparse
import pickle
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from scipy.io import loadmat

EXPECTED_EEG_CHANNELS = 19
EXPECTED_ET_CHANNELS = 6
EXPECTED_EEG_FS = 300.0
EXPECTED_ET_FS = 120.0
EXPECTED_PAGES = 6
EXPECTED_PRODUCTS = 24
MIN_SEGMENT_SECONDS = 0.5


def mat_to_python(obj: Any) -> Any:
    if hasattr(obj, "_fieldnames"):
        return {k: mat_to_python(getattr(obj, k)) for k in obj._fieldnames}
    if isinstance(obj, np.ndarray):
        if obj.dtype.names is not None:
            return [mat_to_python(x) for x in obj]
        if obj.dtype == object:
            if obj.ndim == 0:
                return mat_to_python(obj.item())
            return np.array(
                [mat_to_python(x) for x in obj.flat], dtype=object
            ).reshape(obj.shape)
        return obj
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_subject(path: Path) -> Dict[str, Any]:
    raw = loadmat(str(path), squeeze_me=True, struct_as_record=False)
    return {
        k: mat_to_python(v)
        for k, v in raw.items()
        if not k.startswith("__")
    }


def unwrap(value: Any) -> Any:
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def field(obj: Any, name: str) -> Any:
    obj = unwrap(obj)
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise KeyError(f"Missing field '{name}'")


def nested_field(obj: Any, *names: str) -> Any:
    for name in names:
        obj = field(obj, name)
    return obj


def scalar(value: Any, name: str) -> float:
    value = unwrap(value)
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} is not scalar: shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def labels_from_chans(chans: Any) -> Optional[List[str]]:
    """
    Extract channel labels from the actual MATLAB chan metadata.
    Returns None only if the representation cannot be interpreted safely.
    """
    chans = unwrap(chans)

    def as_string(x: Any) -> str:
        x = unwrap(x)
        if isinstance(x, bytes):
            return x.decode(errors="ignore").strip()
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, np.ndarray):
            if x.dtype.kind in {"U", "S"}:
                if x.ndim == 0:
                    return str(x.item()).strip()
                return "".join(str(v) for v in x.flat).strip()
            if x.size == 1:
                return as_string(x.reshape(-1)[0])
        return str(x).strip()

    if isinstance(chans, dict):
        if "labels" in chans:
            return labels_from_chans(chans["labels"])
        return None

    if isinstance(chans, (list, tuple)):
        out = []
        for x in chans:
            if isinstance(x, dict) and "labels" in x:
                x = x["labels"]
            out.append(as_string(x))
        return out if out else None

    if isinstance(chans, np.ndarray):
        if chans.dtype == object:
            out = []
            for x in chans.flat:
                if isinstance(x, dict) and "labels" in x:
                    x = x["labels"]
                out.append(as_string(x))
            return out if out else None
        if chans.dtype.kind in {"U", "S"}:
            # Character arrays can represent one label or a character matrix.
            if chans.ndim == 0:
                return [as_string(chans)]
            if chans.ndim == 1:
                return [as_string(x) for x in chans]
            return ["".join(str(x) for x in row).strip()
                    for row in chans]
    return None


def segment_pairs(value: Any, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.size == 0:
        return np.empty((0, 2), dtype=int)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(f"{name}: expected [start,end]")
        arr = arr.reshape(1, 2)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name}: unexpected shape {arr.shape}")
    return arr.astype(int)


def extract_segment(data: np.ndarray, start: int, end: int,
                    name: str) -> np.ndarray:
    if start < 1 or end < start:
        raise ValueError(f"{name}: invalid MATLAB interval [{start},{end}]")
    if end > data.shape[1]:
        raise ValueError(
            f"{name}: end={end} exceeds data length={data.shape[1]}"
        )
    # MATLAB start:end is inclusive.
    # Python equivalent is [start-1:end].
    return np.asarray(data[:, start - 1:end], dtype=np.float64)


def prepare_subject(path: Path) -> Dict[str, Any]:
    root = load_subject(path)
    subject = root.get(path.stem)
    if subject is None:
        # Fallback only if there is exactly one non-meta variable.
        vals = list(root.values())
        if len(vals) != 1:
            raise ValueError(f"Could not identify subject struct in {path}")
        subject = vals[0]

    eeg = field(subject, "EEG_clean")
    et = field(subject, "ET_clean")

    eeg_data = np.asarray(field(eeg, "Data"), dtype=float)
    et_data = np.asarray(field(et, "Data"), dtype=float)
    eeg_fs = scalar(field(eeg, "Fs"), "EEG Fs")
    et_fs = scalar(field(et, "Fs"), "ET Fs")

    if eeg_data.ndim != 2:
        raise ValueError(f"{path.name}: EEG Data must be [channels,samples]")
    if et_data.ndim != 2:
        raise ValueError(f"{path.name}: ET Data must be [channels,samples]")

    if eeg_data.shape[0] != EXPECTED_EEG_CHANNELS:
        raise ValueError(
            f"{path.name}: expected 19 EEG channels, found "
            f"{eeg_data.shape[0]}"
        )
    if et_data.shape[0] != EXPECTED_ET_CHANNELS:
        raise ValueError(
            f"{path.name}: expected 6 ET channels, found "
            f"{et_data.shape[0]}"
        )
    if not np.isclose(eeg_fs, EXPECTED_EEG_FS):
        raise ValueError(f"{path.name}: unexpected EEG Fs={eeg_fs}")
    if not np.isclose(et_fs, EXPECTED_ET_FS):
        raise ValueError(f"{path.name}: unexpected ET Fs={et_fs}")

    try:
        channel_names = labels_from_chans(field(eeg, "chans"))
    except KeyError:
        channel_names = None

    if channel_names is None or len(channel_names) != EXPECTED_EEG_CHANNELS:
        raise ValueError(
            f"{path.name}: could not safely recover all 19 EEG channel labels."
        )

    products = []
    counts = {
        "products_total": 0,
        "products_with_eeg": 0,
        "products_with_et": 0,
        "products_with_both": 0,
        "eeg_segments_total": 0,
        "eeg_segments_kept": 0,
        "et_segments_total": 0,
        "et_segments_kept": 0,
    }

    for page_no in range(1, EXPECTED_PAGES + 1):
        page = field(subject, f"Page{page_no}")

        for product_no in range(1, EXPECTED_PRODUCTS + 1):
            product = field(page, f"Product{product_no}")
            counts["products_total"] += 1

            bought = scalar(
                nested_field(product, "ProductInfo", "Bought"),
                "ProductInfo.Bought"
            )
            if bought not in (0.0, 1.0):
                raise ValueError(
                    f"{path.name} Page{page_no} Product{product_no}: "
                    f"Bought={bought}, expected 0/1"
                )

            eeg_idx = segment_pairs(
                field(product, "EEG_segments"),
                f"{path.name} Page{page_no} Product{product_no} EEG_segments"
            )
            et_idx = segment_pairs(
                field(product, "ET_segments"),
                f"{path.name} Page{page_no} Product{product_no} ET_segments"
            )

            counts["eeg_segments_total"] += len(eeg_idx)
            counts["et_segments_total"] += len(et_idx)

            eeg_segments = []
            for start, end in eeg_idx:
                duration = (end - start + 1) / eeg_fs
                if duration < MIN_SEGMENT_SECONDS:
                    continue
                eeg_segments.append({
                    "start_matlab": int(start),
                    "end_matlab": int(end),
                    "start_python": int(start - 1),
                    "end_python_exclusive": int(end),
                    "duration_seconds": float(duration),
                    "data": extract_segment(
                        eeg_data, start, end,
                        f"{path.name} EEG"
                    ),
                })
                counts["eeg_segments_kept"] += 1

            et_segments = []
            for start, end in et_idx:
                duration = (end - start + 1) / et_fs
                if duration < MIN_SEGMENT_SECONDS:
                    continue
                et_segments.append({
                    "start_matlab": int(start),
                    "end_matlab": int(end),
                    "start_python": int(start - 1),
                    "end_python_exclusive": int(end),
                    "duration_seconds": float(duration),
                    "data": extract_segment(
                        et_data, start, end,
                        f"{path.name} ET"
                    ),
                })
                counts["et_segments_kept"] += 1

            if eeg_segments:
                counts["products_with_eeg"] += 1
            if et_segments:
                counts["products_with_et"] += 1
            if eeg_segments and et_segments:
                counts["products_with_both"] += 1

            products.append({
                "page": page_no,
                "product": product_no,
                "label": int(bought),
                "eeg_segments": eeg_segments,
                "et_segments": et_segments,
            })

    return {
        "subject_id": path.stem,
        "source_file": str(path.resolve()),
        "eeg_fs": eeg_fs,
        "et_fs": et_fs,
        "eeg_channel_names": channel_names,
        "eeg_n_channels": int(eeg_data.shape[0]),
        "et_n_channels": int(et_data.shape[0]),
        "products": products,
        "counts": counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    subject_files = sorted(
        [
            p for p in args.data_dir.glob("S*.mat")
            if re.fullmatch(r"S\d+\.mat", p.name, flags=re.I)
        ],
        key=lambda p: int(re.search(r"\d+", p.stem).group())
    )
    if not subject_files:
        raise FileNotFoundError(f"No S*.mat files found in {args.data_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    subjects = []
    for path in subject_files:
        print(f"Preparing {path.name}")
        subjects.append(prepare_subject(path))

    artifact = {
        "stage": 1,
        "subjects": subjects,
        "dataset_constants": {
            "eeg_fs": EXPECTED_EEG_FS,
            "et_fs": EXPECTED_ET_FS,
            "eeg_channels": EXPECTED_EEG_CHANNELS,
            "et_channels": EXPECTED_ET_CHANNELS,
            "min_segment_seconds": MIN_SEGMENT_SECONDS,
        },
        "dataset_provided_preprocessing": {
            "eeg": "0.5-45 Hz 3rd-order zero-phase Butterworth + ASR + FORCe",
            "et": "blink-related missing samples linearly interpolated",
        },
        "stage1_operations": [
            "load actual subject .mat files",
            "verify 19 EEG / 6 ET channels",
            "convert MATLAB inclusive segment indices",
            "discard segments shorter than 0.5 s",
            "preserve channel metadata",
        ],
    }

    output = args.output_dir / "stage1_dataset.pkl"
    with output.open("wb") as f:
        pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
