#!/usr/bin/env python3
"""
Stage 1 — NeuMa dataset access and epoch preparation
Implements only points 1–6 of the Kalaganis et al. (2025) pipeline:
  1) load NeuMa preprocessed subject files
  2) enumerate pages/products
  3) read product EEG/ET segment indices
  4) discard segments shorter than 0.5 s
  5) read Buy/NoBuy labels
  6) build a product-level intermediate representation for later stages

No filtering, GFT, Hjorth features, normalization, feature selection, CV, or
classification is performed here.

Expected input:
    <data_dir>/S01.mat, S02.mat, ...

Outputs:
    stage1_dataset.pkl   Variable-length Python representation with actual
                         EEG/ET segments plus metadata and original indices.
    stage1_summary.csv   Human-readable validation/summary table.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
from scipy.io import loadmat


EXPECTED_PAGES = 6
EXPECTED_PRODUCTS_PER_PAGE = 24
MIN_SEGMENT_SECONDS = 0.5


def _mat_to_python(obj: Any) -> Any:
    """Recursively convert scipy.io MATLAB objects into Python containers."""
    if hasattr(obj, "_fieldnames"):
        return {
            name: _mat_to_python(getattr(obj, name))
            for name in obj._fieldnames
        }

    if isinstance(obj, np.ndarray):
        if obj.dtype.names is not None:
            return [_mat_to_python(item) for item in obj]

        if obj.dtype == object:
            if obj.ndim == 0:
                return _mat_to_python(obj.item())
            return np.array(
                [_mat_to_python(x) for x in obj.flat], dtype=object
            ).reshape(obj.shape)

        return obj

    if isinstance(obj, np.generic):
        return obj.item()

    return obj


def _load_subject_mat(path: Path) -> Dict[str, Any]:
    """Load a standard MATLAB .mat subject file."""
    try:
        raw = loadmat(
            str(path),
            squeeze_me=True,
            struct_as_record=False,
        )
    except NotImplementedError as exc:
        raise RuntimeError(
            f"{path.name} appears to be MATLAB v7.3/HDF5 format. "
            "This script currently expects standard .mat files readable "
            "by scipy.io.loadmat."
        ) from exc

    return {
        key: _mat_to_python(value)
        for key, value in raw.items()
        if not key.startswith("__")
    }


def _unwrap_scalar(value: Any) -> Any:
    """Unwrap singleton MATLAB/object arrays."""
    while isinstance(value, np.ndarray) and value.size == 1:
        value = value.reshape(-1)[0]
    return value


def _get_field(obj: Any, field: str) -> Any:
    obj = _unwrap_scalar(obj)
    if isinstance(obj, dict):
        if field not in obj:
            raise KeyError(field)
        return obj[field]
    raise TypeError(
        f"Expected MATLAB struct/dict for field '{field}', got {type(obj)}"
    )


def _get_nested_field(obj: Any, *fields: str) -> Any:
    for field in fields:
        obj = _get_field(obj, field)
    return obj


def _as_float_scalar(value: Any, name: str) -> float:
    value = _unwrap_scalar(value)
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{name} is not scalar; shape={arr.shape}")
    return float(arr.reshape(-1)[0])


def _as_segment_matrix(value: Any, name: str) -> np.ndarray:
    """
    Convert MATLAB [N x 2] start/end indices into integer indices.

    The returned indices remain MATLAB-style (1-based, inclusive).
    """
    value = _unwrap_scalar(value)

    if value is None:
        return np.empty((0, 2), dtype=np.int64)

    arr = np.asarray(value)
    if arr.size == 0:
        return np.empty((0, 2), dtype=np.int64)

    if arr.dtype == object:
        try:
            arr = np.asarray(arr.tolist(), dtype=float)
        except Exception as exc:
            raise ValueError(
                f"Cannot parse {name} as numeric segment indices"
            ) from exc
    else:
        arr = arr.astype(float, copy=False)

    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError(
                f"{name} expected [N x 2], got shape {arr.shape}"
            )
        arr = arr.reshape(1, 2)

    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(
            f"{name} expected [N x 2], got shape {arr.shape}"
        )

    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite indices")

    rounded = np.rint(arr).astype(np.int64)
    if not np.allclose(arr, rounded):
        raise ValueError(f"{name} contains non-integer sample indices")

    return rounded


def _get_subject_struct(
    raw: Dict[str, Any], subject_id: str
) -> Dict[str, Any]:
    """Find the subject struct, normally stored under SXX."""
    if subject_id in raw and isinstance(raw[subject_id], dict):
        return raw[subject_id]

    candidates = [
        (key, value)
        for key, value in raw.items()
        if isinstance(value, dict)
    ]

    if len(candidates) == 1:
        key, value = candidates[0]
        warnings.warn(
            f"{subject_id}: top-level variable is '{key}', not "
            f"'{subject_id}'. Using the only MATLAB struct found."
        )
        return value

    raise KeyError(
        f"Could not find subject struct '{subject_id}' in "
        f"{subject_id}.mat. Top-level variables: {list(raw.keys())}"
    )


def _iter_subject_ids(data_dir: Path) -> List[str]:
    files = sorted(
        p for p in data_dir.glob("S*.mat")
        if p.is_file() and p.stem[1:].isdigit()
    )
    return [p.stem for p in files]


def _segment_duration_seconds(
    start_matlab: int, end_matlab: int, fs: float
) -> float:
    # MATLAB start:end is inclusive, hence +1 sample.
    return (end_matlab - start_matlab + 1) / fs


def _extract_segment(
    data: np.ndarray,
    start_matlab: int,
    end_matlab: int,
    name: str,
) -> np.ndarray:
    """
    Equivalent to MATLAB:
        data(:, start:end)

    using Python:
        data[:, start-1:end]
    """
    arr = np.asarray(data)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be 2-D [channels x samples], got {arr.shape}"
        )

    if start_matlab < 1 or end_matlab < start_matlab:
        raise ValueError(
            f"Invalid {name} segment indices: "
            f"{start_matlab}, {end_matlab}"
        )

    n_samples = arr.shape[1]
    if end_matlab > n_samples:
        raise IndexError(
            f"{name} segment [{start_matlab}, {end_matlab}] exceeds "
            f"available samples 1..{n_samples}"
        )

    return np.asarray(
        arr[:, start_matlab - 1:end_matlab],
        dtype=np.float64,
    )


def _safe_label(value: Any) -> int:
    value = _as_float_scalar(value, "ProductInfo.Bought")
    if value not in (0.0, 1.0):
        raise ValueError(
            f"ProductInfo.Bought must be 0 or 1, got {value}"
        )
    return int(value)


def _process_subject(
    mat_path: Path,
    min_segment_seconds: float,
    keep_signal_segments: bool = True,
) -> Dict[str, Any]:

    subject_id = mat_path.stem
    raw = _load_subject_mat(mat_path)
    subject = _get_subject_struct(raw, subject_id)

    eeg_clean = _get_field(subject, "EEG_clean")
    et_clean = _get_field(subject, "ET_clean")

    eeg_data = np.asarray(
        _get_field(eeg_clean, "Data"), dtype=np.float64
    )
    et_data = np.asarray(
        _get_field(et_clean, "Data"), dtype=np.float64
    )

    eeg_fs = _as_float_scalar(
        _get_field(eeg_clean, "Fs"), "EEG_clean.Fs"
    )
    et_fs = _as_float_scalar(
        _get_field(et_clean, "Fs"), "ET_clean.Fs"
    )

    if eeg_data.ndim != 2:
        raise ValueError(
            f"{subject_id}.EEG_clean.Data must be 2-D, "
            f"got {eeg_data.shape}"
        )

    if et_data.ndim != 2:
        raise ValueError(
            f"{subject_id}.ET_clean.Data must be 2-D, "
            f"got {et_data.shape}"
        )

    products: List[Dict[str, Any]] = []

    counts = {
        "products_total": 0,
        "products_with_valid_eeg": 0,
        "products_with_valid_et": 0,
        "products_with_both_valid": 0,
        "products_with_no_valid_segment": 0,
        "eeg_segments_total": 0,
        "eeg_segments_kept": 0,
        "et_segments_total": 0,
        "et_segments_kept": 0,
    }

    for page_no in range(1, EXPECTED_PAGES + 1):
        page_name = f"Page{page_no}"
        page = _get_field(subject, page_name)

        for product_no in range(1, EXPECTED_PRODUCTS_PER_PAGE + 1):
            product_name = f"Product{product_no}"
            product = _get_field(page, product_name)

            counts["products_total"] += 1

            bought = _safe_label(
                _get_nested_field(product, "ProductInfo", "Bought")
            )

            eeg_idx = _as_segment_matrix(
                _get_field(product, "EEG_segments"),
                f"{subject_id}.{page_name}.{product_name}."
                "EEG_segments",
            )

            et_idx = _as_segment_matrix(
                _get_field(product, "ET_segments"),
                f"{subject_id}.{page_name}.{product_name}."
                "ET_segments",
            )

            counts["eeg_segments_total"] += len(eeg_idx)
            counts["et_segments_total"] += len(et_idx)

            eeg_records = []

            for start, end in eeg_idx:
                duration = _segment_duration_seconds(
                    int(start), int(end), eeg_fs
                )

                if duration >= min_segment_seconds:
                    segment = (
                        _extract_segment(
                            eeg_data,
                            int(start),
                            int(end),
                            f"{subject_id}.{page_name}."
                            f"{product_name}.EEG",
                        )
                        if keep_signal_segments
                        else None
                    )

                    eeg_records.append({
                        "start_sample_matlab": int(start),
                        "end_sample_matlab": int(end),
                        "start_sample_python": int(start - 1),
                        "end_sample_python_exclusive": int(end),
                        "duration_seconds": float(duration),
                        "data": segment,
                    })

                    counts["eeg_segments_kept"] += 1

            et_records = []

            for start, end in et_idx:
                duration = _segment_duration_seconds(
                    int(start), int(end), et_fs
                )

                if duration >= min_segment_seconds:
                    segment = (
                        _extract_segment(
                            et_data,
                            int(start),
                            int(end),
                            f"{subject_id}.{page_name}."
                            f"{product_name}.ET",
                        )
                        if keep_signal_segments
                        else None
                    )

                    et_records.append({
                        "start_sample_matlab": int(start),
                        "end_sample_matlab": int(end),
                        "start_sample_python": int(start - 1),
                        "end_sample_python_exclusive": int(end),
                        "duration_seconds": float(duration),
                        "data": segment,
                    })

                    counts["et_segments_kept"] += 1

            has_eeg = bool(eeg_records)
            has_et = bool(et_records)

            if has_eeg:
                counts["products_with_valid_eeg"] += 1
            if has_et:
                counts["products_with_valid_et"] += 1
            if has_eeg and has_et:
                counts["products_with_both_valid"] += 1
            if not has_eeg and not has_et:
                counts["products_with_no_valid_segment"] += 1

            description = None
            try:
                description = _unwrap_scalar(
                    _get_nested_field(
                        product, "ProductInfo", "Description"
                    )
                )
            except (KeyError, TypeError):
                pass

            products.append({
                "subject_id": subject_id,
                "page": page_no,
                "product": product_no,
                "label": bought,
                "label_name": "Buy" if bought == 1 else "NoBuy",
                "description": description,
                "eeg_segments": eeg_records,
                "et_segments": et_records,
            })

    return {
        "subject_id": subject_id,
        "sampling_rates": {
            "eeg_hz": eeg_fs,
            "et_hz": et_fs,
        },
        "source_file": mat_path.name,
        "eeg_shape": tuple(eeg_data.shape),
        "et_shape": tuple(et_data.shape),
        "products": products,
        "counts": counts,
    }


def _write_summary(
    path: Path, subjects: Iterable[Dict[str, Any]]
) -> None:

    fieldnames = [
        "subject_id",
        "products_total",
        "products_with_valid_eeg",
        "products_with_valid_et",
        "products_with_both_valid",
        "products_with_no_valid_segment",
        "eeg_segments_total",
        "eeg_segments_kept",
        "et_segments_total",
        "et_segments_kept",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for subject in subjects:
            row = {"subject_id": subject["subject_id"]}
            row.update(subject["counts"])
            writer.writerow(row)


def _validate_stage1(dataset: Dict[str, Any]) -> Dict[str, Any]:
    """Run internal consistency checks without modifying the data."""

    subjects = dataset["subjects"]

    checks = {
        "subject_count": len(subjects),
        "all_product_counts_144": True,
        "all_labels_binary": True,
        "all_kept_segments_meet_threshold": True,
        "all_segment_arrays_2d": True,
    }

    for subject in subjects:
        products = subject["products"]

        if len(products) != EXPECTED_PAGES * EXPECTED_PRODUCTS_PER_PAGE:
            checks["all_product_counts_144"] = False

        for product in products:
            if product["label"] not in (0, 1):
                checks["all_labels_binary"] = False

            for modality in ("eeg_segments", "et_segments"):
                for record in product[modality]:
                    if (
                        record["duration_seconds"]
                        < dataset["config"]["min_segment_seconds"]
                    ):
                        checks["all_kept_segments_meet_threshold"] = False

                    if (
                        record["data"] is not None
                        and np.asarray(record["data"]).ndim != 2
                    ):
                        checks["all_segment_arrays_2d"] = False

    checks["passed"] = all(
        value is True
        for key, value in checks.items()
        if key != "subject_count"
    )

    return checks


def main() -> None:

    parser = argparse.ArgumentParser(
        description="NeuMa Stage 1: dataset access and epoch preparation."
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
        help="Directory containing SXX.mat subject files.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for stage1_dataset.pkl and stage1_summary.csv.",
    )

    parser.add_argument(
        "--min-segment-seconds",
        type=float,
        default=MIN_SEGMENT_SECONDS,
        help="Minimum retained segment duration in seconds (default: 0.5).",
    )

    parser.add_argument(
        "--subjects",
        nargs="*",
        default=None,
        help="Optional subject IDs, e.g. S01 S02 S05. "
             "Default: all S*.mat files.",
    )

    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Do not store actual EEG/ET signal arrays; keep "
             "segment indices and metadata only.",
    )

    args = parser.parse_args()

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset directory not found: {data_dir}"
        )

    if args.min_segment_seconds < 0:
        raise ValueError("--min-segment-seconds must be >= 0")

    output_dir.mkdir(parents=True, exist_ok=True)

    discovered = _iter_subject_ids(data_dir)

    if args.subjects:
        requested = []

        for subject_id in args.subjects:
            sid = subject_id.upper()
            if not sid.startswith("S"):
                sid = "S" + sid.zfill(2)
            requested.append(sid)

        subject_ids = requested
    else:
        subject_ids = discovered

    if not subject_ids:
        raise FileNotFoundError(
            f"No S*.mat subject files found in {data_dir}"
        )

    missing = [
        sid for sid in subject_ids
        if not (data_dir / f"{sid}.mat").exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Requested subject files were not found: "
            + ", ".join(missing)
        )

    if len(discovered) != 42:
        warnings.warn(
            f"Discovered {len(discovered)} subject files, whereas "
            "the paper describes 42 final participants. The script "
            "uses the files actually present and does not hard-code "
            "S01–S42."
        )

    subjects = []

    for i, subject_id in enumerate(subject_ids, start=1):
        path = data_dir / f"{subject_id}.mat"
        print(
            f"[{i}/{len(subject_ids)}] Processing {subject_id} ..."
        )

        subject = _process_subject(
            path,
            min_segment_seconds=args.min_segment_seconds,
            keep_signal_segments=not args.metadata_only,
        )

        subjects.append(subject)

    dataset = {
        "stage": "1_dataset_epoch_preparation",
        "paper": {
            "title": (
                "A hybrid neuromarketing approach exploiting EEG "
                "graph signal processing and gaze dynamic patterning"
            ),
            "year": 2025,
            "authors": "Kalaganis et al.",
        },
        "config": {
            "min_segment_seconds": float(args.min_segment_seconds),
            "expected_pages": EXPECTED_PAGES,
            "expected_products_per_page":
                EXPECTED_PRODUCTS_PER_PAGE,
            "matlab_indices_are_1_based_in_source": True,
            "matlab_end_index_is_inclusive": True,
            "stored_python_end_index_is_exclusive": True,
            "signals_stored": not args.metadata_only,
        },
        "subjects": subjects,
    }

    validation = _validate_stage1(dataset)
    dataset["validation"] = validation

    dataset_path = output_dir / "stage1_dataset.pkl"
    summary_path = output_dir / "stage1_summary.csv"

    with dataset_path.open("wb") as f:
        pickle.dump(
            dataset,
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    _write_summary(summary_path, subjects)

    print("\nStage 1 complete.")
    print(f"Dataset output : {dataset_path}")
    print(f"Summary output : {summary_path}")
    print(f"Subjects       : {len(subjects)}")
    print(
        "Validation     : "
        + ("PASS" if validation["passed"] else "FAIL")
    )

    if not validation["passed"]:
        raise RuntimeError(
            f"Stage 1 validation failed: {validation}"
        )


if __name__ == "__main__":
    main()
