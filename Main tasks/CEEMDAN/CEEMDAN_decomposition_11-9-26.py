"""
run_ceemdan.py

NeuMa EEG segment-level CEEMDAN decomposition.

IMPORTANT:
    This script calls the original MATLAB CEEMDAN implementation:
        ceemdan.m
        emd.m

The decomposition is therefore performed by the supplied repository
implementation rather than by the previous custom Python EMD implementation.

Processing rules
----------------
1. Load EEG/product/segment information using Extraction_code.py.
2. Evaluate every EEG segment independently.
3. Discard ONLY when:
       Bought == 0 AND duration <= 0.5 seconds
4. Keep every other segment.
5. Decompose every EEG channel independently.
6. Do NOT concatenate segments.
7. Do NOT apply an additional bandpass filter.
8. Do NOT impose an IMF-count limit.
9. CEEMDAN continues until its natural EMD stopping condition.
10. MaxIter limits only the sifting iterations of an individual IMF.
"""

from pathlib import Path
import argparse
import json
import time
import traceback

import numpy as np

# Existing NeuMa extraction implementation
from Extraction_code import load_all_subjects


# ============================================================
# CONFIGURATION
# ============================================================

DURATION_THRESHOLD_SEC = 0.5

# These correspond to the parameters supplied to the repository
# ceemdan.m implementation.
CEEMDAN_NSTD = 0.2
CEEMDAN_NR = 100
CEEMDAN_MAX_ITER = 300
CEEMDAN_SNR_FLAG = 1

# No MAXMODES is supplied to EMD inside CEEMDAN.
# Therefore IMF extraction is uncapped.
MAX_IMFS = None


# ============================================================
# MATLAB ENGINE
# ============================================================

def start_matlab():
    """
    Start MATLAB Engine.

    Requires:
        pip install matlabengine
    or the MATLAB Engine for Python corresponding to the
    installed MATLAB version.
    """

    try:
        import matlab.engine
    except ImportError as exc:
        raise RuntimeError(
            "\nMATLAB Engine for Python is not installed.\n"
            "Install the MATLAB Engine for Python corresponding "
            "to your MATLAB installation.\n"
        ) from exc

    print("\nStarting MATLAB Engine...")
    eng = matlab.engine.start_matlab()

    return eng


# ============================================================
# MATLAB CEEMDAN WRAPPER
# ============================================================

def matlab_ceemdan(
    eng,
    signal,
    nstd,
    nr,
    max_iter,
    snr_flag,
):
    """
    Run the supplied ceemdan.m implementation.

    Parameters
    ----------
    signal : 1-D numpy array
        One EEG channel from one EEG segment.

    Returns
    -------
    modes : numpy.ndarray
        Shape:
            (n_modes, n_samples)

        Following the supplied MATLAB implementation:
            modes[:-1] = extracted IMFs
            modes[-1]  = final residue

    iterations : numpy.ndarray
        Sifting-iteration information returned by ceemdan.m.
    """

    signal = np.asarray(
        signal,
        dtype=np.float64
    ).reshape(-1)

    if signal.size < 3:
        raise ValueError(
            f"Signal too short for CEEMDAN: "
            f"{signal.size} samples"
        )

    # MATLAB numeric vector
    matlab_signal = signal.tolist()

    # nargout=2 -> modes, its
    modes, iterations = eng.ceemdan(
        matlab_signal,
        float(nstd),
        float(nr),
        float(max_iter),
        float(snr_flag),
        nargout=2
    )

    modes = np.asarray(
        modes,
        dtype=np.float64
    )

    iterations = np.asarray(
        iterations,
        dtype=np.float64
    )

    # MATLAB ceemdan forces x to a row vector.
    # MATLAB output should therefore be:
    #
    #     n_modes x n_samples
    #
    if modes.ndim == 1:
        modes = modes.reshape(1, -1)

    return modes, iterations


# ============================================================
# RECONSTRUCTION VALIDATION
# ============================================================

def reconstruction_error(
    original_signal,
    modes,
):
    """
    Verify:

        original ≈ sum(IMFs) + residue

    The MATLAB CEEMDAN implementation returns all modes,
    including the final residue.
    """

    original_signal = np.asarray(
        original_signal,
        dtype=np.float64
    ).reshape(-1)

    modes = np.asarray(
        modes,
        dtype=np.float64
    )

    reconstructed = np.sum(
        modes,
        axis=0
    )

    difference = (
        original_signal - reconstructed
    )

    rmse = float(
        np.sqrt(
            np.mean(
                difference ** 2
            )
        )
    )

    max_abs_error = float(
        np.max(
            np.abs(difference)
        )
    )

    return {
        "rmse": rmse,
        "max_abs_error": max_abs_error,
    }


# ============================================================
# PROCESS ONE EEG SEGMENT
# ============================================================

def process_segment(
    eng,
    subject_id,
    page_number,
    product_number,
    product,
    segment,
    output_root,
    global_segment_index,
):
    """
    Process one EEG viewing segment.

    Exclusion is performed BEFORE CEEMDAN.
    """

    label = int(
        product["label"]
    )

    segment_id = int(
        segment["segment_id"]
    )

    duration_sec = float(
        segment["duration_sec"]
    )

    n_samples = int(
        segment["n_samples"]
    )

    # --------------------------------------------------------
    # EXCLUSION RULE
    # --------------------------------------------------------

    discard = (
        label == 0
        and duration_sec <= DURATION_THRESHOLD_SEC
    )

    if discard:

        return {
            "status": "DISCARDED",

            "subject_id": subject_id,
            "page": page_number,
            "product": product_number,
            "segment_id": segment_id,

            "label": label,

            "duration_sec": duration_sec,
            "n_samples": n_samples,

            "start_sample_matlab": int(
                segment["start_sample_matlab"]
            ),

            "end_sample_matlab": int(
                segment["end_sample_matlab"]
            ),

            "reason":
                "NO_BUY_AND_DURATION_LE_0.5_SEC",
        }

    # --------------------------------------------------------
    # EEG DATA
    # --------------------------------------------------------

    eeg = np.asarray(
        segment["data"],
        dtype=np.float64
    )

    if eeg.ndim != 2:
        raise ValueError(
            f"{subject_id} Page {page_number} "
            f"Product {product_number} "
            f"Segment {segment_id}: "
            f"EEG must be 2-D, got {eeg.shape}"
        )

    n_channels, actual_samples = eeg.shape

    if actual_samples != n_samples:
        raise ValueError(
            f"{subject_id} Page {page_number} "
            f"Product {product_number} "
            f"Segment {segment_id}: "
            f"metadata says {n_samples} samples but "
            f"EEG contains {actual_samples}"
        )

    # --------------------------------------------------------
    # OUTPUT DIRECTORY
    # --------------------------------------------------------

    subject_dir = (
        output_root /
        subject_id
    )

    segment_dir = (
        subject_dir /
        f"Page{page_number:02d}" /
        f"Product{product_number:02d}" /
        f"Segment{segment_id:02d}"
    )

    segment_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # PROCESS ALL EEG CHANNELS
    # --------------------------------------------------------

    channel_results = []

    segment_start = time.time()

    for channel_index in range(
        n_channels
    ):

        signal = eeg[
            channel_index,
            :
        ]

        channel_start = time.time()

        try:

            modes, iterations = matlab_ceemdan(
                eng=eng,
                signal=signal,

                nstd=CEEMDAN_NSTD,
                nr=CEEMDAN_NR,
                max_iter=CEEMDAN_MAX_ITER,
                snr_flag=CEEMDAN_SNR_FLAG,
            )

            # ------------------------------------------------
            # MATLAB CEEMDAN OUTPUT
            #
            # Last row = residue
            # Previous rows = IMFs
            # ------------------------------------------------

            if modes.shape[0] < 2:

                raise RuntimeError(
                    "CEEMDAN returned fewer than "
                    "two modes."
                )

            imfs = modes[:-1, :]
            residual = modes[-1, :]

            n_imfs = int(
                imfs.shape[0]
            )

            # ------------------------------------------------
            # RECONSTRUCTION CHECK
            # ------------------------------------------------

            validation = (
                reconstruction_error(
                    signal,
                    modes
                )
            )

            # ------------------------------------------------
            # SAVE CHANNEL OUTPUT
            # ------------------------------------------------

            channel_file = (
                segment_dir /
                f"channel_{channel_index + 1:02d}.npz"
            )

            np.savez_compressed(
                channel_file,

                imfs=imfs,
                residual=residual,

                iterations=iterations,

                original_signal=signal,

                channel_index=channel_index,

                n_imfs=n_imfs,
            )

            channel_time = (
                time.time() - channel_start
            )

            channel_results.append({
                "channel_index":
                    channel_index,

                "n_imfs":
                    n_imfs,

                "n_modes":
                    int(modes.shape[0]),

                "n_samples":
                    int(modes.shape[1]),

                "reconstruction_rmse":
                    validation["rmse"],

                "reconstruction_max_abs_error":
                    validation["max_abs_error"],

                "processing_time_sec":
                    channel_time,

                "status":
                    "SUCCESS",

                "output_file":
                    str(channel_file),
            })

            print(
                f"      Channel "
                f"{channel_index + 1:02d}/{n_channels} "
                f"| IMFs={n_imfs:02d} "
                f"| RMSE={validation['rmse']:.3e}"
            )

        except Exception as exc:

            channel_results.append({
                "channel_index":
                    channel_index,

                "status":
                    "FAILED",

                "error":
                    repr(exc),

                "traceback":
                    traceback.format_exc(),
            })

            print(
                f"      Channel "
                f"{channel_index + 1:02d}/{n_channels} "
                f"| FAILED"
            )

    # --------------------------------------------------------
    # SEGMENT METADATA
    # --------------------------------------------------------

    segment_time = (
        time.time() - segment_start
    )

    successful_channels = sum(
        1
        for x in channel_results
        if x.get("status") == "SUCCESS"
    )

    failed_channels = sum(
        1
        for x in channel_results
        if x.get("status") == "FAILED"
    )

    if failed_channels == 0:
        status = "SUCCESS"

    elif successful_channels > 0:
        status = "PARTIAL_FAILURE"

    else:
        status = "FAILED"

    metadata = {
        "status": status,

        "subject_id": subject_id,

        "page": page_number,

        "product": product_number,

        "segment_id": segment_id,

        "label": label,

        "description":
            product.get(
                "description",
                ""
            ),

        "duration_sec":
            duration_sec,

        "n_samples":
            n_samples,

        "start_sample_matlab":
            int(
                segment[
                    "start_sample_matlab"
                ]
            ),

        "end_sample_matlab":
            int(
                segment[
                    "end_sample_matlab"
                ]
            ),

        "start_sample_python":
            int(
                segment[
                    "start_sample_python"
                ]
            ),

        "end_sample_python":
            int(
                segment[
                    "end_sample_python"
                ]
            ),

        "ceemdan": {
            "Nstd":
                CEEMDAN_NSTD,

            "NR":
                CEEMDAN_NR,

            "MaxIter":
                CEEMDAN_MAX_ITER,

            "SNRFlag":
                CEEMDAN_SNR_FLAG,

            "MAXMODES":
                None,

            "decomposition":
                "UNCAPPED",

            "implementation":
                "Uploaded MATLAB ceemdan.m + Rilling-Flandrin emd.m",
        },

        "n_channels":
            n_channels,

        "successful_channels":
            successful_channels,

        "failed_channels":
            failed_channels,

        "processing_time_sec":
            segment_time,

        "channels":
            channel_results,
    }

    metadata_file = (
        segment_dir /
        "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            indent=2
        )

    return metadata


# ============================================================
# MAIN PROCESSING
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Apply uncapped repository CEEMDAN "
            "to NeuMa EEG segments."
        )
    )

    parser.add_argument(
        "--data_dir",
        required=True,
        help=(
            "Directory containing SXX.mat files."
        )
    )

    parser.add_argument(
        "--out_dir",
        required=True,
        help=(
            "Directory where CEEMDAN results "
            "will be stored."
        )
    )

    parser.add_argument(
        "--matlab_dir",
        required=True,
        help=(
            "Directory containing ceemdan.m "
            "and emd.m."
        )
    )

    args = parser.parse_args()

    data_dir = Path(
        args.data_dir
    )

    output_root = Path(
        args.out_dir
    )

    matlab_dir = Path(
        args.matlab_dir
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # CHECK MATLAB SOURCE
    # --------------------------------------------------------

    ceemdan_file = (
        matlab_dir /
        "ceemdan.m"
    )

    emd_file = (
        matlab_dir /
        "emd.m"
    )

    if not ceemdan_file.exists():
        raise FileNotFoundError(
            f"Missing:\n{ceemdan_file}"
        )

    if not emd_file.exists():
        raise FileNotFoundError(
            f"Missing:\n{emd_file}"
        )

    # --------------------------------------------------------
    # LOAD DATA THROUGH EXISTING EXTRACTION PIPELINE
    # --------------------------------------------------------

    print("=" * 75)
    print("NeuMa — CEEMDAN EEG DECOMPOSITION")
    print("=" * 75)

    print(
        f"\nInput directory:\n"
        f"{data_dir}"
    )

    print(
        f"\nOutput directory:\n"
        f"{output_root}"
    )

    print(
        f"\nMATLAB CEEMDAN directory:\n"
        f"{matlab_dir}"
    )

    print("\nCEEMDAN configuration")
    print("-" * 75)

    print(
        f"Nstd        : "
        f"{CEEMDAN_NSTD}"
    )

    print(
        f"NR          : "
        f"{CEEMDAN_NR}"
    )

    print(
        f"MaxIter     : "
        f"{CEEMDAN_MAX_ITER}"
    )

    print(
        f"SNRFlag     : "
        f"{CEEMDAN_SNR_FLAG}"
    )

    print(
        "MAXMODES    : "
        "UNLIMITED"
    )

    print(
        "Filtering   : "
        "NONE"
    )

    print(
        "Concatenate : "
        "NO"
    )

    print(
        "Exclusion   : "
        "Bought=0 AND duration<=0.5 s"
    )

    print("=" * 75)

    subjects = load_all_subjects(
        data_dir
    )

    print(
        f"\nLoaded subjects: "
        f"{len(subjects)}"
    )

    # --------------------------------------------------------
    # START MATLAB
    # --------------------------------------------------------

    eng = start_matlab()

    try:

        # Add directory containing the exact
        # ceemdan.m and emd.m to MATLAB path.
        eng.addpath(
            str(matlab_dir),
            nargout=0
        )

        # Verify MATLAB sees the expected function.
        ceemdan_location = eng.which(
            "ceemdan"
        )

        emd_location = eng.which(
            "emd"
        )

        print(
            "\nMATLAB CEEMDAN:"
        )
        print(
            ceemdan_location
        )

        print(
            "\nMATLAB EMD:"
        )
        print(
            emd_location
        )

        # ----------------------------------------------------
        # GLOBAL METADATA
        # ----------------------------------------------------

        dataset_metadata = {
            "implementation":
                "Uploaded MATLAB CEEMDAN repository",

            "ceemdan_file":
                str(ceemdan_file),

            "emd_file":
                str(emd_file),

            "nstd":
                CEEMDAN_NSTD,

            "nr":
                CEEMDAN_NR,

            "max_iter":
                CEEMDAN_MAX_ITER,

            "snr_flag":
                CEEMDAN_SNR_FLAG,

            "max_modes":
                None,

            "imf_limit":
                "NONE",

            "duration_exclusion_sec":
                DURATION_THRESHOLD_SEC,

            "exclusion_rule":
                (
                    "Bought == 0 AND "
                    "duration <= 0.5 sec"
                ),

            "filtering":
                "NONE",

            "segment_concatenation":
                False,

            "subjects":
                list(subjects.keys()),
        }

        # ----------------------------------------------------
        # PROCESS SUBJECTS
        # ----------------------------------------------------

        all_results = []

        total_segments = 0
        discarded_segments = 0
        processed_segments = 0
        failed_segments = 0

        for subject_index, (
            subject_id,
            subject
        ) in enumerate(
            subjects.items(),
            start=1
        ):

            print("\n")
            print("#" * 75)

            print(
                f"SUBJECT "
                f"{subject_index}/"
                f"{len(subjects)}: "
                f"{subject_id}"
            )

            print("#" * 75)

            eeg_fs = float(
                subject["eeg_fs"]
            )

            print(
                f"EEG sampling rate: "
                f"{eeg_fs} Hz"
            )

            print(
                f"Channels: "
                f"{len(subject['chan_labels'])}"
            )

            for product in subject[
                "products"
            ]:

                page_number = int(
                    product["page"]
                )

                product_number = int(
                    product["product"]
                )

                label = int(
                    product["label"]
                )

                for segment in product[
                    "eeg_segments"
                ]:

                    total_segments += 1

                    print(
                        f"\n  Page "
                        f"{page_number:02d} | "
                        f"Product "
                        f"{product_number:02d} | "
                        f"Segment "
                        f"{segment['segment_id']:02d} | "
                        f"Bought={label} | "
                        f"Duration="
                        f"{segment['duration_sec']:.4f}s"
                    )

                    result = process_segment(
                        eng=eng,

                        subject_id=subject_id,

                        page_number=page_number,

                        product_number=product_number,

                        product=product,

                        segment=segment,

                        output_root=output_root,

                        global_segment_index=
                            total_segments,
                    )

                    all_results.append(
                        result
                    )

                    if result[
                        "status"
                    ] == "DISCARDED":

                        discarded_segments += 1

                        print(
                            "      -> DISCARDED"
                        )

                    elif result[
                        "status"
                    ] == "FAILED":

                        failed_segments += 1

                        print(
                            "      -> FAILED"
                        )

                    else:

                        processed_segments += 1

                        print(
                            "      -> "
                            f"{result['status']}"
                        )

        # ----------------------------------------------------
        # SAVE DATASET SUMMARY
        # ----------------------------------------------------

        dataset_metadata[
            "total_segments"
        ] = total_segments

        dataset_metadata[
            "discarded_segments"
        ] = discarded_segments

        dataset_metadata[
            "processed_segments"
        ] = processed_segments

        dataset_metadata[
            "failed_segments"
        ] = failed_segments

        dataset_metadata[
            "results"
        ] = all_results

        summary_file = (
            output_root /
            "dataset_metadata.json"
        )

        with open(
            summary_file,
            "w",
            encoding="utf-8"
        ) as f:

            json.dump(
                dataset_metadata,
                f,
                indent=2
            )

        # ----------------------------------------------------
        # FINAL SUMMARY
        # ----------------------------------------------------

        print("\n")
        print("=" * 75)
        print("CEEMDAN PROCESSING COMPLETE")
        print("=" * 75)

        print(
            f"Subjects             : "
            f"{len(subjects)}"
        )

        print(
            f"Total EEG segments   : "
            f"{total_segments}"
        )

        print(
            f"Discarded            : "
            f"{discarded_segments}"
        )

        print(
            f"Processed            : "
            f"{processed_segments}"
        )

        print(
            f"Failed               : "
            f"{failed_segments}"
        )

        print(
            f"\nResults saved to:\n"
            f"{output_root}"
        )

    finally:

        print(
            "\nClosing MATLAB Engine..."
        )

        eng.quit()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()