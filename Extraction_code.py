import os
import re
from pathlib import Path

import numpy as np
import scipy.io


# ============================================================
# CONFIGURATION
# ============================================================

N_PAGES = 6
N_PRODUCTS = 24

SUBJECT_PATTERN = re.compile(
    r"^S(\d+)\.mat$",
    re.IGNORECASE
)


# ============================================================
# EXTRACT ACTUAL EEG / ET SEGMENTS
# ============================================================

def extract_segments(
    data,
    segment_array,
    fs,
    signal_name,
    subject_id,
    page_number,
    product_number
):
    """
    Extract actual EEG or ET signal segments from NeuMa.

    NeuMa stores segment boundaries as MATLAB indices:

        [start_sample, end_sample]

    MATLAB:
        - 1-based
        - start inclusive
        - end inclusive

    NumPy:
        - 0-based
        - start inclusive
        - end exclusive

    Therefore:

        MATLAB [start, end]
        -> NumPy [start - 1 : end]
    """

    segments = []

    if segment_array is None:
        return segments

    segment_array = np.asarray(segment_array)

    if segment_array.size == 0:
        return segments

    segment_array = np.atleast_2d(segment_array)

    if segment_array.shape[1] != 2:
        raise ValueError(
            f"{subject_id} | Page {page_number} | "
            f"Product {product_number}: "
            f"{signal_name}_segments has unexpected shape "
            f"{segment_array.shape}"
        )

    n_recording_samples = data.shape[1]

    for segment_id, row in enumerate(
        segment_array,
        start=1
    ):

        matlab_start = int(row[0])
        matlab_end = int(row[1])

        # Ignore malformed intervals
        if matlab_end < matlab_start:
            continue

        # ----------------------------------------------------
        # MATLAB -> NumPy indexing
        # ----------------------------------------------------

        python_start = matlab_start - 1
        python_end = matlab_end

        # ----------------------------------------------------
        # Bounds validation
        # ----------------------------------------------------

        if (
            matlab_start < 1
            or matlab_end > n_recording_samples
        ):
            raise IndexError(
                f"{subject_id} | Page {page_number} | "
                f"Product {product_number} | "
                f"{signal_name} segment {segment_id}: "
                f"[{matlab_start}, {matlab_end}] "
                f"outside recording range "
                f"[1, {n_recording_samples}]"
            )

        # ----------------------------------------------------
        # Extract actual signal
        # ----------------------------------------------------

        segment_data = data[
            :,
            python_start:python_end
        ].copy()

        # ----------------------------------------------------
        # Validate sample count
        # ----------------------------------------------------

        expected_samples = (
            matlab_end - matlab_start + 1
        )

        actual_samples = segment_data.shape[1]

        if actual_samples != expected_samples:
            raise RuntimeError(
                f"{subject_id} | Page {page_number} | "
                f"Product {product_number} | "
                f"{signal_name} segment {segment_id}: "
                f"expected {expected_samples} samples, "
                f"got {actual_samples}"
            )

        # ----------------------------------------------------
        # Store segment
        # ----------------------------------------------------

        segments.append({
            "segment_id": segment_id,

            # Original MATLAB indices
            "start_sample_matlab": matlab_start,
            "end_sample_matlab": matlab_end,

            # Python indices
            "start_sample_python": python_start,
            "end_sample_python": python_end,

            # Actual extracted signal
            "data": segment_data,

            # Metadata
            "n_samples": actual_samples,
            "duration_sec": (
                actual_samples / float(fs)
            ),
        })

    return segments


# ============================================================
# LOAD ONE SUBJECT
# ============================================================

def load_subject(mat_path):
    """
    Load one NeuMa SXX.mat file.

    Extracts:
        - continuous EEG
        - continuous ET
        - EEG channel labels
        - all 6 pages
        - all 24 products per page
        - actual EEG segments
        - actual ET segments
        - product labels

    No preprocessing is performed.
    """

    mat_path = Path(mat_path)

    subject_id = mat_path.stem

    print("\n" + "=" * 75)
    print(f"LOADING SUBJECT: {subject_id}")
    print("=" * 75)

    # --------------------------------------------------------
    # Load MATLAB file
    # --------------------------------------------------------

    mat = scipy.io.loadmat(
        str(mat_path),
        simplify_cells=False
    )

    if subject_id not in mat:
        raise KeyError(
            f"Top-level variable '{subject_id}' "
            f"not found in {mat_path}"
        )

    S = mat[subject_id][0, 0]

    # --------------------------------------------------------
    # EEG
    # --------------------------------------------------------

    eeg_struct = S["EEG_clean"][0, 0]

    eeg_data = np.asarray(
        eeg_struct["Data"],
        dtype=np.float64
    )

    eeg_fs = int(
        eeg_struct["Fs"].flat[0]
    )

    chan_info = eeg_struct["chans"][0]

    chan_labels = [
        str(chan_info[i]["labels"][0])
        for i in range(len(chan_info))
    ]

    # --------------------------------------------------------
    # ET
    # --------------------------------------------------------

    et_struct = S["ET_clean"][0, 0]

    et_data = np.asarray(
        et_struct["Data"],
        dtype=np.float64
    )

    et_fs = int(
        et_struct["Fs"].flat[0]
    )

    # --------------------------------------------------------
    # Basic validation
    # --------------------------------------------------------

    if eeg_data.ndim != 2:
        raise ValueError(
            f"{subject_id}: EEG data is not 2-D: "
            f"{eeg_data.shape}"
        )

    if et_data.ndim != 2:
        raise ValueError(
            f"{subject_id}: ET data is not 2-D: "
            f"{et_data.shape}"
        )

    if eeg_data.shape[0] != len(chan_labels):
        raise ValueError(
            f"{subject_id}: EEG channel count "
            f"does not match channel labels"
        )

    # --------------------------------------------------------
    # Print continuous recording information
    # --------------------------------------------------------

    print("\nContinuous recordings")
    print("-" * 75)

    print(
        f"EEG : {eeg_data.shape[0]} channels × "
        f"{eeg_data.shape[1]} samples "
        f"@ {eeg_fs} Hz"
    )

    print(
        f"ET  : {et_data.shape[0]} signals × "
        f"{et_data.shape[1]} samples "
        f"@ {et_fs} Hz"
    )

    print(
        f"EEG duration : "
        f"{eeg_data.shape[1] / eeg_fs:.3f} sec"
    )

    print(
        f"ET duration  : "
        f"{et_data.shape[1] / et_fs:.3f} sec"
    )

    print(
        "EEG channels : "
        + ", ".join(chan_labels)
    )

    # --------------------------------------------------------
    # Containers
    # --------------------------------------------------------

    products = []

    total_products = 0
    total_bought = 0
    total_not_bought = 0

    total_eeg_segments = 0
    total_et_segments = 0

    total_eeg_samples = 0
    total_et_samples = 0

    products_without_segments = 0

    # --------------------------------------------------------
    # PAGE × PRODUCT EXTRACTION
    # --------------------------------------------------------

    print("\nPage / Product extraction")
    print("-" * 75)

    for page_number in range(1, N_PAGES + 1):

        page_name = f"Page{page_number}"

        if page_name not in S.dtype.names:
            raise KeyError(
                f"{subject_id}: {page_name} not found"
            )

        page = S[page_name][0, 0]

        page_eeg_segments = 0
        page_et_segments = 0
        page_bought = 0

        print(f"\n--- Page {page_number} ---")

        for product_number in range(
            1,
            N_PRODUCTS + 1
        ):

            product_name = (
                f"Product{product_number}"
            )

            if product_name not in page.dtype.names:
                raise KeyError(
                    f"{subject_id}: "
                    f"{page_name}.{product_name} "
                    f"not found"
                )

            product = page[
                product_name
            ][0, 0]

            # ------------------------------------------------
            # Product information
            # ------------------------------------------------

            product_info = product[
                "ProductInfo"
            ][0, 0]

            label = int(
                product_info["Bought"].flat[0]
            )

            # Product description
            description_raw = (
                product_info["Description"]
            )

            try:
                description = str(
                    description_raw.flat[0]
                )
            except Exception:
                description = str(
                    description_raw
                )

            # ------------------------------------------------
            # EEG extraction
            # ------------------------------------------------

            eeg_segments = extract_segments(
                data=eeg_data,
                segment_array=product["EEG_segments"],
                fs=eeg_fs,
                signal_name="EEG",
                subject_id=subject_id,
                page_number=page_number,
                product_number=product_number
            )

            # ------------------------------------------------
            # ET extraction
            # ------------------------------------------------

            et_segments = extract_segments(
                data=et_data,
                segment_array=product["ET_segments"],
                fs=et_fs,
                signal_name="ET",
                subject_id=subject_id,
                page_number=page_number,
                product_number=product_number
            )

            # ------------------------------------------------
            # EEG / ET correspondence
            # ------------------------------------------------

            if len(eeg_segments) != len(et_segments):
                raise ValueError(
                    f"{subject_id} | Page {page_number} | "
                    f"Product {product_number}: "
                    f"EEG segments = {len(eeg_segments)}, "
                    f"ET segments = {len(et_segments)}"
                )

            # ------------------------------------------------
            # Statistics
            # ------------------------------------------------

            n_eeg = len(eeg_segments)
            n_et = len(et_segments)

            eeg_samples = sum(
                segment["n_samples"]
                for segment in eeg_segments
            )

            et_samples = sum(
                segment["n_samples"]
                for segment in et_segments
            )

            eeg_duration = (
                eeg_samples / eeg_fs
            )

            et_duration = (
                et_samples / et_fs
            )

            # ------------------------------------------------
            # Update counters
            # ------------------------------------------------

            total_products += 1

            total_eeg_segments += n_eeg
            total_et_segments += n_et

            total_eeg_samples += eeg_samples
            total_et_samples += et_samples

            page_eeg_segments += n_eeg
            page_et_segments += n_et

            if label == 1:
                total_bought += 1
                page_bought += 1
            else:
                total_not_bought += 1

            if n_eeg == 0:
                products_without_segments += 1

            # ------------------------------------------------
            # Store product
            # ------------------------------------------------

            products.append({
                "page": page_number,
                "product": product_number,

                "description": description,

                "label": label,

                "eeg_segments": eeg_segments,
                "et_segments": et_segments,

                "n_eeg_segments": n_eeg,
                "n_et_segments": n_et,

                "total_eeg_samples": eeg_samples,
                "total_et_samples": et_samples,

                "total_eeg_duration_sec":
                    eeg_duration,

                "total_et_duration_sec":
                    et_duration,
            })

            # ------------------------------------------------
            # Print product information
            # ------------------------------------------------

            print(
                f"Product {product_number:02d} | "
                f"Label={label} | "
                f"EEG={n_eeg:2d} seg | "
                f"ET={n_et:2d} seg | "
                f"EEG={eeg_duration:7.3f}s | "
                f"ET={et_duration:7.3f}s | "
                f"{description}"
            )

        # ----------------------------------------------------
        # Page summary
        # ----------------------------------------------------

        print(
            f"Page {page_number} summary: "
            f"{page_eeg_segments} EEG segments | "
            f"{page_et_segments} ET segments | "
            f"{page_bought}/24 bought"
        )

    # ========================================================
    # SUBJECT VALIDATION
    # ========================================================

    expected_products = (
        N_PAGES * N_PRODUCTS
    )

    if total_products != expected_products:
        raise RuntimeError(
            f"{subject_id}: expected "
            f"{expected_products} products, "
            f"loaded {total_products}"
        )

    if total_eeg_segments != total_et_segments:
        raise RuntimeError(
            f"{subject_id}: EEG/ET segment mismatch: "
            f"{total_eeg_segments} vs "
            f"{total_et_segments}"
        )

    # ========================================================
    # SUBJECT SUMMARY
    # ========================================================

    print("\n" + "=" * 75)
    print(f"SUBJECT {subject_id} — EXTRACTION SUMMARY")
    print("=" * 75)

    print(
        f"Products loaded           : "
        f"{total_products}/{expected_products}"
    )

    print(
        f"Bought products           : "
        f"{total_bought}"
    )

    print(
        f"Non-bought products       : "
        f"{total_not_bought}"
    )

    print(
        f"Total EEG segments        : "
        f"{total_eeg_segments}"
    )

    print(
        f"Total ET segments         : "
        f"{total_et_segments}"
    )

    print(
        f"Total extracted EEG       : "
        f"{total_eeg_samples:,} samples"
    )

    print(
        f"Total extracted ET        : "
        f"{total_et_samples:,} samples"
    )

    print(
        f"Products without segments : "
        f"{products_without_segments}"
    )

    print(
        "EEG / ET segment match    : PASS"
    )

    print("=" * 75)

    # ========================================================
    # RETURN
    # ========================================================

    return {
        "subject_id": subject_id,

        # Continuous EEG
        "eeg_data": eeg_data,
        "eeg_fs": eeg_fs,
        "chan_labels": chan_labels,

        # Continuous ET
        "et_data": et_data,
        "et_fs": et_fs,

        # Page × Product × Segment data
        "products": products,
    }


# ============================================================
# LOAD ALL SUBJECTS
# ============================================================

def load_all_subjects(data_dir):
    """
    Find SXX.mat subject files in the specified folder
    and load them sequentially.

    Returns
    -------
    dict
        {
            "S01": subject_data,
            "S02": subject_data,
            ...
        }
    """

    data_dir = Path(data_dir)

    if not data_dir.is_dir():
        raise NotADirectoryError(
            f"Invalid directory:\n{data_dir}"
        )

    # --------------------------------------------------------
    # Find subject files
    # --------------------------------------------------------

    subject_files = []

    for path in data_dir.iterdir():

        if not path.is_file():
            continue

        match = SUBJECT_PATTERN.fullmatch(
            path.name
        )

        if match is not None:

            subject_number = int(
                match.group(1)
            )

            subject_files.append(
                (subject_number, path)
            )

    # --------------------------------------------------------
    # Sort numerically
    # --------------------------------------------------------

    subject_files.sort(
        key=lambda item: item[0]
    )

    if not subject_files:
        raise FileNotFoundError(
            f"No SXX.mat subject files found in:\n"
            f"{data_dir}"
        )

    # --------------------------------------------------------
    # Print discovered subjects
    # --------------------------------------------------------

    print("\n")
    print("#" * 75)
    print("NeuMa EEG + ET EXTRACTION")
    print("#" * 75)

    print(
        f"\nInput folder:\n{data_dir}"
    )

    print(
        f"\nSubject files found: "
        f"{len(subject_files)}"
    )

    print("\nSubjects:")

    for _, path in subject_files:
        print(
            f"  - {path.name}"
        )

    # --------------------------------------------------------
    # Load subjects
    # --------------------------------------------------------

    subjects = {}

    successful = []
    failed = []

    for _, mat_path in subject_files:

        subject_id = mat_path.stem

        try:

            subject = load_subject(
                mat_path
            )

            subjects[subject_id] = subject
            successful.append(subject_id)

        except Exception as error:

            failed.append(
                (subject_id, str(error))
            )

            print("\n" + "!" * 75)
            print(
                f"FAILED TO LOAD {subject_id}"
            )
            print(
                f"Reason: {error}"
            )
            print("!" * 75)

    # ========================================================
    # OVERALL SUMMARY
    # ========================================================

    print("\n\n")
    print("#" * 75)
    print("OVERALL EXTRACTION SUMMARY")
    print("#" * 75)

    print(
        f"Subject files found : "
        f"{len(subject_files)}"
    )

    print(
        f"Successfully loaded  : "
        f"{len(successful)}"
    )

    print(
        f"Failed               : "
        f"{len(failed)}"
    )

    if successful:

        print(
            "\nSuccessfully loaded subjects:"
        )

        for subject_id in successful:
            print(
                f"  [OK] {subject_id}"
            )

    if failed:

        print(
            "\nFailed subjects:"
        )

        for subject_id, error in failed:
            print(
                f"  [FAILED] {subject_id}: {error}"
            )

    print("#" * 75)

    return subjects


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # CHANGE THIS PATH
    # ========================================================

    DATA_DIR = r"C:\Users\tiet\Desktop\Paramveer\Neuromarketing\Dataset_NeuMa"

    # ========================================================
    # LOAD ALL SUBJECTS
    # ========================================================

    subjects = load_all_subjects(
        DATA_DIR
    )

    # ========================================================
    # EXAMPLE ACCESS
    # ========================================================

    if subjects:

        first_subject_id = next(
            iter(subjects)
        )

        subject = subjects[
            first_subject_id
        ]

        print("\n")
        print("Example extracted data")
        print("-" * 75)

        print(
            f"Subject: "
            f"{first_subject_id}"
        )

        # First product = Page 1 / Product 1
        product = subject[
            "products"
        ][0]

        print(
            f"Page: "
            f"{product['page']}"
        )

        print(
            f"Product: "
            f"{product['product']}"
        )

        print(
            f"Description: "
            f"{product['description']}"
        )

        print(
            f"Label: "
            f"{product['label']}"
        )

        print(
            f"EEG segments: "
            f"{product['n_eeg_segments']}"
        )

        print(
            f"ET segments: "
            f"{product['n_et_segments']}"
        )

        # ----------------------------------------------------
        # First actual EEG segment
        # ----------------------------------------------------

        if product["eeg_segments"]:

            eeg = product[
                "eeg_segments"
            ][0]["data"]

            print(
                f"First EEG segment shape: "
                f"{eeg.shape}"
            )

        # ----------------------------------------------------
        # First actual ET segment
        # ----------------------------------------------------

        if product["et_segments"]:

            et = product[
                "et_segments"
            ][0]["data"]

            print(
                f"First ET segment shape: "
                f"{et.shape}"
            )