from pathlib import Path
from typing import Any
import time

import matlab
import matlab.engine
import numpy as np

from Extraction_code import load_subject


# ------------------------------------------------------------
# EDIT ONLY THESE TWO PATHS
# ------------------------------------------------------------

DATA_DIR = Path(
    r"C:\Users\tiet\Desktop\Paramveer\Neuromarketing\Dataset_NeuMa"
)

MATLAB_DIR = Path(
    r"C:\Users\tiet\Desktop\Paramveer\Neuromarketing\Main_tasks\CEEMDAN"
)


# ------------------------------------------------------------
# LOAD ONE SUBJECT
# ------------------------------------------------------------

subject_file = DATA_DIR / "S01.mat"

if not subject_file.is_file():
    raise FileNotFoundError(
        f"Could not find subject file:\n{subject_file}"
    )

print(f"Loading one subject:\n{subject_file}")

subject = load_subject(subject_file)


# ------------------------------------------------------------
# SELECT THE LONGEST RETAINED SEGMENT
# ------------------------------------------------------------

candidates = []

for product in subject["products"]:
    for segment in product["eeg_segments"]:

        discard = (
            product["label"] == 0
            and segment["duration_sec"] <= 0.5
        )

        if not discard:
            candidates.append((product, segment))

if not candidates:
    raise RuntimeError(
        "No valid EEG segments found."
    )

selected_product, selected_segment = max(
    candidates,
    key=lambda item: item[1]["duration_sec"]
)

print("\nSelected segment")
print("----------------")
print("Label:", selected_product["label"])
print(
    "Duration:",
    selected_segment["duration_sec"],
    "seconds"
)


# ------------------------------------------------------------
# PREPARE ONE EEG CHANNEL
# ------------------------------------------------------------

eeg = np.asarray(
    selected_segment["data"],
    dtype=np.float64
)

if eeg.ndim != 2:
    raise ValueError(
        f"Expected 2-D EEG data, got {eeg.shape}"
    )

# EEG is stored as channels × samples.
signal = eeg[0, :]


# ------------------------------------------------------------
# RUN MATLAB CEEMDAN
# ------------------------------------------------------------

eng: Any = None

try:

    print("\nStarting MATLAB Engine...")
    eng = matlab.engine.start_matlab()

    eng.addpath(
        str(MATLAB_DIR),
        nargout=0
    )

    print("ceemdan.m:", eng.which("ceemdan"))
    print("emd.m:", eng.which("emd"))

    print("\nRunning CEEMDAN on one real EEG channel...")

    start_time = time.perf_counter()

    modes, iterations = eng.ceemdan(
        matlab.double(signal.tolist()),
        matlab.double(0.2),       # NSTD
        matlab.double(100.0),     # NR
        matlab.double(300.0),     # MAX_ITER
        matlab.double(1.0),       # SNR_FLAG
        nargout=2
    )

    elapsed_time = (
        time.perf_counter() - start_time
    )

    modes = np.asarray(
        modes,
        dtype=np.float64
    )

    if modes.ndim == 1:
        modes = modes.reshape(1, -1)

    if modes.ndim != 2:
        raise ValueError(
            f"Unexpected mode dimensions: {modes.shape}"
        )

    if modes.shape[1] != signal.size:
        raise ValueError(
            f"Unexpected mode shape: {modes.shape}; "
            f"expected {signal.size} samples"
        )

    reconstructed = np.sum(
        modes,
        axis=0
    )

    rmse = np.sqrt(
        np.mean(
            (signal - reconstructed) ** 2
        )
    )

    print("\nValidation result")
    print("-----------------")
    print("Signal samples:", signal.size)
    print("Modes shape:", modes.shape)
    print("IMF count:", modes.shape[0] - 1)
    print("CEEMDAN time:", elapsed_time, "seconds")
    print("Reconstruction RMSE:", rmse)

finally:

    if eng is not None:
        print("\nClosing MATLAB Engine...")
        eng.quit()