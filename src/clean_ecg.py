"""
2025-10-30
Author: Dan Schumacher

Purpose
-------
Segment the PhysioNet / CinC 2017 single-lead ECG recordings into fixed-length
windows for classification, validate shapes/labels, and save NumPy arrays.

Dataset summary (sanity checks)
-------------------------------
name:          ecg (PhysioNet/CinC Challenge 2017)
variables:     1 (forced single-lead)
series length: 1500 samples (5 s @ 300 Hz)
classes:       4 → {'N':0, 'A':1, 'O':2, '~':3}
segments:      ~43673 NOTE: expecting 43673 total, we got train 54593, test 1904, total 56497
source:        https://physionet.org/content/challenge-2017/1.0.0/

Inputs (expected layout)
------------------------
./raw_data/training2017/
  Axxxxx.mat    : MATLAB file with 'val' shaped (C, T)
  Axxxxx.hea    : WFDB header (not required here)
  REFERENCE.csv : "record_id,label" with labels in {'N','A','O','~'}

./raw_data/sample2017/validation/
  Axxxxx.mat
  Axxxxx.hea
  REFERENCE.csv

Windowing policy
----------------
- window_size = 1500, stride = 1500 (non-overlapping 5-second windows)
- Final leftover < 1500 samples are dropped (no padding).
- Each window inherits the parent recording's label.

Outputs
-------
./Classification/data/ecg/
  X_train.npy : float32, shape (N_train_segments, 1, 1500)
  y_train.npy : int64,   shape (N_train_segments,)
  X_test.npy  : float32, shape (N_test_segments,  1, 1500)
  y_test.npy  : int64,   shape (N_test_segments,)

Run
---
    python ./src/clean_ecg.py
"""


import csv
import glob
import os
import numpy as np
from scipy.io import loadmat


CLS2IDX = {"N": 0, "A": 1, "O": 2, "~": 3}

def load_reference_csv(reference_csv_path: str) -> dict:
    """
    Load REFERENCE.csv into a dictionary mapping record IDs to labels.

    Parameters
    ----------
    reference_csv_path : str
        Path to the REFERENCE.csv file, formatted as "record_id,label".

    Returns
    -------
    dict
        Example: {"A00001": "N", "A00002": "A", ...}
    """
    label_mapping = {}
    with open(reference_csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 2:
                continue
            record_id = row[0].strip()
            label = row[1].strip()
            label_mapping[record_id] = label
    return label_mapping


def load_single_lead_signal(mat_file_path: str) -> np.ndarray:
    """
    Load a single-lead ECG signal from a PhysioNet .mat file.

    Parameters
    ----------
    mat_file_path : str
        Path to the .mat file (containing 'val').

    Returns
    -------
    np.ndarray
        A NumPy array of shape (1, n_samples), dtype float32.
    """
    data_dict = loadmat(mat_file_path)
    signal = data_dict["val"].astype(np.float32)  # shape: (num_leads, num_samples)

    # The dataset uses single-lead ECGs. Keep only the first channel if multiple exist.
    if signal.shape[0] > 1:
        signal = signal[:1, :]

    return signal


def segment_signal(signal: np.ndarray, window_size: int = 1500, stride: int = 1500) -> list:
    """
    Split a long ECG signal into fixed-length windows.

    Parameters
    ----------
    signal : np.ndarray
        Input ECG array of shape (1, num_samples).
    window_size : int
        Number of samples per segment (default: 1500 = 5 seconds @ 300 Hz).
    stride : int
        Step size between consecutive windows (default: 1500 = non-overlapping).

    Returns
    -------
    list[np.ndarray]
        List of windows, each of shape (1, window_size).
    """
    _, total_samples = signal.shape
    segments = []

    for start_idx in range(0, total_samples - window_size + 1, stride):
        end_idx = start_idx + window_size
        segment = signal[:, start_idx:end_idx]
        segments.append(segment)

    return segments


def build_segmented_dataset(
    mat_directory: str,
    reference_csv_path: str,
    window_size: int = 1500,
    stride: int = 1500,
    drop_noisy: bool = False
) -> tuple:
    """
    Build a dataset of fixed-length ECG segments with labels.

    Parameters
    ----------
    mat_directory : str
        Path to the directory containing .mat files.
    reference_csv_path : str
        Path to the REFERENCE.csv file with record-label pairs.
    window_size : int
        Number of samples per segment (default: 1500).
    stride : int
        Step size between segments (default: 1500 for non-overlapping windows).
    drop_noisy : bool
        If True, discard recordings labeled as '~' (noisy/unclassifiable).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, list[str]]
        (X, y, record_ids)
        - X: array of shape (N, 1, window_size)
        - y: array of shape (N,)
        - record_ids: list of record IDs corresponding to each segment
    """
    label_mapping = load_reference_csv(reference_csv_path)
    mat_files = sorted(glob.glob(os.path.join(mat_directory, "**", "*.mat"), recursive=True))

    X_segments = []
    y_labels = []
    record_ids = []

    for mat_file in mat_files:
        record_id = os.path.splitext(os.path.basename(mat_file))[0]
        label = label_mapping.get(record_id)

        if label is None:
            # Some validation sets may not include REFERENCE.csv
            continue
        if drop_noisy and label == "~":
            # Optionally skip noisy recordings
            continue

        signal = load_single_lead_signal(mat_file)
        windows = segment_signal(signal, window_size, stride)

        for window in windows:
            X_segments.append(window)
            y_labels.append(CLS2IDX[label])
            record_ids.append(record_id)

    # Stack segments into arrays
    X_array = np.stack(X_segments, axis=0)   # shape: (N, 1, window_size)
    y_array = np.array(y_labels, dtype=np.int64)

    return X_array, y_array, record_ids


# -------------------------------------------------------------
# Main execution
# -------------------------------------------------------------
if __name__ == "__main__":
    # Build dataset with non-overlapping 5-second windows
    X_train, y_train, record_ids = build_segmented_dataset(
        mat_directory="./raw_data/training2017",
        reference_csv_path="./raw_data/training2017/REFERENCE.csv",
        window_size=1500,
        stride=1500,
    )
    X_test, y_test, record_ids = build_segmented_dataset(
        mat_directory="./raw_data/sample2017/validation",
        reference_csv_path="./raw_data/sample2017/validation/REFERENCE.csv",
        window_size=1500,
        stride=1500,
    )


    # Print summary
    print(f"\nTotal train segments: {len(X_train)}")
    print(f"Shape of X_train: {X_train.shape}")
    print(f"Shape of y_train: {y_train.shape}")
    print("Label distribution:", {lbl: int((y_train == idx).sum()) for lbl, idx in CLS2IDX.items()})

    print(f"\nTotal test segments: {len(X_test)}")
    print(f"Shape of X_test: {X_test.shape}")
    print(f"Shape of y_test: {y_test.shape}")
    print("Label distribution:", {lbl: int((y_test == idx).sum()) for lbl, idx in CLS2IDX.items()})


    # Save arrays to disk
    os.makedirs("./Classification/data/ecg/", exist_ok=True)
    np.save("./Classification/data/ecg/X_train.npy", X_train)
    np.save("./Classification/data/ecg/y_train.npy", y_train)
    np.save("./Classification/data/ecg/X_test.npy", X_test)
    np.save("./Classification/data/ecg/y_test.npy", y_test)

    print(f"Expected total ~ 43673 | actual={len(y_train) + len(y_test)}")
    print("\nSAVED FILES SUCCESSSFULLY  |  ./Classification/data/ecg/")
