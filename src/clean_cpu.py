"""
2025-10-22
Author: Dan Schumacher

Purpose
-------
Load the "Computers" time-series classification dataset (from TSC/UEA),
validate shapes/labels, and save clean NumPy arrays for training and testing.

Expected dataset stats (for sanity checks)
------------------------------------------
name:            cpu / dev / device  (Computers)
n_variables:     1                    (univariate)
series length:   720
n_classes:       2
n_samples:       ~500 total (train + test)
source:          https://www.timeseriesclassification.com/description.php?Dataset=Computers

Input file format
-----------------
Plain text files with one example per line:

    <label> v1 v2 ... v720

Notes:
- Values can be floats (including scientific notation).
- Labels are numeric (e.g., 0/1 or 1/2). We remap to {0,1} if necessary.

Outputs
-------
OUT_DIR/
    X_train.npy : float32, shape (N_train, 1, 720)
    y_train.npy : int64,   shape (N_train,)
    X_test.npy  : float32, shape (N_test, 1, 720)
    y_test.npy  : int64,   shape (N_test,)


Run
---
   python ./src/clean_cpu.py
"""

from typing import Tuple, List
import os
import sys
import numpy as np

# ----------------------------
# CONFIG
# ----------------------------
BASE_PATH = "./raw_data/cpu/"                       # Root folder containing *_TRAIN.txt and *_TEST.txt
DATASET_BASENAME = "Computers"                      # Files: Computers_TRAIN.txt, Computers_TEST.txt
OUT_DIR = "./Classification/data/datasets/cpu/"              # Where to write npy files
SAMP_DIR = "./Classification/data/samples/cpu/"
SERIES_LENGTH = 720                                 # Each time series length (columns after label)
EXPECTED_NUM_CLASSES = 2

# ----------------------------
# I/O helpers
# ----------------------------
def read_txt_split(file_path: str, series_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Read a TSC-style plain-text split where each line is:
        <label> v1 v2 ... v{series_length}

    Parameters
    ----------
    file_path : str
        Path to the *_TRAIN.txt or *_TEST.txt file.
    series_length : int
        Expected number of time steps per example (e.g., 720).

    Returns
    -------
    X : np.ndarray
        Float32 array of shape (N_examples, series_length).
    y : np.ndarray
        Int64 array of shape (N_examples,) with raw labels as found in file (not remapped).

    Raises
    ------
    ValueError
        If any line has the wrong number of columns.
    """
    features_list: List[np.ndarray] = []
    labels_list: List[int] = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue  # skip blank lines

            tokens = stripped.split()
            if len(tokens) != 1 + series_length:
                raise ValueError(
                    f"{file_path}: line {line_num} has {len(tokens)-1} values, "
                    f"expected {series_length}."
                )

            # First token is the label; robustly parse numeric (handles "1", "1.0", "0.0e0", etc.)
            try:
                label_val = int(float(tokens[0]))
            except Exception as e:
                raise ValueError(f"{file_path}: line {line_num} label parse failed: {tokens[0]!r}") from e

            # Remaining tokens are the time-series values
            try:
                values = np.array([float(v) for v in tokens[1:]], dtype=np.float32)
            except Exception as e:
                raise ValueError(f"{file_path}: line {line_num} value parse failed.") from e

            labels_list.append(label_val)
            features_list.append(values)

    if not features_list:
        raise ValueError(f"No data read from {file_path}.")

    X = np.vstack(features_list).astype(np.float32)   # (N, L)
    y = np.asarray(labels_list, dtype=np.int64)       # (N,)
    return X, y


def remap_labels_to_zero_one(y: np.ndarray) -> np.ndarray:
    """
    Ensure labels are {0,1}. If labels are already {0,1}, return unchanged.
    If labels are {1,2}, remap -> {0,1}. Otherwise, raise with a helpful message.

    Parameters
    ----------
    y : np.ndarray (int64)
        Label vector.

    Returns
    -------
    np.ndarray
        Remapped label vector (int64) with values in {0,1}.
    """
    unique_vals = np.unique(y)
    if np.array_equal(unique_vals, np.array([0, 1])):
        return y
    if np.array_equal(unique_vals, np.array([1, 2])):
        return (y - 1).astype(np.int64)
    raise ValueError(
        f"Unexpected label set {unique_vals.tolist()}; expected {{0,1}} or {{1,2}}."
    )


def summarize_split(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """
    Print a concise summary for quick sanity checks.

    Parameters
    ----------
    name : str
        Split name, e.g., 'TRAIN' or 'TEST'.
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Label vector.
    """
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique.tolist(), counts.tolist()))
    print(f"[{name}] X shape: {X.shape} | y shape: {y.shape} | class dist (after remap): {dist}")


def to_nyz(X: np.ndarray) -> np.ndarray:
    """Force (N, Y, Z). If (N, Z) -> (N, 1, Z). If already 3D, return as-is."""
    X = np.asarray(X)
    if X.ndim == 2:
        N, Z = X.shape
        return X.reshape(N, 1, Z)
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected 2D or 3D, got {X.shape}")

# ----------------------------
# MAIN
# ----------------------------
def main() -> None:
    """
    Load the Computers dataset splits, validate, remap labels to {0,1}, and save .npy files.
    """
    print("CLEANING CPU DATASET")
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(SAMP_DIR, exist_ok=True)

    train_path = os.path.join(BASE_PATH, f"{DATASET_BASENAME}_TRAIN.txt")
    test_path  = os.path.join(BASE_PATH, f"{DATASET_BASENAME}_TEST.txt")

    print("Loading training split ...")
    X_train, y_train_raw = read_txt_split(train_path, SERIES_LENGTH)

    print("Loading test split ...")
    X_test, y_test_raw = read_txt_split(test_path, SERIES_LENGTH)
    X_train = to_nyz(X_train)
    X_test  = to_nyz(X_test)

    # Basic shape checks: univariate and correct series length
    assert X_train.ndim == 3 and X_train.shape[-1] == SERIES_LENGTH, "TRAIN series length mismatch!"
    assert X_test.ndim == 3  and X_test.shape[-1] == SERIES_LENGTH,  "TEST series length mismatch!"
    assert X_train.shape[1] == 1, "TRAIN should be univariate: got Y != 1"
    assert X_test.shape[1]  == 1, "TEST should be univariate: got Y != 1"

    # Remap labels to {0,1} if necessary
    y_train = remap_labels_to_zero_one(y_train_raw)
    y_test  = remap_labels_to_zero_one(y_test_raw)

    # Final class count check
    num_classes_train = np.unique(y_train).size
    num_classes_test  = np.unique(y_test).size
    if num_classes_train != EXPECTED_NUM_CLASSES or num_classes_test != EXPECTED_NUM_CLASSES:
        print(f"WARNING: expected {EXPECTED_NUM_CLASSES} classes, "
              f"got {num_classes_train} (train) and {num_classes_test} (test).")
        
    # sample
    X_tr_samp = np.random.permutation(X_train)[:500]
    y_tr_samp = np.random.permutation(y_train)[:500]
    X_te_samp = np.random.permutation(X_test)[:100]
    y_te_samp = np.random.permutation(y_test)[:100]
    np.save(os.path.join(SAMP_DIR, "X_train.npy"), X_tr_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "X_test.npy"),  X_te_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_train.npy"), y_tr_samp.astype(np.int64, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_test.npy"),  y_te_samp.astype(np.int64, copy=False))

    # Save outputs
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train.astype(np.float32, copy=False))
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train.astype(np.int64, copy=False))
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test.astype(np.float32, copy=False))
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test.astype(np.int64, copy=False))

    # Summaries
    summarize_split("TRAIN", X_train, y_train)
    summarize_split("TEST",  X_test,  y_test)

    total_samples = X_train.shape[0] + X_test.shape[0]
    print(f"Expected (approx): ~500 | Your total: {total_samples}")

if __name__ == "__main__":
    main()
