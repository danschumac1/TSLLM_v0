"""
2025-10-22
Author: Dan Schumacher

Purpose
-------
Load the Lightning7 (TEE) dataset from raw .txt files, validate dimensions,
and save NumPy arrays for classification experiments.

Dataset summary (sanity checks)
-------------------------------
name:          tee (Lightning / TEE)
variables:     1 (univariate)
series length: 319
classes:       7
samples:       143 total (train + test)
source:        https://www.timeseriesclassification.com/description.php?Dataset=Lightning7

Input file format
-----------------
Each line of the .txt files is:
    <label> v1 v2 ... v319
where `label` is numeric and values are whitespace-separated floats.

Outputs
-------
OUT_DIR/
    X_train.npy : float32, shape (N_train, 1, 319)
    y_train.npy : int64,   shape (N_train,)
    X_test.npy  : float32, shape (N_test,  1, 319)
    y_test.npy  : int64,   shape (N_test,)

Run
---
python ./src/clean_tee.py
"""

import os
import numpy as np

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
BASE_PATH  = "raw_data/tee"
OUT_DIR    = "Classification/data/datasets/tee"
SAMP_DIR   = "./Classification/data/samples/tee/"
SERIES_LEN = 319  # expected time series length
DTYPE_X    = np.float32
DTYPE_Y    = np.int64

# ---------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------
def read_txt_file(file_path: str):
    """
    Read a plain-text time series file with rows of the form:
        <label> v1 v2 ... vL

    Handles floats and scientific notation. Returns arrays:
        X : np.ndarray, shape (N, L), dtype=float32
        y : np.ndarray, shape (N,),   dtype=int64
    """
    X_list, y_list = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            toks = line.split()
            # First token is label
            try:
                label = int(float(toks[0]))
            except ValueError:
                raise ValueError(f"Invalid label in line: {line[:50]}")
            # Remaining tokens are series values
            try:
                vals = [float(v) for v in toks[1:]]
            except ValueError:
                raise ValueError(f"Non-numeric value in line: {line[:50]}")
            X_list.append(np.array(vals, dtype=DTYPE_X))
            y_list.append(label)

    if not X_list:
        raise RuntimeError(f"No valid data rows found in {file_path}")

    X = np.vstack(X_list).astype(DTYPE_X)
    y = np.asarray(y_list, dtype=DTYPE_Y)
    return X, y

def to_nyz(X: np.ndarray) -> np.ndarray:
    """Force (N, Y, Z). If (N, Z) -> (N, 1, Z)."""
    X = np.asarray(X)
    if X.ndim == 2:
        N, Z = X.shape
        return X.reshape(N, 1, Z)
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected 2D or 3D, got {X.shape}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    base = os.path.join(BASE_PATH, "Lightning7")

    print("Loading train/test splits ...")
    X_train, y_train = read_txt_file(base + "_TRAIN.txt")
    X_test,  y_test  = read_txt_file(base + "_TEST.txt")

    # --- Sanity checks ---
    assert X_train.shape[1] == SERIES_LEN, f"Train series length mismatch: {X_train.shape[1]}"
    assert X_test.shape[1]  == SERIES_LEN, f"Test series length mismatch: {X_test.shape[1]}"
    assert X_train.shape[0] > 0 and X_test.shape[0] > 0, "Empty split detected."
    X_train = to_nyz(X_train)
    X_test  = to_nyz(X_test)
    assert X_train.ndim == 3 and X_train.shape[1] == 1 and X_train.shape[2] == SERIES_LEN
    assert X_test.ndim  == 3 and X_test.shape[1]  == 1 and X_test.shape[2]  == SERIES_LEN


    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    print(f"Label distribution (train): { {int(k): int(v) for k, v in zip(*np.unique(y_train, return_counts=True))} }")
    print(f"Label distribution (test):  { {int(k): int(v) for k, v in zip(*np.unique(y_test,  return_counts=True))} }")

    # --- Save arrays ---
    print("Saving .npy files ...")
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train.astype(DTYPE_X, copy=False))
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train.astype(DTYPE_Y, copy=False))
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test.astype(DTYPE_X,  copy=False))
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test.astype(DTYPE_Y,  copy=False))
    
    os.makedirs(SAMP_DIR, exist_ok=True)
    # sample
    X_tr_samp = np.random.permutation(X_train)[:500]
    y_tr_samp = np.random.permutation(y_train)[:500]
    X_te_samp = np.random.permutation(X_test)[:100]
    y_te_samp = np.random.permutation(y_test)[:100]
    np.save(os.path.join(SAMP_DIR, "X_train.npy"), X_tr_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_train.npy"), y_tr_samp.astype(np.int64, copy=False))
    np.save(os.path.join(SAMP_DIR, "X_test.npy"),  X_te_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_test.npy"),  y_te_samp.astype(np.int64, copy=False)) 
    print("✅ Done. Saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
