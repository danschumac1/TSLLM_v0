"""
2025-10-23
Author: Dan Schumacher

Purpose
-------
Slice three long EMG class recordings into fixed-length windows for
classification, validate shapes/labels, and save NumPy splits.

Dataset summary (sanity checks)
-------------------------------
name:          emg
variables:     1 (univariate)
series length: 1500
classes:       3 → {'healthy':0, 'myopathy':1, 'neuropathy':2}
samples:       ≈205 total windows                               NOTE: expected 205, we got 314
source:        https://physionet.org/content/emgdb/1.0.0/

Inputs (expected layout)
------------------------
RAW_DIR/
  emg_healthy.txt
  emg_myopathy.txt
  emg_neuropathy.txt

Each file is a single long signal with lines like:
  <time> <value>
  or values separated by spaces/commas (we read the 2nd numeric as the value).

Windowing policy
----------------
- window_size = 1500, stride = 1500 (non-overlapping)
- Leftovers < 1500 at the end of a file are dropped.
- Each window inherits the file's class label.

Outputs
-------
OUT_DIR/
    X_train.npy : float32, shape (N_train, 1, 1500)
    y_train.npy : int64,   shape (N_train,)
    X_test.npy  : float32, shape (N_test, 1, 1500)
    y_test.npy  : int64,   shape (N_test,)


Run
---
python ./src/clean_emg.py
"""

import os
from typing import Dict, List, Tuple

import numpy as np

# ----------------------------- CONFIG --------------------------------
RAW_DIR     = "./raw_data/emg"
OUT_DIR     = "Classification/data/datasets/emg"
SAMP_DIR    = "./Classification/data/samples/emg/"
WINDOW_SIZE = 1500                   # samples per window
STRIDE      = 1500                   # non-overlapping
DTYPE       = np.float32
TRAIN_FRAC  = 0.85
SEED        = 1337

CLASS_FILES: Dict[str, str] = {
    "healthy":    "emg_healthy.txt",
    "myopathy":   "emg_myopathy.txt",
    "neuropathy": "emg_neuropathy.txt",
}
CLASS_LABELS: Dict[str, int] = {"healthy": 0, "myopathy": 1, "neuropathy": 2}

# --------------------------- IO HELPERS ------------------------------

def stratified_train_test_split(y: np.ndarray, train_frac: float, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    train_idx = []
    test_idx = []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        k = int(round(train_frac * len(cls_idx)))
        train_idx.append(cls_idx[:k])
        test_idx.append(cls_idx[k:])
    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _decode_lines(path: str):
    """Yield lines from a text file with tolerant decoding."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            yield from f; return
    except UnicodeDecodeError:
        pass
    try:
        with open(path, "r", encoding="latin-1") as f:
            yield from f; return
    except UnicodeDecodeError:
        pass
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        yield from f

def read_time_value_txt(path: str) -> np.ndarray:
    """
    Read a time–value style text file and return a 1D float32 array of values.

    Accepts lines like:
      "t v", "t, v", or extra whitespace/commas. Ignores comments starting with '#'.
    Takes the SECOND token as the numeric value.

    Returns
    -------
    np.ndarray of shape (T,), dtype=DTYPE
    """
    values: List[float] = []
    for raw in _decode_lines(path):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.replace(",", " ").split()
        if len(parts) < 2:
            continue
        try:
            values.append(float(parts[1]))
        except ValueError:
            continue
    if not values:
        raise RuntimeError(f"No numeric values parsed from {path}")
    return np.asarray(values, dtype=DTYPE)

def sliding_windows_1d(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    """
    Create 2D array of non-overlapping windows from 1D signal.

    Drops final remainder if shorter than window_size.

    Returns
    -------
    np.ndarray of shape (num_windows, window_size)
    """
    x = x.ravel()
    if x.size < window_size:
        return np.empty((0, window_size), dtype=x.dtype)
    n_windows = (x.size - window_size) // stride + 1
    out = np.empty((n_windows, window_size), dtype=x.dtype)
    for i in range(n_windows):
        start = i * stride
        out[i] = x[start : start + window_size]
    return out

def to_nyz(X: np.ndarray) -> np.ndarray:
    """Force (N, Y, Z). If (N, Z) -> (N, 1, Z)."""
    X = np.asarray(X)
    if X.ndim == 2:
        N, Z = X.shape
        return X.reshape(N, 1, Z)
    if X.ndim == 3:
        return X
    raise ValueError(f"Expected 2D or 3D, got {X.shape}")


# ------------------------------- MAIN --------------------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    all_windows: List[np.ndarray] = []
    all_labels:  List[np.ndarray] = []

    print(f"Loading class files from {RAW_DIR} and slicing into {WINDOW_SIZE}-sample windows ...")
    for class_name, filename in CLASS_FILES.items():
        file_path = os.path.join(RAW_DIR, filename)
        long_signal = read_time_value_txt(file_path)
        windows = sliding_windows_1d(long_signal, window_size=WINDOW_SIZE, stride=STRIDE)

        labels = np.full((windows.shape[0],), CLASS_LABELS[class_name], dtype=np.int64)

        all_windows.append(windows)
        all_labels.append(labels)

        print(f"  {class_name:<12} → {windows.shape[0]} windows of length {WINDOW_SIZE} "
              f"(from {long_signal.size} samples)")

    X_all = np.vstack(all_windows).astype(DTYPE)   # (N, 1500)
    y_all = np.concatenate(all_labels).astype(np.int64)  # (N,)
    print(f"Total windows: {y_all.size} | X shape: {X_all.shape}")

    # Stratified split (preserves per-class proportions)
    train_idx, test_idx = stratified_train_test_split(y_all, train_frac=TRAIN_FRAC, seed=SEED)
    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test,  y_test  = X_all[test_idx],  y_all[test_idx]
    X_train = to_nyz(X_train)
    X_test  = to_nyz(X_test)



    # Sanity checks & summary
    assert X_train.ndim == 3 and X_test.ndim == 3, "Expect 3D (N,Y,Z) arrays"
    assert X_train.shape[1] == 1 and X_test.shape[1] == 1, "EMG should be univariate (Y=1)"
    assert X_train.shape[2] == WINDOW_SIZE and X_test.shape[2] == WINDOW_SIZE, "Series length mismatch!"
    
    def dist(y: np.ndarray) -> dict:
        u, c = np.unique(y, return_counts=True)
        return dict(zip(u.tolist(), c.tolist()))

    print("  Train:", X_train.shape, "| label dist:", dist(y_train))
    print("  Test: ", X_test.shape,  "| label dist:", dist(y_test))
    print(f"  Total: {X_train.shape[0] + X_test.shape[0]} (expect ≈205 if using the same raw sources)\n")


        # Save to disk using project helper (writes X.npy/y.npy under subfolders)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train.astype(np.float32, copy=False))
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train.astype(np.int64,   copy=False))
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test.astype(np.float32,  copy=False))
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test.astype(np.int64,    copy=False))

    os.makedirs(SAMP_DIR, exist_ok=True)
    # sample
    X_tr_samp = np.random.permutation(X_train)[:500]
    y_tr_samp = np.random.permutation(y_train)[:500]
    X_te_samp = np.random.permutation(X_test)[:100]
    y_te_samp = np.random.permutation(y_test)[:100]
    np.save(os.path.join(SAMP_DIR, "X_train.npy"), X_tr_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "X_test.npy"),  X_te_samp.astype(np.float32, copy=False))   
    np.save(os.path.join(SAMP_DIR, "y_train.npy"), y_tr_samp.astype(np.int64, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_test.npy"),  y_te_samp.astype(np.int64, copy=False))
    print(f"\nSAVED FILES SUCCESSSFULLY  |  {OUT_DIR}")


if __name__ == "__main__":
    main()
