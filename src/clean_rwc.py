"""
2025-10-23
Author: Dan Schumacher

Purpose
-------
Load Cornell Whale (RWC) AIFF audio files + labels, convert to fixed-length
mono waveforms, perform a stratified 85/15 split, and save NumPy arrays.

Dataset summary (sanity checks)
-------------------------------
name:          whale (rwc)
variables:     1 (mono)
series length: 4000 samples (center-crop/pad)
classes:       2 (labels expected as {0,1} or {1,2} in train.csv)
source:        train/*.aiff|*.aif with labels in train.csv (filename,label)

Inputs (expected layout)
------------------------
RAW_ROOT/
  train/
    <audio>.aiff (or .aif)
  train.csv      : "filename,label" (header optional). filename may include extension.

Outputs
-------
OUT_DIR/
    X_train.npy : float32, shape (N_train, 1, 4000)
    y_train.npy : int64,   shape (N_train,)
    X_test.npy  : float32, shape (N_test,  1, 4000)
    y_test.npy  : int64,   shape (N_test,)

Run
---
python ./src/clean_rwc.py
"""

import csv
import os
from typing import List, Dict, Tuple

import numpy as np
import soundfile as sf

# ----------------------------- CONFIG --------------------------------
RAW_ROOT   = "raw_data/rwc"
TRAIN_DIR  = os.path.join(RAW_ROOT, "train")      # AIFFs here
TRAIN_CSV  = os.path.join(RAW_ROOT, "train.csv")  # filename,label

OUT_DIR    = "./Classification/data/datasets/rwc"
SAMP_DIR   = "./Classification/data/samples/rwc/"
TARGET_LEN = 4000
DTYPE      = np.float32

TRAIN_FRAC = 0.85     # stratified 85/15 split
SEED       = 1337
SHUFFLE_WITHIN_SPLITS = True

# ----------------------------- HELPERS --------------------------------
def stratified_train_test_split(y: np.ndarray, train_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stratified split indices for train/test.

    Ensures that if a class has >=2 examples, both train and test receive
    at least one (when feasible).

    Parameters
    ----------
    y : np.ndarray
        Integer labels of shape (N,).
    train_frac : float
        Fraction of samples per class to assign to train.
    seed : int
        RNG seed.

    Returns
    -------
    (train_idx, test_idx) : Tuple[np.ndarray, np.ndarray]
        Integer index arrays.
    """
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    train_idx, test_idx = [], []

    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        # choose k with guards to avoid empty class in either split when n >= 2
        k = int(np.floor(train_frac * n))
        if n >= 2:
            k = min(max(1, k), n - 1)
        # if n == 1 → all go to train (k=0 would put it in test only)
        if n == 1:
            k = 1
        train_idx.append(cls_idx[:k])
        test_idx.append(cls_idx[k:])

    train_idx = np.concatenate(train_idx) if train_idx else np.array([], dtype=int)
    test_idx  = np.concatenate(test_idx)  if test_idx  else np.array([], dtype=int)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _to_mono(wave: np.ndarray) -> np.ndarray:
    """Ensure mono: average channel dimension if stereo/multi-channel."""
    return wave if wave.ndim == 1 else wave.mean(axis=1)


def _pad_trim_center(wave: np.ndarray, target_len: int, pad_value: float = 0.0) -> np.ndarray:
    """
    Center-crop if longer than target; center-pad if shorter.

    Returns
    -------
    np.ndarray of shape (target_len,)
    """
    n = wave.shape[0]
    if n == target_len:
        return wave
    if n > target_len:
        start = (n - target_len) // 2
        return wave[start:start + target_len]
    # pad
    pad_total = target_len - n
    left = pad_total // 2
    right = pad_total - left
    return np.pad(wave, (left, right), mode="constant", constant_values=pad_value)


def read_aiff(path: str, dtype: type = DTYPE) -> np.ndarray:
    """
    Read an AIFF/AIF file as mono float32 (or provided dtype).
    """
    wave, _sr = sf.read(path, always_2d=False)  # wave → (n,) or (n,ch)
    wave = np.asarray(wave, dtype=dtype)
    wave = _to_mono(wave)
    return wave


def load_aiff_row(path: str, target_len: int, dtype: type = DTYPE) -> np.ndarray:
    """
    Read, mono-ize, center pad/trim to target_len, and return as row (1, L).
    """
    wave = read_aiff(path, dtype=dtype)
    wave = _pad_trim_center(wave, target_len, pad_value=0.0).astype(dtype, copy=False)
    return wave[None, :]


def list_aiff_files(root: str) -> List[str]:
    """
    Recursively list AIFF/AIF files under root. Sorted for reproducibility.
    """
    paths: List[str] = []
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith((".aiff", ".aif", ".aifc")):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def stem(path: str) -> str:
    """Filename without extension."""
    return os.path.splitext(os.path.basename(path))[0]


def load_labels_csv(csv_path: str) -> Dict[str, int]:
    """
    Load filename → label mapping from CSV.

    CSV format:
      filename,label
    - Header is optional (auto-detected).
    - Filenames may include or omit extension; we match by stem.
    - Labels may be {0,1} or {1,2}; we remap {1,2} → {0,1}.

    Returns
    -------
    dict: {file_stem: int_label_in_{0,1}}
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing labels CSV: {csv_path}")

    mapping: Dict[str, int] = {}
    with open(csv_path, "r", encoding="utf-8") as f:
        sniff = f.read(2048)
        f.seek(0)
        has_header = any(h in sniff.lower() for h in ("filename", "file", "label"))
        reader = csv.reader(f)
        if has_header:
            _ = next(reader, None)

        for row in reader:
            if not row or len(row) < 2:
                continue
            filename_str = row[0].strip()
            label_str = row[1].strip()
            s = stem(filename_str)
            try:
                raw_label = int(float(label_str))
            except ValueError:
                raise ValueError(f"Invalid label {label_str!r} for {filename_str!r}")

            if raw_label in (0, 1):
                y = raw_label
            elif raw_label in (1, 2):
                y = raw_label - 1
            else:
                raise ValueError(f"Unexpected label {raw_label} for {filename_str}; expected 0/1 or 1/2.")
            mapping[s] = y
    if not mapping:
        raise RuntimeError(f"No valid (filename,label) pairs read from {csv_path}")
    return mapping


def summarize(name: str, X: np.ndarray, y: np.ndarray) -> None:
    """Print shapes and per-class counts."""
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    print(f"[{name}] X: {X.shape}  y: {y.shape}  class_dist: {dist}")

# ------------------------------- MAIN --------------------------------
def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- Load labels ---
    print("Reading labels CSV ...")
    label_map = load_labels_csv(TRAIN_CSV)  # stem -> {0,1}

    # --- Gather TRAIN audio files ---
    print("Scanning TRAIN AIFF files ...")
    train_files = list_aiff_files(TRAIN_DIR)
    if not train_files:
        raise RuntimeError(f"No AIFF files found in {TRAIN_DIR}")

    # --- Load waveforms matched to labels ---
    rows: List[np.ndarray] = []
    labels: List[int] = []
    skipped_missing_label = 0

    for path in train_files:
        s = stem(path)
        y = label_map.get(s)
        if y is None:
            skipped_missing_label += 1
            continue  # skip files missing labels
        try:
            row = load_aiff_row(path, target_len=TARGET_LEN, dtype=DTYPE)  # (1, L)
        except Exception as e:
            print(f"[WARN] Skipping unreadable file: {path} ({e})")
            continue
        rows.append(row)
        labels.append(int(y))

    if not rows:
        raise RuntimeError("No labeled samples loaded from TRAIN.")
    if skipped_missing_label:
        print(f"[INFO] Skipped {skipped_missing_label} files not present in train.csv.")

    X_all = np.vstack(rows).astype(DTYPE)            # (N, L)
    y_all = np.asarray(labels, dtype=np.int64)       # (N,)
    assert X_all.shape[1] == TARGET_LEN, "Unexpected waveform length after pad/trim."

    # --- Stratified split (TRAIN → train/test) ---
    print("Performing stratified 85/15 split ...")
    train_idx, test_idx = stratified_train_test_split(y_all, TRAIN_FRAC, SEED)

    X_train, y_train = X_all[train_idx], y_all[train_idx]
    X_test,  y_test  = X_all[test_idx],  y_all[test_idx]

    # Expand channel dim to (N, L, 1) for consistency with conv models
    if X_train.ndim == 2:
        X_train = X_train[:, None, :]
    if X_test.ndim == 2:
        X_test = X_test[:, None, :]


    # Optional: shuffle within splits for randomness
    if SHUFFLE_WITHIN_SPLITS:
        rng = np.random.default_rng(SEED)
        p = rng.permutation(len(y_train)); X_train, y_train = X_train[p], y_train[p]
        p = rng.permutation(len(y_test));  X_test,  y_test  = X_test[p],  y_test[p]

    # --- Assertions & summaries ---
    assert X_train.ndim == 3 and X_train.shape[1] == 1 and X_train.shape[2] == TARGET_LEN, "Expected (N, 1, 4000) for train."
    assert X_test.ndim  == 3 and X_test.shape[1]  == 1 and X_test.shape[2]  == TARGET_LEN, "Expected (N, 1, 4000) for test."
    assert X_train.shape[2] == TARGET_LEN and X_test.shape[2] == TARGET_LEN, "Length mismatch."

    summarize("TRAIN", X_train, y_train)
    summarize("TEST",  X_test,  y_test)

    # --- Save ---
    print("Saving arrays ...")
    np.save(os.path.join(OUT_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(OUT_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(OUT_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT_DIR, "y_test.npy"),  y_test)

    # sample
    os.makedirs(SAMP_DIR, exist_ok=True)
    X_tr_samp = np.random.permutation(X_train)[:500]
    y_tr_samp = np.random.permutation(y_train)[:500]
    X_te_samp = np.random.permutation(X_test)[:100]
    y_te_samp = np.random.permutation(y_test)[:100]
    np.save(os.path.join(SAMP_DIR, "X_train.npy"), X_tr_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "X_test.npy"),  X_te_samp.astype(np.float32, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_train.npy"), y_tr_samp.astype(np.int64, copy=False))
    np.save(os.path.join(SAMP_DIR, "y_test.npy"),  y_te_samp.astype(np.int64, copy=False))

    print("✅ Saved to", OUT_DIR)

if __name__ == "__main__":
    main()
