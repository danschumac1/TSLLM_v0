"""
2025-10-31
Author: Dan Schumacher

Purpose
-------
Load the UCI HAR dataset either as:
  (A) raw triaxial body acceleration windows → shape (N, 3, 128), or
  (B) engineered 561-D feature vectors       → shape (N, 561),
then normalize labels and save NumPy arrays.

Dataset summary (sanity checks)
-------------------------------
name:          har (UCI HAR)
n_variables:   3  (for raw body_acc x,y,z)
series length: 128 (2.56 s @ 50 Hz)
n_classes:     6  (activities 1..6 → remapped to 0..5)
n_samples:     10,299 (7,352 train + 2,947 test)
source:        https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

Inputs (expected layout)
------------------------
RAW/
  train/X_train.txt, y_train.txt
  train/Inertial Signals/body_acc_{x,y,z}_train.txt
  test/X_test.txt,  y_test.txt
  test/Inertial Signals/body_acc_{x,y,z}_test.txt

Outputs
-------
OUT/
  X_train.npy, y_train.npy
  X_test.npy,  y_test.npy

Run
---
python ./src/clean_har.py

"""

import os
import numpy as np

RAW = "./raw_data/har"
OUT = "./Classification/data/har"
DTYPE = np.float32
SEED = 1337


def load_labels(path: str) -> np.ndarray:
    y = np.loadtxt(path, dtype=np.int64)
    # remap activity IDs 1..6 -> 0..5
    y = y - 1
    return y

def load_raw3(split: str):
    base = os.path.join(RAW, split, "Inertial Signals")
    # Each file is (N, 128)
    Xx = np.loadtxt(os.path.join(base, f"body_acc_x_{split}.txt"), dtype=DTYPE)
    Xy = np.loadtxt(os.path.join(base, f"body_acc_y_{split}.txt"), dtype=DTYPE)
    Xz = np.loadtxt(os.path.join(base, f"body_acc_z_{split}.txt"), dtype=DTYPE)

    # Stack into (N, 3, 128)
    X = np.stack([Xx, Xy, Xz], axis=1)
    y = load_labels(os.path.join(RAW, split, f"y_{split}.txt"))
    return X, y

def summarize(name: str, X: np.ndarray, y: np.ndarray):
    uniq, cnt = np.unique(y, return_counts=True)
    dist = {int(k): int(v) for k, v in zip(uniq, cnt)}
    print(f"[{name}] X: {X.shape}, y: {y.shape}, label dist: {dist}")

def main():
    X_train, y_train = load_raw3("train")
    X_test, y_test = load_raw3("test")

    # Optional: shuffle train for randomness
    rng = np.random.default_rng(SEED)
    p = rng.permutation(len(y_train))
    X_train, y_train = X_train[p], y_train[p]

    summarize("TRAIN", X_train, y_train)
    summarize("TEST",  X_test, y_test)
    print(f"TOTAL samples: {len(y_train) + len(y_test)}")
    assert X_train.shape[1:] == (3, 128) and X_test.shape[1:] == (3, 128), "Expected (N, 3, 128) for raw3."

    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "X_train.npy"), X_train)
    np.save(os.path.join(OUT, "y_train.npy"), y_train)
    np.save(os.path.join(OUT, "X_test.npy"),  X_test)
    np.save(os.path.join(OUT, "y_test.npy"),  y_test)


    print("✅ Saved to", OUT)

if __name__ == "__main__":
    main()
