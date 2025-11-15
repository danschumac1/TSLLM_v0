# ./Classification/src/method/classification/build_dtw_matrices.py
"""
2025-11-10
Author: Dan Schumacher

Build and cache:
  - D_full (train × train)
  - D_te_tr (test × train)

Run:
  python ./Classification/src/build_dtw_matrices.py \
      --input_folder ./Classification/sample/datasets/tee \
      --window_frac 0.1 \
      --znorm 1 \
      --verbose 1
"""

import os, sys, json, argparse, time
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from global_utils.file_io import load_train_test
from global_utils.ts import z_norm_series
from global_utils.logging_utils import MasterLogger
from Classification.src.utils.dtw_utils import (
    pairwise_dtw_train_train, pairwise_dtw_val_train,
)

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build DTW distance matrices (no training).")
    p.add_argument("--input_folder", type=str, required=True)
    p.add_argument("--window", type=int, default=None)
    p.add_argument("--window_frac", type=float, default=0.10)
    p.add_argument("--znorm", type=int, choices=[0,1], default=1)
    p.add_argument("--verbose", type=int, choices=[0,1,2], default=1)
    return p.parse_args(argv)

def main():
    args = parse_args()

    # Load data
    train, test = load_train_test(args.input_folder)
    X_tr, y_tr = np.asarray(train.X), np.asarray(train.y).ravel()
    X_te, y_te = np.asarray(test.X),  np.asarray(test.y).ravel()

    if X_tr.ndim != 3 or X_te.ndim != 3:
        raise ValueError(f"Expected (N, V, T). Got train {X_tr.shape}, test {X_te.shape}")

    nvars, T = X_tr.shape[1], X_tr.shape[2]

    # Window
    used_frac = args.window is None
    window = args.window if not used_frac else max(1, int(round(args.window_frac * T)))

    # Paths
    dataset = os.path.basename(os.path.normpath(args.input_folder))
    out_dir = os.path.join("./Classification/data/samples", dataset, "dtw_knn")
    os.makedirs(out_dir, exist_ok=True)


    # Logger
    logs_dir = "./Classification/logs/dtw_knn"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}_mats.log")
    logger = MasterLogger(log_path=log_path, init=True, clear=True)

    print("Building DTW matrices")
    print(f"Dataset: {dataset}")
    print(f"Train size: {len(y_tr)} | Test size: {len(y_te)}")
    print(f"T: {T} | d: {nvars}")
    print(f"Window: {window} ({'frac' if used_frac else 'abs'})")
    print(f"Z-norm: {args.znorm}")

    # --- Z-normalize ---
    if args.znorm:
        print("Z-normalizing...")
        X_tr = z_norm_series(X_tr)
        X_te = z_norm_series(X_te)

    # --- D_full ---
    print("Computing D_full (train × train)...")
    t0 = time.perf_counter()
    D_full = pairwise_dtw_train_train(X_tr, window=window, logger=logger, verbose=args.verbose)
    D_full = np.asarray(D_full, float)
    D_full = 0.5 * (D_full + D_full.T)
    np.fill_diagonal(D_full, 0.0)
    np.save(os.path.join(out_dir, "D_full.npy"), D_full)
    print(f"D_full done in {time.perf_counter()-t0:.2f}s")
    print(f"Saved D_full.npy @ {out_dir}")


    # --- D_te_tr ---
    print("Computing D_te_tr (test × train)...")
    t0 = time.perf_counter()
    D_te_tr = pairwise_dtw_val_train(X_te, X_tr, window=window, logger=logger, verbose=args.verbose)
    D_te_tr = np.asarray(D_te_tr, float)
    np.save(os.path.join(out_dir, "D_te_tr.npy"), D_te_tr)
    print(f"D_te_tr done in {time.perf_counter()-t0:.2f}s")
    print(f"Saved D_te_tr.npy @ {out_dir}")

    # --- Metadata ---
    meta = {
        "dataset": dataset,
        "window": int(window),
        "window_frac": float(args.window_frac),
        "znorm": int(args.znorm),
        "nvars": int(nvars),
        "T": int(T),
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "used_frac": bool(used_frac),
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("✅ Finished building DTW matrices.")

if __name__ == "__main__":
    np.random.seed(42)
    main()
