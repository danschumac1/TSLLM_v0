# ./Classification/src/method/classification/dtw_knn.py
"""
2025-10-31
Author: Dan Schumacher

Run:
  python ./Classification/src/dtw_knn.py \
    --input_folder ./Classification/data/datasets/tee \
    --verbose 2
"""
import os, sys, json, argparse, time
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from global_utils.file_io import load_train_test  # returns (train_split, test_split)
from global_utils.ts import z_norm_series
from global_utils.logging_utils import MasterLogger


# -------------------- CLI --------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description="DTW-kNN classifier with precomputed distances (+ caching), multivariate-aware")
    p.add_argument("--input_folder", type=str, required=True,
                   help="Folder with y_train.npy/y_test.npy and X_train.npy/X_test.npy")
    p.add_argument("--splits", type=int, default=3, help="KFold splits for CV")
    p.add_argument("--kmax", type=int, default=10, help="Max k to try (1..kmax-1 bounded by n)")
    return p.parse_args(argv)

# -------------------- Main --------------------
def main():
    args = parse_args()

    # Load first (so we know T/nvars before cache/window building)
    train, test = load_train_test(args.input_folder)
    X_tr, y_tr = np.asarray(train.X), np.asarray(train.y).ravel()
    X_te, y_te = np.asarray(test.X),  np.asarray(test.y).ravel()

    # Enforce new invariant: 3D (N, V, T)
    if X_tr.ndim != 3 or X_te.ndim != 3:
        raise ValueError(f"Expected 3D arrays (N, V, T). Got train {X_tr.shape}, test {X_te.shape}")

    nvars = X_tr.shape[1]
    T = X_tr.shape[2]

    # Paths
    dataset = os.path.basename(os.path.normpath(args.input_folder))
    out_dir = os.path.join("./Classification/data/sample_generations", dataset, "dtw_knn")
    os.makedirs(out_dir, exist_ok=True)

    # Logger
    logs_dir = "./Classification/logs/dtw_knn"
    os.makedirs(logs_dir, exist_ok=True)
    log_path = os.path.join(logs_dir, f"{dataset}.log")
    logger = MasterLogger(log_path=log_path, init=True, clear=True)

    # load chached datas 
    base = f"Classification/data/samples/{dataset}/dtw_knn"
    D_full = np.load(os.path.join(base, "D_full.npy"))
    D_te_tr = np.load(os.path.join(base, "D_te_tr.npy"))
    metadata= json.load(open(os.path.join(base, "metadata.json"), "r"))
    logger.info(f"Loaded cached distance matrices from {base}")
    logger.info(f"Metadata: {metadata}")
    logger.info(f"X | Train shape: {X_tr.shape} | Test shape: {X_te.shape}")
    logger.info(f"y | Train shape: {y_tr.shape} | Test shape: {y_te.shape}")

    # Optional z-norm (per series; per variable if multivariate)
    if metadata.get("znorm", False): # either 1 or 0
        logger.info("Z-normalizing X-train and X-test (per series, per variable)…")
        tnorm = time.perf_counter()
        X_tr = z_norm_series(X_tr)
        X_te = z_norm_series(X_te)
        logger.info(f"Z-norm done in {(time.perf_counter()-tnorm):.2f}s")

    # ---- CV: choose k and weights (reusing slices of D_full) ----
    n = X_tr.shape[0]
    ks = list(range(1, min(n, args.kmax)))
    weight_opts = ["uniform", "distance"]
    cv = StratifiedKFold(n_splits=args.splits, shuffle=True, random_state=42)
    
    # set up for visual progress bar
    total_iters = len(ks) * len(weight_opts) * args.splits
    pbar = tqdm(total=total_iters, desc="Cross-validation", ncols=100)

    # initialize to crap values
    best_score, best_k, best_w = -1.0, None, None
    cv_results = []  # store detailed results

    logger.info(f"[CV] ks={ks} | weights={weight_opts} | splits={args.splits}")

    for k in ks:
        for w in weight_opts:
            scores = []
            fold_details = []
            for fold_id, (tr_idx, va_idx) in enumerate(cv.split(X_tr, y_tr), start=1):

                D_tr = D_full[np.ix_(tr_idx, tr_idx)]
                D_va = D_full[np.ix_(va_idx, tr_idx)]

                clf = KNeighborsClassifier(metric='precomputed', n_neighbors=k, weights=w)
                tfit = time.perf_counter()
                clf.fit(D_tr, y_tr[tr_idx])
                tfit = time.perf_counter() - tfit

                tpred = time.perf_counter()
                va_pred = clf.predict(D_va)
                tpred = time.perf_counter() - tpred

                fold_acc = accuracy_score(y_tr[va_idx], va_pred)
                scores.append(fold_acc)
                pbar.update(1)


                fold_details.append({
                    "fold": fold_id,
                    "k": k,
                    "w": w,
                    "acc": float(fold_acc),
                    "fit_s": float(tfit),
                    "pred_s": float(tpred),
                    "n_tr": int(len(tr_idx)),
                    "n_va": int(len(va_idx)),
                })

            mean_score = float(np.mean(scores))
            cv_results.append({
                "k": k, "weights": w, "scores": scores,
                "mean_acc": mean_score, "folds": fold_details
            })

            if mean_score > best_score:
                best_score, best_k, best_w = mean_score, k, w

    pbar.close()
    logger.info(f"Best k     : {best_k}")
    logger.info(f"Best weight: {best_w}")
    logger.info(f"CV acc     : {best_score:.4f}")

    # Dump CV results
    cv_json = os.path.join(out_dir, "cv_results.json")
    with open(cv_json, "w", encoding="utf-8") as f:
        json.dump(cv_results, f, indent=2)
    logger.info(f"[OUT] Saved CV details → {cv_json}")

    # ---- Fit on full train with best hyperparams ----
    logger.info("[FINAL] Fitting kNN on full train with best hyperparams…")
    clf = KNeighborsClassifier(metric='precomputed', n_neighbors=best_k, weights=best_w)
    tfit = time.perf_counter()
    clf.fit(D_full, y_tr)
    logger.info(f"[FINAL] Fit time: {time.perf_counter()-tfit:.2f}s")


    # ---- Predict test ----
    logger.info("[FINAL] Predicting test set…")
    tpred = time.perf_counter()
    y_pred = clf.predict(D_te_tr)

    pred_s = time.perf_counter() - tpred
    logger.info(f"[FINAL] Prediction time: {pred_s:.2f}s (N_test={len(y_te)})")

    # ---- Save JSONL with concise tag ----
    out_file = os.path.join(out_dir, f"dtw_knn.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for idx, pred, gt in zip(test.idx.tolist(), y_pred.tolist(), y_te.tolist()):
            f.write(json.dumps({"idx": int(idx), "pred": int(pred), "gt": int(gt)}) + "\n")
    logger.info(f"[OUT] Saved predictions JSONL → {out_file}")

    # ---- Metrics ----
    acc = accuracy_score(y_te, y_pred)
    f1m = f1_score(y_te, y_pred, average="macro")
    logger.info(f"Accuracy   : {acc:.4f} ({acc:.2%})")
    logger.info(f"Macro F1   : {f1m:.4f} ({f1m:.2%})")

    # Confusion matrix + report
    cm = confusion_matrix(y_te, y_pred)
    logger.info("[CONFUSION MATRIX]\n" + np.array2string(cm, max_line_width=150))
    cls_rep = classification_report(y_te, y_pred, digits=4)
    logger.info("[CLASSIFICATION REPORT]\n" + cls_rep)

    # Save metrics bundle
    metrics_path = os.path.join(out_dir, f"metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset,
            "acc": float(acc),
            "macro_f1": float(f1m),
            "best_k": int(best_k),
            "best_weights": str(best_w),
            "n_train": int(len(y_tr)),
            "n_test": int(len(y_te)),
            "T": int(T),
            "d": int(nvars),
            "window": int(metadata['window']),
            "used_frac": bool(metadata['used_frac']),
            "znorm": int(metadata['znorm']),
            "pred_time_s": float(pred_s),
        }, f, indent=2)
    logger.info(f"[OUT] Saved metrics JSON → {metrics_path}")

    logger.info("✅ Done.")

if __name__ == "__main__":
    np.random.seed(42)
    main()
