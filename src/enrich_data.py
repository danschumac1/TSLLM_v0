'''
python ./src/enrich_data.py --dataset <dataset>
E.g.,
    python ./src/enrich_data.py --dataset cpu
    python ./src/enrich_data.py --dataset ecg
    python ./src/enrich_data.py --dataset emg
    python ./src/enrich_data.py --dataset har
    python ./src/enrich_data.py --dataset rwc
    python ./src/enrich_data.py --dataset tee
'''


#!/usr/bin/env python
import os, sys, json, argparse, warnings

# Absolute path to the project root (TSLLM_v0)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import numpy as np
from pathlib import Path

# project imports
from global_utils.file_io import load_train_test
from Classification.src.utils.dtw_utils import ensure_dir
from Classification.src.utils.build_questions import (
    LABEL_MAPPING, _letters, _sort_key_for_label_id, build_question_text
)

# -------------------------------------------------
# Load distance matrices
# -------------------------------------------------
def load_dtw_mats(cache_dir: str = None):
    d_full = d_te = None
    if not os.path.isdir(cache_dir):
        raise FileNotFoundError(f"[CACHE] dir not found: {cache_dir}")

    for f in os.listdir(cache_dir):
        if not f.endswith(".npy"):
            continue
        fp = os.path.join(cache_dir, f)
        if f.startswith("D_full_"):
            d_full = fp
        elif f.startswith("D_te_tr_"):
            d_te = fp

    if not d_full or not d_te:
        raise FileNotFoundError("Missing D_full_*.npy or D_te_tr_*.npy in cache.")

    return np.load(d_full), np.load(d_te)

# -------------------------------------------------
# Letter maps
# -------------------------------------------------
def build_letter_maps(dataset: str):
    key = dataset.strip().upper()
    id_to_name = LABEL_MAPPING[key]
    items = sorted(id_to_name.items(), key=lambda kv: _sort_key_for_label_id(kv[0]))
    id_to_letter = {int(cid): _letters(i+1) for i,(cid,_) in enumerate(items)}
    letter_to_id = {letter: cid for cid, letter in id_to_letter.items()}
    return id_to_letter, letter_to_id, id_to_name

# -------------------------------------------------
# Args
# -------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    return p.parse_args()

# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    args = parse_args()
    dataset = args.dataset

    # where to write
    # out_dir = f"./Classification/data/datasets/{dataset}/"
    out_dir = f"./Classification/data/samples/{dataset}/"

    ensure_dir(out_dir)

    # Load train/test to know ordering
    # data_dir = f"./Classification/data/datasets/{dataset}"
    data_dir = f"./Classification/data/samples/{dataset}/"

    train, test = load_train_test(data_dir)
    X_tr, _ = np.asarray(train.X), np.asarray(train.y).ravel()
    X_te, _ = np.asarray(test.X),  np.asarray(test.y).ravel()

    # Load matrices
    _, D_te_tr = load_dtw_mats(f"./Classification/data/samples/cache/{dataset}/")
    print(f"D_te_tr shape: {D_te_tr.shape}")
    print(f"X_tr shape: {X_tr.shape}, X_te shape: {X_te.shape}")
    if D_te_tr.shape != (len(X_te), len(X_tr)):
        raise ValueError("D_te_tr shape mismatch with train/test sizes.")

    # Build maps + question
    id_to_letter, letter_to_id, id_to_name = build_letter_maps(dataset)
    question = build_question_text(dataset).strip()

    # -----------------------------
    # ALWAYS K=10
    # -----------------------------
    K = 10
    N_te = len(X_te)
    top_same = np.empty((N_te, K), dtype=np.int32)

    for i in range(N_te):
        dists = D_te_tr[i]
        order = np.argsort(dists)

        # ---- same-label (straight nearest 10)
        top_same[i] = order[:K]


    # -----------------------------
    # Write artifacts
    # -----------------------------
    np.save(os.path.join(out_dir, "top10_similar.npy"), top_same)

    with open(os.path.join(out_dir, "general_question.txt"), "w") as f:
        f.write(question)

    maps = {
        "letter_to_id": letter_to_id,                       # {"A": 3, "B": 7, ...}
        "id_to_letter": {str(k): v for k, v in id_to_letter.items()},
        "id_to_name":   {str(k): v for k, v in id_to_name.items()},
    }
    maps_path = os.path.join(out_dir, "label_maps.json")
    with open(maps_path, "w") as f:
        json.dump(maps, f, indent=2)

    print(f"[OK] wrote artifacts → {out_dir}")

if __name__ == "__main__":
    import numpy as np
    main()
