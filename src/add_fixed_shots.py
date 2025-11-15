'''
python ./src/add_fixed_shots.py
'''
import json
import sys, os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

import numpy as np

from global_utils.file_io import load_train_test
DATASETS = ["cpu", "ecg","emg","har","rwc","tee"]
OUTPUT_PATH_BASE = "./Classification/data/samples"
def main():
    np.random.seed(0)

    input_folder_base = "./Classification/data/samples"
    
    for dataset in DATASETS:
        out_path = f"{OUTPUT_PATH_BASE}/{dataset}/fixed_shot_indices.npy"
        train, _ = load_train_test(f"{input_folder_base}/{dataset}")

        # keys from JSON are strings; cast to int
        label_ids = [int(k) for k in train.label_maps["id_to_letter"].keys()]
        print(f"\n=== DATASET: {dataset} ===")
        print(f"label_ids: {label_ids}")

        shot_indices = []
        for label in label_ids:
            # get indices for this label
            indices = [i for i, l in enumerate(train.y) if int(l) == int(label)]

            if len(indices) == 0:
                print(f"No samples found for label {label} in dataset {dataset}")
                continue

            selected_index = int(np.random.choice(indices, 1, replace=False)[0])
            shot_indices.append(selected_index)
        with open(out_path, "wb") as f:
            np.save(f, np.array(shot_indices, dtype=np.int64))

if __name__ == "__main__":
    main()
