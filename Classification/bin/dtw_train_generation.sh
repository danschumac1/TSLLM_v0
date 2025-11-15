#!/bin/bash
# 3680240
# chmod +x ./Classification/bin/dtw_train_generation.sh
# ./Classification/bin/dtw_train_generation.sh
# nohup ./Classification/bin/dtw_train_generation.sh > ./logs/dtw_train_generation.log 2>&1 &
# tail -f ./logs/dtw_train_generation.log

DATASETS=(
    "cpu"
    "tee"
    "emg"
    "har"
    "rwc"
    "ecg"
)

SPLITS=3          # StratifiedKFold splits
KMAX=10           # max k to test
WINDOW_FRAC=0.01  # 10% Sakoe–Chiba band
ZNORM=1           # 1 = normalize, 0 = raw

for dataset in "${DATASETS[@]}"; do
    echo "=============================================================="
    echo "Running DTW-kNN on dataset: $dataset"
    echo "=============================================================="
    python ./Classification/src/dtw_knn.py \
        --input_folder ./Classification/data/samples/$dataset \
        --splits $SPLITS \
        --kmax $KMAX 


    # ------------------------------------------------------------------
    # Run eval
    # ------------------------------------------------------------------
    python ./Classification/src/eval.py \
        --pred_path "./Classification/data/sample_generations/${dataset}/dtw_knn/dtw_knn.jsonl"
done
