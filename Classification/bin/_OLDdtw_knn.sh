#!/bin/bash
# 3680240
# chmod +x ./Classification/bin/dtw_knn.sh
# ./Classification/bin/dtw_knn.sh
# nohup ./Classification/bin/dtw_knn.sh > ./logs/dtw_knn.log 2>&1 &
# tail -f ./logs/dtw_knn.log


DATASETS=(
    # "cpu"
    # "tee"
    # "emg"
    # "har"
    "rwc"
    # "ecg"
)

SPLITS=3          # StratifiedKFold splits
KMAX=1           # max k to test
WINDOW_FRAC=0.01  # 10% Sakoe–Chiba band
ZNORM=1           # 1 = normalize, 0 = raw

for dataset in "${DATASETS[@]}"; do
    echo "=============================================================="
    echo "Running DTW-kNN on dataset: $dataset"
    echo "=============================================================="
    python ./Classification/src/dtw_knn.py \
        --input_folder ./Classification/data/samples/$dataset \
        --splits $SPLITS \
        --kmax $KMAX \
        --window_frac $WINDOW_FRAC \
        --znorm $ZNORM
done
