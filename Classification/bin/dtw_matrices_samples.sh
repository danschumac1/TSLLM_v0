#!/bin/bash
# 3680240
# chmod +x ./Classification/bin/dtw_matrices_samples.sh
# ./Classification/bin/dtw_matrices_samples.sh
# nohup ./Classification/bin/dtw_matrices_samples.sh > ./logs/dtw_mat.log 2>&1 &
# tail -f ./logs/dtw_mat.log

DATASETS=(
    "cpu"
    "tee"
    "emg"
    "har"
    "rwc"
    "ecg"
)

SPLITS=3          # StratifiedKFold splits
KMAX=1           # max k to test
WINDOW_FRAC=0.01  # 10% Sakoe–Chiba band
ZNORM=1           # 1 = normalize, 0 = raw

for dataset in "${DATASETS[@]}"; do
    echo "=============================================================="
    echo "Running DTW-kNN on dataset: $dataset"
    echo "=============================================================="
    python ./Classification/src/build_dtw_matrices.py \
        --input_folder ./Classification/data/samples/$dataset \
        --window_frac $WINDOW_FRAC \
        --znorm $ZNORM
done
