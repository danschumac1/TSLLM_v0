#!/bin/bash
# chmod +x ./Classification/bin/random_baseline.sh
# ./Classification/bin/random_baseline.sh

DATASETS=(
    # "cpu"
    "ecg"
    "emg"
    "har"
    "rwc"
    "tee"
)

MODES=(
    "uniform"
    "prior"
    "majority"
)

for dataset in "${DATASETS[@]}"; do
    for mode in "${MODES[@]}"; do
        python ./Classification/src/random_baseline.py \
            --input_folder "./Classification/data/samples/$dataset" \
            --mode $mode

        python ./Classification/src/eval.py \
            --pred_path "Classification/data/sample_generations/${dataset}/random/${mode}.jsonl"
    done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"

# not samples
# --input_folder ./Classification/data/datasets/$dataset \