#!/bin/bash
# 3702720
# ================================================================
# 2025-10-31
# Author: Dan Schumacher
# How to run:
#   chmod +x ./Classification/bin/simple_prompting.sh
#   nohup ./Classification/bin/simple_prompting.sh> ./logs/simple_prompting.log &
#   tail -f  ./logs/simple_prompting.log
# ================================================================

DATASETS=(
    "cpu"
    # "ecg"
    # "emg"
    # "har"
    # "rwc"
    # "tee"
)

MODELS=(
    "gpt"
    # "llama"
    # "mistral"
    # "gemma"
)

PROMPT_PATH="./Classification/prompts/zs.yaml"
OUT_ROOT="./Classification/data/generations"
IN_ROOT="./Classification/data/datasets"

for dataset in "${DATASETS[@]}"; do
    for model in "${MODELS[@]}"; do
        echo "=============================================================="
        echo "Running model: $model on dataset: $dataset"
        echo "=============================================================="

        python ./Classification/src/simple_prompting.py \
            --input_folder "${IN_ROOT}/${dataset}/" \
            --prompt_path "$PROMPT_PATH" \
            --model_type "$model" \
            --n_shots 3 \
            --temperature 0.7 \
            --batch_size 12 \
            --device_map 0 \
            --show_prompt 0
    done
done
