#!/bin/bash
# 1478950
# ================================================================
# 2025-10-31
# Author: Dan Schumacher
# chmod +x ./Classification/bin/text-only_prompting.sh
# nohup ./Classification/bin/text-only_prompting.sh > ./Classification/logs/text-only_prompting.log 2>&1 &
# tail -f ./Classification/logs/text-only_prompting.log
# ./Classification/bin/text-only_prompting.sh
# ================================================================

# exit on error
set -e

DATASETS=(
    # "cpu"
    # "ecg"
    # "emg"
    # "har"
    # "rwc"
    "tee"
)

MODELS=(
    "gpt-4o-mini"
)

NUMBER_OF_SHOTS=( # doesn't matter if SHOT_MODE=="random_per_class" (just something needs to be here)
    0
    # 3
    # 5
)

SHOT_MODES=(
    "random_per_class"
    # "dtw"
)

NORMALIZATION_BOOLS=(
    # 0
    1
)


# ---- Global options ----
BATCH_SIZE=2
TEMPERATURE=0.0
# ------------------------

IN_ROOT="./Classification/data/samples"

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    for shots in "${NUMBER_OF_SHOTS[@]}"; do
      for shot_mode in "${SHOT_MODES[@]}"; do
        for normalize in "${NORMALIZATION_BOOLS[@]}"; do


                  echo "=============================================================="
                  echo "STARTING PROMPTING..."
                  echo "=============================================================="

                  echo "Dataset:       $dataset"
                  echo "Model:         $model"
                  echo "Shots:         $shots"
                  echo "Shot mode:     $shot_mode"
                  echo "Normalize:     $normalize"

                # -------------------------------------------------------------------- #
                #   ---------- Run prompting script ----------
                #   python ./Classification/src/text-only_prompting_v1.py \
                #     --input_folder "${IN_ROOT}/${dataset}/" \
                #     --model_name "$model" \
                #     --n_shots "$shots" \
                #     --batch_size "$BATCH_SIZE" \
                #     --temperature "$TEMPERATURE" \
                #     --normalize "$normalize" \
                #     --shot_mode "$shot_mode" \
                #     --print_to_console 1 
                #     wait $!
                #     echo "Prompting complete."
                  # -------------------------------------------------------------------- #



                  # -------------------------------------------------------------------- #
                  printf "\n\n"
                  echo "=============================================================="
                  echo "STARTING EVALUATION..."
                  echo "=============================================================="

                  # ---------- Determine output directory ----------
                  if [[ $shot_mode == "random_per_class" ]]; then
                    FILE_NAME="rpc.jsonl"
                  elif [[ $shot_mode == "dtw" ]]; then
                    FILE_NAME="$n_shots-shot.jsonl"
                  else
                    echo "ERROR: Unknown shot_mode: $shot_mode"
                    exit 1
                  fi

                  if [[ $normalize -eq 1 ]]; then
                    NORM_TAG="normalized"
                  else
                    NORM_TAG="raw"
                  fi
                  
                  OUT_JSONL="./Classification/data/sample_generations/${dataset}/text-only_prompting/${NORM_TAG}/${FILE_NAME}"
                  # -------------------------------------------------------------------- #


                  # -------------------------------------------------------------------- #
                  # ---------- Run evaluation ----------
                  python ./Classification/src/eval.py \
                    --pred_path "$OUT_JSONL"
                  # -------------------------------------------------------------------- #



        done
      done
    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
