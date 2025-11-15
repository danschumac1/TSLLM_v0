#!/bin/bash
# 1478950
# ================================================================
# 2025-10-31
# Author: Dan Schumacher
# chmod +x ./Classification/bin/prompting.sh
# nohup ./Classification/bin/prompting.sh > ./Classification/logs/prompting.log 2>&1 &
# tail -f ./Classification/logs/prompting.log
# ./Classification/bin/prompting.sh
# ================================================================

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

VISION_BOOLS=(
    0
    # 1
)

INCLUDE_TS_BOOLS=(
    # 0
    1
)

NORMALIZATION_BOOLS=(
    # 0
    1
)

VIZ_METHODS=(
    "line"
    # "spectrogram"
)

IMG_DETAILS=(
    "auto"
    # "low"
    # "high"
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
        for use_vision in "${VISION_BOOLS[@]}"; do
          for include_ts in "${INCLUDE_TS_BOOLS[@]}"; do
            for normalize in "${NORMALIZATION_BOOLS[@]}"; do
              for viz in "${VIZ_METHODS[@]}"; do
                for img_detail in "${IMG_DETAILS[@]}"; do

                  # skip invalid combo: both off will crash python (ValueError)
                  if [[ $use_vision -eq 0 && $include_ts -eq 0 ]]; then
                    echo "Skipping: dataset=${dataset}, model=${model}, shots=${shots}, shot_mode=${shot_mode}, use_vision=${use_vision}, include_ts=${include_ts} (invalid combo)"
                    continue
                  fi

                  echo "=============================================================="
                  echo "Dataset:       $dataset"
                  echo "Model:         $model"
                  echo "Shots:         $shots"
                  echo "Shot mode:     $shot_mode"
                  echo "Use vision:    $use_vision"
                  echo "Include TS:    $include_ts"
                  echo "Normalize:     $normalize"
                  echo "Viz method:    $viz"
                  echo "Img detail:    $img_detail"
                  echo "=============================================================="

                  ---------- Run prompting script ----------
                  python ./Classification/src/prompting.py \
                    --input_folder "${IN_ROOT}/${dataset}/" \
                    --model_name "$model" \
                    --n_shots "$shots" \
                    --batch_size "$BATCH_SIZE" \
                    --temperature "$TEMPERATURE" \
                    --use_vision "$use_vision" \
                    --include_ts "$include_ts" \
                    --normalize "$normalize" \
                    --img_detail "$img_detail" \
                    --visualization_method "$viz" \
                    --shot_mode "$shot_mode" \
                    --print_to_console 1 \
                    --clear_images 1
                    wait $!
                    echo "Prompting complete."

                  # ---------- Determine output directory ----------
                  if [[ $use_vision -eq 1 ]]; then
                    TYPE_DIR="visual-prompting/${viz}"
                    VISION_STR="vision"
                  else
                    TYPE_DIR="text-only-prompting"
                    VISION_STR="text-only"
                  fi

                  OUT_JSONL="./Classification/data/sample_generations/${dataset}/${TYPE_DIR}/${shots}-shot_${shot_mode}_${VISION_STR}_${viz}.jsonl"

                  echo "Expecting JSONL at: $OUT_JSONL"

                  # ---------- Run evaluation ----------
                  python ./Classification/src/eval.py \
                    --pred_path "$OUT_JSONL"

                done
              done
            done
          done
        done
      done
    done
  done
done

printf "\n\nFILE DONE RUNNING 🎉🎉🎉\n\n"
