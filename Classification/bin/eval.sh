#!/usr/bin/env bash
# chmod +x ./Classification/bin/eval.sh
# ./Classification/bin/eval.sh

set -euo pipefail
IFS=$'\n\t'

# ----------------------------
# CONFIG
# ----------------------------
DATASETS=(
  cpu 
  ecg 
  emg 
  har 
  rwc 
  tee
  )
METHODS=(
  # random 
  # simple_prompting 
  # dtw_knn
  visual_prompting
)

# ----------------------------
# MAIN LOOP
# ----------------------------
for dataset in "${DATASETS[@]}"; do
  for method in "${METHODS[@]}"; do

    # Fresh MODES each method/dataset
    MODES=()

    if [[ "$method" == "random" ]]; then
      MODES=("uniform" "prior" "majority")

    elif [[ "$method" == "simple_prompting" ]]; then
      MODES=("NS0" "NS3")
    elif [[ "$method" == "visual_prompting" ]]; then
      MODES=("0_shot" "3_shot" "5_shot")


    elif [[ "$method" == "dtw_knn" ]]; then
      # Build MODES from filenames like S3_KM5_WF10p_Wd_Z1_Nd1.jsonl
      base_dir="./Classification/data/sample_generations/${dataset}/${method}"

      if [[ ! -d "$base_dir" ]]; then
        echo "⚠️  Directory missing for $dataset/$method: $base_dir"
        continue
      fi

      shopt -s nullglob
      files=("$base_dir"/*.jsonl)
      shopt -u nullglob

      if (( ${#files[@]} == 0 )); then
        echo "⚠️  No .jsonl files found for $dataset/$method in $base_dir"
        continue
      fi

      for f in "${files[@]}"; do
        fname="$(basename "$f")"
        # Skip any metrics sidecar files, just in case they appear as .jsonl
        if [[ "$fname" == *_metrics.jsonl ]]; then
          continue
        fi
        MODES+=("${fname%.jsonl}")
      done
    fi

    # Nothing to do?
    if (( ${#MODES[@]} == 0 )); then
      echo "⚠️  No modes resolved for dataset=$dataset method=$method"
      continue
    fi

    # Evaluate
    for mode in "${MODES[@]}"; do
      pred_path="./Classification/data/sample_generations/${dataset}/${method}/${mode}.jsonl"
      if [[ -f "$pred_path" ]]; then
        echo "-------------------------------------------"
        echo " Evaluating: dataset=$dataset | method=$method | mode=$mode "
        echo "-------------------------------------------"
        python ./Classification/src/eval.py --pred_path "$pred_path"
      else
        echo "⚠️  Skipping missing file: $pred_path"
      fi
      echo ""
    done

  done
done
