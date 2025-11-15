#!/usr/bin/env bash
# 3218600
# to run:
#   chmod +x ./Clasification/bin/knn_infer.sh
#   ./Clasification/bin/knn_infer.sh
#   nohup ./Clasification/bin/knn_infer.sh > ./logs/knn_infer_nohup.log 2>&1 &
#   tail -f ./logs/knn_infer_nohup.log 2>&1 &

# Batch inference using params chosen by CV and stored in the TSV.

# set -euo pipefail
debug=1

datasets=("TimerBed")
splits=("test")   # add "dev" if you like
cv_metric="accuracy"   # must match rows in TSV (or it will pick best overall for that ds/subset)
n_jobs=-1

params_tsv="./Classification/data/generations/knn_cv_params.tsv"
gen_root="./data/generations/knn_classifier"
mkdir -p "${gen_root}"

for dataset in "${datasets[@]}"; do

  input_folder="./Classification/data/datasets/${dataset}/"
  if [[ ! -f "${input_path}" ]]; then
    echo "SKIP: missing train ${train_path}"
    continue
  fi

  for split in "${splits[@]}"; do
    input_path="./data/datasets/${dataset}/${subset}/${split}.jsonl"
    if [[ ! -f "${input_path}" ]]; then
      echo "SKIP: missing input ${input_path}"
      continue
    fi

    out_dir="${gen_root}/${dataset}/${subset}/auto_from_tsv"
    out_path="${out_dir}/${split}.jsonl"
    mkdir -p "${out_dir}"

    echo "INFER ${dataset}/${subset}/${split}"
    echo "  TSV: ${params_tsv}"
    echo "  Out: ${out_path}"

    python ./src/method/knn_infer.py \
      --input_path "${input_path}" \
      --train_path "${train_path}" \
      --params_tsv "${params_tsv}" \
      --output_path "${out_path}" \
      --cv_metric "${cv_metric}" \
      --n_jobs "${n_jobs}" \
      --debug "${debug}"
  done
done
