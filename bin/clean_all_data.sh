#!/usr/bin/env bash
# ------------------------------------------------------------------------------
# Clean and preprocess all InstructionTime datasets
# Author: Dan Schumacher
# Date: 2025-10-30
#
# Usage:
#   chmod +x ./bin/clean_all_data.sh
#   ./bin/clean_all_data.sh
#
# Notes:
# - Each called script is self-contained and writes to Classification/data/<dataset>/
# - Output and errors are logged to ./logs/clean_<dataset>.log
# - Stops immediately if any script fails (set -e)
# ------------------------------------------------------------------------------

set -e  # stop on first error
set -o pipefail

SRC_DIR="./src"
LOG_DIR="./logs"

mkdir -p "$LOG_DIR"

declare -a datasets=(
  "cpu"
  "ecg"
  "emg"
  "har"
  "rwc"
  "tee"
)

echo "=============================================="
echo " Cleaning all datasets..."
echo "=============================================="

for ds in "${datasets[@]}"; do
  echo ""
  echo "➡️  Cleaning ${ds^^} ..."
  python "${SRC_DIR}/clean_${ds}.py" \
    >"${LOG_DIR}/clean_${ds}.log" 2>&1 \
    && echo "✅ ${ds^^} done." \
    || { echo "❌ ${ds^^} failed! Check ${LOG_DIR}/clean_${ds}.log"; exit 1; }
done

echo ""
echo "=============================================="
echo "✅ All dataset cleaning complete!"
echo "Logs saved to: $LOG_DIR"
echo "=============================================="
