# ./Classification/src/append_manual_results.py
"""
2025-10-30
Author: Dan Schumacher

Purpose:
Append GPT-4o manual accuracy results to Classification/data/results.tsv
using the same schema as eval.py.

WARNING: RUN FIRST! ---> ./Classification/bin/eval.sh 

Run:
  python ./Classification/src/write_vltime_results_manually.py
"""

import os
from datetime import datetime

import sys; sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))  # → adds project root
from global_utils.file_io import ensure_header, append_rows

# --------------------------------------------------------
# CONFIG
# --------------------------------------------------------
TSV_PATH = "./Classification/data/results.tsv"

# dataset | model | prompt_type | setting | timestamp | acc
raw_rows = [
    ["rwc", "GPT-4o", "vl_time", "zs", "2025-10-30 00:00:00", "70.02"],
    ["tee", "GPT-4o", "vl_time", "zs", "2025-10-30 00:00:00", "24.88"],
    ["ecg", "GPT-4o", "vl_time", "zs", "2025-10-30 00:00:00", "26.33"],
    ["emg", "GPT-4o", "vl_time", "zs", "2025-10-30 00:00:00", "33.33"],
    ["har", "GPT-4o", "vl_time", "zs", "2025-10-30 00:00:00", "50.71"],
    ["cpu", "GPT-4o", "vl_time", "zs", "2025-10-30 00:00:00", "37.50"],
    ["rwc", "GPT-4o", "vl_time", "fs", "2025-10-30 00:00:00", "91.03"],
    ["tee", "GPT-4o", "vl_time", "fs", "2025-10-30 00:00:00", "64.29"],
    ["ecg", "GPT-4o", "vl_time", "fs", "2025-10-30 00:00:00", "43.75"],
    ["emg", "GPT-4o", "vl_time", "fs", "2025-10-30 00:00:00", "91.67"],
    ["har", "GPT-4o", "vl_time", "fs", "2025-10-30 00:00:00", "63.64"],
    ["cpu", "GPT-4o", "vl_time", "fs", "2025-10-30 00:00:00", "66.67"],
]

# target TSV header (same as eval.py)
header = [
    "dataset",
    "method",
    "mode",
    "accuracy",
    "macro_f1",
    "pred_path",
    "timestamp",
]

# --------------------------------------------------------
# MAIN
# --------------------------------------------------------
def main():
    os.makedirs(os.path.dirname(TSV_PATH), exist_ok=True)
    ensure_header(TSV_PATH, header)

    formatted_rows = []
    for dataset, model, prompt, setting, timestamp, acc in raw_rows:
        method = prompt  # "vl_time"
        mode = f"{model}_{setting}"  # e.g., "GPT-4o_zs"
        formatted_rows.append([
            dataset,
            method,
            mode,
            f"{float(acc):.6f}",
            "=NA()",           # macro_f1 missing
            "=NA()",           # pred_path not applicable
            timestamp,
        ])

    append_rows(TSV_PATH, formatted_rows)
    print(f"✅ Appended {len(formatted_rows)} GPT-4o rows to {TSV_PATH}")


if __name__ == "__main__":
    main()
