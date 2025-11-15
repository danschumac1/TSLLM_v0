'''
2025-09-29
Author: Dan Schumacher
How to run:
   python ./src/utils/baseline_utils.py
'''
import json
import re

import sys; sys.path.append("./src")
import numpy as np
from typing import Dict, List

def corr_sim(x, y):
    """Pearson-based similarity in [0,1], safe for constants/NaNs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n == 0:
        return 0.0
    x = x[:n]; y = y[:n]
    # If either std is ~0, correlation is undefined; treat as no linear relation
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.5  # neutral similarity; change if you prefer 0.0
    rho = np.corrcoef(x, y)[0, 1]
    if np.isnan(rho):
        return 0.5
    return float((rho + 1) / 2)

def select_most_similar(current_row: Dict, candidates: List[Dict], n_examples: int = 3) -> List[Dict]:
    """
    Find the top-n candidates most similar to current_row based on mean per-dimension Pearson similarity.
    Expects structure:
      current_row["subset"] : hashable (optional)
      current_row["ts"]     : dict[dim_name] -> 1D array-like
      candidate["ts"]       : dict[dim_name] -> 1D array-like
    """
    # Optional subset filter
    if "subset" in current_row and current_row["subset"] is not None:
        # this means that 
        candidates = [row for row in candidates if row.get("subset") == current_row["subset"]]

    if not candidates:
        return []

    dims = list(current_row["ts"].keys())

    scored = []
    for i, cand in enumerate(candidates):
        # Compute per-dimension similarity; skip dims missing in candidate
        sims = []
        for dim in dims:
            if "ts" not in cand or dim not in cand["ts"]:
                continue
            sims.append(corr_sim(current_row["ts"][dim], cand["ts"][dim]))
        if not sims:
            mean_sim = 0.0
        else:
            mean_sim = float(np.mean(sims))
        scored.append((i, mean_sim))

    # Sort by descending similarity
    scored.sort(key=lambda kv: kv[1], reverse=True)

    # Take top-k indices (tie-breaker is stable: earlier candidates win ties)
    top_idxs = [idx for idx, _ in scored[:n_examples]]

    # Return the actual candidate rows in that order
    return [candidates[idx] for idx in top_idxs]

if __name__ == "__main__":
    # JUST TO TEST TO MAKE SURE THINGS WORK.
    from utils.file_io import load_jsonl

    data = load_jsonl("data/datasets/TimerBed/_TINYTEST/test.jsonl")
    candidates = load_jsonl("data/datasets/TimerBed/_TINYTEST/train.jsonl")

    current_row = data[0]

    examples = select_most_similar(current_row, candidates, 3)
    print(len(examples))

_LETTER_RE = re.compile(r"The answer is:\s*([A-Z])\b")

def extract_letter(answer_text: str) -> str:
    if not isinstance(answer_text, str):
        answer_text = str(answer_text)
    m = _LETTER_RE.search(answer_text)
    if m:
        return m.group(1)
    for ch in answer_text:
        if "A" <= ch <= "Z":
            return ch
    return ""

def to_text(result) -> str:
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for k in ("answer", "output", "text", "prediction"):
            if k in result:
                v = result[k]
                return v if isinstance(v, str) else json.dumps(v)
        return json.dumps(result)
    try:
        return str(result)
    except Exception:
        return repr(result)
