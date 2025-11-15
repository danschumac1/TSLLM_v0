from __future__ import annotations
import warnings
import json, textwrap
import csv
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import os
import numpy as np

from global_utils.logging_utils import MasterLogger
from global_utils.prompter import QAs
from Classification.src.utils.build_questions import (
    LABEL_MAPPING, _letters, _sort_key_for_label_id
)

# SIMPLE HELPERS -----------------------------------------------------------------------------------
def load_jsonl(file_path):
    """Load a JSON Lines file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def yield_jsonl(file_path, max_rows=None):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_rows is not None and i >= max_rows:
                break
            data.append(json.loads(line))
    return data

def save_jsonl(data, file_path):
    """Save data to a JSON Lines file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

def append_jsonl(output_path: str, data: dict):
    """
    Append a dictionary to the specified output JSONL file.
    Creates parent directories if needed.
    """
    output_file = output_path
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'a') as f:
        json.dump(data, f)
        f.write('\n')
    
    logger = MasterLogger.get_instance()
    # logger.info(f"Output saved to {output_file}")

def load_json(file_path):
    """Load a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_json(data, file_path):
    """Save data to a JSON file."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)

def load_csv(file_path):
    """Load a CSV file."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data

def save_csv(data, file_path):
    """Save data to a CSV file."""
    if not data:
        return
    with open(file_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=data[0].keys())
        writer.writeheader()
        writer.writerows(data)

# WRITING RESULTS ----------------------------------------------------------------------------------
def ensure_header(results_path: str, header_cols: List[str]):
    # create parent dir
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    write_header = (not os.path.exists(results_path)) or (os.path.getsize(results_path) == 0)
    if write_header:
        with open(results_path, "w", encoding="utf-8") as f:
            f.write("\t".join(header_cols) + "\n")

def append_row(results_path: str, row_vals: List[str]):
    with open(results_path, "a", encoding="utf-8") as f:
        f.write("\t".join(map(str, row_vals)) + "\n")
        
def append_rows(path, rows):
    """Append new rows to the TSV file."""
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write("\t".join(map(str, r)) + "\n")

# DATA LOADERS -------------------------------------------------------------------------------------
def _letters(n: int) -> str:
    # 1-indexed: 1->A
    s = ""
    while n > 0:
        n, r = divmod(n - 1, 26)
        s = chr(65 + r) + s
    return s

@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    idx: np.ndarray
    shot_idxs: Optional[np.ndarray] = None
    fixed_shot_idxs: Optional[np.ndarray] = None
    label_maps: Optional[Dict] = None
    general_question: Optional[str] = None
    dataset: Optional[str] = None

    @property
    def unique_classes(self) -> np.ndarray:
        return np.unique(self.y)

    @property
    def n_classes(self) -> int:
        return int(self.unique_classes.size)

    def __len__(self) -> int:
        return int(self.X.shape[0])

    def class_dist(self) -> Dict[int, int]:
        u, c = np.unique(self.y, return_counts=True)
        return {int(k): int(v) for k, v in zip(u, c)}

    def __repr__(self) -> str:
        """
        Detailed, structured summary of the Split object with
        pretty label_maps and wrapped general_question.
        """

        # ---------- small helpers ----------
        def _indent(s: str, n: int = 4) -> str:
            pad = " " * n
            return "\n".join(pad + line if line else pad for line in s.splitlines())

        def _fmt_dict(d: Dict, max_items=10) -> str:
            items = list(d.items())
            if len(items) <= max_items:
                return str(d)
            head = ", ".join(f"{k}: {v}" for k, v in items[:max_items])
            return f"{{{head}, ... ({len(items)} total)}}"

        def _wrap_block(s: str, width: int = 88, max_lines: int = 12) -> str:
            s = (s or "").strip()
            if not s:
                return "none"
            wrapped = textwrap.fill(s, width=width)
            lines = wrapped.splitlines()
            if len(lines) > max_lines:
                lines = lines[:max_lines] + ["..."]
            return "\n" + _indent("\n".join(lines), 4)

        def _label_maps_table(maps: Dict, max_rows: int = 20) -> str:
            if not isinstance(maps, dict):
                return "none"

            id2name = maps.get("id_to_name")
            id2letter = maps.get("id_to_letter")
            # If expected keys missing, pretty-print JSON
            if not isinstance(id2name, dict) or not isinstance(id2letter, dict):
                pretty = json.dumps(maps, indent=4, sort_keys=True)
                return "\n" + _indent(pretty, 4)

            # Normalize keys to ints
            try:
                id2name_i = {int(k): str(v) for k, v in id2name.items()}
                id2letter_i = {int(k): str(v) for k, v in id2letter.items()}
            except Exception:
                pretty = json.dumps(maps, indent=4, sort_keys=True)
                return "\n" + _indent(pretty, 4)

            rows = []
            for k in sorted(id2name_i.keys()):
                rows.append((k, id2letter_i.get(k, "?"), id2name_i.get(k, str(k))))

            total = len(rows)
            if total == 0:
                return "none"

            if total > max_rows:
                head = rows[:max_rows]
                tail_note = f"... ({total - max_rows} more)"
            else:
                head = rows
                tail_note = ""

            # Compute column widths
            col1 = "id"
            col2 = "letter"
            col3 = "name"
            w1 = max(len(col1), max(len(str(r[0])) for r in head))
            w2 = max(len(col2), max(len(str(r[1])) for r in head))
            w3 = max(len(col3), max(len(str(r[2])) for r in head))

            # Build table
            header = f"{col1:<{w1}} | {col2:<{w2}} | {col3:<{w3}}"
            sep = "-" * len(header)
            body_lines = [f"{r[0]:<{w1}} | {r[1]:<{w2}} | {r[2]:<{w3}}" for r in head]
            if tail_note:
                body_lines.append(tail_note)

            table = "\n".join([header, sep, *body_lines])
            return "\n" + _indent(table, 4)

        # ---------- core fields ----------
        N = len(self)
        shape = "×".join(map(str, self.X.shape))
        classes = self.unique_classes
        cls_dist = self.class_dist()
        dist_str = _fmt_dict(cls_dist, max_items=10)
        shot_shape = None if self.shot_idxs is None else "×".join(map(str, self.shot_idxs.shape))

        # ---------- pretty blocks ----------
        label_maps_block = _label_maps_table(self.label_maps, max_rows=20)
        gq_block = _wrap_block(self.general_question, width=88, max_lines=12)

        # ---------- final string ----------
        lines = []
        lines.append("Split(")
        lines.append(f"  dataset          = {self.dataset!r},")
        lines.append(f"  N                = {N},")
        lines.append(f"  X.shape          = ({shape}),")
        lines.append(f"  y.shape          = {tuple(self.y.shape)},")
        lines.append(f"  n_classes        = {self.n_classes},")
        lines.append(f"  classes          = {classes.tolist()},")
        lines.append(f"  class_dist       = {dist_str},")
        lines.append(f"  shot_idxs        = {shot_shape},")
        lines.append(f"  label_maps       ={label_maps_block},")
        lines.append(f"  general_question ={gq_block},")
        lines.append(")")
        return "\n".join(lines)
    
    def _take(self, idxs: np.ndarray) -> "Split":
        """Return a new Split containing rows at idxs (0-based)."""
        idxs = np.asarray(idxs, dtype=int)
        new_X = self.X[idxs]
        new_y = self.y[idxs]
        new_idx = self.idx[idxs] if self.idx is not None else idxs.copy()

        # If shot indices exist, keep only those that land in this subset and remap to the new 0..len-1
        if self.shot_idxs is not None:
            mask = np.isin(self.shot_idxs, idxs)
            kept = self.shot_idxs[mask]
            if kept.size:
                remap = {int(old): i for i, old in enumerate(idxs.tolist())}
                new_shot = np.array([remap[int(s)] for s in kept if int(s) in remap], dtype=int)
            else:
                new_shot = np.empty((0,), dtype=int)
        else:
            new_shot = None

        return Split(
            X=new_X,
            y=new_y,
            idx=new_idx,
            shot_idxs=new_shot,
            label_maps=self.label_maps,
            general_question=self.general_question,
            dataset=self.dataset,
        )

    def __getitem__(self, key: Union[int, slice, Sequence[int], np.ndarray]) -> "Split":
        """
        Support int, slice, list/ndarray of indices.
        Always returns a Split (so batch ops are uniform).
        """
        if isinstance(key, slice):
            idxs = np.arange(len(self))[key]
        elif isinstance(key, (list, tuple, np.ndarray)):
            idxs = np.asarray(key, dtype=int)
        else:  # int / np.integer
            idxs = np.asarray([int(key)], dtype=int)
        return self._take(idxs)

    
    def qas_from_indices(
        self,
        neighbor_indices: List[int],
        *,
        n_shots: Optional[int] = None,
        general_question: Optional[str] = None,
        include_ts: bool = True
    ):
        """
        Convert neighbor indices into few-shot QAs using only self.label_maps.
        Expects label_maps.json to contain:
          - "id_to_name": { "<int>": "<label name>", ... }
          - optionally "id_to_letter": { "<int>": "A", ... }  (else synthesized)
        Caps to at most 10 shots.
        """
        # local import to avoid circulars if needed
        from global_utils.prompter import QAs

        if n_shots is None:
            n_shots = 10
        n_shots = max(0, min(int(n_shots), 10))

        q_text = (general_question or self.general_question or "").strip()

        if not isinstance(self.label_maps, dict):
            warnings.warn("[qas_from_indices] label_maps is missing; returning empty shots.")
            return []

        # Parse maps (string keys -> ints)
        id_to_name_raw = self.label_maps.get("id_to_name", {})
        if not id_to_name_raw:
            warnings.warn("[qas_from_indices] 'id_to_name' missing in label_maps; returning empty shots.")
            return []
        id_to_name: Dict[int, str] = {int(k): str(v) for k, v in id_to_name_raw.items()}

        id_to_letter_raw = self.label_maps.get("id_to_letter", {})
        if id_to_letter_raw:
            id_to_letter: Dict[int, str] = {int(k): str(v) for k, v in id_to_letter_raw.items()}
        else:
            # Synthesize letters by sorted class ids
            sorted_cls = sorted(id_to_name.keys())
            id_to_letter = {cls_id: _letters(i + 1) for i, cls_id in enumerate(sorted_cls)}

        # Build QAs
        qas_list: List[QAs] = []
        y_arr = np.asarray(self.y).ravel()
        n_train = len(y_arr)

        for j in list(neighbor_indices)[:n_shots]:
            j = int(j)
            if j < 0 or j >= n_train:
                continue
            cls_id = int(y_arr[j])
            label_name = id_to_name.get(cls_id, str(cls_id))
            letter = id_to_letter.get(cls_id, "A")
            ts = self.X[j]
            ts_json = json.dumps(np.asarray(ts).tolist())
            if include_ts:
                qas_list.append(
                    QAs(
                        question={
                            "question": q_text, 
                            "timeseries": ts_json},
                        answer=f"The answer is: {letter} | {label_name}",
                    )
                )
            else:
                qas_list.append(
                    QAs(
                        question={"question": q_text},
                        answer=f"The answer is: {letter} | {label_name}",
                    )
                )
        return qas_list



# ---- Low-level split loader ----
def _load_split(root: str, split: str, mmap: bool = False, normalize: bool = False) -> Split:
    x_path = os.path.join(root, f"X_{split}.npy")
    y_path = os.path.join(root, f"y_{split}.npy")
    fixed_shots_path = os.path.join(root, "fixed_shot_indices.npy")
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Missing features for split={split} in {root}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Missing labels for split={split} in {root}")
    
    if os.path.exists(fixed_shots_path):
        fixed_shot_idxs = np.load(fixed_shots_path) # deleted mmap_mode=mmap_mode and works now?
        if fixed_shot_idxs.dtype != np.int64:
            fixed_shot_idxs = fixed_shot_idxs.astype(np.int64, copy=False)
    else:
        fixed_shot_idxs = None

    mmap_mode = "r" if mmap else None
    X = np.load(x_path, mmap_mode=mmap_mode)
    y = np.load(y_path, mmap_mode=mmap_mode)

    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)
    if y.dtype != np.int64:
        y = y.astype(np.int64, copy=False)
    if y.ndim != 1:
        raise ValueError(f"y must be 1D; got shape {y.shape} for split={split}")

    if normalize:

        X_mean = np.mean(X, axis=-1, keepdims=True)
        X_std = np.std(X, axis=-1, keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        for x in X:
            if np.all(x == 0):
                raise ValueError(f"[NORMALIZE] Warning: found all-zero time series in split={split} after normalization.")

        
    idx = np.arange(len(X), dtype=np.int64)
    return Split(X=X, y=y, idx=idx, fixed_shot_idxs=fixed_shot_idxs)


# ---- Artifact loader (tolerant) ----
def _load_artifacts(artifact_root: str) -> Tuple[Optional[np.ndarray], Optional[Dict], Optional[str]]:
    """
    Load common artifacts from <artifact_root> if present:
      - top10_similar.npy          -> shot indices (N_test, 10)
      - label_maps.json            -> maps dict
      - general_question.txt       -> question string
    Returns: (shot_indices, label_maps, general_question); any may be None.
    """
    shot_indices: Optional[np.ndarray] = None
    label_maps: Optional[Dict] = None
    general_question: Optional[str] = None

    # shots
    shot_path = os.path.join(artifact_root, "top10_similar.npy")
    if os.path.isfile(shot_path):
        arr = np.load(shot_path)
        if arr.ndim != 2:
            warnings.warn(f"[ARTIFACT] Expected 2D shot indices; got shape {arr.shape} at {shot_path}")
        else:
            # clip to at most 10 columns
            if arr.shape[1] > 10:
                arr = arr[:, :10]
            shot_indices = arr.astype(np.int32, copy=False)
    else:
        warnings.warn(f"[ARTIFACT] Missing top10_similar.npy at {shot_path}; shot indices will be None.")

    # maps
    maps_path = os.path.join(artifact_root, "label_maps.json")
    if os.path.isfile(maps_path):
        with open(maps_path, "r", encoding="utf-8") as f:
            try:
                label_maps = json.load(f)
                # Optional sanity check
                for key in ("letter_to_id", "id_to_letter"):
                    if key not in label_maps:
                        warnings.warn(f"[ARTIFACT] '{key}' missing in label_maps.json at {maps_path}")
            except json.JSONDecodeError as e:
                warnings.warn(f"[ARTIFACT] Could not parse label_maps.json: {e}")
                label_maps = None
    else:
        warnings.warn(f"[ARTIFACT] Missing label_maps.json at {maps_path}")

    # question
    q_path = os.path.join(artifact_root, "general_question.txt")
    if os.path.isfile(q_path):
        with open(q_path, "r", encoding="utf-8") as f:
            general_question = f.read().strip()
    else:
        warnings.warn(f"[ARTIFACT] Missing general_question.txt at {q_path}")

    return shot_indices, label_maps, general_question


# ---- Public API ----
def load_train_test(
        input_folder: str, 
        mmap: bool = False, 
        attach_artifacts: bool = True,
        normalize: bool = False) -> Tuple[Split, Split]:
    """
    Load train/test splits from <root>.
    Expects: X_train.npy, y_train.npy, X_test.npy, y_test.npy

    If attach_artifacts=True:
      - Resolve artifact root:
          * if `artifact_root` is provided, use it
          * else use `root` (same directory as the splits)
      - Load artifacts and attach:
          * test.shot_idxs           (np.ndarray | None)
          * {train,test}.label_maps  (Dict | None)
          * {train,test}.general_question (str | None)
    """

    train = _load_split(input_folder, "train", mmap=mmap, normalize=normalize)
    test  = _load_split(input_folder, "test",  mmap=mmap, normalize=normalize)

    if attach_artifacts:
        shots, maps, q = _load_artifacts(input_folder)
        test.shot_idxs = shots
        train.label_maps = maps
        test.label_maps = maps
        train.general_question = q
        test.general_question = q

        # Optional consistency check
        if shots is not None and shots.shape[0] != len(test):
            warnings.warn(
                f"[ARTIFACT] shot_idxs rows ({shots.shape[0]}) != N_test ({len(test)}). "
                "Make sure artifacts correspond to this split."
            )

    return train, test