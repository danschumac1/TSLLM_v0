
import os, sys, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from global_utils.logging_utils import MasterLogger 

from dtaidistance import dtw_ndim
from tqdm import tqdm
import numpy as np

def _as_list_nd(A: np.ndarray):
    """
    Convert (N, V, T) -> list of (T, V) float arrays for dtw_ndim.
    """
    if A.ndim != 3:
        raise ValueError(f"Expected (N, V, T); got {A.shape}")
    # transpose per-sample from (V, T) to (T, V)
    return [a.transpose(1, 0).astype(float, copy=False) for a in A]

def pairwise_dtw_train_train(A: np.ndarray, window: int, logger=None, verbose=1) -> np.ndarray:
    """
    Train * Train distances using exact multivariate DTW.
    A: (N, V, T) -> returns (N, N)
    """
    if A.ndim != 3:
        raise ValueError(f"pairwise_dtw_train_train expects (N, V, T); got {A.shape}")
    lst = _as_list_nd(A)   # each (T, V)
    n = len(lst)
    if logger and verbose:
        print(f"[DTW] Computing train*train distance matrix for N={n}, window={window}")
    t0 = time.perf_counter()
    # Note: dtw_ndim.distance_matrix_fast has internal parallelization
    D = dtw_ndim.distance_matrix_fast(lst, parallel=True, window=window)
    dt = time.perf_counter() - t0
    if logger and verbose:
        print(f"[DTW] Finished train*train in {dt:.2f}s; shape={np.asarray(D).shape}")
    return D

def pairwise_dtw_val_train(A: np.ndarray, B: np.ndarray, window: int, logger=None, verbose=1) -> np.ndarray:
    """
    Validation/Test * Train distances for multivariate DTW.
    A: (N_va, V, T), B: (N_tr, V, T) -> (N_va, N_tr)
    """
    if A.ndim != 3 or B.ndim != 3:
        raise ValueError(f"pairwise_dtw_val_train expects (N, V, T); got {A.shape} and {B.shape}")
    Al = _as_list_nd(A)    # (T, V)
    Bl = _as_list_nd(B)    # (T, V)
    D = np.empty((len(Al), len(Bl)), dtype=float)
    if logger and verbose:
        print(f"[DTW] Computing val/test*train distances: {len(Al)}*{len(Bl)} with window={window}")

    # Explicit loop with tqdm and per-row timing
    for i, a in enumerate(tqdm(Al, desc="DTW val*train", ncols=100)):
        trow = time.perf_counter()
        for j, b in enumerate(Bl):
            D[i, j] = dtw_ndim.distance_fast(a, b, window=window)
        dtrow = time.perf_counter() - trow
        if (logger and verbose >= 2):
            print(f"[DTW] Row {i+1}/{len(Al)} computed in {dtrow:.2f}s; min={D[i].min():.4f} max={D[i].max():.4f} mean={D[i].mean():.4f}")
    return D

