import numpy as np

def z_norm_series(A: np.ndarray) -> np.ndarray:
    """
    Per-series z-normalization on 3D arrays shaped (N, V, T).
    Mean/std are computed along the time axis only.
    """
    A = np.asarray(A, dtype=float)
    if A.ndim != 3:
        raise ValueError(f"Expected (N, V, T); got {A.shape}")
    mu = A.mean(axis=-1, keepdims=True)            # (N, V, 1)
    sd = A.std(axis=-1, keepdims=True) + 1e-8      # (N, V, 1)
    return (A - mu) / sd
