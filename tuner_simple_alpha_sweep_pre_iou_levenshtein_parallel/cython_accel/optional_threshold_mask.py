from __future__ import annotations

"""Optional Cython acceleration for fixed-threshold score masks."""

import numpy as np

try:
    from .threshold_mask_core import threshold_mask_at_or_above as _cython_threshold_mask_at_or_above
except Exception:
    _cython_threshold_mask_at_or_above = None


def cython_threshold_mask_available() -> bool:
    """Return True when the compiled threshold-mask helper is importable."""
    return _cython_threshold_mask_at_or_above is not None


def threshold_mask_at_or_above(score_matrix: np.ndarray, threshold: float) -> np.ndarray | None:
    """Return a compiled threshold mask, or None when the extension is unavailable."""
    if _cython_threshold_mask_at_or_above is None:
        return None
    try:
        matrix = np.ascontiguousarray(score_matrix, dtype=np.float64)
        return np.asarray(_cython_threshold_mask_at_or_above(matrix, float(threshold)), dtype=bool)
    except (TypeError, ValueError):
        return None


__all__ = ["cython_threshold_mask_available", "threshold_mask_at_or_above"]
