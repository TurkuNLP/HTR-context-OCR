from __future__ import annotations

"""Optional Cython acceleration for the column-ownership scan."""

from typing import Any

import numpy as np

try:
    from .ownership_core import assign_columns_to_candidate_lines as _cython_assign_columns
except Exception:
    _cython_assign_columns = None


def cython_ownership_available() -> bool:
    """Return True when the compiled ownership helper can be imported."""
    return _cython_assign_columns is not None


def assign_columns_to_candidate_lines_with_optional_accelerator(
    *,
    score_matrix: np.ndarray,
    voter_mask: np.ndarray,
    candidate_lines: list[dict[str, Any]],
) -> dict[str, np.ndarray] | None:
    """Return compiled ownership arrays, or None when the helper is unavailable."""
    if _cython_assign_columns is None:
        return None
    if not candidate_lines:
        return None

    try:
        matrix = np.ascontiguousarray(score_matrix, dtype=np.float64)
        mask_uint8 = np.ascontiguousarray(voter_mask, dtype=np.uint8)
        candidate_x0 = np.ascontiguousarray([float(line["x0"]) for line in candidate_lines], dtype=np.float64)
        candidate_y0 = np.ascontiguousarray([float(line["y0"]) for line in candidate_lines], dtype=np.float64)
        candidate_x1 = np.ascontiguousarray([float(line["x1"]) for line in candidate_lines], dtype=np.float64)
        candidate_y1 = np.ascontiguousarray([float(line["y1"]) for line in candidate_lines], dtype=np.float64)
        result = _cython_assign_columns(
            matrix,
            mask_uint8,
            candidate_x0,
            candidate_y0,
            candidate_x1,
            candidate_y1,
        )
    except (TypeError, ValueError, KeyError, IndexError):
        return None

    if result is None:
        return None
    return {
        "mapped_y": np.asarray(result["mapped_y"], dtype=float),
        "mapped_candidate_id": np.asarray(result["mapped_candidate_id"], dtype=int),
        "owned_counts": np.asarray(result["owned_counts"], dtype=int),
    }


__all__ = [
    "assign_columns_to_candidate_lines_with_optional_accelerator",
    "cython_ownership_available",
]
