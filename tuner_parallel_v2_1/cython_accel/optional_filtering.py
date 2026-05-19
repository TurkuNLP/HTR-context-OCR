from __future__ import annotations

"""Optional Cython acceleration for filtering helper loops.

The ownership filter is behavior-sensitive, so this module accelerates only
small helper boundaries that have straightforward Python-equivalent semantics:

- exact set IoU
- coverage-index construction by prediction column

If the compiled extension is absent, the Python fallback functions are used.
"""

try:
    from .filter_core import (
        build_coverage_indices_by_prediction_column as _cython_build_coverage_indices,
        set_iou as _cython_set_iou,
    )
except Exception:
    _cython_build_coverage_indices = None
    _cython_set_iou = None


def cython_filtering_helpers_available() -> bool:
    """Return ``True`` when filtering helper extensions are importable."""
    return _cython_build_coverage_indices is not None and _cython_set_iou is not None


def set_iou(values_a: set[int], values_b: set[int]) -> float:
    """Return exact set IoU with the reference empty-union behavior."""
    if _cython_set_iou is not None:
        return float(_cython_set_iou(values_a, values_b))

    union_values = values_a | values_b
    if not union_values:
        return 0.0
    return float(len(values_a & values_b) / len(union_values))


def build_coverage_indices_by_prediction_column(
    coverages: list[dict],
    n_prediction_columns: int,
) -> list[list[int]]:
    """Index coverages by prediction column with optional Cython acceleration."""
    if _cython_build_coverage_indices is not None:
        return _cython_build_coverage_indices(coverages, int(n_prediction_columns))

    coverage_indices_by_prediction_column: list[list[int]] = [
        [] for _ in range(int(n_prediction_columns))
    ]

    for coverage_index, coverage in enumerate(coverages):
        for prediction_column in coverage["x_to_y"]:
            if 0 <= int(prediction_column) < int(n_prediction_columns):
                coverage_indices_by_prediction_column[int(prediction_column)].append(int(coverage_index))

    return coverage_indices_by_prediction_column


__all__ = [
    "build_coverage_indices_by_prediction_column",
    "cython_filtering_helpers_available",
    "set_iou",
]
