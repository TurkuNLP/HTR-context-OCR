from __future__ import annotations

"""Optional Cython acceleration for filtering helper loops.

The ownership filter is behavior-sensitive, so this module accelerates only
small helper boundaries that have straightforward Python-equivalent semantics:

- exact set IoU
- coverage-index construction by prediction column
- line support/path sampling
- final ownership assignment

If the compiled extension is absent, the Python fallback functions are used.
"""

try:
    from .filter_core import (
        build_coverage_indices_by_prediction_column as _cython_build_coverage_indices,
        compute_final_assignment_from_coverages as _cython_compute_final_assignment,
        mean_line_support_from_endpoints as _cython_mean_line_support_from_endpoints,
        sample_line_path as _cython_sample_line_path,
        set_iou as _cython_set_iou,
    )
except Exception:
    _cython_build_coverage_indices = None
    _cython_compute_final_assignment = None
    _cython_mean_line_support_from_endpoints = None
    _cython_sample_line_path = None
    _cython_set_iou = None


def cython_filtering_helpers_available() -> bool:
    """Return ``True`` when filtering helper extensions are importable."""
    return _cython_build_coverage_indices is not None and _cython_set_iou is not None


def cython_line_sampling_available() -> bool:
    """Return ``True`` when compiled line-sampling helpers are importable."""
    return (
        _cython_mean_line_support_from_endpoints is not None
        and _cython_sample_line_path is not None
    )


def cython_final_assignment_available() -> bool:
    """Return ``True`` when the compiled final-assignment helper is importable."""
    return _cython_compute_final_assignment is not None


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


def compute_final_assignment_from_coverages(
    *,
    coverages: list[dict],
    coverage_indices_by_prediction_column: list[list[int]],
    n_reference_rows: int,
    n_prediction_columns: int,
) -> dict | None:
    """Return Cython-computed assignment lists, or ``None`` when unavailable."""
    if _cython_compute_final_assignment is None:
        return None
    try:
        assignment = _cython_compute_final_assignment(
            coverages,
            coverage_indices_by_prediction_column,
            int(n_reference_rows),
            int(n_prediction_columns),
        )
    except (TypeError, ValueError, KeyError, IndexError):
        return None
    return None if assignment is None else dict(assignment)


def mean_line_support_from_endpoints(
    matrix,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> float | None:
    """Return Cython-sampled mean line support, or ``None`` when unavailable.

    The caller owns the Python fallback.  Returning ``None`` instead of raising
    keeps the tuner portable on environments where the extension was not built
    or where a matrix has an unsupported dtype/layout.
    """
    if _cython_mean_line_support_from_endpoints is None:
        return None
    try:
        return float(
            _cython_mean_line_support_from_endpoints(
                matrix,
                float(x0),
                float(y0),
                float(x1),
                float(y1),
            )
        )
    except (TypeError, ValueError):
        return None


def sample_line_path(
    matrix,
    *,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
) -> dict | None:
    """Return Cython-sampled path data, or ``None`` when unavailable.

    The returned object deliberately contains normal Python containers.  That
    keeps the rest of the filtering implementation simple and preserves the
    exact public output shape.
    """
    if _cython_sample_line_path is None:
        return None
    try:
        sampled_path = _cython_sample_line_path(
            matrix,
            float(x0),
            float(y0),
            float(x1),
            float(y1),
        )
    except (TypeError, ValueError):
        return None
    return None if sampled_path is None else dict(sampled_path)


__all__ = [
    "build_coverage_indices_by_prediction_column",
    "compute_final_assignment_from_coverages",
    "cython_final_assignment_available",
    "cython_filtering_helpers_available",
    "cython_line_sampling_available",
    "mean_line_support_from_endpoints",
    "sample_line_path",
    "set_iou",
]
