from __future__ import annotations

"""Geometry and line-fitting helpers for true-IoU filtering.

This module owns the small mathematical operations used by the filter:

- clipping a line to valid matrix columns;
- sampling mean support for a line;
- expanding a sampled path into covered reference rows;
- fitting one representative straight line through a merged path.

The functions here do not decide which candidate lines survive.  They only
provide deterministic geometry primitives used by the candidate, merge, and
final-assignment modules.
"""

import math

import numpy as np

from . import filter_cython_accelerators as accelerators

try:
    from ..runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

# Direct script execution can place only ``filtering/`` on sys.path.  The shared
# runtime helper adds the tuner root and project helper roots before we import
# geometry functions from sibling packages.
ensure_tuner_runtime_paths()

try:
    from ..alignment.hough_segment_endpoint_records import (
        line_record_length as line_length,
        line_y_at_prediction_column as line_y_at_x,
        mean_line_support_from_score_matrix as mean_line_support,
    )
except ImportError:
    from alignment.hough_segment_endpoint_records import (  # type: ignore
        line_record_length as line_length,
        line_y_at_prediction_column as line_y_at_x,
        mean_line_support_from_score_matrix as mean_line_support,
    )


def line_x_bounds(line: dict, n_pred: int) -> tuple[int, int] | None:
    """Return inclusive prediction-column bounds for one line inside the matrix."""
    if int(n_pred) <= 0:
        return None

    x_min = max(0, int(math.floor(min(line["x0"], line["x1"]))))
    x_max = min(int(n_pred) - 1, int(math.ceil(max(line["x0"], line["x1"]))))
    if x_max < x_min:
        return None
    return x_min, x_max


def mean_line_support_for_filter(matrix: np.ndarray, line: dict) -> float:
    """Return the mean score-matrix support sampled along one line.

    The optional Cython helper implements the same sampling rule as
    ``mean_line_support``.  If the compiled helper is unavailable or cannot
    handle the provided matrix, the readable Python reference path is used.
    """
    if accelerators.accelerated_mean_line_support_from_endpoints is not None:
        accelerated_support = accelerators.accelerated_mean_line_support_from_endpoints(
            matrix,
            x0=float(line["x0"]),
            y0=float(line["y0"]),
            x1=float(line["x1"]),
            y1=float(line["y1"]),
        )
        if accelerated_support is not None:
            return float(accelerated_support)
    return float(mean_line_support(matrix, line))


def ref_segments_from_path(x_to_y: dict[int, int]) -> set[int]:
    """Expand a sampled x-to-y path into all reference rows it covers.

    Adjacent sampled columns can jump by more than one reference row.  The filter
    fills those vertical gaps so the true-IoU overlap check measures the full
    band crossed by the line, not only the exact integer samples.
    """
    if not x_to_y:
        return set()

    covered_reference_rows: set[int] = set()
    previous_row: int | None = None

    for prediction_column in sorted(x_to_y):
        current_row = int(x_to_y[prediction_column])
        covered_reference_rows.add(current_row)
        if previous_row is not None:
            row_start, row_end = sorted((previous_row, current_row))
            for reference_row in range(row_start, row_end + 1):
                covered_reference_rows.add(int(reference_row))
        previous_row = current_row

    return covered_reference_rows


def weighted_degree_one_fit(
    sampled_prediction_columns: np.ndarray,
    sampled_reference_rows: np.ndarray,
    sampled_weights: np.ndarray,
) -> tuple[float, float] | None:
    """Return the weighted straight-line ``(slope, intercept)`` fit.

    This is the exact degree-1 form of the historical
    ``np.polyfit(x, y, deg=1, w=weights)`` rule.  NumPy applies ``weights`` to
    the residual before squaring, so the equivalent normal-equation weights are
    ``weights ** 2``.
    """
    squared_weights = np.square(np.asarray(sampled_weights, dtype=float))
    total_weight = float(np.sum(squared_weights))
    if total_weight <= 0.0 or not math.isfinite(total_weight):
        return None

    x_values = np.asarray(sampled_prediction_columns, dtype=float)
    y_values = np.asarray(sampled_reference_rows, dtype=float)
    weighted_x_mean = float(np.dot(squared_weights, x_values) / total_weight)
    weighted_y_mean = float(np.dot(squared_weights, y_values) / total_weight)

    centered_x_values = x_values - weighted_x_mean
    centered_y_values = y_values - weighted_y_mean
    weighted_x_variance = float(np.dot(squared_weights, centered_x_values * centered_x_values))
    if weighted_x_variance <= 0.0 or not math.isfinite(weighted_x_variance):
        return None

    fitted_slope = float(
        np.dot(squared_weights, centered_x_values * centered_y_values)
        / weighted_x_variance
    )
    fitted_intercept = float(weighted_y_mean - (fitted_slope * weighted_x_mean))
    return fitted_slope, fitted_intercept


def fit_line_from_path(
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    *,
    fallback_line: dict | None = None,
) -> dict:
    """Fit the representative straight segment used downstream.

    A merged coverage can be a jagged path assembled from multiple Hough
    candidates.  Downstream scoring and visualisation still need one clean line
    record with ``x0, y0, x1, y1`` endpoints.  Stronger local score-matrix cells
    get larger weights, so high-confidence samples influence the representative
    line more than weak samples.
    """
    if not x_to_y:
        fallback_geometry = {} if fallback_line is None else dict(fallback_line)
        fallback_geometry.setdefault("x0", 0.0)
        fallback_geometry.setdefault("y0", 0.0)
        fallback_geometry.setdefault("x1", 0.0)
        fallback_geometry.setdefault("y1", 0.0)
        fallback_geometry["length"] = line_length(fallback_geometry)
        fallback_geometry["support"] = (
            mean_line_support_for_filter(matrix, fallback_geometry)
            if matrix.size
            else 0.0
        )
        fallback_geometry["score"] = float(
            fallback_geometry.get("score", fallback_geometry.get("support", 0.0))
        )
        return fallback_geometry

    sampled_prediction_columns = np.asarray(sorted(x_to_y), dtype=float)
    sampled_reference_rows = np.asarray(
        [float(x_to_y[int(prediction_column)]) for prediction_column in sampled_prediction_columns],
        dtype=float,
    )
    sampled_weights = np.asarray(
        [max(float(x_to_score[int(prediction_column)]), 1e-6) for prediction_column in sampled_prediction_columns],
        dtype=float,
    )

    if len(sampled_prediction_columns) == 1 or np.allclose(
        sampled_prediction_columns,
        sampled_prediction_columns[0],
    ):
        fitted_x0 = fitted_x1 = float(sampled_prediction_columns[0])
        fitted_y0 = fitted_y1 = float(sampled_reference_rows[0])
    else:
        fit_result = weighted_degree_one_fit(
            sampled_prediction_columns,
            sampled_reference_rows,
            sampled_weights,
        )
        if fit_result is None:
            # Degenerate or non-finite input should not normally reach this
            # branch.  Keeping the first/last samples is the safest fallback
            # because it preserves a line spanning the sampled path.
            fitted_x0 = float(sampled_prediction_columns[0])
            fitted_x1 = float(sampled_prediction_columns[-1])
            fitted_y0 = float(sampled_reference_rows[0])
            fitted_y1 = float(sampled_reference_rows[-1])
        else:
            fitted_slope, fitted_intercept = fit_result
            fitted_x0 = float(sampled_prediction_columns.min())
            fitted_x1 = float(sampled_prediction_columns.max())
            fitted_y0 = float((fitted_slope * fitted_x0) + fitted_intercept)
            fitted_y1 = float((fitted_slope * fitted_x1) + fitted_intercept)

    representative_line = {} if fallback_line is None else dict(fallback_line)
    representative_line["x0"] = fitted_x0
    representative_line["y0"] = fitted_y0
    representative_line["x1"] = fitted_x1
    representative_line["y1"] = fitted_y1
    representative_line["length"] = line_length(representative_line)
    representative_line["support"] = (
        mean_line_support_for_filter(matrix, representative_line)
        if matrix.size
        else 0.0
    )
    representative_line["score"] = float(representative_line["support"])
    return representative_line


def set_iou(values_a: set[int], values_b: set[int]) -> float:
    """Return exact set IoU for the true-IoU overlap rule."""
    if accelerators.accelerated_set_iou is not None:
        return float(accelerators.accelerated_set_iou(values_a, values_b))

    union_values = values_a | values_b
    if not union_values:
        return 0.0
    return float(len(values_a & values_b) / len(union_values))


__all__ = [
    "fit_line_from_path",
    "line_length",
    "line_x_bounds",
    "line_y_at_x",
    "mean_line_support_for_filter",
    "ref_segments_from_path",
    "set_iou",
    "weighted_degree_one_fit",
]
