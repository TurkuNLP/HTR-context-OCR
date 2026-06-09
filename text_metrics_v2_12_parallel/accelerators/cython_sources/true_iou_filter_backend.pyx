# cython: language_level=3
"""Exact-result Cython backend for the true-IoU filter helper hot paths.

This backend accelerates two narrow helper boundaries that the Python reference
implementation already isolates:

1. sampling one geometric line onto score-matrix columns,
2. selecting per-column local winners while merging one overlap component.

The returned dictionaries, list ordering, rounding behavior, and tie-breaking
must remain identical to the Python reference path.
"""

from libc.math cimport fabs
import numpy as np
cimport numpy as cnp

cnp.import_array()


cdef inline double _line_y_at_x_exact(object line, double prediction_column):
    """Return the preserved line interpolation result at one prediction column."""
    cdef double x0 = float(line["x0"])
    cdef double y0 = float(line["y0"])
    cdef double x1 = float(line["x1"])
    cdef double y1 = float(line["y1"])
    cdef double dx = x1 - x0
    cdef double t

    if fabs(dx) < 1e-8:
        return y0

    t = (prediction_column - x0) / dx
    return y0 + t * (y1 - y0)


cpdef object sample_line_path_exact(object line, object matrix):
    """Return the exact sampled line path used by the production true-IoU filter."""
    cdef int n_reference_rows
    cdef int n_prediction_columns
    cdef double x0 = float(line["x0"])
    cdef double x1 = float(line["x1"])
    cdef int x_min
    cdef int x_max
    cdef int prediction_column
    cdef double line_y_value
    cdef int sampled_reference_row
    cdef dict sampled_x_to_y = {}
    cdef dict sampled_x_to_score = {}

    if matrix.size == 0:
        return None

    n_reference_rows = int(matrix.shape[0])
    n_prediction_columns = int(matrix.shape[1])
    x_min = max(0, int(np.floor(min(x0, x1))))
    x_max = min(n_prediction_columns - 1, int(np.ceil(max(x0, x1))))

    if x_max < x_min:
        return None

    for prediction_column in range(x_min, x_max + 1):
        line_y_value = _line_y_at_x_exact(line, prediction_column)
        sampled_reference_row = int(round(line_y_value))
        if sampled_reference_row < 0:
            sampled_reference_row = 0
        elif sampled_reference_row >= n_reference_rows:
            sampled_reference_row = n_reference_rows - 1
        sampled_x_to_y[int(prediction_column)] = int(sampled_reference_row)
        sampled_x_to_score[int(prediction_column)] = float(matrix[sampled_reference_row, prediction_column])

    if not sampled_x_to_y:
        return None

    return sampled_x_to_y, sampled_x_to_score


cpdef object select_component_local_winners_exact(object component_coverages):
    """Return exact local-winner dictionaries for one overlap component."""
    cdef dict best_path_sample_by_column = {}
    cdef object coverage
    cdef object prediction_column
    cdef object sampled_reference_row
    cdef tuple local_winner_key
    cdef object current_best_sample
    cdef dict merged_x_to_y
    cdef dict merged_x_to_score
    cdef list merged_source_raw_line_ids

    for coverage in component_coverages:
        for prediction_column, sampled_reference_row in coverage["x_to_y"].items():
            local_winner_key = (
                float(coverage["x_to_score"][prediction_column]),
                float(coverage.get("total_score", 0.0)),
                float(coverage.get("mean_score", 0.0)),
                -float(int(sampled_reference_row)),
            )
            current_best_sample = best_path_sample_by_column.get(int(prediction_column))
            if current_best_sample is None or local_winner_key > current_best_sample[0]:
                best_path_sample_by_column[int(prediction_column)] = (
                    local_winner_key,
                    int(sampled_reference_row),
                    float(coverage["x_to_score"][prediction_column]),
                )

    merged_x_to_y = {
        int(prediction_column): int(best_path_sample_by_column[int(prediction_column)][1])
        for prediction_column in sorted(best_path_sample_by_column)
    }
    merged_x_to_score = {
        int(prediction_column): float(best_path_sample_by_column[int(prediction_column)][2])
        for prediction_column in sorted(best_path_sample_by_column)
    }
    merged_source_raw_line_ids = sorted(
        {
            int(raw_line_id)
            for coverage in component_coverages
            for raw_line_id in coverage.get("source_raw_line_ids", [])
            if int(raw_line_id) >= 0
        }
    )

    return merged_x_to_y, merged_x_to_score, merged_source_raw_line_ids
