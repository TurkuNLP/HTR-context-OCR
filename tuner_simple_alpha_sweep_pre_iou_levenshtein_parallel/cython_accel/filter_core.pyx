# cython: language_level=3

"""Cython helpers for behavior-preserving filtering internals.

The functions in this file intentionally operate on the same Python dict/set
objects as the reference implementation.  That keeps semantics identical while
moving high-frequency loops into compiled code.
"""

import numpy as np
from libc.math cimport ceil, fabs, floor


cdef long _round_half_to_even(double value):
    """Return Python-compatible ``round(value)`` for a C double.

    Python uses bankers rounding: values exactly halfway between two integers
    round to the nearest even integer.  The filtering code samples matrix rows
    with ``round(...)``, so this helper intentionally preserves that rule
    instead of using the C library rounding mode.
    """
    cdef double floored_value = floor(value)
    cdef double fractional_part = value - floored_value
    cdef long lower_integer = <long>floored_value

    if fractional_part < 0.5:
        return lower_integer
    if fractional_part > 0.5:
        return lower_integer + 1
    if lower_integer % 2 == 0:
        return lower_integer
    return lower_integer + 1


cdef int _clip_int(long value, int minimum_value, int maximum_value):
    """Clamp one integer into an inclusive integer interval."""
    if value < minimum_value:
        return minimum_value
    if value > maximum_value:
        return maximum_value
    return <int>value


cdef int _line_x_min(double x0, double x1):
    """Return the unclamped inclusive start column for one line segment."""
    if x0 < x1:
        return <int>floor(x0)
    return <int>floor(x1)


cdef int _line_x_max(double x0, double x1):
    """Return the unclamped inclusive end column for one line segment."""
    if x0 > x1:
        return <int>ceil(x0)
    return <int>ceil(x1)


def set_iou(values_a, values_b):
    """Return exact set IoU with the same empty-union behavior as Python."""
    cdef object union_values = values_a | values_b
    if not union_values:
        return 0.0
    return float(len(values_a & values_b) / len(union_values))


def build_coverage_indices_by_prediction_column(coverages, int n_prediction_columns):
    """Index coverage ids by prediction column using the Python output shape."""
    cdef int column_count = n_prediction_columns
    cdef Py_ssize_t coverage_index
    cdef object coverage
    cdef object prediction_column
    cdef int prediction_column_int
    cdef list coverage_indices_by_prediction_column

    if column_count < 0:
        column_count = 0

    coverage_indices_by_prediction_column = [[] for _ in range(column_count)]

    for coverage_index, coverage in enumerate(coverages):
        for prediction_column in coverage["x_to_y"]:
            prediction_column_int = int(prediction_column)
            if 0 <= prediction_column_int < column_count:
                coverage_indices_by_prediction_column[prediction_column_int].append(int(coverage_index))

    return coverage_indices_by_prediction_column


def mean_line_support_from_endpoints(double[:, ::1] matrix, double x0, double y0, double x1, double y1):
    """Return the exact mean support sampled by ``line_filtering.mean_line_support``.

    The Python reference samples integer prediction columns from
    ``ceil(min(x0, x1))`` through ``floor(max(x0, x1))``.  It then rounds the
    interpolated row with Python's half-to-even rule and clips the row into the
    matrix.  This helper keeps that contract while avoiding a Python function
    call and NumPy indexing operation for every sampled column.
    """
    cdef Py_ssize_t n_reference_rows = matrix.shape[0]
    cdef Py_ssize_t n_prediction_columns = matrix.shape[1]
    cdef int x_start
    cdef int x_end
    cdef int prediction_column
    cdef int sampled_reference_row
    cdef double dx
    cdef double y_estimate
    cdef double support_sum = 0.0
    cdef Py_ssize_t support_count = 0

    if n_reference_rows <= 0 or n_prediction_columns <= 0:
        return 0.0

    if x0 < x1:
        x_start = <int>ceil(x0)
        x_end = <int>floor(x1)
    else:
        x_start = <int>ceil(x1)
        x_end = <int>floor(x0)

    if x_start < 0:
        x_start = 0
    if x_end > n_prediction_columns - 1:
        x_end = <int>n_prediction_columns - 1
    if x_end < x_start:
        return 0.0

    dx = x1 - x0
    for prediction_column in range(x_start, x_end + 1):
        if fabs(dx) < 1e-8:
            y_estimate = y0
        else:
            y_estimate = y0 + (((prediction_column - x0) / dx) * (y1 - y0))
        sampled_reference_row = _clip_int(
            _round_half_to_even(y_estimate),
            0,
            <int>n_reference_rows - 1,
        )
        support_sum += matrix[sampled_reference_row, prediction_column]
        support_count += 1

    if support_count <= 0:
        return 0.0
    return support_sum / support_count


def sample_line_path(double[:, ::1] matrix, double x0, double y0, double x1, double y1):
    """Sample one line into the Python path objects used by the ownership filter.

    The returned dictionary intentionally contains ordinary Python dicts, sets,
    and lists because the rest of the filtering code still consumes that shape.
    This keeps the acceleration boundary small and easy to audit: only the
    repeated per-column interpolation, row clipping, matrix lookup, and
    reference-row bridging loop moves into Cython.
    """
    cdef Py_ssize_t n_reference_rows = matrix.shape[0]
    cdef Py_ssize_t n_prediction_columns = matrix.shape[1]
    cdef int x_min
    cdef int x_max
    cdef int prediction_column
    cdef int sampled_reference_row
    cdef int previous_reference_row = 0
    cdef int bridge_start
    cdef int bridge_end
    cdef int bridge_reference_row
    cdef bint has_previous_reference_row = False
    cdef double dx
    cdef double y_estimate
    cdef double local_score
    cdef double total_score = 0.0
    cdef Py_ssize_t sample_count = 0
    cdef dict x_to_y
    cdef dict x_to_score
    cdef set prediction_segments
    cdef set reference_segments
    cdef list sampled_reference_rows

    if n_reference_rows <= 0 or n_prediction_columns <= 0:
        return None

    x_min = _line_x_min(x0, x1)
    x_max = _line_x_max(x0, x1)
    if x_min < 0:
        x_min = 0
    if x_max > n_prediction_columns - 1:
        x_max = <int>n_prediction_columns - 1
    if x_max < x_min:
        return None

    x_to_y = {}
    x_to_score = {}
    prediction_segments = set()
    reference_segments = set()
    sampled_reference_rows = []
    dx = x1 - x0

    for prediction_column in range(x_min, x_max + 1):
        if fabs(dx) < 1e-8:
            y_estimate = y0
        else:
            y_estimate = y0 + (((prediction_column - x0) / dx) * (y1 - y0))

        sampled_reference_row = _clip_int(
            _round_half_to_even(y_estimate),
            0,
            <int>n_reference_rows - 1,
        )
        local_score = matrix[sampled_reference_row, prediction_column]

        x_to_y[int(prediction_column)] = int(sampled_reference_row)
        x_to_score[int(prediction_column)] = float(local_score)
        prediction_segments.add(int(prediction_column))
        reference_segments.add(int(sampled_reference_row))
        sampled_reference_rows.append(int(sampled_reference_row))
        total_score += local_score
        sample_count += 1

        if has_previous_reference_row:
            if previous_reference_row <= sampled_reference_row:
                bridge_start = previous_reference_row
                bridge_end = sampled_reference_row
            else:
                bridge_start = sampled_reference_row
                bridge_end = previous_reference_row
            for bridge_reference_row in range(bridge_start, bridge_end + 1):
                reference_segments.add(int(bridge_reference_row))

        previous_reference_row = sampled_reference_row
        has_previous_reference_row = True

    if sample_count <= 0:
        return None

    return {
        "x_to_y": x_to_y,
        "x_to_score": x_to_score,
        "pred_segments": prediction_segments,
        "ref_segments": reference_segments,
        "sampled_reference_rows": sampled_reference_rows,
        "total_score": float(total_score),
        "mean_score": float(total_score / sample_count),
        "pred_min": int(x_min),
        "pred_max": int(x_max),
        "ref_min": int(min(reference_segments)) if reference_segments else 0,
        "ref_max": int(max(reference_segments)) if reference_segments else -1,
    }


cdef bint _candidate_owner_key_is_better(
    double local_score,
    double total_score,
    double mean_score,
    double negative_reference_row,
    double line_length,
    double negative_coverage_index,
    double best_local_score,
    double best_total_score,
    double best_mean_score,
    double best_negative_reference_row,
    double best_line_length,
    double best_negative_coverage_index,
):
    """Return the exact lexicographic comparison used for column ownership."""
    if local_score != best_local_score:
        return local_score > best_local_score
    if total_score != best_total_score:
        return total_score > best_total_score
    if mean_score != best_mean_score:
        return mean_score > best_mean_score
    if negative_reference_row != best_negative_reference_row:
        return negative_reference_row > best_negative_reference_row
    if line_length != best_line_length:
        return line_length > best_line_length
    return negative_coverage_index > best_negative_coverage_index


def compute_final_assignment_from_coverages(
    coverages,
    coverage_indices_by_prediction_column,
    int n_reference_rows,
    int n_prediction_columns,
):
    """Return final assignment lists for already-merged coverage objects.

    This mirrors ``_compute_final_assignment`` in the Python filter.  It keeps
    the same Python coverage dictionaries and the same ownership key, but it
    moves the repeated per-column owner-selection loop into compiled code.
    """
    cdef int prediction_column
    cdef int coverage_index
    cdef int sampled_reference_row
    cdef int winning_reference_row
    cdef int winning_coverage_index
    cdef double local_score
    cdef double total_score
    cdef double mean_score
    cdef double negative_reference_row
    cdef double line_length
    cdef double negative_coverage_index
    cdef double best_local_score = 0.0
    cdef double best_total_score = 0.0
    cdef double best_mean_score = 0.0
    cdef double best_negative_reference_row = 0.0
    cdef double best_line_length = 0.0
    cdef double best_negative_coverage_index = 0.0
    cdef bint has_best_owner
    cdef list mapped_y
    cdef list mapped_line_id
    cdef object coverage
    cdef object prediction_column_coverages
    cdef object line

    if n_prediction_columns < 0:
        n_prediction_columns = 0
    if n_reference_rows < 0:
        n_reference_rows = 0

    mapped_y = [float("nan") for _ in range(n_prediction_columns)]
    mapped_line_id = [-1 for _ in range(n_prediction_columns)]

    for prediction_column in range(n_prediction_columns):
        has_best_owner = False
        winning_reference_row = -1
        winning_coverage_index = -1
        prediction_column_coverages = coverage_indices_by_prediction_column[prediction_column]

        for coverage_index in prediction_column_coverages:
            coverage_index = int(coverage_index)
            coverage = coverages[coverage_index]
            sampled_reference_row = int(coverage["x_to_y"][prediction_column])
            line = coverage["line"]

            local_score = float(coverage["x_to_score"][prediction_column])
            total_score = float(coverage.get("total_score", 0.0))
            mean_score = float(coverage.get("mean_score", 0.0))
            negative_reference_row = -float(sampled_reference_row)
            line_length = float(line.get("length", 0.0))
            negative_coverage_index = -float(coverage_index)

            if (
                not has_best_owner
                or _candidate_owner_key_is_better(
                    local_score,
                    total_score,
                    mean_score,
                    negative_reference_row,
                    line_length,
                    negative_coverage_index,
                    best_local_score,
                    best_total_score,
                    best_mean_score,
                    best_negative_reference_row,
                    best_line_length,
                    best_negative_coverage_index,
                )
            ):
                has_best_owner = True
                best_local_score = local_score
                best_total_score = total_score
                best_mean_score = mean_score
                best_negative_reference_row = negative_reference_row
                best_line_length = line_length
                best_negative_coverage_index = negative_coverage_index
                winning_reference_row = sampled_reference_row
                winning_coverage_index = coverage_index

        if not has_best_owner:
            continue

        if n_reference_rows > 0:
            if winning_reference_row < 0:
                winning_reference_row = 0
            elif winning_reference_row > n_reference_rows - 1:
                winning_reference_row = n_reference_rows - 1

        mapped_line_id[prediction_column] = int(winning_coverage_index)
        mapped_y[prediction_column] = float(winning_reference_row)

    return {
        "mapped_y": mapped_y,
        "mapped_line_id": mapped_line_id,
    }


def unique_reference_rows_from_path_slice(
    sampled_reference_rows,
    int row_start,
    int row_end,
    int reference_window_count,
):
    """Return deduplicated reference rows for one column slice of a sampled path.

    Uses a C uint8 seen-array instead of a Python set for O(1) membership checks.
    ``row_start`` and ``row_end`` are slice indices into ``sampled_reference_rows``,
    not column numbers.  Rows outside ``[0, reference_window_count)`` are silently
    skipped so callers do not need boundary checks.
    """
    cdef int i, row_val, n_input, end
    cdef unsigned char[:] seen

    if row_end <= row_start or reference_window_count <= 0:
        return []

    n_input = len(sampled_reference_rows)
    end = row_end if row_end <= n_input else n_input
    if end <= row_start:
        return []

    seen_arr = np.zeros(reference_window_count, dtype=np.uint8)
    seen = seen_arr
    unique_rows = []

    for i in range(row_start, end):
        row_val = int(sampled_reference_rows[i])
        if row_val < 0 or row_val >= reference_window_count:
            continue
        if seen[row_val]:
            continue
        seen[row_val] = 1
        unique_rows.append(row_val)

    return unique_rows
