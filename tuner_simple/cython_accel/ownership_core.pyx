# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False

"""Compiled column-ownership scan for tuner_simple."""

from libc.math cimport fabs, floor, isfinite
import numpy as np


cdef long _round_half_to_even(double value):
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


cdef double _minimum_double(double left_value, double right_value):
    if left_value <= right_value:
        return left_value
    return right_value


cdef double _maximum_double(double left_value, double right_value):
    if left_value >= right_value:
        return left_value
    return right_value


def assign_columns_to_candidate_lines(
    double[:, ::1] score_matrix,
    unsigned char[:, ::1] voter_mask,
    double[::1] candidate_x0,
    double[::1] candidate_y0,
    double[::1] candidate_x1,
    double[::1] candidate_y1,
):
    """Assign each prediction column to the strongest candidate line crossing it."""
    cdef Py_ssize_t row_count = score_matrix.shape[0]
    cdef Py_ssize_t column_count = score_matrix.shape[1]
    cdef Py_ssize_t candidate_count = candidate_x0.shape[0]
    cdef Py_ssize_t column_index
    cdef Py_ssize_t candidate_id
    cdef double x0
    cdef double y0
    cdef double x1
    cdef double y1
    cdef double x_position
    cdef double y_value
    cdef double dx
    cdef long row_index_long
    cdef Py_ssize_t row_index
    cdef double score_value
    cdef double best_score
    cdef Py_ssize_t best_candidate_id
    cdef double best_y_value

    if row_count <= 0 or column_count <= 0 or candidate_count <= 0:
        return None
    if voter_mask.shape[0] != row_count or voter_mask.shape[1] != column_count:
        raise ValueError("voter_mask shape must match score_matrix shape")
    if candidate_y0.shape[0] != candidate_count or candidate_x1.shape[0] != candidate_count or candidate_y1.shape[0] != candidate_count:
        raise ValueError("candidate coordinate arrays must have the same length")

    mapped_y = np.full(<int>column_count, np.nan, dtype=np.float64)
    mapped_candidate_id = np.full(<int>column_count, -1, dtype=np.int64)
    owned_counts = np.zeros(<int>candidate_count, dtype=np.int64)

    cdef double[::1] mapped_y_view = mapped_y
    cdef long[::1] mapped_candidate_id_view = mapped_candidate_id
    cdef long[::1] owned_counts_view = owned_counts

    for column_index in range(column_count):
        x_position = <double>column_index
        best_candidate_id = -1
        best_y_value = 0.0
        best_score = -1.7976931348623157e308

        for candidate_id in range(candidate_count):
            x0 = candidate_x0[candidate_id]
            y0 = candidate_y0[candidate_id]
            x1 = candidate_x1[candidate_id]
            y1 = candidate_y1[candidate_id]

            if x_position < _minimum_double(x0, x1) - 1e-9:
                continue
            if x_position > _maximum_double(x0, x1) + 1e-9:
                continue

            dx = x1 - x0
            if fabs(dx) <= 1e-12:
                continue

            y_value = y0 + (((x_position - x0) / dx) * (y1 - y0))
            if not isfinite(y_value):
                continue

            row_index_long = _round_half_to_even(y_value)
            if row_index_long < 0 or row_index_long >= row_count:
                continue
            row_index = <Py_ssize_t>row_index_long

            if voter_mask[row_index, column_index] == 0:
                continue

            score_value = score_matrix[row_index, column_index]
            if score_value > best_score:
                best_candidate_id = candidate_id
                best_y_value = y_value
                best_score = score_value

        if best_candidate_id >= 0:
            mapped_candidate_id_view[column_index] = <long>best_candidate_id
            mapped_y_view[column_index] = best_y_value
            owned_counts_view[best_candidate_id] += 1

    return {
        "mapped_y": mapped_y,
        "mapped_candidate_id": mapped_candidate_id,
        "owned_counts": owned_counts,
    }
