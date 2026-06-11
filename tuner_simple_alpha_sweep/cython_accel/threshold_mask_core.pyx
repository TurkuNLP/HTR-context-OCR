from __future__ import annotations

import numpy as np
cimport numpy as cnp


def threshold_mask_at_or_above(double[:, ::1] matrix, double threshold):
    """Return a boolean mask where finite matrix cells are at or above threshold."""
    cdef Py_ssize_t row_count = matrix.shape[0]
    cdef Py_ssize_t column_count = matrix.shape[1]
    cdef Py_ssize_t row_index
    cdef Py_ssize_t column_index
    cdef double value
    cdef cnp.ndarray[cnp.npy_bool, ndim=2] mask = np.empty((row_count, column_count), dtype=np.bool_)
    cdef cnp.npy_bool[:, ::1] mask_view = mask

    for row_index in range(row_count):
        for column_index in range(column_count):
            value = matrix[row_index, column_index]
            mask_view[row_index, column_index] = value == value and value >= threshold
    return mask
