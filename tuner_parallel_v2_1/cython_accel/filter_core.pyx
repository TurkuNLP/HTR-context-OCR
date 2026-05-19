# cython: language_level=3

"""Cython helpers for behavior-preserving filtering internals.

The functions in this file intentionally operate on the same Python dict/set
objects as the reference implementation.  That keeps semantics identical while
moving high-frequency loops into compiled code.
"""


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
