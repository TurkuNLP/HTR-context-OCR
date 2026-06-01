# cython: boundscheck=False, wraparound=False, initializedcheck=False, language_level=3

import numpy as np
cimport numpy as cnp


def label_connected_components_uint8(cnp.ndarray[cnp.uint8_t, ndim=2] active_mask):
    """Label eight-connected active cells in a binary matrix."""
    cdef Py_ssize_t row_count = active_mask.shape[0]
    cdef Py_ssize_t column_count = active_mask.shape[1]
    cdef cnp.ndarray[cnp.int32_t, ndim=2] component_labels = np.zeros(
        (row_count, column_count), dtype=np.int32
    )
    cdef cnp.ndarray[cnp.intp_t, ndim=1] stack_rows = np.empty(row_count * column_count, dtype=np.intp)
    cdef cnp.ndarray[cnp.intp_t, ndim=1] stack_columns = np.empty(row_count * column_count, dtype=np.intp)
    cdef Py_ssize_t row_index, column_index, active_row, active_column
    cdef Py_ssize_t neighbour_row, neighbour_column
    cdef Py_ssize_t row_start, row_end, column_start, column_end
    cdef Py_ssize_t stack_size
    cdef int current_label = 0

    for row_index in range(row_count):
        for column_index in range(column_count):
            if active_mask[row_index, column_index] == 0 or component_labels[row_index, column_index] != 0:
                continue

            current_label += 1
            component_labels[row_index, column_index] = current_label
            stack_size = 1
            stack_rows[0] = row_index
            stack_columns[0] = column_index

            while stack_size > 0:
                stack_size -= 1
                active_row = stack_rows[stack_size]
                active_column = stack_columns[stack_size]

                row_start = active_row - 1 if active_row > 0 else 0
                row_end = active_row + 2 if active_row + 2 < row_count else row_count
                column_start = active_column - 1 if active_column > 0 else 0
                column_end = active_column + 2 if active_column + 2 < column_count else column_count

                for neighbour_row in range(row_start, row_end):
                    for neighbour_column in range(column_start, column_end):
                        if neighbour_row == active_row and neighbour_column == active_column:
                            continue
                        if (
                            active_mask[neighbour_row, neighbour_column] != 0
                            and component_labels[neighbour_row, neighbour_column] == 0
                        ):
                            component_labels[neighbour_row, neighbour_column] = current_label
                            stack_rows[stack_size] = neighbour_row
                            stack_columns[stack_size] = neighbour_column
                            stack_size += 1

    return component_labels, int(current_label)
