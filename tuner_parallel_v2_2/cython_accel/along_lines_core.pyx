# cython: language_level=3

"""Cython helpers for hot-path along-line bookkeeping.

The extension intentionally accelerates only small deterministic loops.  It does
not call Hough, alter filtering decisions, or change Levenshtein semantics.
"""

from libc.math cimport isfinite


def group_owned_columns_by_line(mapped_line_id, int line_count_hint):
    """Return ``list[list[int]]`` mapping line ids to owned prediction columns."""
    cdef int group_count = line_count_hint
    cdef Py_ssize_t column_index
    cdef Py_ssize_t column_count
    cdef int line_id
    cdef list owned_columns_by_line

    if group_count < 0:
        group_count = 0

    owned_columns_by_line = [[] for _ in range(group_count)]
    column_count = len(mapped_line_id)

    for column_index in range(column_count):
        line_id = int(mapped_line_id[column_index])
        if 0 <= line_id < group_count:
            owned_columns_by_line[line_id].append(int(column_index))

    return owned_columns_by_line


def weighted_mean_from_scores_and_lengths(scores, lengths):
    """Return ``sum(score * length) / sum(length)`` for valid positive lengths.

    The Python fallback has identical behavior.  Invalid/non-finite scores and
    non-positive lengths are skipped because they cannot contribute to a
    normalized length-weighted mean.
    """
    cdef Py_ssize_t index
    cdef Py_ssize_t count = len(scores)
    cdef double score
    cdef double length
    cdef double weighted_sum = 0.0
    cdef double total_length = 0.0

    if count != len(lengths):
        raise ValueError("scores and lengths must have the same length")

    for index in range(count):
        score = float(scores[index])
        length = float(lengths[index])
        if isfinite(score) and isfinite(length) and length > 0.0:
            weighted_sum += score * length
            total_length += length

    if total_length <= 0.0:
        return None
    return weighted_sum / total_length
