from __future__ import annotations

"""Optional Cython acceleration for along-line bookkeeping.

The public functions in this module always preserve Python fallback behavior.
When ``cython_accel.along_lines_core`` is compiled, small hot loops use compiled
helpers; otherwise the pure-Python implementations run with the same outputs.
"""

from collections.abc import Sequence
import math

try:
    from .along_lines_core import group_owned_columns_by_line as _cython_group_owned_columns_by_line
    from .along_lines_core import weighted_mean_from_scores_and_lengths as _cython_weighted_mean
except Exception:
    _cython_group_owned_columns_by_line = None
    _cython_weighted_mean = None


def _python_group_owned_columns_by_line(mapped_line_id: Sequence[int], line_count_hint: int) -> list[list[int]]:
    """Group prediction columns by line id using the reference Python logic."""
    group_count = max(0, int(line_count_hint))
    owned_columns_by_line: list[list[int]] = [[] for _ in range(group_count)]

    for column_index, raw_line_id in enumerate(mapped_line_id):
        line_id = int(raw_line_id)
        if 0 <= line_id < group_count:
            owned_columns_by_line[line_id].append(int(column_index))

    return owned_columns_by_line


def _python_weighted_mean_from_scores_and_lengths(scores: Sequence[float], lengths: Sequence[float]) -> float | None:
    """Compute the normalized length-weighted mean with exact fallback semantics."""
    if len(scores) != len(lengths):
        raise ValueError("scores and lengths must have the same length")

    weighted_sum = 0.0
    total_length = 0.0
    for score_value, length_value in zip(scores, lengths):
        score = float(score_value)
        length = float(length_value)
        if math.isfinite(score) and math.isfinite(length) and length > 0.0:
            weighted_sum += score * length
            total_length += length

    if total_length <= 0.0:
        return None
    return float(weighted_sum / total_length)


def cython_line_grouping_available() -> bool:
    """Return ``True`` when the compiled grouping extension can be imported."""
    return _cython_group_owned_columns_by_line is not None


def cython_weighted_mean_available() -> bool:
    """Return ``True`` when the compiled weighted-mean helper is available."""
    return _cython_weighted_mean is not None


def group_owned_columns_by_line(mapped_line_id: Sequence[int], line_count_hint: int) -> list[list[int]]:
    """Group prediction columns by line id with optional Cython acceleration."""
    if _cython_group_owned_columns_by_line is None:
        return _python_group_owned_columns_by_line(mapped_line_id, int(line_count_hint))
    return _cython_group_owned_columns_by_line(mapped_line_id, int(line_count_hint))


def weighted_mean_from_scores_and_lengths(scores: Sequence[float], lengths: Sequence[float]) -> float | None:
    """Compute length-weighted mean with optional Cython acceleration."""
    if _cython_weighted_mean is None:
        return _python_weighted_mean_from_scores_and_lengths(scores, lengths)
    result = _cython_weighted_mean(scores, lengths)
    return None if result is None else float(result)


__all__ = [
    "cython_line_grouping_available",
    "cython_weighted_mean_available",
    "group_owned_columns_by_line",
    "weighted_mean_from_scores_and_lengths",
]
