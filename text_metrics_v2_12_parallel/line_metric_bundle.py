"""Build per-line metric bundles and accumulate per-character coverage counts.

This module builds the stable v2.12 bundle schema used by the downstream
Levenshtein and coverage metrics. The previous internal coverage field names are
replaced here with non-legacy names while preserving the exact interval values.
"""

from __future__ import annotations

import numpy as np

from shared.ordered_sequence_helpers import (
    is_non_decreasing,
    ordered_unique,
    reference_rows_for_mapped_columns,
)
from shared.project_line_to_text_windows import (
    line_window_ids_from_endpoint,
    normalize_line_endpoints,
    num_windows_for_text_len,
    parse_point_xy,
    window_ids_to_merged_char_intervals,
)


# Return mapped reference rows for one line together with the reorder flag used by Levenshtein.
def reference_rows_for_levenshtein(
    owned_cols: list[int],
    mapped_y: np.ndarray,
    n_ref_windows: int,
) -> tuple[list[int], bool]:
    """Return mapped reference rows for one line and reordering flag."""
    return reference_rows_for_mapped_columns(owned_cols, mapped_y, int(n_ref_windows))


# Build one canonical line bundle with ownership ids and coverage intervals.
def build_line_metric_bundle(
    *,
    lines_used: list[dict],
    column_assignment: dict,
    n_ref_windows: int,
    n_other_windows: int,
    ref_text_len: int,
    other_text_len: int,
    window_size: int,
    window_stride: int,
) -> dict:
    """Build one canonical bundle with per-line ids and precomputed intervals."""
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    if mapped_y.shape != (int(n_other_windows),) or mapped_line_id.shape != (int(n_other_windows),):
        raise ValueError(
            "column_assignment must provide mapped_y and mapped_line_id arrays with shape "
            f"({int(n_other_windows)},), got {mapped_y.shape} and {mapped_line_id.shape}"
        )

    coverage_n_other_windows = num_windows_for_text_len(
        int(other_text_len),
        int(window_size),
        int(window_stride),
    )
    coverage_n_ref_windows = num_windows_for_text_len(
        int(ref_text_len),
        int(window_size),
        int(window_stride),
    )

    line_entries: list[dict] = []
    for line_id, line in enumerate(lines_used):
        owned_prediction_columns = [int(x) for x in np.flatnonzero(mapped_line_id == int(line_id))]
        mapped_reference_rows_per_x = [
            int(np.clip(round(float(mapped_y[x])), 0, int(n_ref_windows) - 1))
            for x in owned_prediction_columns
            if int(n_ref_windows) > 0 and np.isfinite(mapped_y[x])
        ]
        reference_rows_for_line_levenshtein, rows_reordered_for_monotonicity = reference_rows_for_levenshtein(
            owned_prediction_columns,
            mapped_y,
            int(n_ref_windows),
        )

        x_char_intervals_owned = window_ids_to_merged_char_intervals(
            owned_prediction_columns,
            text_len=int(other_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        y_char_intervals_owned = window_ids_to_merged_char_intervals(
            ordered_unique(mapped_reference_rows_per_x),
            text_len=int(ref_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )

        coverage_x_window_ids, coverage_y_window_ids = line_window_ids_from_endpoint(
            line,
            n_x_windows=int(coverage_n_other_windows),
            n_y_windows=int(coverage_n_ref_windows),
        )
        x_char_intervals_for_coverage = window_ids_to_merged_char_intervals(
            coverage_x_window_ids,
            text_len=int(other_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        y_char_intervals_for_coverage = window_ids_to_merged_char_intervals(
            coverage_y_window_ids,
            text_len=int(ref_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )

        line_entries.append(
            {
                "line_id": int(line_id),
                "x_window_ids_owned": owned_prediction_columns,
                "y_window_ids_mapped_per_x": mapped_reference_rows_per_x,
                "y_window_ids_for_levenshtein": reference_rows_for_line_levenshtein,
                "y_rows_reordered_for_monotonicity": bool(rows_reordered_for_monotonicity),
                "x_char_intervals_owned": x_char_intervals_owned,
                "y_char_intervals_owned": y_char_intervals_owned,
                "x_char_intervals_for_coverage": x_char_intervals_for_coverage,
                "y_char_intervals_for_coverage": y_char_intervals_for_coverage,
            }
        )

    return {
        "n_ref_windows": int(n_ref_windows),
        "n_other_windows": int(n_other_windows),
        "coverage_n_ref_windows": int(coverage_n_ref_windows),
        "coverage_n_other_windows": int(coverage_n_other_windows),
        "ref_text_len": int(ref_text_len),
        "other_text_len": int(other_text_len),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
        "line_guided_columns": int(np.sum(mapped_line_id >= 0)),
        "fallback_columns": int(np.sum(mapped_line_id < 0)),
        "lines": line_entries,
    }


# Accumulate per-character counts from many interval groups.
def accumulate_counts_from_interval_groups(
    *,
    text_len: int,
    interval_groups: list[list[tuple[int, int]]],
) -> np.ndarray:
    """Accumulate per-character counts from many interval groups.

    Each group represents one line. Intervals inside one group are assumed to be
    pre-merged so each line contributes +1 coverage per covered character.
    """
    if int(text_len) <= 0:
        return np.zeros(0, dtype=np.int32)

    diff = np.zeros(int(text_len) + 1, dtype=np.int64)
    for intervals in interval_groups:
        for start, end in intervals:
            interval_start = max(0, min(int(start), int(text_len)))
            interval_end = max(0, min(int(end), int(text_len)))
            if interval_end <= interval_start:
                continue
            diff[interval_start] += 1
            diff[interval_end] -= 1
    return np.cumsum(diff[:-1], dtype=np.int64).astype(np.int32)


__all__ = [
    "accumulate_counts_from_interval_groups",
    "build_line_metric_bundle",
    "is_non_decreasing",
    "line_window_ids_from_endpoint",
    "normalize_line_endpoints",
    "num_windows_for_text_len",
    "ordered_unique",
    "parse_point_xy",
    "reference_rows_for_levenshtein",
    "window_ids_to_merged_char_intervals",
]
