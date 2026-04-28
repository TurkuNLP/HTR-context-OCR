"""Build per-line metric bundles and accumulate per-character coverage counts.

This module keeps the exact bundle schema used by the v2.1 pipeline while
reusing shared helper utilities for sequence ordering and line-window mapping.
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


def reference_rows_for_levenshtein(
    owned_cols: list[int],
    mapped_y: np.ndarray,
    n_ref_windows: int,
) -> tuple[list[int], bool]:
    """Return mapped reference rows for one line and reordering flag."""
    return reference_rows_for_mapped_columns(owned_cols, mapped_y, int(n_ref_windows))


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

    # Coverage clipping mirrors previous count_text_on_lne semantics exactly.
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
    for lid, line in enumerate(lines_used):
        owned_cols = [int(x) for x in np.flatnonzero(mapped_line_id == int(lid))]
        mapped_rows_per_x = [
            int(np.clip(round(float(mapped_y[x])), 0, int(n_ref_windows) - 1))
            for x in owned_cols
            if int(n_ref_windows) > 0 and np.isfinite(mapped_y[x])
        ]
        y_for_lev, y_reordered = reference_rows_for_levenshtein(
            owned_cols,
            mapped_y,
            int(n_ref_windows),
        )

        x_char_intervals_owned = window_ids_to_merged_char_intervals(
            owned_cols,
            text_len=int(other_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        y_char_intervals_owned = window_ids_to_merged_char_intervals(
            ordered_unique(mapped_rows_per_x),
            text_len=int(ref_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )

        x_ids_legacy, y_ids_legacy = line_window_ids_from_endpoint(
            line,
            n_x_windows=int(coverage_n_other_windows),
            n_y_windows=int(coverage_n_ref_windows),
        )
        x_char_intervals_coverage_legacy = window_ids_to_merged_char_intervals(
            x_ids_legacy,
            text_len=int(other_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        y_char_intervals_coverage_legacy = window_ids_to_merged_char_intervals(
            y_ids_legacy,
            text_len=int(ref_text_len),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )

        line_entries.append(
            {
                "line_id": int(lid),
                "x_window_ids_owned": owned_cols,
                "y_window_ids_mapped_per_x": mapped_rows_per_x,
                "y_window_ids_for_levenshtein": y_for_lev,
                "y_rows_reordered_for_monotonicity": bool(y_reordered),
                "x_char_intervals_owned": x_char_intervals_owned,
                "y_char_intervals_owned": y_char_intervals_owned,
                "x_char_intervals_coverage_legacy": x_char_intervals_coverage_legacy,
                "y_char_intervals_coverage_legacy": y_char_intervals_coverage_legacy,
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
            s = max(0, min(int(start), int(text_len)))
            e = max(0, min(int(end), int(text_len)))
            if e <= s:
                continue
            diff[s] += 1
            diff[e] -= 1
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
