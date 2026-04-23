from __future__ import annotations

import numpy as np


def ordered_unique(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue in seen:
            continue
        out.append(ivalue)
        seen.add(ivalue)
    return out


def is_non_decreasing(values: list[int]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def num_windows_for_text_len(text_len: int, window_size: int, window_stride: int) -> int:
    text_len = int(text_len)
    window_size = int(window_size)
    window_stride = int(window_stride)
    if text_len < window_size:
        return 0
    return ((text_len - window_size) // window_stride) + 1


def reference_rows_for_levenshtein(
    owned_cols: list[int],
    mapped_y: np.ndarray,
    n_ref_windows: int,
) -> tuple[list[int], bool]:
    if n_ref_windows <= 0:
        return [], False

    rows = [
        int(np.clip(round(float(mapped_y[x])), 0, n_ref_windows - 1))
        for x in owned_cols
        if 0 <= int(x) < mapped_y.shape[0] and np.isfinite(mapped_y[x])
    ]
    if not rows:
        return [], False

    unique_rows = ordered_unique(rows)
    if is_non_decreasing(unique_rows):
        return unique_rows, False
    return sorted(set(unique_rows)), True


def _parse_point_xy(point) -> tuple[float, float]:
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Invalid point format: {point!r}")
    return float(arr[0]), float(arr[1])


def _normalize_line_endpoints(line) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(line, dict):
        try:
            return (
                (float(line["x0"]), float(line["y0"])),
                (float(line["x1"]), float(line["y1"])),
            )
        except KeyError as exc:
            raise ValueError(f"Line dict missing endpoint key: {exc}") from exc

    if isinstance(line, (list, tuple)) and len(line) == 2:
        p0, p1 = line
        x0, y0 = _parse_point_xy(p0)
        x1, y1 = _parse_point_xy(p1)
        return (x0, y0), (x1, y1)

    raise ValueError(f"Unsupported line format: {line!r}")


def line_window_ids_from_endpoint(
    line,
    *,
    n_x_windows: int,
    n_y_windows: int,
) -> tuple[list[int], list[int]]:
    if n_x_windows <= 0 and n_y_windows <= 0:
        return [], []

    (x0, y0), (x1, y1) = _normalize_line_endpoints(line)
    n_steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n_steps = max(n_steps, 1)

    xs = np.rint(np.linspace(x0, x1, n_steps)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n_steps)).astype(int)

    x_ids: list[int] = []
    y_ids: list[int] = []
    if n_x_windows > 0:
        xs = np.clip(xs, 0, n_x_windows - 1)
        x_ids = sorted(set(int(v) for v in xs.tolist()))
    if n_y_windows > 0:
        ys = np.clip(ys, 0, n_y_windows - 1)
        y_ids = sorted(set(int(v) for v in ys.tolist()))
    return x_ids, y_ids


def window_ids_to_merged_char_intervals(
    window_ids: list[int],
    *,
    text_len: int,
    window_size: int,
    window_stride: int,
) -> list[tuple[int, int]]:
    if not window_ids or text_len <= 0:
        return []

    raw: list[tuple[int, int]] = []
    for idx in sorted(set(int(v) for v in window_ids)):
        start = int(idx) * int(window_stride)
        end = min(start + int(window_size), int(text_len))
        if start >= text_len or end <= start:
            continue
        raw.append((start, end))
    if not raw:
        return []

    merged: list[tuple[int, int]] = [raw[0]]
    for start, end in raw[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


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
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    if mapped_y.shape != (int(n_other_windows),) or mapped_line_id.shape != (int(n_other_windows),):
        raise ValueError(
            "column_assignment must provide mapped_y and mapped_line_id arrays with shape "
            f"({int(n_other_windows)},), got {mapped_y.shape} and {mapped_line_id.shape}"
        )

    # Coverage clipping must mirror count_text_on_lne semantics exactly.
    # That logic derives window counts from text length + window params,
    # not from score-matrix shape.
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
