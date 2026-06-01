from __future__ import annotations

"""Line-to-text-window projection helpers copied from v2.12.

Source:
`text_metrics_v2_12_parallel/shared/project_line_to_text_windows.py`

Copied on: 2026-05-25.
"""

import numpy as np


def num_windows_for_text_len(text_len: int, window_size: int, window_stride: int) -> int:
    """Return number of stride windows for a text length and window config."""
    text_len = int(text_len)
    window_size = int(window_size)
    window_stride = int(window_stride)
    if text_len < window_size:
        return 0
    return ((text_len - window_size) // window_stride) + 1


def parse_point_xy(point) -> tuple[float, float]:
    """Convert a point-like input into an ``(x, y)`` float pair."""
    arr = np.asarray(point, dtype=float).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Invalid point format: {point!r}")
    return float(arr[0]), float(arr[1])


def normalize_line_endpoints(line) -> tuple[tuple[float, float], tuple[float, float]]:
    """Normalize one line into endpoint form ``((x0, y0), (x1, y1))``."""
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
        x0, y0 = parse_point_xy(p0)
        x1, y1 = parse_point_xy(p1)
        return (x0, y0), (x1, y1)

    raise ValueError(f"Unsupported line format: {line!r}")


def line_window_ids_from_endpoint(
    line,
    *,
    n_x_windows: int,
    n_y_windows: int,
) -> tuple[list[int], list[int]]:
    """Sample one line and return covered x/y window ids."""
    if int(n_x_windows) <= 0 and int(n_y_windows) <= 0:
        return [], []

    (x0, y0), (x1, y1) = normalize_line_endpoints(line)
    n_steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    n_steps = max(n_steps, 1)

    xs = np.rint(np.linspace(x0, x1, n_steps)).astype(int)
    ys = np.rint(np.linspace(y0, y1, n_steps)).astype(int)

    x_ids: list[int] = []
    y_ids: list[int] = []
    if int(n_x_windows) > 0:
        xs = np.clip(xs, 0, int(n_x_windows) - 1)
        x_ids = sorted(set(int(v) for v in xs.tolist()))
    if int(n_y_windows) > 0:
        ys = np.clip(ys, 0, int(n_y_windows) - 1)
        y_ids = sorted(set(int(v) for v in ys.tolist()))
    return x_ids, y_ids


def window_ids_to_merged_char_intervals(
    window_ids: list[int],
    *,
    text_len: int,
    window_size: int,
    window_stride: int,
) -> list[tuple[int, int]]:
    """Convert window ids into merged character intervals ``[start, end)``."""
    if not window_ids or int(text_len) <= 0:
        return []

    raw: list[tuple[int, int]] = []
    for idx in sorted(set(int(v) for v in window_ids)):
        start = int(idx) * int(window_stride)
        end = min(start + int(window_size), int(text_len))
        if start >= int(text_len) or end <= start:
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


__all__ = [
    "line_window_ids_from_endpoint",
    "normalize_line_endpoints",
    "num_windows_for_text_len",
    "parse_point_xy",
    "window_ids_to_merged_char_intervals",
]
