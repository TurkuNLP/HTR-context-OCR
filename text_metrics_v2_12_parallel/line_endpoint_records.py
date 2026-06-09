from __future__ import annotations

import numpy as np

from alignment_utils.line_geometry_support import line_length, mean_line_support

__all__ = ["lines_from_hough_segments", "lines_from_merged_segments"]


# Convert geometric Hough segments into the stable line-dictionary records used downstream.
def lines_from_hough_segments(
    matrix: np.ndarray,
    hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[dict]:
    """Build downstream line records from Hough endpoint segments.

    The conversion is intentionally minimal and deterministic: each segment is
    re-expressed as the line-dictionary structure used by the true-IoU filter and
    the later reporting pipeline, then the current support/score semantics are
    populated exactly as before.
    """
    line_records: list[dict] = []
    for point_0, point_1 in hough_segments:
        line_record = {
            "x0": float(point_0[0]),
            "y0": float(point_0[1]),
            "x1": float(point_1[0]),
            "y1": float(point_1[1]),
        }
        line_record["length"] = line_length(line_record)
        line_record["support"] = mean_line_support(matrix, line_record) if matrix.size else 0.0
        line_record["score"] = float(line_record["support"])
        line_records.append(line_record)
    return line_records


# Backward-compatible wrapper name kept while the current default path still uses merged Hough output.
def lines_from_merged_segments(
    matrix: np.ndarray,
    merged_lines: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[dict]:
    """Backward-compatible wrapper around :func:`lines_from_hough_segments`."""
    return lines_from_hough_segments(matrix, merged_lines)
