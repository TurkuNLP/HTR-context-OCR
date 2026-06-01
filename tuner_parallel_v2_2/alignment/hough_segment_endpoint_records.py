from __future__ import annotations

"""Convert raw Hough endpoint segments into tuner line-record dictionaries.

The tuner passes raw probabilistic-Hough segments directly into true-IoU
filtering.  This small module keeps that conversion local to the tuner so the
hot path no longer depends on the historical ``text_metrics_v2_1_parallel``
``line_endpoint_records`` import.

The numerical behaviour mirrors the v2.12 endpoint conversion and geometry
support exactly: every segment becomes a dictionary with endpoints, Euclidean
length, mean matrix support, and score equal to that support.
"""

import math
from typing import Iterable

import numpy as np


def line_y_at_prediction_column(line_record: dict, prediction_column_index: int) -> float:
    """Interpolate the reference-row coordinate at one prediction column.

    The near-vertical fallback intentionally matches the v2.12 helper.  It keeps
    old numerical behaviour stable for rare segments where both endpoints share
    almost the same x coordinate.
    """
    x0 = float(line_record["x0"])
    y0 = float(line_record["y0"])
    x1 = float(line_record["x1"])
    y1 = float(line_record["y1"])
    delta_x = x1 - x0
    if abs(delta_x) < 1e-8:
        return y0

    interpolation_fraction = (int(prediction_column_index) - x0) / delta_x
    return y0 + interpolation_fraction * (y1 - y0)


def line_record_length(line_record: dict) -> float:
    """Return the Euclidean length of a line-record endpoint dictionary."""
    return float(
        math.hypot(
            float(line_record["x1"]) - float(line_record["x0"]),
            float(line_record["y1"]) - float(line_record["y0"]),
        )
    )


def mean_line_support_from_score_matrix(score_matrix: np.ndarray, line_record: dict) -> float:
    """Return mean score-matrix support sampled along one line segment.

    The sampling range, rounding, and clipping match the v2.12 text-metrics
    helper so converting this import into tuner-local code does not change
    downstream filter scores or best-parameter ranking.
    """
    if score_matrix.size == 0:
        return 0.0

    reference_window_count, prediction_window_count = score_matrix.shape
    x_start = int(max(0, math.ceil(min(float(line_record["x0"]), float(line_record["x1"])))))
    x_end = int(min(prediction_window_count - 1, math.floor(max(float(line_record["x0"]), float(line_record["x1"])))))
    if x_end < x_start:
        return 0.0

    sampled_values: list[float] = []
    for prediction_column_index in range(x_start, x_end + 1):
        interpolated_reference_row = line_y_at_prediction_column(line_record, prediction_column_index)
        reference_row_index = int(np.clip(round(interpolated_reference_row), 0, reference_window_count - 1))
        sampled_values.append(float(score_matrix[reference_row_index, prediction_column_index]))

    return float(np.mean(sampled_values)) if sampled_values else 0.0


def line_records_from_raw_hough_segments(
    score_matrix: np.ndarray,
    raw_hough_segments: Iterable[tuple[tuple[float, float], tuple[float, float]]],
) -> list[dict]:
    """Build true-IoU filter line records from raw Hough endpoint segments."""
    line_records: list[dict] = []
    for point_0, point_1 in raw_hough_segments:
        line_record = {
            "x0": float(point_0[0]),
            "y0": float(point_0[1]),
            "x1": float(point_1[0]),
            "y1": float(point_1[1]),
        }
        line_record["length"] = line_record_length(line_record)
        line_record["support"] = mean_line_support_from_score_matrix(score_matrix, line_record)
        line_record["score"] = float(line_record["support"])
        line_records.append(line_record)

    return line_records


__all__ = [
    "line_records_from_raw_hough_segments",
    "line_record_length",
    "line_y_at_prediction_column",
    "mean_line_support_from_score_matrix",
]
