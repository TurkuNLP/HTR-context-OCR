"""Shared geometry helpers used by line alignment and filtering modules.

These functions are intentionally kept behavior-compatible with the prior
implementations so numerical results remain unchanged.
"""

from __future__ import annotations

import math

import numpy as np


def line_y_at_x(line: dict, x: int) -> float:
    """Interpolate a line's y-position at prediction-column ``x``.

    The interpolation rule matches legacy behavior exactly, including the
    near-vertical fallback to ``y0``.
    """
    x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
    dx = x1 - x0
    if abs(dx) < 1e-8:
        return y0
    t = (x - x0) / dx
    return y0 + t * (y1 - y0)


def line_length(line: dict) -> float:
    """Return Euclidean segment length for a line endpoint dictionary."""
    return float(math.hypot(line["x1"] - line["x0"], line["y1"] - line["y0"]))


def mean_line_support(matrix: np.ndarray, line: dict) -> float:
    """Return mean matrix support sampled along one line segment.

    Sampling and clipping behavior is preserved from the original helper so
    downstream ranking and filtering scores remain identical.
    """
    if matrix.size == 0:
        return 0.0

    n_ref, n_pred = matrix.shape
    x_start = int(max(0, math.ceil(min(line["x0"], line["x1"]))))
    x_end = int(min(n_pred - 1, math.floor(max(line["x0"], line["x1"]))))
    if x_end < x_start:
        return 0.0

    values = []
    for x in range(x_start, x_end + 1):
        y_idx = int(np.clip(round(line_y_at_x(line, x)), 0, n_ref - 1))
        values.append(float(matrix[y_idx, x]))
    return float(np.mean(values)) if values else 0.0
