from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from alignment_utils.line_geometry_support import line_length, mean_line_support


def lines_from_merged_segments(
    matrix: np.ndarray,
    merged_lines: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[dict]:
    lines: list[dict] = []
    for p0, p1 in merged_lines:
        line = {
            "x0": float(p0[0]),
            "y0": float(p0[1]),
            "x1": float(p1[0]),
            "y1": float(p1[1]),
        }
        line["length"] = line_length(line)
        line["support"] = mean_line_support(matrix, line) if matrix.size else 0.0
        line["score"] = float(line["support"])
        lines.append(line)
    return lines
