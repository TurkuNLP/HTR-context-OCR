from __future__ import annotations

"""Configuration primitives and shared types for tuner_parallel_v2."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from .line_filtering_v2_1_IoU_fast import DEFAULT_MIN_IOU_THRESHOLD
except ImportError:
    from line_filtering_v2_1_IoU_fast import DEFAULT_MIN_IOU_THRESHOLD  # type: ignore


LogFn = Callable[[str], None]

MATRIX_CACHE_VERSION = "score_matrix_v2_strict_theta_30_60"

PARAM_HOUGH_THRESHOLD = "hough_threshold"
PARAM_HOUGH_LINE_LENGTH = "hough_line_length"
PARAM_HOUGH_LINE_GAP = "hough_line_gap"
PARAM_HOUGH_SEED = "hough_seed"

SUPPORTED_SWEEP_PARAMETERS: tuple[str, ...] = (
    PARAM_HOUGH_THRESHOLD,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_LINE_GAP,
)

# User-requested fixed tuning grid.
# NOTE: threshold now starts from 1 (not 0).
HOUGH_THRESHOLD_MIN = 1
HOUGH_THRESHOLD_MAX = 40
HOUGH_LINE_LENGTH_MIN = 1
HOUGH_LINE_LENGTH_MAX = 50
HOUGH_LINE_GAP_MIN = 1
HOUGH_LINE_GAP_MAX = 30

# Read-only default index-cache directory produced by text_metrics_v2_1_parallel.
DEFAULT_SCORE_INDEX_CACHE_DIR = (
    Path(__file__).resolve().parent.parent / "text_metrics_v2_1_parallel" / ".score_index_cache"
)


@dataclass(frozen=True)
class HoughBaselineConfig:
    """Base configuration for non-grid parameters and fixed seed."""

    hough_threshold: int = 26
    hough_line_length: int = 10
    hough_line_gap: int = 15
    hough_seed: int = 0
    hough_start: float = 2.6
    align_abs_min_len: float = 8.0
    align_min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD


@dataclass
class SweepDocument:
    """Prepared document payload reused across all combinations."""

    index: int
    fname: str
    pred: str
    ref: str
    matrix: np.ndarray
    whole_document_nls: float
    pred_blocks: list[str]
    ref_blocks: list[str]
    hough_ctx: dict


def fixed_parameter_ranges() -> dict[str, tuple[int, int]]:
    """Return hardcoded fixed tuner ranges for threshold/line_length/line_gap."""
    return {
        PARAM_HOUGH_THRESHOLD: (HOUGH_THRESHOLD_MIN, HOUGH_THRESHOLD_MAX),
        PARAM_HOUGH_LINE_LENGTH: (HOUGH_LINE_LENGTH_MIN, HOUGH_LINE_LENGTH_MAX),
        PARAM_HOUGH_LINE_GAP: (HOUGH_LINE_GAP_MIN, HOUGH_LINE_GAP_MAX),
    }


def fixed_values_for(param: str) -> list[int]:
    """Expand fixed min/max range for one parameter into integer values."""
    ranges = fixed_parameter_ranges()
    if param not in ranges:
        raise KeyError(f"Unsupported parameter {param!r}")
    lo, hi = ranges[param]
    return list(range(int(lo), int(hi) + 1))


__all__ = [
    "DEFAULT_MIN_IOU_THRESHOLD",
    "DEFAULT_SCORE_INDEX_CACHE_DIR",
    "HOUGH_THRESHOLD_MIN",
    "HOUGH_THRESHOLD_MAX",
    "HOUGH_LINE_LENGTH_MIN",
    "HOUGH_LINE_LENGTH_MAX",
    "HOUGH_LINE_GAP_MIN",
    "HOUGH_LINE_GAP_MAX",
    "HoughBaselineConfig",
    "LogFn",
    "MATRIX_CACHE_VERSION",
    "PARAM_HOUGH_THRESHOLD",
    "PARAM_HOUGH_LINE_LENGTH",
    "PARAM_HOUGH_LINE_GAP",
    "PARAM_HOUGH_SEED",
    "SUPPORTED_SWEEP_PARAMETERS",
    "SweepDocument",
    "fixed_parameter_ranges",
    "fixed_values_for",
]
