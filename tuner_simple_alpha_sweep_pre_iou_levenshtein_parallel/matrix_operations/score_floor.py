from __future__ import annotations

"""Pre-Hough score-mask construction for tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel."""

from dataclasses import dataclass
import math

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.cython_accel.optional_threshold_mask import threshold_mask_at_or_above


@dataclass(frozen=True)
class ScoreFloorStatistics:
    """Reusable finite-cell score statistics for one score matrix."""

    score_mean: float
    score_standard_deviation: float


@dataclass(frozen=True)
class ScoreFloorResult:
    """Statistics and binary masks derived from one score matrix."""

    score_mean: float
    score_standard_deviation: float
    score_floor_alpha: float
    score_floor: float
    active_cell_count: int
    active_fraction: float
    hough_input_mask: np.ndarray


def compute_score_floor_statistics(score_matrix: np.ndarray) -> ScoreFloorStatistics:
    """Compute reusable finite-cell statistics for score-floor masks."""
    matrix_values = np.asarray(score_matrix, dtype=float)
    finite_values = matrix_values[np.isfinite(matrix_values)]
    if finite_values.size == 0:
        return ScoreFloorStatistics(score_mean=0.0, score_standard_deviation=0.0)
    return ScoreFloorStatistics(
        score_mean=float(np.mean(finite_values)),
        score_standard_deviation=float(np.std(finite_values, ddof=0)),
    )


def build_boolean_threshold_mask(score_matrix: np.ndarray, *, threshold: float) -> np.ndarray:
    """Return cells at or above threshold, using Cython when the helper is built."""
    matrix_values = np.ascontiguousarray(score_matrix, dtype=float)
    cython_mask = threshold_mask_at_or_above(matrix_values, float(threshold))
    if cython_mask is not None:
        return np.asarray(cython_mask, dtype=bool)
    return np.asarray(matrix_values >= float(threshold), dtype=bool)


def score_floor_result_from_threshold(
    score_matrix: np.ndarray,
    *,
    score_floor_alpha: float,
    score_floor: float,
    statistics: ScoreFloorStatistics,
) -> ScoreFloorResult:
    """Build the shared result object after a numeric pre-Hough threshold is known."""
    hough_input_mask = build_boolean_threshold_mask(score_matrix, threshold=float(score_floor))
    active_cell_count = int(np.count_nonzero(hough_input_mask))
    total_cell_count = int(hough_input_mask.size)
    active_fraction = 0.0 if total_cell_count <= 0 else float(active_cell_count / total_cell_count)
    return ScoreFloorResult(
        score_mean=float(statistics.score_mean),
        score_standard_deviation=float(statistics.score_standard_deviation),
        score_floor_alpha=float(score_floor_alpha),
        score_floor=float(score_floor),
        active_cell_count=active_cell_count,
        active_fraction=active_fraction,
        hough_input_mask=hough_input_mask,
    )


def compute_score_floor_mask_from_statistics(
    score_matrix: np.ndarray,
    *,
    alpha: float,
    statistics: ScoreFloorStatistics,
) -> ScoreFloorResult:
    """Build the current mean-plus-alpha-standard-deviation Hough mask."""
    score_floor = float(statistics.score_mean + float(alpha) * statistics.score_standard_deviation)
    return score_floor_result_from_threshold(
        score_matrix,
        score_floor_alpha=float(alpha),
        score_floor=score_floor,
        statistics=statistics,
    )


def infer_score_matrix_scale(score_matrix: np.ndarray) -> str:
    """Return percent for 0-100 matrices and unit for 0-1 matrices."""
    matrix_values = np.asarray(score_matrix, dtype=float)
    finite_values = matrix_values[np.isfinite(matrix_values)]
    if finite_values.size == 0:
        return "percent"
    return "percent" if float(np.max(finite_values)) > 1.5 else "unit"


def convert_minimum_levenshtein_to_matrix_threshold(
    score_matrix: np.ndarray,
    *,
    minimum_levenshtein: float,
) -> float:
    """Convert 0.30 and 30.0 into the same threshold on the matrix scale."""
    minimum_value = float(minimum_levenshtein)
    if not math.isfinite(minimum_value) or minimum_value < 0.0:
        raise ValueError("minimum Levenshtein threshold must be finite and non-negative")
    matrix_scale = infer_score_matrix_scale(score_matrix)
    if matrix_scale == "percent":
        return minimum_value * 100.0 if minimum_value <= 1.0 else minimum_value
    return minimum_value / 100.0 if minimum_value > 1.0 else minimum_value


def compute_minimum_levenshtein_mask(
    score_matrix: np.ndarray,
    *,
    minimum_levenshtein: float,
    statistics: ScoreFloorStatistics,
) -> ScoreFloorResult:
    """Build one fixed-threshold pre-Hough mask from a minimum Levenshtein score."""
    matrix_threshold = convert_minimum_levenshtein_to_matrix_threshold(
        score_matrix,
        minimum_levenshtein=float(minimum_levenshtein),
    )
    return score_floor_result_from_threshold(
        score_matrix,
        score_floor_alpha=0.0,
        score_floor=float(matrix_threshold),
        statistics=statistics,
    )


def compute_score_floor_mask(score_matrix: np.ndarray, *, alpha: float) -> ScoreFloorResult:
    """Build the current simple Hough mask from mean + alpha * standard deviation."""
    return compute_score_floor_mask_from_statistics(
        score_matrix,
        alpha=float(alpha),
        statistics=compute_score_floor_statistics(score_matrix),
    )


__all__ = [
    "ScoreFloorResult",
    "ScoreFloorStatistics",
    "build_boolean_threshold_mask",
    "compute_minimum_levenshtein_mask",
    "compute_score_floor_mask",
    "compute_score_floor_mask_from_statistics",
    "compute_score_floor_statistics",
    "convert_minimum_levenshtein_to_matrix_threshold",
    "infer_score_matrix_scale",
]
