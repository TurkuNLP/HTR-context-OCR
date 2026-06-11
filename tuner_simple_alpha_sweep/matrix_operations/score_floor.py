from __future__ import annotations

"""Score-floor preprocessing for Hough voter masks."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ScoreFloorStatistics:
    """Reusable finite-cell score statistics for one score matrix."""

    score_mean: float
    score_standard_deviation: float


# Ask Python to generate common data-container methods for the class defined next.
@dataclass(frozen=True)
# Define the ScoreFloorResult class, which groups related state and behavior for this part of the pipeline.
class ScoreFloorResult:
    """Statistics and binary masks derived from one score matrix."""

    # Define the score_mean field; it stores the average score of all finite matrix cells, used as the center of the score distribution.
    score_mean: float
    # Define the score_standard_deviation field; it stores how widely matrix scores spread around the mean, used to raise the floor above ordinary cells.
    score_standard_deviation: float
    # Define the score_floor_alpha field; it stores the user-selected multiplier that controls how strongly the standard deviation raises the floor.
    score_floor_alpha: float
    # Define the score_floor field; it stores the numeric cutoff a score must meet before it can become a Hough voter.
    score_floor: float
    # Define the active_cell_count field; it stores how many matrix cells survived the score floor and became active candidates.
    active_cell_count: int
    # Define the active_fraction field; it stores the active candidate count divided by the total number of matrix cells.
    active_fraction: float
    # Define the hough_input_mask field; it stores the boolean matrix passed to Hough, where True means this cell can vote for a line.
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


def compute_score_floor_mask_from_statistics(
    score_matrix: np.ndarray,
    *,
    alpha: float,
    statistics: ScoreFloorStatistics,
) -> ScoreFloorResult:
    """Build the Hough mask for one alpha using precomputed matrix statistics."""

    matrix_values = np.asarray(score_matrix, dtype=float)
    score_mean = float(statistics.score_mean)
    score_standard_deviation = float(statistics.score_standard_deviation)
    score_floor = float(score_mean + float(alpha) * score_standard_deviation)
    hough_input_mask = np.asarray(matrix_values >= score_floor, dtype=bool)
    active_cell_count = int(np.count_nonzero(hough_input_mask))
    total_cell_count = int(hough_input_mask.size)
    active_fraction = 0.0 if total_cell_count <= 0 else float(active_cell_count / total_cell_count)
    return ScoreFloorResult(
        score_mean=score_mean,
        score_standard_deviation=score_standard_deviation,
        score_floor_alpha=float(alpha),
        score_floor=score_floor,
        active_cell_count=active_cell_count,
        active_fraction=active_fraction,
        hough_input_mask=hough_input_mask,
    )


# Define the compute_score_floor_mask function; its body below performs one named step of the pipeline.
def compute_score_floor_mask(score_matrix: np.ndarray, *, alpha: float) -> ScoreFloorResult:
    """Build the simple Hough mask from `mean + alpha * standard deviation`."""

    return compute_score_floor_mask_from_statistics(
        score_matrix,
        alpha=float(alpha),
        statistics=compute_score_floor_statistics(score_matrix),
    )


__all__ = [
    "ScoreFloorResult",
    "ScoreFloorStatistics",
    "compute_score_floor_mask",
    "compute_score_floor_mask_from_statistics",
    "compute_score_floor_statistics",
]
