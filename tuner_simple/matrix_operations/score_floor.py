from __future__ import annotations

"""Score-floor preprocessing for Hough voter masks."""

from dataclasses import dataclass

import numpy as np


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


# Define the compute_score_floor_mask function; its body below performs one named step of the pipeline.
def compute_score_floor_mask(score_matrix: np.ndarray, *, alpha: float) -> ScoreFloorResult:
    """Build the simple Hough mask from `mean + alpha * standard deviation`."""
    # A float array keeps statistics stable even when matrices arrive as lists or integer arrays.
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    matrix_values = np.asarray(score_matrix, dtype=float)
    # Only finite values should influence the floor; invalid cells cannot be meaningful voters.
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    finite_values = matrix_values[np.isfinite(matrix_values)]

    # Check whether finite_values.size == 0; the indented block handles that specific case.
    if finite_values.size == 0:
        # Compute or store score_mean so later code can reuse this named value clearly.
        score_mean = 0.0
        # Compute or store score_standard_deviation so later code can reuse this named value clearly.
        score_standard_deviation = 0.0
    # Define the else field so this data object records that value explicitly.
    else:
        # Use NumPy here because matrix operations should run on compact numeric arrays.
        score_mean = float(np.mean(finite_values))
        # Use NumPy here because matrix operations should run on compact numeric arrays.
        score_standard_deviation = float(np.std(finite_values, ddof=0))

    # Compute or store score_floor so later code can reuse this named value clearly.
    score_floor = float(score_mean + float(alpha) * score_standard_deviation)
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    hough_input_mask = np.asarray(matrix_values >= score_floor, dtype=bool)
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    active_cell_count = int(np.count_nonzero(hough_input_mask))
    # Compute or store total_cell_count so later code can reuse this named value clearly.
    total_cell_count = int(hough_input_mask.size)
    # Compute or store active_fraction so later code can reuse this named value clearly.
    active_fraction = 0.0 if total_cell_count <= 0 else float(active_cell_count / total_cell_count)

    # Return this computed value to the caller so the next pipeline stage can use it.
    return ScoreFloorResult(
        # Pass score_mean into the surrounding call; this supplies the average score of all finite matrix cells, used as the center of the score distribution.
        score_mean=score_mean,
        # Pass score_standard_deviation into the surrounding call; this supplies how widely matrix scores spread around the mean, used to raise the floor above ordinary cells.
        score_standard_deviation=score_standard_deviation,
        # Pass score_floor_alpha into the surrounding call; this supplies the user-selected multiplier that controls how strongly the standard deviation raises the floor.
        score_floor_alpha=float(alpha),
        # Pass score_floor into the surrounding call; this supplies the numeric cutoff a score must meet before it can become a Hough voter.
        score_floor=score_floor,
        # Pass active_cell_count into the surrounding call; this supplies how many matrix cells survived the score floor and became active candidates.
        active_cell_count=active_cell_count,
        # Pass active_fraction into the surrounding call; this supplies the active candidate count divided by the total number of matrix cells.
        active_fraction=active_fraction,
        # Pass hough_input_mask into the surrounding call; this supplies the boolean matrix passed to Hough, where True means this cell can vote for a line.
        hough_input_mask=hough_input_mask,
    )


__all__ = ["ScoreFloorResult", "compute_score_floor_mask"]
