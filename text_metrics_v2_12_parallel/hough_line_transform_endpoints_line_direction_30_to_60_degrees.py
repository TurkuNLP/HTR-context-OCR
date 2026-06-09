"""Detect diagonal Hough line endpoints from score matrices.

This module is the single probabilistic-Hough detector implementation used by
``text_metrics_v2_12_parallel``. The detector always runs with fixed diagonal
normal-angle bands that correspond to line-direction angles in ``[30 deg, 60 deg)``.

The raw Hough detection and the post-Hough merge heuristic are intentionally
separated now:

- this module owns preprocessing, adaptive thresholding, and probabilistic Hough,
- ``hough_postprocessing.greedy_diagonal_segment_merging`` owns the current
  default greedy merge heuristic.

That keeps the default behavior identical while making the merge stage much
clearer to profile, accelerate, and eventually remove in a later version.
"""

from __future__ import annotations

import numpy as np
from skimage.transform import probabilistic_hough_line

from hough_postprocessing.greedy_diagonal_segment_merging import merge_diagonal_segments

DEFAULT_HOUGH_THRESHOLD = 26
DEFAULT_HOUGH_LINE_LENGTH = 10
DEFAULT_HOUGH_LINE_GAP = 15
DEFAULT_HOUGH_SEED = 0
DEFAULT_START_INIT = 2.6

# Fixed theta range used by downstream alignment scripts.
# In Hough space this is the normal angle, not the line-direction angle.
# Keeping line-direction angles in [30 deg, 60 deg) maps to normal-angle
# bands [-60 deg, -30 deg) U (30 deg, 60 deg], which preserves both diagonal
# slants while excluding near-horizontal and near-vertical lines.
DIAGONAL_THETA_DEG = np.r_[
    np.arange(-60, -30, 0.5),
    np.arange(30.5, 60 + 0.5, 0.5),
]
DIAGONAL_THETA_RAD = np.deg2rad(DIAGONAL_THETA_DEG)


# Normalize a score matrix into the dense-style range expected by this detector.
def normalize_for_dense_style(mat: np.ndarray) -> np.ndarray:
    """Normalize a score matrix to the dense-style range expected by Hough.

    The legacy detector transformed matrices into a ``[0, 1)`` range before the
    reciprocal-emphasis step. That normalization is preserved exactly.
    """
    if mat.size == 0:
        return mat

    max_val = float(np.max(mat))
    if max_val <= 1.0:
        normalized_matrix = mat.copy()
    elif max_val <= 100.0:
        normalized_matrix = mat / 100.0
    else:
        normalized_matrix = mat / max_val

    return np.clip(normalized_matrix, 0.0, 0.999999)


# Build the thresholded active-point list needed by the default merge heuristic.
def _active_mask_points_xy(active_mask: np.ndarray) -> list[tuple[int, int]]:
    """Return active thresholded-mask points as ``(x, y)`` integer coordinates."""
    active_rows, active_columns = np.nonzero(active_mask)
    return [
        (int(column_index), int(row_index))
        for row_index, column_index in zip(active_rows, active_columns)
    ]


# Run the dense-style Hough detector with an explicit theta configuration.
def detect_lines_dense_style(
    matrix: np.ndarray,
    *,
    theta: np.ndarray,
    threshold: int = DEFAULT_HOUGH_THRESHOLD,
    line_length: int = DEFAULT_HOUGH_LINE_LENGTH,
    line_gap: int = DEFAULT_HOUGH_LINE_GAP,
    rng: np.random.Generator | None = None,
    start_init: float = DEFAULT_START_INIT,
    compute_merged_hough_segments: bool = True,
) -> dict:
    """Run the dense-style Hough detector with an explicit theta configuration."""
    theta_array = np.asarray(theta, dtype=float)
    if theta_array.ndim != 1 or theta_array.size == 0:
        raise ValueError("theta must be a non-empty 1D array of normal angles")

    normalized_matrix = normalize_for_dense_style(matrix)
    reciprocal_emphasis_matrix = 1.0 / (1.0 - normalized_matrix)

    threshold_start = float(start_init)
    active_mask_dense_enough = False
    target_active_count = 1.4 * matrix.shape[0]
    thresholded_matrix = reciprocal_emphasis_matrix.copy()

    while not active_mask_dense_enough:
        if threshold_start < 0:
            break
        threshold_start -= 0.2
        thresholded_matrix = reciprocal_emphasis_matrix.copy()
        thresholded_matrix[thresholded_matrix < threshold_start] = 0
        active_mask_dense_enough = (thresholded_matrix > 0).sum() > target_active_count

    active_mask = thresholded_matrix > 0

    raw_hough_segments = list(
        probabilistic_hough_line(
            thresholded_matrix,
            threshold=int(threshold),
            line_length=int(line_length),
            line_gap=int(line_gap),
            theta=theta_array,
            rng=rng,
        )
    )
    if bool(compute_merged_hough_segments):
        # The compatibility handoff path still needs the historical greedy
        # postprocessing output, so build the active-point list only then.
        active_mask_points_xy = _active_mask_points_xy(active_mask)
        merged_hough_segments = merge_diagonal_segments(
            raw_hough_segments,
            active_mask,
            active_mask_points_xy,
        )
    else:
        # The raw-Hough default skips merge_diag entirely; callers that select
        # raw handoff do not consume the merged segment list.
        merged_hough_segments = []

    return {
        "threshold_start": threshold_start,
        "mask": thresholded_matrix,
        "raw_lines": raw_hough_segments,
        "selected_lines": raw_hough_segments,
        "merged_lines": merged_hough_segments,
    }


# Run the detector with the pipeline's fixed diagonal theta bands.
def detect_lines_dense_style_diagonal_fixed_theta(
    matrix: np.ndarray,
    *,
    seed: int,
    threshold: int = DEFAULT_HOUGH_THRESHOLD,
    line_length: int = DEFAULT_HOUGH_LINE_LENGTH,
    line_gap: int = DEFAULT_HOUGH_LINE_GAP,
    start_init: float = DEFAULT_START_INIT,
    compute_merged_hough_segments: bool = True,
) -> dict:
    """Run the Hough detector with the pipeline's fixed diagonal theta bands."""
    return detect_lines_dense_style(
        matrix,
        threshold=threshold,
        line_length=line_length,
        line_gap=line_gap,
        theta=DIAGONAL_THETA_RAD,
        rng=np.random.default_rng(int(seed)),
        start_init=start_init,
        compute_merged_hough_segments=bool(compute_merged_hough_segments),
    )
