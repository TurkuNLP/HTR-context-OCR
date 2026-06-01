from __future__ import annotations

"""Regression tests for v2_2 Hough context precomputation."""

import numpy as np

from tuner_parallel_v2_2.alignment.line_alignment_pipeline_fast import precompute_hough_context
from tuner_parallel_v2_2.hough_preprocessing import HoughPreprocessingConfig


def test_hough_precompute_uses_binary_region_of_interest_mask() -> None:
    """A compact strong diagonal should become a sparse binary Hough input."""
    matrix = np.zeros((5, 20), dtype=float)
    for row_index in range(matrix.shape[0]):
        matrix[row_index, row_index] = 90.0

    config = HoughPreprocessingConfig(
        minimum_score_floor=20.0,
        median_absolute_deviation_multiplier=0.0,
        maximum_active_fraction=0.50,
    )
    hough_context = precompute_hough_context(matrix, config=config, keep_debug_arrays=True)
    hough_image = hough_context["hough_image"]

    assert hough_context["preprocessing_mode"] == "region_of_interest"
    assert bool(hough_context["hough_preprocessing_accepted"])
    assert hough_image.shape == matrix.shape
    assert set(np.unique(hough_image)).issubset({0, 1})
    assert np.count_nonzero(hough_image) == 5
    assert np.count_nonzero(hough_image) < hough_image.size
    assert "debug_strong_match_mask" in hough_context
    assert "debug_region_of_interest_mask" in hough_context
