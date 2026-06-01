from __future__ import annotations

"""Regression tests for dense-style Hough context precomputation."""

import numpy as np
import pytest

from tuner_parallel_v2_1.alignment.line_alignment_pipeline_fast import precompute_hough_context


def test_hough_precompute_stops_on_intensity_sum_not_active_cell_count() -> None:
    """A compact strong diagonal should stop before weak background fills the image."""
    matrix = np.zeros((5, 20), dtype=float)
    for row in range(matrix.shape[0]):
        matrix[row, row] = 90.0

    hough_ctx = precompute_hough_context(matrix, start_init=8.0, keep_debug_arrays=True)
    thresholded_hough_image = hough_ctx["hough_image"]

    assert hough_ctx["hough_stop_criteria"] == pytest.approx(1.4 * 20)
    assert hough_ctx["threshold_start"] == pytest.approx(7.8)

    # Each 90/100 diagonal score becomes 1 / (1 - 0.9) = 10.  Five such cells
    # give enough total Hough intensity even though the active-cell count is
    # far below the stopping criterion.
    assert hough_ctx["hough_image_intensity_sum"] == pytest.approx(50.0)
    assert np.count_nonzero(thresholded_hough_image) == 5
    assert np.count_nonzero(thresholded_hough_image) < hough_ctx["hough_stop_criteria"]
    assert np.count_nonzero(thresholded_hough_image) < thresholded_hough_image.size

    # The clearer debug names are available; old aliases remain only for
    # compatibility with exploratory scripts/notebooks.
    assert "normalized_score_matrix" in hough_ctx
    assert "reciprocal_emphasis_image" in hough_ctx
    assert hough_ctx["norm"] is hough_ctx["normalized_score_matrix"]
    assert hough_ctx["test"] is hough_ctx["reciprocal_emphasis_image"]
