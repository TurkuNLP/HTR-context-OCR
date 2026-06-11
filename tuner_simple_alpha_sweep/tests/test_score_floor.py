from __future__ import annotations

import numpy as np

from tuner_simple_alpha_sweep.matrix_operations.score_floor import compute_score_floor_mask


# Define the test_score_floor_uses_mean_plus_alpha_times_standard_deviation function; its body below performs one named step of the pipeline.
def test_score_floor_uses_mean_plus_alpha_times_standard_deviation() -> None:
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    matrix = np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=float)
    # Compute or store result so later code can reuse this named value clearly.
    result = compute_score_floor_mask(matrix, alpha=1.0)

    # Use NumPy here because matrix operations should run on compact numeric arrays.
    expected_mean = float(np.mean(matrix))
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    expected_standard_deviation = float(np.std(matrix, ddof=0))
    # Compute or store expected_floor so later code can reuse this named value clearly.
    expected_floor = expected_mean + expected_standard_deviation

    # Verify this expected condition during tests so regressions fail clearly.
    assert result.score_mean == expected_mean
    # Verify this expected condition during tests so regressions fail clearly.
    assert result.score_standard_deviation == expected_standard_deviation
    # Verify this expected condition during tests so regressions fail clearly.
    assert result.score_floor == expected_floor
    # Verify this expected condition during tests so regressions fail clearly.
    assert result.active_cell_count == int(np.count_nonzero(matrix >= expected_floor))
    # Verify this expected condition during tests so regressions fail clearly.
    assert np.array_equal(result.hough_input_mask, matrix >= expected_floor)


# Define the test_score_floor_empty_matrix_creates_empty_masks function; its body below performs one named step of the pipeline.
def test_score_floor_empty_matrix_creates_empty_masks() -> None:
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    matrix = np.empty((3, 0), dtype=float)
    # Compute or store result so later code can reuse this named value clearly.
    result = compute_score_floor_mask(matrix, alpha=1.0)

    # Verify this expected condition during tests so regressions fail clearly.
    assert result.score_floor == 0.0
    # Verify this expected condition during tests so regressions fail clearly.
    assert result.active_cell_count == 0
    # Verify this expected condition during tests so regressions fail clearly.
    assert result.hough_input_mask.shape == (3, 0)
