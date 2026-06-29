from __future__ import annotations

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.matrix_operations.score_floor import (
    compute_minimum_levenshtein_mask,
    compute_score_floor_mask,
    compute_score_floor_statistics,
    convert_minimum_levenshtein_to_matrix_threshold,
)


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



def test_minimum_levenshtein_threshold_accepts_unit_value_for_percent_matrix() -> None:
    matrix = np.asarray([[10.0, 29.9], [30.0, 80.0]], dtype=float)
    statistics = compute_score_floor_statistics(matrix)

    result = compute_minimum_levenshtein_mask(matrix, minimum_levenshtein=0.30, statistics=statistics)

    assert result.score_floor == 30.0
    assert result.score_floor_alpha == 0.0
    assert result.active_cell_count == 2
    assert np.array_equal(result.hough_input_mask, matrix >= 30.0)


def test_minimum_levenshtein_threshold_accepts_percent_value_for_percent_matrix() -> None:
    matrix = np.asarray([[10.0, 29.9], [30.0, 80.0]], dtype=float)

    assert convert_minimum_levenshtein_to_matrix_threshold(matrix, minimum_levenshtein=30.0) == 30.0


def test_minimum_levenshtein_threshold_accepts_percent_value_for_unit_matrix() -> None:
    matrix = np.asarray([[0.10, 0.299], [0.30, 0.80]], dtype=float)
    statistics = compute_score_floor_statistics(matrix)

    result = compute_minimum_levenshtein_mask(matrix, minimum_levenshtein=30.0, statistics=statistics)

    assert result.score_floor == 0.30
    assert result.active_cell_count == 2
    assert np.array_equal(result.hough_input_mask, matrix >= 0.30)


def test_minimum_levenshtein_threshold_accepts_unit_value_for_unit_matrix() -> None:
    matrix = np.asarray([[0.10, 0.299], [0.30, 0.80]], dtype=float)

    assert convert_minimum_levenshtein_to_matrix_threshold(matrix, minimum_levenshtein=0.30) == 0.30
