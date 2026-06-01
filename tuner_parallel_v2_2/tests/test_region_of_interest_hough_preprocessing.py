from __future__ import annotations

import numpy as np
import pytest

from tuner_parallel_v2_2.hough_preprocessing import (
    CONNECTED_COMPONENT_BACKEND_CYTHON,
    CONNECTED_COMPONENT_BACKEND_SCIPY,
    HoughPreprocessingConfig,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
    build_region_of_interest_hough_context,
)
from tuner_parallel_v2_2.hough_preprocessing.matrix_statistics import (
    scaled_median_absolute_deviation,
    summarize_score_matrix,
)


def test_region_of_interest_context_preserves_matrix_shape() -> None:
    matrix = np.eye(6, dtype=float) * 90.0
    context = build_region_of_interest_hough_context(matrix)

    assert context["hough_image"].shape == matrix.shape
    assert context["hough_mask_bool"].shape == matrix.shape
    assert context["region_of_interest_mask_bool"].shape == matrix.shape
    assert context["strong_match_mask_bool"].shape == matrix.shape


def test_repeated_high_scores_in_one_reference_row_are_preserved() -> None:
    matrix = np.zeros((4, 5), dtype=float)
    matrix[1, 1] = 91.0
    matrix[1, 2] = 88.0

    config = HoughPreprocessingConfig(
        minimum_score_floor=20.0,
        median_absolute_deviation_multiplier=0.0,
        near_peak_ratio=0.90,
        minimum_active_cells=2,
        minimum_active_rows=1,
        minimum_active_columns=2,
        minimum_y_span=1,
    )
    context = build_region_of_interest_hough_context(matrix, config=config)

    strong_match_mask = context["strong_match_mask_bool"]
    assert bool(strong_match_mask[1, 1])
    assert bool(strong_match_mask[1, 2])


def test_weak_row_winners_do_not_survive_the_absolute_score_floor() -> None:
    matrix = np.zeros((4, 4), dtype=float)
    matrix[2, 2] = 12.0

    config = HoughPreprocessingConfig(minimum_score_floor=20.0, median_absolute_deviation_multiplier=0.0)
    context = build_region_of_interest_hough_context(matrix, config=config)

    assert not bool(context["hough_preprocessing_accepted"])
    assert context["hough_preprocessing_rejection_reason"] == "no_strong_match_evidence"
    assert int(np.count_nonzero(context["hough_mask_bool"])) == 0


def test_dense_full_matrix_evidence_is_rejected_as_ambiguous() -> None:
    matrix = np.full((10, 10), 50.0, dtype=float)
    config = HoughPreprocessingConfig(
        minimum_score_floor=20.0,
        median_absolute_deviation_multiplier=0.0,
        maximum_active_fraction=0.08,
    )
    context = build_region_of_interest_hough_context(matrix, config=config)

    assert not bool(context["hough_preprocessing_accepted"])
    assert context["hough_preprocessing_rejection_reason"] == "ambiguous_or_too_dense"
    assert context["hough_preprocessing_summary"]["active_fraction"] == pytest.approx(1.0)


def test_dilation_does_not_add_weak_cells_to_hough_input() -> None:
    matrix = np.zeros((5, 5), dtype=float)
    matrix[2, 2] = 90.0
    matrix[2, 3] = 89.0

    config = HoughPreprocessingConfig(
        minimum_score_floor=20.0,
        median_absolute_deviation_multiplier=0.0,
        region_dilation_radius=2,
        minimum_active_cells=2,
        minimum_active_rows=1,
        minimum_active_columns=2,
        minimum_y_span=1,
        maximum_active_fraction=1.0,
    )
    context = build_region_of_interest_hough_context(matrix, config=config)

    assert int(np.count_nonzero(context["region_of_interest_mask_bool"])) > 2
    assert int(np.count_nonzero(context["hough_mask_bool"])) == 2


def test_manual_median_absolute_deviation_matches_scipy_normal_scale() -> None:
    pytest.importorskip("scipy.stats")
    values = np.asarray([1.0, 2.0, 2.0, 4.0, 9.0], dtype=float)
    median_value = float(np.median(values))

    manual_value = scaled_median_absolute_deviation(
        values,
        median_value=median_value,
        backend=MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    )
    scipy_value = scaled_median_absolute_deviation(
        values,
        median_value=median_value,
        backend=MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
    )

    assert manual_value == pytest.approx(scipy_value, rel=1e-12, abs=1e-12)


def test_score_matrix_statistics_records_selected_deviation_backend() -> None:
    matrix = np.asarray([[0.0, 10.0], [20.0, 30.0]], dtype=float)
    stats = summarize_score_matrix(
        matrix,
        median_absolute_deviation_backend=MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    )

    assert stats.finite_value_count == 4
    assert stats.score_median == pytest.approx(15.0)
    assert stats.score_maximum == pytest.approx(30.0)
    assert stats.median_absolute_deviation_backend == MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY


def test_default_connected_component_backend_uses_cython_when_extension_is_built() -> None:
    pytest.importorskip("tuner_parallel_v2_2.cython_accel.roi_preprocessing_core")
    matrix = np.eye(8, dtype=float) * 90.0

    context = build_region_of_interest_hough_context(matrix)
    summary = context["hough_preprocessing_summary"]

    assert summary["connected_component_backend_requested"] == CONNECTED_COMPONENT_BACKEND_CYTHON
    assert summary["connected_component_backend_used"] == CONNECTED_COMPONENT_BACKEND_CYTHON


def test_scipy_connected_component_backend_is_available_as_explicit_option() -> None:
    pytest.importorskip("scipy.ndimage")
    matrix = np.eye(8, dtype=float) * 90.0
    config = HoughPreprocessingConfig(connected_component_backend=CONNECTED_COMPONENT_BACKEND_SCIPY)

    context = build_region_of_interest_hough_context(matrix, config=config)
    summary = context["hough_preprocessing_summary"]

    assert summary["connected_component_backend_requested"] == CONNECTED_COMPONENT_BACKEND_SCIPY
    assert summary["connected_component_backend_used"] == CONNECTED_COMPONENT_BACKEND_SCIPY

def test_cython_and_scipy_connected_component_backends_build_the_same_hough_input() -> None:
    pytest.importorskip("scipy.ndimage")
    pytest.importorskip("tuner_parallel_v2_2.cython_accel.roi_preprocessing_core")

    matrix = np.zeros((8, 9), dtype=float)
    matrix[1, 1] = 94.0
    matrix[1, 2] = 91.0
    matrix[2, 2] = 95.0
    matrix[5, 6] = 89.0
    matrix[6, 7] = 92.0

    common_settings = dict(
        minimum_score_floor=20.0,
        median_absolute_deviation_multiplier=0.0,
        near_peak_ratio=0.90,
        minimum_active_cells=1,
        minimum_active_rows=1,
        minimum_active_columns=1,
        minimum_x_span=1,
        minimum_y_span=1,
        maximum_active_fraction=1.0,
    )
    cython_context = build_region_of_interest_hough_context(
        matrix,
        config=HoughPreprocessingConfig(
            connected_component_backend=CONNECTED_COMPONENT_BACKEND_CYTHON,
            **common_settings,
        ),
    )
    scipy_context = build_region_of_interest_hough_context(
        matrix,
        config=HoughPreprocessingConfig(
            connected_component_backend=CONNECTED_COMPONENT_BACKEND_SCIPY,
            **common_settings,
        ),
    )

    assert np.array_equal(cython_context["hough_mask_bool"], scipy_context["hough_mask_bool"])
    assert np.array_equal(
        cython_context["region_of_interest_mask_bool"],
        scipy_context["region_of_interest_mask_bool"],
    )

