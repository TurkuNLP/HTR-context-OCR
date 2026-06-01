from __future__ import annotations

"""Equivalence tests for optional Cython line-sampling helpers."""

import math

import numpy as np
import pytest

from tuner_parallel_v2_1.alignment.hough_segment_endpoint_records import (
    line_y_at_prediction_column,
    mean_line_support_from_score_matrix,
)
from tuner_parallel_v2_1.cython_accel.optional_filtering import (
    cython_line_sampling_available,
    mean_line_support_from_endpoints,
    sample_line_path,
)
import tuner_parallel_v2_1.filtering.filter_cython_accelerators as filtering_accelerators
import tuner_parallel_v2_1.filtering.line_filtering_v2_1_IoU_fast as filtering_module


def _reference_sample_line_path(matrix: np.ndarray, line: dict) -> dict | None:
    """Return the pure-Python path sample used as the exact reference."""
    if matrix.size == 0:
        return None

    reference_row_count, prediction_column_count = matrix.shape
    x_min = max(0, int(math.floor(min(line["x0"], line["x1"]))))
    x_max = min(prediction_column_count - 1, int(math.ceil(max(line["x0"], line["x1"]))))
    if x_max < x_min:
        return None

    x_to_y: dict[int, int] = {}
    x_to_score: dict[int, float] = {}
    reference_segments: set[int] = set()
    previous_reference_row: int | None = None

    for prediction_column in range(x_min, x_max + 1):
        sampled_reference_row = int(
            np.clip(
                round(line_y_at_prediction_column(line, prediction_column)),
                0,
                reference_row_count - 1,
            )
        )
        x_to_y[int(prediction_column)] = int(sampled_reference_row)
        x_to_score[int(prediction_column)] = float(matrix[sampled_reference_row, prediction_column])
        reference_segments.add(int(sampled_reference_row))

        if previous_reference_row is not None:
            row_start, row_end = sorted((previous_reference_row, sampled_reference_row))
            for bridged_reference_row in range(row_start, row_end + 1):
                reference_segments.add(int(bridged_reference_row))
        previous_reference_row = sampled_reference_row

    total_score = float(sum(x_to_score.values()))
    return {
        "x_to_y": x_to_y,
        "x_to_score": x_to_score,
        "pred_segments": set(x_to_y),
        "ref_segments": reference_segments,
        "sampled_reference_rows": [int(x_to_y[x]) for x in sorted(x_to_y)],
        "total_score": total_score,
        "mean_score": float(total_score / len(x_to_score)) if x_to_score else 0.0,
        "pred_min": min(x_to_y) if x_to_y else 0,
        "pred_max": max(x_to_y) if x_to_y else -1,
        "ref_min": min(reference_segments) if reference_segments else 0,
        "ref_max": max(reference_segments) if reference_segments else -1,
    }


def _assert_numeric_lines_match(left_lines: list[dict], right_lines: list[dict]) -> None:
    """Compare final line dictionaries while tolerating harmless float noise."""
    assert len(left_lines) == len(right_lines)
    for left_line, right_line in zip(left_lines, right_lines):
        assert set(left_line) == set(right_line)
        for field_name in sorted(left_line):
            left_value = left_line[field_name]
            right_value = right_line[field_name]
            if isinstance(left_value, float) or isinstance(right_value, float):
                assert float(left_value) == pytest.approx(float(right_value), abs=1e-12, rel=0.0)
            else:
                assert left_value == right_value


def test_cython_line_path_sampling_matches_reference_python_sampling() -> None:
    """The Cython line sampler must reproduce the Python path exactly."""
    if not cython_line_sampling_available():
        pytest.skip("compiled line-sampling helpers are not available")

    matrix = np.ascontiguousarray(np.arange(80, dtype=float).reshape(8, 10) / 10.0)
    line = {"x0": -0.25, "y0": 0.50, "x1": 9.25, "y1": 7.50}

    accelerated_sample = sample_line_path(
        matrix,
        x0=float(line["x0"]),
        y0=float(line["y0"]),
        x1=float(line["x1"]),
        y1=float(line["y1"]),
    )
    reference_sample = _reference_sample_line_path(matrix, line)

    assert accelerated_sample is not None
    assert reference_sample is not None
    assert accelerated_sample["x_to_y"] == reference_sample["x_to_y"]
    assert accelerated_sample["x_to_score"] == reference_sample["x_to_score"]
    assert accelerated_sample["pred_segments"] == reference_sample["pred_segments"]
    assert accelerated_sample["ref_segments"] == reference_sample["ref_segments"]
    assert accelerated_sample["sampled_reference_rows"] == reference_sample["sampled_reference_rows"]
    assert accelerated_sample["total_score"] == pytest.approx(reference_sample["total_score"], abs=1e-12)
    assert accelerated_sample["mean_score"] == pytest.approx(reference_sample["mean_score"], abs=1e-12)


def test_cython_mean_support_matches_reference_python_sampling() -> None:
    """The Cython support sampler must keep the exact mean-support rule."""
    if not cython_line_sampling_available():
        pytest.skip("compiled line-sampling helpers are not available")

    matrix = np.ascontiguousarray(np.arange(132, dtype=float).reshape(11, 12) / 7.0)
    line = {"x0": 1.25, "y0": 9.50, "x1": 10.75, "y1": 0.50}

    accelerated_support = mean_line_support_from_endpoints(
        matrix,
        x0=float(line["x0"]),
        y0=float(line["y0"]),
        x1=float(line["x1"]),
        y1=float(line["y1"]),
    )
    reference_support = mean_line_support_from_score_matrix(matrix, line)

    assert accelerated_support is not None
    assert accelerated_support == pytest.approx(reference_support, abs=1e-12, rel=0.0)


def test_used_coverage_indices_helper_matches_old_repeated_scan_rule() -> None:
    """The one-pass finalization helper must keep the old kept-id order."""
    mapped_line_id = np.asarray(
        [-1, 2, 0, 2, 5, 1, -1, 4, 2, 99, 0],
        dtype=int,
    )
    coverage_count = 6

    expected_indices = [
        coverage_index
        for coverage_index in range(coverage_count)
        if np.any(mapped_line_id == coverage_index)
    ]
    actual_indices = filtering_module._used_coverage_indices_from_assignment(
        mapped_line_id,
        coverage_count=coverage_count,
    )

    assert actual_indices == expected_indices


def test_weighted_degree_one_fit_matches_numpy_polyfit_rule() -> None:
    """The faster line fit must keep the old weighted ``np.polyfit`` semantics."""
    sampled_prediction_columns = np.asarray([0.0, 1.0, 3.0, 4.0, 7.0, 9.0], dtype=float)
    sampled_reference_rows = np.asarray([1.0, 2.2, 4.1, 5.3, 8.0, 9.7], dtype=float)
    sampled_weights = np.asarray([0.25, 1.0, 0.75, 1.5, 0.5, 2.0], dtype=float)

    expected_slope, expected_intercept = np.polyfit(
        sampled_prediction_columns,
        sampled_reference_rows,
        deg=1,
        w=sampled_weights,
    )
    actual_fit = filtering_module._weighted_degree_one_fit(
        sampled_prediction_columns,
        sampled_reference_rows,
        sampled_weights,
    )

    assert actual_fit is not None
    actual_slope, actual_intercept = actual_fit
    assert actual_slope == pytest.approx(float(expected_slope), abs=1e-12, rel=0.0)
    assert actual_intercept == pytest.approx(float(expected_intercept), abs=1e-12, rel=0.0)


def test_filter_output_matches_when_line_sampling_acceleration_is_disabled() -> None:
    """Full filter output must match the Python path on a small synthetic case."""
    matrix = np.ascontiguousarray(np.eye(20, dtype=float) * 3.0)
    matrix += np.ascontiguousarray(np.eye(20, k=1, dtype=float) * 2.0)
    matrix += 0.05
    mask_bool = matrix > 1.0
    lines = [
        {"x0": 0.0, "y0": 0.0, "x1": 19.0, "y1": 19.0, "score": 10.0},
        {"x0": 0.0, "y0": 1.0, "x1": 18.0, "y1": 19.0, "score": 8.0},
        {"x0": 3.0, "y0": 3.0, "x1": 17.0, "y1": 17.0, "score": 7.0},
    ]

    original_final_assignment = filtering_accelerators.accelerated_compute_final_assignment
    original_sample_line_path = filtering_accelerators.accelerated_sample_line_path
    original_mean_support = filtering_accelerators.accelerated_mean_line_support_from_endpoints
    try:
        filtering_accelerators.accelerated_compute_final_assignment = None
        filtering_accelerators.accelerated_sample_line_path = None
        filtering_accelerators.accelerated_mean_line_support_from_endpoints = None
        reference_lines, reference_assignment = filtering_module.filter_lines_for_alignment_by_ownership(
            lines,
            matrix,
            mask_bool,
            abs_min_len=1.0,
            min_iou_threshold=0.0,
        )
    finally:
        filtering_accelerators.accelerated_compute_final_assignment = original_final_assignment
        filtering_accelerators.accelerated_sample_line_path = original_sample_line_path
        filtering_accelerators.accelerated_mean_line_support_from_endpoints = original_mean_support

    accelerated_lines, accelerated_assignment = filtering_module.filter_lines_for_alignment_by_ownership(
        lines,
        matrix,
        mask_bool,
        abs_min_len=1.0,
        min_iou_threshold=0.0,
    )

    _assert_numeric_lines_match(accelerated_lines, reference_lines)
    np.testing.assert_allclose(
        accelerated_assignment["mapped_y"],
        reference_assignment["mapped_y"],
        equal_nan=True,
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(
        accelerated_assignment["mapped_line_id"],
        reference_assignment["mapped_line_id"],
    )
