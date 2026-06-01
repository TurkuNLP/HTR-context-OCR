from __future__ import annotations

"""Tests for optional true-IoU filter profiling."""

import numpy as np

from tuner_parallel_v2_2.filtering.line_filtering_v2_1_IoU_fast import (
    FILTER_PROFILE_DEFAULTS,
    filter_lines_for_alignment_by_ownership,
)


def _small_matrix() -> np.ndarray:
    """Return a tiny matrix with two visible diagonal guides."""
    matrix = np.zeros((8, 8), dtype=float)
    for index in range(8):
        matrix[index, index] = 90.0
    for index in range(4):
        matrix[index + 3, index] = 70.0
    return matrix


def _candidate_lines() -> list[dict]:
    """Return overlapping and non-overlapping candidates for profile coverage."""
    return [
        {"x0": 0.0, "y0": 0.0, "x1": 7.0, "y1": 7.0, "length": 9.9, "support": 90.0, "score": 90.0},
        {"x0": 0.0, "y0": 0.0, "x1": 6.0, "y1": 6.0, "length": 8.5, "support": 80.0, "score": 80.0},
        {"x0": 0.0, "y0": 3.0, "x1": 3.0, "y1": 6.0, "length": 4.3, "support": 70.0, "score": 70.0},
    ]


def _assert_same_filter_outputs(actual, expected) -> None:
    """Compare filter outputs without relying on object identity."""
    actual_lines, actual_assignment = actual
    expected_lines, expected_assignment = expected

    assert len(actual_lines) == len(expected_lines)
    for actual_line, expected_line in zip(actual_lines, expected_lines):
        assert actual_line.keys() == expected_line.keys()
        for key, expected_value in expected_line.items():
            actual_value = actual_line[key]
            if isinstance(expected_value, float):
                assert float(actual_value) == float(expected_value)
            else:
                assert actual_value == expected_value

    np.testing.assert_array_equal(actual_assignment["mapped_line_id"], expected_assignment["mapped_line_id"])
    np.testing.assert_allclose(
        actual_assignment["mapped_y"],
        expected_assignment["mapped_y"],
        equal_nan=True,
        rtol=0.0,
        atol=0.0,
    )


def test_filter_profile_does_not_change_filter_outputs() -> None:
    """Supplying a profile dict must leave final lines and assignments unchanged."""
    matrix = _small_matrix()
    mask_bool = matrix > 0
    lines = _candidate_lines()

    expected = filter_lines_for_alignment_by_ownership(lines, matrix, mask_bool)
    profile: dict = {}
    actual = filter_lines_for_alignment_by_ownership(lines, matrix, mask_bool, profile=profile)

    _assert_same_filter_outputs(actual, expected)
    for field_name in FILTER_PROFILE_DEFAULTS:
        assert field_name in profile
    assert profile["filter_input_line_count"] == len(lines)
    assert profile["filter_prepared_candidate_count"] >= 1
    assert profile["filter_candidate_coverage_count"] >= 1
    assert profile["filter_final_line_count"] == len(actual[0])
    assert profile["filter_total_profiled_seconds"] >= 0.0


def test_filter_profile_records_fallback_candidate_use() -> None:
    """The profile should show when coarse gates fall back to the best raw line."""
    matrix = np.zeros((4, 4), dtype=float)
    mask_bool = matrix > 0
    lines = [
        {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0, "length": 1.4, "support": 0.0, "score": 1.0},
        {"x0": 2.0, "y0": 2.0, "x1": 3.0, "y1": 3.0, "length": 1.4, "support": 0.0, "score": 2.0},
    ]

    profile: dict = {}
    filter_lines_for_alignment_by_ownership(
        lines,
        matrix,
        mask_bool,
        abs_min_len=10.0,
        profile=profile,
    )

    assert profile["filter_input_line_count"] == 2
    assert profile["filter_fallback_candidate_used"] == 1
    assert profile["filter_prepared_candidate_count"] == 1
