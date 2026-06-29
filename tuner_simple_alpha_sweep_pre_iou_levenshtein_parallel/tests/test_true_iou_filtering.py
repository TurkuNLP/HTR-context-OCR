from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.probabilistic_hough import hough_detection as hough_detection_module
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.probabilistic_hough.hough_detection import filter_lines_by_column_ownership
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.filtering.filter_overlap_merging import (
    coverages_have_enough_true_iou,
    coverages_merge_candidate,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.filtering.filter_candidate_coverages import build_line_coverage, coverage_from_path
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.alignment.hough_segment_endpoint_records import line_records_from_raw_hough_segments


def diagonal_score_matrix(size: int = 20) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=float)
    for index in range(size):
        matrix[index, index] = 100.0
    return matrix


def test_align_min_iou_threshold_controls_true_iou_merging() -> None:
    matrix = diagonal_score_matrix(20)
    detection_result = {
        "candidate_segments": [
            ((0.0, 0.0), (12.0, 12.0)),
            ((7.0, 7.0), (19.0, 19.0)),
        ],
        "mask_bool": np.ones((20, 20), dtype=bool),
    }

    permissive_result = filter_lines_by_column_ownership(
        score_matrix=matrix,
        detection_result=detection_result,
        hough_input_mask=np.ones((20, 20), dtype=bool),
        align_min_iou_threshold=0.035,
    )
    strict_result = filter_lines_by_column_ownership(
        score_matrix=matrix,
        detection_result=detection_result,
        hough_input_mask=np.ones((20, 20), dtype=bool),
        align_min_iou_threshold=0.4,
    )

    assert len(permissive_result["lines_used"]) == 1
    assert len(strict_result["lines_used"]) == 2
    assert permissive_result["filtered_by"] == "true_iou_v2_2"
    assert strict_result["filtered_by"] == "true_iou_v2_2"


def test_true_iou_decision_uses_minimum_of_prediction_and_reference_iou() -> None:
    matrix = diagonal_score_matrix(20)
    raw_lines = line_records_from_raw_hough_segments(
        matrix,
        [
            ((0.0, 0.0), (12.0, 12.0)),
            ((7.0, 7.0), (19.0, 19.0)),
        ],
    )
    coverage_a = build_line_coverage(raw_lines[0], matrix)
    coverage_b = build_line_coverage(raw_lines[1], matrix)

    assert coverage_a is not None
    assert coverage_b is not None
    assert coverages_merge_candidate(coverage_a, coverage_b, matrix=matrix, min_iou_threshold=0.035)
    assert not coverages_merge_candidate(coverage_a, coverage_b, matrix=matrix, min_iou_threshold=0.4)


def test_final_line_sorting_remaps_column_assignment_ids() -> None:
    from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.filtering.filter_final_assignment import sort_final_lines_and_remap_assignment

    unsorted_lines = [
        {"x0": 12.0, "y0": 1.0, "x1": 17.0, "y1": 6.0, "anchor_y": 3.5},
        {"x0": 0.0, "y0": 0.0, "x1": 7.0, "y1": 7.0, "anchor_y": 0.0},
    ]
    assignment = {
        "mapped_y": np.asarray([0.0, 1.0, 2.0, 3.0, np.nan], dtype=float),
        "mapped_line_id": np.asarray([1, 1, 0, 0, -1], dtype=int),
    }

    sorted_lines, remapped_assignment = sort_final_lines_and_remap_assignment(unsorted_lines, assignment)

    assert sorted_lines[0]["x0"] == 0.0
    assert sorted_lines[1]["x0"] == 12.0
    np.testing.assert_array_equal(
        remapped_assignment["mapped_line_id"],
        np.asarray([0, 0, 1, 1, -1], dtype=int),
    )


def synthetic_coverage(matrix: np.ndarray, *, slope: float, intercept: float, high_score_on_right: bool) -> dict:
    x_to_y = {int(x): int(round((float(slope) * x) + float(intercept))) for x in range(0, 101)}
    x_to_score = {
        int(x): (100.0 if (int(x) >= 50) == bool(high_score_on_right) else 1.0)
        for x in range(0, 101)
    }
    return coverage_from_path(
        x_to_y=x_to_y,
        x_to_score=x_to_score,
        matrix=matrix,
        fallback_line={
            "x0": 0.0,
            "y0": float(x_to_y[0]),
            "x1": 100.0,
            "y1": float(x_to_y[100]),
        },
        source_raw_line_ids=[],
    )


def test_true_iou_refuses_merge_when_fitted_pair_leaves_hough_angle_range() -> None:
    matrix = np.zeros((400, 120), dtype=float)
    lower_valid_line = synthetic_coverage(matrix, slope=1.7, intercept=0.0, high_score_on_right=False)
    upper_valid_line = synthetic_coverage(matrix, slope=1.7, intercept=155.0, high_score_on_right=True)

    assert coverages_have_enough_true_iou(lower_valid_line, upper_valid_line, min_iou_threshold=0.035)
    assert not coverages_merge_candidate(
        lower_valid_line,
        upper_valid_line,
        matrix=matrix,
        min_iou_threshold=0.035,
    )


def test_pre_iou_levenshtein_filter_does_not_overwrite_pure_raw_hough_lines(monkeypatch) -> None:
    pure_hough_segments = [((0.0, 0.0), (5.0, 5.0)), ((10.0, 10.0), (15.0, 15.0))]
    filtered_segments = [pure_hough_segments[0]]

    def fake_detect_falling_diagonal_hough_lines(**kwargs):
        return {
            "raw_lines": list(pure_hough_segments),
            "candidate_segments": list(pure_hough_segments),
            "mask": np.ones((20, 20), dtype=bool),
        }

    def fake_filter_raw_hough_segments_by_line_levenshtein(**kwargs):
        return SimpleNamespace(
            filtered_segments=list(filtered_segments),
            filter_enabled=True,
            input_line_count=2,
            surviving_line_count=1,
            removed_line_count=1,
            threshold=0.5,
            score_minimum=0.4,
            score_maximum=0.9,
            score_mean=0.65,
            seconds=0.0,
            line_score_records=[],
        )

    def fake_filter_lines_by_column_ownership(**kwargs):
        detection_result = kwargs["detection_result"]
        assert detection_result["raw_lines"] == pure_hough_segments
        assert detection_result["candidate_segments"] == filtered_segments
        return {
            "lines_for_filtering": [],
            "lines_used": [],
            "column_assignment": {},
        }

    monkeypatch.setattr(hough_detection_module, "detect_falling_diagonal_hough_lines", fake_detect_falling_diagonal_hough_lines)
    monkeypatch.setattr(hough_detection_module, "filter_raw_hough_segments_by_line_levenshtein", fake_filter_raw_hough_segments_by_line_levenshtein)
    monkeypatch.setattr(hough_detection_module, "filter_lines_by_column_ownership", fake_filter_lines_by_column_ownership)

    payload = hough_detection_module.run_probabilistic_hough_and_filter(
        score_matrix=np.ones((20, 20), dtype=float),
        hough_input_mask=np.ones((20, 20), dtype=bool),
        score_floor=0.0,
        hough_threshold=3,
        hough_line_length=3,
        hough_line_gap=4,
        hough_seed=0,
        align_min_iou_threshold=0.035,
        reference_windows=["reference window"],
        prediction_windows=["prediction window"],
        reference_window_count=20,
        minimum_raw_line_nls=0.5,
    )

    assert payload.detection_result["raw_lines"] == pure_hough_segments
    assert payload.detection_result["raw_lines_before_pre_iou_levenshtein"] == pure_hough_segments
    assert payload.detection_result["candidate_segments"] == filtered_segments
