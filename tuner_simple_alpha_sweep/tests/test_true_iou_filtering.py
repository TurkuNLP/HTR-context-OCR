from __future__ import annotations

import numpy as np

from tuner_simple_alpha_sweep.probabilistic_hough.hough_detection import filter_lines_by_column_ownership
from tuner_simple_alpha_sweep.filtering.filter_overlap_merging import coverages_merge_candidate
from tuner_simple_alpha_sweep.filtering.filter_candidate_coverages import build_line_coverage
from tuner_simple_alpha_sweep.alignment.hough_segment_endpoint_records import line_records_from_raw_hough_segments


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
    assert coverages_merge_candidate(coverage_a, coverage_b, min_iou_threshold=0.035)
    assert not coverages_merge_candidate(coverage_a, coverage_b, min_iou_threshold=0.4)
