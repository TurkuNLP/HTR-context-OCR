from __future__ import annotations

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein.cython_accel.optional_ownership import (
    assign_columns_to_candidate_lines_with_optional_accelerator,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.probabilistic_hough.hough_detection import (
    assign_columns_to_candidate_lines_with_python_reference,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.line_text_similarity import LineTextFilterResult, WeightedAlongLinesResult
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.scoring_pipeline import score_document_alignment
from tuner_simple_alpha_sweep_pre_iou_levenshtein.probabilistic_hough.hough_detection import HoughFilteredPayload


def test_cython_ownership_matches_python_reference_when_extension_is_available():
    score_matrix = np.array(
        [
            [92.0, 10.0, 10.0, 10.0],
            [10.0, 91.0, 20.0, 10.0],
            [10.0, 20.0, 90.0, 30.0],
            [10.0, 10.0, 30.0, 89.0],
        ],
        dtype=float,
    )
    voter_mask = score_matrix >= 80.0
    candidate_lines = [
        {"x0": 0.0, "y0": 0.0, "x1": 3.0, "y1": 3.0},
        {"x0": 0.0, "y0": 1.0, "x1": 3.0, "y1": 3.0},
    ]

    python_result = assign_columns_to_candidate_lines_with_python_reference(
        score_matrix=score_matrix,
        voter_mask=voter_mask,
        candidate_lines=candidate_lines,
    )
    cython_result = assign_columns_to_candidate_lines_with_optional_accelerator(
        score_matrix=score_matrix,
        voter_mask=voter_mask,
        candidate_lines=candidate_lines,
    )

    if cython_result is None:
        return

    assert np.array_equal(cython_result["mapped_candidate_id"], python_result["mapped_candidate_id"])
    assert np.allclose(cython_result["mapped_y"], python_result["mapped_y"], equal_nan=True)
    assert np.array_equal(cython_result["owned_counts"], python_result["owned_counts"])


def test_invalid_coverage_keeps_final_ref_to_pred_lines_available():
    final_line = {"x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 2.0, "length": 3.0}
    empty_assignment = {"mapped_y": np.array([], dtype=float), "mapped_line_id": np.array([], dtype=int)}
    ref_to_pred_payload = HoughFilteredPayload(
        hough_context={"mask": np.ones((3, 3), dtype=bool)},
        detection_result={"raw_lines": []},
        filtered_result={"lines_used": [final_line], "column_assignment": empty_assignment},
        raw_line_count=1,
        candidate_line_count=1,
        used_line_count=1,
        detect_seconds=0.0,
        filter_seconds=0.0,
    )
    ref_to_ref_payload = HoughFilteredPayload(
        hough_context={"mask": np.ones((3, 3), dtype=bool)},
        detection_result={"raw_lines": []},
        filtered_result={"lines_used": [final_line, final_line, final_line], "column_assignment": empty_assignment},
        raw_line_count=3,
        candidate_line_count=3,
        used_line_count=3,
        detect_seconds=0.0,
        filter_seconds=0.0,
    )
    line_text_filter_result = LineTextFilterResult(
        filtered_result={"lines_used": [final_line], "column_assignment": empty_assignment},
        weighted_result=WeightedAlongLinesResult(weighted_along_lines_nls=None, scored_line_count=0, covered_column_count=0),
        filter_enabled=True,
        input_line_count=1,
        scored_line_count=0,
        removed_line_count=0,
        surviving_line_count=1,
        removed_column_count=0,
        surviving_column_count=0,
        all_lines_removed=False,
        seconds=0.0,
    )

    scored = score_document_alignment(
        fname="synthetic.jpeg",
        reference_text="A" * 30,
        prediction_text="A" * 30,
        reference_windows=["A" * 10, "A" * 10, "A" * 10],
        prediction_windows=["A" * 10, "A" * 10, "A" * 10],
        ref_to_pred_hough_payload=ref_to_pred_payload,
        ref_to_ref_hough_payload=ref_to_ref_payload,
        line_text_filter_result=line_text_filter_result,
        window_size=10,
        window_stride=10,
    )

    assert scored.coverage_invalid_reason == "coverage_y_diff_below_minus_one"
    assert scored.metrics.correct_ref_coverage is None
    assert scored.ref_to_pred_payload.hough_payload.filtered_result["lines_used"] == [final_line]
