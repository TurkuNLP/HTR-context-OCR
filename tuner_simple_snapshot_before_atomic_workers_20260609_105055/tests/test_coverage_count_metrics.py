from __future__ import annotations

from tuner_simple.scoring.coverage_count_metrics import compute_coverage_count_metrics


def payload(lines, *, reference_text_length=30, other_text_length=30):
    return {
        "lines_used": list(lines),
        "reference_text_length": int(reference_text_length),
        "other_text_length": int(other_text_length),
        "window_size": 10,
        "window_stride": 10,
    }


def diagonal_line():
    return {"x0": 0.0, "y0": 0.0, "x1": 2.0, "y1": 2.0}


def test_matching_ref_to_ref_and_ref_to_pred_is_correct_coverage():
    result = compute_coverage_count_metrics(
        ref_to_pred_scoring_payload=payload([diagonal_line()]),
        ref_to_ref_scoring_payload=payload([diagonal_line()]),
    )

    assert result.correct_ref_coverage == 1.0
    assert result.missing_ref_coverage == 0.0
    assert result.repetition_on_reference == 0.0
    assert result.hallucination == 0.0
    assert result.invalid_reason is None


def test_missing_reference_coverage_comes_from_minus_one_difference():
    result = compute_coverage_count_metrics(
        ref_to_pred_scoring_payload=payload([]),
        ref_to_ref_scoring_payload=payload([diagonal_line()]),
    )

    assert result.correct_ref_coverage == 0.0
    assert result.missing_ref_coverage == 1.0
    assert result.repetition_on_reference == 0.0
    assert result.hallucination == 1.0
    assert result.diagnostics["coverage_y_diff_min"] == -1


def test_repetition_on_reference_comes_from_positive_difference():
    result = compute_coverage_count_metrics(
        ref_to_pred_scoring_payload=payload([diagonal_line(), diagonal_line()]),
        ref_to_ref_scoring_payload=payload([diagonal_line()]),
    )

    assert result.correct_ref_coverage == 0.0
    assert result.missing_ref_coverage == 0.0
    assert result.repetition_on_reference == 1.0
    assert result.hallucination == 0.0
    assert result.diagnostics["coverage_y_diff_max"] == 1


def test_reference_missing_repetition_below_minus_one_is_invalid_like_v2_2():
    result = compute_coverage_count_metrics(
        ref_to_pred_scoring_payload=payload([]),
        ref_to_ref_scoring_payload=payload([diagonal_line(), diagonal_line()]),
    )

    assert result.correct_ref_coverage is None
    assert result.missing_ref_coverage is None
    assert result.repetition_on_reference is None
    assert result.hallucination is None
    assert result.invalid_reason == "coverage_y_diff_below_minus_one"
    assert result.diagnostics["coverage_y_diff_lt_minus_one_count"] == 30
    assert result.diagnostics["coverage_y_diff_below_minus_one_counts_json"] == {"-2": 30}
