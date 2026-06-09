from __future__ import annotations

"""Tests for the optional alignment-evidence Hough selection objective."""

import pytest

from tuner_parallel_v2_2.metrics.alignment_quality_score import (
    compute_alignment_evidence_selection_score,
    compute_harmonic_tuning_score,
    compute_line_guided_fraction,
    compute_non_hallucination_weighted_tuning_score,
    compute_score_matrix_support_from_lines,
)
from tuner_parallel_v2_2.tuner.hough_eval import (
    SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE,
    SELECTION_OBJECTIVE_NON_HALLUCINATION_WEIGHTED,
    SELECTION_OBJECTIVE_STRICT_QUALITY,
    pick_better_eval,
)


def test_score_matrix_support_uses_owned_score_mean_on_zero_to_one_scale() -> None:
    """Final line support should be a weighted mean of 0..100 matrix scores."""
    lines = [
        {"owned_score_mean": 80.0, "owned_cols": 3, "x0": 0, "y0": 0, "x1": 3, "y1": 3},
        {"owned_score_mean": 20.0, "owned_cols": 1, "x0": 0, "y0": 1, "x1": 1, "y1": 2},
    ]

    support = compute_score_matrix_support_from_lines(lines)

    assert support == pytest.approx(0.65)


def test_alignment_evidence_selection_can_choose_lower_tuning_score_when_matrix_support_is_stronger() -> None:
    """The optional objective should not change strict-quality ranking."""
    text_focused_row = {
        "is_valid": True,
        "tuning_score": 0.80,
        "weighted_along_lines_nls": 0.90,
        "correct_ref_coverage": 0.90,
        "hallucination": 0.05,
        "line_guided_columns": 4,
        "fallback_columns": 4,
        "score_matrix_support": 0.35,
        "line_guided_fraction": compute_line_guided_fraction(line_guided_columns=4, fallback_columns=4),
    }
    text_focused_row["alignment_selection_score"] = compute_alignment_evidence_selection_score(
        weighted_along_lines_nls=text_focused_row["weighted_along_lines_nls"],
        score_matrix_support=text_focused_row["score_matrix_support"],
        line_guided_fraction=text_focused_row["line_guided_fraction"],
        hallucination=text_focused_row["hallucination"],
    )

    matrix_supported_row = {
        "is_valid": True,
        "tuning_score": 0.70,
        "weighted_along_lines_nls": 0.86,
        "correct_ref_coverage": 0.75,
        "hallucination": 0.05,
        "line_guided_columns": 8,
        "fallback_columns": 0,
        "score_matrix_support": 0.92,
        "line_guided_fraction": compute_line_guided_fraction(line_guided_columns=8, fallback_columns=0),
    }
    matrix_supported_row["alignment_selection_score"] = compute_alignment_evidence_selection_score(
        weighted_along_lines_nls=matrix_supported_row["weighted_along_lines_nls"],
        score_matrix_support=matrix_supported_row["score_matrix_support"],
        line_guided_fraction=matrix_supported_row["line_guided_fraction"],
        hallucination=matrix_supported_row["hallucination"],
    )

    assert (
        pick_better_eval(
            text_focused_row,
            matrix_supported_row,
            selection_objective=SELECTION_OBJECTIVE_STRICT_QUALITY,
        )
        is text_focused_row
    )
    assert (
        pick_better_eval(
            text_focused_row,
            matrix_supported_row,
            selection_objective=SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE,
        )
        is matrix_supported_row
    )


def _row_with_harmonic_scores(*, weighted_nls: float, coverage: float, hallucination: float) -> dict:
    """Build the smallest valid eval row needed by the objective-ranking tests."""
    return {
        "is_valid": True,
        "tuning_score": compute_harmonic_tuning_score(
            weighted_along_lines_nls=weighted_nls,
            correct_ref_coverage=coverage,
            hallucination=hallucination,
        ),
        "non_hallucination_weighted_tuning_score": compute_non_hallucination_weighted_tuning_score(
            weighted_along_lines_nls=weighted_nls,
            correct_ref_coverage=coverage,
            hallucination=hallucination,
        ),
        "weighted_along_lines_nls": weighted_nls,
        "correct_ref_coverage": coverage,
        "hallucination": hallucination,
        "line_guided_columns": 0,
        "fallback_columns": 0,
    }


def test_non_hallucination_weighted_objective_can_prefer_less_hallucination() -> None:
    """The new option should only change ranking when explicitly selected."""
    high_text_quality_row = _row_with_harmonic_scores(
        weighted_nls=0.99,
        coverage=0.99,
        hallucination=0.30,
    )
    low_hallucination_row = _row_with_harmonic_scores(
        weighted_nls=0.80,
        coverage=0.80,
        hallucination=0.0,
    )

    assert high_text_quality_row["tuning_score"] > low_hallucination_row["tuning_score"]
    assert (
        high_text_quality_row["non_hallucination_weighted_tuning_score"]
        < low_hallucination_row["non_hallucination_weighted_tuning_score"]
    )
    assert (
        pick_better_eval(
            high_text_quality_row,
            low_hallucination_row,
            selection_objective=SELECTION_OBJECTIVE_STRICT_QUALITY,
        )
        is high_text_quality_row
    )
    assert (
        pick_better_eval(
            high_text_quality_row,
            low_hallucination_row,
            selection_objective=SELECTION_OBJECTIVE_NON_HALLUCINATION_WEIGHTED,
        )
        is low_hallucination_row
    )
