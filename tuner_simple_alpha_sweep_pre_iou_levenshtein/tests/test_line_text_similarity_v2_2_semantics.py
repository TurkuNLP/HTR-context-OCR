from __future__ import annotations

import math

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.line_text_similarity import (
    LineTextRecord,
    compute_line_text_record,
    weighted_result_from_records,
)


def test_line_text_uses_reference_document_order_and_contiguous_window_text() -> None:
    reference_windows = ["ab", "cd"]
    prediction_windows = ["ab", "cd"]
    column_assignment = {
        "mapped_y": np.asarray([1.0, 0.0], dtype=float),
        "mapped_line_id": np.asarray([0, 0], dtype=int),
    }
    line_record = {"x0": 0.0, "y0": 0.0, "x1": 1.0, "y1": 1.0}

    scored_line = compute_line_text_record(
        line_id=0,
        line_record=line_record,
        column_assignment=column_assignment,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
        reference_window_count=len(reference_windows),
    )

    assert scored_line is not None
    assert scored_line.normalized_levenshtein_similarity == 1.0
    assert scored_line.mapped_reference_row_count == 2
    assert scored_line.owned_prediction_column_count == 2


def test_weighted_along_lines_uses_geometric_line_length_not_column_count() -> None:
    records = [
        LineTextRecord(
            line_id=0,
            normalized_levenshtein_similarity=1.0,
            line_length=10.0,
            owned_prediction_column_count=1,
            mapped_reference_row_count=1,
        ),
        LineTextRecord(
            line_id=1,
            normalized_levenshtein_similarity=0.0,
            line_length=1.0,
            owned_prediction_column_count=100,
            mapped_reference_row_count=1,
        ),
    ]

    weighted_result = weighted_result_from_records(records)

    assert math.isclose(weighted_result.weighted_along_lines_nls, 10.0 / 11.0)
    assert math.isclose(weighted_result.unweighted_along_lines_nls, 0.5)
    assert weighted_result.scored_line_count == 2
    assert math.isclose(weighted_result.total_line_length, 11.0)
    assert weighted_result.covered_column_count == 101
