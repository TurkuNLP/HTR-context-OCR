from __future__ import annotations

"""Tests for compact per-combination profiling records and CSV export."""

import csv
from pathlib import Path

import numpy as np

from tuner_parallel_v2_1.outputs.tuner_profile_exports import write_combination_profile_csv
from tuner_parallel_v2_1.tuner.sweep_scheduler import _combination_profile_record_from_eval_row
from tuner_parallel_v2_1.tuner.tuner_config import (
    PARAM_HOUGH_LINE_GAP,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_SEED,
    PARAM_HOUGH_THRESHOLD,
    SweepDocument,
)


def _dummy_sweep_document() -> SweepDocument:
    """Return a tiny prepared document for scalar profile-row construction."""
    matrix = np.eye(3, dtype=float)
    hough_ctx = {
        "hough_image": matrix,
        "hough_mask_bool": matrix > 0,
        "mask": matrix,
        "threshold_start": 2.4,
    }
    return SweepDocument(
        index=7,
        fname="profile_fixture.jpeg",
        pred="abc",
        ref="abc",
        window_size=4,
        window_stride=2,
        ref_to_pred_matrix=matrix,
        ref_to_ref_matrix=matrix,
        whole_document_nls=1.0,
        pred_blocks=["a", "b", "c"],
        ref_blocks=["a", "b", "c"],
        ref_to_pred_hough_ctx=hough_ctx,
        ref_to_ref_hough_ctx=hough_ctx,
    )


def test_combination_profile_record_contains_scalar_identity_and_profile_fields() -> None:
    """A profile row should be compact and sufficient for timing analysis."""
    doc = _dummy_sweep_document()
    eval_row = {
        PARAM_HOUGH_THRESHOLD: 10,
        PARAM_HOUGH_LINE_LENGTH: 5,
        PARAM_HOUGH_LINE_GAP: 0,
        PARAM_HOUGH_SEED: 1,
        "is_valid": True,
        "tuning_score": 0.75,
        "weighted_along_lines_nls": 0.8,
        "correct_ref_coverage": 0.9,
        "missing_ref_coverage": 0.1,
        "repetition_on_ref": 0.0,
        "hallucination": 0.05,
        "line_count": 1,
        "used_line_count": 1,
        "used_line_count_ref_to_ref": 1,
        "raw_line_count": 2,
        "candidate_line_count": 2,
        "raw_line_count_ref_to_ref": 1,
        "candidate_line_count_ref_to_ref": 1,
        "line_guided_columns": 3,
        "fallback_columns": 0,
        "timing_total_seconds": 0.25,
        "timing_filter_total_profiled_ref_to_pred_seconds": 0.12,
        "filter_input_line_count_ref_to_pred": 2,
    }

    record = _combination_profile_record_from_eval_row(doc=doc, eval_row=eval_row)

    assert record["doc_index"] == 7
    assert record["fname"] == "profile_fixture.jpeg"
    assert record[PARAM_HOUGH_THRESHOLD] == 10
    assert record["matrix_rows_ref_to_pred"] == 3
    assert record["timing_total_seconds"] == 0.25
    assert record["timing_filter_total_profiled_ref_to_pred_seconds"] == 0.12
    assert record["filter_input_line_count_ref_to_pred"] == 2
    assert "raw_lines" not in record
    assert "lines_used" not in record


def test_write_combination_profile_csv_round_trips_scalar_rows(tmp_path: Path) -> None:
    """The profile CSV writer should create one row per evaluated combination."""
    output_csv = tmp_path / "combination_profile.csv"
    rows = [
        {"doc_index": 1, "fname": "a.jpeg", "hough_threshold": 10, "tuning_score": 0.5},
        {"doc_index": 2, "fname": "b.jpeg", "hough_threshold": 11, "tuning_score": 0.6},
    ]

    written_path = write_combination_profile_csv(rows=rows, output_csv=output_csv)

    assert written_path == output_csv
    with output_csv.open("r", encoding="utf-8", newline="") as handle:
        loaded_rows = list(csv.DictReader(handle))
    assert len(loaded_rows) == 2
    assert loaded_rows[0]["fname"] == "a.jpeg"
    assert loaded_rows[1]["hough_threshold"] == "11"
