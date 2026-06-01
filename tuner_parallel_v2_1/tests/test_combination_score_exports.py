from __future__ import annotations

"""Tests for compact per-combination score rows and gzip CSV export."""

import csv
import gzip
from pathlib import Path

import numpy as np

from tuner_parallel_v2_1.outputs.tuner_combination_score_exports import (
    COMBINATION_SCORE_FIELDNAMES,
    CombinationScoreTableWriter,
)
from tuner_parallel_v2_1.tuner.hough_eval import evaluate_single_combination_values
from tuner_parallel_v2_1.tuner.sweep_scheduler import _combination_score_record_from_eval_row
from tuner_parallel_v2_1.tuner.tuner_config import (
    PARAM_HOUGH_LINE_GAP,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_SEED,
    PARAM_HOUGH_THRESHOLD,
    SweepDocument,
)


def _dummy_sweep_document() -> SweepDocument:
    """Return a tiny prepared document for scalar score-row construction."""
    matrix = np.eye(3, dtype=float)
    hough_ctx = {
        "hough_image": matrix,
        "hough_mask_bool": matrix > 0,
        "mask": matrix,
        "threshold_start": 2.4,
    }
    return SweepDocument(
        index=7,
        fname="score_fixture.jpeg",
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


def test_combination_score_record_is_scalar_and_keeps_document_nls() -> None:
    """The score table row should contain analysis fields but no geometry."""
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
        "raw_line_count_ref_to_ref": 2,
        "candidate_line_count": 2,
        "candidate_line_count_ref_to_ref": 2,
        "line_guided_columns": 3,
        "fallback_columns": 0,
        "timing_total_seconds": 0.25,
        "timing_filter_total_profiled_ref_to_pred_seconds": 0.12,
        "filter_input_line_count_ref_to_pred": 2,
    }

    record = _combination_score_record_from_eval_row(doc=doc, eval_row=eval_row)

    assert record["doc_index"] == 7
    assert record["whole_document_nls"] == 1.0
    assert record[PARAM_HOUGH_THRESHOLD] == 10
    assert record["timing_filter_total_profiled_ref_to_pred_seconds"] == 0.12
    assert record["filter_input_line_count_ref_to_pred"] == 2
    assert "raw_lines" not in record
    assert "lines_used" not in record
    assert "bundle" not in record


def test_combination_score_table_writer_writes_gzip_csv_rows(tmp_path: Path) -> None:
    """The score writer should produce a readable gzip CSV with fixed columns."""
    output_csv_gz = tmp_path / "combination_scores.csv.gz"
    writer = CombinationScoreTableWriter(output_csv_gz=output_csv_gz)
    writer.submit_document_rows(
        [
            {
                "doc_index": 1,
                "fname": "a.jpeg",
                "whole_document_nls": 0.9,
                "hough_threshold": 10,
                "hough_line_length": 5,
                "hough_line_gap": 0,
                "hough_seed": 1,
                "is_valid": True,
                "tuning_score": 0.5,
            },
            {
                "doc_index": 1,
                "fname": "a.jpeg",
                "whole_document_nls": 0.9,
                "hough_threshold": 11,
                "hough_line_length": 5,
                "hough_line_gap": 0,
                "hough_seed": 1,
                "is_valid": False,
                "invalid_reason": "fixture_invalid",
            },
        ]
    )
    writer.close()

    assert writer.summary()["row_count"] == 2
    with gzip.open(output_csv_gz, mode="rt", encoding="utf-8", newline="") as handle:
        loaded_rows = list(csv.DictReader(handle))

    assert len(loaded_rows) == 2
    assert loaded_rows[0]["fname"] == "a.jpeg"
    assert loaded_rows[0]["is_valid"] == "1"
    assert loaded_rows[1]["invalid_reason"] == "fixture_invalid"
    assert set(COMBINATION_SCORE_FIELDNAMES).issubset(set(loaded_rows[0].keys()))


def test_evaluator_profile_fields_are_populated_when_filter_profile_is_enabled() -> None:
    """The evaluator must attach real filter-profile fields to profiled rows."""
    doc = _dummy_sweep_document()

    profiled_row = evaluate_single_combination_values(
        doc=doc,
        hough_threshold=1,
        hough_line_length=1,
        hough_line_gap=0,
        hough_seed=1,
        align_abs_min_len=1.0,
        align_min_iou_threshold=0.0,
        levenshtein_backend="c",
        profile_filters=True,
    )

    assert "timing_filter_total_profiled_ref_to_pred_seconds" in profiled_row
    assert "timing_filter_total_profiled_ref_to_ref_seconds" in profiled_row
    assert "filter_input_line_count_ref_to_pred" in profiled_row
    assert "filter_input_line_count_ref_to_ref" in profiled_row
    assert profiled_row["timing_filter_total_profiled_ref_to_pred_seconds"] >= 0.0
    assert profiled_row["timing_filter_total_profiled_ref_to_ref_seconds"] >= 0.0
