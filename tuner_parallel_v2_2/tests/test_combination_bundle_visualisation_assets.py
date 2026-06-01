from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from tuner_parallel_v2_2.outputs.combination_bundle_logger import CombinationBundleLogger
from tuner_parallel_v2_2.outputs.combination_bundle_records import serialize_pickle_stream_record
from tuner_parallel_v2_2.tools.language_hough_parameter_metric_analysis import (
    attach_best_geometry_source_pointers,
    render_best_combination_visual_panel,
    render_best_combination_visual_panels_for_pair,
    split_documents_by_bundle_availability_and_prediction,
)
from tuner_parallel_v2_2.tuner.tuner_config import SweepDocument


def _hough_context(matrix: np.ndarray) -> dict:
    mask = np.asarray(matrix > 0, dtype=bool)
    return {
        "hough_image": mask.astype(np.uint8),
        "hough_mask_bool": mask,
        "mask": mask.astype(np.uint8),
        "region_of_interest_mask_bool": mask,
        "strong_match_mask_bool": mask,
        "hough_preprocessing_summary": {
            "accepted": True,
            "active_cell_count": int(np.count_nonzero(mask)),
            "score_floor": 20.0,
        },
    }


def test_visualisation_bundle_keeps_current_assets_and_adds_preprocessing_masks(tmp_path: Path) -> None:
    """Visual bundles should keep score matrices while adding preprocessing masks."""
    matrix = np.eye(3, dtype=float)
    document = SweepDocument(
        index=3,
        fname="bundle_fixture.jpeg",
        pred="abc",
        ref="abc",
        window_size=4,
        window_stride=2,
        ref_to_pred_matrix=matrix,
        ref_to_ref_matrix=matrix,
        whole_document_nls=1.0,
        pred_blocks=["a", "b", "c"],
        ref_blocks=["a", "b", "c"],
        ref_to_pred_hough_ctx=_hough_context(matrix),
        ref_to_ref_hough_ctx=_hough_context(matrix),
    )
    logger = CombinationBundleLogger(root_dir=tmp_path, scope="all", include_candidate_lines=False)

    logger.submit_completed_document(
        doc=document,
        records_by_threshold={
            1: [
                {
                    "schema_version": "test",
                    "document": {"index": 3, "fname": "bundle_fixture.jpeg"},
                    "metrics": {"is_valid": True},
                    "ref_to_pred": {"hough_detection": {"raw_lines": []}, "filtering": {"lines_used": []}},
                }
            ]
        },
    )
    logger.close()

    document_dir = logger.document_dir(document)
    metadata = json.loads((document_dir / "document_metadata.json").read_text(encoding="utf-8"))

    assert (document_dir / "ref_to_pred_score_matrix.npy").exists()
    assert (document_dir / "ref_to_ref_score_matrix.npy").exists()
    assert (document_dir / "ref_to_pred_hough_input_mask.npy").exists()
    assert (document_dir / "ref_to_pred_region_of_interest_mask.npy").exists()
    assert (document_dir / "ref_to_pred_strong_match_mask.npy").exists()
    assert metadata["score_matrices"]["ref_to_pred"] == "ref_to_pred_score_matrix.npy"
    assert metadata["hough_preprocessing"]["ref_to_pred_hough_input_mask"] == "ref_to_pred_hough_input_mask.npy"
    assert metadata["hough_preprocessing"]["ref_to_pred_summary"]["active_cell_count"] == 3

def _winner_only_combination_record() -> dict:
    return {
        "schema_version": "test",
        "document": {
            "index": 556,
            "fname": "winner_only_fixture.jpeg",
            "whole_document_nls": 0.5,
        },
        "hough_parameters": {
            "hough_threshold": 9,
            "hough_line_length": 11,
            "hough_line_gap": 0,
            "hough_seed": 1,
            "effective_hough_seed": 1,
        },
        "metrics": {
            "tuning_score": 0.25,
            "weighted_along_lines_nls": 0.5,
            "correct_ref_coverage": 0.75,
            "missing_ref_coverage": 0.25,
            "repetition_on_ref": 0.0,
            "hallucination": 0.1,
            "raw_line_count": 1,
            "candidate_line_count": 1,
            "used_line_count": 1,
            "line_guided_columns": 12,
            "fallback_columns": 0,
            "is_valid": True,
            "invalid_reason": "",
        },
        "ref_to_pred": {
            "hough_detection": {"raw_lines": [[[0, 0], [10, 10]]]},
            "filtering": {"lines_used": [{"x0": 0, "y0": 0, "x1": 10, "y1": 10}]},
        },
    }


def test_winner_only_visualisation_uses_the_single_saved_geometry_record(tmp_path: Path) -> None:
    """Winner-only bundles should plot the saved winner when CSV ties select another row."""
    bundle_dir = tmp_path / "document_000556_winner_only_fixture.jpeg"
    bundle_dir.mkdir()
    record_path = bundle_dir / "threshold_009.pklstream"
    record_path.write_bytes(serialize_pickle_stream_record(_winner_only_combination_record()))

    csv_selected_row = {
        "main_language": "Greek",
        "document_type": "print",
        "document_index": 556,
        "fname": "winner_only_fixture.jpeg",
        "bundle_dir": str(bundle_dir),
        "shard_index": 11,
        "whole_document_nls": 0.5,
        "hough_threshold": 5,
        "hough_line_length": 7,
        "hough_line_gap": 0,
        "hough_seed": 1,
        "effective_hough_seed": 1,
        "tuning_score": 0.0,
        "weighted_along_lines_nls": 0.0,
        "correct_ref_coverage": 0.0,
        "missing_ref_coverage": 1.0,
        "repetition_on_ref": 0.0,
        "hallucination": 1.0,
        "non_hallucination": 0.0,
        "raw_line_count": 0,
        "candidate_line_count": 0,
        "used_line_count": 0,
        "line_guided_columns": 0,
        "fallback_columns": 0,
        "is_valid": True,
        "invalid_reason": "",
        "source_jsonl_path": "",
        "source_line_number": 0,
    }

    resolved_rows = attach_best_geometry_source_pointers(pd.DataFrame([csv_selected_row]))
    resolved_row = resolved_rows.iloc[0]

    assert resolved_row["hough_threshold"] == 9
    assert resolved_row["hough_line_length"] == 11
    assert resolved_row["hough_line_gap"] == 0
    assert resolved_row["tuning_score"] == 0.25
    assert resolved_row["source_jsonl_path"] == str(record_path)
    assert resolved_row["source_line_number"] == 1

def test_best_combination_panel_renders_region_of_interest_and_hough_input_masks(tmp_path: Path) -> None:
    """The best-combination panel should render score matrices and the two binary masks."""
    bundle_dir = tmp_path / "document_000556_winner_only_fixture.jpeg"
    bundle_dir.mkdir()
    record_path = bundle_dir / "threshold_009.pklstream"
    record_path.write_bytes(serialize_pickle_stream_record(_winner_only_combination_record()))

    score_matrix = np.eye(12, dtype=float) * 100.0
    region_of_interest_mask = np.ones_like(score_matrix, dtype=bool)
    hough_input_mask = np.eye(12, dtype=bool)
    np.save(bundle_dir / "ref_to_pred_score_matrix.npy", score_matrix)
    np.save(bundle_dir / "ref_to_ref_score_matrix.npy", score_matrix)
    np.save(bundle_dir / "ref_to_pred_region_of_interest_mask.npy", region_of_interest_mask)
    np.save(bundle_dir / "ref_to_pred_hough_input_mask.npy", hough_input_mask)

    best_row = pd.Series(
        {
            "main_language": "Greek",
            "document_type": "print",
            "document_index": 556,
            "fname": "winner_only_fixture.jpeg",
            "bundle_dir": str(bundle_dir),
            "shard_index": 11,
            "whole_document_nls": 0.5,
            "hough_threshold": 9,
            "hough_line_length": 11,
            "hough_line_gap": 0,
            "hough_seed": 1,
            "effective_hough_seed": 1,
            "tuning_score": 0.25,
            "weighted_along_lines_nls": 0.5,
            "correct_ref_coverage": 0.75,
            "missing_ref_coverage": 0.25,
            "repetition_on_ref": 0.0,
            "hallucination": 0.1,
            "non_hallucination": 0.9,
            "raw_line_count": 1,
            "candidate_line_count": 1,
            "used_line_count": 1,
            "line_guided_columns": 12,
            "fallback_columns": 0,
            "is_valid": True,
            "invalid_reason": "",
            "source_jsonl_path": str(record_path),
            "source_line_number": 1,
        }
    )

    output_path = render_best_combination_visual_panel(
        best_row=best_row,
        temporary_panel_output_dir=tmp_path / "temporary_panels",
        ref_to_pred_scores_pkl=tmp_path / "missing_ref_to_pred.pkl",
        ref_to_ref_scores_pkl=tmp_path / "missing_ref_to_ref.pkl",
        saved_figure_dpi=50,
        show_line_labels=False,
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0

def test_visualisation_uses_worker_skip_record_when_bundle_is_missing(tmp_path: Path) -> None:
    """Worker skipped-document CSVs should override the generic missing-bundle reason."""
    shards_dir = tmp_path / "shards"
    skipped_csv_dir = shards_dir / "dynamic_worker_000" / "csv"
    skipped_csv_dir.mkdir(parents=True)
    diagnostic_bundle_dir = tmp_path / "diagnostics" / "document_000690_arabic_fixture.jpeg"
    pd.DataFrame(
        [
            {
                "index": 690,
                "fname": "arabic_fixture.jpeg",
                "skip_reason": "ref_to_pred_hough_preprocessing_rejected",
                "skip_stage": "hough_preprocessing",
                "preprocessing_rejection_reason": "no_strong_match_evidence",
                "diagnostic_bundle_dir": str(diagnostic_bundle_dir),
            }
        ]
    ).to_csv(skipped_csv_dir / "skipped_documents.csv", index=False)

    loadable_documents, skipped_documents = split_documents_by_bundle_availability_and_prediction(
        runfile_documents=[
            {
                "document_index": 690,
                "fname": "arabic_fixture.jpeg",
                "main_language": "Arabic",
                "document_type": "handwriting",
            }
        ],
        shards_dir=shards_dir,
        documents_per_shard=50,
    )

    assert loadable_documents == []
    assert skipped_documents[0]["skip_reason"] == "ref_to_pred_hough_preprocessing_rejected"
    assert skipped_documents[0]["preprocessing_rejection_reason"] == "no_strong_match_evidence"
    assert skipped_documents[0]["bundle_dir"] == str(diagnostic_bundle_dir)


def test_stitched_visual_panel_includes_skipped_document_diagnostics(tmp_path: Path) -> None:
    """Skipped documents with diagnostic bundles should still appear in the stitched PNG."""
    diagnostic_bundle_dir = tmp_path / "diagnostics" / "document_000690_arabic_fixture.jpeg"
    diagnostic_bundle_dir.mkdir(parents=True)
    score_matrix = np.eye(8, dtype=float) * 100.0
    region_of_interest_mask = np.ones_like(score_matrix, dtype=bool)
    hough_input_mask = np.eye(8, dtype=bool)
    np.save(diagnostic_bundle_dir / "ref_to_pred_score_matrix.npy", score_matrix)
    np.save(diagnostic_bundle_dir / "ref_to_ref_score_matrix.npy", score_matrix)
    np.save(diagnostic_bundle_dir / "ref_to_pred_region_of_interest_mask.npy", region_of_interest_mask)
    np.save(diagnostic_bundle_dir / "ref_to_pred_hough_input_mask.npy", hough_input_mask)

    stitched_path = render_best_combination_visual_panels_for_pair(
        best_rows_dataframe=pd.DataFrame(),
        skipped_documents=[
            {
                "document_index": 690,
                "fname": "arabic_fixture.jpeg",
                "main_language": "Arabic",
                "document_type": "handwriting",
                "skip_reason": "ref_to_pred_hough_preprocessing_rejected",
                "skip_stage": "hough_preprocessing",
                "preprocessing_rejection_reason": "no_strong_match_evidence",
                "diagnostic_bundle_dir": str(diagnostic_bundle_dir),
                "ref_to_pred_matrix_rows": 8,
                "ref_to_pred_matrix_cols": 8,
                "message": "Skipped by test fixture.",
            }
        ],
        output_dir=tmp_path / "visuals",
        language_name="Arabic",
        document_type="handwriting",
        ref_to_pred_scores_pkl=tmp_path / "missing_ref_to_pred.pkl",
        ref_to_ref_scores_pkl=tmp_path / "missing_ref_to_ref.pkl",
        saved_figure_dpi=50,
        show_line_labels=False,
    )

    assert stitched_path is not None
    assert stitched_path.exists()
    assert stitched_path.stat().st_size > 0

