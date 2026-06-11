from __future__ import annotations

"""Process one document through loading, Hough detection, filtering, and scoring."""

from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from tuner_simple.config.pipeline_config import PipelineConfig
from tuner_simple.document_selection.runfile_loader import RunfileDocument
from tuner_simple.matrix_operations.matrix_loader import (
    # Pass this value into the surrounding multi-line call or collection.
    ScoreMatrixIndexBundle,
    # Pass this value into the surrounding multi-line call or collection.
    load_or_compute_ref_to_pred_matrix,
    # Pass this value into the surrounding multi-line call or collection.
    load_or_compute_ref_to_ref_matrix,
)
from tuner_simple.matrix_operations.matrix_shape import (
    # Pass this value into the surrounding multi-line call or collection.
    count_sliding_windows,
    # Pass this value into the surrounding multi-line call or collection.
    matrix_size_skip_reason,
    # Pass this value into the surrounding multi-line call or collection.
    sliding_text_windows,
)
from tuner_simple.matrix_operations.score_floor import ScoreFloorResult, compute_score_floor_mask
from tuner_simple.probabilistic_hough.hough_detection import HoughFilteredPayload, run_probabilistic_hough_and_filter
from tuner_simple.scoring.line_text_similarity import filter_lines_by_minimum_normalised_levenshtein
from tuner_simple.scoring.scoring_pipeline import ScoredDocumentResult, score_document_alignment, zero_alignment_metrics
from tuner_simple.scoring.scoring_pipeline import DocumentAlignmentMetrics


# Ask Python to generate common data-container methods for the class defined next.
@dataclass
# Define the DocumentRunResult class, which groups related state and behavior for this part of the pipeline.
class DocumentRunResult:
    """Serializable output from processing one document."""

    # Define the result_row field so this data object records that value explicitly.
    result_row: dict[str, Any] | None
    # Define the skipped_row field so this data object records that value explicitly.
    skipped_row: dict[str, Any] | None
    # Define the loadable_row field so this data object records that value explicitly.
    loadable_row: dict[str, Any] | None
    # Define the loaded_row field so this data object records that value explicitly.
    loaded_row: dict[str, Any] | None
    # Define the plot_payload field so this data object records that value explicitly.
    plot_payload: dict[str, Any] | None


# Define the document_table_row function; its body below performs one named step of the pipeline.
def document_table_row(document: RunfileDocument, *, window_size: int, window_stride: int) -> dict[str, Any]:
    """Build the shared document-audit row used by several CSV files."""
    # Return this computed value to the caller so the next pipeline stage can use it.
    return {
        # Add the document_index field to the surrounding dictionary so it appears in outputs or returned metadata.
        "document_index": int(document.document_index),
        # Add the fname field to the surrounding dictionary so it appears in outputs or returned metadata.
        "fname": str(document.fname),
        # Add the main_language field to the surrounding dictionary so it appears in outputs or returned metadata.
        "main_language": str(document.main_language),
        # Add the document_type field to the surrounding dictionary so it appears in outputs or returned metadata.
        "document_type": str(document.document_type),
        # Add the reference_text_length field to the surrounding dictionary so it appears in outputs or returned metadata.
        "reference_text_length": int(len(document.reference_text)),
        # Add the prediction_text_length field to the surrounding dictionary so it appears in outputs or returned metadata.
        "prediction_text_length": int(len(document.prediction_text)),
        # Add the reference_window_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "reference_window_count": count_sliding_windows(
            # Pass this value into the surrounding multi-line call or collection.
            document.reference_text,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=int(window_size),
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=int(window_stride),
        ),
        # Add the prediction_window_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "prediction_window_count": count_sliding_windows(
            # Pass this value into the surrounding multi-line call or collection.
            document.prediction_text,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=int(window_size),
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=int(window_stride),
        ),
    }


def format_log_value(value: Any, *, digits: int = 6) -> str:
    """Render one scalar value compactly for timestamped progress logs."""
    if value is None:
        return "None"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "nan"
        return f"{value:.{int(digits)}f}"
    return str(value)


def matrix_shape_text(matrix_shape: tuple[int, int]) -> str:
    """Render a matrix shape as rows x columns for log messages."""
    return f"{int(matrix_shape[0])}x{int(matrix_shape[1])}"


def log_score_floor_summary(log, *, fname: str, label: str, score_floor_result: ScoreFloorResult) -> None:
    """Log the score-floor preprocessing result for one matrix direction."""
    log(
        f"[preprocess] {fname} {label} "
        f"mean={format_log_value(score_floor_result.score_mean)} "
        f"std={format_log_value(score_floor_result.score_standard_deviation)} "
        f"alpha={format_log_value(score_floor_result.score_floor_alpha)} "
        f"floor={format_log_value(score_floor_result.score_floor)} "
        f"active_cells={int(score_floor_result.active_cell_count)} "
        f"active_fraction={format_log_value(score_floor_result.active_fraction)} "
        f"mask_shape={matrix_shape_text(tuple(int(value) for value in score_floor_result.hough_input_mask.shape))}"
    )


def log_hough_summary(log, *, fname: str, label: str, hough_payload: HoughFilteredPayload) -> None:
    """Log raw, candidate, and final line counts after one Hough call."""
    log(
        f"[hough] {fname} {label} done "
        f"raw_lines={int(hough_payload.raw_line_count)} "
        f"candidate_lines={int(hough_payload.candidate_line_count)} "
        f"used_lines={int(hough_payload.used_line_count)} "
        f"detect_seconds={format_log_value(hough_payload.detect_seconds)} "
        f"filter_seconds={format_log_value(hough_payload.filter_seconds)} "
        f"total_seconds={format_log_value(float(hough_payload.detect_seconds) + float(hough_payload.filter_seconds))}"
    )


def log_line_filter_summary(log, *, fname: str, line_filter_result) -> None:
    """Log the line-level normalised Levenshtein filter result."""
    log(
        f"[text-filter] {fname} "
        f"enabled={bool(line_filter_result.filter_enabled)} "
        f"input_lines={int(line_filter_result.input_line_count)} "
        f"scored_lines={int(line_filter_result.scored_line_count)} "
        f"removed_lines={int(line_filter_result.removed_line_count)} "
        f"surviving_lines={int(line_filter_result.surviving_line_count)} "
        f"removed_columns={int(line_filter_result.removed_column_count)} "
        f"surviving_columns={int(line_filter_result.surviving_column_count)} "
        f"all_lines_removed={bool(line_filter_result.all_lines_removed)} "
        f"seconds={format_log_value(line_filter_result.seconds)}"
    )


def log_scoring_summary(log, *, fname: str, scored: ScoredDocumentResult, total_seconds: float) -> None:
    """Log the final public metrics and coverage-subtraction diagnostics."""
    metrics = scored.metrics
    diagnostics = dict(scored.coverage_diagnostics or {})
    log(
        f"[scoring] {fname} metrics "
        f"document_nls={format_log_value(metrics.document_normalised_levenshtein)} "
        f"weighted_along_lines_nls={format_log_value(metrics.weighted_along_lines_normalised_levenshtein)} "
        f"correct_ref_coverage={format_log_value(metrics.correct_ref_coverage)} "
        f"missing_ref_coverage={format_log_value(metrics.missing_ref_coverage)} "
        f"repetition_on_reference={format_log_value(metrics.repetition_on_reference)} "
        f"hallucination={format_log_value(metrics.hallucination)} "
        f"coverage_invalid_reason={format_log_value(scored.coverage_invalid_reason)} "
        f"y_diff_min={format_log_value(diagnostics.get('coverage_y_diff_min'))} "
        f"y_diff_max={format_log_value(diagnostics.get('coverage_y_diff_max'))} "
        f"y_diff_lt_minus_one={format_log_value(diagnostics.get('coverage_y_diff_lt_minus_one_count'))} "
        f"coverage_seconds={format_log_value(scored.coverage_seconds)} "
        f"levenshtein_seconds={format_log_value(scored.levenshtein_seconds)} "
        f"total_seconds={format_log_value(total_seconds)}"
    )


# Define the skipped_row_from_document function; its body below performs one named step of the pipeline.
def skipped_row_from_document(
    # Define the document field so this data object records that value explicitly.
    document: RunfileDocument,
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the config field so this data object records that value explicitly.
    config: PipelineConfig,
    # Define the skip_stage field so this data object records that value explicitly.
    skip_stage: str,
    # Define the skip_reason field so this data object records that value explicitly.
    skip_reason: str,
    # Compute or store matrix_shape: tuple[int, int] so later code can reuse this named value clearly.
    matrix_shape: tuple[int, int] = (0, 0),
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> dict[str, Any]:
    """Build one skip row with enough context to understand the failure."""
    # Compute or store row so later code can reuse this named value clearly.
    row = document_table_row(document, window_size=config.window_size, window_stride=config.window_stride)
    # Start a multi-line call or data structure so related arguments stay readable.
    row.update(
        # Start a multi-line collection so related values can be listed clearly.
        {
            # Add the skip_stage field to the surrounding dictionary so it appears in outputs or returned metadata.
            "skip_stage": str(skip_stage),
            # Add the skip_reason field to the surrounding dictionary so it appears in outputs or returned metadata.
            "skip_reason": str(skip_reason),
            # Add the row_count field to the surrounding dictionary so it appears in outputs or returned metadata.
            "row_count": int(matrix_shape[0]),
            # Add the column_count field to the surrounding dictionary so it appears in outputs or returned metadata.
            "column_count": int(matrix_shape[1]),
        }
    )
    # Return this computed value to the caller so the next pipeline stage can use it.
    return row


# Define the empty_hough_payload function; its body below performs one named step of the pipeline.
def empty_hough_payload(*, score_floor_result: ScoreFloorResult) -> HoughFilteredPayload:
    """Return a zero-line payload when no Hough call should be attempted."""
    from tuner_simple.probabilistic_hough.hough_input import build_simple_hough_context

    # Compute or store context so later code can reuse this named value clearly.
    context = build_simple_hough_context(
        # Pass hough_input_mask into the surrounding call; this supplies the boolean matrix passed to Hough, where True means this cell can vote for a line.
        hough_input_mask=score_floor_result.hough_input_mask,
        # Pass score_floor into the surrounding call; this supplies the numeric cutoff a score must meet before it can become a Hough voter.
        score_floor=score_floor_result.score_floor,
    )
    # Compute or store filtered so later code can reuse this named value clearly.
    filtered = {
        # Add the lines_used field to the surrounding dictionary so it appears in outputs or returned metadata.
        "lines_used": [],
        # Add the lines_for_filtering field to the surrounding dictionary so it appears in outputs or returned metadata.
        "lines_for_filtering": [],
        # Add the column_assignment field to the surrounding dictionary so it appears in outputs or returned metadata.
        "column_assignment": {
            # Add the mapped_y field to the surrounding dictionary so it appears in outputs or returned metadata.
            "mapped_y": np.full(score_floor_result.hough_input_mask.shape[1], np.nan, dtype=float),
            # Add the mapped_line_id field to the surrounding dictionary so it appears in outputs or returned metadata.
            "mapped_line_id": np.full(score_floor_result.hough_input_mask.shape[1], -1, dtype=int),
        },
    }
    # Return this computed value to the caller so the next pipeline stage can use it.
    return HoughFilteredPayload(
        # Pass the hough_context argument into the surrounding call so the callee receives that setting explicitly.
        hough_context=context,
        # Pass the detection_result argument into the surrounding call so the callee receives that setting explicitly.
        detection_result={"raw_lines": [], "candidate_segments": [], "threshold_start": score_floor_result.score_floor},
        # Pass the filtered_result argument into the surrounding call so the callee receives that setting explicitly.
        filtered_result=filtered,
        # Pass the raw_line_count argument into the surrounding call so the callee receives that setting explicitly.
        raw_line_count=0,
        # Pass the candidate_line_count argument into the surrounding call so the callee receives that setting explicitly.
        candidate_line_count=0,
        # Pass the used_line_count argument into the surrounding call so the callee receives that setting explicitly.
        used_line_count=0,
        # Pass the detect_seconds argument into the surrounding call so the callee receives that setting explicitly.
        detect_seconds=0.0,
        # Pass the filter_seconds argument into the surrounding call so the callee receives that setting explicitly.
        filter_seconds=0.0,
    )


# Define the metrics_to_row_fields function; its body below performs one named step of the pipeline.
def metrics_to_row_fields(metrics: DocumentAlignmentMetrics) -> dict[str, Any]:
    """Expose the six scientific metrics with their public names."""
    # Return this computed value to the caller so the next pipeline stage can use it.
    return {
        # Add the document_normalised_levenshtein field to the surrounding dictionary so it appears in outputs or returned metadata.
        "document_normalised_levenshtein": metrics.document_normalised_levenshtein,
        # Add the weighted_along_lines_normalised_levenshtein field to the surrounding dictionary so it appears in outputs or returned metadata.
        "weighted_along_lines_normalised_levenshtein": metrics.weighted_along_lines_normalised_levenshtein,
        # Add the correct_ref_coverage field to the surrounding dictionary so it appears in outputs or returned metadata.
        "correct_ref_coverage": metrics.correct_ref_coverage,
        # Add the missing_ref_coverage field to the surrounding dictionary so it appears in outputs or returned metadata.
        "missing_ref_coverage": metrics.missing_ref_coverage,
        # Add the repetition_on_reference field to the surrounding dictionary so it appears in outputs or returned metadata.
        "repetition_on_reference": metrics.repetition_on_reference,
        # Add the hallucination field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hallucination": metrics.hallucination,
    }


# Define the build_result_row function; its body below performs one named step of the pipeline.
def build_result_row(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the document field so this data object records that value explicitly.
    document: RunfileDocument,
    # Define the config field so this data object records that value explicitly.
    config: PipelineConfig,
    # Define the ref_to_pred_matrix_source field so this data object records that value explicitly.
    ref_to_pred_matrix_source: str,
    # Define the ref_to_ref_matrix_source field so this data object records that value explicitly.
    ref_to_ref_matrix_source: str,
    # Define the ref_to_pred_matrix_reason field so this data object records that value explicitly.
    ref_to_pred_matrix_reason: str | None,
    # Define the ref_to_ref_matrix_reason field so this data object records that value explicitly.
    ref_to_ref_matrix_reason: str | None,
    # Define the ref_to_pred_shape field so this data object records that value explicitly.
    ref_to_pred_shape: tuple[int, int],
    # Define the ref_to_ref_shape field so this data object records that value explicitly.
    ref_to_ref_shape: tuple[int, int],
    # Define the ref_to_pred_floor field so this data object records that value explicitly.
    ref_to_pred_floor: ScoreFloorResult,
    # Define the ref_to_ref_floor field so this data object records that value explicitly.
    ref_to_ref_floor: ScoreFloorResult,
    # Define the ref_to_pred_hough field so this data object records that value explicitly.
    ref_to_pred_hough: HoughFilteredPayload,
    # Define the ref_to_ref_hough field so this data object records that value explicitly.
    ref_to_ref_hough: HoughFilteredPayload | None,
    # Define the scored field so this data object records that value explicitly.
    scored: ScoredDocumentResult,
    # Define the timing_matrix_seconds field so this data object records that value explicitly.
    timing_matrix_seconds: float,
    # Define the timing_preprocessing_seconds field so this data object records that value explicitly.
    timing_preprocessing_seconds: float,
    # Define the timing_total_seconds field so this data object records that value explicitly.
    timing_total_seconds: float,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> dict[str, Any]:
    """Build the flat document result row without selection or support scores."""
    # Compute or store hough so later code can reuse this named value clearly.
    hough = config.hough_parameters
    # Compute or store row so later code can reuse this named value clearly.
    row = {
        # Add the document_index field to the surrounding dictionary so it appears in outputs or returned metadata.
        "document_index": int(document.document_index),
        # Add the fname field to the surrounding dictionary so it appears in outputs or returned metadata.
        "fname": str(document.fname),
        # Add the main_language field to the surrounding dictionary so it appears in outputs or returned metadata.
        "main_language": str(document.main_language),
        # Add the document_type field to the surrounding dictionary so it appears in outputs or returned metadata.
        "document_type": str(document.document_type),
        # Add the matrix_source_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "matrix_source_ref_to_pred": ref_to_pred_matrix_source,
        # Add the matrix_source_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "matrix_source_ref_to_ref": ref_to_ref_matrix_source,
        # Add the matrix_load_reason_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "matrix_load_reason_ref_to_pred": ref_to_pred_matrix_reason,
        # Add the matrix_load_reason_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "matrix_load_reason_ref_to_ref": ref_to_ref_matrix_reason,
        # Add the row_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "row_count": int(ref_to_pred_shape[0]),
        # Add the column_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "column_count": int(ref_to_pred_shape[1]),
        # Add the ref_to_ref_row_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "ref_to_ref_row_count": int(ref_to_ref_shape[0]),
        # Add the ref_to_ref_column_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "ref_to_ref_column_count": int(ref_to_ref_shape[1]),
        # Add the score_floor_alpha field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_floor_alpha": float(config.score_floor_alpha),
        # Add the score_mean_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_mean_ref_to_pred": ref_to_pred_floor.score_mean,
        # Add the score_standard_deviation_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_standard_deviation_ref_to_pred": ref_to_pred_floor.score_standard_deviation,
        # Add the score_floor_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_floor_ref_to_pred": ref_to_pred_floor.score_floor,
        # Add the active_cell_count_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "active_cell_count_ref_to_pred": ref_to_pred_floor.active_cell_count,
        # Add the active_fraction_ref_to_pred field to the surrounding dictionary so it appears in outputs or returned metadata.
        "active_fraction_ref_to_pred": ref_to_pred_floor.active_fraction,
        # Add the score_mean_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_mean_ref_to_ref": ref_to_ref_floor.score_mean,
        # Add the score_standard_deviation_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_standard_deviation_ref_to_ref": ref_to_ref_floor.score_standard_deviation,
        # Add the score_floor_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "score_floor_ref_to_ref": ref_to_ref_floor.score_floor,
        # Add the active_cell_count_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "active_cell_count_ref_to_ref": ref_to_ref_floor.active_cell_count,
        # Add the active_fraction_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "active_fraction_ref_to_ref": ref_to_ref_floor.active_fraction,
        # Add the hough_threshold field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_threshold": int(hough.hough_threshold),
        # Add the hough_line_length field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_line_length": int(hough.hough_line_length),
        # Add the hough_line_gap field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_line_gap": int(hough.hough_line_gap),
        # Add the hough_seed field to the surrounding dictionary so it appears in outputs or returned metadata.
        "hough_seed": int(hough.hough_seed),
        # Add the align_min_iou_threshold field to the surrounding dictionary so it appears in outputs or returned metadata.
        "align_min_iou_threshold": float(config.align_min_iou_threshold),
        # Add the min_surviving_line_nls field to the surrounding dictionary so it appears in outputs or returned metadata.
        "min_surviving_line_nls": config.min_surviving_line_nls,
        # Add the raw_line_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "raw_line_count": int(ref_to_pred_hough.raw_line_count),
        # Add the candidate_line_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "candidate_line_count": int(ref_to_pred_hough.candidate_line_count),
        # Add the used_line_count field to the surrounding dictionary so it appears in outputs or returned metadata.
        "used_line_count": int(len(scored.ref_to_pred_payload.hough_payload.filtered_result.get("lines_used", []))),
        # Add the raw_line_count_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "raw_line_count_ref_to_ref": 0 if ref_to_ref_hough is None else int(ref_to_ref_hough.raw_line_count),
        # Add the candidate_line_count_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "candidate_line_count_ref_to_ref": 0 if ref_to_ref_hough is None else int(ref_to_ref_hough.candidate_line_count),
        # Add the used_line_count_ref_to_ref field to the surrounding dictionary so it appears in outputs or returned metadata.
        "used_line_count_ref_to_ref": 0 if ref_to_ref_hough is None else int(ref_to_ref_hough.used_line_count),
        # Add the timing_matrix_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_matrix_seconds": float(timing_matrix_seconds),
        # Add the timing_preprocessing_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_preprocessing_seconds": float(timing_preprocessing_seconds),
        # Add the timing_hough_detect_ref_to_pred_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_hough_detect_ref_to_pred_seconds": float(ref_to_pred_hough.detect_seconds),
        # Add the timing_filter_ref_to_pred_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_filter_ref_to_pred_seconds": float(ref_to_pred_hough.filter_seconds),
        # Add the timing_hough_detect_ref_to_ref_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_hough_detect_ref_to_ref_seconds": 0.0 if ref_to_ref_hough is None else float(ref_to_ref_hough.detect_seconds),
        # Add the timing_filter_ref_to_ref_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_filter_ref_to_ref_seconds": 0.0 if ref_to_ref_hough is None else float(ref_to_ref_hough.filter_seconds),
        # Add the timing_coverage_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_coverage_seconds": float(scored.coverage_seconds),
        # Add the timing_levenshtein_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_levenshtein_seconds": float(scored.levenshtein_seconds),
        # Add the timing_total_seconds field to the surrounding dictionary so it appears in outputs or returned metadata.
        "timing_total_seconds": float(timing_total_seconds),
    }
    # Execute this statement as the next small step in the surrounding pipeline logic.
    row.update(metrics_to_row_fields(scored.metrics))
    # Store v2.12-compatible reference-axis subtraction diagnostics next to the public metrics.
    row.update(
        {
            "coverage_invalid_reason": scored.coverage_invalid_reason,
            "coverage_invalid_error_message": scored.coverage_invalid_error_message,
            **dict(scored.coverage_diagnostics or {}),
        }
    )
    # Return this computed value to the caller so the next pipeline stage can use it.
    return row


# Define the process_one_document function; its body below performs one named step of the pipeline.
def process_one_document(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the document field so this data object records that value explicitly.
    document: RunfileDocument,
    # Define the config field so this data object records that value explicitly.
    config: PipelineConfig,
    # Define the indexes field so this data object records that value explicitly.
    indexes: ScoreMatrixIndexBundle,
    # Pass this value into the surrounding multi-line call or collection.
    log,
    # Define the keep_plot_payload field so this data object records that value explicitly.
    keep_plot_payload: bool,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> DocumentRunResult:
    """Run one document and release large arrays when the caller discards the result."""
    # Compute or store document_started_at so later code can reuse this named value clearly.
    document_started_at = time.perf_counter()
    # Compute or store base_document_row so later code can reuse this named value clearly.
    base_document_row = document_table_row(document, window_size=config.window_size, window_stride=config.window_stride)
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        f"[document] {document.fname} metadata "
        f"language={document.main_language} "
        f"document_type={document.document_type} "
        f"reference_chars={len(document.reference_text)} "
        f"prediction_chars={len(document.prediction_text)} "
        f"reference_windows={base_document_row['reference_window_count']} "
        f"prediction_windows={base_document_row['prediction_window_count']} "
        f"window_size={int(config.window_size)} "
        f"window_stride={int(config.window_stride)} "
        f"plot_payload_requested={bool(keep_plot_payload)}"
    )

    # Define the try field so this data object records that value explicitly.
    try:
        # Compute or store matrix_started_at so later code can reuse this named value clearly.
        matrix_started_at = time.perf_counter()
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[matrix] {document.fname} load start")
        # Compute or store ref_to_pred_loaded so later code can reuse this named value clearly.
        ref_to_pred_loaded = load_or_compute_ref_to_pred_matrix(
            # Pass the scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
            scores_pkl=config.scores_pkl_ref_to_pred,
            # Pass the score_index_by_fname argument into the surrounding call so the callee receives that setting explicitly.
            score_index_by_fname=indexes.ref_to_pred_index,
            # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
            fname=document.fname,
            # Pass reference_text into the surrounding call; this supplies the normalized reference transcription for this document.
            reference_text=document.reference_text,
            # Pass prediction_text into the surrounding call; this supplies the normalized model prediction for this document.
            prediction_text=document.prediction_text,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=config.window_size,
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=config.window_stride,
            # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
            log=log,
        )
        # Compute or store ref_to_ref_loaded so later code can reuse this named value clearly.
        ref_to_ref_loaded = load_or_compute_ref_to_ref_matrix(
            # Pass the scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
            scores_pkl=config.scores_pkl_ref_to_ref,
            # Pass the score_index_by_fname argument into the surrounding call so the callee receives that setting explicitly.
            score_index_by_fname=indexes.ref_to_ref_index,
            # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
            fname=document.fname,
            # Pass reference_text into the surrounding call; this supplies the normalized reference transcription for this document.
            reference_text=document.reference_text,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=config.window_size,
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=config.window_stride,
            # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
            log=log,
        )
        # Compute or store timing_matrix_seconds so later code can reuse this named value clearly.
        timing_matrix_seconds = time.perf_counter() - matrix_started_at

        # Use NumPy here because matrix operations should run on compact numeric arrays.
        ref_to_pred_matrix = np.asarray(ref_to_pred_loaded.matrix, dtype=float)
        # Use NumPy here because matrix operations should run on compact numeric arrays.
        ref_to_ref_matrix = np.asarray(ref_to_ref_loaded.matrix, dtype=float)
        # Compute or store ref_to_pred_shape so later code can reuse this named value clearly.
        ref_to_pred_shape = tuple(int(value) for value in ref_to_pred_matrix.shape)
        # Compute or store ref_to_ref_shape so later code can reuse this named value clearly.
        ref_to_ref_shape = tuple(int(value) for value in ref_to_ref_matrix.shape)
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(
            f"[matrix] {document.fname} load done "
            f"ref_to_pred_source={ref_to_pred_loaded.source} "
            f"ref_to_pred_reason={format_log_value(ref_to_pred_loaded.reason)} "
            f"ref_to_pred_shape={matrix_shape_text(ref_to_pred_shape)} "
            f"ref_to_ref_source={ref_to_ref_loaded.source} "
            f"ref_to_ref_reason={format_log_value(ref_to_ref_loaded.reason)} "
            f"ref_to_ref_shape={matrix_shape_text(ref_to_ref_shape)} "
            f"seconds={format_log_value(timing_matrix_seconds)}"
        )

        # Compute or store size_skip_reason so later code can reuse this named value clearly.
        size_skip_reason = matrix_size_skip_reason(
            # Pass this value into the surrounding multi-line call or collection.
            ref_to_pred_shape,
            # Pass the minimum_rows argument into the surrounding call so the callee receives that setting explicitly.
            minimum_rows=config.minimum_matrix_rows,
            # Pass the minimum_columns argument into the surrounding call so the callee receives that setting explicitly.
            minimum_columns=config.minimum_matrix_columns,
        )
        # Check whether size_skip_reason is not None; the indented block handles that specific case.
        if size_skip_reason is not None:
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(
                f"[matrix] {document.fname} size check failed "
                f"reason={size_skip_reason} "
                f"minimum_rows={int(config.minimum_matrix_rows)} "
                f"minimum_columns={int(config.minimum_matrix_columns)} "
                f"seconds={format_log_value(time.perf_counter() - document_started_at)}"
            )
            # Return this computed value to the caller so the next pipeline stage can use it.
            return DocumentRunResult(
                # Pass the result_row argument into the surrounding call so the callee receives that setting explicitly.
                result_row=None,
                # Compute or store skipped_row so later code can reuse this named value clearly.
                skipped_row=skipped_row_from_document(
                    # Pass this value into the surrounding multi-line call or collection.
                    document,
                    # Pass the config argument into the surrounding call so the callee receives that setting explicitly.
                    config=config,
                    # Pass the skip_stage argument into the surrounding call so the callee receives that setting explicitly.
                    skip_stage="matrix_size",
                    # Pass the skip_reason argument into the surrounding call so the callee receives that setting explicitly.
                    skip_reason=size_skip_reason,
                    # Pass the matrix_shape argument into the surrounding call so the callee receives that setting explicitly.
                    matrix_shape=ref_to_pred_shape,
                ),
                # Pass the loadable_row argument into the surrounding call so the callee receives that setting explicitly.
                loadable_row=None,
                # Pass the loaded_row argument into the surrounding call so the callee receives that setting explicitly.
                loaded_row=None,
                # Pass the plot_payload argument into the surrounding call so the callee receives that setting explicitly.
                plot_payload=None,
            )

        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(
            f"[matrix] {document.fname} size check passed "
            f"minimum_rows={int(config.minimum_matrix_rows)} "
            f"minimum_columns={int(config.minimum_matrix_columns)}"
        )
        # Compute or store preprocessing_started_at so later code can reuse this named value clearly.
        preprocessing_started_at = time.perf_counter()
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[preprocess] {document.fname} score floor start")
        # Compute or store ref_to_pred_floor so later code can reuse this named value clearly.
        ref_to_pred_floor = compute_score_floor_mask(ref_to_pred_matrix, alpha=config.score_floor_alpha)
        # Compute or store ref_to_ref_floor so later code can reuse this named value clearly.
        ref_to_ref_floor = compute_score_floor_mask(ref_to_ref_matrix, alpha=config.score_floor_alpha)
        # Compute or store timing_preprocessing_seconds so later code can reuse this named value clearly.
        timing_preprocessing_seconds = time.perf_counter() - preprocessing_started_at
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log_score_floor_summary(log, fname=document.fname, label="ref_to_pred", score_floor_result=ref_to_pred_floor)
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log_score_floor_summary(log, fname=document.fname, label="ref_to_ref", score_floor_result=ref_to_ref_floor)
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[preprocess] {document.fname} score floor done seconds={format_log_value(timing_preprocessing_seconds)}")

        # Compute or store hough so later code can reuse this named value clearly.
        hough = config.hough_parameters
        # Check whether ref_to_pred_floor.active_cell_count > 0; the indented block handles that specific case.
        if ref_to_pred_floor.active_cell_count > 0:
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(
                f"[hough] {document.fname} ref_to_pred start "
                f"active_cells={int(ref_to_pred_floor.active_cell_count)} "
                f"threshold={int(hough.hough_threshold)} "
                f"line_length={int(hough.hough_line_length)} "
                f"line_gap={int(hough.hough_line_gap)} "
                f"seed={int(hough.hough_seed)}"
            )
            # Compute or store ref_to_pred_hough so later code can reuse this named value clearly.
            ref_to_pred_hough = run_probabilistic_hough_and_filter(
                # Pass the score_matrix argument into the surrounding call so the callee receives that setting explicitly.
                score_matrix=ref_to_pred_matrix,
                # Pass hough_input_mask into the surrounding call; this supplies the boolean matrix passed to Hough, where True means this cell can vote for a line.
                hough_input_mask=ref_to_pred_floor.hough_input_mask,
                # Pass score_floor into the surrounding call; this supplies the numeric cutoff a score must meet before it can become a Hough voter.
                score_floor=ref_to_pred_floor.score_floor,
                # Pass the hough_threshold argument into the surrounding call so the callee receives that setting explicitly.
                hough_threshold=hough.hough_threshold,
                # Pass the hough_line_length argument into the surrounding call so the callee receives that setting explicitly.
                hough_line_length=hough.hough_line_length,
                # Pass the hough_line_gap argument into the surrounding call so the callee receives that setting explicitly.
                hough_line_gap=hough.hough_line_gap,
                # Pass the hough_seed argument into the surrounding call so the callee receives that setting explicitly.
                hough_seed=hough.hough_seed,
                # Pass align_min_iou_threshold into the surrounding call; this supplies the overlap threshold used when assigning line coverage to text windows.
                align_min_iou_threshold=config.align_min_iou_threshold,
            )
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log_hough_summary(log, fname=document.fname, label="ref_to_pred", hough_payload=ref_to_pred_hough)
        # Define the else field so this data object records that value explicitly.
        else:
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[hough] {document.fname} ref_to_pred skipped reason=no_active_cells seconds=0.000000")
            # Compute or store ref_to_pred_hough so later code can reuse this named value clearly.
            ref_to_pred_hough = empty_hough_payload(score_floor_result=ref_to_pred_floor)

        # Compute or store reference_windows so later code can reuse this named value clearly.
        reference_windows = sliding_text_windows(
            # Pass this value into the surrounding multi-line call or collection.
            document.reference_text,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=config.window_size,
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=config.window_stride,
        )
        # Compute or store prediction_windows so later code can reuse this named value clearly.
        prediction_windows = sliding_text_windows(
            # Pass this value into the surrounding multi-line call or collection.
            document.prediction_text,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=config.window_size,
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=config.window_stride,
        )
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(
            f"[windows] {document.fname} built "
            f"reference_windows={len(reference_windows)} "
            f"prediction_windows={len(prediction_windows)}"
        )
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(
            f"[text-filter] {document.fname} start "
            f"minimum_line_nls={format_log_value(config.min_surviving_line_nls)} "
            f"input_lines={len(ref_to_pred_hough.filtered_result.get('lines_used', []))}"
        )
        # Compute or store line_filter_result so later code can reuse this named value clearly.
        line_filter_result = filter_lines_by_minimum_normalised_levenshtein(
            # Pass the filtered_result argument into the surrounding call so the callee receives that setting explicitly.
            filtered_result=ref_to_pred_hough.filtered_result,
            # Pass the reference_windows argument into the surrounding call so the callee receives that setting explicitly.
            reference_windows=reference_windows,
            # Pass the prediction_windows argument into the surrounding call so the callee receives that setting explicitly.
            prediction_windows=prediction_windows,
            # Pass the reference_window_count argument into the surrounding call so the callee receives that setting explicitly.
            reference_window_count=ref_to_pred_shape[0],
            # Pass the minimum_line_nls argument into the surrounding call so the callee receives that setting explicitly.
            minimum_line_nls=config.min_surviving_line_nls,
        )
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log_line_filter_summary(log, fname=document.fname, line_filter_result=line_filter_result)

        # Check whether line_filter_result.filtered_result.get("lines_used") and ref_to_ref_floor.active_cell_count > 0; the indented block handles that specific case.
        if line_filter_result.filtered_result.get("lines_used") and ref_to_ref_floor.active_cell_count > 0:
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(
                f"[hough] {document.fname} ref_to_ref start "
                f"active_cells={int(ref_to_ref_floor.active_cell_count)} "
                f"threshold={int(hough.hough_threshold)} "
                f"line_length={int(hough.hough_line_length)} "
                f"line_gap={int(hough.hough_line_gap)} "
                f"seed={int(hough.hough_seed)}"
            )
            # Compute or store ref_to_ref_hough so later code can reuse this named value clearly.
            ref_to_ref_hough = run_probabilistic_hough_and_filter(
                # Pass the score_matrix argument into the surrounding call so the callee receives that setting explicitly.
                score_matrix=ref_to_ref_matrix,
                # Pass hough_input_mask into the surrounding call; this supplies the boolean matrix passed to Hough, where True means this cell can vote for a line.
                hough_input_mask=ref_to_ref_floor.hough_input_mask,
                # Pass score_floor into the surrounding call; this supplies the numeric cutoff a score must meet before it can become a Hough voter.
                score_floor=ref_to_ref_floor.score_floor,
                # Pass the hough_threshold argument into the surrounding call so the callee receives that setting explicitly.
                hough_threshold=hough.hough_threshold,
                # Pass the hough_line_length argument into the surrounding call so the callee receives that setting explicitly.
                hough_line_length=hough.hough_line_length,
                # Pass the hough_line_gap argument into the surrounding call so the callee receives that setting explicitly.
                hough_line_gap=hough.hough_line_gap,
                # Pass the hough_seed argument into the surrounding call so the callee receives that setting explicitly.
                hough_seed=hough.hough_seed,
                # Pass align_min_iou_threshold into the surrounding call; this supplies the overlap threshold used when assigning line coverage to text windows.
                align_min_iou_threshold=config.align_min_iou_threshold,
            )
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log_hough_summary(log, fname=document.fname, label="ref_to_ref", hough_payload=ref_to_ref_hough)
        # Define the else field so this data object records that value explicitly.
        else:
            # Compute or store ref_to_ref_skip_reason so later code can reuse this named value clearly.
            ref_to_ref_skip_reason = "no_surviving_ref_to_pred_lines"
            # Check whether ref_to_ref_floor.active_cell_count <= 0; the indented block handles that specific case.
            if ref_to_ref_floor.active_cell_count <= 0:
                # Compute or store ref_to_ref_skip_reason so later code can reuse this named value clearly.
                ref_to_ref_skip_reason = "no_ref_to_ref_active_cells"
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[hough] {document.fname} ref_to_ref skipped reason={ref_to_ref_skip_reason} seconds=0.000000")
            # Compute or store ref_to_ref_hough so later code can reuse this named value clearly.
            ref_to_ref_hough = None

        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[scoring] {document.fname} start")
        # Compute or store scored so later code can reuse this named value clearly.
        scored = score_document_alignment(
            # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
            fname=document.fname,
            # Pass reference_text into the surrounding call; this supplies the normalized reference transcription for this document.
            reference_text=document.reference_text,
            # Pass prediction_text into the surrounding call; this supplies the normalized model prediction for this document.
            prediction_text=document.prediction_text,
            # Pass the reference_windows argument into the surrounding call so the callee receives that setting explicitly.
            reference_windows=reference_windows,
            # Pass the prediction_windows argument into the surrounding call so the callee receives that setting explicitly.
            prediction_windows=prediction_windows,
            # Pass the ref_to_pred_hough_payload argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_pred_hough_payload=ref_to_pred_hough,
            # Pass the ref_to_ref_hough_payload argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_ref_hough_payload=ref_to_ref_hough,
            # Pass the line_text_filter_result argument into the surrounding call so the callee receives that setting explicitly.
            line_text_filter_result=line_filter_result,
            # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
            window_size=config.window_size,
            # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
            window_stride=config.window_stride,
        )

        # Compute or store timing_total_seconds so later code can reuse this named value clearly.
        timing_total_seconds = time.perf_counter() - document_started_at
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log_scoring_summary(log, fname=document.fname, scored=scored, total_seconds=timing_total_seconds)
        # Compute or store result_row so later code can reuse this named value clearly.
        result_row = build_result_row(
            # Pass the document argument into the surrounding call so the callee receives that setting explicitly.
            document=document,
            # Pass the config argument into the surrounding call so the callee receives that setting explicitly.
            config=config,
            # Pass the ref_to_pred_matrix_source argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_pred_matrix_source=ref_to_pred_loaded.source,
            # Pass the ref_to_ref_matrix_source argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_ref_matrix_source=ref_to_ref_loaded.source,
            # Pass the ref_to_pred_matrix_reason argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_pred_matrix_reason=ref_to_pred_loaded.reason,
            # Pass the ref_to_ref_matrix_reason argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_ref_matrix_reason=ref_to_ref_loaded.reason,
            # Pass the ref_to_pred_shape argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_pred_shape=ref_to_pred_shape,
            # Pass the ref_to_ref_shape argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_ref_shape=ref_to_ref_shape,
            # Pass the ref_to_pred_floor argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_pred_floor=ref_to_pred_floor,
            # Pass the ref_to_ref_floor argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_ref_floor=ref_to_ref_floor,
            # Pass the ref_to_pred_hough argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_pred_hough=ref_to_pred_hough,
            # Pass the ref_to_ref_hough argument into the surrounding call so the callee receives that setting explicitly.
            ref_to_ref_hough=ref_to_ref_hough,
            # Pass the scored argument into the surrounding call so the callee receives that setting explicitly.
            scored=scored,
            # Pass the timing_matrix_seconds argument into the surrounding call so the callee receives that setting explicitly.
            timing_matrix_seconds=timing_matrix_seconds,
            # Pass the timing_preprocessing_seconds argument into the surrounding call so the callee receives that setting explicitly.
            timing_preprocessing_seconds=timing_preprocessing_seconds,
            # Pass the timing_total_seconds argument into the surrounding call so the callee receives that setting explicitly.
            timing_total_seconds=timing_total_seconds,
        )

        # Compute or store plot_payload so later code can reuse this named value clearly.
        plot_payload = None
        # Check whether keep_plot_payload; the indented block handles that specific case.
        if keep_plot_payload:
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[plot] {document.fname} payload build start")
            # Compute or store plot_payload so later code can reuse this named value clearly.
            plot_payload = {
                # Add the document field to the surrounding dictionary so it appears in outputs or returned metadata.
                "document": document,
                # Add the result_row field to the surrounding dictionary so it appears in outputs or returned metadata.
                "result_row": result_row,
                # Add the ref_to_pred_score_matrix field to the surrounding dictionary so it appears in outputs or returned metadata.
                "ref_to_pred_score_matrix": ref_to_pred_matrix,
                # Add the ref_to_ref_score_matrix field to the surrounding dictionary so it appears in outputs or returned metadata.
                "ref_to_ref_score_matrix": ref_to_ref_matrix,
                # Add the ref_to_pred_hough_input_mask field to the surrounding dictionary so it appears in outputs or returned metadata.
                "ref_to_pred_hough_input_mask": ref_to_pred_floor.hough_input_mask,
                # Add the raw_ref_to_pred_hough_lines field to the surrounding dictionary so it appears in outputs or returned metadata.
                "raw_ref_to_pred_hough_lines": list(ref_to_pred_hough.detection_result.get("raw_lines", [])),
                # Add the final_surviving_ref_to_pred_lines field to the surrounding dictionary so it appears in outputs or returned metadata.
                "final_surviving_ref_to_pred_lines": list(scored.ref_to_pred_payload.hough_payload.filtered_result.get("lines_used", [])),
            }
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(
                f"[plot] {document.fname} payload ready "
                f"raw_lines={len(plot_payload['raw_ref_to_pred_hough_lines'])} "
                f"final_lines={len(plot_payload['final_surviving_ref_to_pred_lines'])}"
            )
        # Define the else field so this data object records that value explicitly.
        else:
            # Write a progress message so long runs are understandable from terminal or Slurm output.
            log(f"[plot] {document.fname} payload skipped reason=plotting_disabled")

        # Return this computed value to the caller so the next pipeline stage can use it.
        return DocumentRunResult(
            # Pass the result_row argument into the surrounding call so the callee receives that setting explicitly.
            result_row=result_row,
            # Pass the skipped_row argument into the surrounding call so the callee receives that setting explicitly.
            skipped_row=None,
            # Pass the loadable_row argument into the surrounding call so the callee receives that setting explicitly.
            loadable_row=base_document_row,
            # Pass the loaded_row argument into the surrounding call so the callee receives that setting explicitly.
            loaded_row=base_document_row,
            # Pass the plot_payload argument into the surrounding call so the callee receives that setting explicitly.
            plot_payload=plot_payload,
        )
    # Catch the matching failure type and turn it into explicit handling instead of crashing silently.
    except Exception as exc:
        # Write a progress message so long runs are understandable from terminal or Slurm output.
        log(f"[exception] {document.fname} error={repr(exc)} seconds={format_log_value(time.perf_counter() - document_started_at)}")
        # Return this computed value to the caller so the next pipeline stage can use it.
        return DocumentRunResult(
            # Pass the result_row argument into the surrounding call so the callee receives that setting explicitly.
            result_row=None,
            # Compute or store skipped_row so later code can reuse this named value clearly.
            skipped_row=skipped_row_from_document(
                # Pass this value into the surrounding multi-line call or collection.
                document,
                # Pass the config argument into the surrounding call so the callee receives that setting explicitly.
                config=config,
                # Pass the skip_stage argument into the surrounding call so the callee receives that setting explicitly.
                skip_stage="exception",
                # Pass the skip_reason argument into the surrounding call so the callee receives that setting explicitly.
                skip_reason=repr(exc),
            ),
            # Pass the loadable_row argument into the surrounding call so the callee receives that setting explicitly.
            loadable_row=None,
            # Pass the loaded_row argument into the surrounding call so the callee receives that setting explicitly.
            loaded_row=None,
            # Pass the plot_payload argument into the surrounding call so the callee receives that setting explicitly.
            plot_payload=None,
        )


__all__ = ["DocumentRunResult", "document_table_row", "process_one_document", "skipped_row_from_document"]
