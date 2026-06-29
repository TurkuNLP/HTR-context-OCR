from __future__ import annotations

"""Process one document through loading, Hough detection, filtering, and scoring."""

from dataclasses import dataclass, replace
import math
import os
from pathlib import Path
import pickle
import time
from typing import Any

import json

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein.config.pipeline_config import PipelineConfig
from tuner_simple_alpha_sweep_pre_iou_levenshtein.document_selection.runfile_loader import RunfileDocument
from tuner_simple_alpha_sweep_pre_iou_levenshtein.matrix_operations.matrix_loader import (
    # Pass this value into the surrounding multi-line call or collection.
    ScoreMatrixIndexBundle,
    # Pass this value into the surrounding multi-line call or collection.
    load_or_compute_ref_to_pred_matrix,
    # Pass this value into the surrounding multi-line call or collection.
    load_or_compute_ref_to_ref_matrix,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.matrix_operations.matrix_shape import (
    # Pass this value into the surrounding multi-line call or collection.
    count_sliding_windows,
    # Pass this value into the surrounding multi-line call or collection.
    matrix_size_skip_reason,
    # Pass this value into the surrounding multi-line call or collection.
    sliding_text_windows,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.matrix_operations.score_floor import (
    ScoreFloorResult,
    ScoreFloorStatistics,
    compute_minimum_levenshtein_mask,
    compute_score_floor_mask_from_statistics,
    compute_score_floor_statistics,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.probabilistic_hough.hough_detection import HoughFilteredPayload, run_probabilistic_hough_and_filter
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.levenshtein import normalized_levenshtein_similarity
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.line_text_similarity import filter_lines_by_minimum_normalised_levenshtein
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.scoring_pipeline import ScoredDocumentResult, score_document_alignment, zero_alignment_metrics
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.scoring_pipeline import DocumentAlignmentMetrics


def _serialise_per_run_counts(detection_result: dict) -> str:
    """Return the per-run raw falling-line counts as a compact JSON array."""
    counts = detection_result.get("hough_per_run_raw_counts", [])
    return json.dumps(counts)


def _serialise_segment_hit_counts(detection_result: dict) -> str:
    """Return a JSON array describing how many runs detected each union segment.

    Each element is {"x0": ..., "y0": ..., "x1": ..., "y1": ...,
                     "hit_count": N, "miss_count": M}
    sorted by (x0, y0, x1, y1) so the upper-left-most segment is first.
    M = hough_num_runs - hit_count, i.e. the number of runs where the
    segment was absent entirely.
    """
    hit_counts: dict = detection_result.get("hough_segment_run_hit_counts", {})
    num_runs: int = int(detection_result.get("hough_num_runs", 1))
    records = []
    for seg, hit in sorted(hit_counts.items()):
        (x0, y0), (x1, y1) = seg
        records.append({
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "hit_count": hit,
            "miss_count": num_runs - hit,
        })
    return json.dumps(records)


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
    # Path to the per-document alpha sweep pickle, when sweep mode wrote one.
    alpha_sweep_pickle_path: str | None = None


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
    from tuner_simple_alpha_sweep_pre_iou_levenshtein.probabilistic_hough.hough_input import build_simple_hough_context

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
        "hough_skipped_reason": "empty_pre_hough_mask",
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
        detection_result={
            "raw_lines": [],
            "candidate_segments": [],
            "threshold_start": score_floor_result.score_floor,
            "hough_skipped_reason": "empty_pre_hough_mask",
            "skimage_raw_line_count_before_direction_filter": 0,
            "direction_rejected_line_count": 0,
        },
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
        "pre_hough_mask_kind": pre_hough_mask_kind(config),
        "minimum_pre_hough_levenshtein": config.minimum_pre_hough_levenshtein,
        "harmonic_mode": str(config.alpha_selection_harmonic_mode),
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
        "hough_seed": None if hough.hough_seed is None else int(hough.hough_seed),
        # Add the align_min_iou_threshold field to the surrounding dictionary so it appears in outputs or returned metadata.
        "align_min_iou_threshold": float(config.align_min_iou_threshold),
        # Add the min_surviving_line_nls field to the surrounding dictionary so it appears in outputs or returned metadata.
        "min_surviving_line_nls": config.min_surviving_line_nls,
        "line_levenshtein_filter_stage": str(ref_to_pred_hough.detection_result.get("pre_iou_line_levenshtein_filter_stage", "pre_iou_raw_hough")),
        "pre_iou_line_levenshtein_filter_enabled": bool(ref_to_pred_hough.detection_result.get("pre_iou_line_levenshtein_filter_enabled", False)),
        "raw_falling_line_count_before_pre_iou_levenshtein": int(ref_to_pred_hough.detection_result.get("raw_falling_line_count_before_pre_iou_levenshtein", ref_to_pred_hough.raw_line_count)),
        "raw_falling_line_count_after_pre_iou_levenshtein": int(ref_to_pred_hough.detection_result.get("raw_falling_line_count_after_pre_iou_levenshtein", ref_to_pred_hough.raw_line_count)),
        "raw_falling_line_levenshtein_removed_count": int(ref_to_pred_hough.detection_result.get("raw_falling_line_levenshtein_removed_count", 0)),
        "raw_falling_line_levenshtein_threshold": ref_to_pred_hough.detection_result.get("raw_falling_line_levenshtein_threshold"),
        "pre_iou_line_levenshtein_min": ref_to_pred_hough.detection_result.get("pre_iou_line_levenshtein_min"),
        "pre_iou_line_levenshtein_max": ref_to_pred_hough.detection_result.get("pre_iou_line_levenshtein_max"),
        "pre_iou_line_levenshtein_mean": ref_to_pred_hough.detection_result.get("pre_iou_line_levenshtein_mean"),
        "pre_iou_line_levenshtein_seconds": float(ref_to_pred_hough.detection_result.get("pre_iou_line_levenshtein_seconds", 0.0)),
        "hough_num_runs": int(ref_to_pred_hough.detection_result.get("hough_num_runs", 1)),
        "hough_union_unique_segment_count": int(ref_to_pred_hough.detection_result.get("hough_union_unique_segment_count", ref_to_pred_hough.raw_line_count)),
        "hough_per_run_counts_json": _serialise_per_run_counts(ref_to_pred_hough.detection_result),
        "hough_segment_run_hit_counts_json": _serialise_segment_hit_counts(ref_to_pred_hough.detection_result),
        "hough_skip_reason_ref_to_pred": str(ref_to_pred_hough.detection_result.get("hough_skipped_reason", "")),
        "hough_skip_reason_ref_to_ref": (
            str(ref_to_ref_hough.detection_result.get("hough_skipped_reason", ""))
            if ref_to_ref_hough is not None
            else (
                "empty_ref_to_ref_pre_hough_mask"
                if bool(scored.ref_to_pred_payload.hough_payload.filtered_result.get("lines_used"))
                and int(ref_to_ref_floor.active_cell_count) <= 0
                else "no_ref_to_pred_lines_after_filter"
            )
        ),
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



@dataclass
class AlphaCandidateRun:
    """All intermediate state for one alpha candidate while a document is being scored."""

    alpha: float
    selection_harmonic_score: float
    result_row: dict[str, Any]
    ref_to_pred_floor: ScoreFloorResult
    ref_to_ref_floor: ScoreFloorResult
    ref_to_pred_hough: HoughFilteredPayload
    ref_to_ref_hough: HoughFilteredPayload | None
    scored: ScoredDocumentResult
    line_filter_result: Any
    timing_preprocessing_seconds: float
    timing_total_seconds: float


def fixed_minimum_levenshtein_mask_enabled(config: PipelineConfig) -> bool:
    """Return True when the user requested a fixed Levenshtein pre-Hough mask."""

    return config.minimum_pre_hough_levenshtein is not None


def pre_hough_mask_kind(config: PipelineConfig) -> str:
    """Return the public name of the mask-building rule used by this run."""

    if fixed_minimum_levenshtein_mask_enabled(config):
        return "minimum_levenshtein"
    return "score_mean_plus_alpha_standard_deviation"


def alpha_values_for_config(config: PipelineConfig) -> tuple[float, ...]:
    """Return the alpha candidates for this run, including the configured upper bound."""

    if fixed_minimum_levenshtein_mask_enabled(config):
        return (0.0,)
    if not bool(config.alpha_sweep_enabled):
        return (round(float(config.score_floor_alpha), 10),)

    start_alpha = float(config.alpha_sweep_min)
    stop_alpha = float(config.alpha_sweep_max)
    step = float(config.alpha_sweep_step)
    values: list[float] = []
    candidate_index = 0
    epsilon = abs(step) * 1e-7
    while True:
        alpha = start_alpha + candidate_index * step
        if alpha > stop_alpha + epsilon:
            break
        rounded_alpha = round(float(alpha), 10)
        if not values or rounded_alpha != values[-1]:
            values.append(rounded_alpha)
        candidate_index += 1
    if not values:
        values.append(round(start_alpha, 10))
    return tuple(values)


def harmonic_selection_score(metrics: DocumentAlignmentMetrics, *, mode: str = "balanced") -> float:
    """Return a harmonic mean score used to compare and rank alpha candidates.

    The three available modes weight the same three metrics differently:

        balanced
            Equal weight on NLS, coverage, and non-hallucination.
            Formula: 3 / (1/NLS + 1/coverage + 1/(1-hallucination))

        coverage-hallucination-priority
            Coverage and non-hallucination each carry twice the weight of NLS.
            Formula: 5 / (1/NLS + 2/coverage + 2/(1-hallucination))

        coverage-hallucination-only
            NLS is excluded entirely; selection is driven by coverage and
            non-hallucination alone.
            Formula: 2 / (1/coverage + 1/(1-hallucination))

        nls-priority
            NLS carries twice the weight of coverage and non-hallucination.
            Formula: 4 / (2/NLS + 1/coverage + 1/(1-hallucination))

    Returns 0.0 when any required metric is missing, non-finite, or not strictly positive.
    """

    def _safe(value: Any) -> float | None:
        """Return value as a strictly positive finite float, or None when unusable."""
        if value is None:
            return None
        try:
            v = float(value)
        except Exception:
            return None
        return v if math.isfinite(v) and v > 0.0 else None

    coverage = _safe(metrics.correct_ref_coverage)
    non_hallucination = _safe(
        None if metrics.hallucination is None else 1.0 - float(metrics.hallucination)
    )

    if mode == "coverage-hallucination-only":
        # NLS is intentionally excluded from this formula. Selection is driven entirely by how
        # much of the reference is covered and how little is hallucinated.
        if coverage is None or non_hallucination is None:
            return 0.0
        return float(2.0 / (1.0 / coverage + 1.0 / non_hallucination))

    nls = _safe(metrics.weighted_along_lines_normalised_levenshtein)
    if nls is None or coverage is None or non_hallucination is None:
        return 0.0

    if mode == "coverage-hallucination-priority":
        # NLS carries weight 1; coverage and non-hallucination each carry weight 2.
        return float(5.0 / (1.0 / nls + 2.0 / coverage + 2.0 / non_hallucination))

    if mode == "nls-priority":
        # NLS carries weight 2; coverage and non-hallucination each carry weight 1.
        return float(4.0 / (2.0 / nls + 1.0 / coverage + 1.0 / non_hallucination))

    # Default: "balanced" — all three terms carry equal weight.
    return float(3.0 / (1.0 / nls + 1.0 / coverage + 1.0 / non_hallucination))


def numeric_or_negative_one(value: Any) -> float:
    """Convert a metric to a finite tie-breaker value."""

    if value is None:
        return -1.0
    try:
        numeric_value = float(value)
    except Exception:
        return -1.0
    if not math.isfinite(numeric_value):
        return -1.0
    return numeric_value


def candidate_selection_key(candidate: AlphaCandidateRun) -> tuple[float, float, float, float, int, float]:
    """Return a deterministic ordering key for choosing the best alpha candidate."""

    metrics = candidate.scored.metrics
    used_line_count = int(candidate.result_row.get("used_line_count") or 0)
    non_hallucination = None if metrics.hallucination is None else 1.0 - float(metrics.hallucination)
    return (
        float(candidate.selection_harmonic_score),
        numeric_or_negative_one(metrics.weighted_along_lines_normalised_levenshtein),
        numeric_or_negative_one(metrics.correct_ref_coverage),
        numeric_or_negative_one(non_hallucination),
        used_line_count,
        -float(candidate.alpha),
    )


def safe_path_component(value: str) -> str:
    """Return a filesystem-safe path component without importing plotting modules."""

    cleaned = "".join(character if character.isalnum() or character in ("-", "_", ".") else "_" for character in str(value))
    return cleaned.strip("._") or "unknown"


def alpha_sweep_pickle_path_for_document(config: PipelineConfig, document: RunfileDocument) -> Path:
    """Return the final per-document alpha sweep pickle path."""

    language_name = safe_path_component(str(document.main_language or "UNKNOWN"))
    document_name = safe_path_component(str(document.fname))
    return Path(config.output_dir) / "alpha_sweep_pickles" / language_name / f"{document_name}.pkl"


def write_pickle_atomically(path: Path, payload: dict[str, Any]) -> None:
    """Write a pickle via os.replace so dynamic workers never expose partial files."""

    final_path = Path(path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = final_path.with_name(f".{final_path.name}.{os.getpid()}.tmp")
    with temporary_path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(temporary_path, final_path)


def score_floor_summary(score_floor_result: ScoreFloorResult) -> dict[str, Any]:
    """Return compact score-floor statistics without storing the full mask."""

    return {
        "score_mean": float(score_floor_result.score_mean),
        "score_standard_deviation": float(score_floor_result.score_standard_deviation),
        "score_floor_alpha": float(score_floor_result.score_floor_alpha),
        "score_floor": float(score_floor_result.score_floor),
        "active_cell_count": int(score_floor_result.active_cell_count),
        "active_fraction": float(score_floor_result.active_fraction),
        "mask_shape": tuple(int(value) for value in score_floor_result.hough_input_mask.shape),
    }


def hough_summary(
    hough_payload: HoughFilteredPayload | None,
    *,
    final_lines_after_text_filter: list[dict] | None = None,
) -> dict[str, Any] | None:
    """Return Hough counts and line bundles for an alpha candidate."""

    if hough_payload is None:
        return None
    filtered_result = hough_payload.filtered_result or {}
    return {
        "raw_line_count": int(hough_payload.raw_line_count),
        "candidate_line_count": int(hough_payload.candidate_line_count),
        "used_line_count_before_text_filter": int(hough_payload.used_line_count),
        "used_line_count_after_text_filter": int(
            len(final_lines_after_text_filter) if final_lines_after_text_filter is not None else len(filtered_result.get("lines_used", []))
        ),
        "detect_seconds": float(hough_payload.detect_seconds),
        "filter_seconds": float(hough_payload.filter_seconds),
        "raw_lines": list(hough_payload.detection_result.get("raw_lines", [])),
        "raw_lines_before_pre_iou_levenshtein": list(hough_payload.detection_result.get("raw_lines_before_pre_iou_levenshtein", hough_payload.detection_result.get("raw_lines", []))),
        "pre_iou_line_levenshtein_filter_stage": hough_payload.detection_result.get("pre_iou_line_levenshtein_filter_stage"),
        "pre_iou_line_levenshtein_filter_enabled": hough_payload.detection_result.get("pre_iou_line_levenshtein_filter_enabled"),
        "raw_falling_line_count_before_pre_iou_levenshtein": hough_payload.detection_result.get("raw_falling_line_count_before_pre_iou_levenshtein"),
        "raw_falling_line_count_after_pre_iou_levenshtein": hough_payload.detection_result.get("raw_falling_line_count_after_pre_iou_levenshtein"),
        "raw_falling_line_levenshtein_removed_count": hough_payload.detection_result.get("raw_falling_line_levenshtein_removed_count"),
        "raw_falling_line_levenshtein_threshold": hough_payload.detection_result.get("raw_falling_line_levenshtein_threshold"),
        "pre_iou_line_levenshtein_min": hough_payload.detection_result.get("pre_iou_line_levenshtein_min"),
        "pre_iou_line_levenshtein_max": hough_payload.detection_result.get("pre_iou_line_levenshtein_max"),
        "pre_iou_line_levenshtein_mean": hough_payload.detection_result.get("pre_iou_line_levenshtein_mean"),
        "pre_iou_line_levenshtein_seconds": hough_payload.detection_result.get("pre_iou_line_levenshtein_seconds"),
        "pre_iou_line_levenshtein_records": list(hough_payload.detection_result.get("pre_iou_line_levenshtein_records", [])),
        "candidate_lines_for_filtering": list(filtered_result.get("lines_for_filtering", [])),
        "final_lines": list(final_lines_after_text_filter if final_lines_after_text_filter is not None else filtered_result.get("lines_used", [])),
        "hough_skipped_reason": str(hough_payload.detection_result.get("hough_skipped_reason", "")),
    }


def line_filter_summary(line_filter_result: Any) -> dict[str, Any]:
    """Return the line text filter audit fields for a candidate."""

    weighted_result = getattr(line_filter_result, "weighted_result", None)
    if weighted_result is None:
        weighted_summary = None
    else:
        weighted_summary = {
            "weighted_along_lines_nls": weighted_result.weighted_along_lines_nls,
            "unweighted_along_lines_nls": weighted_result.unweighted_along_lines_nls,
            "scored_line_count": int(weighted_result.scored_line_count),
            "total_line_length": float(weighted_result.total_line_length),
            "covered_column_count": int(weighted_result.covered_column_count),
        }
    return {
        "filter_enabled": bool(line_filter_result.filter_enabled),
        "input_line_count": int(line_filter_result.input_line_count),
        "scored_line_count": int(line_filter_result.scored_line_count),
        "removed_line_count": int(line_filter_result.removed_line_count),
        "surviving_line_count": int(line_filter_result.surviving_line_count),
        "removed_column_count": int(line_filter_result.removed_column_count),
        "surviving_column_count": int(line_filter_result.surviving_column_count),
        "all_lines_removed": bool(line_filter_result.all_lines_removed),
        "seconds": float(line_filter_result.seconds),
        "weighted_result": weighted_summary,
    }


def build_plot_payload(
    *,
    document: RunfileDocument,
    result_row: dict[str, Any],
    ref_to_pred_matrix: np.ndarray,
    ref_to_ref_matrix: np.ndarray,
    candidate: AlphaCandidateRun,
) -> dict[str, Any]:
    """Build the existing renderer payload for the selected candidate only."""

    return {
        "document": document,
        "result_row": result_row,
        "ref_to_pred_score_matrix": ref_to_pred_matrix,
        "ref_to_ref_score_matrix": ref_to_ref_matrix,
        "ref_to_pred_hough_input_mask": candidate.ref_to_pred_floor.hough_input_mask,
        "raw_ref_to_pred_hough_lines": list(candidate.ref_to_pred_hough.detection_result.get("raw_lines", [])),
        "final_surviving_ref_to_pred_lines": list(
            candidate.scored.ref_to_pred_payload.hough_payload.filtered_result.get("lines_used", [])
        ),
    }


def build_alpha_candidate_summary(candidate: AlphaCandidateRun) -> dict[str, Any]:
    """Return a compact, pickle-friendly audit record for one alpha candidate."""

    final_ref_to_pred_lines = list(candidate.scored.ref_to_pred_payload.hough_payload.filtered_result.get("lines_used", []))
    return {
        "alpha": float(candidate.alpha),
        "selection_harmonic_score": float(candidate.selection_harmonic_score),
        "result_row": dict(candidate.result_row),
        "metrics": metrics_to_row_fields(candidate.scored.metrics),
        "coverage_invalid_reason": candidate.scored.coverage_invalid_reason,
        "coverage_invalid_error_message": candidate.scored.coverage_invalid_error_message,
        "coverage_diagnostics": dict(candidate.scored.coverage_diagnostics or {}),
        "ref_to_pred_score_floor": score_floor_summary(candidate.ref_to_pred_floor),
        "ref_to_ref_score_floor": score_floor_summary(candidate.ref_to_ref_floor),
        "ref_to_pred_hough": hough_summary(candidate.ref_to_pred_hough, final_lines_after_text_filter=final_ref_to_pred_lines),
        "ref_to_ref_hough": hough_summary(candidate.ref_to_ref_hough),
        "line_text_filter": line_filter_summary(candidate.line_filter_result),
        "timings": {
            "preprocessing_seconds": float(candidate.timing_preprocessing_seconds),
            "hough_detect_ref_to_pred_seconds": float(candidate.ref_to_pred_hough.detect_seconds),
            "filter_ref_to_pred_seconds": float(candidate.ref_to_pred_hough.filter_seconds),
            "hough_detect_ref_to_ref_seconds": 0.0 if candidate.ref_to_ref_hough is None else float(candidate.ref_to_ref_hough.detect_seconds),
            "filter_ref_to_ref_seconds": 0.0 if candidate.ref_to_ref_hough is None else float(candidate.ref_to_ref_hough.filter_seconds),
            "coverage_seconds": float(candidate.scored.coverage_seconds),
            "levenshtein_seconds": float(candidate.scored.levenshtein_seconds),
            "candidate_total_seconds": float(candidate.timing_total_seconds),
        },
    }


def build_pre_hough_mask_result(
    *,
    score_matrix: np.ndarray,
    statistics: ScoreFloorStatistics,
    config: PipelineConfig,
    alpha: float,
) -> ScoreFloorResult:
    """Build the mask selected by the current preprocessing configuration."""

    if fixed_minimum_levenshtein_mask_enabled(config):
        return compute_minimum_levenshtein_mask(
            score_matrix,
            minimum_levenshtein=float(config.minimum_pre_hough_levenshtein),
            statistics=statistics,
        )
    return compute_score_floor_mask_from_statistics(
        score_matrix,
        alpha=float(alpha),
        statistics=statistics,
    )


def run_alpha_candidate(
    *,
    document: RunfileDocument,
    config: PipelineConfig,
    alpha: float,
    ref_to_pred_matrix: np.ndarray,
    ref_to_ref_matrix: np.ndarray,
    ref_to_pred_shape: tuple[int, int],
    ref_to_ref_shape: tuple[int, int],
    ref_to_pred_floor_statistics: ScoreFloorStatistics,
    ref_to_ref_floor_statistics: ScoreFloorStatistics,
    ref_to_pred_matrix_source: str,
    ref_to_ref_matrix_source: str,
    ref_to_pred_matrix_reason: str | None,
    ref_to_ref_matrix_reason: str | None,
    reference_windows: list[str],
    prediction_windows: list[str],
    timing_matrix_seconds: float,
    document_normalised_levenshtein: float | None = None,
) -> AlphaCandidateRun:
    """Run preprocessing, Hough, text filtering, and scoring for one alpha."""

    candidate_started_at = time.perf_counter()
    candidate_config = replace(config, score_floor_alpha=float(alpha))
    preprocessing_started_at = time.perf_counter()
    ref_to_pred_floor = build_pre_hough_mask_result(
        score_matrix=ref_to_pred_matrix,
        statistics=ref_to_pred_floor_statistics,
        config=candidate_config,
        alpha=float(alpha),
    )
    ref_to_ref_floor = build_pre_hough_mask_result(
        score_matrix=ref_to_ref_matrix,
        statistics=ref_to_ref_floor_statistics,
        config=candidate_config,
        alpha=float(alpha),
    )
    timing_preprocessing_seconds = time.perf_counter() - preprocessing_started_at

    hough = candidate_config.hough_parameters
    # Derive the number of characters shared between adjacent sliding windows.
    # With window_size=50 and window_stride=35 this is 15.  Both the pre-IoU
    # Hough-segment NLS filter and the post-IoU line NLS filter need this value
    # so that consecutive windows contribute only their unique characters to the
    # concatenated text strings used for Levenshtein comparison.
    window_overlap = max(0, int(candidate_config.window_size) - int(candidate_config.window_stride))
    if ref_to_pred_floor.active_cell_count > 0:
        ref_to_pred_hough = run_probabilistic_hough_and_filter(
            score_matrix=ref_to_pred_matrix,
            hough_input_mask=ref_to_pred_floor.hough_input_mask,
            score_floor=ref_to_pred_floor.score_floor,
            hough_threshold=hough.hough_threshold,
            hough_line_length=hough.hough_line_length,
            hough_line_gap=hough.hough_line_gap,
            hough_seed=hough.hough_seed,
            align_min_iou_threshold=candidate_config.align_min_iou_threshold,
            reference_windows=reference_windows,
            prediction_windows=prediction_windows,
            reference_window_count=ref_to_pred_shape[0],
            minimum_raw_line_nls=candidate_config.min_surviving_line_nls,
            hough_num_runs=hough.hough_num_runs,
            window_overlap=window_overlap,
        )
    else:
        ref_to_pred_hough = empty_hough_payload(score_floor_result=ref_to_pred_floor)

    line_filter_result = filter_lines_by_minimum_normalised_levenshtein(
        filtered_result=ref_to_pred_hough.filtered_result,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
        reference_window_count=ref_to_pred_shape[0],
        minimum_line_nls=None,
        window_overlap=window_overlap,
    )

    if line_filter_result.filtered_result.get("lines_used") and ref_to_ref_floor.active_cell_count > 0:
        ref_to_ref_hough = run_probabilistic_hough_and_filter(
            score_matrix=ref_to_ref_matrix,
            hough_input_mask=ref_to_ref_floor.hough_input_mask,
            score_floor=ref_to_ref_floor.score_floor,
            hough_threshold=hough.hough_threshold,
            hough_line_length=hough.hough_line_length,
            hough_line_gap=hough.hough_line_gap,
            hough_seed=hough.hough_seed,
            align_min_iou_threshold=candidate_config.align_min_iou_threshold,
        )
    else:
        ref_to_ref_hough = None

    scored = score_document_alignment(
        fname=document.fname,
        reference_text=document.reference_text,
        prediction_text=document.prediction_text,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
        ref_to_pred_hough_payload=ref_to_pred_hough,
        ref_to_ref_hough_payload=ref_to_ref_hough,
        line_text_filter_result=line_filter_result,
        window_size=candidate_config.window_size,
        window_stride=candidate_config.window_stride,
        document_normalised_levenshtein=document_normalised_levenshtein,
    )
    timing_total_seconds = time.perf_counter() - candidate_started_at
    selection_score = harmonic_selection_score(scored.metrics, mode=config.alpha_selection_harmonic_mode)
    result_row = build_result_row(
        document=document,
        config=candidate_config,
        ref_to_pred_matrix_source=ref_to_pred_matrix_source,
        ref_to_ref_matrix_source=ref_to_ref_matrix_source,
        ref_to_pred_matrix_reason=ref_to_pred_matrix_reason,
        ref_to_ref_matrix_reason=ref_to_ref_matrix_reason,
        ref_to_pred_shape=ref_to_pred_shape,
        ref_to_ref_shape=ref_to_ref_shape,
        ref_to_pred_floor=ref_to_pred_floor,
        ref_to_ref_floor=ref_to_ref_floor,
        ref_to_pred_hough=ref_to_pred_hough,
        ref_to_ref_hough=ref_to_ref_hough,
        scored=scored,
        timing_matrix_seconds=timing_matrix_seconds,
        timing_preprocessing_seconds=timing_preprocessing_seconds,
        timing_total_seconds=timing_total_seconds,
    )
    result_row["selection_harmonic_score"] = float(selection_score)
    result_row["alpha_sweep_pickle_path"] = ""
    return AlphaCandidateRun(
        alpha=float(alpha),
        selection_harmonic_score=float(selection_score),
        result_row=result_row,
        ref_to_pred_floor=ref_to_pred_floor,
        ref_to_ref_floor=ref_to_ref_floor,
        ref_to_pred_hough=ref_to_pred_hough,
        ref_to_ref_hough=ref_to_ref_hough,
        scored=scored,
        line_filter_result=line_filter_result,
        timing_preprocessing_seconds=float(timing_preprocessing_seconds),
        timing_total_seconds=float(timing_total_seconds),
    )


def build_alpha_sweep_pickle_payload(
    *,
    document: RunfileDocument,
    config: PipelineConfig,
    base_document_row: dict[str, Any],
    alpha_values: tuple[float, ...],
    candidate_summaries: list[dict[str, Any]],
    selected_result_row: dict[str, Any],
    selected_candidate_summary: dict[str, Any] | None,
    selected_plot_payload: dict[str, Any] | None,
    timing_matrix_seconds: float,
    timing_total_seconds: float,
) -> dict[str, Any]:
    """Build the per-document pickle payload containing every candidate and the selected result."""

    hough = config.hough_parameters
    return {
        "schema_version": "tuner_simple_alpha_sweep_pre_iou_levenshtein_document_v1",
        "selection_formula": "3 / ((1 / weighted_along_lines_normalised_levenshtein) + (1 / correct_ref_coverage) + (1 / (1 - hallucination)))",
        "document": dict(base_document_row),
        "fname": str(document.fname),
        "main_language": str(document.main_language),
        "document_type": str(document.document_type),
        "alpha_values": [float(value) for value in alpha_values],
        "selected_alpha": float(selected_result_row.get("score_floor_alpha")),
        "selected_harmonic_score": float(selected_result_row.get("selection_harmonic_score") or 0.0),
        "selected_result_row": dict(selected_result_row),
        "selected_candidate_summary": selected_candidate_summary,
        "candidate_summaries": candidate_summaries,
        "selected_plot_payload": selected_plot_payload,
        "timings": {
            "matrix_seconds": float(timing_matrix_seconds),
            "document_total_seconds": float(timing_total_seconds),
        },
        "run_parameters": {
            "alpha_sweep_enabled": bool(config.alpha_sweep_enabled),
            "alpha_sweep_min": float(config.alpha_sweep_min),
            "alpha_sweep_max": float(config.alpha_sweep_max),
            "alpha_sweep_step": float(config.alpha_sweep_step),
            "window_size": int(config.window_size),
            "window_stride": int(config.window_stride),
            "minimum_matrix_rows": int(config.minimum_matrix_rows),
            "minimum_matrix_columns": int(config.minimum_matrix_columns),
            "hough_threshold": int(hough.hough_threshold),
            "hough_line_length": int(hough.hough_line_length),
            "hough_line_gap": int(hough.hough_line_gap),
            "hough_seed": None if hough.hough_seed is None else int(hough.hough_seed),
            "align_min_iou_threshold": float(config.align_min_iou_threshold),
            "min_surviving_line_nls": config.min_surviving_line_nls,
        },
    }


# Define the process_one_document function; its body below performs one named step of the pipeline.
def process_one_document(
    *,
    document: RunfileDocument,
    config: PipelineConfig,
    indexes: ScoreMatrixIndexBundle,
    log,
    keep_plot_payload: bool,
) -> DocumentRunResult:
    """Run one document and select the alpha with the best harmonic score."""

    document_started_at = time.perf_counter()
    base_document_row = document_table_row(document, window_size=config.window_size, window_stride=config.window_stride)
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

    try:
        matrix_started_at = time.perf_counter()
        log(f"[matrix] {document.fname} load start")
        ref_to_pred_loaded = load_or_compute_ref_to_pred_matrix(
            scores_pkl=config.scores_pkl_ref_to_pred,
            score_index_by_fname=indexes.ref_to_pred_index,
            fname=document.fname,
            reference_text=document.reference_text,
            prediction_text=document.prediction_text,
            window_size=config.window_size,
            window_stride=config.window_stride,
            log=log,
        )
        ref_to_ref_loaded = load_or_compute_ref_to_ref_matrix(
            scores_pkl=config.scores_pkl_ref_to_ref,
            score_index_by_fname=indexes.ref_to_ref_index,
            fname=document.fname,
            reference_text=document.reference_text,
            window_size=config.window_size,
            window_stride=config.window_stride,
            log=log,
        )
        timing_matrix_seconds = time.perf_counter() - matrix_started_at

        ref_to_pred_matrix = np.asarray(ref_to_pred_loaded.matrix, dtype=float)
        ref_to_ref_matrix = np.asarray(ref_to_ref_loaded.matrix, dtype=float)
        ref_to_pred_shape = tuple(int(value) for value in ref_to_pred_matrix.shape)
        ref_to_ref_shape = tuple(int(value) for value in ref_to_ref_matrix.shape)
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

        size_skip_reason = matrix_size_skip_reason(
            ref_to_pred_shape,
            minimum_rows=config.minimum_matrix_rows,
            minimum_columns=config.minimum_matrix_columns,
        )
        if size_skip_reason is not None:
            log(
                f"[matrix] {document.fname} size check failed "
                f"reason={size_skip_reason} "
                f"minimum_rows={int(config.minimum_matrix_rows)} "
                f"minimum_columns={int(config.minimum_matrix_columns)} "
                f"seconds={format_log_value(time.perf_counter() - document_started_at)}"
            )
            return DocumentRunResult(
                result_row=None,
                skipped_row=skipped_row_from_document(
                    document,
                    config=config,
                    skip_stage="matrix_size",
                    skip_reason=size_skip_reason,
                    matrix_shape=ref_to_pred_shape,
                ),
                loadable_row=None,
                loaded_row=None,
                plot_payload=None,
                alpha_sweep_pickle_path=None,
            )

        log(
            f"[matrix] {document.fname} size check passed "
            f"minimum_rows={int(config.minimum_matrix_rows)} "
            f"minimum_columns={int(config.minimum_matrix_columns)}"
        )
        reference_windows = sliding_text_windows(
            document.reference_text,
            window_size=config.window_size,
            window_stride=config.window_stride,
        )
        prediction_windows = sliding_text_windows(
            document.prediction_text,
            window_size=config.window_size,
            window_stride=config.window_stride,
        )
        log(
            f"[windows] {document.fname} built "
            f"reference_windows={len(reference_windows)} "
            f"prediction_windows={len(prediction_windows)}"
        )

        # Opt A: skip expensive statistics computation when running in fixed-mask mode.
        # The score_floor in that mode is a fixed Levenshtein threshold, not mean+alpha*std, so
        # mean and std are never used for mask construction. They would only appear as audit
        # columns (score_mean_ref_to_pred, etc.) in the result CSV, where NaN is acceptable.
        # In alpha-sweep mode the statistics ARE required, so they are computed normally there.
        if fixed_minimum_levenshtein_mask_enabled(config):
            _nan_stats = ScoreFloorStatistics(
                score_mean=float("nan"),
                score_standard_deviation=float("nan"),
            )
            ref_to_pred_floor_statistics = _nan_stats
            ref_to_ref_floor_statistics = _nan_stats
            log(f"[preprocess] {document.fname} score statistics skipped (fixed Levenshtein mask mode)")
        else:
            floor_statistics_started_at = time.perf_counter()
            ref_to_pred_floor_statistics = compute_score_floor_statistics(ref_to_pred_matrix)
            ref_to_ref_floor_statistics = compute_score_floor_statistics(ref_to_ref_matrix)
            log(
                f"[preprocess] {document.fname} reusable score statistics ready "
                f"ref_to_pred_mean={format_log_value(ref_to_pred_floor_statistics.score_mean)} "
                f"ref_to_pred_std={format_log_value(ref_to_pred_floor_statistics.score_standard_deviation)} "
                f"ref_to_ref_mean={format_log_value(ref_to_ref_floor_statistics.score_mean)} "
                f"ref_to_ref_std={format_log_value(ref_to_ref_floor_statistics.score_standard_deviation)} "
                f"seconds={format_log_value(time.perf_counter() - floor_statistics_started_at)}"
            )

        document_nls_precomputed = float(normalized_levenshtein_similarity(
            str(document.prediction_text),
            str(document.reference_text),
        ))
        log(
            f"[preprocess] {document.fname} document_nls precomputed "
            f"document_nls={format_log_value(document_nls_precomputed)}"
        )

        # Opt C: separate the fixed-mask single-candidate path from the full alpha sweep path.
        # When --minimum-pre-hough-levenshtein is set, there is exactly one candidate (alpha=0.0),
        # no candidate comparison is needed, and no alpha-sweep pickle is written. Keeping the
        # two paths visually separate makes the intent of each branch immediately clear.
        if fixed_minimum_levenshtein_mask_enabled(config):
            log(
                f"[pre-hough] {document.fname} fixed minimum Levenshtein mask enabled "
                f"minimum={float(config.minimum_pre_hough_levenshtein):.6g} "
                f"alpha_sweep_skipped=True"
            )
            candidate = run_alpha_candidate(
                document=document,
                config=config,
                alpha=0.0,
                ref_to_pred_matrix=ref_to_pred_matrix,
                ref_to_ref_matrix=ref_to_ref_matrix,
                ref_to_pred_shape=ref_to_pred_shape,
                ref_to_ref_shape=ref_to_ref_shape,
                ref_to_pred_floor_statistics=ref_to_pred_floor_statistics,
                ref_to_ref_floor_statistics=ref_to_ref_floor_statistics,
                ref_to_pred_matrix_source=ref_to_pred_loaded.source,
                ref_to_ref_matrix_source=ref_to_ref_loaded.source,
                ref_to_pred_matrix_reason=ref_to_pred_loaded.reason,
                ref_to_ref_matrix_reason=ref_to_ref_loaded.reason,
                reference_windows=reference_windows,
                prediction_windows=prediction_windows,
                timing_matrix_seconds=timing_matrix_seconds,
                document_normalised_levenshtein=document_nls_precomputed,
            )
            selected_result_row: dict[str, Any] = dict(candidate.result_row)
            # Opt B: skip building the candidate summary dict in fixed-mask mode.
            # The summary is only consumed when writing the alpha-sweep pickle, which is never
            # written in this path. Building the dict would be pure waste.
            selected_candidate_summary: dict[str, Any] | None = None
            candidate_summaries: list[dict[str, Any]] = []
            selected_plot_payload: dict[str, Any] | None = (
                build_plot_payload(
                    document=document,
                    result_row=selected_result_row,
                    ref_to_pred_matrix=ref_to_pred_matrix,
                    ref_to_ref_matrix=ref_to_ref_matrix,
                    candidate=candidate,
                )
                if keep_plot_payload
                else None
            )
            log(
                f"[pre-hough] {document.fname} fixed mask candidate done "
                f"score={format_log_value(candidate.selection_harmonic_score)} "
                f"used_lines={int(selected_result_row.get('used_line_count') or 0)} "
                f"weighted_nls={format_log_value(candidate.scored.metrics.weighted_along_lines_normalised_levenshtein)} "
                f"correct_ref_coverage={format_log_value(candidate.scored.metrics.correct_ref_coverage)} "
                f"hallucination={format_log_value(candidate.scored.metrics.hallucination)} "
                f"seconds={format_log_value(candidate.timing_total_seconds)}"
            )
        else:
            # Alpha sweep path: evaluate every candidate alpha and keep the highest-scoring one.
            # When --no-alpha-sweep is set, there is a single candidate and the selection is trivial.
            alpha_values = alpha_values_for_config(config)
            log(
                f"[alpha-sweep] {document.fname} start "
                f"enabled={bool(config.alpha_sweep_enabled)} "
                f"candidate_count={len(alpha_values)} "
                f"values={','.join(f'{value:.6g}' for value in alpha_values)}"
            )

            best_key: tuple[float, float, float, float, int, float] | None = None
            selected_result_row = None
            selected_plot_payload = None
            selected_candidate_summary = None
            candidate_summaries = []

            for alpha_index, alpha in enumerate(alpha_values):
                candidate = run_alpha_candidate(
                    document=document,
                    config=config,
                    alpha=float(alpha),
                    ref_to_pred_matrix=ref_to_pred_matrix,
                    ref_to_ref_matrix=ref_to_ref_matrix,
                    ref_to_pred_shape=ref_to_pred_shape,
                    ref_to_ref_shape=ref_to_ref_shape,
                    ref_to_pred_floor_statistics=ref_to_pred_floor_statistics,
                    ref_to_ref_floor_statistics=ref_to_ref_floor_statistics,
                    ref_to_pred_matrix_source=ref_to_pred_loaded.source,
                    ref_to_ref_matrix_source=ref_to_ref_loaded.source,
                    ref_to_pred_matrix_reason=ref_to_pred_loaded.reason,
                    ref_to_ref_matrix_reason=ref_to_ref_loaded.reason,
                    reference_windows=reference_windows,
                    prediction_windows=prediction_windows,
                    timing_matrix_seconds=timing_matrix_seconds,
                    document_normalised_levenshtein=document_nls_precomputed,
                )
                candidate_summary = build_alpha_candidate_summary(candidate)
                candidate_summaries.append(candidate_summary)
                current_key = candidate_selection_key(candidate)
                log(
                    f"[alpha-sweep] {document.fname} alpha={float(alpha):.6f} "
                    f"score={format_log_value(candidate.selection_harmonic_score)} "
                    f"used_lines={int(candidate.result_row.get('used_line_count') or 0)} "
                    f"weighted_nls={format_log_value(candidate.scored.metrics.weighted_along_lines_normalised_levenshtein)} "
                    f"correct_ref_coverage={format_log_value(candidate.scored.metrics.correct_ref_coverage)} "
                    f"hallucination={format_log_value(candidate.scored.metrics.hallucination)} "
                    f"seconds={format_log_value(candidate.timing_total_seconds)}"
                )
                if best_key is None or current_key > best_key:
                    best_key = current_key
                    selected_result_row = dict(candidate.result_row)
                    selected_candidate_summary = candidate_summary
                    if keep_plot_payload:
                        selected_plot_payload = build_plot_payload(
                            document=document,
                            result_row=selected_result_row,
                            ref_to_pred_matrix=ref_to_pred_matrix,
                            ref_to_ref_matrix=ref_to_ref_matrix,
                            candidate=candidate,
                        )
                if candidate.ref_to_pred_floor.active_cell_count == 0:
                    remaining = len(alpha_values) - alpha_index - 1
                    log(
                        f"[alpha-sweep] {document.fname} early exit alpha={float(alpha):.6f} "
                        f"reason=empty_ref_to_pred_mask remaining_skipped={remaining}"
                    )
                    break

            if selected_result_row is None:
                raise RuntimeError("alpha sweep produced no candidates")

        timing_total_seconds = time.perf_counter() - document_started_at
        selected_result_row["timing_total_seconds"] = float(timing_total_seconds)
        alpha_sweep_pickle_path: str | None = None
        if bool(config.alpha_sweep_enabled) and not fixed_minimum_levenshtein_mask_enabled(config):
            pickle_path = alpha_sweep_pickle_path_for_document(config, document)
            alpha_sweep_pickle_path = str(pickle_path)
            selected_result_row["alpha_sweep_pickle_path"] = alpha_sweep_pickle_path
            for candidate_summary in candidate_summaries:
                candidate_summary["result_row"]["alpha_sweep_pickle_path"] = alpha_sweep_pickle_path
            if selected_candidate_summary is not None:
                selected_candidate_summary["result_row"]["alpha_sweep_pickle_path"] = alpha_sweep_pickle_path
        else:
            selected_result_row["alpha_sweep_pickle_path"] = ""

        if selected_plot_payload is not None:
            selected_plot_payload["result_row"] = selected_result_row
            log(
                f"[plot] {document.fname} selected payload ready "
                f"raw_lines={len(selected_plot_payload['raw_ref_to_pred_hough_lines'])} "
                f"final_lines={len(selected_plot_payload['final_surviving_ref_to_pred_lines'])}"
            )
        else:
            log(f"[plot] {document.fname} payload skipped reason=plotting_disabled")

        if bool(config.alpha_sweep_enabled) and not fixed_minimum_levenshtein_mask_enabled(config) and not bool(config.suppress_output_files):
            pickle_payload = build_alpha_sweep_pickle_payload(
                document=document,
                config=config,
                base_document_row=base_document_row,
                alpha_values=alpha_values,
                candidate_summaries=candidate_summaries,
                selected_result_row=selected_result_row,
                selected_candidate_summary=selected_candidate_summary,
                selected_plot_payload=selected_plot_payload,
                timing_matrix_seconds=timing_matrix_seconds,
                timing_total_seconds=timing_total_seconds,
            )
            write_pickle_atomically(Path(alpha_sweep_pickle_path), pickle_payload)
            log(f"[alpha-sweep] {document.fname} wrote pickle {alpha_sweep_pickle_path}")
        elif bool(config.suppress_output_files):
            log(f"[alpha-sweep] {document.fname} pickle suppressed; suppress_output_files=True")

        if fixed_minimum_levenshtein_mask_enabled(config):
            log(
                f"[pre-hough] {document.fname} selected fixed minimum Levenshtein mask "
                f"minimum={float(config.minimum_pre_hough_levenshtein):.6g} "
                f"threshold_ref_to_pred={format_log_value(selected_result_row.get('score_floor_ref_to_pred'))} "
                f"score={format_log_value(selected_result_row.get('selection_harmonic_score'))} "
                f"used_lines={int(selected_result_row.get('used_line_count') or 0)} "
                f"seconds={format_log_value(timing_total_seconds)}"
            )
        else:
            log(
                f"[alpha-sweep] {document.fname} selected "
                f"alpha={float(selected_result_row.get('score_floor_alpha')):.6f} "
                f"score={format_log_value(selected_result_row.get('selection_harmonic_score'))} "
                f"used_lines={int(selected_result_row.get('used_line_count') or 0)} "
                f"seconds={format_log_value(timing_total_seconds)}"
            )
        return DocumentRunResult(
            result_row=selected_result_row,
            skipped_row=None,
            loadable_row=base_document_row,
            loaded_row=base_document_row,
            plot_payload=selected_plot_payload,
            alpha_sweep_pickle_path=alpha_sweep_pickle_path,
        )
    except Exception as exc:
        log(f"[exception] {document.fname} error={repr(exc)} seconds={format_log_value(time.perf_counter() - document_started_at)}")
        return DocumentRunResult(
            result_row=None,
            skipped_row=skipped_row_from_document(
                document,
                config=config,
                skip_stage="exception",
                skip_reason=repr(exc),
            ),
            loadable_row=None,
            loaded_row=None,
            plot_payload=None,
            alpha_sweep_pickle_path=None,
        )


__all__ = ["DocumentRunResult", "document_table_row", "process_one_document", "skipped_row_from_document"]
