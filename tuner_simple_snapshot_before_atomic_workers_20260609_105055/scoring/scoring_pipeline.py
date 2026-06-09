from __future__ import annotations

"""Compute the six scientific metrics for one processed document."""

from dataclasses import dataclass
import time

import numpy as np

from tuner_simple.probabilistic_hough.hough_detection import HoughFilteredPayload
from tuner_simple.scoring.coverage_count_metrics import CoverageCountMetricResult, compute_coverage_count_metrics
from tuner_simple.scoring.levenshtein import normalized_levenshtein_similarity
from .line_text_similarity import LineTextFilterResult, compute_weighted_along_lines_from_payload


@dataclass(frozen=True)
class DocumentAlignmentMetrics:
    """The only scientific metrics exposed by the simple tuner."""

    # Store whole-document normalized Levenshtein similarity between prediction and reference text.
    document_normalised_levenshtein: float
    # Store line-guided normalized Levenshtein similarity, weighted by how many prediction columns each line covers.
    weighted_along_lines_normalised_levenshtein: float | None
    # Store the fraction of reference windows covered by final ref-to-pred lines.
    correct_ref_coverage: float | None
    # Store the fraction of reference windows not covered by final ref-to-pred lines.
    missing_ref_coverage: float | None
    # Store how much final line coverage repeats already-covered reference windows.
    repetition_on_reference: float | None
    # Store the fraction of prediction windows not assigned to any final line.
    hallucination: float | None


@dataclass(frozen=True)
class DirectionScoringPayload:
    """Compact local ownership payload built after Hough and filtering."""

    # Store raw and filtered Hough data for one matrix direction.
    hough_payload: HoughFilteredPayload
    # Store the compact local scoring dictionary used by text and coverage metrics.
    scoring_payload: dict
    # Store covered reference rows for reference-to-reference self alignment, when available.
    refref_y: np.ndarray | None


@dataclass(frozen=True)
class ScoredDocumentResult:
    """Metrics plus lightweight diagnostic counts for output tables and plots."""

    # Store the six final scientific metrics.
    metrics: DocumentAlignmentMetrics
    # Store the reference-to-prediction scoring payload.
    ref_to_pred_payload: DirectionScoringPayload
    # Store the reference-to-reference scoring payload, or None when it was not needed.
    ref_to_ref_payload: DirectionScoringPayload | None
    # Store the line-level text filtering summary.
    line_text_filter_result: LineTextFilterResult
    # Store whether v2.12-compatible coverage subtraction rejected this result.
    coverage_invalid_reason: str | None
    # Store the explicit coverage subtraction error message when the result is invalid.
    coverage_invalid_error_message: str | None
    # Store compact y-axis subtraction diagnostics for audits and debugging.
    coverage_diagnostics: dict
    # Store how long coverage metrics took to compute.
    coverage_seconds: float
    # Store how long document and line Levenshtein metrics took to compute.
    levenshtein_seconds: float


def finite_unit_interval(value: float) -> float:
    """Clamp a numeric metric into [0, 1], using 0 for non-finite values."""
    # Convert the value to float so callers can pass NumPy scalar values safely.
    value = float(value)
    # Treat NaN and infinity as missing evidence rather than letting them leak into CSV output.
    if not np.isfinite(value):
        return 0.0
    # Clamp below zero to zero and above one to one.
    return max(0.0, min(1.0, value))


def assignment_arrays(column_assignment: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return mapped-y and mapped-line-id arrays from a local column assignment dictionary."""
    # Convert mapped reference rows into float values because unassigned columns are represented with NaN.
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    # Convert mapped line ids into integer values where -1 means unassigned.
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    # Return both arrays in the same order as prediction/self columns.
    return mapped_y, mapped_line_id


def rounded_valid_rows(mapped_y: np.ndarray, mapped_line_id: np.ndarray, row_count: int) -> list[int]:
    """Return valid rounded reference-row indices from assigned columns."""
    # Store rows in column order so repetition can be measured later.
    rows: list[int] = []
    # Inspect each mapped y coordinate with its owning line id.
    for y_value, line_id in zip(mapped_y, mapped_line_id):
        # Skip columns that no final line owns.
        if int(line_id) < 0:
            continue
        # Skip NaN or infinite y coordinates.
        if not np.isfinite(float(y_value)):
            continue
        # Round the continuous line coordinate to the nearest reference-window row.
        row_index = int(round(float(y_value)))
        # Keep only rows inside the score matrix.
        if 0 <= row_index < int(row_count):
            rows.append(int(row_index))
    # Return valid assigned reference rows.
    return rows


def build_direction_scoring_payload(
    *,
    hough_payload: HoughFilteredPayload,
    reference_text_length: int,
    other_text_length: int,
    window_size: int,
    window_stride: int,
    include_reference_self_coverage_array: bool,
) -> DirectionScoringPayload:
    """Build compact local ownership data for one matrix direction."""
    # Read the filtered Hough result produced by local ownership filtering.
    filtered_result = hough_payload.filtered_result
    # Read the Hough mask only for its matrix shape; values are not used for scoring here.
    mask = np.asarray(hough_payload.hough_context.get("mask"))
    # Derive the reference-window count from matrix rows.
    reference_window_count = int(mask.shape[0]) if mask.ndim == 2 else 0
    # Derive the comparison-window count from matrix columns.
    other_window_count = int(mask.shape[1]) if mask.ndim == 2 else 0
    # Store local compact fields needed by text and coverage metrics.
    scoring_payload = {
        "lines_used": list(filtered_result.get("lines_used", [])),
        "column_assignment": filtered_result.get("column_assignment", {}),
        "reference_window_count": int(reference_window_count),
        "other_window_count": int(other_window_count),
        "reference_text_length": int(reference_text_length),
        "other_text_length": int(other_text_length),
        "window_size": int(window_size),
        "window_stride": int(window_stride),
    }
    # Build reference-self covered rows only for the ref-to-ref direction.
    refref_y = None
    # Include the self-coverage array when the caller is building reference-to-reference evidence.
    if include_reference_self_coverage_array:
        # Read assignment arrays from the ref-to-ref line result.
        mapped_y, mapped_line_id = assignment_arrays(scoring_payload["column_assignment"])
        # Convert assigned self-alignment rows into a compact integer array.
        refref_y = np.asarray(
            rounded_valid_rows(mapped_y, mapped_line_id, int(reference_window_count)),
            dtype=np.int32,
        )
    # Return a named payload so downstream code does not need to know how the dictionary is built.
    return DirectionScoringPayload(hough_payload=hough_payload, scoring_payload=scoring_payload, refref_y=refref_y)


def zero_alignment_metrics(*, document_normalised_levenshtein: float) -> DocumentAlignmentMetrics:
    """Return a stable metric row when no final lines remain for alignment."""
    # No final lines means no line-guided text score, no coverage, full missing reference, and full hallucination.
    return DocumentAlignmentMetrics(
        document_normalised_levenshtein=float(document_normalised_levenshtein),
        weighted_along_lines_normalised_levenshtein=None,
        correct_ref_coverage=0.0,
        missing_ref_coverage=1.0,
        repetition_on_reference=0.0,
        hallucination=1.0,
    )


def empty_coverage_diagnostics() -> dict:
    """Return the stable v2.2 diagnostic fields for a zero-line result."""
    return {
        "coverage_y_diff_size": 0,
        "coverage_y_diff_min": None,
        "coverage_y_diff_max": None,
        "coverage_y_diff_le_minus_one_count": 0,
        "coverage_y_diff_lt_minus_one_count": 0,
        "coverage_y_diff_below_minus_one_counts_json": {},
    }


def compute_local_coverage_metrics(
    *,
    ref_to_pred_payload: DirectionScoringPayload,
    ref_to_ref_payload: DirectionScoringPayload | None,
) -> CoverageCountMetricResult:
    """Compute coverage metrics from the v2.2 reference-axis count subtraction."""
    return compute_coverage_count_metrics(
        ref_to_pred_scoring_payload=ref_to_pred_payload.scoring_payload,
        ref_to_ref_scoring_payload=None if ref_to_ref_payload is None else ref_to_ref_payload.scoring_payload,
    )


def score_document_alignment(
    *,
    fname: str,
    reference_text: str,
    prediction_text: str,
    reference_windows: list[str],
    prediction_windows: list[str],
    ref_to_pred_hough_payload: HoughFilteredPayload,
    ref_to_ref_hough_payload: HoughFilteredPayload | None,
    line_text_filter_result: LineTextFilterResult,
    window_size: int,
    window_stride: int,
) -> ScoredDocumentResult:
    """Compute only the six scientific metrics for one document."""
    # Start timing document-level and line-level Levenshtein work.
    levenshtein_started_at = time.perf_counter()
    # Compute whole-document normalized Levenshtein similarity.
    document_nls = float(
        normalized_levenshtein_similarity(
            str(prediction_text),
            str(reference_text),
        )
    )
    # Build a ref-to-pred payload that uses the line-text-pruned final lines and assignments.
    ref_to_pred_payload = build_direction_scoring_payload(
        hough_payload=HoughFilteredPayload(
            hough_context=ref_to_pred_hough_payload.hough_context,
            detection_result=ref_to_pred_hough_payload.detection_result,
            filtered_result=line_text_filter_result.filtered_result,
            raw_line_count=ref_to_pred_hough_payload.raw_line_count,
            candidate_line_count=ref_to_pred_hough_payload.candidate_line_count,
            used_line_count=int(len(line_text_filter_result.filtered_result.get("lines_used", []))),
            detect_seconds=ref_to_pred_hough_payload.detect_seconds,
            filter_seconds=ref_to_pred_hough_payload.filter_seconds,
        ),
        reference_text_length=len(reference_text),
        other_text_length=len(prediction_text),
        window_size=int(window_size),
        window_stride=int(window_stride),
        include_reference_self_coverage_array=False,
    )
    # If no final lines survive, return stable zero-alignment metrics immediately.
    if not line_text_filter_result.filtered_result.get("lines_used"):
        # Build the no-line metrics row.
        metrics = zero_alignment_metrics(document_normalised_levenshtein=document_nls)
        # Return the scored document result without a ref-to-ref payload.
        return ScoredDocumentResult(
            metrics=metrics,
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=None,
            line_text_filter_result=line_text_filter_result,
            coverage_invalid_reason=None,
            coverage_invalid_error_message=None,
            coverage_diagnostics=empty_coverage_diagnostics(),
            coverage_seconds=0.0,
            levenshtein_seconds=float(time.perf_counter() - levenshtein_started_at),
        )
    # Reuse the weighted line-text score computed during line filtering when available.
    if line_text_filter_result.weighted_result is not None:
        # Store the weighted result from the text filter.
        weighted_result = line_text_filter_result.weighted_result
    else:
        # Compute weighted along-line similarity from the local compact assignment payload.
        weighted_result = compute_weighted_along_lines_from_payload(
            reference_windows=reference_windows,
            prediction_windows=prediction_windows,
            lines_used=list(line_text_filter_result.filtered_result.get("lines_used", [])),
            compact_payload=ref_to_pred_payload.scoring_payload,
        )
    # Finish Levenshtein timing after document and line text scores are known.
    levenshtein_seconds = float(time.perf_counter() - levenshtein_started_at)
    # Build ref-to-ref self-coverage payload when reference self lines are available.
    ref_to_ref_payload = None
    # Include ref-to-ref evidence only when the Hough stage produced a payload.
    if ref_to_ref_hough_payload is not None:
        # Build a local self-coverage payload with reference rows covered by ref-to-ref lines.
        ref_to_ref_payload = build_direction_scoring_payload(
            hough_payload=ref_to_ref_hough_payload,
            reference_text_length=len(reference_text),
            other_text_length=len(reference_text),
            window_size=int(window_size),
            window_stride=int(window_stride),
            include_reference_self_coverage_array=True,
        )
    # Start timing the coverage calculations separately from text similarity.
    coverage_started_at = time.perf_counter()
    # Compute the four coverage-style metrics from reference-axis character-count subtraction.
    coverage_result = compute_local_coverage_metrics(
        ref_to_pred_payload=ref_to_pred_payload,
        ref_to_ref_payload=ref_to_ref_payload,
    )
    # Store coverage runtime for audit output.
    coverage_seconds = float(time.perf_counter() - coverage_started_at)
    # Build the final six-metric object.
    metrics = DocumentAlignmentMetrics(
        document_normalised_levenshtein=float(document_nls),
        weighted_along_lines_normalised_levenshtein=(
            None if weighted_result.weighted_along_lines_nls is None else float(weighted_result.weighted_along_lines_nls)
        ),
        correct_ref_coverage=coverage_result.correct_ref_coverage,
        missing_ref_coverage=coverage_result.missing_ref_coverage,
        repetition_on_reference=coverage_result.repetition_on_reference,
        hallucination=coverage_result.hallucination,
    )
    # Return metrics plus the compact payloads needed by output and plotting code.
    return ScoredDocumentResult(
        metrics=metrics,
        ref_to_pred_payload=ref_to_pred_payload,
        ref_to_ref_payload=ref_to_ref_payload,
        line_text_filter_result=line_text_filter_result,
        coverage_invalid_reason=coverage_result.invalid_reason,
        coverage_invalid_error_message=coverage_result.invalid_error_message,
        coverage_diagnostics=coverage_result.diagnostics,
        coverage_seconds=coverage_seconds,
        levenshtein_seconds=levenshtein_seconds,
    )


# Declare the public helpers that other tuner_simple modules may import.
__all__ = [
    "DirectionScoringPayload",
    "DocumentAlignmentMetrics",
    "ScoredDocumentResult",
    "build_direction_scoring_payload",
    "compute_local_coverage_metrics",
    "empty_coverage_diagnostics",
    "score_document_alignment",
    "zero_alignment_metrics",
]
