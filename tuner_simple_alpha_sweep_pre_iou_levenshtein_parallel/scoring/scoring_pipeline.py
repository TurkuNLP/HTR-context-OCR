from __future__ import annotations

"""Compute the public scientific metrics for one processed document."""

from dataclasses import dataclass
import time

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.probabilistic_hough.hough_detection import HoughFilteredPayload
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.scoring.coverage_count_metrics import CoverageCountMetricResult, compute_coverage_count_metrics
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.scoring.levenshtein import normalized_levenshtein_similarity
from .line_text_similarity import LineTextFilterResult, compute_weighted_along_lines_from_payload


@dataclass(frozen=True)
class DocumentAlignmentMetrics:
    """The scientific metrics written by the simple tuner."""

    # Store whole-document normalized Levenshtein similarity between the prediction text and the reference text.
    document_normalised_levenshtein: float
    # Store normalized Levenshtein similarity measured only along final accepted lines, weighted by covered prediction text.
    weighted_along_lines_normalised_levenshtein: float | None
    # Store the fraction of reference characters covered exactly once by final ref-to-pred lines.
    correct_ref_coverage: float | None
    # Store the fraction of reference characters covered by ref-to-ref evidence but not covered by final ref-to-pred lines.
    missing_ref_coverage: float | None
    # Store the fraction of reference characters covered more often by ref-to-pred lines than by ref-to-ref evidence.
    repetition_on_reference: float | None
    # Store the fraction of prediction characters that no final ref-to-pred line assigns to the reference.
    hallucination: float | None


@dataclass(frozen=True)
class DirectionScoringPayload:
    """Compact Hough result data shared by text scoring and reference-axis coverage scoring."""

    # Store the full Hough payload so plotting and audit output can still inspect the final lines.
    hough_payload: HoughFilteredPayload
    # Store the minimal dictionary consumed by text-similarity and coverage-count metric code.
    scoring_payload: dict


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


def build_direction_scoring_payload(
    *,
    hough_payload: HoughFilteredPayload,
    reference_text_length: int,
    other_text_length: int,
    window_size: int,
    window_stride: int,
) -> DirectionScoringPayload:
    """Build compact scoring data for one score-matrix direction."""
    # Read the filtered Hough result; this contains final accepted lines and their column ownership arrays.
    filtered_result = hough_payload.filtered_result
    # Read only the shape of the Hough mask because the metric code needs window counts, not mask values.
    mask_shape = tuple(int(value) for value in getattr(hough_payload.hough_context.get("mask"), "shape", ()) or ())
    # The first matrix axis is the reference-window axis used by both ref-to-pred and ref-to-ref matrices.
    reference_window_count = int(mask_shape[0]) if len(mask_shape) == 2 else 0
    # The second matrix axis is either prediction windows for ref-to-pred or reference windows for ref-to-ref.
    other_window_count = int(mask_shape[1]) if len(mask_shape) == 2 else 0
    # Keep only the fields that downstream scoring code needs so the payload stays small and easy to audit.
    scoring_payload = {
        # Store final line dictionaries after Hough filtering and line-text filtering.
        "lines_used": list(filtered_result.get("lines_used", [])),
        # Store per-column ownership arrays; ref-to-pred uses prediction columns and ref-to-ref uses reference columns.
        "column_assignment": filtered_result.get("column_assignment", {}),
        # Store the number of reference sliding windows represented by matrix rows.
        "reference_window_count": int(reference_window_count),
        # Store the number of comparison sliding windows represented by matrix columns.
        "other_window_count": int(other_window_count),
        # Store the original reference text length so coverage ratios are measured in characters.
        "reference_text_length": int(reference_text_length),
        # Store the original comparison text length so hallucination can be measured over prediction characters.
        "other_text_length": int(other_text_length),
        # Store the sliding-window width so window ids can be converted back into character spans.
        "window_size": int(window_size),
        # Store the sliding-window stride so neighboring window ids map back to the correct character offsets.
        "window_stride": int(window_stride),
    }
    # Return a named payload so callers pass one object instead of a loose dictionary.
    return DirectionScoringPayload(hough_payload=hough_payload, scoring_payload=scoring_payload)


def zero_alignment_metrics(*, document_normalised_levenshtein: float) -> DocumentAlignmentMetrics:
    """Return stable metric values when no final lines remain for alignment."""
    # No final lines means no line-guided text score, no reference coverage, full missing reference, and full hallucination.
    return DocumentAlignmentMetrics(
        document_normalised_levenshtein=float(document_normalised_levenshtein),
        weighted_along_lines_normalised_levenshtein=None,
        correct_ref_coverage=0.0,
        missing_ref_coverage=1.0,
        repetition_on_reference=0.0,
        hallucination=1.0,
    )


def empty_coverage_diagnostics() -> dict:
    """Return stable coverage diagnostic fields for a zero-line result."""
    # These fields match normal coverage diagnostics so CSV readers do not need special zero-line handling.
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
    """Compute coverage metrics from v2.12-style reference-axis count subtraction."""
    # Delegate to the local coverage-count module so the higher-level scoring code stays focused on orchestration.
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
    document_normalised_levenshtein: float | None = None,
) -> ScoredDocumentResult:
    """Compute the six public metrics for one document."""
    # Start timing document-level and line-level Levenshtein work.
    levenshtein_started_at = time.perf_counter()
    # Use pre-computed whole-document NLS when the caller already computed it once outside the alpha loop.
    if document_normalised_levenshtein is not None:
        document_nls = float(document_normalised_levenshtein)
    else:
        document_nls = float(normalized_levenshtein_similarity(str(prediction_text), str(reference_text)))
    # Rebuild the ref-to-pred Hough payload with the line-text-pruned final lines and column assignments.
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
    )
    # If the text filter removed every line, return explicit zero-line metrics while keeping the plot payload available.
    if not line_text_filter_result.filtered_result.get("lines_used"):
        # Build the no-line metrics row.
        metrics = zero_alignment_metrics(document_normalised_levenshtein=document_nls)
        # Return the scored document result without running reference-self coverage subtraction.
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
    # Reuse the weighted line-text score computed during line filtering when that cached result is available.
    if line_text_filter_result.weighted_result is not None:
        # Store the cached weighted result so the same line text is not compared twice.
        weighted_result = line_text_filter_result.weighted_result
    else:
        # Compute weighted along-line similarity from final lines and their owned prediction columns.
        weighted_result = compute_weighted_along_lines_from_payload(
            reference_windows=reference_windows,
            prediction_windows=prediction_windows,
            lines_used=list(line_text_filter_result.filtered_result.get("lines_used", [])),
            compact_payload=ref_to_pred_payload.scoring_payload,
        )
    # Finish Levenshtein timing after document-level and line-level similarity are both known.
    levenshtein_seconds = float(time.perf_counter() - levenshtein_started_at)
    # Start with no ref-to-ref payload; documents without final ref-to-pred lines do not need self-coverage evidence.
    ref_to_ref_payload = None
    # Build ref-to-ref self-coverage payload only when the Hough stage actually produced one.
    if ref_to_ref_hough_payload is not None:
        # Convert the ref-to-ref Hough result into the same compact payload format used by ref-to-pred scoring.
        ref_to_ref_payload = build_direction_scoring_payload(
            hough_payload=ref_to_ref_hough_payload,
            reference_text_length=len(reference_text),
            other_text_length=len(reference_text),
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
    # Start timing the coverage-count calculation separately from text similarity.
    coverage_started_at = time.perf_counter()
    # Compute correct coverage, missing coverage, repetition, hallucination, and invalid `-2` diagnostics.
    coverage_result = compute_local_coverage_metrics(
        ref_to_pred_payload=ref_to_pred_payload,
        ref_to_ref_payload=ref_to_ref_payload,
    )
    # Store coverage runtime for audit output.
    coverage_seconds = float(time.perf_counter() - coverage_started_at)
    # Build the final six-metric object from text similarity and reference-axis count subtraction.
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
    # Return metrics plus compact payloads needed by output rows and plotting code.
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
