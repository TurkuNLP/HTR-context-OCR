from __future__ import annotations

"""Per-combination Hough evaluation helpers.

Each call evaluates exactly one ``(threshold, line_length, line_gap, seed)``
combination on one already-prepared ``SweepDocument``.  Matrix building and Hough
context preparation happen once in document preparation; this hot path only runs
Hough detection, filtering, v2.12 bundle/coverage logic, and line-level
Levenshtein scoring.
"""

import math
import time

import numpy as np

try:
    from ..alignment.line_alignment_pipeline_fast import detect_lines_only_from_hough_ctx, filter_lines_after_hough
    from ..metrics.alignment_quality_score import (
        compute_harmonic_tuning_score,
        compute_weighted_along_lines_similarity_from_bundle,
        normalize_v212_coverage_metrics,
    )
    from ..metrics.v2_12_metric_adapter import (
        build_v212_line_coverage_arrays_from_cached_refref_y,
        build_v212_line_coverage_arrays_from_bundles,
        build_v212_line_metric_bundle,
        build_v212_refref_y_coverage_array_from_bundle,
        compute_v212_line_coverage_ratio_metrics_from_arrays,
    )
    from .tuner_config import (
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )
except ImportError:
    from alignment.line_alignment_pipeline_fast import detect_lines_only_from_hough_ctx, filter_lines_after_hough  # type: ignore
    from metrics.alignment_quality_score import (  # type: ignore
        compute_harmonic_tuning_score,
        compute_weighted_along_lines_similarity_from_bundle,
        normalize_v212_coverage_metrics,
    )
    from metrics.v2_12_metric_adapter import (  # type: ignore
        build_v212_line_coverage_arrays_from_cached_refref_y,
        build_v212_line_coverage_arrays_from_bundles,
        build_v212_line_metric_bundle,
        build_v212_refref_y_coverage_array_from_bundle,
        compute_v212_line_coverage_ratio_metrics_from_arrays,
    )
    from tuner.tuner_config import (  # type: ignore
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )


INVALID_REASON_COVERAGE_Y_DIFF_BELOW_MINUS_ONE = "coverage_y_diff_below_minus_one"


def is_finite_tuning_score(value) -> bool:
    """Return ``True`` only for finite numeric tuner objective scores."""
    if value is None:
        return False
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def is_finite_along_lines(value) -> bool:
    """Return ``True`` only for finite numeric along-lines scores."""
    if value is None:
        return False
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def _finite_float_for_rank(value, default: float) -> float:
    """Convert a value into a finite float for ranking tuples."""
    if value is None:
        return float(default)
    try:
        converted = float(value)
    except Exception:
        return float(default)
    if not math.isfinite(converted):
        return float(default)
    return float(converted)


def evaluation_rank_key(row: dict) -> tuple[float, float, float, float, int, int, int, int, int, int]:
    """Return a strict deterministic ranking tuple for best-evaluation selection."""
    hallucination = _finite_float_for_rank(row.get("hallucination"), 1.0)
    return (
        _finite_float_for_rank(row.get("tuning_score"), float("-inf")),
        _finite_float_for_rank(row.get("weighted_along_lines_nls"), float("-inf")),
        _finite_float_for_rank(row.get("correct_ref_coverage"), float("-inf")),
        -float(hallucination),
        int(row.get("line_guided_columns", 0)),
        -int(row.get("fallback_columns", 0)),
        -int(row.get(PARAM_HOUGH_THRESHOLD, 0)),
        -int(row.get(PARAM_HOUGH_LINE_LENGTH, 0)),
        -int(row.get(PARAM_HOUGH_LINE_GAP, 0)),
        -int(row.get(PARAM_HOUGH_SEED, 0)),
    )


def pick_better_eval(current: dict | None, candidate: dict | None) -> dict | None:
    """Return the better of two evaluation rows according to the tuner ranking."""
    if candidate is None:
        return current

    candidate_is_valid = bool(candidate.get("is_valid", True)) and is_finite_tuning_score(candidate.get("tuning_score"))
    if current is None:
        return candidate if candidate_is_valid else None

    current_is_valid = bool(current.get("is_valid", True)) and is_finite_tuning_score(current.get("tuning_score"))
    if not candidate_is_valid:
        return current
    if not current_is_valid:
        return candidate
    if evaluation_rank_key(candidate) > evaluation_rank_key(current):
        return candidate
    return current


def _coverage_y_diff_diagnostics(y_diff) -> dict:
    """Return compact diagnostics for v2.12 reference-axis coverage subtraction.

    V2.12 defines ``-1`` as a valid missing-reference category.  Values below
    ``-1`` are not valid v2.12 categories because they mean reference-self
    coverage overlapped itself more than the ref-to-pred coverage can explain.
    """
    y_diff_array = np.asarray(y_diff, dtype=np.int32)
    if y_diff_array.size == 0:
        return {
            "coverage_y_diff_size": 0,
            "coverage_y_diff_min": None,
            "coverage_y_diff_max": None,
            "coverage_y_diff_le_minus_one_count": 0,
            "coverage_y_diff_lt_minus_one_count": 0,
            "coverage_y_diff_below_minus_one_counts_json": {},
        }

    unique_values, unique_counts = np.unique(y_diff_array, return_counts=True)
    below_minus_one_counts = {
        str(int(value)): int(count)
        for value, count in zip(unique_values, unique_counts)
        if int(value) < -1
    }
    return {
        "coverage_y_diff_size": int(y_diff_array.size),
        "coverage_y_diff_min": int(np.min(y_diff_array)),
        "coverage_y_diff_max": int(np.max(y_diff_array)),
        "coverage_y_diff_le_minus_one_count": int(np.count_nonzero(y_diff_array <= -1)),
        "coverage_y_diff_lt_minus_one_count": int(np.count_nonzero(y_diff_array < -1)),
        "coverage_y_diff_below_minus_one_counts_json": below_minus_one_counts,
    }


def _timing_fields(
    *,
    ref_to_pred_payload: dict,
    ref_to_ref_payload: dict,
    coverage_seconds: float,
    levenshtein_seconds: float,
    eval_started_at: float,
) -> dict:
    """Build shared timing fields for valid and invalid evaluation rows."""
    timing_hough_detect_ref_to_pred_seconds = float(ref_to_pred_payload["timing_hough_detect_seconds"])
    timing_filter_ref_to_pred_seconds = float(ref_to_pred_payload["timing_filter_seconds"])
    timing_hough_detect_ref_to_ref_seconds = float(ref_to_ref_payload["timing_hough_detect_seconds"])
    timing_filter_ref_to_ref_seconds = float(ref_to_ref_payload["timing_filter_seconds"])
    timing_build_bundle_seconds = float(
        ref_to_pred_payload["timing_build_bundle_seconds"] + ref_to_ref_payload["timing_build_bundle_seconds"]
    )
    timing_hough_detect_seconds = float(
        timing_hough_detect_ref_to_pred_seconds + timing_hough_detect_ref_to_ref_seconds
    )
    timing_filter_seconds = float(timing_filter_ref_to_pred_seconds + timing_filter_ref_to_ref_seconds)
    timing_detect_filter_seconds = float(timing_hough_detect_seconds + timing_filter_seconds)

    return {
        "timing_hough_detect_ref_to_pred_seconds": timing_hough_detect_ref_to_pred_seconds,
        "timing_filter_ref_to_pred_seconds": timing_filter_ref_to_pred_seconds,
        "timing_hough_detect_ref_to_ref_seconds": timing_hough_detect_ref_to_ref_seconds,
        "timing_filter_ref_to_ref_seconds": timing_filter_ref_to_ref_seconds,
        "timing_hough_detect_seconds": timing_hough_detect_seconds,
        "timing_filter_seconds": timing_filter_seconds,
        "timing_detect_filter_seconds": timing_detect_filter_seconds,
        "timing_build_bundle_seconds": timing_build_bundle_seconds,
        "timing_coverage_seconds": float(coverage_seconds),
        "timing_levenshtein_seconds": float(levenshtein_seconds),
        "timing_total_seconds": float(time.perf_counter() - eval_started_at),
    }


def _line_count_fields(*, ref_to_pred_payload: dict, ref_to_ref_payload: dict) -> dict:
    """Build shared line-count fields for valid and invalid evaluation rows."""
    return {
        "used_line_count": int(ref_to_pred_payload["used_line_count"]),
        "used_line_count_ref_to_ref": int(ref_to_ref_payload["used_line_count"]),
        "line_guided_columns": int(ref_to_pred_payload["line_guided_columns"]),
        "fallback_columns": int(ref_to_pred_payload["fallback_columns"]),
        "raw_line_count": int(ref_to_pred_payload["raw_line_count"]),
        "raw_line_count_ref_to_ref": int(ref_to_ref_payload["raw_line_count"]),
        "candidate_line_count": int(ref_to_pred_payload["candidate_line_count"]),
        "candidate_line_count_ref_to_ref": int(ref_to_ref_payload["candidate_line_count"]),
        "threshold_start": float(ref_to_pred_payload["threshold_start"]),
        "threshold_start_ref_to_ref": float(ref_to_ref_payload["threshold_start"]),
    }


def _invalid_coverage_eval_row(
    *,
    error: Exception,
    weighted_result,
    ref_to_pred_payload: dict,
    ref_to_ref_payload: dict,
    y_diff_diagnostics: dict,
    coverage_seconds: float,
    levenshtein_seconds: float,
    eval_started_at: float,
) -> dict:
    """Return a stable invalid row when v2.12 rejects coverage categories."""
    weighted_along_lines_nls = weighted_result.weighted_along_lines_nls
    return {
        "is_valid": False,
        "invalid_reason": INVALID_REASON_COVERAGE_Y_DIFF_BELOW_MINUS_ONE,
        "invalid_error_message": str(error),
        "tuning_score": None,
        # Keep the single internal weighted value used by ranking; the public
        # report writer renames it to the long human-facing metric label.
        "weighted_along_lines_nls": None if weighted_along_lines_nls is None else float(weighted_along_lines_nls),
        "line_count": int(weighted_result.scored_line_count),
        "total_line_length": float(weighted_result.total_line_length),
        "correct_ref_coverage": None,
        "missing_ref_coverage": None,
        "repetition_on_ref": None,
        "hallucination": None,
        **_line_count_fields(ref_to_pred_payload=ref_to_pred_payload, ref_to_ref_payload=ref_to_ref_payload),
        **y_diff_diagnostics,
        **_timing_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
            coverage_seconds=float(coverage_seconds),
            levenshtein_seconds=float(levenshtein_seconds),
            eval_started_at=float(eval_started_at),
        ),
    }


def _detect_filter_and_build_bundle(
    *,
    matrix: np.ndarray,
    hough_ctx: dict,
    document_index: int,
    ref_text_len: int,
    other_text_len: int,
    window_size: int,
    window_stride: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
    include_reference_self_coverage_array: bool = False,
) -> dict:
    """Detect, filter, and bundle one matrix direction for one combination."""
    direction_started_at = time.perf_counter()

    t_detect = time.perf_counter()
    det = detect_lines_only_from_hough_ctx(
        hough_ctx=hough_ctx,
        seed=int(hough_seed) + int(document_index),
        threshold=int(hough_threshold),
        line_length=int(hough_line_length),
        line_gap=int(hough_line_gap),
    )
    detect_seconds = time.perf_counter() - t_detect

    t_filter = time.perf_counter()
    filtered = filter_lines_after_hough(
        matrix=matrix,
        det_result=det,
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
        matrix_is_prepared=True,
    )
    filter_seconds = time.perf_counter() - t_filter

    t_bundle = time.perf_counter()
    n_ref_windows = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    n_other_windows = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    bundle = build_v212_line_metric_bundle(
        lines_used=filtered["lines_used"],
        column_assignment=filtered["column_assignment"],
        n_ref_windows=n_ref_windows,
        n_other_windows=n_other_windows,
        ref_text_len=int(ref_text_len),
        other_text_len=int(other_text_len),
        window_size=int(window_size),
        window_stride=int(window_stride),
    )
    bundle_seconds = time.perf_counter() - t_bundle

    mapped_line_id = np.asarray(filtered["column_assignment"].get("mapped_line_id", []), dtype=int)
    line_guided_columns = int(np.sum(mapped_line_id >= 0)) if mapped_line_id.size else 0
    fallback_columns = int(np.sum(mapped_line_id < 0)) if mapped_line_id.size else 0
    refref_y = (
        build_v212_refref_y_coverage_array_from_bundle(refref_bundle=bundle)
        if bool(include_reference_self_coverage_array)
        else None
    )

    return {
        "det": det,
        "filtered": filtered,
        "bundle": bundle,
        "refref_y": None if refref_y is None else np.asarray(refref_y, dtype=np.int32),
        "line_guided_columns": int(line_guided_columns),
        "fallback_columns": int(fallback_columns),
        "raw_line_count": int(len(det.get("raw_lines", []))),
        "candidate_line_count": int(len(filtered.get("lines_for_filtering", []))),
        "used_line_count": int(len(filtered.get("lines_used", []))),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "timing_hough_detect_seconds": float(detect_seconds),
        "timing_filter_seconds": float(filter_seconds),
        "timing_build_bundle_seconds": float(bundle_seconds),
        "timing_direction_total_seconds": float(time.perf_counter() - direction_started_at),
    }


def compute_reference_self_payload_for_combination(
    *,
    doc: SweepDocument,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> dict:
    """Compute the exact reference-self payload for one Hough combination.

    This is the single implementation used both by the normal evaluator and by
    the cache warm-up stage.  Keeping it here prevents the warm path from having
    its own subtly different copy of the ref-to-ref Hough/filter/bundle logic.
    """
    return _detect_filter_and_build_bundle(
        matrix=doc.ref_to_ref_matrix,
        hough_ctx=doc.ref_to_ref_hough_ctx,
        document_index=int(doc.index),
        ref_text_len=len(doc.ref),
        other_text_len=len(doc.ref),
        window_size=int(doc.window_size),
        window_stride=int(doc.window_stride),
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
        include_reference_self_coverage_array=True,
    )


def evaluate_single_combination(
    *,
    doc: SweepDocument,
    cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    combination_bundle_logger=None,
) -> dict:
    """Evaluate one ``(threshold, line_length, line_gap, seed)`` combination."""
    return evaluate_single_combination_values(
        doc=doc,
        hough_threshold=int(cfg.hough_threshold),
        hough_line_length=int(cfg.hough_line_length),
        hough_line_gap=int(cfg.hough_line_gap),
        hough_seed=int(cfg.hough_seed),
        align_abs_min_len=float(cfg.align_abs_min_len),
        align_min_iou_threshold=float(cfg.align_min_iou_threshold),
        levenshtein_backend=str(levenshtein_backend),
        combination_bundle_logger=combination_bundle_logger,
    )


def evaluate_single_combination_values(
    *,
    doc: SweepDocument,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
    levenshtein_backend: str,
    ref_to_ref_cache=None,
    combination_bundle_logger=None,
) -> dict:
    """Evaluate one combination from scalar values without allocating a config."""
    eval_started_at = time.perf_counter()

    ref_to_pred_payload = _detect_filter_and_build_bundle(
        matrix=doc.ref_to_pred_matrix,
        hough_ctx=doc.ref_to_pred_hough_ctx,
        document_index=int(doc.index),
        ref_text_len=len(doc.ref),
        other_text_len=len(doc.pred),
        window_size=int(doc.window_size),
        window_stride=int(doc.window_stride),
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
    )

    t_lev = time.perf_counter()
    weighted_result = compute_weighted_along_lines_similarity_from_bundle(
        ref_blocks=doc.ref_blocks,
        other_blocks=doc.pred_blocks,
        lines_used=ref_to_pred_payload["filtered"]["lines_used"],
        bundle=ref_to_pred_payload["bundle"],
        levenshtein_backend=str(levenshtein_backend),
    )
    levenshtein_seconds = time.perf_counter() - t_lev

    def compute_ref_to_ref_payload() -> dict:
        """Compute the exact reference-self payload for this combination."""
        return compute_reference_self_payload_for_combination(
            doc=doc,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
        )

    if ref_to_ref_cache is None:
        ref_to_ref_payload = compute_ref_to_ref_payload()
    else:
        ref_to_ref_payload = ref_to_ref_cache.get_or_compute(
            doc=doc,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
            compute_payload=compute_ref_to_ref_payload,
        )

    t_coverage = time.perf_counter()
    if ref_to_ref_payload.get("refref_y") is not None:
        coverage_arrays = build_v212_line_coverage_arrays_from_cached_refref_y(
            refref_y=ref_to_ref_payload["refref_y"],
            other_bundle=ref_to_pred_payload["bundle"],
        )
    else:
        coverage_arrays = build_v212_line_coverage_arrays_from_bundles(
            refref_bundle=ref_to_ref_payload["bundle"],
            other_bundle=ref_to_pred_payload["bundle"],
        )
    y_diff_diagnostics = _coverage_y_diff_diagnostics(coverage_arrays["y_diff"])
    try:
        coverage_ratio_metrics = compute_v212_line_coverage_ratio_metrics_from_arrays(
            y_diff=coverage_arrays["y_diff"],
            other_x=coverage_arrays["other_x"],
            file_name=str(doc.fname),
        )
    except ValueError as exc:
        coverage_seconds = time.perf_counter() - t_coverage
        if int(y_diff_diagnostics.get("coverage_y_diff_lt_minus_one_count", 0)) > 0:
            del coverage_arrays
            eval_row = _invalid_coverage_eval_row(
                error=exc,
                weighted_result=weighted_result,
                ref_to_pred_payload=ref_to_pred_payload,
                ref_to_ref_payload=ref_to_ref_payload,
                y_diff_diagnostics=y_diff_diagnostics,
                coverage_seconds=float(coverage_seconds),
                levenshtein_seconds=float(levenshtein_seconds),
                eval_started_at=float(eval_started_at),
            )
            if combination_bundle_logger is not None:
                combination_bundle_logger.write_combination(
                    doc=doc,
                    hough_threshold=int(hough_threshold),
                    hough_line_length=int(hough_line_length),
                    hough_line_gap=int(hough_line_gap),
                    hough_seed=int(hough_seed),
                    align_abs_min_len=float(align_abs_min_len),
                    align_min_iou_threshold=float(align_min_iou_threshold),
                    eval_row=eval_row,
                    ref_to_pred_payload=ref_to_pred_payload,
                    ref_to_ref_payload=ref_to_ref_payload,
                )
            return eval_row
        raise

    normalized_coverage_metrics = normalize_v212_coverage_metrics(coverage_ratio_metrics)
    coverage_seconds = time.perf_counter() - t_coverage

    weighted_along_lines_nls = weighted_result.weighted_along_lines_nls
    tuning_score = compute_harmonic_tuning_score(
        weighted_along_lines_nls=weighted_along_lines_nls,
        correct_ref_coverage=normalized_coverage_metrics["correct_ref_coverage"],
        hallucination=normalized_coverage_metrics["hallucination"],
    )

    # Drop large temporary arrays as soon as scalar metrics are extracted.
    del coverage_arrays

    eval_row = {
        "is_valid": True,
        "invalid_reason": None,
        "invalid_error_message": None,
        "tuning_score": float(tuning_score),
        # This internal key is intentionally short because it is read often by
        # reducers; output serializers expose the longer report label.
        "weighted_along_lines_nls": None if weighted_along_lines_nls is None else float(weighted_along_lines_nls),
        "line_count": int(weighted_result.scored_line_count),
        "total_line_length": float(weighted_result.total_line_length),
        "correct_ref_coverage": float(normalized_coverage_metrics["correct_ref_coverage"]),
        "missing_ref_coverage": float(normalized_coverage_metrics["missing_ref_coverage"]),
        "repetition_on_ref": float(normalized_coverage_metrics["repetition_on_ref"]),
        "hallucination": float(normalized_coverage_metrics["hallucination"]),
        **_line_count_fields(ref_to_pred_payload=ref_to_pred_payload, ref_to_ref_payload=ref_to_ref_payload),
        **y_diff_diagnostics,
        **_timing_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
            coverage_seconds=float(coverage_seconds),
            levenshtein_seconds=float(levenshtein_seconds),
            eval_started_at=float(eval_started_at),
        ),
    }
    if combination_bundle_logger is not None:
        combination_bundle_logger.write_combination(
            doc=doc,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
            eval_row=eval_row,
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
        )
    return eval_row


__all__ = [
    "INVALID_REASON_COVERAGE_Y_DIFF_BELOW_MINUS_ONE",
    "is_finite_tuning_score",
    "is_finite_along_lines",
    "evaluation_rank_key",
    "pick_better_eval",
    "compute_reference_self_payload_for_combination",
    "evaluate_single_combination",
    "evaluate_single_combination_values",
]
