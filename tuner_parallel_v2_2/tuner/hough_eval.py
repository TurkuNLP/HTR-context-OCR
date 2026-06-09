from __future__ import annotations

"""Per-combination Hough evaluation helpers.

Each call evaluates exactly one ``(threshold, line_length, line_gap, seed)``
combination on one already-prepared ``SweepDocument``.  Matrix building and Hough
context preparation happen once in document preparation; this hot path only runs
Hough detection, filtering, compact v2.12-compatible scoring-payload/coverage
logic, and line-level Levenshtein scoring.
"""

import math
import time

import numpy as np

try:
    from ..alignment.line_alignment_pipeline_fast import detect_lines_only_from_hough_ctx, filter_lines_after_hough
    from ..metrics.alignment_quality_score import (
        WeightedAlongLinesResult,
        compute_alignment_evidence_selection_score,
        compute_harmonic_tuning_score,
        compute_non_hallucination_weighted_tuning_score,
        compute_line_guided_fraction,
        compute_line_level_similarity_records_from_assignment,
        compute_score_matrix_support_from_lines,
        compute_weighted_along_lines_similarity_from_compact_payload,
        normalize_v212_coverage_metrics,
        weighted_along_lines_result_from_line_similarity_records,
    )
    from ..metrics.v2_12_metric_adapter import (
        build_v212_compact_line_scoring_payload,
        build_v212_line_coverage_arrays_from_cached_refref_y,
        build_v212_line_coverage_arrays_from_bundles,
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
        WeightedAlongLinesResult,
        compute_alignment_evidence_selection_score,
        compute_harmonic_tuning_score,
        compute_non_hallucination_weighted_tuning_score,
        compute_line_guided_fraction,
        compute_line_level_similarity_records_from_assignment,
        compute_score_matrix_support_from_lines,
        compute_weighted_along_lines_similarity_from_compact_payload,
        normalize_v212_coverage_metrics,
        weighted_along_lines_result_from_line_similarity_records,
    )
    from metrics.v2_12_metric_adapter import (  # type: ignore
        build_v212_compact_line_scoring_payload,
        build_v212_line_coverage_arrays_from_cached_refref_y,
        build_v212_line_coverage_arrays_from_bundles,
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
METRIC_OUTCOME_REASON_LINE_NLS_FILTER_REMOVED_ALL_LINES = "line_nls_filter_removed_all_ref_to_pred_lines"

SELECTION_OBJECTIVE_STRICT_QUALITY = "strict_quality"
SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE = "alignment_evidence"
SELECTION_OBJECTIVE_NON_HALLUCINATION_WEIGHTED = "non_hallucination_weighted"
DEFAULT_SELECTION_OBJECTIVE = SELECTION_OBJECTIVE_STRICT_QUALITY
SUPPORTED_SELECTION_OBJECTIVES = (
    SELECTION_OBJECTIVE_STRICT_QUALITY,
    SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE,
    SELECTION_OBJECTIVE_NON_HALLUCINATION_WEIGHTED,
)

FILTER_PROFILE_TIMING_FIELDS = (
    "filter_prepare_candidates_seconds",
    "filter_build_candidate_coverages_seconds",
    "filter_possible_pair_generation_seconds",
    "filter_exact_iou_seconds",
    "filter_component_build_seconds",
    "filter_merge_components_seconds",
    "filter_final_assignment_seconds",
    "filter_finalize_outputs_seconds",
    "filter_total_profiled_seconds",
)

FILTER_PROFILE_COUNT_FIELDS = (
    "filter_input_line_count",
    "filter_prepared_candidate_count",
    "filter_candidate_coverage_count",
    "filter_possible_overlap_pair_count",
    "filter_merge_edge_count",
    "filter_component_count",
    "filter_merged_coverage_count",
    "filter_finalize_prune_iteration_count",
    "filter_final_line_count",
    "filter_fallback_candidate_used",
)


def _directional_filter_profile_fields(
    *,
    profile: dict | None,
    direction_label: str,
) -> dict:
    """Return scalar filter-profile fields for one matrix direction."""
    if profile is None:
        return {}

    fields: dict[str, float | int] = {}
    for field_name in FILTER_PROFILE_TIMING_FIELDS:
        base_name = field_name.removesuffix("_seconds")
        fields[f"timing_{base_name}_{direction_label}_seconds"] = float(
            profile.get(field_name, 0.0) or 0.0
        )
    for field_name in FILTER_PROFILE_COUNT_FIELDS:
        fields[f"{field_name}_{direction_label}"] = int(profile.get(field_name, 0) or 0)
    return fields


def _filter_profile_fields(*, ref_to_pred_payload: dict, ref_to_ref_payload: dict) -> dict:
    """Return compact directional profiling fields when profiling was enabled."""
    fields: dict = {}
    fields.update(
        _directional_filter_profile_fields(
            profile=ref_to_pred_payload.get("filter_profile"),
            direction_label="ref_to_pred",
        )
    )
    fields.update(
        _directional_filter_profile_fields(
            profile=ref_to_ref_payload.get("filter_profile"),
            direction_label="ref_to_ref",
        )
    )
    return fields


def _line_nls_filter_payload_fields(
    *,
    enabled: bool,
    minimum: float | None,
    input_line_count: int = 0,
    scored_line_count: int = 0,
    removed_line_count: int = 0,
    surviving_line_count: int = 0,
    removed_column_count: int = 0,
    surviving_column_count: int = 0,
    all_lines_removed: bool = False,
    seconds: float = 0.0,
) -> dict:
    """Return stable scalar fields for the optional final-line NLS gate."""
    return {
        "line_nls_filter_enabled": bool(enabled),
        "min_surviving_line_nls": None if minimum is None else float(minimum),
        "line_nls_filter_input_line_count": int(input_line_count),
        "line_nls_filter_scored_line_count": int(scored_line_count),
        "line_nls_filter_removed_line_count": int(removed_line_count),
        "line_nls_filter_surviving_line_count": int(surviving_line_count),
        "line_nls_filter_removed_column_count": int(removed_column_count),
        "line_nls_filter_surviving_column_count": int(surviving_column_count),
        "line_nls_filter_all_lines_removed": bool(all_lines_removed),
        "timing_line_nls_filter_seconds": float(seconds),
    }


def _line_nls_filter_fields_from_payload(ref_to_pred_payload: dict) -> dict:
    """Copy line-NLS filter fields from the ref-to-pred direction payload."""
    return _line_nls_filter_payload_fields(
        enabled=bool(ref_to_pred_payload.get("line_nls_filter_enabled", False)),
        minimum=ref_to_pred_payload.get("min_surviving_line_nls"),
        input_line_count=int(ref_to_pred_payload.get("line_nls_filter_input_line_count", 0) or 0),
        scored_line_count=int(ref_to_pred_payload.get("line_nls_filter_scored_line_count", 0) or 0),
        removed_line_count=int(ref_to_pred_payload.get("line_nls_filter_removed_line_count", 0) or 0),
        surviving_line_count=int(ref_to_pred_payload.get("line_nls_filter_surviving_line_count", 0) or 0),
        removed_column_count=int(ref_to_pred_payload.get("line_nls_filter_removed_column_count", 0) or 0),
        surviving_column_count=int(ref_to_pred_payload.get("line_nls_filter_surviving_column_count", 0) or 0),
        all_lines_removed=bool(ref_to_pred_payload.get("line_nls_filter_all_lines_removed", False)),
        seconds=float(ref_to_pred_payload.get("timing_line_nls_filter_seconds", 0.0) or 0.0),
    )


def _filter_ref_to_pred_lines_by_minimum_nls(
    *,
    filtered: dict,
    ref_blocks: list[str],
    pred_blocks: list[str],
    n_ref_windows: int,
    min_surviving_line_nls: float,
    levenshtein_backend: str,
) -> tuple[dict, WeightedAlongLinesResult, dict]:
    """Remove final ref-to-pred lines whose exact line-level NLS is too low.

    This runs after true-IoU geometry filtering, when each prediction column has
    its final line owner.  The helper never changes raw Hough output or ref-to-ref
    baselines; it only prunes the final ref-to-pred line list and ownership
    arrays before metric payload construction.
    """
    original_lines = list(filtered.get("lines_used", []))
    original_assignment = filtered.get("column_assignment", {})
    original_mapped_y = np.asarray(original_assignment.get("mapped_y", []), dtype=float)
    original_mapped_line_id = np.asarray(original_assignment.get("mapped_line_id", []), dtype=int)
    original_guided_column_count = int(np.count_nonzero(original_mapped_line_id >= 0))

    line_similarity_records = compute_line_level_similarity_records_from_assignment(
        ref_blocks=ref_blocks,
        other_blocks=pred_blocks,
        lines_used=original_lines,
        column_assignment=original_assignment,
        n_ref_windows=int(n_ref_windows),
        levenshtein_backend=str(levenshtein_backend),
    )
    scored_record_by_line_id = {
        int(record.line_id): record
        for record in line_similarity_records
    }
    kept_original_line_ids = [
        int(line_id)
        for line_id in range(len(original_lines))
        if (
            int(line_id) in scored_record_by_line_id
            and float(scored_record_by_line_id[int(line_id)].normalized_levenshtein_similarity)
            >= float(min_surviving_line_nls)
        )
    ]
    kept_original_line_id_set = set(kept_original_line_ids)

    pruned_lines: list[dict] = []
    original_to_new_line_id: dict[int, int] = {}
    for new_line_id, original_line_id in enumerate(kept_original_line_ids):
        line_copy = dict(original_lines[int(original_line_id)])
        score_record = scored_record_by_line_id[int(original_line_id)]
        # Keep the exact text score on the surviving line so diagnostics and
        # optional visuals can explain why this line passed the text-quality gate.
        line_copy["line_nls"] = float(score_record.normalized_levenshtein_similarity)
        pruned_lines.append(line_copy)
        original_to_new_line_id[int(original_line_id)] = int(new_line_id)

    pruned_mapped_y = np.full(original_mapped_y.shape, np.nan, dtype=float)
    pruned_mapped_line_id = np.full(original_mapped_line_id.shape, -1, dtype=int)
    for original_line_id, new_line_id in original_to_new_line_id.items():
        owned_column_mask = original_mapped_line_id == int(original_line_id)
        pruned_mapped_line_id[owned_column_mask] = int(new_line_id)
        pruned_mapped_y[owned_column_mask] = original_mapped_y[owned_column_mask]

    pruned_assignment = {
        "mapped_y": pruned_mapped_y,
        "mapped_line_id": pruned_mapped_line_id,
    }
    pruned_filtered = {
        **filtered,
        "lines_used": pruned_lines,
        "column_assignment": pruned_assignment,
    }

    surviving_records = [
        scored_record_by_line_id[int(line_id)]
        for line_id in kept_original_line_ids
        if int(line_id) in scored_record_by_line_id
    ]
    weighted_result = weighted_along_lines_result_from_line_similarity_records(surviving_records)
    surviving_guided_column_count = int(np.count_nonzero(pruned_mapped_line_id >= 0))
    removed_column_count = max(0, int(original_guided_column_count - surviving_guided_column_count))
    all_lines_removed = bool(len(original_lines) > 0 and len(pruned_lines) == 0)

    filter_fields = _line_nls_filter_payload_fields(
        enabled=True,
        minimum=float(min_surviving_line_nls),
        input_line_count=int(len(original_lines)),
        scored_line_count=int(len(line_similarity_records)),
        removed_line_count=int(len(original_lines) - len(pruned_lines)),
        surviving_line_count=int(len(pruned_lines)),
        removed_column_count=int(removed_column_count),
        surviving_column_count=int(surviving_guided_column_count),
        all_lines_removed=all_lines_removed,
    )
    return pruned_filtered, weighted_result, filter_fields


def _empty_ref_to_ref_payload_for_line_nls_short_circuit() -> dict:
    """Return a zero-work ref-to-ref payload for line-NLS all-removed rows."""
    return {
        "det": {},
        "filtered": {"lines_used": [], "column_assignment": {"mapped_y": np.asarray([]), "mapped_line_id": np.asarray([])}},
        "scoring_payload": {},
        "refref_y": None,
        "line_guided_columns": 0,
        "fallback_columns": 0,
        "raw_line_count": 0,
        "skimage_raw_line_count_before_direction_filter": 0,
        "direction_rejected_line_count": 0,
        "candidate_line_count": 0,
        "used_line_count": 0,
        "threshold_start": float("nan"),
        "timing_hough_detect_seconds": 0.0,
        "timing_filter_seconds": 0.0,
        "timing_build_bundle_seconds": 0.0,
        "timing_direction_total_seconds": 0.0,
        "timing_line_nls_filter_seconds": 0.0,
    }


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


def normalize_selection_objective(selection_objective: str | None) -> str:
    """Return a supported objective name or raise a clear configuration error."""
    normalized = DEFAULT_SELECTION_OBJECTIVE if selection_objective is None else str(selection_objective).strip()
    if normalized not in SUPPORTED_SELECTION_OBJECTIVES:
        raise ValueError(
            f"Unsupported selection_objective {selection_objective!r}; "
            f"expected one of {SUPPORTED_SELECTION_OBJECTIVES!r}"
        )
    return normalized


def evaluation_rank_key(row: dict, *, selection_objective: str = DEFAULT_SELECTION_OBJECTIVE) -> tuple:
    """Return a strict deterministic ranking tuple for best-evaluation selection."""
    objective = normalize_selection_objective(selection_objective)
    hallucination = _finite_float_for_rank(row.get("hallucination"), 1.0)

    if objective == SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE:
        return (
            _finite_float_for_rank(row.get("alignment_selection_score"), float("-inf")),
            _finite_float_for_rank(row.get("score_matrix_support"), float("-inf")),
            _finite_float_for_rank(row.get("line_guided_fraction"), float("-inf")),
            _finite_float_for_rank(row.get("weighted_along_lines_nls"), float("-inf")),
            -float(hallucination),
            _finite_float_for_rank(row.get("tuning_score"), float("-inf")),
            _finite_float_for_rank(row.get("correct_ref_coverage"), float("-inf")),
            int(row.get("line_guided_columns", 0)),
            -int(row.get("fallback_columns", 0)),
            -int(row.get(PARAM_HOUGH_THRESHOLD, 0)),
            -int(row.get(PARAM_HOUGH_LINE_LENGTH, 0)),
            -int(row.get(PARAM_HOUGH_LINE_GAP, 0)),
            -int(row.get(PARAM_HOUGH_SEED, 0)),
        )

    if objective == SELECTION_OBJECTIVE_NON_HALLUCINATION_WEIGHTED:
        return (
            _finite_float_for_rank(row.get("non_hallucination_weighted_tuning_score"), float("-inf")),
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


def pick_better_eval(
    current: dict | None,
    candidate: dict | None,
    *,
    selection_objective: str = DEFAULT_SELECTION_OBJECTIVE,
) -> dict | None:
    """Return the better of two evaluation rows according to the selected objective."""
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
    if evaluation_rank_key(candidate, selection_objective=selection_objective) > evaluation_rank_key(
        current,
        selection_objective=selection_objective,
    ):
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
    include_filter_profile: bool = False,
) -> dict:
    """Build shared timing fields for valid and invalid evaluation rows."""
    timing_hough_detect_ref_to_pred_seconds = float(ref_to_pred_payload["timing_hough_detect_seconds"])
    timing_filter_ref_to_pred_seconds = float(ref_to_pred_payload["timing_filter_seconds"])
    timing_hough_detect_ref_to_ref_seconds = float(ref_to_ref_payload["timing_hough_detect_seconds"])
    timing_filter_ref_to_ref_seconds = float(ref_to_ref_payload["timing_filter_seconds"])
    timing_build_bundle_seconds = float(
        ref_to_pred_payload["timing_build_bundle_seconds"] + ref_to_ref_payload["timing_build_bundle_seconds"]
    )
    timing_line_nls_filter_seconds = float(
        ref_to_pred_payload.get("timing_line_nls_filter_seconds", 0.0)
        + ref_to_ref_payload.get("timing_line_nls_filter_seconds", 0.0)
    )
    timing_hough_detect_seconds = float(
        timing_hough_detect_ref_to_pred_seconds + timing_hough_detect_ref_to_ref_seconds
    )
    timing_filter_seconds = float(timing_filter_ref_to_pred_seconds + timing_filter_ref_to_ref_seconds)
    timing_detect_filter_seconds = float(timing_hough_detect_seconds + timing_filter_seconds)

    fields = {
        "timing_hough_detect_ref_to_pred_seconds": timing_hough_detect_ref_to_pred_seconds,
        "timing_filter_ref_to_pred_seconds": timing_filter_ref_to_pred_seconds,
        "timing_hough_detect_ref_to_ref_seconds": timing_hough_detect_ref_to_ref_seconds,
        "timing_filter_ref_to_ref_seconds": timing_filter_ref_to_ref_seconds,
        "timing_hough_detect_seconds": timing_hough_detect_seconds,
        "timing_filter_seconds": timing_filter_seconds,
        "timing_detect_filter_seconds": timing_detect_filter_seconds,
        "timing_build_bundle_seconds": timing_build_bundle_seconds,
        "timing_line_nls_filter_seconds": timing_line_nls_filter_seconds,
        "timing_coverage_seconds": float(coverage_seconds),
        "timing_levenshtein_seconds": float(levenshtein_seconds),
        "timing_total_seconds": float(time.perf_counter() - eval_started_at),
    }
    if bool(include_filter_profile):
        fields.update(
            _filter_profile_fields(
                ref_to_pred_payload=ref_to_pred_payload,
                ref_to_ref_payload=ref_to_ref_payload,
            )
        )
    return fields


def _line_count_fields(*, ref_to_pred_payload: dict, ref_to_ref_payload: dict) -> dict:
    """Build shared line-count fields for valid and invalid evaluation rows."""
    return {
        "used_line_count": int(ref_to_pred_payload["used_line_count"]),
        "used_line_count_ref_to_ref": int(ref_to_ref_payload["used_line_count"]),
        "line_guided_columns": int(ref_to_pred_payload["line_guided_columns"]),
        "fallback_columns": int(ref_to_pred_payload["fallback_columns"]),
        "raw_line_count": int(ref_to_pred_payload["raw_line_count"]),
        "raw_line_count_ref_to_ref": int(ref_to_ref_payload["raw_line_count"]),
        "skimage_raw_line_count_before_direction_filter": int(
            ref_to_pred_payload.get("skimage_raw_line_count_before_direction_filter", ref_to_pred_payload["raw_line_count"])
        ),
        "skimage_raw_line_count_before_direction_filter_ref_to_ref": int(
            ref_to_ref_payload.get("skimage_raw_line_count_before_direction_filter", ref_to_ref_payload["raw_line_count"])
        ),
        "direction_rejected_line_count": int(ref_to_pred_payload.get("direction_rejected_line_count", 0)),
        "direction_rejected_line_count_ref_to_ref": int(ref_to_ref_payload.get("direction_rejected_line_count", 0)),
        "candidate_line_count": int(ref_to_pred_payload["candidate_line_count"]),
        "candidate_line_count_ref_to_ref": int(ref_to_ref_payload["candidate_line_count"]),
        "threshold_start": float(ref_to_pred_payload["threshold_start"]),
        "threshold_start_ref_to_ref": float(ref_to_ref_payload["threshold_start"]),
    }


def _selection_metric_fields(
    *,
    ref_to_pred_payload: dict,
    weighted_along_lines_nls,
    hallucination,
) -> dict:
    """Return scalar fields used by the optional alignment-evidence selector."""
    filtered_payload = ref_to_pred_payload.get("filtered", {}) if isinstance(ref_to_pred_payload, dict) else {}
    lines_used = filtered_payload.get("lines_used", []) if isinstance(filtered_payload, dict) else []
    line_guided_fraction = compute_line_guided_fraction(
        line_guided_columns=ref_to_pred_payload.get("line_guided_columns", 0),
        fallback_columns=ref_to_pred_payload.get("fallback_columns", 0),
    )
    score_matrix_support = compute_score_matrix_support_from_lines(lines_used if isinstance(lines_used, list) else [])
    alignment_selection_score = compute_alignment_evidence_selection_score(
        weighted_along_lines_nls=weighted_along_lines_nls,
        score_matrix_support=score_matrix_support,
        line_guided_fraction=line_guided_fraction,
        hallucination=hallucination,
    )
    return {
        "alignment_selection_score": float(alignment_selection_score),
        "score_matrix_support": float(score_matrix_support),
        "line_guided_fraction": float(line_guided_fraction),
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
    include_filter_profile: bool = False,
) -> dict:
    """Return a stable invalid row when v2.12 rejects coverage categories."""
    weighted_along_lines_nls = weighted_result.weighted_along_lines_nls
    return {
        "is_valid": False,
        "invalid_reason": INVALID_REASON_COVERAGE_Y_DIFF_BELOW_MINUS_ONE,
        "invalid_error_message": str(error),
        "metric_outcome_reason": None,
        "tuning_score": None,
        "non_hallucination_weighted_tuning_score": None,
        # Keep the single internal weighted value used by ranking; the public
        # report writer renames it to the long human-facing metric label.
        "weighted_along_lines_nls": None if weighted_along_lines_nls is None else float(weighted_along_lines_nls),
        "line_count": int(weighted_result.scored_line_count),
        "total_line_length": float(weighted_result.total_line_length),
        "correct_ref_coverage": None,
        "missing_ref_coverage": None,
        "repetition_on_ref": None,
        "hallucination": None,
        **_selection_metric_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            weighted_along_lines_nls=weighted_along_lines_nls,
            hallucination=1.0,
        ),
        **_line_count_fields(ref_to_pred_payload=ref_to_pred_payload, ref_to_ref_payload=ref_to_ref_payload),
        **_line_nls_filter_fields_from_payload(ref_to_pred_payload),
        **y_diff_diagnostics,
        **_timing_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
            coverage_seconds=float(coverage_seconds),
            levenshtein_seconds=float(levenshtein_seconds),
            eval_started_at=float(eval_started_at),
            include_filter_profile=bool(include_filter_profile),
        ),
    }


def _line_nls_removed_all_eval_row(
    *,
    weighted_result: WeightedAlongLinesResult,
    ref_to_pred_payload: dict,
    ref_to_ref_payload: dict,
    levenshtein_seconds: float,
    eval_started_at: float,
    include_filter_profile: bool = False,
) -> dict:
    """Return a valid zero-score row when the line-NLS gate removes all lines."""
    return {
        "is_valid": True,
        "invalid_reason": None,
        "invalid_error_message": None,
        "metric_outcome_reason": METRIC_OUTCOME_REASON_LINE_NLS_FILTER_REMOVED_ALL_LINES,
        "tuning_score": 0.0,
        "non_hallucination_weighted_tuning_score": 0.0,
        "weighted_along_lines_nls": None,
        "line_count": int(weighted_result.scored_line_count),
        "total_line_length": float(weighted_result.total_line_length),
        "correct_ref_coverage": 0.0,
        "missing_ref_coverage": 1.0,
        "repetition_on_ref": 0.0,
        "hallucination": 1.0,
        **_selection_metric_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            weighted_along_lines_nls=None,
            hallucination=1.0,
        ),
        **_line_count_fields(ref_to_pred_payload=ref_to_pred_payload, ref_to_ref_payload=ref_to_ref_payload),
        **_line_nls_filter_fields_from_payload(ref_to_pred_payload),
        "coverage_y_diff_size": 0,
        "coverage_y_diff_min": None,
        "coverage_y_diff_max": None,
        "coverage_y_diff_le_minus_one_count": 0,
        "coverage_y_diff_lt_minus_one_count": 0,
        "coverage_y_diff_below_minus_one_counts_json": {},
        **_timing_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
            coverage_seconds=0.0,
            levenshtein_seconds=float(levenshtein_seconds),
            eval_started_at=float(eval_started_at),
            include_filter_profile=bool(include_filter_profile),
        ),
    }


def _detect_filter_and_build_scoring_payload(
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
    filter_profile: dict | None = None,
    min_surviving_line_nls: float | None = None,
    line_nls_filter_ref_blocks: list[str] | None = None,
    line_nls_filter_other_blocks: list[str] | None = None,
    levenshtein_backend: str | None = None,
) -> dict:
    """Detect, filter, and build one compact scoring payload for one direction."""
    direction_started_at = time.perf_counter()

    t_detect = time.perf_counter()
    det = detect_lines_only_from_hough_ctx(
        hough_ctx=hough_ctx,
        # The Hough seed is now passed literally.  Document identity must not
        # change the probabilistic Hough random stream for the same parameters.
        seed=int(hough_seed),
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
        profile=filter_profile,
    )
    filter_seconds = time.perf_counter() - t_filter

    precomputed_weighted_result = None
    line_nls_filter_fields = _line_nls_filter_payload_fields(
        enabled=min_surviving_line_nls is not None,
        minimum=min_surviving_line_nls,
    )
    if min_surviving_line_nls is not None:
        if line_nls_filter_ref_blocks is None or line_nls_filter_other_blocks is None:
            raise ValueError("line-NLS filtering requires reference and prediction text blocks")
        if levenshtein_backend is None:
            raise ValueError("line-NLS filtering requires a Levenshtein backend")

        t_line_nls_filter = time.perf_counter()
        filtered, precomputed_weighted_result, line_nls_filter_fields = (
            _filter_ref_to_pred_lines_by_minimum_nls(
                filtered=filtered,
                ref_blocks=line_nls_filter_ref_blocks,
                pred_blocks=line_nls_filter_other_blocks,
                n_ref_windows=int(matrix.shape[0]) if matrix.ndim == 2 else 0,
                min_surviving_line_nls=float(min_surviving_line_nls),
                levenshtein_backend=str(levenshtein_backend),
            )
        )
        line_nls_filter_fields["timing_line_nls_filter_seconds"] = float(
            time.perf_counter() - t_line_nls_filter
        )

    # Keep the historical timing field name in the exported CSV schema, but the
    # measured work is now compact scoring-payload construction rather than the
    # full verbose v2.12 diagnostic bundle.
    t_scoring_payload = time.perf_counter()
    n_ref_windows = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    n_other_windows = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    scoring_payload = build_v212_compact_line_scoring_payload(
        lines_used=filtered["lines_used"],
        column_assignment=filtered["column_assignment"],
        n_ref_windows=n_ref_windows,
        n_other_windows=n_other_windows,
        ref_text_len=int(ref_text_len),
        other_text_len=int(other_text_len),
        window_size=int(window_size),
        window_stride=int(window_stride),
    )
    scoring_payload_seconds = time.perf_counter() - t_scoring_payload

    mapped_line_id = np.asarray(filtered["column_assignment"].get("mapped_line_id", []), dtype=int)
    line_guided_columns = int(np.sum(mapped_line_id >= 0)) if mapped_line_id.size else 0
    fallback_columns = int(np.sum(mapped_line_id < 0)) if mapped_line_id.size else 0
    refref_y = (
        build_v212_refref_y_coverage_array_from_bundle(refref_bundle=scoring_payload)
        if bool(include_reference_self_coverage_array)
        else None
    )

    payload = {
        "det": det,
        "filtered": filtered,
        "scoring_payload": scoring_payload,
        "refref_y": None if refref_y is None else np.asarray(refref_y, dtype=np.int32),
        "line_guided_columns": int(line_guided_columns),
        "fallback_columns": int(fallback_columns),
        "raw_line_count": int(len(det.get("raw_lines", []))),
        "skimage_raw_line_count_before_direction_filter": int(
            det.get("skimage_raw_line_count_before_direction_filter", len(det.get("raw_lines", [])))
        ),
        "direction_rejected_line_count": int(det.get("direction_rejected_line_count", 0)),
        "candidate_line_count": int(len(filtered.get("lines_for_filtering", []))),
        "used_line_count": int(len(filtered.get("lines_used", []))),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "timing_hough_detect_seconds": float(detect_seconds),
        "timing_filter_seconds": float(filter_seconds),
        "timing_build_bundle_seconds": float(scoring_payload_seconds),
        "timing_direction_total_seconds": float(time.perf_counter() - direction_started_at),
        "line_nls_filter_weighted_result": precomputed_weighted_result,
        **line_nls_filter_fields,
    }
    if filter_profile is not None:
        payload["filter_profile"] = dict(filter_profile)
    return payload


def compute_reference_self_payload_for_combination(
    *,
    doc: SweepDocument,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
    filter_profile: dict | None = None,
) -> dict:
    """Compute the exact reference-self payload for one Hough combination.

    This is the single implementation used both by the normal evaluator and by
    the cache warm-up stage.  Keeping it here prevents the warm path from having
    its own subtly different copy of the ref-to-ref Hough/filter/scoring-payload
    logic.
    """
    return _detect_filter_and_build_scoring_payload(
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
        filter_profile=filter_profile,
    )


def evaluate_single_combination(
    *,
    doc: SweepDocument,
    cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    min_surviving_line_nls: float | None = None,
    combination_bundle_logger=None,
    combination_bundle_records: list[dict] | None = None,
    combination_bundle_candidate_out: dict | None = None,
    profile_filters: bool = False,
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
        min_surviving_line_nls=min_surviving_line_nls,
        combination_bundle_logger=combination_bundle_logger,
        combination_bundle_records=combination_bundle_records,
        combination_bundle_candidate_out=combination_bundle_candidate_out,
        profile_filters=bool(profile_filters),
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
    min_surviving_line_nls: float | None = None,
    ref_to_ref_cache=None,
    combination_bundle_logger=None,
    combination_bundle_records: list[dict] | None = None,
    combination_bundle_candidate_out: dict | None = None,
    profile_filters: bool = False,
) -> dict:
    """Evaluate one combination from scalar values without allocating a config."""
    eval_started_at = time.perf_counter()
    ref_to_pred_filter_profile = {} if bool(profile_filters) else None
    ref_to_ref_filter_profile = {} if bool(profile_filters) else None

    ref_to_pred_payload = _detect_filter_and_build_scoring_payload(
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
        filter_profile=ref_to_pred_filter_profile,
        min_surviving_line_nls=min_surviving_line_nls,
        line_nls_filter_ref_blocks=doc.ref_blocks,
        line_nls_filter_other_blocks=doc.pred_blocks,
        levenshtein_backend=str(levenshtein_backend),
    )

    precomputed_weighted_result = ref_to_pred_payload.get("line_nls_filter_weighted_result")
    if precomputed_weighted_result is not None:
        weighted_result = precomputed_weighted_result
        levenshtein_seconds = float(ref_to_pred_payload.get("timing_line_nls_filter_seconds", 0.0) or 0.0)
    else:
        t_lev = time.perf_counter()
        weighted_result = compute_weighted_along_lines_similarity_from_compact_payload(
            ref_blocks=doc.ref_blocks,
            other_blocks=doc.pred_blocks,
            lines_used=ref_to_pred_payload["filtered"]["lines_used"],
            compact_payload=ref_to_pred_payload["scoring_payload"],
            levenshtein_backend=str(levenshtein_backend),
        )
        levenshtein_seconds = time.perf_counter() - t_lev

    if bool(ref_to_pred_payload.get("line_nls_filter_all_lines_removed", False)):
        ref_to_ref_payload = _empty_ref_to_ref_payload_for_line_nls_short_circuit()
        eval_row = _line_nls_removed_all_eval_row(
            weighted_result=weighted_result,
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
            levenshtein_seconds=float(levenshtein_seconds),
            eval_started_at=float(eval_started_at),
            include_filter_profile=bool(profile_filters),
        )
        if combination_bundle_candidate_out is not None:
            combination_bundle_candidate_out.clear()
            combination_bundle_candidate_out.update(
                {
                    "eval_row": eval_row,
                    "ref_to_pred_payload": ref_to_pred_payload,
                    "ref_to_ref_payload": ref_to_ref_payload,
                }
            )
        if combination_bundle_logger is not None and combination_bundle_records is not None:
            combination_record = combination_bundle_logger.build_combination_record(
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
            if combination_record is not None:
                combination_bundle_records.append(combination_record)
        return eval_row

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
            filter_profile=ref_to_ref_filter_profile,
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
            other_bundle=ref_to_pred_payload["scoring_payload"],
        )
    else:
        coverage_arrays = build_v212_line_coverage_arrays_from_bundles(
            refref_bundle=ref_to_ref_payload["scoring_payload"],
            other_bundle=ref_to_pred_payload["scoring_payload"],
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
                include_filter_profile=bool(profile_filters),
            )
            if combination_bundle_candidate_out is not None:
                combination_bundle_candidate_out.clear()
                combination_bundle_candidate_out.update(
                    {
                        "eval_row": eval_row,
                        "ref_to_pred_payload": ref_to_pred_payload,
                        "ref_to_ref_payload": ref_to_ref_payload,
                    }
                )
            if combination_bundle_logger is not None and combination_bundle_records is not None:
                # Visualization bundles are observational.  Build the record in
                # memory and let the threshold worker return it with the normal
                # metric payload so disk I/O never runs inside this hot loop.
                combination_record = combination_bundle_logger.build_combination_record(
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
                if combination_record is not None:
                    combination_bundle_records.append(combination_record)
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
    non_hallucination_weighted_tuning_score = compute_non_hallucination_weighted_tuning_score(
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
        "metric_outcome_reason": None,
        "tuning_score": float(tuning_score),
        "non_hallucination_weighted_tuning_score": float(non_hallucination_weighted_tuning_score),
        # This internal key is intentionally short because it is read often by
        # reducers; output serializers expose the longer report label.
        "weighted_along_lines_nls": None if weighted_along_lines_nls is None else float(weighted_along_lines_nls),
        "line_count": int(weighted_result.scored_line_count),
        "total_line_length": float(weighted_result.total_line_length),
        "correct_ref_coverage": float(normalized_coverage_metrics["correct_ref_coverage"]),
        "missing_ref_coverage": float(normalized_coverage_metrics["missing_ref_coverage"]),
        "repetition_on_ref": float(normalized_coverage_metrics["repetition_on_ref"]),
        "hallucination": float(normalized_coverage_metrics["hallucination"]),
        **_selection_metric_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            weighted_along_lines_nls=weighted_along_lines_nls,
            hallucination=normalized_coverage_metrics["hallucination"],
        ),
        **_line_count_fields(ref_to_pred_payload=ref_to_pred_payload, ref_to_ref_payload=ref_to_ref_payload),
        **_line_nls_filter_fields_from_payload(ref_to_pred_payload),
        **y_diff_diagnostics,
        **_timing_fields(
            ref_to_pred_payload=ref_to_pred_payload,
            ref_to_ref_payload=ref_to_ref_payload,
            coverage_seconds=float(coverage_seconds),
            levenshtein_seconds=float(levenshtein_seconds),
            eval_started_at=float(eval_started_at),
            include_filter_profile=bool(profile_filters),
        ),
    }
    if combination_bundle_candidate_out is not None:
        combination_bundle_candidate_out.clear()
        combination_bundle_candidate_out.update(
            {
                "eval_row": eval_row,
                "ref_to_pred_payload": ref_to_pred_payload,
                "ref_to_ref_payload": ref_to_ref_payload,
            }
        )
    if combination_bundle_logger is not None and combination_bundle_records is not None:
        # Store the visualization record in the threshold-local bucket.  The
        # completed document writer serializes these records after all threshold
        # workers finish, which keeps per-combination scoring focused on CPU work.
        combination_record = combination_bundle_logger.build_combination_record(
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
        if combination_record is not None:
            combination_bundle_records.append(combination_record)
    return eval_row


__all__ = [
    "DEFAULT_SELECTION_OBJECTIVE",
    "INVALID_REASON_COVERAGE_Y_DIFF_BELOW_MINUS_ONE",
    "SELECTION_OBJECTIVE_ALIGNMENT_EVIDENCE",
    "SELECTION_OBJECTIVE_NON_HALLUCINATION_WEIGHTED",
    "SELECTION_OBJECTIVE_STRICT_QUALITY",
    "SUPPORTED_SELECTION_OBJECTIVES",
    "is_finite_tuning_score",
    "is_finite_along_lines",
    "evaluation_rank_key",
    "normalize_selection_objective",
    "pick_better_eval",
    "compute_reference_self_payload_for_combination",
    "evaluate_single_combination",
    "evaluate_single_combination_values",
]
