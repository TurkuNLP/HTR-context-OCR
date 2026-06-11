from __future__ import annotations

"""Public entry point for the tuner's production true-IoU line filter.

Historically this file contained the whole filtering implementation.  It is now
a small compatibility wrapper around focused modules in the same directory:

- ``filter_candidate_coverages.py`` prepares raw Hough lines and samples paths.
- ``filter_overlap_merging.py`` builds the true-IoU overlap graph and merges it.
- ``filter_final_assignment.py`` assigns each prediction column to one final line.
- ``filter_geometry_helpers.py`` owns line support, path expansion, and fitting.
- ``filter_profile_fields.py`` owns optional scalar profiling fields.

Keep this file as the stable import path for the rest of the tuner.  New helper
logic should go into the focused modules above instead of growing this wrapper.
"""

import time

import numpy as np

from .filter_candidate_coverages import (
    build_candidate_coverages as _build_candidate_coverages,
    build_line_coverage as _build_line_coverage,
    coverage_from_path as _coverage_from_path,
    coverage_from_sampled_path as _coverage_from_sampled_path,
    local_path_key as _local_path_key,
    normalize_candidate_line as _normalize_candidate_line,
    prepare_candidate_lines as _prepare_candidates,
)
from .filter_final_assignment import (
    build_coverage_indices_by_prediction_column as _build_coverage_indices_by_prediction_column,
    compute_final_assignment as _compute_final_assignment,
    empty_assignment as _empty_assignment,
    finalize_outputs as _finalize_outputs,
    used_coverage_indices_from_assignment as _used_coverage_indices_from_assignment,
)
from .filter_geometry_helpers import (
    fit_line_from_path as _fit_line_from_path,
    line_x_bounds as _line_x_bounds,
    mean_line_support_for_filter as _mean_line_support_for_filter,
    ref_segments_from_path as _ref_segments_from_path,
    set_iou as _set_iou,
    weighted_degree_one_fit as _weighted_degree_one_fit,
)
from .filter_overlap_merging import (
    components_from_adjacency as _components_from_adjacency,
    coverage_components_for_production as _coverage_components_for_production,
    coverages_merge_candidate as _coverages_merge_candidate,
    iter_possible_overlap_pairs_for_production as _iter_possible_overlap_pairs_for_production,
    merge_overlap_component as _merge_component,
    run_production_filtering_pipeline as _run_production_filtering_pipeline,
    segments_bounds_disjoint as _segments_bounds_disjoint,
    segments_bounds_overlap as _segments_bounds_overlap,
)
from .filter_profile_fields import (
    FILTER_PROFILE_DEFAULTS,
    add_profile_seconds as _add_profile_seconds,
    ensure_profile_defaults as _ensure_profile_defaults,
    set_profile_count as _set_profile_count,
)


DEFAULT_MIN_IOU_THRESHOLD = 0.035

__all__ = [
    "DEFAULT_MIN_IOU_THRESHOLD",
    "FILTER_PROFILE_DEFAULTS",
    "filter_lines_for_alignment_by_ownership",
]


def filter_lines_for_alignment_by_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    profile: dict | None = None,
    **_ignored,
):
    """Filter raw Hough lines and return final lines plus column ownership.

    Args:
        lines: Raw Hough line dictionaries already expressed in matrix
            coordinates.  The caller normally gets these from
            ``alignment/line_alignment_pipeline_fast.py``.
        matrix: Score matrix whose rows are reference windows and columns are
            prediction windows.
        mask_bool: Boolean active-cell mask from the same Hough context as the
            matrix.  It is used only for final diagnostic fields on surviving
            lines; it must have the same shape as ``matrix``.
        min_iou_threshold: Minimum overlap threshold used when deciding whether
            two candidate coverages should be merged.
        profile: Optional dictionary filled with scalar timing/count fields.

    Returns:
        ``(final_lines, assignment)`` where ``assignment["mapped_y"]`` gives the
        selected reference row per prediction column and
        ``assignment["mapped_line_id"]`` gives the final line owner per
        prediction column.  Unowned columns use ``NaN`` and ``-1``.
    """
    _ensure_profile_defaults(profile)
    filter_started_at = time.perf_counter() if profile is not None else 0.0
    _set_profile_count(profile, "filter_input_line_count", len(lines))

    if not lines or matrix.size == 0:
        n_prediction_columns = matrix.shape[1] if matrix.ndim == 2 else 0
        _set_profile_count(profile, "filter_final_line_count", 0)
        if profile is not None:
            profile["filter_total_profiled_seconds"] = float(time.perf_counter() - filter_started_at)
        return [], _empty_assignment(n_prediction_columns)

    if mask_bool.shape != matrix.shape:
        raise ValueError(f"mask_bool shape {mask_bool.shape} does not match matrix shape {matrix.shape}")

    merged_coverages = _run_production_filtering_pipeline(
        lines,
        matrix,
        min_iou_threshold=float(min_iou_threshold),
        profile=profile,
    )

    finalize_started_at = time.perf_counter() if profile is not None else 0.0
    final_lines, assignment = _finalize_outputs(
        list(merged_coverages),
        matrix,
        mask_bool,
        profile=profile,
    )
    if profile is not None:
        _add_profile_seconds(
            profile,
            "filter_finalize_outputs_seconds",
            float(time.perf_counter() - finalize_started_at),
        )
        profile["filter_total_profiled_seconds"] = float(time.perf_counter() - filter_started_at)
    return final_lines, assignment
