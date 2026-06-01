from __future__ import annotations

"""Small profiling helpers for the line-filtering hot path.

The filter can optionally receive a plain dictionary named ``profile``.  When it
is provided, each stage records scalar timing and count values into that
dictionary.  When it is ``None``, these helpers become no-ops so the normal
production hot loop does not pay for profiling bookkeeping.
"""


FILTER_PROFILE_DEFAULTS = {
    "filter_prepare_candidates_seconds": 0.0,
    "filter_build_candidate_coverages_seconds": 0.0,
    "filter_possible_pair_generation_seconds": 0.0,
    "filter_exact_iou_seconds": 0.0,
    "filter_component_build_seconds": 0.0,
    "filter_merge_components_seconds": 0.0,
    "filter_final_assignment_seconds": 0.0,
    "filter_finalize_outputs_seconds": 0.0,
    "filter_total_profiled_seconds": 0.0,
    "filter_input_line_count": 0,
    "filter_prepared_candidate_count": 0,
    "filter_candidate_coverage_count": 0,
    "filter_possible_overlap_pair_count": 0,
    "filter_merge_edge_count": 0,
    "filter_component_count": 0,
    "filter_merged_coverage_count": 0,
    "filter_finalize_prune_iteration_count": 0,
    "filter_final_line_count": 0,
    "filter_fallback_candidate_used": 0,
}


def ensure_profile_defaults(profile: dict | None) -> None:
    """Populate every expected profiling key when profiling is enabled."""
    if profile is None:
        return
    for field_name, default_value in FILTER_PROFILE_DEFAULTS.items():
        profile.setdefault(field_name, default_value)


def add_profile_seconds(profile: dict | None, field_name: str, seconds: float) -> None:
    """Accumulate one timing value into the optional profile dictionary."""
    if profile is None:
        return
    profile[field_name] = float(profile.get(field_name, 0.0) or 0.0) + float(seconds)


def set_profile_count(profile: dict | None, field_name: str, value: int) -> None:
    """Set one integer counter in the optional profile dictionary."""
    if profile is None:
        return
    profile[field_name] = int(value)


__all__ = [
    "FILTER_PROFILE_DEFAULTS",
    "add_profile_seconds",
    "ensure_profile_defaults",
    "set_profile_count",
]
