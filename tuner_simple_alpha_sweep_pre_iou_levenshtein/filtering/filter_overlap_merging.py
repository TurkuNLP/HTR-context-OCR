from __future__ import annotations

"""True-IoU overlap graph and component merging for line filtering."""

import time

import numpy as np

from .filter_candidate_coverages import (
    build_candidate_coverages,
    coverage_from_path,
    local_path_key,
    prepare_candidate_lines,
)
from .filter_geometry_helpers import line_is_falling_diagonal_in_hough_angle_range, set_iou
from .filter_profile_fields import add_profile_seconds, set_profile_count


def segments_bounds_disjoint(min_a: int, max_a: int, min_b: int, max_b: int) -> bool:
    """Return whether two inclusive integer ranges cannot overlap."""
    return int(max_a) < int(min_b) or int(max_b) < int(min_a)


def segments_bounds_overlap(min_a: int, max_a: int, min_b: int, max_b: int) -> bool:
    """Return whether two inclusive integer ranges overlap at least once."""
    return not segments_bounds_disjoint(min_a, max_a, min_b, max_b)


def coverages_have_enough_true_iou(coverage_a: dict, coverage_b: dict, *, min_iou_threshold: float) -> bool:
    """Return whether two coverages pass the existing prediction/reference true-IoU threshold."""
    if segments_bounds_disjoint(
        int(coverage_a.get("pred_min", 0)),
        int(coverage_a.get("pred_max", -1)),
        int(coverage_b.get("pred_min", 0)),
        int(coverage_b.get("pred_max", -1)),
    ):
        return False

    if segments_bounds_disjoint(
        int(coverage_a.get("ref_min", 0)),
        int(coverage_a.get("ref_max", -1)),
        int(coverage_b.get("ref_min", 0)),
        int(coverage_b.get("ref_max", -1)),
    ):
        return False

    prediction_iou = set_iou(coverage_a["pred_segments"], coverage_b["pred_segments"])
    reference_iou = set_iou(coverage_a["ref_segments"], coverage_b["ref_segments"])
    return bool(min(prediction_iou, reference_iou) > float(min_iou_threshold))


def merged_pair_keeps_hough_angle_range(coverage_a: dict, coverage_b: dict, matrix: np.ndarray) -> bool:
    """Return whether the fitted two-coverage merge still follows the Hough falling-angle range."""
    merged_pair_coverage = merge_overlap_component([coverage_a, coverage_b], matrix)
    return line_is_falling_diagonal_in_hough_angle_range(merged_pair_coverage.get("line", {}))


def coverages_merge_candidate(coverage_a: dict, coverage_b: dict, *, matrix: np.ndarray, min_iou_threshold: float) -> bool:
    """Return whether true-IoU may merge two coverages without breaking the Hough angle rule."""
    if not coverages_have_enough_true_iou(coverage_a, coverage_b, min_iou_threshold=min_iou_threshold):
        return False
    return merged_pair_keeps_hough_angle_range(coverage_a, coverage_b, matrix)


def merge_overlap_component(component_coverages: list[dict], matrix: np.ndarray) -> dict:
    """Merge one connected overlap component into one coverage object.

    Every prediction column in the component keeps the locally strongest sample.
    This is why a final surviving line may be assembled from several raw Hough
    segments instead of matching one raw segment exactly.
    """
    if len(component_coverages) == 1:
        return dict(component_coverages[0])

    fallback_coverage = max(
        component_coverages,
        key=lambda coverage: (
            float(coverage.get("total_score", 0.0)),
            float(coverage.get("mean_score", 0.0)),
            int(len(coverage.get("pred_segments", ()))),
            float(coverage["line"].get("support", 0.0)),
            float(coverage["line"].get("length", 0.0)),
        ),
    )

    best_path_sample_by_column: dict[int, tuple[tuple, int, float]] = {}

    for coverage in component_coverages:
        for prediction_column, sampled_reference_row in coverage["x_to_y"].items():
            local_winner_key = local_path_key(coverage, int(prediction_column))
            current_best_sample = best_path_sample_by_column.get(int(prediction_column))
            if current_best_sample is None or local_winner_key > current_best_sample[0]:
                best_path_sample_by_column[int(prediction_column)] = (
                    local_winner_key,
                    int(sampled_reference_row),
                    float(coverage["x_to_score"][int(prediction_column)]),
                )

    merged_x_to_y = {
        int(prediction_column): int(best_path_sample_by_column[int(prediction_column)][1])
        for prediction_column in sorted(best_path_sample_by_column)
    }
    merged_x_to_score = {
        int(prediction_column): float(best_path_sample_by_column[int(prediction_column)][2])
        for prediction_column in sorted(best_path_sample_by_column)
    }

    merged_source_raw_line_ids = sorted(
        {
            int(raw_line_id)
            for coverage in component_coverages
            for raw_line_id in coverage.get("source_raw_line_ids", [])
            if int(raw_line_id) >= 0
        }
    )

    return coverage_from_path(
        x_to_y=merged_x_to_y,
        x_to_score=merged_x_to_score,
        matrix=matrix,
        fallback_line=fallback_coverage["line"],
        source_raw_line_ids=merged_source_raw_line_ids,
    )


def components_from_adjacency(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Return connected components from an adjacency graph in stable index order."""
    if not adjacency:
        return []

    connected_components: list[list[int]] = []
    visited_indices: set[int] = set()

    for start_index in sorted(adjacency):
        if start_index in visited_indices:
            continue

        stack = [int(start_index)]
        visited_indices.add(int(start_index))
        component_indices: list[int] = []

        while stack:
            current_index = stack.pop()
            component_indices.append(int(current_index))
            for adjacent_index in sorted(adjacency[current_index]):
                if adjacent_index in visited_indices:
                    continue
                visited_indices.add(int(adjacent_index))
                stack.append(int(adjacent_index))

        connected_components.append(sorted(component_indices))

    return connected_components


def iter_possible_overlap_pairs_for_production(coverages: list[dict]):
    """Yield only coverage pairs whose prediction and reference bounds overlap."""
    sorted_coverage_indices = sorted(
        range(len(coverages)),
        key=lambda coverage_index: (
            int(coverages[coverage_index].get("pred_min", 0)),
            int(coverages[coverage_index].get("pred_max", -1)),
            int(coverages[coverage_index].get("ref_min", 0)),
            int(coverages[coverage_index].get("ref_max", -1)),
            int(coverage_index),
        ),
    )

    active_prediction_overlap_indices: list[int] = []

    for current_index in sorted_coverage_indices:
        current_coverage = coverages[int(current_index)]
        current_prediction_min = int(current_coverage.get("pred_min", 0))
        current_reference_min = int(current_coverage.get("ref_min", 0))
        current_reference_max = int(current_coverage.get("ref_max", -1))

        still_active_indices: list[int] = []
        for active_index in active_prediction_overlap_indices:
            active_coverage = coverages[int(active_index)]
            active_prediction_max = int(active_coverage.get("pred_max", -1))

            if active_prediction_max < current_prediction_min:
                continue

            still_active_indices.append(int(active_index))

            if not segments_bounds_overlap(
                int(active_coverage.get("ref_min", 0)),
                int(active_coverage.get("ref_max", -1)),
                current_reference_min,
                current_reference_max,
            ):
                continue

            left_index = min(int(active_index), int(current_index))
            right_index = max(int(active_index), int(current_index))
            yield left_index, right_index

        still_active_indices.append(int(current_index))
        active_prediction_overlap_indices = still_active_indices


def coverage_components_for_production(
    coverages: list[dict],
    matrix: np.ndarray,
    *,
    min_iou_threshold: float,
    profile: dict | None = None,
) -> list[list[int]]:
    """Build connected overlap components for the production filtering path."""
    if not coverages:
        set_profile_count(profile, "filter_possible_overlap_pair_count", 0)
        set_profile_count(profile, "filter_merge_edge_count", 0)
        set_profile_count(profile, "filter_component_count", 0)
        return []

    adjacency: dict[int, set[int]] = {coverage_index: set() for coverage_index in range(len(coverages))}
    possible_overlap_pair_count = 0
    merge_edge_count = 0
    angle_rejected_merge_edge_count = 0
    exact_iou_seconds = 0.0

    pair_loop_started_at = time.perf_counter() if profile is not None else 0.0
    for left_index, right_index in iter_possible_overlap_pairs_for_production(coverages):
        possible_overlap_pair_count += 1
        exact_iou_started_at = time.perf_counter() if profile is not None else 0.0
        left_coverage = coverages[int(left_index)]
        right_coverage = coverages[int(right_index)]
        should_merge_by_iou = coverages_have_enough_true_iou(
            left_coverage,
            right_coverage,
            min_iou_threshold=min_iou_threshold,
        )
        should_merge = False
        if should_merge_by_iou:
            should_merge = merged_pair_keeps_hough_angle_range(left_coverage, right_coverage, matrix)
            if not should_merge:
                angle_rejected_merge_edge_count += 1
        if profile is not None:
            exact_iou_seconds += float(time.perf_counter() - exact_iou_started_at)
        if not should_merge:
            continue
        adjacency[int(left_index)].add(int(right_index))
        adjacency[int(right_index)].add(int(left_index))
        merge_edge_count += 1

    if profile is not None:
        pair_loop_seconds = float(time.perf_counter() - pair_loop_started_at)
        add_profile_seconds(
            profile,
            "filter_possible_pair_generation_seconds",
            max(0.0, pair_loop_seconds - exact_iou_seconds),
        )
        add_profile_seconds(profile, "filter_exact_iou_seconds", exact_iou_seconds)
        set_profile_count(profile, "filter_possible_overlap_pair_count", possible_overlap_pair_count)
        set_profile_count(profile, "filter_merge_edge_count", merge_edge_count)
        set_profile_count(profile, "filter_angle_rejected_merge_edge_count", angle_rejected_merge_edge_count)

    component_started_at = time.perf_counter() if profile is not None else 0.0
    components = components_from_adjacency(adjacency)
    if profile is not None:
        add_profile_seconds(
            profile,
            "filter_component_build_seconds",
            float(time.perf_counter() - component_started_at),
        )
        set_profile_count(profile, "filter_component_count", len(components))
    return components


def run_production_filtering_pipeline(
    lines: list[dict],
    matrix: np.ndarray,
    *,
    min_iou_threshold: float,
    profile: dict | None = None,
) -> list[dict]:
    """Run candidate preparation, coverage building, overlap graph, and merge."""
    prepare_started_at = time.perf_counter() if profile is not None else 0.0
    candidate_lines = prepare_candidate_lines(lines, matrix, profile=profile)
    if profile is not None:
        add_profile_seconds(
            profile,
            "filter_prepare_candidates_seconds",
            float(time.perf_counter() - prepare_started_at),
        )

    coverage_started_at = time.perf_counter() if profile is not None else 0.0
    candidate_coverages = build_candidate_coverages(candidate_lines, matrix)
    if profile is not None:
        add_profile_seconds(
            profile,
            "filter_build_candidate_coverages_seconds",
            float(time.perf_counter() - coverage_started_at),
        )
        set_profile_count(profile, "filter_candidate_coverage_count", len(candidate_coverages))

    component_indices = coverage_components_for_production(
        candidate_coverages,
        matrix,
        min_iou_threshold=min_iou_threshold,
        profile=profile,
    )

    merge_started_at = time.perf_counter() if profile is not None else 0.0
    merged_coverages: list[dict] = []
    angle_rejected_merged_component_count = 0
    for component in component_indices:
        component_coverages = [candidate_coverages[coverage_index] for coverage_index in component]
        merged_coverage = merge_overlap_component(component_coverages, matrix)
        if line_is_falling_diagonal_in_hough_angle_range(merged_coverage.get("line", {})):
            merged_coverages.append(merged_coverage)
            continue

        angle_rejected_merged_component_count += 1
        merged_coverages.extend(
            coverage
            for coverage in component_coverages
            if line_is_falling_diagonal_in_hough_angle_range(coverage.get("line", {}))
        )
    if profile is not None:
        add_profile_seconds(
            profile,
            "filter_merge_components_seconds",
            float(time.perf_counter() - merge_started_at),
        )
        set_profile_count(profile, "filter_angle_rejected_merged_component_count", angle_rejected_merged_component_count)
        set_profile_count(profile, "filter_merged_coverage_count", len(merged_coverages))
    return merged_coverages


__all__ = [
    "components_from_adjacency",
    "coverage_components_for_production",
    "coverages_have_enough_true_iou",
    "coverages_merge_candidate",
    "iter_possible_overlap_pairs_for_production",
    "merge_overlap_component",
    "merged_pair_keeps_hough_angle_range",
    "run_production_filtering_pipeline",
    "segments_bounds_disjoint",
    "segments_bounds_overlap",
]
