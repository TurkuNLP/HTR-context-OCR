"""Detailed debug analysis helpers for the v2.12 true-IoU line filter.

This module is intentionally imported only when a caller explicitly wants the
full pairwise IoU analysis. The production report pipeline does not import this
module during normal runs.
"""

from __future__ import annotations

import numpy as np

# Import the shared production helpers so the debug analysis reuses the exact
# same candidate preparation, coverage construction, merge math, and component
# construction logic instead of duplicating any filtering behavior.
from line_filtering_v2_12_IoU import (
    DEFAULT_ABS_MIN_LEN,
    DEFAULT_MIN_IOU_THRESHOLD,
    _build_candidate_coverages,
    _components_from_adjacency,
    _merge_component,
    _prepare_candidates,
    _segments_bounds_disjoint,
    _set_iou,
)

__all__ = [
    "analyze_line_filtering",
]


# Build the full x/y IoU payload for one pair of coverage objects.
def _coverage_iou_stats(cov_a: dict, cov_b: dict, *, min_iou_threshold: float) -> dict:
    """Return the detailed pairwise IoU diagnostics for one coverage pair."""
    # Read the sampled prediction-column coverage for the first coverage.
    prediction_segments_a = cov_a["pred_segments"]
    # Read the sampled prediction-column coverage for the second coverage.
    prediction_segments_b = cov_b["pred_segments"]
    # Read the expanded reference-row coverage for the first coverage.
    reference_segments_a = cov_a["ref_segments"]
    # Read the expanded reference-row coverage for the second coverage.
    reference_segments_b = cov_b["ref_segments"]

    # Check whether the prediction-axis bounds are already disjoint.
    prediction_disjoint = _segments_bounds_disjoint(
        int(cov_a.get("pred_min", 0)),
        int(cov_a.get("pred_max", -1)),
        int(cov_b.get("pred_min", 0)),
        int(cov_b.get("pred_max", -1)),
    )
    # Check whether the reference-axis bounds are already disjoint.
    reference_disjoint = _segments_bounds_disjoint(
        int(cov_a.get("ref_min", 0)),
        int(cov_a.get("ref_max", -1)),
        int(cov_b.get("ref_min", 0)),
        int(cov_b.get("ref_max", -1)),
    )

    # Handle the cheapest case where both axes are disjoint.
    if prediction_disjoint and reference_disjoint:
        # No prediction overlap is possible in this case.
        x_iou = 0.0
        # No reference overlap is possible in this case.
        y_iou = 0.0
        # There are no shared prediction segments.
        shared_prediction_segments: set[int] = set()
        # There are no shared reference segments.
        shared_reference_segments: set[int] = set()
        # The prediction union count is just the sum of both sizes.
        union_prediction_count = int(len(prediction_segments_a) + len(prediction_segments_b))
        # The reference union count is just the sum of both sizes.
        union_reference_count = int(len(reference_segments_a) + len(reference_segments_b))
    # Handle the case where prediction bounds are disjoint but reference may overlap.
    elif prediction_disjoint:
        # Prediction IoU is forced to zero when prediction bounds do not overlap.
        x_iou = 0.0
        # No shared prediction segments can exist here.
        shared_prediction_segments = set()
        # The prediction union count is just the sum of both sizes.
        union_prediction_count = int(len(prediction_segments_a) + len(prediction_segments_b))
        # Compute the shared reference rows exactly.
        shared_reference_segments = reference_segments_a & reference_segments_b
        # Compute the exact reference union set.
        union_reference_segments = reference_segments_a | reference_segments_b
        # Count the exact reference-union size.
        union_reference_count = int(len(union_reference_segments))
        # Compute the exact reference IoU.
        y_iou = _set_iou(reference_segments_a, reference_segments_b)
    # Handle the symmetric case where reference bounds are disjoint but prediction may overlap.
    elif reference_disjoint:
        # Reference IoU is forced to zero when reference bounds do not overlap.
        y_iou = 0.0
        # No shared reference segments can exist here.
        shared_reference_segments = set()
        # The reference union count is just the sum of both sizes.
        union_reference_count = int(len(reference_segments_a) + len(reference_segments_b))
        # Compute the shared prediction segments exactly.
        shared_prediction_segments = prediction_segments_a & prediction_segments_b
        # Compute the exact prediction union set.
        union_prediction_segments = prediction_segments_a | prediction_segments_b
        # Count the exact prediction-union size.
        union_prediction_count = int(len(union_prediction_segments))
        # Compute the exact prediction IoU.
        x_iou = _set_iou(prediction_segments_a, prediction_segments_b)
    else:
        # Compute the exact shared prediction segments.
        shared_prediction_segments = prediction_segments_a & prediction_segments_b
        # Compute the exact shared reference segments.
        shared_reference_segments = reference_segments_a & reference_segments_b
        # Compute the exact prediction union set.
        union_prediction_segments = prediction_segments_a | prediction_segments_b
        # Compute the exact reference union set.
        union_reference_segments = reference_segments_a | reference_segments_b
        # Count the exact prediction-union size.
        union_prediction_count = int(len(union_prediction_segments))
        # Count the exact reference-union size.
        union_reference_count = int(len(union_reference_segments))
        # Compute the exact prediction IoU.
        x_iou = _set_iou(prediction_segments_a, prediction_segments_b)
        # Compute the exact reference IoU.
        y_iou = _set_iou(reference_segments_a, reference_segments_b)

    # The merge rule uses the smaller IoU across the two axes.
    min_iou_value = float(min(x_iou, y_iou))

    # Return the full debug payload exactly as the earlier implementation did.
    return {
        "raw_line_ids_a": sorted(int(raw_line_id) for raw_line_id in cov_a.get("source_raw_line_ids", [])),
        "raw_line_ids_b": sorted(int(raw_line_id) for raw_line_id in cov_b.get("source_raw_line_ids", [])),
        "shared_pred_count": int(len(shared_prediction_segments)),
        "union_pred_count": int(union_prediction_count),
        "shared_ref_count": int(len(shared_reference_segments)),
        "union_ref_count": int(union_reference_count),
        "shared_pred_segments": [int(prediction_segment) for prediction_segment in sorted(shared_prediction_segments)],
        "shared_ref_segments": [int(reference_segment) for reference_segment in sorted(shared_reference_segments)],
        "x_iou": float(x_iou),
        "y_iou": float(y_iou),
        "min_iou": min_iou_value,
        "min_iou_threshold": float(min_iou_threshold),
        "merge_candidate": bool(min_iou_value > float(min_iou_threshold)),
    }


# Decide whether two coverages overlap strongly enough under the true-IoU rule.
def _coverages_overlap(cov_a: dict, cov_b: dict, *, min_iou_threshold: float) -> tuple[bool, dict]:
    """Return the exact merge decision together with the full debug payload."""
    # Build the detailed pairwise statistics for this exact coverage pair.
    pairwise_stats = _coverage_iou_stats(cov_a, cov_b, min_iou_threshold=min_iou_threshold)
    # Reuse the exact production merge decision from the debug payload.
    return bool(pairwise_stats["merge_candidate"]), pairwise_stats


# Build connected components while also storing every pairwise IoU payload.
def _coverage_components(
    coverages: list[dict],
    *,
    min_iou_threshold: float,
) -> tuple[list[list[int]], list[dict]]:
    """Build the full all-pairs overlap graph and pairwise IoU diagnostics."""
    # Short-circuit when there are no coverage objects to compare.
    if not coverages:
        return [], []

    # Start one empty adjacency set for every coverage index.
    adjacency: dict[int, set[int]] = {coverage_index: set() for coverage_index in range(len(coverages))}
    # Collect the full IoU payload for every tested pair.
    pairwise_iou_stats: list[dict] = []

    # Intentionally keep the complete all-pairs traversal for debug analysis.
    for left_index in range(len(coverages)):
        # Compare the current coverage against every later coverage once.
        for right_index in range(left_index + 1, len(coverages)):
            # Compute the exact overlap decision and the full debug payload.
            overlaps, pairwise_stats = _coverages_overlap(
                coverages[int(left_index)],
                coverages[int(right_index)],
                min_iou_threshold=min_iou_threshold,
            )
            # Record which coverage indices were compared.
            pairwise_stats["coverage_index_a"] = int(left_index)
            # Record the second coverage index as well.
            pairwise_stats["coverage_index_b"] = int(right_index)
            # Store the debug payload even when the pair does not merge.
            pairwise_iou_stats.append(pairwise_stats)
            # Skip adjacency updates when the exact merge rule says the pair does not merge.
            if not overlaps:
                continue
            # Add the merge edge in the left-to-right direction.
            adjacency[int(left_index)].add(int(right_index))
            # Add the symmetric merge edge in the right-to-left direction.
            adjacency[int(right_index)].add(int(left_index))

    # Return the deterministic connected components together with the full debug payload list.
    return _components_from_adjacency(adjacency), pairwise_iou_stats


# Convert one coverage object into a compact debug summary for reports or notebooks.
def _coverage_debug_summary(cov: dict, *, coverage_index: int | None = None) -> dict:
    """Build the compact summary stored in the top-level debug analysis result."""
    # Read the representative line geometry from the coverage object.
    line = cov["line"]
    # Build the compact coverage summary that is easier to inspect than the full object.
    summary = {
        "source_raw_line_ids": sorted(int(raw_line_id) for raw_line_id in cov.get("source_raw_line_ids", [])),
        "pred_segment_count": int(len(cov.get("pred_segments", ()))),
        "ref_segment_count": int(len(cov.get("ref_segments", ()))),
        "total_score": float(cov.get("total_score", 0.0)),
        "mean_score": float(cov.get("mean_score", 0.0)),
        "anchor_y": float(cov.get("anchor_y", 0.0)),
        "x0": float(line.get("x0", 0.0)),
        "y0": float(line.get("y0", 0.0)),
        "x1": float(line.get("x1", 0.0)),
        "y1": float(line.get("y1", 0.0)),
        "length": float(line.get("length", 0.0)),
        "support": float(line.get("support", 0.0)),
        "score": float(line.get("score", 0.0)),
    }
    # Include the coverage index when the caller asked for it.
    if coverage_index is not None:
        summary["coverage_index"] = int(coverage_index)
    # Return the compact summary object.
    return summary


# Public debug entrypoint for inspecting the full true-IoU filtering state.
def analyze_line_filtering(
    lines: list[dict],
    matrix: np.ndarray,
    *,
    abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
) -> dict:
    """Run the full debug analysis path, including all pairwise IoU payloads."""
    # Reuse the exact production candidate preparation rules.
    candidate_lines = _prepare_candidates(lines, matrix, abs_min_len=abs_min_len)
    # Reuse the exact production coverage construction logic.
    candidate_coverages = _build_candidate_coverages(candidate_lines, matrix)
    # Build the full all-pairs overlap graph and debug payloads.
    component_indices, pairwise_iou_stats = _coverage_components(
        candidate_coverages,
        min_iou_threshold=min_iou_threshold,
    )
    # Reuse the exact production component merge behavior for every component.
    merged_coverages = [
        _merge_component([candidate_coverages[coverage_index] for coverage_index in component], matrix)
        for component in component_indices
    ]

    # Return the structured top-level debug bundle.
    return {
        "candidate_lines": [dict(candidate_line) for candidate_line in candidate_lines],
        "candidate_coverages": [
            _coverage_debug_summary(coverage_object, coverage_index=coverage_index)
            for coverage_index, coverage_object in enumerate(candidate_coverages)
        ],
        "pairwise_iou": pairwise_iou_stats,
        "components": [
            {
                "component_index": int(component_index),
                "coverage_indices": [int(coverage_index) for coverage_index in component],
                "source_raw_line_ids": sorted(
                    {
                        int(raw_line_id)
                        for coverage_index in component
                        for raw_line_id in candidate_coverages[coverage_index].get("source_raw_line_ids", [])
                        if int(raw_line_id) >= 0
                    }
                ),
            }
            for component_index, component in enumerate(component_indices)
        ],
        "merged_coverages": [
            _coverage_debug_summary(coverage_object, coverage_index=coverage_index)
            for coverage_index, coverage_object in enumerate(merged_coverages)
        ],
        "merged_coverage_objects": merged_coverages,
    }
