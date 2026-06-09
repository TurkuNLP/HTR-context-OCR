"""Greedy diagonal Hough-segment merging used by the default v2.12 pipeline.

This module contains the exact Python reference implementation of the historical
``merging_diag()`` behavior. The default production path still depends on this
merge heuristic, so the code stays fully behavior-compatible. Keeping it here
instead of inside the raw Hough detector makes three things much clearer:

1. raw probabilistic Hough detection,
2. post-Hough greedy diagonal segment merging,
3. the later handoff into the true-IoU filter.

That separation is also the cleanest foundation for a future exact-result
accelerator and for a later version where the merge stage is removed entirely.
"""

from __future__ import annotations

import math

import numpy as np

from accelerators.greedy_diagonal_segment_merging_backend import (
    merge_diagonal_segments_with_optional_accelerator,
)

__all__ = ["merge_diagonal_segments"]


# Measure Euclidean distance between two 2D points.
def _line_magnitude(x1: float, y1: float, x2: float, y2: float) -> float:
    """Return Euclidean length between two points."""
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# Measure the distance from a point to a finite line segment.
def _point_line_distance(
    entry: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
) -> float:
    """Measure point-to-segment distance using the preserved legacy geometry."""
    line_point_0, line_point_1, query_point = entry
    query_x, query_y = query_point
    line_x0, line_y0 = line_point_0
    line_x1, line_y1 = line_point_1

    segment_length = _line_magnitude(line_x0, line_y0, line_x1, line_y1)
    if segment_length < 1e-8:
        return 9999.0

    projection_numerator = (
        ((query_x - line_x0) * (line_x1 - line_x0))
        + ((query_y - line_y0) * (line_y1 - line_y0))
    )
    projection_position = projection_numerator / (segment_length * segment_length)

    if (projection_position < 0.00001) or (projection_position > 1):
        distance_to_first_endpoint = _line_magnitude(query_x, query_y, line_x0, line_y0)
        distance_to_second_endpoint = _line_magnitude(query_x, query_y, line_x1, line_y1)
        return (
            distance_to_second_endpoint
            if distance_to_first_endpoint > distance_to_second_endpoint
            else distance_to_first_endpoint
        )

    projected_x = line_x0 + projection_position * (line_x1 - line_x0)
    projected_y = line_y0 + projection_position * (line_y1 - line_y0)
    return _line_magnitude(query_x, query_y, projected_x, projected_y)


# Count how many active thresholded-mask points lie near candidate segments.
def _count_points_in_range(
    segments: list[tuple[tuple[float, float], tuple[float, float]]],
    points: list[tuple[int, int]],
    max_distance: float,
) -> int:
    """Count how many global mask points lie within ``max_distance`` of segments."""
    count = 0
    for segment in segments:
        point_0, point_1 = segment
        for point in points:
            if _point_line_distance((point_0, point_1, point)) <= max_distance:
                count += 1
    return count


# Sample one segment onto matrix cells so bridge support can be measured.
def _sample_line_pixels(
    point_0: tuple[float, float],
    point_1: tuple[float, float],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Sample pixel coordinates along a line segment inside matrix bounds."""
    (x0, y0), (x1, y1) = point_0, point_1
    sample_count = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    sample_count = max(sample_count, 2)
    sampled_x = np.clip(np.rint(np.linspace(x0, x1, sample_count)).astype(int), 0, shape[1] - 1)
    sampled_y = np.clip(np.rint(np.linspace(y0, y1, sample_count)).astype(int), 0, shape[0] - 1)
    return sampled_x, sampled_y


# Find the longest gap of inactive bridge cells.
def _longest_false_run(values: np.ndarray) -> int:
    """Return the longest contiguous run of ``False`` values in a boolean array."""
    best_run = 0
    current_run = 0
    for value in values:
        if not value:
            current_run += 1
            best_run = max(best_run, current_run)
        else:
            current_run = 0
    return best_run


# Measure Euclidean length of one Hough segment.
def _segment_length(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """Return Euclidean length of one Hough segment."""
    (x0, y0), (x1, y1) = segment
    return float(math.hypot(x1 - x0, y1 - y0))


# Measure orientation of one Hough segment.
def _segment_angle(segment: tuple[tuple[float, float], tuple[float, float]]) -> float:
    """Return segment direction angle in degrees mapped into ``[0, 180)``."""
    (x0, y0), (x1, y1) = segment
    segment_degrees = math.degrees(math.atan2(y1 - y0, x1 - x0))
    return float((segment_degrees + 180.0) % 180.0)


# Find the closest endpoint pair between two Hough segments.
def _nearest_endpoints(
    segment_a: tuple[tuple[float, float], tuple[float, float]],
    segment_b: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Find the closest endpoint pair between two line segments."""
    segment_a_points = [segment_a[0], segment_a[1]]
    segment_b_points = [segment_b[0], segment_b[1]]
    best_a_point, best_b_point = segment_a_points[0], segment_b_points[0]
    best_distance = float("inf")
    for point_a in segment_a_points:
        for point_b in segment_b_points:
            distance = math.hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])
            if distance < best_distance:
                best_distance = distance
                best_a_point, best_b_point = point_a, point_b
    return best_a_point, best_b_point, best_distance


# Measure support and longest gap along the bridge between two endpoints.
def _bridge_stats(
    point_0: tuple[float, float],
    point_1: tuple[float, float],
    active_mask: np.ndarray,
) -> tuple[float, float]:
    """Measure support and longest gap along the bridge between two endpoints."""
    sampled_x, sampled_y = _sample_line_pixels(point_0, point_1, active_mask.shape)
    sampled_values = active_mask[sampled_y, sampled_x]
    if sampled_values.size == 0:
        return 0.0, 1.0
    bridge_support = float(sampled_values.mean())
    normalized_bridge_gap = float(_longest_false_run(sampled_values) / len(sampled_values))
    return bridge_support, normalized_bridge_gap


# Python reference implementation of the exact historical merge heuristic.
def _merge_diagonal_segments_python_reference(
    raw_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    active_mask: np.ndarray,
    active_mask_points_xy: list[tuple[int, int]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Merge nearby diagonal segments using the preserved legacy heuristics."""
    merged_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    sorted_lines = sorted(raw_hough_segments, key=lambda line: line[1])
    sorted_lines = np.array(sorted_lines, dtype=object)

    for line in sorted_lines:
        if not merged_segments:
            merged_segments.append((line[0], line[1]))
            continue

        previous_point_0, previous_point_1 = merged_segments[-1]
        current_point_0, current_point_1 = line

        previous_length = _segment_length((previous_point_0, previous_point_1))
        current_length = _segment_length((current_point_0, current_point_1))
        minimum_length = max(min(previous_length, current_length), 1.0)
        merge_distance_limit = max(3.0, 0.25 * minimum_length)

        previous_angle = _segment_angle((previous_point_0, previous_point_1))
        current_angle = _segment_angle((current_point_0, current_point_1))
        angular_difference = abs(previous_angle - current_angle)
        if angular_difference > 90.0:
            angular_difference = 180.0 - angular_difference

        nearest_previous_endpoint, nearest_current_endpoint, endpoint_distance = _nearest_endpoints(
            (previous_point_0, previous_point_1),
            (current_point_0, current_point_1),
        )
        bridge_support, bridge_gap = _bridge_stats(
            nearest_previous_endpoint,
            nearest_current_endpoint,
            active_mask,
        )

        if (
            endpoint_distance <= merge_distance_limit
            and angular_difference <= 12.0
            and bridge_support >= 0.60
            and bridge_gap <= 0.20
        ):
            candidate_pairs = [
                (previous_point_0, previous_point_1),
                (previous_point_0, current_point_1),
                (current_point_0, previous_point_1),
                (current_point_0, current_point_1),
            ]
            best_pair = ((0, 0), (0, 0))
            best_score = 0
            for candidate_pair in candidate_pairs:
                pair_score = _count_points_in_range(
                    [candidate_pair],
                    active_mask_points_xy,
                    20,
                )
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = candidate_pair
            merged_segments.pop()
            merged_segments.append(best_pair)
        else:
            merged_segments.append((current_point_0, current_point_1))

    return merged_segments


# Public merge entrypoint used by the default production handoff path.
def merge_diagonal_segments(
    raw_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    active_mask: np.ndarray,
    active_mask_points_xy: list[tuple[int, int]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the exact default merged-Hough segment list.

    The implementation currently delegates to the Python reference path. This
    wrapper is the stable boundary where an exact-result accelerator can be
    introduced later without changing detector call sites or the downstream
    pipeline contract.
    """
    return merge_diagonal_segments_with_optional_accelerator(
        raw_hough_segments,
        active_mask,
        active_mask_points_xy,
        python_reference_merge_function=_merge_diagonal_segments_python_reference,
    )
