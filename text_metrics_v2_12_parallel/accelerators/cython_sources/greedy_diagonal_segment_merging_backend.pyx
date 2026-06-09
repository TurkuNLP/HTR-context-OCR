# cython: language_level=3
"""Exact-result Cython backend for the default greedy diagonal merge stage.

This module mirrors the Python reference implementation in
``hough_postprocessing.greedy_diagonal_segment_merging``. The goal here is not
algorithmic change; the goal is to keep the same segment order, the same merge
rule, the same tie-breaking, and the same returned geometry while moving the
heaviest numeric loops into compiled code.
"""

from libc.math cimport atan2, fabs, sqrt
import numpy as np
cimport numpy as cnp

cnp.import_array()


# Preserve the historical segment sort key in one reusable top-level helper.
def _line_sort_key(line):
    """Return the preserved sort key for one raw Hough segment."""
    return line[1]


cdef inline double _line_magnitude(double x1, double y1, double x2, double y2):
    """Return Euclidean point-to-point distance."""
    return sqrt(((x2 - x1) * (x2 - x1)) + ((y2 - y1) * (y2 - y1)))


cdef double _point_line_distance(object point_0, object point_1, object query_point):
    """Return point-to-segment distance using the preserved legacy formula."""
    cdef double query_x = float(query_point[0])
    cdef double query_y = float(query_point[1])
    cdef double line_x0 = float(point_0[0])
    cdef double line_y0 = float(point_0[1])
    cdef double line_x1 = float(point_1[0])
    cdef double line_y1 = float(point_1[1])
    cdef double segment_length = _line_magnitude(line_x0, line_y0, line_x1, line_y1)
    cdef double projection_numerator
    cdef double projection_position
    cdef double projected_x
    cdef double projected_y
    cdef double distance_to_first_endpoint
    cdef double distance_to_second_endpoint

    if segment_length < 1e-8:
        return 9999.0

    projection_numerator = (
        ((query_x - line_x0) * (line_x1 - line_x0))
        + ((query_y - line_y0) * (line_y1 - line_y0))
    )
    projection_position = projection_numerator / (segment_length * segment_length)

    if projection_position < 0.00001 or projection_position > 1.0:
        distance_to_first_endpoint = _line_magnitude(query_x, query_y, line_x0, line_y0)
        distance_to_second_endpoint = _line_magnitude(query_x, query_y, line_x1, line_y1)
        if distance_to_first_endpoint > distance_to_second_endpoint:
            return distance_to_second_endpoint
        return distance_to_first_endpoint

    projected_x = line_x0 + projection_position * (line_x1 - line_x0)
    projected_y = line_y0 + projection_position * (line_y1 - line_y0)
    return _line_magnitude(query_x, query_y, projected_x, projected_y)


cdef int _count_points_in_range(object segments, object points, double max_distance):
    """Count active mask points whose segment distance stays within the radius."""
    cdef int count = 0
    cdef object segment
    cdef object point_0
    cdef object point_1
    cdef object point

    for segment in segments:
        point_0 = segment[0]
        point_1 = segment[1]
        for point in points:
            if _point_line_distance(point_0, point_1, point) <= max_distance:
                count += 1
    return count


cdef tuple _sample_line_pixels(object point_0, object point_1, tuple shape):
    """Sample one bridge line onto matrix coordinates exactly like the reference path."""
    cdef double x0 = float(point_0[0])
    cdef double y0 = float(point_0[1])
    cdef double x1 = float(point_1[0])
    cdef double y1 = float(point_1[1])
    cdef int sample_count = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    if sample_count < 2:
        sample_count = 2

    sampled_x = np.clip(np.rint(np.linspace(x0, x1, sample_count)).astype(int), 0, shape[1] - 1)
    sampled_y = np.clip(np.rint(np.linspace(y0, y1, sample_count)).astype(int), 0, shape[0] - 1)
    return sampled_x, sampled_y


cdef int _longest_false_run(object values):
    """Return the longest contiguous False run in a boolean array."""
    cdef int best_run = 0
    cdef int current_run = 0
    cdef object value

    for value in values:
        if not bool(value):
            current_run += 1
            if current_run > best_run:
                best_run = current_run
        else:
            current_run = 0
    return best_run


cdef double _segment_length(object segment):
    """Return Euclidean segment length."""
    return _line_magnitude(
        float(segment[0][0]),
        float(segment[0][1]),
        float(segment[1][0]),
        float(segment[1][1]),
    )


cdef double _segment_angle(object segment):
    """Return the preserved segment direction angle in degrees mapped into [0, 180)."""
    cdef double x0 = float(segment[0][0])
    cdef double y0 = float(segment[0][1])
    cdef double x1 = float(segment[1][0])
    cdef double y1 = float(segment[1][1])
    cdef double segment_degrees = atan2(y1 - y0, x1 - x0) * (180.0 / np.pi)
    return (segment_degrees + 180.0) % 180.0


cdef tuple _nearest_endpoints(object segment_a, object segment_b):
    """Return the closest endpoint pair between two segments."""
    cdef object best_a_point = segment_a[0]
    cdef object best_b_point = segment_b[0]
    cdef double best_distance = 1e300
    cdef double distance
    cdef object point_a
    cdef object point_b

    for point_a in (segment_a[0], segment_a[1]):
        for point_b in (segment_b[0], segment_b[1]):
            distance = _line_magnitude(
                float(point_a[0]),
                float(point_a[1]),
                float(point_b[0]),
                float(point_b[1]),
            )
            if distance < best_distance:
                best_distance = distance
                best_a_point = point_a
                best_b_point = point_b

    return best_a_point, best_b_point, best_distance


cdef tuple _bridge_stats(object point_0, object point_1, object active_mask):
    """Return bridge support and normalized bridge gap."""
    cdef object sampled_x
    cdef object sampled_y
    cdef object sampled_values
    cdef double bridge_support
    cdef double normalized_bridge_gap

    sampled_x, sampled_y = _sample_line_pixels(point_0, point_1, active_mask.shape)
    sampled_values = active_mask[sampled_y, sampled_x]
    if sampled_values.size == 0:
        return 0.0, 1.0

    bridge_support = float(sampled_values.mean())
    normalized_bridge_gap = float(_longest_false_run(sampled_values) / len(sampled_values))
    return bridge_support, normalized_bridge_gap


cpdef object merge_diagonal_segments_exact(object raw_hough_segments, object active_mask, object active_mask_points_xy):
    """Return the exact merged segment list for the default greedy merge stage."""
    cdef list merged_segments = []
    cdef list sorted_lines = sorted(raw_hough_segments, key=_line_sort_key)
    cdef object line
    cdef object previous_point_0
    cdef object previous_point_1
    cdef object current_point_0
    cdef object current_point_1
    cdef object previous_segment
    cdef double previous_length
    cdef double current_length
    cdef double minimum_length
    cdef double merge_distance_limit
    cdef double previous_angle
    cdef double current_angle
    cdef double angular_difference
    cdef object nearest_previous_endpoint
    cdef object nearest_current_endpoint
    cdef double endpoint_distance
    cdef double bridge_support
    cdef double bridge_gap
    cdef list candidate_pairs
    cdef object candidate_pair
    cdef object best_pair = ((0, 0), (0, 0))
    cdef int best_score
    cdef int pair_score

    for line in sorted_lines:
        if not merged_segments:
            merged_segments.append((line[0], line[1]))
            continue

        previous_segment = merged_segments[len(merged_segments) - 1]
        previous_point_0 = previous_segment[0]
        previous_point_1 = previous_segment[1]
        current_point_0 = line[0]
        current_point_1 = line[1]

        previous_length = _segment_length((previous_point_0, previous_point_1))
        current_length = _segment_length((current_point_0, current_point_1))
        minimum_length = max(min(previous_length, current_length), 1.0)
        merge_distance_limit = max(3.0, 0.25 * minimum_length)

        previous_angle = _segment_angle((previous_point_0, previous_point_1))
        current_angle = _segment_angle((current_point_0, current_point_1))
        angular_difference = fabs(previous_angle - current_angle)
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
                pair_score = _count_points_in_range([candidate_pair], active_mask_points_xy, 20.0)
                if pair_score > best_score:
                    best_score = pair_score
                    best_pair = candidate_pair
            merged_segments.pop()
            merged_segments.append(best_pair)
        else:
            merged_segments.append((current_point_0, current_point_1))

    return merged_segments
