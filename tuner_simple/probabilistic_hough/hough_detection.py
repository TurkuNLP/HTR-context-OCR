from __future__ import annotations

"""Local probabilistic Hough detection and simple line ownership filtering."""

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from skimage.transform import probabilistic_hough_line

from .hough_input import build_simple_hough_context

# A falling line in matrix coordinates moves right and down, matching reference/prediction progression.
FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES = 30.0
# Lines steeper than this are rejected because they are unlikely to represent normal text progression.
FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES = 60.0
# scikit-image uses normal angles, so this range corresponds to falling visual diagonals.
FALLING_DIAGONAL_NORMAL_THETA_DEGREES = np.arange(-59.5, -30.0, 0.5)
# Precompute radians once because probabilistic_hough_line expects theta values in radians.
FALLING_DIAGONAL_NORMAL_THETA_RADIANS = np.deg2rad(FALLING_DIAGONAL_NORMAL_THETA_DEGREES)


@dataclass
class HoughFilteredPayload:
    """All line geometry needed by scoring and plotting for one matrix direction."""

    # Store the Hough input mask and score-floor metadata used to create the lines.
    hough_context: dict
    # Store raw Hough output before ownership filtering.
    detection_result: dict
    # Store filtered line records and column-to-line assignment arrays.
    filtered_result: dict
    # Store how many raw Hough segments scikit-image returned after direction filtering.
    raw_line_count: int
    # Store how many candidate line records were available to ownership filtering.
    candidate_line_count: int
    # Store how many lines owned at least one prediction column after filtering.
    used_line_count: int
    # Store the Hough detection runtime for audit output.
    detect_seconds: float
    # Store the ownership filtering runtime for audit output.
    filter_seconds: float


def canonicalize_segment_left_to_right(raw_segment: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return a Hough segment with the smaller x endpoint first."""
    # Try to unpack the scikit-image endpoint pair into numeric coordinates.
    try:
        # Extract the two endpoints from the raw segment.
        (x0, y0), (x1, y1) = raw_segment
    # Reject malformed segments by returning None instead of raising inside the main loop.
    except (TypeError, ValueError):
        # Return None so the caller can skip this malformed segment.
        return None
    # Convert all coordinates to floats because later interpolation uses fractional values.
    first_endpoint = (float(x0), float(y0))
    # Convert the second endpoint in the same way.
    second_endpoint = (float(x1), float(y1))
    # Keep the endpoint with the smaller x coordinate first so slope checks are consistent.
    if first_endpoint[0] <= second_endpoint[0]:
        # Return the endpoints in their original order when they already go left to right.
        return first_endpoint, second_endpoint
    # Return the endpoints swapped so the segment always goes left to right.
    return second_endpoint, first_endpoint


def segment_is_falling_diagonal(segment: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    """Return True when a segment moves right and down between 30 and 60 visual degrees."""
    # Unpack the left and right endpoints after canonicalization.
    (left_x, left_y), (right_x, right_y) = segment
    # Compute horizontal movement from left endpoint to right endpoint.
    delta_x = float(right_x) - float(left_x)
    # Compute vertical movement from left endpoint to right endpoint.
    delta_y = float(right_y) - float(left_y)
    # Reject vertical, horizontal, reversed, and upward lines before calculating an angle.
    if delta_x <= 0.0 or delta_y <= 0.0:
        # Return False because a valid alignment line must move down as prediction columns increase.
        return False
    # Convert the line slope into the visual angle shown on score-matrix plots.
    visual_angle_degrees = math.degrees(math.atan2(delta_y, delta_x))
    # Accept only the angle band that represents plausible falling diagonal text alignment.
    return bool(
        FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES
        <= visual_angle_degrees
        <= FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES
    )


def line_y_at_x(line_record: dict, x_position: float) -> float | None:
    """Interpolate the row coordinate where a line crosses one prediction column."""
    # Read the left endpoint x coordinate stored in the line record.
    x0 = float(line_record["x0"])
    # Read the right endpoint x coordinate stored in the line record.
    x1 = float(line_record["x1"])
    # A line owns only columns that lie between its endpoints, with a small tolerance for rounding.
    if x_position < min(x0, x1) - 1e-9 or x_position > max(x0, x1) + 1e-9:
        # Return None when the line does not span this column.
        return None
    # Avoid division by zero if a malformed vertical line reaches this helper.
    if abs(x1 - x0) <= 1e-12:
        # Return None because vertical lines cannot map prediction columns to a progression of reference rows.
        return None
    # Compute the interpolation fraction between the two endpoints.
    interpolation_fraction = (float(x_position) - x0) / (x1 - x0)
    # Interpolate the y coordinate at this x position.
    return float(line_record["y0"] + interpolation_fraction * (line_record["y1"] - line_record["y0"]))


def raw_segment_to_line_record(
    *,
    raw_segment: tuple[tuple[float, float], tuple[float, float]],
    raw_line_id: int,
) -> dict:
    """Convert one canonical raw Hough segment into the dictionary shape used downstream."""
    # Unpack the canonical left-to-right endpoints.
    (x0, y0), (x1, y1) = raw_segment
    # Compute the Euclidean segment length in matrix-cell units.
    line_length = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
    # Return one explicit dictionary so plotting, filtering, and scoring share the same geometry fields.
    return {
        "raw_line_id": int(raw_line_id),
        "source_raw_line_ids": [int(raw_line_id)],
        "x0": float(x0),
        "y0": float(y0),
        "x1": float(x1),
        "y1": float(y1),
        "length": float(line_length),
    }


def detect_falling_diagonal_hough_lines(
    *,
    hough_context: dict,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
) -> dict:
    """Run scikit-image Hough and keep only falling diagonal segments."""
    # Prefer the prepared Hough image and fall back to the boolean mask if no image key exists.
    hough_image = hough_context.get("hough_image", hough_context["mask"])
    # Run probabilistic Hough with a deterministic NumPy random generator.
    raw_segments_from_skimage = list(
        probabilistic_hough_line(
            hough_image,
            threshold=int(hough_threshold),
            line_length=int(hough_line_length),
            line_gap=int(hough_line_gap),
            theta=FALLING_DIAGONAL_NORMAL_THETA_RADIANS,
            rng=np.random.default_rng(int(hough_seed)),
        )
    )
    # Keep accepted segments in a separate list so rejected segment counts remain easy to explain.
    accepted_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    # Inspect every segment returned by scikit-image.
    for raw_segment in raw_segments_from_skimage:
        # Convert endpoint order to left-to-right before checking slope.
        canonical_segment = canonicalize_segment_left_to_right(raw_segment)
        # Skip malformed raw segments defensively.
        if canonical_segment is None:
            # Continue with the next raw segment.
            continue
        # Keep only falling diagonal candidates in the allowed angle band.
        if segment_is_falling_diagonal(canonical_segment):
            # Add the canonical segment to the accepted list.
            accepted_segments.append(canonical_segment)
    # Return raw line geometry and counts for filtering, plotting, and audit output.
    return {
        "threshold_start": float(hough_context.get("threshold_start", float("nan"))),
        "mask": hough_image,
        "mask_bool": hough_context.get("hough_mask_bool"),
        "raw_lines": accepted_segments,
        "candidate_segments": list(accepted_segments),
        "skimage_raw_line_count_before_direction_filter": int(len(raw_segments_from_skimage)),
        "direction_rejected_line_count": int(len(raw_segments_from_skimage) - len(accepted_segments)),
    }


def empty_column_assignment(column_count: int) -> dict[str, np.ndarray]:
    """Return assignment arrays for a matrix with no usable lines."""
    # Fill mapped_y with NaN because no prediction column has a reference-row estimate.
    mapped_y = np.full(int(column_count), np.nan, dtype=float)
    # Fill mapped_line_id with -1 because no prediction column is owned by a line.
    mapped_line_id = np.full(int(column_count), -1, dtype=int)
    # Return both arrays in the shared assignment dictionary shape.
    return {"mapped_y": mapped_y, "mapped_line_id": mapped_line_id}


def filter_lines_by_column_ownership(
    *,
    score_matrix: np.ndarray,
    detection_result: dict,
    hough_input_mask: np.ndarray,
    align_abs_min_len: float,
) -> dict:
    """Assign each prediction column to the strongest candidate line that crosses it."""
    # Convert the score matrix to float values for safe indexing and comparison.
    matrix = np.asarray(score_matrix, dtype=float)
    # Convert the Hough input to a boolean mask so only voting cells can support ownership.
    voter_mask = np.asarray(hough_input_mask, dtype=bool)
    # Derive the matrix shape, falling back to zero columns if the matrix is malformed.
    row_count, column_count = matrix.shape if matrix.ndim == 2 else (0, 0)
    # Convert raw Hough segments into line dictionaries and apply the absolute line-length filter.
    candidate_lines: list[dict] = []
    # Iterate through accepted raw Hough segments in stable order.
    for raw_line_id, raw_segment in enumerate(detection_result.get("candidate_segments", []) or []):
        # Convert the raw endpoint tuple into a local line record.
        line_record = raw_segment_to_line_record(raw_segment=raw_segment, raw_line_id=int(raw_line_id))
        # Keep only lines whose geometric length meets the configured minimum.
        if float(line_record["length"]) >= float(align_abs_min_len):
            # Add the line to the candidate list used for column ownership.
            candidate_lines.append(line_record)
    # If there is no matrix or no candidate line, return an empty assignment.
    if row_count <= 0 or column_count <= 0 or not candidate_lines:
        # Return the candidate list for diagnostics even when no final line survives.
        return {
            "lines_used": [],
            "column_assignment": empty_column_assignment(column_count),
            "lines_for_filtering": candidate_lines,
        }
    # Start every prediction column as unassigned.
    mapped_y = np.full(column_count, np.nan, dtype=float)
    # Start every prediction column with line id -1, meaning no owning line.
    mapped_line_id = np.full(column_count, -1, dtype=int)
    # Track which columns each candidate owns before candidate ids are compacted.
    owned_columns_by_candidate: dict[int, list[int]] = {candidate_id: [] for candidate_id in range(len(candidate_lines))}
    # Process every prediction column independently.
    for column_index in range(column_count):
        # Store the best candidate seen so far for this prediction column.
        best_candidate_id: int | None = None
        # Store the row coordinate associated with the best candidate.
        best_y_value = float("nan")
        # Store the score at the candidate cell so the strongest crossing line wins the column.
        best_score = float("-inf")
        # Test every candidate line against this column.
        for candidate_id, line_record in enumerate(candidate_lines):
            # Interpolate where this line crosses the current prediction column.
            y_value = line_y_at_x(line_record, float(column_index))
            # Skip lines that do not span this column.
            if y_value is None:
                # Continue with the next candidate line.
                continue
            # Round the interpolated row to the nearest score-matrix cell.
            row_index = int(round(float(y_value)))
            # Skip crossings outside the score matrix.
            if row_index < 0 or row_index >= row_count:
                # Continue with the next candidate line.
                continue
            # Require that the crossing cell is part of the binary Hough input.
            if not bool(voter_mask[row_index, column_index]):
                # Continue with the next candidate line because this cell did not survive preprocessing.
                continue
            # Read the score at the crossing cell.
            score_value = float(matrix[row_index, column_index])
            # Keep the candidate with the highest score at this column.
            if score_value > best_score:
                # Store the candidate id as the current owner.
                best_candidate_id = int(candidate_id)
                # Store the interpolated y coordinate, not only the rounded row, for smoother downstream plots.
                best_y_value = float(y_value)
                # Store the score used for this ownership decision.
                best_score = float(score_value)
        # Leave the column unassigned if no candidate crossed a surviving voter cell.
        if best_candidate_id is None:
            # Continue with the next prediction column.
            continue
        # Record the winning line id for this column using the temporary candidate id.
        mapped_line_id[column_index] = int(best_candidate_id)
        # Record the winning y coordinate for this column.
        mapped_y[column_index] = float(best_y_value)
        # Add this column to the candidate's owned-column list.
        owned_columns_by_candidate[int(best_candidate_id)].append(int(column_index))
    # Compact candidate ids so final line ids are contiguous and only owned lines survive.
    final_lines: list[dict] = []
    # Map temporary candidate ids to final line ids.
    candidate_to_final_id: dict[int, int] = {}
    # Visit candidates in original order for stable output.
    for candidate_id, line_record in enumerate(candidate_lines):
        # Keep only candidates that own at least one prediction column.
        if not owned_columns_by_candidate.get(candidate_id):
            # Continue with the next candidate line.
            continue
        # Copy the line record so adding ownership metadata does not mutate candidate_lines unexpectedly.
        final_line = dict(line_record)
        # Store the prediction columns owned by this final line.
        final_line["owned_columns"] = list(owned_columns_by_candidate[int(candidate_id)])
        # Store how many columns this line owns for easy output and debugging.
        final_line["owned_column_count"] = int(len(final_line["owned_columns"]))
        # Assign the next compact final line id.
        final_line_id = int(len(final_lines))
        # Store the mapping from temporary candidate id to final line id.
        candidate_to_final_id[int(candidate_id)] = int(final_line_id)
        # Append the line to the final used-line list.
        final_lines.append(final_line)
    # Rewrite temporary candidate ids in the assignment array to compact final line ids.
    compact_mapped_line_id = np.full(column_count, -1, dtype=int)
    # Iterate over every prediction column to rewrite the ids.
    for column_index, candidate_id in enumerate(mapped_line_id):
        # Skip unassigned columns.
        if int(candidate_id) < 0:
            # Continue with the next column.
            continue
        # Copy the compact final id for assigned columns whose candidate survived.
        compact_mapped_line_id[column_index] = int(candidate_to_final_id.get(int(candidate_id), -1))
    # Remove mapped y values for any column whose candidate did not survive compaction.
    mapped_y[compact_mapped_line_id < 0] = np.nan
    # Return the local filtered result in the same broad shape expected by the rest of tuner_simple.
    return {
        "lines_used": final_lines,
        "column_assignment": {"mapped_y": mapped_y, "mapped_line_id": compact_mapped_line_id},
        "lines_for_filtering": candidate_lines,
    }


def run_probabilistic_hough_and_filter(
    *,
    score_matrix: np.ndarray,
    hough_input_mask: np.ndarray,
    score_floor: float,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> HoughFilteredPayload:
    """Run local Hough detection and local ownership filtering once."""
    # Build the shared context dictionary from the binary Hough input mask.
    hough_context = build_simple_hough_context(hough_input_mask=hough_input_mask, score_floor=float(score_floor))
    # Mark the start of Hough detection timing.
    detect_started_at = time.perf_counter()
    # Run scikit-image Hough and keep only falling diagonal raw segments.
    detection_result = detect_falling_diagonal_hough_lines(
        hough_context=hough_context,
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
    )
    # Store how long Hough detection took.
    detect_seconds = time.perf_counter() - detect_started_at
    # Mark the start of local ownership filtering timing.
    filter_started_at = time.perf_counter()
    # Assign columns to the strongest candidate line crossing surviving voter cells.
    filtered_result = filter_lines_by_column_ownership(
        score_matrix=np.asarray(score_matrix, dtype=float),
        detection_result=detection_result,
        hough_input_mask=np.asarray(hough_input_mask, dtype=bool),
        align_abs_min_len=float(align_abs_min_len),
    )
    # Store how long local filtering took.
    filter_seconds = time.perf_counter() - filter_started_at
    # Return one payload that carries raw lines, final lines, assignments, and timings.
    return HoughFilteredPayload(
        hough_context=hough_context,
        detection_result=detection_result,
        filtered_result=filtered_result,
        raw_line_count=int(len(detection_result.get("raw_lines", []))),
        candidate_line_count=int(len(filtered_result.get("lines_for_filtering", []))),
        used_line_count=int(len(filtered_result.get("lines_used", []))),
        detect_seconds=float(detect_seconds),
        filter_seconds=float(filter_seconds),
    )


# Declare the public helpers that other tuner_simple modules may import.
__all__ = [
    "FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES",
    "FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES",
    "HoughFilteredPayload",
    "empty_column_assignment",
    "run_probabilistic_hough_and_filter",
]
