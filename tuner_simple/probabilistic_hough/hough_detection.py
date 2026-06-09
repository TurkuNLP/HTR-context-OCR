from __future__ import annotations

"""Local probabilistic Hough detection and line ownership filtering."""

from dataclasses import dataclass
import math
import time
from typing import Any

import numpy as np
from skimage.transform import probabilistic_hough_line

from tuner_simple.cython_accel.optional_ownership import (
    assign_columns_to_candidate_lines_with_optional_accelerator,
)

from .hough_input import build_simple_hough_context

FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES = 30.0
FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES = 60.0
FALLING_DIAGONAL_NORMAL_THETA_DEGREES = np.arange(-59.5, -30.0, 0.5)
FALLING_DIAGONAL_NORMAL_THETA_RADIANS = np.deg2rad(FALLING_DIAGONAL_NORMAL_THETA_DEGREES)


@dataclass
class HoughFilteredPayload:
    """All line geometry needed by scoring and plotting for one matrix direction."""

    hough_context: dict
    detection_result: dict
    filtered_result: dict
    raw_line_count: int
    candidate_line_count: int
    used_line_count: int
    detect_seconds: float
    filter_seconds: float


def canonicalize_segment_left_to_right(raw_segment: Any) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """Return a Hough segment with the smaller x endpoint first."""
    try:
        (x0, y0), (x1, y1) = raw_segment
    except (TypeError, ValueError):
        return None

    first_endpoint = (float(x0), float(y0))
    second_endpoint = (float(x1), float(y1))
    if first_endpoint[0] <= second_endpoint[0]:
        return first_endpoint, second_endpoint
    return second_endpoint, first_endpoint


def segment_is_falling_diagonal(segment: tuple[tuple[float, float], tuple[float, float]]) -> bool:
    """Return True when a segment moves right and down between 30 and 60 visual degrees."""
    (left_x, left_y), (right_x, right_y) = segment
    delta_x = float(right_x) - float(left_x)
    delta_y = float(right_y) - float(left_y)
    if delta_x <= 0.0 or delta_y <= 0.0:
        return False

    visual_angle_degrees = math.degrees(math.atan2(delta_y, delta_x))
    return bool(
        FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES
        <= visual_angle_degrees
        <= FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES
    )


def line_y_at_x(line_record: dict, x_position: float) -> float | None:
    """Interpolate the row coordinate where a line crosses one prediction column."""
    x0 = float(line_record["x0"])
    x1 = float(line_record["x1"])
    if x_position < min(x0, x1) - 1e-9 or x_position > max(x0, x1) + 1e-9:
        return None
    if abs(x1 - x0) <= 1e-12:
        return None

    interpolation_fraction = (float(x_position) - x0) / (x1 - x0)
    return float(line_record["y0"] + interpolation_fraction * (line_record["y1"] - line_record["y0"]))


def raw_segment_to_line_record(
    *,
    raw_segment: tuple[tuple[float, float], tuple[float, float]],
    raw_line_id: int,
) -> dict:
    """Convert one canonical raw Hough segment into the dictionary shape used downstream."""
    (x0, y0), (x1, y1) = raw_segment
    line_length = math.hypot(float(x1) - float(x0), float(y1) - float(y0))
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
    hough_image = hough_context.get("hough_image", hough_context["mask"])
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

    accepted_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for raw_segment in raw_segments_from_skimage:
        canonical_segment = canonicalize_segment_left_to_right(raw_segment)
        if canonical_segment is not None and segment_is_falling_diagonal(canonical_segment):
            accepted_segments.append(canonical_segment)

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
    return {
        "mapped_y": np.full(int(column_count), np.nan, dtype=float),
        "mapped_line_id": np.full(int(column_count), -1, dtype=int),
    }


def assign_columns_to_candidate_lines_with_python_reference(
    *,
    score_matrix: np.ndarray,
    voter_mask: np.ndarray,
    candidate_lines: list[dict],
) -> dict[str, np.ndarray]:
    """Assign columns with the readable Python implementation used as the fallback."""
    matrix = np.asarray(score_matrix, dtype=float)
    mask = np.asarray(voter_mask, dtype=bool)
    row_count, column_count = matrix.shape if matrix.ndim == 2 else (0, 0)

    mapped_y = np.full(column_count, np.nan, dtype=float)
    mapped_candidate_id = np.full(column_count, -1, dtype=int)
    owned_counts = np.zeros(len(candidate_lines), dtype=int)

    for column_index in range(column_count):
        best_candidate_id: int | None = None
        best_y_value = float("nan")
        best_score = float("-inf")

        for candidate_id, line_record in enumerate(candidate_lines):
            y_value = line_y_at_x(line_record, float(column_index))
            if y_value is None:
                continue

            row_index = int(round(float(y_value)))
            if row_index < 0 or row_index >= row_count:
                continue
            if not bool(mask[row_index, column_index]):
                continue

            score_value = float(matrix[row_index, column_index])
            if score_value > best_score:
                best_candidate_id = int(candidate_id)
                best_y_value = float(y_value)
                best_score = float(score_value)

        if best_candidate_id is None:
            continue

        mapped_candidate_id[column_index] = int(best_candidate_id)
        mapped_y[column_index] = float(best_y_value)
        owned_counts[int(best_candidate_id)] += 1

    return {
        "mapped_y": mapped_y,
        "mapped_candidate_id": mapped_candidate_id,
        "owned_counts": owned_counts,
    }


def assign_columns_to_candidate_lines(
    *,
    score_matrix: np.ndarray,
    voter_mask: np.ndarray,
    candidate_lines: list[dict],
) -> tuple[dict[str, np.ndarray], str]:
    """Use the compiled ownership scan when available, otherwise use Python."""
    accelerated_result = assign_columns_to_candidate_lines_with_optional_accelerator(
        score_matrix=score_matrix,
        voter_mask=voter_mask,
        candidate_lines=candidate_lines,
    )
    if accelerated_result is not None:
        return accelerated_result, "cython"

    return assign_columns_to_candidate_lines_with_python_reference(
        score_matrix=score_matrix,
        voter_mask=voter_mask,
        candidate_lines=candidate_lines,
    ), "python"


def compact_owned_candidate_lines(
    *,
    candidate_lines: list[dict],
    mapped_y: np.ndarray,
    mapped_candidate_id: np.ndarray,
    owned_counts: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Drop candidates that own no columns and rewrite ids to compact final ids."""
    column_count = int(mapped_candidate_id.shape[0])
    final_lines: list[dict] = []
    candidate_to_final_id: dict[int, int] = {}

    for candidate_id, line_record in enumerate(candidate_lines):
        if int(owned_counts[int(candidate_id)]) <= 0:
            continue

        owned_columns = [int(column_index) for column_index in np.flatnonzero(mapped_candidate_id == int(candidate_id))]
        final_line = dict(line_record)
        final_line["owned_columns"] = owned_columns
        final_line["owned_column_count"] = int(len(owned_columns))
        candidate_to_final_id[int(candidate_id)] = int(len(final_lines))
        final_lines.append(final_line)

    compact_mapped_line_id = np.full(column_count, -1, dtype=int)
    for column_index, candidate_id in enumerate(mapped_candidate_id):
        if int(candidate_id) < 0:
            continue
        compact_mapped_line_id[int(column_index)] = int(candidate_to_final_id.get(int(candidate_id), -1))

    compact_mapped_y = np.asarray(mapped_y, dtype=float).copy()
    compact_mapped_y[compact_mapped_line_id < 0] = np.nan
    return final_lines, {"mapped_y": compact_mapped_y, "mapped_line_id": compact_mapped_line_id}


def filter_lines_by_column_ownership(
    *,
    score_matrix: np.ndarray,
    detection_result: dict,
    hough_input_mask: np.ndarray,
    align_abs_min_len: float,
) -> dict:
    """Assign each prediction column to the strongest candidate line that crosses it."""
    matrix = np.asarray(score_matrix, dtype=float)
    voter_mask = np.asarray(hough_input_mask, dtype=bool)
    row_count, column_count = matrix.shape if matrix.ndim == 2 else (0, 0)

    candidate_lines: list[dict] = []
    for raw_line_id, raw_segment in enumerate(detection_result.get("candidate_segments", []) or []):
        line_record = raw_segment_to_line_record(raw_segment=raw_segment, raw_line_id=int(raw_line_id))
        if float(line_record["length"]) >= float(align_abs_min_len):
            candidate_lines.append(line_record)

    if row_count <= 0 or column_count <= 0 or not candidate_lines:
        return {
            "lines_used": [],
            "column_assignment": empty_column_assignment(column_count),
            "lines_for_filtering": candidate_lines,
            "ownership_backend": "none",
        }

    ownership_result, ownership_backend = assign_columns_to_candidate_lines(
        score_matrix=matrix,
        voter_mask=voter_mask,
        candidate_lines=candidate_lines,
    )
    final_lines, column_assignment = compact_owned_candidate_lines(
        candidate_lines=candidate_lines,
        mapped_y=np.asarray(ownership_result["mapped_y"], dtype=float),
        mapped_candidate_id=np.asarray(ownership_result["mapped_candidate_id"], dtype=int),
        owned_counts=np.asarray(ownership_result["owned_counts"], dtype=int),
    )

    return {
        "lines_used": final_lines,
        "column_assignment": column_assignment,
        "lines_for_filtering": candidate_lines,
        "ownership_backend": ownership_backend,
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
    hough_context = build_simple_hough_context(hough_input_mask=hough_input_mask, score_floor=float(score_floor))

    detect_started_at = time.perf_counter()
    detection_result = detect_falling_diagonal_hough_lines(
        hough_context=hough_context,
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
    )
    detect_seconds = time.perf_counter() - detect_started_at

    filter_started_at = time.perf_counter()
    filtered_result = filter_lines_by_column_ownership(
        score_matrix=np.asarray(score_matrix, dtype=float),
        detection_result=detection_result,
        hough_input_mask=np.asarray(hough_input_mask, dtype=bool),
        align_abs_min_len=float(align_abs_min_len),
    )
    filter_seconds = time.perf_counter() - filter_started_at

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


__all__ = [
    "FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES",
    "FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES",
    "HoughFilteredPayload",
    "assign_columns_to_candidate_lines_with_python_reference",
    "empty_column_assignment",
    "filter_lines_by_column_ownership",
    "run_probabilistic_hough_and_filter",
]
