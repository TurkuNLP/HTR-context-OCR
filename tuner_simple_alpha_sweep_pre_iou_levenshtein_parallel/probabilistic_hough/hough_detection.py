from __future__ import annotations

"""Local probabilistic Hough detection plus v2.2 true-IoU line filtering."""

from dataclasses import dataclass
import time
from typing import Any, Sequence

import numpy as np
from skimage.transform import probabilistic_hough_line

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.alignment.hough_segment_endpoint_records import line_records_from_raw_hough_segments
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.cython_accel.optional_ownership import (
    assign_columns_to_candidate_lines_with_optional_accelerator,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.filtering.filter_geometry_helpers import (
    FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES,
    FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES,
    line_is_falling_diagonal_in_hough_angle_range,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.filtering.line_filtering_v2_1_IoU_fast import filter_lines_for_alignment_by_ownership
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.scoring.raw_hough_line_text_filter import (
    filter_raw_hough_segments_by_line_levenshtein,
)

from .hough_input import build_simple_hough_context

FALLING_DIAGONAL_NORMAL_THETA_DEGREES = np.arange(
    -(FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES - 0.5),
    -FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES,
    0.5,
)
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
    """Return True when a segment moves right and down inside the shared Hough angle range."""
    (left_x, left_y), (right_x, right_y) = segment
    return line_is_falling_diagonal_in_hough_angle_range(
        {"x0": left_x, "y0": left_y, "x1": right_x, "y1": right_y}
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


def detect_falling_diagonal_hough_lines(
    *,
    hough_context: dict,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int | None,
) -> dict:
    """Run scikit-image Hough and keep only falling diagonal segments."""
    hough_image = hough_context.get("hough_image", hough_context["mask"])
    # When hough_seed is None, omit rng entirely so skimage uses its own PCG64 generator.
    rng_kwargs: dict = {} if hough_seed is None else {"rng": np.random.default_rng(int(hough_seed))}
    raw_segments_from_skimage = list(
        probabilistic_hough_line(
            hough_image,
            threshold=int(hough_threshold),
            line_length=int(hough_line_length),
            line_gap=int(hough_line_gap),
            theta=FALLING_DIAGONAL_NORMAL_THETA_RADIANS,
            **rng_kwargs,
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
    """Assign columns with the old readable fallback kept only for accelerator tests."""
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
    """Use the old compiled ownership scan when available, otherwise use Python."""
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


def filter_lines_by_column_ownership(
    *,
    score_matrix: np.ndarray,
    detection_result: dict,
    hough_input_mask: np.ndarray,
    align_min_iou_threshold: float,
) -> dict:
    """Run the v2.2 true-IoU ownership filter on raw Hough segments."""
    matrix = np.asarray(score_matrix, dtype=float)
    row_count, column_count = matrix.shape if matrix.ndim == 2 else (0, 0)
    candidate_segments = list(detection_result.get("candidate_segments", []) or [])
    lines_for_filtering = line_records_from_raw_hough_segments(matrix, candidate_segments)

    if row_count <= 0 or column_count <= 0 or not lines_for_filtering:
        return {
            "lines_used": [],
            "column_assignment": empty_column_assignment(column_count),
            "lines_for_filtering": lines_for_filtering,
            "ownership_backend": "true_iou_none",
            "filtered_by": "true_iou_v2_2",
        }

    cached_mask_bool = detection_result.get("mask_bool")
    if cached_mask_bool is None:
        mask_bool = np.asarray(hough_input_mask, dtype=bool)
    else:
        mask_bool = np.asarray(cached_mask_bool, dtype=bool)

    lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
        lines_for_filtering,
        matrix,
        mask_bool,
        min_iou_threshold=float(align_min_iou_threshold),
    )

    return {
        "lines_used": list(lines_used),
        "column_assignment": column_assignment,
        "lines_for_filtering": lines_for_filtering,
        "ownership_backend": "true_iou_v2_2",
        "filtered_by": "true_iou_v2_2",
        "align_min_iou_threshold": float(align_min_iou_threshold),
    }


def collect_multi_seed_hough_segments(
    *,
    hough_context: dict,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int | None,
    hough_num_runs: int,
) -> tuple[list[tuple[tuple[float, float], tuple[float, float]]], dict]:
    """Run Hough hough_num_runs times and return the deduplicated union of falling segments.

    When hough_seed is None each run omits the rng argument, letting skimage's PCG64
    generator vary naturally between calls.  When hough_seed is an integer run i uses
    seed hough_seed+i so the full ensemble is deterministic and reproducible.
    """
    # When seed is None: each call omits rng → skimage's own PCG64 randomness per run.
    # When seed is int: derive run-specific seeds arithmetically for reproducibility.
    seeds: list[int | None] = (
        [None] * int(hough_num_runs)
        if hough_seed is None
        else [int(hough_seed) + i for i in range(int(hough_num_runs))]
    )
    union: set[tuple[tuple[float, float], tuple[float, float]]] = set()
    per_run_counts: list[int] = []
    # Maps each unique segment to the number of individual runs that detected it.
    segment_hit_counts: dict[tuple[tuple[float, float], tuple[float, float]], int] = {}
    total_skimage_raw = 0
    total_direction_rejected = 0
    for run_seed in seeds:
        run_result = detect_falling_diagonal_hough_lines(
            hough_context=hough_context,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=run_seed,
        )
        run_segments: list[tuple[tuple[float, float], tuple[float, float]]] = run_result["raw_lines"]
        per_run_counts.append(int(len(run_segments)))
        total_skimage_raw += int(run_result["skimage_raw_line_count_before_direction_filter"])
        total_direction_rejected += int(run_result["direction_rejected_line_count"])
        for seg in run_segments:
            segment_hit_counts[seg] = segment_hit_counts.get(seg, 0) + 1
        union.update(run_segments)
    union_sorted: list[tuple[tuple[float, float], tuple[float, float]]] = sorted(union)
    return union_sorted, {
        "per_run_counts": per_run_counts,
        "segment_hit_counts": segment_hit_counts,
        "total_skimage_raw": total_skimage_raw,
        "total_direction_rejected": total_direction_rejected,
    }


def run_probabilistic_hough_and_filter(
    *,
    score_matrix: np.ndarray,
    hough_input_mask: np.ndarray,
    score_floor: float,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int | None,
    align_min_iou_threshold: float,
    reference_windows: Sequence[str] | None = None,
    prediction_windows: Sequence[str] | None = None,
    reference_window_count: int | None = None,
    minimum_raw_line_nls: float | None = None,
    hough_num_runs: int = 1,
    window_overlap: int = 0,
) -> HoughFilteredPayload:
    """Run local Hough detection and v2.2 true-IoU filtering.

    When hough_num_runs > 1, detection is repeated that many times using the same
    mask and parameters but different seeds; the union of all raw segments is
    assembled before the pre-IoU Levenshtein filter and IoU filter run once on
    the combined result.
    """
    hough_context = build_simple_hough_context(hough_input_mask=hough_input_mask, score_floor=float(score_floor))

    detect_started_at = time.perf_counter()
    if int(hough_num_runs) > 1:
        union_segments, run_info = collect_multi_seed_hough_segments(
            hough_context=hough_context,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=hough_seed,
            hough_num_runs=int(hough_num_runs),
        )
        detection_result: dict = {
            "threshold_start": float(hough_context.get("threshold_start", float("nan"))),
            "mask": hough_context["hough_image"],
            "mask_bool": hough_context.get("hough_mask_bool"),
            "raw_lines": union_segments,
            "candidate_segments": list(union_segments),
            "hough_num_runs": int(hough_num_runs),
            "hough_per_run_raw_counts": run_info["per_run_counts"],
            "hough_segment_run_hit_counts": run_info["segment_hit_counts"],
            "skimage_raw_line_count_before_direction_filter": run_info["total_skimage_raw"],
            "direction_rejected_line_count": run_info["total_direction_rejected"],
            "hough_union_unique_segment_count": int(len(union_segments)),
        }
    else:
        detection_result = detect_falling_diagonal_hough_lines(
            hough_context=hough_context,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=hough_seed,
        )
        detection_result["hough_num_runs"] = 1
        detection_result["hough_per_run_raw_counts"] = [int(len(detection_result["raw_lines"]))]
        detection_result["hough_union_unique_segment_count"] = int(len(detection_result["raw_lines"]))
    detect_seconds = time.perf_counter() - detect_started_at

    original_falling_segments = list(detection_result.get("raw_lines", []))
    if reference_windows is not None and prediction_windows is not None:
        raw_line_filter = filter_raw_hough_segments_by_line_levenshtein(
            score_matrix=np.asarray(score_matrix, dtype=float),
            raw_segments=original_falling_segments,
            reference_windows=reference_windows,
            prediction_windows=prediction_windows,
            reference_window_count=int(reference_window_count if reference_window_count is not None else np.asarray(score_matrix).shape[0]),
            minimum_line_nls=minimum_raw_line_nls,
            window_overlap=int(window_overlap),
        )
        detection_result = dict(detection_result)
        detection_result["raw_lines_before_pre_iou_levenshtein"] = original_falling_segments
        detection_result["candidate_segments"] = list(raw_line_filter.filtered_segments)
        detection_result["candidate_segments_after_pre_iou_levenshtein"] = list(raw_line_filter.filtered_segments)
        detection_result["pre_iou_line_levenshtein_filter_stage"] = "pre_iou_raw_hough"
        detection_result["pre_iou_line_levenshtein_filter_enabled"] = bool(raw_line_filter.filter_enabled)
        detection_result["raw_falling_line_count_before_pre_iou_levenshtein"] = int(raw_line_filter.input_line_count)
        detection_result["raw_falling_line_count_after_pre_iou_levenshtein"] = int(raw_line_filter.surviving_line_count)
        detection_result["raw_falling_line_levenshtein_removed_count"] = int(raw_line_filter.removed_line_count)
        detection_result["raw_falling_line_levenshtein_threshold"] = raw_line_filter.threshold
        detection_result["pre_iou_line_levenshtein_min"] = raw_line_filter.score_minimum
        detection_result["pre_iou_line_levenshtein_max"] = raw_line_filter.score_maximum
        detection_result["pre_iou_line_levenshtein_mean"] = raw_line_filter.score_mean
        detection_result["pre_iou_line_levenshtein_seconds"] = float(raw_line_filter.seconds)
        detection_result["pre_iou_line_levenshtein_records"] = list(raw_line_filter.line_score_records)
    else:
        detection_result = dict(detection_result)
        detection_result["raw_lines_before_pre_iou_levenshtein"] = original_falling_segments
        detection_result["pre_iou_line_levenshtein_filter_stage"] = "not_applied"
        detection_result["pre_iou_line_levenshtein_filter_enabled"] = False
        detection_result["raw_falling_line_count_before_pre_iou_levenshtein"] = int(len(original_falling_segments))
        detection_result["raw_falling_line_count_after_pre_iou_levenshtein"] = int(len(original_falling_segments))
        detection_result["raw_falling_line_levenshtein_removed_count"] = 0
        detection_result["raw_falling_line_levenshtein_threshold"] = None
        detection_result["pre_iou_line_levenshtein_min"] = None
        detection_result["pre_iou_line_levenshtein_max"] = None
        detection_result["pre_iou_line_levenshtein_mean"] = None
        detection_result["pre_iou_line_levenshtein_seconds"] = 0.0
        detection_result["pre_iou_line_levenshtein_records"] = []

    filter_started_at = time.perf_counter()
    filtered_result = filter_lines_by_column_ownership(
        score_matrix=np.asarray(score_matrix, dtype=float),
        detection_result=detection_result,
        hough_input_mask=np.asarray(hough_input_mask, dtype=bool),
        align_min_iou_threshold=float(align_min_iou_threshold),
    )
    filter_seconds = time.perf_counter() - filter_started_at

    return HoughFilteredPayload(
        hough_context=hough_context,
        detection_result=detection_result,
        filtered_result=filtered_result,
        raw_line_count=int(detection_result.get("raw_falling_line_count_before_pre_iou_levenshtein", len(detection_result.get("raw_lines", [])))),
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
    "collect_multi_seed_hough_segments",
    "empty_column_assignment",
    "filter_lines_by_column_ownership",
    "run_probabilistic_hough_and_filter",
]
