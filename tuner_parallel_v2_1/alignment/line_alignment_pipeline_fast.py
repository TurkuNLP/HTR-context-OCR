from __future__ import annotations

"""Fast line-detection + filtering pipeline for parameter sweeps.

This module keeps Hough detection and post-Hough ownership filtering separated
so callers can profile and optimize each stage independently.

Important change from the older tuner path:
- raw probabilistic Hough segments are no longer pre-merged with ``merge_diag``
- filtering is now the only place that merges overlapping/duplicate guides
"""

import math

import numpy as np
from skimage.transform import probabilistic_hough_line

try:
    from ..runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

# Ensure shared helper modules are importable in both module and script usage.
ensure_tuner_runtime_paths()

from score_matrix_builder import coerce_score_matrix, compute_score_matrix  # type: ignore

try:
    from .hough_segment_endpoint_records import line_records_from_raw_hough_segments
    from ..filtering.line_filtering_v2_1_IoU_fast import (
        DEFAULT_MIN_IOU_THRESHOLD,
        filter_lines_for_alignment_by_ownership,
    )
except ImportError:
    from alignment.hough_segment_endpoint_records import line_records_from_raw_hough_segments  # type: ignore
    from filtering.line_filtering_v2_1_IoU_fast import (  # type: ignore
        DEFAULT_MIN_IOU_THRESHOLD,
        filter_lines_for_alignment_by_ownership,
    )


# Falling visual diagonal target in image/matrix coordinates:
# x increases left-to-right, y increases top-to-bottom, so a valid line must
# move from upper-left toward lower-right when read left to right.
FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES = 30.0
FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES = 60.0

# skimage Hough theta values are normal-angle values, not the visual angle of
# the plotted line.  The negative normal-angle band below corresponds to the
# desired falling visual diagonals around 30..60 degrees.
FALLING_DIAGONAL_NORMAL_THETA_DEG = np.arange(-59.5, -30.0, 0.5)
FALLING_DIAGONAL_NORMAL_THETA_RAD = np.deg2rad(FALLING_DIAGONAL_NORMAL_THETA_DEG)

# Backward-compatible constant names remain exported for older callers, but the
# values now intentionally represent falling diagonals only.
STRICT_DIAGONAL_THETA_DEG = FALLING_DIAGONAL_NORMAL_THETA_DEG
STRICT_DIAGONAL_THETA_RAD = FALLING_DIAGONAL_NORMAL_THETA_RAD


def canonicalize_hough_segment_left_to_right(
    hough_segment: tuple[tuple[float, float], tuple[float, float]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return the segment with the left endpoint first.

    skimage does not promise that endpoint 0 is the visual start of the line, so
    every directional check must canonicalize endpoints before looking at slope.
    """
    (x0, y0), (x1, y1) = hough_segment
    if float(x0) <= float(x1):
        return (float(x0), float(y0)), (float(x1), float(y1))
    return (float(x1), float(y1)), (float(x0), float(y0))


def hough_segment_is_falling_diagonal(
    hough_segment: tuple[tuple[float, float], tuple[float, float]],
) -> bool:
    """Return True when a canonicalized segment is a falling 30..60 degree line."""
    (left_x, left_y), (right_x, right_y) = canonicalize_hough_segment_left_to_right(hough_segment)
    delta_x = float(right_x) - float(left_x)
    delta_y = float(right_y) - float(left_y)

    # A falling visual line must move rightward and downward in image/matrix
    # coordinates.  Horizontal, vertical, and upward lines are rejected here.
    if delta_x <= 0.0 or delta_y <= 0.0:
        return False

    visual_angle_degrees = math.degrees(math.atan2(delta_y, delta_x))
    return (
        FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES
        <= visual_angle_degrees
        <= FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES
    )


def keep_only_falling_diagonal_hough_segments(
    hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return only left-to-right canonical falling diagonal Hough segments."""
    falling_diagonal_segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for hough_segment in hough_segments:
        canonical_segment = canonicalize_hough_segment_left_to_right(hough_segment)
        if hough_segment_is_falling_diagonal(canonical_segment):
            falling_diagonal_segments.append(canonical_segment)
    return falling_diagonal_segments


# Normalize a score matrix into the dense-style range expected by Hough.
def normalize_for_dense_style(mat: np.ndarray) -> np.ndarray:
    """Normalize the matrix into the ``[0, 1)`` range used by the Hough path."""
    if mat.size == 0:
        return mat

    max_val = float(np.max(mat))
    if max_val <= 1.0:
        norm = mat.copy()
    elif max_val <= 100.0:
        norm = mat / 100.0
    else:
        norm = mat / max_val

    # Clip below 1.0 so the reciprocal emphasis step never divides by zero.
    return np.clip(norm, 0.0, 0.999999)


# Build the empty ownership assignment used for degenerate matrices.
def _empty_column_assignment(n_other: int) -> dict[str, np.ndarray]:
    """Return empty ownership assignment for degenerate matrices."""
    return {
        "mapped_y": np.full(int(n_other), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_other), -1, dtype=int),
    }


# Precompute the matrix-dependent Hough inputs once per document.
def precompute_hough_context(
    matrix: np.ndarray,
    *,
    start_init: float,
    keep_debug_arrays: bool = True,
) -> dict:
    """Precompute thresholded Hough inputs once per document.

    The returned payload contains only matrix-derived artifacts that are reused
    across all parameter combinations for the same document.
    """
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        empty_hough_image = np.zeros_like(matrix)
        empty_context = {
            "hough_image": empty_hough_image,
            "hough_mask_bool": np.zeros_like(matrix, dtype=bool),
            "mask": empty_hough_image,
            "threshold_start": float("nan"),
        }
        if keep_debug_arrays:
            empty_context["norm"] = np.zeros_like(matrix)
            empty_context["test"] = np.zeros_like(matrix)
        return empty_context

    # Preserve the same dense-style preprocessing used by the earlier pipeline.
    norm = normalize_for_dense_style(matrix)
    test = 1.0 / (1.0 - norm)

    # Preserve the adaptive threshold search exactly so the active mask density
    # remains compatible with the previous tuner behavior.
    start = float(start_init)
    enough = False
    criteria = 1.4 * matrix.shape[0]

    test2 = test.copy()
    while not enough:
        if start < 0:
            break
        start -= 0.2
        test2 = test.copy()
        test2[test2 < start] = 0
        enough = (test2 > 0).sum() > criteria

    # Keep the old "mask" key as a compatibility alias while giving the active
    # sweep path more descriptive names and a precomputed boolean mask.
    context = {
        "hough_image": test2,
        "hough_mask_bool": np.asarray(test2) > 0,
        "mask": test2,
        "threshold_start": float(start),
    }
    if keep_debug_arrays:
        context["norm"] = norm
        context["test"] = test
    return context


# Run probabilistic Hough on a precomputed context, without any geometric merge stage.
def detect_lines_only_from_hough_ctx(
    *,
    hough_ctx: dict,
    seed: int,
    threshold: int,
    line_length: int,
    line_gap: int,
) -> dict:
    """Run Hough detection only and return raw candidate segments."""
    test2 = hough_ctx.get("hough_image", hough_ctx["mask"])

    # Use a deterministic NumPy Generator so results stay reproducible per seed.
    raw_hough_segments_from_skimage = list(
        probabilistic_hough_line(
            test2,
            threshold=int(threshold),
            line_length=int(line_length),
            line_gap=int(line_gap),
            theta=FALLING_DIAGONAL_NORMAL_THETA_RAD,
            rng=np.random.default_rng(int(seed)),
        )
    )

    # Apply an explicit endpoint-based direction check as a second guard after
    # Hough theta restriction.  This keeps upward false positives out of
    # true-IoU filtering and makes the visual definition easy to test.
    raw_lines = keep_only_falling_diagonal_hough_segments(raw_hough_segments_from_skimage)

    # Filtering is now responsible for all merging, so candidate_segments is
    # simply the raw Hough output in stable list form.
    candidate_segments = list(raw_lines)

    return {
        "threshold_start": float(hough_ctx.get("threshold_start", float("nan"))),
        "mask": test2,
        "mask_bool": hough_ctx.get("hough_mask_bool"),
        "raw_lines": raw_lines,
        "candidate_segments": candidate_segments,
    }


# Apply ownership-based filtering after Hough detection.
def filter_lines_after_hough(
    *,
    matrix: np.ndarray,
    det_result: dict,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
    matrix_is_prepared: bool = False,
) -> dict:
    """Convert raw segments into line records, then run the ownership filter."""
    # Public compatibility callers still get defensive matrix coercion.  The
    # tuner hot path passes prepared matrices that were coerced once during
    # document loading, avoiding millions of repeated dtype/NaN checks.
    if matrix_is_prepared:
        mat = matrix
    else:
        mat = coerce_score_matrix(matrix, source_desc="post_hough_filter")

    # Convert raw Hough segments into the line-record dictionaries consumed by
    # true-IoU filtering.  The helper is tuner-local and mirrors the v2.12
    # endpoint conversion, avoiding the old v2.1 import-path dependency.
    candidate_segments = list(det_result.get("candidate_segments", []))
    lines_for_filtering = line_records_from_raw_hough_segments(mat, candidate_segments)

    if mat.size > 0:
        cached_mask_bool = det_result.get("mask_bool")
        if cached_mask_bool is None:
            mask_bool = np.asarray(det_result.get("mask", np.zeros_like(mat))) > 0
        else:
            mask_bool = np.asarray(cached_mask_bool, dtype=bool)
        lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
            lines_for_filtering,
            mat,
            mask_bool,
            abs_min_len=float(align_abs_min_len),
            min_iou_threshold=float(align_min_iou_threshold),
        )
    else:
        n_other = mat.shape[1] if mat.ndim == 2 else 0
        lines_used = []
        column_assignment = _empty_column_assignment(n_other)

    return {
        "lines_used": lines_used,
        "column_assignment": column_assignment,
        "lines_for_filtering": lines_for_filtering,
    }


# Compatibility helper that keeps detect+filter available in one call.
def detect_and_filter_lines_from_matrix(
    matrix: np.ndarray,
    *,
    hough_ctx: dict | None = None,
    item_index: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    hough_start: float = 2.6,
    align_abs_min_len: float,
    align_min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
) -> dict:
    """Compatibility helper: detect + filter in one call."""
    mat = coerce_score_matrix(matrix, source_desc="tuner_parallel_v2:line_alignment_pipeline_fast:matrix")

    resolved_ctx = hough_ctx
    if resolved_ctx is None:
        resolved_ctx = precompute_hough_context(mat, start_init=float(hough_start))

    det = detect_lines_only_from_hough_ctx(
        hough_ctx=resolved_ctx,
        seed=int(hough_seed) + int(item_index),
        threshold=int(hough_threshold),
        line_length=int(hough_line_length),
        line_gap=int(hough_line_gap),
    )

    filtered = filter_lines_after_hough(
        matrix=mat,
        det_result=det,
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
    )

    return {
        "det": det,
        "raw_hough_segments": list(det.get("raw_lines", [])),
        "candidate_segments": list(det.get("candidate_segments", [])),
        "lines_for_filtering": filtered["lines_for_filtering"],
        "lines_used": filtered["lines_used"],
        "column_assignment": filtered["column_assignment"],
    }


# Compatibility API for callers that expect endpoint debug payloads.
def derive_filtered_line_endpoints(
    *,
    ref_text: str,
    pred_text: str,
    item_index: int,
    window_size: int,
    window_stride: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    hough_start: float,
    align_abs_min_len: float,
    align_min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    precomputed_matrix: np.ndarray | None = None,
) -> tuple[list[dict], dict]:
    """Compatibility API for callers that expect endpoint debug payloads."""
    if precomputed_matrix is None:
        matrix = compute_score_matrix(
            ref_text,
            pred_text,
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
    else:
        matrix = coerce_score_matrix(precomputed_matrix, source_desc="tuner_parallel_v2:precomputed_matrix")

    hough_ctx = precompute_hough_context(matrix, start_init=float(hough_start))
    payload = detect_and_filter_lines_from_matrix(
        matrix,
        hough_ctx=hough_ctx,
        item_index=int(item_index),
        hough_threshold=int(hough_threshold),
        hough_line_length=int(hough_line_length),
        hough_line_gap=int(hough_line_gap),
        hough_seed=int(hough_seed),
        hough_start=float(hough_start),
        align_abs_min_len=float(align_abs_min_len),
        align_min_iou_threshold=float(align_min_iou_threshold),
    )

    det = payload["det"]
    lines_for_filtering = payload["lines_for_filtering"]
    lines_used = payload["lines_used"]
    matrix_shape = [int(matrix.shape[0]), int(matrix.shape[1])] if matrix.ndim == 2 else [0, 0]

    endpoint_debug = {
        "matrix_shape": matrix_shape,
        "raw_line_count": int(len(payload["raw_hough_segments"])),
        "candidate_line_count": int(len(lines_for_filtering)),
        "used_line_count": int(len(lines_used)),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "hough_threshold": int(hough_threshold),
        "hough_line_length": int(hough_line_length),
        "hough_line_gap": int(hough_line_gap),
        "hough_seed": int(hough_seed) + int(item_index),
        "hough_start": float(hough_start),
        "align_abs_min_len": float(align_abs_min_len),
        "align_min_iou_threshold": float(align_min_iou_threshold),
    }
    return list(lines_used), endpoint_debug


__all__ = [
    "DEFAULT_MIN_IOU_THRESHOLD",
    "FALLING_DIAGONAL_MAX_VISUAL_ANGLE_DEGREES",
    "FALLING_DIAGONAL_MIN_VISUAL_ANGLE_DEGREES",
    "FALLING_DIAGONAL_NORMAL_THETA_DEG",
    "FALLING_DIAGONAL_NORMAL_THETA_RAD",
    "STRICT_DIAGONAL_THETA_DEG",
    "STRICT_DIAGONAL_THETA_RAD",
    "canonicalize_hough_segment_left_to_right",
    "hough_segment_is_falling_diagonal",
    "keep_only_falling_diagonal_hough_segments",
    "normalize_for_dense_style",
    "precompute_hough_context",
    "detect_lines_only_from_hough_ctx",
    "filter_lines_after_hough",
    "detect_and_filter_lines_from_matrix",
    "derive_filtered_line_endpoints",
]
