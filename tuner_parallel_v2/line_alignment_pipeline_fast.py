from __future__ import annotations

"""Fast line-detection + filtering pipeline for parameter sweeps.

This module keeps Hough detection and post-Hough ownership filtering separated so
callers can profile and optimize each stage independently.
"""

import numpy as np
from skimage.transform import probabilistic_hough_line

try:
    from .runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime_paths import ensure_tuner_runtime_paths  # type: ignore

ensure_tuner_runtime_paths()

from hough_line_transform_endpoints_no_angle_all import (  # type: ignore
    merging_diag,
    normalize_for_dense_style,
)
from line_endpoint_records import lines_from_merged_segments  # type: ignore
from score_matrix_builder import coerce_score_matrix, compute_score_matrix  # type: ignore

try:
    from .line_filtering_v2_1_IoU_fast import (
        DEFAULT_MIN_IOU_THRESHOLD,
        filter_lines_for_alignment_by_ownership,
    )
except ImportError:
    from line_filtering_v2_1_IoU_fast import (  # type: ignore
        DEFAULT_MIN_IOU_THRESHOLD,
        filter_lines_for_alignment_by_ownership,
    )


# Strict line-direction angle target: 30 < x < 60 degrees.
# In skimage Hough this corresponds to normal-angle bands:
#   -60 < theta < -30 and 30 < theta < 60.
STRICT_DIAGONAL_THETA_DEG = np.r_[
    np.arange(-59.5, -30.0, 0.5),
    np.arange(30.5, 60.0, 0.5),
]
STRICT_DIAGONAL_THETA_RAD = np.deg2rad(STRICT_DIAGONAL_THETA_DEG)


def _empty_column_assignment(n_other: int) -> dict[str, np.ndarray]:
    """Return empty ownership assignment for degenerate matrices."""
    return {
        "mapped_y": np.full(int(n_other), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_other), -1, dtype=int),
    }


def precompute_hough_context(matrix: np.ndarray, *, start_init: float) -> dict:
    """Precompute matrix-dependent Hough inputs once per document."""
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return {
            "norm": np.zeros_like(matrix),
            "test": np.zeros_like(matrix),
            "mask": np.zeros_like(matrix),
            "points_glo": [],
            "threshold_start": float("nan"),
        }

    norm = normalize_for_dense_style(matrix)
    test = 1.0 / (1.0 - norm)

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

    ys, xs = np.nonzero(test2)
    points_glo = [(int(x), int(y)) for y, x in zip(ys, xs)]

    return {
        "norm": norm,
        "test": test,
        "mask": test2,
        "points_glo": points_glo,
        "threshold_start": float(start),
    }


def detect_lines_only_from_hough_ctx(
    *,
    hough_ctx: dict,
    seed: int,
    threshold: int,
    line_length: int,
    line_gap: int,
) -> dict:
    """Run Hough detection + diagonal merge only (no filtering)."""
    test2 = hough_ctx["mask"]
    points_glo = hough_ctx["points_glo"]

    lines = probabilistic_hough_line(
        test2,
        threshold=int(threshold),
        line_length=int(line_length),
        line_gap=int(line_gap),
        theta=STRICT_DIAGONAL_THETA_RAD,
        rng=np.random.default_rng(int(seed)),
    )

    selected_lines = list(lines)
    merged_lines = merging_diag(selected_lines, test2 > 0, points_glo)

    return {
        "threshold_start": float(hough_ctx.get("threshold_start", float("nan"))),
        "mask": test2,
        "raw_lines": selected_lines,
        "merged_lines": merged_lines,
    }


def filter_lines_after_hough(
    *,
    matrix: np.ndarray,
    det_result: dict,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> dict:
    """Apply ownership-based filtering after Hough detection."""
    mat = coerce_score_matrix(matrix, source_desc="post_hough_filter")

    merged_lines = list(det_result.get("merged_lines", []))
    lines_for_filtering = lines_from_merged_segments(mat, merged_lines)

    if mat.size > 0:
        mask_bool = np.asarray(det_result.get("mask", np.zeros_like(mat))) > 0
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
        "lines_for_filtering": filtered["lines_for_filtering"],
        "lines_used": filtered["lines_used"],
        "column_assignment": filtered["column_assignment"],
    }


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
        "merged_line_count": int(len(lines_for_filtering)),
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
    "STRICT_DIAGONAL_THETA_DEG",
    "STRICT_DIAGONAL_THETA_RAD",
    "precompute_hough_context",
    "detect_lines_only_from_hough_ctx",
    "filter_lines_after_hough",
    "detect_and_filter_lines_from_matrix",
    "derive_filtered_line_endpoints",
]
