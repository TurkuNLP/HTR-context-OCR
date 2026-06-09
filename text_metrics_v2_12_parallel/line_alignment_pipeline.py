from __future__ import annotations

import numpy as np

from hough_detection.line_handoff_modes import (
    DEFAULT_HOUGH_HANDOFF_MODE,
    HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU,
    select_hough_segments_for_true_iou,
)
from hough_line_transform_endpoints_line_direction_30_to_60_degrees import (
    detect_lines_dense_style_diagonal_fixed_theta,
)
from line_endpoint_records import lines_from_hough_segments
from line_filtering_v2_12_IoU import filter_lines_for_alignment_by_ownership
from score_matrix_builder import coerce_score_matrix


# Build the stable empty fallback assignment used when no lines survive.
def _empty_column_assignment(n_other: int) -> dict[str, np.ndarray]:
    """Return the empty fallback column assignment structure for one matrix.

    The report pipeline expects the assignment arrays to exist even when no line
    survives filtering, so this helper keeps that shape stable.
    """
    return {
        "mapped_y": np.full(int(n_other), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_other), -1, dtype=int),
    }


# Run Hough detection and true-IoU filtering for one score matrix.
def detect_and_filter_lines_from_matrix(
    matrix: np.ndarray,
    *,
    item_index: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    hough_start: float,
    hough_handoff_mode: str = DEFAULT_HOUGH_HANDOFF_MODE,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> dict:
    """Run Hough detection and true-IoU filtering for one score matrix.

    This is the only line-alignment entry point used by
    ``run_text_metrics_report.sh``. The detector uses the fixed line-direction
    angle range encoded by the Hough module, then forwards either the merged or
    raw Hough segments into the exact same production ownership-based true-IoU
    filter depending on the selected handoff mode.
    """
    matrix_for_alignment = coerce_score_matrix(matrix, source_desc="line_alignment_pipeline:matrix")

    if matrix_for_alignment.size > 0 and matrix_for_alignment.shape[0] > 0 and matrix_for_alignment.shape[1] > 0:
        detector_payload = detect_lines_dense_style_diagonal_fixed_theta(
            matrix_for_alignment,
            seed=int(hough_seed) + int(item_index),
            threshold=int(hough_threshold),
            line_length=int(hough_line_length),
            line_gap=int(hough_line_gap),
            start_init=float(hough_start),
            compute_merged_hough_segments=(
                str(hough_handoff_mode) == HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU
            ),
        )
        raw_hough_segments = list(detector_payload.get("raw_lines", []))
        merged_hough_segments = list(detector_payload.get("merged_lines", []))
    else:
        detector_payload = {
            "threshold_start": float("nan"),
            "mask": np.zeros_like(matrix_for_alignment),
            "raw_lines": [],
            "selected_lines": [],
            "merged_lines": [],
        }
        raw_hough_segments = []
        merged_hough_segments = []

    hough_segments_for_filtering = select_hough_segments_for_true_iou(
        raw_hough_segments=raw_hough_segments,
        merged_hough_segments=merged_hough_segments,
        handoff_mode=hough_handoff_mode,
    )

    # Downstream report code works with line dictionaries, so convert the chosen
    # Hough segments into that stable record format before filtering.
    lines_for_filtering = lines_from_hough_segments(matrix_for_alignment, hough_segments_for_filtering)
    if matrix_for_alignment.size > 0:
        active_mask_bool = np.asarray(detector_payload.get("mask", np.zeros_like(matrix_for_alignment))) > 0
        lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
            lines_for_filtering,
            matrix_for_alignment,
            active_mask_bool,
            abs_min_len=float(align_abs_min_len),
            min_iou_threshold=float(align_min_iou_threshold),
        )
    else:
        n_other = matrix_for_alignment.shape[1] if matrix_for_alignment.ndim == 2 else 0
        lines_used = []
        column_assignment = _empty_column_assignment(n_other)

    return {
        "det": detector_payload,
        "raw_hough_segments": raw_hough_segments,
        "merged_hough_segments": merged_hough_segments,
        "hough_segments_for_filtering": hough_segments_for_filtering,
        "lines_for_filtering": lines_for_filtering,
        "lines_used": lines_used,
        "column_assignment": column_assignment,
        "hough_handoff_mode": str(hough_handoff_mode),
    }
