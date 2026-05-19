from __future__ import annotations

import numpy as np

from hough_line_transform_endpoints_line_direction_30_to_60_degrees import (
    detect_lines_dense_style_diagonal_fixed_theta,
)
from line_endpoint_records import lines_from_merged_segments
from line_filtering_v2_1_IoU import filter_lines_for_alignment_by_ownership
from score_matrix_builder import coerce_score_matrix


def _empty_column_assignment(n_other: int) -> dict[str, np.ndarray]:
    """Return the empty fallback column assignment structure for one matrix.

    The report pipeline expects the assignment arrays to exist even when no line
    survives filtering, so this helper keeps that shape stable.
    """
    return {
        "mapped_y": np.full(int(n_other), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_other), -1, dtype=int),
    }



def detect_and_filter_lines_from_matrix(
    matrix: np.ndarray,
    *,
    item_index: int,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    hough_start: float,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> dict:
    """Run Hough detection and IoU-based line filtering for one score matrix.

    This is the only line-alignment entry point used by
    ``run_text_metrics_report.sh``. The detector uses the fixed line-direction
    angle range encoded by the Hough module, then forwards the merged segments
    into the ownership-based v2.1 IoU filter.
    """
    mat = coerce_score_matrix(matrix, source_desc="line_alignment_pipeline:matrix")

    if mat.size > 0 and mat.shape[0] > 0 and mat.shape[1] > 0:
        det = detect_lines_dense_style_diagonal_fixed_theta(
            mat,
            seed=int(hough_seed) + int(item_index),
            threshold=int(hough_threshold),
            line_length=int(hough_line_length),
            line_gap=int(hough_line_gap),
            start_init=float(hough_start),
        )
        raw_hough_segments = list(det.get("raw_lines", []))
        merged_lines = list(det.get("merged_lines", []))
    else:
        det = {
            "threshold_start": float("nan"),
            "mask": np.zeros_like(mat),
            "raw_lines": [],
            "selected_lines": [],
            "merged_lines": [],
        }
        raw_hough_segments = []
        merged_lines = []

    # Downstream report code works with line dictionaries, so convert the merged
    # Hough segments into that stable record format before filtering.
    lines_for_filtering = lines_from_merged_segments(mat, merged_lines)
    if mat.size > 0:
        mask_bool = np.asarray(det.get("mask", np.zeros_like(mat))) > 0
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
        "det": det,
        "raw_hough_segments": raw_hough_segments,
        "lines_for_filtering": lines_for_filtering,
        "lines_used": lines_used,
        "column_assignment": column_assignment,
    }
