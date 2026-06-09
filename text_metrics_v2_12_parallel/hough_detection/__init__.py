"""Hough-detection helpers shared by the line-alignment pipeline."""

from .line_handoff_modes import (
    DEFAULT_HOUGH_HANDOFF_MODE,
    HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU,
    HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU,
    HoughHandoffMode,
    select_hough_segments_for_true_iou,
)

__all__ = [
    "DEFAULT_HOUGH_HANDOFF_MODE",
    "HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU",
    "HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU",
    "HoughHandoffMode",
    "select_hough_segments_for_true_iou",
]
