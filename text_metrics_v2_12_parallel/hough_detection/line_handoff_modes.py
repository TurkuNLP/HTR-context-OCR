"""Select which Hough-segment set is handed into the production true-IoU filter.

The refactor keeps two supported handoff modes under the same public entrypoint:

- ``merged_hough_to_true_iou``: compatibility behavior; raw Hough segments are
  postprocessed by ``merging_diag()`` first, then the merged segments are passed
  into the true-IoU filter.
- ``raw_hough_to_true_iou``: default behavior; raw Hough segments are
  passed directly into the exact same true-IoU filter.

Keeping both modes here gives the pipeline one explicit switch point instead of
spreading handoff choices across the detector and filter code.
"""

from __future__ import annotations

from typing import Literal

HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU = "merged_hough_to_true_iou"
HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU = "raw_hough_to_true_iou"
DEFAULT_HOUGH_HANDOFF_MODE = HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU

HoughHandoffMode = Literal[
    "merged_hough_to_true_iou",
    "raw_hough_to_true_iou",
]


def select_hough_segments_for_true_iou(
    *,
    raw_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    merged_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    handoff_mode: str,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return the exact Hough-segment set that should enter true-IoU filtering.

    Raw-Hough handoff is now the default.  The merged-Hough mode remains
    available as an explicit compatibility path for old apples-to-apples runs.
    """
    normalized_mode = str(handoff_mode)
    if normalized_mode == HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU:
        return list(merged_hough_segments)
    if normalized_mode == HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU:
        return list(raw_hough_segments)
    raise ValueError(
        "Unsupported hough handoff mode. "
        f"Got {handoff_mode!r}; expected one of "
        f"{HOUGH_HANDOFF_MODE_MERGED_TO_TRUE_IOU!r}, {HOUGH_HANDOFF_MODE_RAW_TO_TRUE_IOU!r}."
    )
