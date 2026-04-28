"""Render all report visuals produced by the text-metrics pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from visualisation.render_alignment_matrix_views import (
    normalize_lines_for_labels,
    raw_segments_to_labeled_lines,
    save_matrix_visualisation,
)
from visualisation.render_line_coverage_views import save_count_line_coverage_visualisations


def save_text_metrics_visualisations(
    *,
    matrix_before: np.ndarray,
    raw_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    pre_filter_lines: list[dict],
    filtered_lines: list[dict],
    matrix_after_reordering: np.ndarray | None,
    case_prefix: str,
    file_name: str,
    output_dir: Path,
    threshold_start: float | None = None,
    line_filter_label: str = "v2.1_true_IoU",
    coverage_refref_y: np.ndarray | None = None,
    coverage_other_y: np.ndarray | None = None,
    coverage_other_x: np.ndarray | None = None,
    coverage_y_diff: np.ndarray | None = None,
) -> dict:
    """Save all pipeline visualisations for one document.

    Paths and keys are intentionally preserved so report consumers remain
    backward-compatible.
    """
    output_dir = Path(output_dir)
    vis_before_hough_dir = output_dir / "visualise_before_hough"
    vis_after_hough_dir = output_dir / "visualise_after_hough_line_transform"
    vis_after_filtering_dir = output_dir / "visualise_after_filtering"
    vis_after_reordering_dir = output_dir / "visualise_after_reordering"

    raw_hough_lines = raw_segments_to_labeled_lines(list(raw_hough_segments))
    normalized_filtered_lines = normalize_lines_for_labels(list(filtered_lines))

    before_path = save_matrix_visualisation(
        matrix=matrix_before,
        title=f"{file_name} | before Hough",
        out_path=vis_before_hough_dir / f"{case_prefix}_before_hough.png",
    )

    hough_suffix = ""
    if threshold_start is not None and np.isfinite(float(threshold_start)):
        hough_suffix = f" | start={float(threshold_start):.2f}"
    after_hough_path = save_matrix_visualisation(
        matrix=matrix_before,
        title=(
            f"{file_name} | after Hough line transform{hough_suffix} "
            f"| raw={len(raw_hough_lines)}"
        ),
        out_path=vis_after_hough_dir / f"{case_prefix}_after_hough_line_transform.png",
        lines=raw_hough_lines,
        line_color="red",
        line_width=2.0,
        line_alpha=1.0,
        show_labels=True,
        label_fontsize=8.0,
    )

    after_filtering_path = save_matrix_visualisation(
        matrix=matrix_before,
        title=(
            f"{file_name} | after filtering ({line_filter_label}) "
            f"| kept={len(normalized_filtered_lines)}/{len(pre_filter_lines)}"
        ),
        out_path=vis_after_filtering_dir / f"{case_prefix}_after_filtering.png",
        lines=normalized_filtered_lines,
        line_color="limegreen",
        line_width=2.4,
        line_alpha=1.0,
        show_labels=True,
        label_fontsize=8.0,
    )

    after_reordering_path = None
    if matrix_after_reordering is not None:
        # Keep this branch unchanged in behavior.
        after_reordering_path = save_matrix_visualisation(
            matrix=matrix_after_reordering,
            title=f"{file_name} | after reordering",
            out_path=vis_after_reordering_dir / f"{case_prefix}_after_reordering.png",
        )

    count_visuals = {
        "visualise_count_line_coverage_y_path": None,
        "visualise_count_line_coverage_x_path": None,
    }
    if (
        coverage_refref_y is not None
        and coverage_other_y is not None
        and coverage_other_x is not None
        and coverage_y_diff is not None
    ):
        count_visuals = save_count_line_coverage_visualisations(
            coverage_refref_y=np.asarray(coverage_refref_y, dtype=np.int32),
            coverage_other_y=np.asarray(coverage_other_y, dtype=np.int32),
            coverage_other_x=np.asarray(coverage_other_x, dtype=np.int32),
            coverage_y_diff=np.asarray(coverage_y_diff, dtype=np.int32),
            case_prefix=case_prefix,
            file_name=file_name,
            output_dir=output_dir,
        )

    return {
        "visualise_before_hough_path": before_path,
        "visualise_after_hough_line_transform_path": after_hough_path,
        "visualise_after_filtering_path": after_filtering_path,
        "visualise_after_reordering_path": after_reordering_path,
        "visualise_raw_hough_path": after_hough_path,
        "visualise_after_v2_1_true_iou_path": after_filtering_path,
        "visualise_after_reorder_path": after_reordering_path,
        "visualise_full_path": after_hough_path,
        "visualise_graph_path": after_hough_path,
        "visualise_mask_path": None,
        **count_visuals,
    }
