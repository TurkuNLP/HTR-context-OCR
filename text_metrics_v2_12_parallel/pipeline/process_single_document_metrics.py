"""Per-document metric processing for the text-metrics report pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from line_alignment_pipeline import detect_and_filter_lines_from_matrix
from levenshtein_metric import LEVENSHTEIN_BACKEND, compute_levenshtein_metrics_from_bundle
from line_coverage_subtract import (
    build_line_coverage_arrays_from_bundles,
    compute_line_coverage_ratio_metrics_from_arrays,
)
from line_metric_bundle import build_line_metric_bundle
from pipeline.load_or_compute_score_matrices import ItemScoreMatrixProvider
from pipeline.resolve_text_metrics_input_sources import (
    KIND_REF_TO_ADJUSTED_PRED,
    KIND_REF_TO_PRED,
    KIND_REF_TO_REF,
)
from runfile_records import safe_name


# Build the serializable per-line report diagnostics from filtered line objects.
def build_line_report(lines_used: list[dict]) -> list[dict]:
    """Build serializable line diagnostics from filtered line objects."""
    out = []
    for line_id, line in enumerate(lines_used):
        out.append(
            {
                "line_id": int(line_id),
                "x0": float(line.get("x0", 0.0)),
                "y0": float(line.get("y0", 0.0)),
                "x1": float(line.get("x1", 0.0)),
                "y1": float(line.get("y1", 0.0)),
                "score": float(line.get("score", 0.0)),
                "length": float(line.get("length", 0.0)),
                "support": float(line.get("support", 0.0)),
                "owned_cols": int(line.get("owned_cols", 0)),
                "owned_fraction": float(line.get("owned_fraction", 0.0)),
                "owned_score_mean": float(line.get("owned_score_mean", 0.0)),
                "owned_mask_hits": int(line.get("owned_mask_hits", 0)),
                "owned_mask_fraction": float(line.get("owned_mask_fraction", 0.0)),
                "anchor_y": float(line.get("anchor_y", min(line.get("y0", 0.0), line.get("y1", 0.0)))),
            }
        )
    return out


# Process one document end-to-end and return the report item payload.
def process_item(
    item: dict,
    args: argparse.Namespace,
    visual_output_dir: Path,
    *,
    score_index_by_kind: dict[str, dict[str, dict]],
    scores_pkl_paths_by_kind: dict[str, Path | None],
    visuals_enabled: bool = False,
    debug_enabled: bool = False,
) -> dict:
    """Process one document end-to-end and return the report item payload.

    Stage order remains intentionally unchanged:
    1) load/build matrices
    2) detect/filter lines and build bundles
    3) build coverage arrays
    4) optional visuals (lazy import)
    5) release matrix-heavy objects
    6) compute metrics from bundles and coverage arrays
    """
    if debug_enabled:
        from debug.per_document_stage_timing import (
            finish_debug_stage_timing,
            start_debug_stage_timing,
        )
    else:
        def start_debug_stage_timing() -> None:
            return None

        def finish_debug_stage_timing(
            timings: dict[str, float],
            *,
            key: str,
            start_time: float | None,
        ) -> None:
            return None

    timings: dict[str, float] = {}
    total_start = start_debug_stage_timing()

    pred_text = str(item["pred"])
    ref_text = str(item["ref"])

    matrix_provider = ItemScoreMatrixProvider(
        item=item,
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        score_index_by_kind=score_index_by_kind,
        scores_pkl_paths_by_kind=scores_pkl_paths_by_kind,
    )

    t0 = start_debug_stage_timing()
    ref_to_pred_matrix = matrix_provider.get_ref_to_pred_matrix()
    finish_debug_stage_timing(timings, key="matrix_ref_to_pred_s", start_time=t0)

    t0 = start_debug_stage_timing()
    ref_to_ref_matrix = matrix_provider.get_ref_to_ref_matrix()
    finish_debug_stage_timing(timings, key="matrix_ref_to_ref_s", start_time=t0)

    matrix_shape = [int(ref_to_pred_matrix.shape[0]), int(ref_to_pred_matrix.shape[1])]
    matrix_shape_ref_to_ref = [int(ref_to_ref_matrix.shape[0]), int(ref_to_ref_matrix.shape[1])]

    t0 = start_debug_stage_timing()
    pred_lines_payload = detect_and_filter_lines_from_matrix(
        ref_to_pred_matrix,
        item_index=int(item["index"]),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        hough_handoff_mode=str(args.hough_handoff_mode),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )
    finish_debug_stage_timing(timings, key="hough_filter_ref_to_pred_s", start_time=t0)

    detector_payload = pred_lines_payload["det"]
    raw_hough_segments = pred_lines_payload["raw_hough_segments"]
    merged_hough_segments = pred_lines_payload["merged_hough_segments"]
    lines_for_filtering = pred_lines_payload["lines_for_filtering"]
    lines_used = pred_lines_payload["lines_used"]
    column_assignment = pred_lines_payload["column_assignment"]

    raw_line_count = int(len(raw_hough_segments))
    merged_hough_line_count = int(len(merged_hough_segments))
    filter_input_line_count = int(len(lines_for_filtering))
    used_line_count = int(len(lines_used))

    n_ref_windows = int(ref_to_pred_matrix.shape[0]) if ref_to_pred_matrix.ndim == 2 else 0
    n_other_windows = int(ref_to_pred_matrix.shape[1]) if ref_to_pred_matrix.ndim == 2 else 0

    t0 = start_debug_stage_timing()
    bundle_ref_to_pred = build_line_metric_bundle(
        lines_used=lines_used,
        column_assignment=column_assignment,
        n_ref_windows=n_ref_windows,
        n_other_windows=n_other_windows,
        ref_text_len=len(ref_text),
        other_text_len=len(pred_text),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
    )
    finish_debug_stage_timing(timings, key="bundle_ref_to_pred_s", start_time=t0)

    t0 = start_debug_stage_timing()
    refref_lines_payload = detect_and_filter_lines_from_matrix(
        ref_to_ref_matrix,
        item_index=int(item["index"]),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        hough_handoff_mode=str(args.hough_handoff_mode),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
    )
    finish_debug_stage_timing(timings, key="hough_filter_ref_to_ref_s", start_time=t0)

    lines_used_ref_to_ref = refref_lines_payload["lines_used"]
    column_assignment_ref_to_ref = refref_lines_payload["column_assignment"]
    used_line_count_ref_to_ref = int(len(lines_used_ref_to_ref))

    n_ref_ref_windows = int(ref_to_ref_matrix.shape[0]) if ref_to_ref_matrix.ndim == 2 else 0
    n_other_ref_windows = int(ref_to_ref_matrix.shape[1]) if ref_to_ref_matrix.ndim == 2 else 0

    t0 = start_debug_stage_timing()
    bundle_ref_to_ref = build_line_metric_bundle(
        lines_used=lines_used_ref_to_ref,
        column_assignment=column_assignment_ref_to_ref,
        n_ref_windows=n_ref_ref_windows,
        n_other_windows=n_other_ref_windows,
        ref_text_len=len(ref_text),
        other_text_len=len(ref_text),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
    )
    finish_debug_stage_timing(timings, key="bundle_ref_to_ref_s", start_time=t0)

    t0 = start_debug_stage_timing()
    coverage_arrays = build_line_coverage_arrays_from_bundles(
        refref_bundle=bundle_ref_to_ref,
        other_bundle=bundle_ref_to_pred,
    )
    finish_debug_stage_timing(timings, key="coverage_arrays_from_bundles_s", start_time=t0)

    line_guided_columns = int(bundle_ref_to_pred.get("line_guided_columns", 0))
    fallback_columns = int(bundle_ref_to_pred.get("fallback_columns", 0))

    case_prefix = f"{item['index']:04d}_{safe_name(Path(item['fname']).name)}"
    matrix_source_adjusted = "not_needed"
    if visuals_enabled:
        t0 = start_debug_stage_timing()
        from visualisation.render_text_metrics_visualisations import save_text_metrics_visualisations

        matrix_after_reordering = matrix_provider.get_ref_to_adjusted_pred_matrix(pred_text)
        matrix_source_adjusted = matrix_provider.source_for(KIND_REF_TO_ADJUSTED_PRED)

        vis_paths = save_text_metrics_visualisations(
            matrix_before=ref_to_pred_matrix,
            raw_hough_segments=raw_hough_segments,
            pre_filter_lines=lines_for_filtering,
            filtered_lines=lines_used,
            matrix_after_reordering=matrix_after_reordering,
            case_prefix=case_prefix,
            file_name=Path(item["fname"]).name,
            output_dir=visual_output_dir,
            threshold_start=float(detector_payload.get("threshold_start", float("nan"))),
            line_filter_label=f"v2.12_true_IoU @ {float(args.align_min_iou_threshold):.3f}",
            coverage_refref_y=coverage_arrays["refref_y"],
            coverage_other_y=coverage_arrays["other_y"],
            coverage_other_x=coverage_arrays["other_x"],
            coverage_y_diff=coverage_arrays["y_diff"],
        )
        finish_debug_stage_timing(timings, key="visualisations_s", start_time=t0)
        del matrix_after_reordering
    else:
        vis_paths = {
            "visualise_before_hough_path": None,
            "visualise_after_hough_line_transform_path": None,
            "visualise_after_filtering_path": None,
            "visualise_after_reordering_path": None,
            "visualise_raw_hough_path": None,
            "visualise_after_v2_12_true_iou_path": None,
            "visualise_after_reorder_path": None,
            "visualise_full_path": None,
            "visualise_graph_path": None,
            "visualise_mask_path": None,
            "visualise_count_line_coverage_y_path": None,
            "visualise_count_line_coverage_x_path": None,
        }

    matrix_source_ref_to_pred = matrix_provider.source_for(KIND_REF_TO_PRED)
    matrix_source_ref_to_ref = matrix_provider.source_for(KIND_REF_TO_REF)

    del ref_to_pred_matrix
    del ref_to_ref_matrix
    del pred_lines_payload
    del refref_lines_payload

    t0 = start_debug_stage_timing()
    line_metric = compute_levenshtein_metrics_from_bundle(
        ref_text=ref_text,
        other_text=pred_text,
        lines_used=lines_used,
        bundle=bundle_ref_to_pred,
    )
    finish_debug_stage_timing(timings, key="levenshtein_metrics_s", start_time=t0)

    t0 = start_debug_stage_timing()
    line_coverage_metrics = compute_line_coverage_ratio_metrics_from_arrays(
        y_diff=coverage_arrays["y_diff"],
        other_x=coverage_arrays["other_x"],
        file_name=Path(item["fname"]).name,
    )
    finish_debug_stage_timing(timings, key="coverage_subtract_s", start_time=t0)

    before_nls = float(line_metric["whole_document_normalized_levenshtein_similarity"])
    along_lines_nls = line_metric.get("document_normalized_levenshtein_similarity_along_lines")
    weighted_along_lines_nls = line_metric.get(
        "document_weighted_normalized_levenshtein_similarity_along_lines"
    )
    after_nls = float(before_nls if along_lines_nls is None else along_lines_nls)

    adjusted_pred = pred_text
    del coverage_arrays

    if debug_enabled:
        finish_debug_stage_timing(timings, key="total_item_s", start_time=total_start)

    return {
        "index": int(item["index"]),
        "fname": Path(item["fname"]).name,
        "normalized_levenshtein_before": float(before_nls),
        "average_normalized_levenshtein_along_lines": None if along_lines_nls is None else float(along_lines_nls),
        "average_weighted_normalized_levenshtein_along_lines": (
            None if weighted_along_lines_nls is None else float(weighted_along_lines_nls)
        ),
        "correct_ref_coverage": float(line_coverage_metrics["correct_ref_coverage"]),
        "missing_ref_coverage": float(line_coverage_metrics["missing_ref_coverage"]),
        "repetition_on_ref": float(line_coverage_metrics["repetition_on_ref"]),
        "hallucination": float(line_coverage_metrics["hallucination"]),
        "adjusted_pred": adjusted_pred,
        "initial_matrix_source": matrix_source_ref_to_pred,
        "matrix_source_ref_to_pred": matrix_source_ref_to_pred,
        "matrix_source_ref_to_ref": matrix_source_ref_to_ref,
        "matrix_source_ref_to_adjusted_pred": matrix_source_adjusted,
        "matrix_shape": matrix_shape,
        "matrix_shape_ref_to_ref": matrix_shape_ref_to_ref,
        "visuals_enabled": bool(visuals_enabled),
        "before_normalized_levenshtein_similarity": float(before_nls),
        "after_normalized_levenshtein_similarity": float(after_nls),
        "delta": float(after_nls - before_nls),
        "whole_document_normalized_levenshtein_similarity": float(before_nls),
        "document_normalized_levenshtein_similarity_along_lines": along_lines_nls,
        "document_weighted_normalized_levenshtein_similarity_along_lines": weighted_along_lines_nls,
        "levenshtein_backend": LEVENSHTEIN_BACKEND,
        "line_metric_line_count": int(line_metric.get("line_count", 0)),
        "line_metric_lines": line_metric.get("lines", []),
        "line_coverage_metrics": line_coverage_metrics,
        "line_guided_columns": int(line_guided_columns),
        "fallback_columns": int(fallback_columns),
        "attached_between_columns": 0,
        "attached_between_runs": 0,
        "movable_components": 0,
        "raw_line_count": int(raw_line_count),
        "merged_line_count": int(filter_input_line_count),
        "merged_hough_line_count": int(merged_hough_line_count),
        "used_line_count": int(used_line_count),
        "used_line_count_ref_to_ref": int(used_line_count_ref_to_ref),
        "threshold_start": float(detector_payload.get("threshold_start", float("nan"))),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed) + int(item["index"]),
        "hough_start": float(args.hough_start),
        "hough_handoff_mode": str(args.hough_handoff_mode),
        "align_abs_min_len": float(args.align_abs_min_len),
        "line_filter_version": "v2_12_true_iou",
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "lines_used": build_line_report(lines_used),
        "__timing": timings if debug_enabled else {},
        **vis_paths,
    }
