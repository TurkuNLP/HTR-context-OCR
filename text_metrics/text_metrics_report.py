import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from align_text_blocks_from_endpoints_no_pkl_v2 import (
    compute_score_matrix,
    lines_from_merged_segments,
    load_run_items,
    same_file,
    safe_name,
)
from line_filtering_v2_1_IoU import (
    DEFAULT_MIN_IOU_THRESHOLD,
    filter_lines_for_alignment_by_ownership,
)
from hough_line_transform_endpoints_no_angle_all import detect_lines_dense_style_diagonal_fixed_theta
from visualise_used_lines_from_report import save_text_metrics_visualisations
from levenshtein_along_lines_metric import run_levenshtein_along_lines_metric
from line_coverage_subtract import compute_line_coverage_percentage_metrics_from_precomputed_endpoints


def coerce_score_matrix(scores, *, source_desc: str) -> np.ndarray:
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2:
        raise ValueError(f"Expected 2D score matrix in {source_desc}, got shape={mat.shape!r}")
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def load_score_items(scores_pkl: Path) -> list[dict]:
    out: list[dict] = []
    with open(scores_pkl, "rb") as f:
        idx = 0
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break

            if not isinstance(item, dict):
                idx += 1
                continue

            fname = str(item.get("fname", f"item_{idx:04d}"))
            out.append(
                {
                    "index": int(idx),
                    "fname": Path(fname).name,
                    "pred": str(item.get("pred", "")),
                    "ref": str(item.get("ref", "")),
                    "has_pred": "pred" in item,
                    "has_ref": "ref" in item,
                    "scores": coerce_score_matrix(item.get("scores"), source_desc=f"{scores_pkl}:{idx}:{fname}"),
                }
            )
            idx += 1
    return out


def build_score_lookup(score_items: list[dict]) -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for item in score_items:
        key = Path(str(item["fname"])).name
        if key in lookup:
            raise ValueError(f"Duplicate fname in scores.pkl stream: {key!r}")
        lookup[key] = item
    return lookup


def load_run_items_from_scores(score_items: list[dict], scores_pkl: Path) -> list[dict]:
    out: list[dict] = []
    for item in score_items:
        if not item.get("has_pred", False) or not item.get("has_ref", False):
            raise ValueError(
                f"scores.pkl item is missing pred/ref text and cannot replace runfile-json: {item['fname']!r} in {scores_pkl}"
            )
        out.append(
            {
                "index": int(item["index"]),
                "fname": Path(str(item["fname"])).name,
                "pred": str(item["pred"]),
                "ref": str(item["ref"]),
            }
        )
    return out



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Align prediction text with fixed-diagonal probabilistic Hough lines "
            "from either runfile JSON or precomputed scores.pkl matrices."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runfile-json", type=Path, default=None, help="Optional path to outputs.json")
    p.add_argument(
        "--scores-pkl",
        type=Path,
        default=None,
        help=(
            "Optional compare.py scores.pkl pickle-stream. If provided, the script reuses the precomputed "
            "initial chrF matrix. If used without --runfile-json, pred/ref/fname are also loaded from this file."
        ),
    )
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--window-size", type=int, default=100, help="Sliding window size")
    p.add_argument("--window-stride", type=int, default=50, help="Sliding window stride")
    p.add_argument("--target-fname", type=str, default=None, help="Optional exact/basename target file")
    p.add_argument("--max-items", type=int, default=None, help="Optional maximum processed items")
    p.add_argument(
        "--skip-visuals",
        dest="skip_visuals",
        action="store_true",
        default=True,
        help="Skip before-Hough, after-Hough-line-transform, after-filtering, and after-reordering PNG outputs.",
    )
    p.add_argument(
        "--with-visuals",
        dest="skip_visuals",
        action="store_false",
        help="Generate before-Hough, after-Hough-line-transform, after-filtering, and after-reordering PNG outputs.",
    )

    # Hough parameters (theta is fixed in Hough module by design).
    p.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold")
    p.add_argument("--hough-line-length", type=int, default=10, help="Minimum accepted line length")
    p.add_argument("--hough-line-gap", type=int, default=15, help="Maximum gap to connect line pixels")
    p.add_argument("--hough-seed", type=int, default=0, help="Base random seed")
    p.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start before decrement loop")
    p.add_argument(
        "--align-abs-min-len",
        type=float,
        default=8.0,
        help="Absolute minimum line length kept before ownership resolution.",
    )
    p.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=DEFAULT_MIN_IOU_THRESHOLD,
        help="Minimum true-IoU threshold used to merge overlapping line coverages in v2.1_true_IoU.",
    )
    return p.parse_args()


def _line_report(lines_used: list[dict]) -> list[dict]:
    out = []
    for lid, ln in enumerate(lines_used):
        out.append(
            {
                "line_id": int(lid),
                "x0": float(ln.get("x0", 0.0)),
                "y0": float(ln.get("y0", 0.0)),
                "x1": float(ln.get("x1", 0.0)),
                "y1": float(ln.get("y1", 0.0)),
                "score": float(ln.get("score", 0.0)),
                "length": float(ln.get("length", 0.0)),
                "support": float(ln.get("support", 0.0)),
                "owned_cols": int(ln.get("owned_cols", 0)),
                "owned_fraction": float(ln.get("owned_fraction", 0.0)),
                "owned_score_mean": float(ln.get("owned_score_mean", 0.0)),
                "owned_mask_hits": int(ln.get("owned_mask_hits", 0)),
                "owned_mask_fraction": float(ln.get("owned_mask_fraction", 0.0)),
                "anchor_y": float(ln.get("anchor_y", min(ln.get("y0", 0.0), ln.get("y1", 0.0)))),
            }
        )
    return out


def process_item(
    item: dict,
    args,
    visual_output_dir: Path,
    *,
    precomputed_matrix: np.ndarray | None = None,
    matrix_source: str = "computed",
    visuals_enabled: bool = False,
) -> dict:
    pred = item["pred"]
    ref = item["ref"]
    if precomputed_matrix is None:
        matrix = compute_score_matrix(ref, pred, window_size=args.window_size, window_stride=args.window_stride)
    else:
        matrix = np.asarray(precomputed_matrix, dtype=float)

    if matrix.size > 0 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        det = detect_lines_dense_style_diagonal_fixed_theta(
            matrix,
            seed=int(args.hough_seed) + int(item["index"]),
            threshold=int(args.hough_threshold),
            line_length=int(args.hough_line_length),
            line_gap=int(args.hough_line_gap),
            start_init=float(args.hough_start),
        )
        raw_hough_segments = list(det.get("raw_lines", []))
        merged_lines = det.get("merged_lines", [])
    else:
        det = {
            "threshold_start": float("nan"),
            "mask": np.zeros_like(matrix),
            "raw_lines": [],
            "selected_lines": [],
            "merged_lines": [],
        }
        raw_hough_segments = []
        merged_lines = []

    lines_for_filtering = lines_from_merged_segments(matrix, merged_lines)
    if matrix.size > 0:
        mask_bool = np.asarray(det.get("mask", np.zeros_like(matrix))) > 0
        lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
            lines_for_filtering,
            matrix,
            mask_bool,
            abs_min_len=float(args.align_abs_min_len),
            min_iou_threshold=float(args.align_min_iou_threshold),
        )
    else:
        lines_used = []
        column_assignment = {
            "mapped_y": np.full(matrix.shape[1] if matrix.ndim == 2 else 0, np.nan, dtype=float),
            "mapped_line_id": np.full(matrix.shape[1] if matrix.ndim == 2 else 0, -1, dtype=int),
        }

    # Legacy alignment path (disabled by request; kept for reference).
    # aligned = align_prediction(
    #     pred_text=pred,
    #     matrix=matrix,
    #     lines_read_order=lines_used,
    #     window_stride=args.window_stride,
    #     column_assignment=column_assignment,
    # )
    # adjusted_pred = aligned["adjusted_pred"]
    # before_nls = normalized_levenshtein_similarity(pred, ref)
    # after_nls = normalized_levenshtein_similarity(adjusted_pred, ref)

    line_metric = run_levenshtein_along_lines_metric(
        pred_text=pred,
        ref_text=ref,
        item_index=int(item["index"]),
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        hough_threshold=int(args.hough_threshold),
        hough_line_length=int(args.hough_line_length),
        hough_line_gap=int(args.hough_line_gap),
        hough_seed=int(args.hough_seed),
        hough_start=float(args.hough_start),
        align_abs_min_len=float(args.align_abs_min_len),
        align_min_iou_threshold=float(args.align_min_iou_threshold),
        precomputed_matrix=matrix,
    )

    adjusted_pred = pred
    before_nls = float(line_metric["whole_document_normalized_levenshtein_similarity"])
    along_lines_nls = line_metric.get("document_normalized_levenshtein_similarity_along_lines")
    after_nls = float(before_nls if along_lines_nls is None else along_lines_nls)

    case_prefix = f"{item['index']:04d}_{safe_name(Path(item['fname']).name)}"
    line_coverage_metrics = compute_line_coverage_percentage_metrics_from_precomputed_endpoints(
        ref_text=ref,
        other_text=adjusted_pred,
        refref_line_endpoints=lines_used,
        other_line_endpoints=lines_used,
        window_size=int(args.window_size),
        window_stride=int(args.window_stride),
        strict_lines=False,
        file_name=Path(item["fname"]).name,
    )

    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    line_guided_columns = int(np.sum(mapped_line_id >= 0))
    fallback_columns = int(np.sum(mapped_line_id < 0))
    if visuals_enabled:
        matrix_after_reordering = compute_score_matrix(
            ref,
            adjusted_pred,
            window_size=args.window_size,
            window_stride=args.window_stride,
        )
        vis_paths = save_text_metrics_visualisations(
            matrix_before=matrix,
            raw_hough_segments=raw_hough_segments,
            pre_filter_lines=lines_for_filtering,
            filtered_lines=lines_used,
            matrix_after_reordering=matrix_after_reordering,
            case_prefix=case_prefix,
            file_name=Path(item["fname"]).name,
            output_dir=visual_output_dir,
            threshold_start=float(det.get("threshold_start", float("nan"))),
            line_filter_label=f"v2.1_true_IoU @ {float(args.align_min_iou_threshold):.3f}",
        )
    else:
        vis_paths = {
            "visualise_before_hough_path": None,
            "visualise_after_hough_line_transform_path": None,
            "visualise_after_filtering_path": None,
            "visualise_after_reordering_path": None,
            "visualise_raw_hough_path": None,
            "visualise_after_v2_1_true_iou_path": None,
            "visualise_after_reorder_path": None,
            "visualise_full_path": None,
            "visualise_graph_path": None,
            "visualise_mask_path": None,
        }

    return {
        "index": int(item["index"]),
        "fname": Path(item["fname"]).name,
        "adjusted_pred": adjusted_pred,
        "initial_matrix_source": str(matrix_source),
        "visuals_enabled": bool(visuals_enabled),
        "before_normalized_levenshtein_similarity": float(before_nls),
        "after_normalized_levenshtein_similarity": float(after_nls),
        "delta": float(after_nls - before_nls),
        "whole_document_normalized_levenshtein_similarity": float(before_nls),
        "document_normalized_levenshtein_similarity_along_lines": along_lines_nls,
        "line_metric_line_count": int(line_metric.get("line_count", 0)),
        "line_metric_lines": line_metric.get("lines", []),
        "line_coverage_metrics": line_coverage_metrics,
        "missing_percent": float(line_coverage_metrics["missing_percent"]),
        "ok_percent": float(line_coverage_metrics["ok_percent"]),
        "repetition_percent": float(line_coverage_metrics["repetition_percent"]),
        "hallucination_percent": float(line_coverage_metrics["hallucination_percent"]),
        "line_guided_columns": int(line_guided_columns),
        "fallback_columns": int(fallback_columns),
        "attached_between_columns": 0,
        "attached_between_runs": 0,
        "movable_components": 0,
        "raw_line_count": int(len(raw_hough_segments)),
        "merged_line_count": int(len(lines_for_filtering)),
        "used_line_count": int(len(lines_used)),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed) + int(item["index"]),
        "hough_start": float(args.hough_start),
        "align_abs_min_len": float(args.align_abs_min_len),
        "line_filter_version": "v2_1_true_iou",
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "lines_used": _line_report(lines_used),
        **vis_paths,
    }



def main():
    args = parse_args()

    if args.runfile_json is None and args.scores_pkl is None:
        raise ValueError("Provide at least one input source: --runfile-json or --scores-pkl")
    if args.runfile_json is not None and not args.runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
    if args.scores_pkl is not None and not args.scores_pkl.exists():
        raise FileNotFoundError(f"Missing scores file: {args.scores_pkl}")
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")
    if args.hough_threshold <= 0:
        raise ValueError("hough-threshold must be positive")
    if args.hough_line_length <= 0:
        raise ValueError("hough-line-length must be positive")
    if args.hough_line_gap < 0:
        raise ValueError("hough-line-gap must be non-negative")
    if args.hough_start <= 0:
        raise ValueError("hough-start must be positive")
    if args.align_abs_min_len <= 0:
        raise ValueError("align-abs-min-len must be positive")
    if not (0.0 <= args.align_min_iou_threshold <= 1.0):
        raise ValueError("align-min-iou-threshold must satisfy 0.0 <= value <= 1.0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_enabled = not bool(args.skip_visuals)
    visual_output_dir = args.output_dir

    score_items = load_score_items(args.scores_pkl) if args.scores_pkl is not None else []
    score_lookup = build_score_lookup(score_items) if score_items else {}

    if args.runfile_json is not None:
        run_items = load_run_items(args.runfile_json)
    else:
        run_items = load_run_items_from_scores(score_items, args.scores_pkl)

    rows = []
    matched = 0

    for item in run_items:
        if args.target_fname is not None and not same_file(item["fname"], args.target_fname):
            continue
        matched += 1

        if args.max_items is not None and len(rows) >= args.max_items:
            break

        score_item = score_lookup.get(Path(item["fname"]).name)
        precomputed_matrix = None
        matrix_source = "computed"
        if score_item is not None:
            if args.runfile_json is not None:
                if score_item.get("has_pred", False) and str(score_item.get("pred", "")) != str(item["pred"]):
                    raise ValueError(f"Prediction text mismatch between runfile-json and scores.pkl for {item['fname']!r}")
                if score_item.get("has_ref", False) and str(score_item.get("ref", "")) != str(item["ref"]):
                    raise ValueError(f"Reference text mismatch between runfile-json and scores.pkl for {item['fname']!r}")
            precomputed_matrix = score_item["scores"]
            matrix_source = "scores_pkl"
        elif args.scores_pkl is not None:
            raise KeyError(f"No scores.pkl entry found for fname={item['fname']!r} in {args.scores_pkl}")

        res = process_item(
            item,
            args,
            visual_output_dir,
            precomputed_matrix=precomputed_matrix,
            matrix_source=matrix_source,
            visuals_enabled=visuals_enabled,
        )

        case_prefix = f"{res['index']:04d}_{safe_name(Path(res['fname']).name)}"
        out_pred = args.output_dir / f"{case_prefix}_adjusted_pred.txt"
        out_report = args.output_dir / f"{case_prefix}_report.json"
        out_pred.write_text(res["adjusted_pred"], encoding="utf-8")

        report = dict(res)
        report["adjusted_pred_path"] = str(out_pred)
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {
            k: v
            for k, v in report.items()
            if k not in {"adjusted_pred", "lines_used", "line_metric_lines"}
        }
        row["report_path"] = str(out_report)
        rows.append(row)

        print(
            f"[{len(rows)}] {row['fname']} | before={row['before_normalized_levenshtein_similarity']:.6f} "
            f"after={row['after_normalized_levenshtein_similarity']:.6f} delta={row['delta']:+.6f} | "
            f"hough(th={row['hough_threshold']}, len={row['hough_line_length']}, "
            f"gap={row['hough_line_gap']}, seed={row['hough_seed']}, start={row['hough_start']:.2f})"
        )

    if args.target_fname is not None and matched == 0:
        raise KeyError(f"Target file not found in provided input items: {args.target_fname!r}")
    if not rows:
        raise RuntimeError("No items processed. Check target filters and inputs.")

    avg_before = sum(r["before_normalized_levenshtein_similarity"] for r in rows) / len(rows)
    avg_after = sum(r["after_normalized_levenshtein_similarity"] for r in rows) / len(rows)
    avg_missing_percent = sum(r["missing_percent"] for r in rows) / len(rows)
    avg_ok_percent = sum(r["ok_percent"] for r in rows) / len(rows)
    avg_repetition_percent = sum(r["repetition_percent"] for r in rows) / len(rows)
    avg_hallucination_percent = sum(r["hallucination_percent"] for r in rows) / len(rows)

    summary = {
        "count": int(len(rows)),
        "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
        "scores_pkl": None if args.scores_pkl is None else str(args.scores_pkl),
        "visuals_enabled": bool(visuals_enabled),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed),
        "hough_start": float(args.hough_start),
        "align_abs_min_len": float(args.align_abs_min_len),
        "line_filter_version": "v2_1_true_iou",
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "average_before_normalized_levenshtein_similarity": float(avg_before),
        "average_after_normalized_levenshtein_similarity": float(avg_after),
        "average_delta": float(avg_after - avg_before),
        "average_missing_percent": float(avg_missing_percent),
        "average_ok_percent": float(avg_ok_percent),
        "average_repetition_percent": float(avg_repetition_percent),
        "average_hallucination_percent": float(avg_hallucination_percent),
        "items": rows,
    }

    out_summary = args.output_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Processed items: {len(rows)}")
    print(f"Average before: {avg_before:.6f}")
    print(f"Average after:  {avg_after:.6f}")
    print(f"Average delta:  {avg_after - avg_before:+.6f}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()