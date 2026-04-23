import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from line_alignment_pipeline import detect_and_filter_lines_from_matrix
from levenshtein_metric import BACKEND_PYTHON, SUPPORTED_BACKENDS, compute_levenshtein_metrics
from line_filtering_v2_1_IoU import (
    DEFAULT_ABS_MIN_LEN,
    DEFAULT_MIN_IOU_THRESHOLD,
)
from runfile_records import load_run_items, same_file, safe_name
from score_matrix_builder import compute_score_matrix

__all__ = ["run_levenshtein_along_lines_metric"]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute normalized Levenshtein similarity along coverage-merged Hough lines. "
            "For each recognized line, prediction and reference text are assembled from the "
            "final line-guided stride blocks, and the document score is the mean of those line scores."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--window-size", type=int, default=100, help="Sliding window size")
    p.add_argument("--window-stride", type=int, default=50, help="Sliding window stride")
    p.add_argument("--target-fname", type=str, default=None, help="Optional exact/basename target file")
    p.add_argument("--max-items", type=int, default=None, help="Optional maximum processed items")
    p.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold")
    p.add_argument("--hough-line-length", type=int, default=10, help="Minimum accepted line length")
    p.add_argument("--hough-line-gap", type=int, default=15, help="Maximum gap to connect line pixels")
    p.add_argument("--hough-seed", type=int, default=0, help="Base random seed")
    p.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start")
    p.add_argument(
        "--align-abs-min-len",
        type=float,
        default=DEFAULT_ABS_MIN_LEN,
        help="Absolute minimum line length kept before ownership resolution.",
    )
    p.add_argument(
        "--align-min-iou-threshold",
        type=float,
        default=DEFAULT_MIN_IOU_THRESHOLD,
        help="Minimum true-IoU threshold used to merge overlapping line coverages.",
    )
    p.add_argument(
        "--levenshtein-backend",
        type=str,
        default=BACKEND_PYTHON,
        choices=tuple(SUPPORTED_BACKENDS),
        help="Levenshtein backend. 'python' keeps current implementation; 'c' uses exact C-backed distance.",
    )
    return p.parse_args()


def _compute_levenshtein_along_lines_metric(
    *,
    pred_text: str,
    ref_text: str,
    item_index: int,
    window_size: int,
    window_stride: int,
    hough_threshold: int = 26,
    hough_line_length: int = 10,
    hough_line_gap: int = 15,
    hough_seed: int = 0,
    hough_start: float = 2.6,
    align_abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    align_min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    precomputed_matrix: np.ndarray | None = None,
    levenshtein_backend: str = BACKEND_PYTHON,
) -> dict:
    if precomputed_matrix is None:
        matrix = compute_score_matrix(
            ref_text,
            pred_text,
            window_size=window_size,
            window_stride=window_stride,
        )
    else:
        matrix = np.asarray(precomputed_matrix, dtype=float)
        if matrix.ndim != 2:
            raise ValueError(f"Expected 2D matrix, got shape={matrix.shape!r}")
        matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    payload = detect_and_filter_lines_from_matrix(
        matrix,
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
    lines_raw = payload["lines_for_filtering"]
    lines_used = payload["lines_used"]
    column_assignment = payload["column_assignment"]

    n_ref = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    n_pred = int(matrix.shape[1]) if matrix.ndim == 2 else 0

    lev = compute_levenshtein_metrics(
        ref_text=ref_text,
        other_text=pred_text,
        lines_used=lines_used,
        column_assignment=column_assignment,
        n_ref=n_ref,
        n_other=n_pred,
        window_stride=window_stride,
        backend=levenshtein_backend,
    )

    return {
        "whole_document_normalized_levenshtein_similarity": float(
            lev["whole_document_normalized_levenshtein_similarity"]
        ),
        "document_normalized_levenshtein_similarity_along_lines": lev[
            "document_normalized_levenshtein_similarity_along_lines"
        ],
        "line_count": int(lev["line_count"]),
        "raw_line_count": int(len(lines_raw)),
        "used_line_count": int(len(lines_used)),
        "line_guided_columns": int(lev["line_guided_columns"]),
        "fallback_columns": int(lev["fallback_columns"]),
        "matrix_shape": [n_ref, n_pred],
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "hough_threshold": int(hough_threshold),
        "hough_line_length": int(hough_line_length),
        "hough_line_gap": int(hough_line_gap),
        "hough_seed": int(hough_seed) + int(item_index),
        "hough_start": float(hough_start),
        "line_filter_version": "v2_1_true_iou",
        "line_filter_abs_min_len": float(align_abs_min_len),
        "line_filter_min_iou_threshold": float(align_min_iou_threshold),
        "levenshtein_backend": str(levenshtein_backend),
        "lines": lev["lines"],
    }


def run_levenshtein_along_lines_metric(
    *,
    pred_text: str,
    ref_text: str,
    item_index: int,
    window_size: int,
    window_stride: int,
    hough_threshold: int = 26,
    hough_line_length: int = 10,
    hough_line_gap: int = 15,
    hough_seed: int = 0,
    hough_start: float = 2.6,
    align_abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    align_min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    precomputed_matrix: np.ndarray | None = None,
    levenshtein_backend: str = BACKEND_PYTHON,
) -> dict:
    return _compute_levenshtein_along_lines_metric(
        pred_text=pred_text,
        ref_text=ref_text,
        item_index=item_index,
        window_size=window_size,
        window_stride=window_stride,
        hough_threshold=hough_threshold,
        hough_line_length=hough_line_length,
        hough_line_gap=hough_line_gap,
        hough_seed=hough_seed,
        hough_start=hough_start,
        align_abs_min_len=align_abs_min_len,
        align_min_iou_threshold=align_min_iou_threshold,
        precomputed_matrix=precomputed_matrix,
        levenshtein_backend=levenshtein_backend,
    )


def process_item(item: dict, args) -> dict:
    metrics = run_levenshtein_along_lines_metric(
        pred_text=item["pred"],
        ref_text=item["ref"],
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
        levenshtein_backend=str(args.levenshtein_backend),
    )
    return {
        "index": int(item["index"]),
        "fname": Path(item["fname"]).name,
        **metrics,
    }


def main():
    args = parse_args()
    if not args.runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
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
    if str(args.levenshtein_backend) not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported Levenshtein backend {args.levenshtein_backend!r}; expected one of {SUPPORTED_BACKENDS!r}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_items = load_run_items(args.runfile_json)

    rows = []
    matched = 0
    for item in run_items:
        if args.target_fname is not None and not same_file(item["fname"], args.target_fname):
            continue
        matched += 1

        if args.max_items is not None and len(rows) >= args.max_items:
            break

        report = process_item(item, args)
        case_prefix = f"{report['index']:04d}_{safe_name(Path(report['fname']).name)}"
        out_report = args.output_dir / f"{case_prefix}_report.json"
        out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

        row = {k: v for k, v in report.items() if k != "lines"}
        row["report_path"] = str(out_report)
        rows.append(row)

        along_lines = report["document_normalized_levenshtein_similarity_along_lines"]
        along_lines_str = "n/a" if along_lines is None else f"{along_lines:.6f}"
        print(
            f"[{len(rows)}] {report['fname']} | whole={report['whole_document_normalized_levenshtein_similarity']:.6f} "
            f"along_lines={along_lines_str} | lines={report['line_count']} "
            f"guided={report['line_guided_columns']} fallback={report['fallback_columns']}"
        )

    if args.target_fname is not None and matched == 0:
        raise KeyError(f"Target file not found in provided input items: {args.target_fname!r}")
    if not rows:
        raise RuntimeError("No items processed. Check target filters and inputs.")

    avg_whole = sum(r["whole_document_normalized_levenshtein_similarity"] for r in rows) / len(rows)
    along_line_scores = [
        r["document_normalized_levenshtein_similarity_along_lines"]
        for r in rows
        if r["document_normalized_levenshtein_similarity_along_lines"] is not None
    ]
    avg_along_lines = None if not along_line_scores else float(sum(along_line_scores) / len(along_line_scores))

    summary = {
        "count": int(len(rows)),
        "documents_with_line_metric": int(len(along_line_scores)),
        "runfile_json": str(args.runfile_json),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed),
        "hough_start": float(args.hough_start),
        "line_filter_version": "v2_1_true_iou",
        "line_filter_abs_min_len": float(args.align_abs_min_len),
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "levenshtein_backend": str(args.levenshtein_backend),
        "average_whole_document_normalized_levenshtein_similarity": float(avg_whole),
        "average_document_normalized_levenshtein_similarity_along_lines": avg_along_lines,
        "items": rows,
    }

    out_summary = args.output_dir / "summary.json"
    out_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print(f"Processed items: {len(rows)}")
    print(f"Average whole document similarity: {avg_whole:.6f}")
    if avg_along_lines is None:
        print("Average along-lines similarity: n/a")
    else:
        print(f"Average along-lines similarity: {avg_along_lines:.6f}")
    print(f"Summary: {out_summary}")


if __name__ == "__main__":
    main()
