import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from align_text_blocks_from_endpoints_no_pkl_v2 import (
    build_pred_blocks,
    compute_score_matrix,
    lines_from_merged_segments,
    load_run_items,
    normalized_levenshtein_similarity,
    same_file,
    safe_name,
)
from line_filtering_v2_IoU import DEFAULT_ABS_MIN_LEN, filter_lines_for_alignment_by_ownership
from hough_line_transform_endpoints_no_angle_all import detect_lines_dense_style_diagonal_fixed_theta


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
    return p.parse_args()


def _empty_column_assignment(n_pred: int) -> dict[str, np.ndarray]:
    return {
        "mapped_y": np.full(n_pred, np.nan, dtype=float),
        "mapped_line_id": np.full(n_pred, -1, dtype=int),
    }


def _ordered_unique(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        if value in seen:
            continue
        out.append(int(value))
        seen.add(int(value))
    return out


def _is_non_decreasing(values: list[int]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def reference_rows_for_line(owned_cols: list[int], mapped_y: np.ndarray, n_ref: int) -> tuple[list[int], bool]:
    if n_ref <= 0:
        return [], False

    rows = [
        int(np.clip(round(float(mapped_y[x])), 0, n_ref - 1))
        for x in owned_cols
        if 0 <= int(x) < mapped_y.shape[0] and np.isfinite(mapped_y[x])
    ]
    if not rows:
        return [], False

    unique_rows = _ordered_unique(rows)
    if _is_non_decreasing(unique_rows):
        return unique_rows, False

    return sorted(set(unique_rows)), True


def build_line_similarity_reports(
    *,
    pred_text: str,
    ref_text: str,
    lines_used: list[dict],
    column_assignment: dict,
    window_stride: int,
    n_ref: int,
    n_pred: int,
) -> list[dict]:
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    if mapped_y.shape != (n_pred,) or mapped_line_id.shape != (n_pred,):
        raise ValueError(
            "column_assignment must provide mapped_y and mapped_line_id arrays with shape "
            f"({n_pred},), got {mapped_y.shape} and {mapped_line_id.shape}"
        )

    pred_blocks = build_pred_blocks(pred_text, n_pred=n_pred, stride=window_stride)
    ref_blocks = build_pred_blocks(ref_text, n_pred=n_ref, stride=window_stride)

    rows: list[dict] = []
    for lid, line in enumerate(lines_used):
        owned_cols = [int(x) for x in np.flatnonzero(mapped_line_id == lid)]
        if not owned_cols:
            continue

        ref_rows, ref_rows_reordered = reference_rows_for_line(owned_cols, mapped_y, n_ref=n_ref)
        pred_line_text = "".join(pred_blocks[x] for x in owned_cols if 0 <= x < len(pred_blocks))
        ref_line_text = "".join(ref_blocks[y] for y in ref_rows if 0 <= y < len(ref_blocks))
        score = normalized_levenshtein_similarity(pred_line_text, ref_line_text)

        rows.append(
            {
                "line_id": int(lid),
                "normalized_levenshtein_similarity": float(score),
                "pred_text": pred_line_text,
                "ref_text": ref_line_text,
                "pred_char_len": int(len(pred_line_text)),
                "ref_char_len": int(len(ref_line_text)),
                "owned_column_count": int(len(owned_cols)),
                "pred_column_start": int(owned_cols[0]),
                "pred_column_end": int(owned_cols[-1]),
                "mapped_ref_row_count": int(len(ref_rows)),
                "mapped_ref_row_start": None if not ref_rows else int(ref_rows[0]),
                "mapped_ref_row_end": None if not ref_rows else int(ref_rows[-1]),
                "mapped_ref_rows": ref_rows,
                "ref_rows_reordered_for_monotonicity": bool(ref_rows_reordered),
                "x0": float(line.get("x0", 0.0)),
                "y0": float(line.get("y0", 0.0)),
                "x1": float(line.get("x1", 0.0)),
                "y1": float(line.get("y1", 0.0)),
                "score": float(line.get("score", 0.0)),
                "length": float(line.get("length", 0.0)),
                "support": float(line.get("support", 0.0)),
                "owned_cols": int(line.get("owned_cols", len(owned_cols))),
                "owned_fraction": float(line.get("owned_fraction", 0.0)),
                "owned_score_mean": float(line.get("owned_score_mean", 0.0)),
                "owned_mask_hits": int(line.get("owned_mask_hits", 0)),
                "owned_mask_fraction": float(line.get("owned_mask_fraction", 0.0)),
                "anchor_y": float(line.get("anchor_y", min(line.get("y0", 0.0), line.get("y1", 0.0)))),
            }
        )

    return rows


def compute_levenshtein_along_lines(
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
    precomputed_matrix: np.ndarray | None = None,
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

    if matrix.size > 0 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        det = detect_lines_dense_style_diagonal_fixed_theta(
            matrix,
            seed=int(hough_seed) + int(item_index),
            threshold=int(hough_threshold),
            line_length=int(hough_line_length),
            line_gap=int(hough_line_gap),
            start_init=float(hough_start),
        )
        merged_lines = det.get("merged_lines", [])
        mask_bool = np.asarray(det.get("mask", np.zeros_like(matrix))) > 0
        lines_raw = lines_from_merged_segments(matrix, merged_lines)
        lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
            lines_raw,
            matrix,
            mask_bool,
        )
    else:
        det = {
            "threshold_start": float("nan"),
            "mask": np.zeros_like(matrix),
            "raw_lines": [],
            "selected_lines": [],
            "merged_lines": [],
        }
        lines_raw = []
        lines_used = []
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        column_assignment = _empty_column_assignment(n_pred)

    n_ref = int(matrix.shape[0]) if matrix.ndim == 2 else 0
    n_pred = int(matrix.shape[1]) if matrix.ndim == 2 else 0
    line_reports = build_line_similarity_reports(
        pred_text=pred_text,
        ref_text=ref_text,
        lines_used=lines_used,
        column_assignment=column_assignment,
        window_stride=window_stride,
        n_ref=n_ref,
        n_pred=n_pred,
    )

    line_scores = [row["normalized_levenshtein_similarity"] for row in line_reports]
    document_along_lines = None if not line_scores else float(sum(line_scores) / len(line_scores))
    mapped_line_id = np.asarray(column_assignment["mapped_line_id"], dtype=int)

    return {
        "whole_document_normalized_levenshtein_similarity": float(
            normalized_levenshtein_similarity(pred_text, ref_text)
        ),
        "document_normalized_levenshtein_similarity_along_lines": document_along_lines,
        "line_count": int(len(line_reports)),
        "raw_line_count": int(len(lines_raw)),
        "used_line_count": int(len(lines_used)),
        "line_guided_columns": int(np.sum(mapped_line_id >= 0)),
        "fallback_columns": int(np.sum(mapped_line_id < 0)),
        "matrix_shape": [n_ref, n_pred],
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "hough_threshold": int(hough_threshold),
        "hough_line_length": int(hough_line_length),
        "hough_line_gap": int(hough_line_gap),
        "hough_seed": int(hough_seed) + int(item_index),
        "hough_start": float(hough_start),
        "line_filter_version": "v2_iou",
        "line_filter_abs_min_len": float(DEFAULT_ABS_MIN_LEN),
        "lines": line_reports,
    }


def process_item(item: dict, args) -> dict:
    metrics = compute_levenshtein_along_lines(
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
        "line_filter_version": "v2_iou",
        "line_filter_abs_min_len": float(DEFAULT_ABS_MIN_LEN),
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
