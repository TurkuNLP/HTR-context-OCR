import argparse, json, sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.transforms import Bbox

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from align_text_blocks_from_endpoints_no_pkl_v2 import (
    align_prediction,
    compute_score_matrix,
    filter_lines_for_alignment_by_ownership,
    lines_from_merged_segments,
    load_run_items,
    same_file,
    safe_name,
)
from hough_line_transform_endpoints_no_angle_all import detect_lines_dense_style_diagonal_fixed_theta
from evaluation.evaluate_page import levenshtein_distance


def normalized_levenshtein_similarity(predicted_text: str, gold_text: str) -> float:
    denom = max(len(predicted_text), len(gold_text))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(predicted_text, gold_text) / denom)



def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Align prediction text with fixed-diagonal probabilistic Hough lines "
            "computed in-memory from runfile JSON."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    p.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    p.add_argument("--window-size", type=int, default=100, help="Sliding window size")
    p.add_argument("--window-stride", type=int, default=50, help="Sliding window stride")
    p.add_argument(
        "--fallback-mode",
        type=str,
        default="argmax",
        choices=["argmax", "skip"],
        help="Fallback handling for columns not covered by any line.",
    )
    p.add_argument("--target-fname", type=str, default=None, help="Optional exact/basename target file")
    p.add_argument("--max-items", type=int, default=None, help="Optional maximum processed items")

    # Hough parameters (theta is fixed in Hough module by design).
    p.add_argument("--hough-threshold", type=int, default=26, help="Hough vote threshold")
    p.add_argument("--hough-line-length", type=int, default=10, help="Minimum accepted line length")
    p.add_argument("--hough-line-gap", type=int, default=15, help="Maximum gap to connect line pixels")
    p.add_argument("--hough-seed", type=int, default=0, help="Base random seed")
    p.add_argument("--hough-start", type=float, default=2.6, help="Initial adaptive threshold start before decrement loop")
    p.add_argument(
        "--align-min-len-ratio",
        type=float,
        default=0.08,
        help="Legacy global-length ratio from the previous filter (kept for compatibility; ownership filtering does not use it).",
    )
    p.add_argument(
        "--align-abs-min-len",
        type=float,
        default=8.0,
        help="Absolute minimum line length kept before ownership resolution.",
    )
    p.add_argument(
        "--align-mask-radius",
        type=int,
        default=1,
        help="Vertical mask search radius when checking whether a line is supported at a given x-column.",
    )
    p.add_argument(
        "--align-min-owned-cols",
        type=int,
        default=6,
        help="Minimum number of x-columns a line must own to survive ownership filtering.",
    )
    p.add_argument(
        "--align-min-owned-fraction",
        type=float,
        default=0.12,
        help="Minimum owned x-columns divided by a line's x-span for the line to survive ownership filtering.",
    )
    return p.parse_args()


def save_plain_matrix_visualisation(
    *,
    matrix: np.ndarray,
    title: str,
    out_path: Path,
) -> str:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys")
    plt.colorbar(im, ax=ax, label="chrF")
    ax.set_xlabel("pred segment")
    ax.set_ylabel("ref segment")
    ax.set_title(title)
    plt.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return str(out_path)


def save_hough_visualisations(
    *,
    matrix: np.ndarray,
    det: dict,
    case_prefix: str,
    file_name: str,
    vis_full_dir: Path,
    vis_graph_dir: Path,
    vis_mask_dir: Path,
) -> dict:
    vis_full_dir.mkdir(parents=True, exist_ok=True)
    vis_graph_dir.mkdir(parents=True, exist_ok=True)
    vis_mask_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="Greys")
    cbar = plt.colorbar(im, ax=ax, label="chrF")
    ax.set_xlabel("pred segment")
    ax.set_ylabel("ref segment")

    for p0, p1 in det.get("merged_lines", []):
        ax.plot((p0[0], p1[0]), (p0[1], p1[1]), color="red", linewidth=2)

    ax.set_title(
        f"{file_name} | start={det.get('threshold_start', float('nan')):.2f} "
        f"raw={len(det.get('raw_lines', []))} merged={len(det.get('merged_lines', []))}"
    )
    plt.tight_layout()

    full_out = vis_full_dir / f"{case_prefix}_full.png"
    graph_out = vis_graph_dir / f"{case_prefix}_graph.png"
    mask_out = vis_mask_dir / f"{case_prefix}_mask.png"

    fig.savefig(full_out, dpi=220, bbox_inches="tight", facecolor="white")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    graph_bbox = Bbox.union([ax.get_tightbbox(renderer), cbar.ax.get_tightbbox(renderer)]).transformed(
        fig.dpi_scale_trans.inverted()
    )
    fig.savefig(graph_out, dpi=260, bbox_inches=graph_bbox, facecolor="white")

    mask = det.get("mask", np.zeros_like(matrix))
    plt.imsave(mask_out, (mask > 0).astype(np.uint8) * 255, cmap="gray")
    plt.close(fig)

    return {
        "visualise_full_path": str(full_out),
        "visualise_graph_path": str(graph_out),
        "visualise_mask_path": str(mask_out),
    }


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
    vis_full_dir: Path,
    vis_graph_dir: Path,
    vis_mask_dir: Path,
    vis_before_hough_dir: Path,
    vis_after_reorder_dir: Path,
) -> dict:
    pred = item["pred"]
    ref = item["ref"]
    matrix = compute_score_matrix(ref, pred, window_size=args.window_size, window_stride=args.window_stride)

    if matrix.size > 0 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
        det = detect_lines_dense_style_diagonal_fixed_theta(
            matrix,
            seed=int(args.hough_seed) + int(item["index"]),
            threshold=int(args.hough_threshold),
            line_length=int(args.hough_line_length),
            line_gap=int(args.hough_line_gap),
            start_init=float(args.hough_start),
        )
        merged_lines = det.get("merged_lines", [])
    else:
        det = {
            "threshold_start": float("nan"),
            "mask": np.zeros_like(matrix),
            "raw_lines": [],
            "selected_lines": [],
            "merged_lines": [],
        }
        merged_lines = []

    lines_raw = lines_from_merged_segments(matrix, merged_lines)
    if matrix.size > 0:
        mask_bool = np.asarray(det.get("mask", np.zeros_like(matrix))) > 0
        lines_used, column_assignment = filter_lines_for_alignment_by_ownership(
            lines_raw,
            matrix,
            mask_bool,
            abs_min_len=float(args.align_abs_min_len),
            mask_radius=int(args.align_mask_radius),
            min_owned_cols=int(args.align_min_owned_cols),
            min_owned_fraction=float(args.align_min_owned_fraction),
        )
    else:
        lines_used = []
        column_assignment = {
            "mapped_y": np.full(matrix.shape[1] if matrix.ndim == 2 else 0, np.nan, dtype=float),
            "mapped_line_id": np.full(matrix.shape[1] if matrix.ndim == 2 else 0, -1, dtype=int),
        }

    aligned = align_prediction(
        pred_text=pred,
        matrix=matrix,
        lines_read_order=lines_used,
        window_stride=args.window_stride,
        fallback_mode=args.fallback_mode,
        column_assignment=column_assignment,
    )

    adjusted_pred = aligned["adjusted_pred"]
    before_nls = normalized_levenshtein_similarity(pred, ref)
    after_nls = normalized_levenshtein_similarity(adjusted_pred, ref)

    matrix_after = compute_score_matrix(
        ref,
        adjusted_pred,
        window_size=args.window_size,
        window_stride=args.window_stride,
    )

    case_prefix = f"{item['index']:04d}_{safe_name(Path(item['fname']).name)}"
    vis_paths = save_hough_visualisations(
        matrix=matrix,
        det=det,
        case_prefix=case_prefix,
        file_name=Path(item["fname"]).name,
        vis_full_dir=vis_full_dir,
        vis_graph_dir=vis_graph_dir,
        vis_mask_dir=vis_mask_dir,
    )
    before_graph_path = save_plain_matrix_visualisation(
        matrix=matrix,
        title=f"{Path(item['fname']).name} | before Hough",
        out_path=vis_before_hough_dir / f"{case_prefix}_before_hough.png",
    )
    after_graph_path = save_plain_matrix_visualisation(
        matrix=matrix_after,
        title=f"{Path(item['fname']).name} | after reordering",
        out_path=vis_after_reorder_dir / f"{case_prefix}_after_reorder.png",
    )

    return {
        "index": int(item["index"]),
        "fname": Path(item["fname"]).name,
        "adjusted_pred": adjusted_pred,
        "before_normalized_levenshtein_similarity": float(before_nls),
        "after_normalized_levenshtein_similarity": float(after_nls),
        "delta": float(after_nls - before_nls),
        "line_guided_columns": int(aligned["line_guided_columns"]),
        "fallback_columns": int(aligned["fallback_columns"]),
        "attached_between_columns": int(aligned.get("attached_between_columns", 0)),
        "attached_between_runs": int(aligned.get("attached_between_runs", 0)),
        "movable_components": int(aligned.get("movable_components", 0)),
        "raw_line_count": int(len(lines_raw)),
        "used_line_count": int(len(lines_used)),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed) + int(item["index"]),
        "hough_start": float(args.hough_start),
        "align_min_len_ratio": float(args.align_min_len_ratio),
        "align_abs_min_len": float(args.align_abs_min_len),
        "align_mask_radius": int(args.align_mask_radius),
        "align_min_owned_cols": int(args.align_min_owned_cols),
        "align_min_owned_fraction": float(args.align_min_owned_fraction),
        "matrix_shape": [int(matrix.shape[0]), int(matrix.shape[1])],
        "lines_used": _line_report(lines_used),
        "visualise_before_hough_path": before_graph_path,
        "visualise_after_reorder_path": after_graph_path,
        **vis_paths,
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
    if args.align_min_len_ratio <= 0:
        raise ValueError("align-min-len-ratio must be positive")
    if args.align_abs_min_len <= 0:
        raise ValueError("align-abs-min-len must be positive")
    if args.align_mask_radius < 0:
        raise ValueError("align-mask-radius must be non-negative")
    if args.align_min_owned_cols <= 0:
        raise ValueError("align-min-owned-cols must be positive")
    if not (0.0 <= args.align_min_owned_fraction <= 1.0):
        raise ValueError("align-min-owned-fraction must satisfy 0.0 <= value <= 1.0")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    vis_full_dir = args.output_dir / "visualise_full"
    vis_graph_dir = args.output_dir / "visualise_graph"
    vis_mask_dir = args.output_dir / "visualise_mask"
    vis_before_hough_dir = args.output_dir / "visualise_before_hough"
    vis_after_reorder_dir = args.output_dir / "visualise_after_reorder"

    run_items = load_run_items(args.runfile_json)
    rows = []
    matched = 0

    for item in run_items:
        if args.target_fname is not None and not same_file(item["fname"], args.target_fname):
            continue
        matched += 1

        if args.max_items is not None and len(rows) >= args.max_items:
            break

        res = process_item(
            item,
            args,
            vis_full_dir,
            vis_graph_dir,
            vis_mask_dir,
            vis_before_hough_dir,
            vis_after_reorder_dir,
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
            if k not in {"adjusted_pred", "lines_used"}
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
        raise KeyError(f"Target file not found in runfile JSON: {args.target_fname!r}")
    if not rows:
        raise RuntimeError("No items processed. Check target filters and inputs.")

    avg_before = sum(r["before_normalized_levenshtein_similarity"] for r in rows) / len(rows)
    avg_after = sum(r["after_normalized_levenshtein_similarity"] for r in rows) / len(rows)

    summary = {
        "count": int(len(rows)),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "fallback_mode": str(args.fallback_mode),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed),
        "hough_start": float(args.hough_start),
        "align_min_len_ratio": float(args.align_min_len_ratio),
        "align_abs_min_len": float(args.align_abs_min_len),
        "align_mask_radius": int(args.align_mask_radius),
        "align_min_owned_cols": int(args.align_min_owned_cols),
        "align_min_owned_fraction": float(args.align_min_owned_fraction),
        "average_before_normalized_levenshtein_similarity": float(avg_before),
        "average_after_normalized_levenshtein_similarity": float(avg_after),
        "average_delta": float(avg_after - avg_before),
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
