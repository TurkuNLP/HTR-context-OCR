import argparse
import csv
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import align_text_blocks_from_endpoints_no_pkl_v2 as align_v2


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect and visualize how alignment reorders prediction columns "
            "(src_x -> dst_x), including line-guided and fallback behavior."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    parser.add_argument("--target-fname", type=str, required=True, help="Exact/basename target file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for plots/tables")
    parser.add_argument("--window-size", type=int, default=60, help="Sliding window size")
    parser.add_argument("--window-stride", type=int, default=40, help="Sliding window stride")
    parser.add_argument(
        "--fallback-mode",
        choices=["argmax", "skip"],
        default="argmax",
        help="Fallback behavior for columns not covered by any kept line",
    )
    parser.add_argument("--hough-seed", type=int, default=0, help="Seed for Hough line detection")
    parser.add_argument("--y-band-min", type=int, default=185, help="Y-band start for focused inspection")
    parser.add_argument("--y-band-max", type=int, default=210, help="Y-band end for focused inspection")
    parser.add_argument("--max-band-rows", type=int, default=400, help="Max rows in y-band snippets CSV")
    return parser.parse_args()


def load_target_item(runfile_json: Path, target_fname: str) -> dict:
    for item in align_v2.load_run_items(runfile_json):
        if align_v2.same_file(item["fname"], target_fname):
            return item
    raise KeyError(f"Target file not found in runfile JSON: {target_fname!r}")


def build_blocks_and_spans(pred_text: str, n_pred: int, stride: int):
    blocks = align_v2.build_pred_blocks(pred_text, n_pred=n_pred, stride=stride)
    src_spans = []
    for j, block in enumerate(blocks):
        start = min(j * stride, len(pred_text))
        src_spans.append((start, start + len(block)))
    return blocks, src_spans


def compute_mapping_debug(
    *,
    pred_text: str,
    matrix: np.ndarray,
    lines: list[dict],
    fallback_mode: str,
    window_stride: int,
    column_assignment: dict | None = None,
):
    n_ref, n_pred = matrix.shape
    lines_sorted = sorted(lines, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))
    if column_assignment is not None:
        mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float).copy()
        mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int).copy()
        if mapped_y.shape != (n_pred,) or mapped_line_id.shape != (n_pred,):
            raise ValueError(
                "column_assignment must provide mapped_y and mapped_line_id arrays with shape "
                f"({n_pred},), got {mapped_y.shape} and {mapped_line_id.shape}"
            )
    else:
        mapped_y = np.full(n_pred, np.nan, dtype=float)
        mapped_line_id = np.full(n_pred, -1, dtype=int)

    overlap_rows = []

    for x in range(n_pred):
        best_score = float("-inf")
        best_y_idx = -1
        best_lid = -1
        candidates = []

        for lid, ln in enumerate(lines_sorted):
            x_min = int(math.floor(min(ln["x0"], ln["x1"])))
            x_max = int(math.ceil(max(ln["x0"], ln["x1"])))
            if x < x_min or x > x_max:
                continue

            y_est = align_v2.line_y_at_x(ln, x)
            y_idx = int(np.clip(round(y_est), 0, n_ref - 1))
            score = float(matrix[y_idx, x])
            candidates.append((lid, y_idx, score))

            if (
                score > best_score
                or (math.isclose(score, best_score) and (best_y_idx < 0 or y_idx < best_y_idx))
                or (math.isclose(score, best_score) and y_idx == best_y_idx and (best_lid < 0 or lid < best_lid))
            ):
                best_score = score
                best_y_idx = y_idx
                best_lid = lid

        if column_assignment is None and best_lid >= 0:
            mapped_y[x] = float(best_y_idx)
            mapped_line_id[x] = best_lid
        elif column_assignment is None and fallback_mode == "argmax":
            mapped_y[x] = float(int(np.argmax(matrix[:, x])) if n_ref > 0 else 0)

        if len(candidates) > 1:
            winner_lid = int(mapped_line_id[x]) if int(mapped_line_id[x]) >= 0 else best_lid
            for lid, y_idx, score in candidates:
                overlap_rows.append(
                    {
                        "x_src_col": x,
                        "candidate_count": len(candidates),
                        "candidate_line_id": lid,
                        "candidate_y_idx": y_idx,
                        "candidate_score": score,
                        "is_winner": int(lid == winner_lid),
                    }
                )

    if fallback_mode == "argmax":
        uncovered = ~np.isfinite(mapped_y)
        if np.any(uncovered):
            for x in np.flatnonzero(uncovered):
                mapped_y[x] = float(int(np.argmax(matrix[:, x])) if n_ref > 0 else 0)

    aligned = align_v2.align_prediction(
        pred_text=pred_text,
        matrix=matrix,
        lines_read_order=lines_sorted,
        window_stride=window_stride,
        fallback_mode=fallback_mode,
        column_assignment=column_assignment,
    )
    order = [int(src) for src in aligned["order"]]

    assert len(order) == n_pred
    assert all(0 <= idx < n_pred for idx in order)
    dst_col_for_src = np.empty(n_pred, dtype=int)
    for dst, src in enumerate(order):
        dst_col_for_src[src] = dst

    default_mapping_type = "fallback_argmax" if fallback_mode == "argmax" else "fallback_skip"
    mapping_type = np.full(n_pred, default_mapping_type, dtype=object)
    mapping_type[mapped_line_id >= 0] = "line_guided"

    return {
        "lines_sorted": lines_sorted,
        "mapped_y": mapped_y,
        "mapped_line_id": mapped_line_id,
        "mapping_type": mapping_type,
        "order": order,
        "dst_col_for_src": dst_col_for_src,
        "overlap_rows": overlap_rows,
        "aligned": aligned,
    }


def write_mapping_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def plot_heatmap_with_lines(path: Path, matrix: np.ndarray, lines: list[dict], y_band_min: int, y_band_max: int):
    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    fig.colorbar(im, ax=ax, label="chrF")
    for i, ln in enumerate(lines):
        ax.plot([ln["x0"], ln["x1"]], [ln["y0"], ln["y1"]], "-", linewidth=1.8, label=f"line {i}")
    ax.axhspan(y_band_min, y_band_max, color="white", alpha=0.12)
    ax.set_title("Score Heatmap + Kept Lines")
    ax.set_xlabel("pred segment (x)")
    ax.set_ylabel("ref segment (y)")
    if lines:
        ax.legend(loc="lower right", fontsize=8, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_src_to_dst(path: Path, dst_col_for_src: np.ndarray, mapping_type: np.ndarray):
    x = np.arange(len(dst_col_for_src))
    fig, ax = plt.subplots(figsize=(10, 7))
    masks = {
        "line_guided": mapping_type == "line_guided",
        "fallback_argmax": mapping_type == "fallback_argmax",
        "fallback_skip": mapping_type == "fallback_skip",
    }
    colors = {"line_guided": "#1f77b4", "fallback_argmax": "#ff7f0e", "fallback_skip": "#7f7f7f"}
    for name, mask in masks.items():
        if np.any(mask):
            ax.scatter(x[mask], dst_col_for_src[mask], s=9, alpha=0.75, label=name, c=colors[name])
    ax.plot([0, len(x) - 1], [0, len(x) - 1], "k--", linewidth=1, label="identity")
    ax.set_title("Reorder Map (source x -> destination col)")
    ax.set_xlabel("source x column")
    ax.set_ylabel("destination column")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_src_to_mapped_y(path: Path, mapped_y: np.ndarray, mapping_type: np.ndarray, y_band_min: int, y_band_max: int):
    x = np.arange(len(mapped_y))
    finite = np.isfinite(mapped_y)
    fig, ax = plt.subplots(figsize=(10, 7))
    for name, color in [("line_guided", "#1f77b4"), ("fallback_argmax", "#ff7f0e"), ("fallback_skip", "#7f7f7f")]:
        mask = finite & (mapping_type == name)
        if np.any(mask):
            ax.scatter(x[mask], mapped_y[mask], s=9, alpha=0.75, label=name, c=color)
    ax.axhspan(y_band_min, y_band_max, color="red", alpha=0.12, label=f"y-band [{y_band_min}, {y_band_max}]")
    ax.set_title("Column Mapping to Reference Y")
    ax.set_xlabel("source x column")
    ax.set_ylabel("mapped y row")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_displacement_hist(path: Path, dst_col_for_src: np.ndarray):
    src = np.arange(len(dst_col_for_src))
    disp = dst_col_for_src - src
    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.hist(disp, bins=80, color="#2ca02c", alpha=0.85)
    ax.set_title("Displacement Histogram (dst_col - src_col)")
    ax.set_xlabel("column displacement")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    if not args.runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    item = load_target_item(args.runfile_json, args.target_fname)
    pred = item["pred"]
    ref = item["ref"]

    matrix = align_v2.compute_score_matrix(ref, pred, window_size=args.window_size, window_stride=args.window_stride)
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise RuntimeError("Score matrix is empty; text may be shorter than window size.")

    det = align_v2.detect_lines_dense_style_no_angle_seeded(matrix, seed=int(args.hough_seed) + int(item["index"]))
    lines_raw = align_v2.lines_from_merged_segments(matrix, det.get("merged_lines", []))
    mask_bool = np.asarray(det.get("mask", np.zeros_like(matrix))) > 0
    lines_used, column_assignment = align_v2.filter_lines_for_alignment_by_ownership(
        lines_raw,
        matrix,
        mask_bool,
    )
    dbg = compute_mapping_debug(
        pred_text=pred,
        matrix=matrix,
        lines=lines_used,
        fallback_mode=args.fallback_mode,
        window_stride=args.window_stride,
        column_assignment=column_assignment,
    )

    n_ref, n_pred = matrix.shape
    blocks, src_spans = build_blocks_and_spans(pred, n_pred=n_pred, stride=args.window_stride)
    adjusted = str(dbg["aligned"]["adjusted_pred"])
    assert len(adjusted) == len(pred)

    dst_spans = {}
    cursor = 0
    for src in dbg["order"]:
        block_len = len(blocks[src])
        dst_spans[src] = (cursor, cursor + block_len)
        cursor += block_len

    mapping_rows = []
    for x in range(n_pred):
        s0, s1 = src_spans[x]
        d0, d1 = dst_spans[x]
        mapping_rows.append(
            {
                "x_src_col": x,
                "dst_col": int(dbg["dst_col_for_src"][x]),
                "displacement_cols": int(dbg["dst_col_for_src"][x] - x),
                "mapped_y": "" if not np.isfinite(dbg["mapped_y"][x]) else int(dbg["mapped_y"][x]),
                "mapped_line_id": int(dbg["mapped_line_id"][x]),
                "mapping_type": str(dbg["mapping_type"][x]),
                "src_char_start": s0,
                "src_char_end": s1,
                "dst_char_start": d0,
                "dst_char_end": d1,
            }
        )

    yb_min, yb_max = int(args.y_band_min), int(args.y_band_max)
    band_rows = [dict(r) for r in mapping_rows if r["mapped_y"] != "" and yb_min <= int(r["mapped_y"]) <= yb_max]
    band_rows = band_rows[: int(args.max_band_rows)]
    for r in band_rows:
        s0, s1 = r["src_char_start"], r["src_char_end"]
        d0, d1 = r["dst_char_start"], r["dst_char_end"]
        r["src_text"] = pred[s0:s1]
        r["dst_text"] = adjusted[d0:d1]
        r["text_equal_after_move"] = int(r["src_text"] == r["dst_text"])

    write_mapping_csv(args.output_dir / "mapping_full.csv", mapping_rows)
    write_mapping_csv(args.output_dir / "mapping_y_band.csv", band_rows)
    write_mapping_csv(args.output_dir / "overlap_candidates.csv", dbg["overlap_rows"])

    plot_heatmap_with_lines(
        args.output_dir / "plot_heatmap_kept_lines.png",
        matrix=matrix,
        lines=dbg["lines_sorted"],
        y_band_min=yb_min,
        y_band_max=yb_max,
    )
    plot_src_to_dst(args.output_dir / "plot_src_to_dst.png", dbg["dst_col_for_src"], dbg["mapping_type"])
    plot_src_to_mapped_y(
        args.output_dir / "plot_src_to_mapped_y.png",
        mapped_y=dbg["mapped_y"],
        mapping_type=dbg["mapping_type"],
        y_band_min=yb_min,
        y_band_max=yb_max,
    )
    plot_displacement_hist(args.output_dir / "plot_displacement_hist.png", dbg["dst_col_for_src"])

    summary = {
        "file": item["fname"],
        "matrix_shape": [int(n_ref), int(n_pred)],
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "fallback_mode": str(args.fallback_mode),
        "raw_line_count": int(len(lines_raw)),
        "used_line_count": int(len(lines_used)),
        "line_guided_columns": int(dbg["aligned"]["line_guided_columns"]),
        "fallback_columns": int(dbg["aligned"]["fallback_columns"]),
        "y_band": [yb_min, yb_max],
        "y_band_column_count": int(len([r for r in mapping_rows if r["mapped_y"] != "" and yb_min <= int(r["mapped_y"]) <= yb_max])),
        "overlap_candidate_rows": int(len(dbg["overlap_rows"])),
        "outputs": {
            "mapping_full_csv": str(args.output_dir / "mapping_full.csv"),
            "mapping_y_band_csv": str(args.output_dir / "mapping_y_band.csv"),
            "overlap_candidates_csv": str(args.output_dir / "overlap_candidates.csv"),
            "plot_heatmap_kept_lines": str(args.output_dir / "plot_heatmap_kept_lines.png"),
            "plot_src_to_dst": str(args.output_dir / "plot_src_to_dst.png"),
            "plot_src_to_mapped_y": str(args.output_dir / "plot_src_to_mapped_y.png"),
            "plot_displacement_hist": str(args.output_dir / "plot_displacement_hist.png"),
        },
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
