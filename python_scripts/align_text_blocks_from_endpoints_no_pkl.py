import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import sacrebleu

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from evaluation.evaluate_page import levenshtein_distance
from hough_line_transform_endpoints_no_angle_all import merging_diag, normalize_for_dense_style
from skimage.transform import probabilistic_hough_line


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Align prediction text using in-memory score matrices and in-memory Hough endpoints "
            "(no scores.pkl/endpoints.pkl files)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runfile-json", type=Path, required=True, help="Path to outputs.json")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--window-size", type=int, default=100, help="Sliding window size")
    parser.add_argument("--window-stride", type=int, default=50, help="Sliding window stride")
    parser.add_argument(
        "--fallback-mode",
        type=str,
        default="argmax",
        choices=["argmax", "skip"],
        help=(
            "How to handle x-columns not covered by any line: "
            "'argmax' maps with matrix argmax(row), 'skip' keeps those columns in place."
        ),
    )
    parser.add_argument("--target-fname", type=str, default=None, help="Optional exact/basename target file")
    parser.add_argument("--max-items", type=int, default=None, help="Optional maximum processed items")
    parser.add_argument("--hough-seed", type=int, default=0, help="Random seed for probabilistic Hough")
    return parser.parse_args()


def normalized_levenshtein_similarity(predicted_text: str, gold_text: str) -> float:
    denom = max(len(predicted_text), len(gold_text))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(predicted_text, gold_text) / denom)


def same_file(a: str, b: str) -> bool:
    return str(a) == str(b) or Path(str(a)).name == Path(str(b)).name


def safe_name(name: str) -> str:
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def sliding_segments(text: str, window_size: int, window_stride: int) -> list[str]:
    if len(text) < window_size:
        return []
    return [text[i : i + window_size] for i in range(0, len(text) - window_size + 1, window_stride)]


def compute_score_matrix(ref_text: str, pred_text: str, window_size: int, window_stride: int) -> np.ndarray:
    ref_segments = sliding_segments(ref_text, window_size, window_stride)
    pred_segments = sliding_segments(pred_text, window_size, window_stride)

    scores = np.zeros((len(ref_segments), len(pred_segments)), dtype=float)
    for i, ref_seg in enumerate(ref_segments):
        for j, pred_seg in enumerate(pred_segments):
            scores[i, j] = sacrebleu.sentence_chrf(ref_seg, [pred_seg]).score
    return scores


def detect_lines_dense_style_no_angle_seeded(matrix: np.ndarray, seed: int):
    norm = normalize_for_dense_style(matrix)
    test = 1.0 / (1.0 - norm)

    start = 3
    enough = False
    criteria = 1.4 * matrix.shape[0]

    test2 = test.copy()
    while not enough:
        if start < 0:
            break
        start -= 0.2
        test2 = test.copy()
        test2[test2 < start] = 0
        enough = (test2 > 0).sum() > criteria

    ys, xs = np.nonzero(test2)
    points_glo = [(int(x), int(y)) for y, x in zip(ys, xs)]

    lines = probabilistic_hough_line(
        test2,
        threshold=26,
        line_length=10,
        line_gap=15,
        rng=np.random.default_rng(seed),
    )
    res_lines = list(lines)
    merged = merging_diag(res_lines, test2 > 0, points_glo)

    return {
        "threshold_start": start,
        "mask": test2,
        "raw_lines": lines,
        "selected_lines": res_lines,
        "merged_lines": merged,
    }


def line_y_at_x(line: dict, x: int) -> float:
    x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
    dx = x1 - x0
    if abs(dx) < 1e-8:
        return y0
    t = (x - x0) / dx
    return y0 + t * (y1 - y0)


def line_length(line: dict) -> float:
    return float(math.hypot(line["x1"] - line["x0"], line["y1"] - line["y0"]))


def mean_line_support(matrix: np.ndarray, line: dict) -> float:
    if matrix.size == 0:
        return 0.0

    n_ref, n_pred = matrix.shape
    x_start = int(max(0, math.ceil(min(line["x0"], line["x1"]))))
    x_end = int(min(n_pred - 1, math.floor(max(line["x0"], line["x1"]))))
    if x_end < x_start:
        return 0.0

    vals = []
    for x in range(x_start, x_end + 1):
        y_idx = int(np.clip(round(line_y_at_x(line, x)), 0, n_ref - 1))
        vals.append(float(matrix[y_idx, x]))
    return float(np.mean(vals)) if vals else 0.0


def lines_from_merged_segments(matrix: np.ndarray, merged_lines) -> list[dict]:
    lines: list[dict] = []
    for p0, p1 in merged_lines:
        try:
            x0, y0 = float(p0[0]), float(p0[1])
            x1, y1 = float(p1[0]), float(p1[1])
        except Exception:
            continue

        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1 = y1, y0

        line = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
        line["score"] = mean_line_support(matrix, line)
        lines.append(line)

    return sorted(lines, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))


def filter_lines_for_alignment(lines: list[dict], matrix: np.ndarray) -> list[dict]:
    if not lines:
        return []

    max_score = max(float(ln.get("score", 0.0)) for ln in lines)
    min_dim = min(matrix.shape) if matrix.size > 0 else 1
    min_len = max(8.0, 0.08 * float(min_dim))
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0

    kept = []
    for ln in lines:
        ln2 = dict(ln)
        ln2["length"] = line_length(ln2)
        ln2["support"] = mean_line_support(matrix, ln2)

        if ln2["length"] < min_len:
            continue
        if max_score > 0 and float(ln2.get("score", 0.0)) < 0.06 * max_score:
            continue
        if ln2["support"] < support_floor:
            continue
        kept.append(ln2)

    if not kept:
        best = max(lines, key=lambda ln: float(ln.get("score", 0.0)))
        best2 = dict(best)
        best2["length"] = line_length(best2)
        best2["support"] = mean_line_support(matrix, best2)
        return [best2]

    return sorted(kept, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))


def build_pred_blocks(pred_text: str, n_pred: int, stride: int) -> list[str]:
    starts = [j * stride for j in range(n_pred)]
    blocks = []
    for j, s in enumerate(starts):
        e = starts[j + 1] if (j + 1) < n_pred else len(pred_text)
        s = min(s, len(pred_text))
        e = min(max(e, s), len(pred_text))
        blocks.append(pred_text[s:e])
    return blocks


def assemble_from_order_with_stride(pred_text: str, order: list[int], n_pred: int, stride: int) -> str:
    if n_pred <= 0:
        return pred_text
    blocks = build_pred_blocks(pred_text, n_pred, stride)
    if not order:
        return pred_text

    out = [blocks[order[0]]]
    for idx in order[1:]:
        block = blocks[idx]
        out.append(block)
    return "".join(out)


def align_prediction(
    pred_text: str,
    matrix: np.ndarray,
    lines_read_order: list[dict],
    window_stride: int,
    fallback_mode: str,
):
    n_ref, n_pred = matrix.shape
    if n_pred == 0:
        return {
            "adjusted_pred": pred_text,
            "line_guided_columns": 0,
            "fallback_columns": 0,
            "order": [],
        }

    lines_sorted = sorted(lines_read_order, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))
    mapped_y = np.full(n_pred, np.nan, dtype=float)
    mapped_line_id = np.full(n_pred, -1, dtype=int)

    for x in range(n_pred):
        best_score = float("-inf")
        best_y_idx = -1
        best_lid = -1

        for lid, ln in enumerate(lines_sorted):
            x_min = int(math.floor(min(ln["x0"], ln["x1"])))
            x_max = int(math.ceil(max(ln["x0"], ln["x1"])))
            if x < x_min or x > x_max:
                continue

            y_est = line_y_at_x(ln, x)
            y_idx = int(np.clip(round(y_est), 0, n_ref - 1))
            score = float(matrix[y_idx, x])

            if (
                score > best_score
                or (math.isclose(score, best_score) and (best_y_idx < 0 or y_idx < best_y_idx))
                or (math.isclose(score, best_score) and y_idx == best_y_idx and (best_lid < 0 or lid < best_lid))
            ):
                best_score = score
                best_y_idx = y_idx
                best_lid = lid

        if best_lid >= 0:
            mapped_y[x] = float(best_y_idx)
            mapped_line_id[x] = best_lid
        elif fallback_mode == "argmax":
            mapped_y[x] = float(int(np.argmax(matrix[:, x])) if n_ref > 0 else 0)

    if fallback_mode == "skip":
        guided_sorted = sorted(
            [x for x in range(n_pred) if mapped_line_id[x] >= 0],
            key=lambda x: (float(mapped_y[x]), x),
        )
        gi = 0
        order = []
        for x in range(n_pred):
            if mapped_line_id[x] >= 0:
                order.append(guided_sorted[gi])
                gi += 1
            else:
                order.append(x)
    else:
        filled_y = np.where(np.isfinite(mapped_y), mapped_y, np.arange(n_pred, dtype=float))
        order = sorted(range(n_pred), key=lambda x: (float(filled_y[x]), x))

    assert len(order) == n_pred, "Order length must match number of prediction columns"
    assert all(0 <= idx < n_pred for idx in order), "Order contains out-of-range column indices"

    adjusted_pred = assemble_from_order_with_stride(pred_text, order, n_pred=n_pred, stride=window_stride)
    assert len(adjusted_pred) == len(pred_text), "Adjusted text length must match original prediction length"
    assert Counter(adjusted_pred) == Counter(pred_text), "Adjusted text character counts must match prediction"
    assert sorted(adjusted_pred) == sorted(pred_text), "Sorted adjusted characters must equal sorted prediction characters"
    line_guided_cols = int(np.sum(mapped_line_id >= 0))

    return {
        "adjusted_pred": adjusted_pred,
        "line_guided_columns": line_guided_cols,
        "fallback_columns": int(n_pred - line_guided_cols),
        "order": order,
    }


def load_run_items(runfile_json: Path) -> list[dict]:
    data = json.loads(runfile_json.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError(f"Expected runfile JSON list, got: {type(data).__name__}")

    out: list[dict] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            continue
        file_name = str(item.get("file_name", item.get("fname", f"item_{idx:04d}")))
        out.append(
            {
                "index": idx,
                "fname": Path(file_name).name,
                "pred": str(item.get("normalized_predicted_text", item.get("pred", ""))),
                "ref": str(item.get("normalized_gold_text", item.get("ref", ""))),
            }
        )
    return out


def write_case_report(path: Path, row: dict, lines_used: list[dict], matrix_shape: tuple[int, int], args):
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"file={row['fname']}\n")
        f.write(f"matrix_shape={matrix_shape[0]}x{matrix_shape[1]}\n")
        f.write(f"window_size={args.window_size} window_stride={args.window_stride}\n")
        f.write(f"fallback_mode={args.fallback_mode}\n")
        f.write(f"lines_used={len(lines_used)}\n")
        f.write(f"line_guided_columns={row['line_guided_columns']} fallback_columns={row['fallback_columns']}\n")
        f.write(f"before_normalized_levenshtein_similarity={row['before_normalized_levenshtein_similarity']:.6f}\n")
        f.write(f"after_normalized_levenshtein_similarity={row['after_normalized_levenshtein_similarity']:.6f}\n")
        f.write(f"delta={row['delta']:.6f}\n\n")
        f.write("[lines used]\n")
        for lid, ln in enumerate(lines_used):
            f.write(
                f"line={lid} x0={ln['x0']:.3f} y0={ln['y0']:.3f} x1={ln['x1']:.3f} y1={ln['y1']:.3f} "
                f"score={float(ln.get('score', 0.0)):.6f} length={float(ln.get('length', 0.0)):.3f} "
                f"support={float(ln.get('support', 0.0)):.6f}\n"
            )


def main():
    args = parse_args()
    if not args.runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {args.runfile_json}")
    if args.window_size <= 0 or args.window_stride <= 0:
        raise ValueError("window-size and window-stride must be positive")
    if args.max_items is not None and args.max_items <= 0:
        raise ValueError("max-items must be positive")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_items = load_run_items(args.runfile_json)

    rows = []
    processed = 0
    matched = 0

    for item in run_items:
        fname = item["fname"]
        if args.target_fname is not None and not same_file(fname, args.target_fname):
            continue
        matched += 1
        if args.max_items is not None and processed >= args.max_items:
            break

        pred = item["pred"]
        ref = item["ref"]

        matrix = compute_score_matrix(ref, pred, window_size=args.window_size, window_stride=args.window_stride)
        if matrix.size > 0 and matrix.shape[0] > 0 and matrix.shape[1] > 0:
            det = detect_lines_dense_style_no_angle_seeded(
                matrix,
                seed=int(args.hough_seed) + int(item["index"]),
            )
            merged_lines = det.get("merged_lines", [])
        else:
            merged_lines = []

        lines_raw = lines_from_merged_segments(matrix, merged_lines)
        lines_used = filter_lines_for_alignment(lines_raw, matrix)

        aligned = align_prediction(
            pred_text=pred,
            matrix=matrix,
            lines_read_order=lines_used,
            window_stride=args.window_stride,
            fallback_mode=args.fallback_mode,
        )

        adjusted_pred = aligned["adjusted_pred"]
        before_nls = normalized_levenshtein_similarity(pred, ref)
        after_nls = normalized_levenshtein_similarity(adjusted_pred, ref)

        case_prefix = f"{item['index']:04d}_{safe_name(Path(fname).name)}"
        out_pred = args.output_dir / f"{case_prefix}_adjusted_pred.txt"
        out_txt = args.output_dir / f"{case_prefix}_line_segments.txt"

        out_pred.write_text(adjusted_pred, encoding="utf-8")

        row = {
            "index": int(item["index"]),
            "fname": Path(fname).name,
            "before_normalized_levenshtein_similarity": float(before_nls),
            "after_normalized_levenshtein_similarity": float(after_nls),
            "delta": float(after_nls - before_nls),
            "line_guided_columns": int(aligned["line_guided_columns"]),
            "fallback_columns": int(aligned["fallback_columns"]),
            "raw_line_count": int(len(lines_raw)),
            "used_line_count": int(len(lines_used)),
            "adjusted_pred_path": str(out_pred),
            "report_path": str(out_txt),
        }
        rows.append(row)

        write_case_report(out_txt, row=row, lines_used=lines_used, matrix_shape=matrix.shape, args=args)

        processed += 1
        print(
            f"[{processed}] {row['fname']} | before={before_nls:.6f} after={after_nls:.6f} "
            f"delta={row['delta']:+.6f} | guided={row['line_guided_columns']} "
            f"fallback={row['fallback_columns']}"
        )

        del matrix

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