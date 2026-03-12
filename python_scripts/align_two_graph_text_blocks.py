import json, math, pickle, re, sys
import numpy as np

from pathlib import Path

from visualise_dorian_dense_matrices_style_no_angle_all import (
    detect_lines_dense_style_no_angle,
)

"""
Align prediction text blocks for target graph cases using line endpoints
detected from their charF score matrices.

High-level pipeline (current logic):
1) Load target entries from scores.pkl.
2) Detect diagonal alignment lines from each score matrix.
3) Filter lines for robustness (length + support).
4) Reorder prediction blocks using a single strategy:
   top-to-bottom line order, left-to-right within each line.
5) Assemble the adjusted prediction with overlap-aware stitching (stride).
6) Report normalized Levenshtein before/after and save outputs per case.

This script is evaluation-oriented in the sense that it reports a score
for the single strategy it applies; it does not search across strategies.
"""


# Project paths and input data source.
PROJECT_ROOT = Path("/scratch/project_2017385/dorian/HTR-context-OCR")
# scores.pkl is produced by the Churro_copy pipeline.
SCORES_PKL = Path("/scratch/project_2017385/dorian/Churro_copy/results/custom_churro_infer_dev_run1/vllm/dev/scores.pkl")
# Reuse project metrics exactly as done by custom_churro_infer's evaluation path.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from evaluation.evaluate_page import levenshtein_distance

# Target graph outputs to process.
TARGET_GRAPHS = [
    "newseye-aus_onb_krz_19330701_010_graph.png",
    "newseye-fre_p1020482_graph.png",
    "0068_salamanca_w0019_page159_graph.png",
]

# compare.py defaults used to build scores.pkl.
# WINDOW_SIZE is kept for reference; WINDOW_STRIDE is used for reassembly.
WINDOW_SIZE = 100
WINDOW_STRIDE = 50

# Output folder for adjusted text + reports (may contain multiple targets).
OUTPUT_DIR = PROJECT_ROOT / "results/aligned_text_blocks_two_cases"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def safe_name(name: str) -> str:
    """
    Normalize filename stem to a filesystem-safe token.
    Keeps only [A-Za-z0-9._-], replaces other chars with "_", and truncates.
    """
    stem = Path(name).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)
    return stem[:120]


def safe_matrix(scores) -> np.ndarray:
    """
    Convert raw score payload to a clean 2D float matrix.
    - invalid shape -> 1x1 zero matrix
    - NaN/Inf -> 0
    """
    mat = np.asarray(scores, dtype=float)
    if mat.ndim != 2 or mat.size == 0:
        return np.zeros((1, 1), dtype=float)
    return np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)


def normalized_levenshtein_similarity(predicted_text: str, gold_text: str) -> float:
    """
    Same formula used in custom_churro_infer evaluation path, and distance function
    is imported directly from evaluation.evaluate_page.
      1 - levenshtein_distance(pred, gold) / max(len(pred), len(gold))
    """
    denom = max(len(predicted_text), len(gold_text))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(predicted_text, gold_text) / denom)


def detect_lines_dense_style_for_alignment(matrix: np.ndarray):
    """
    Call the shared dense-matrices detector and convert merged lines into
    line dictionaries for alignment.
    """
    det = detect_lines_dense_style_no_angle(matrix)
    merged = det["merged_lines"]

    line_dicts = []
    for p0, p1 in merged:
        x0, y0 = float(p0[0]), float(p0[1])
        x1, y1 = float(p1[0]), float(p1[1])
        if x1 < x0:
            x0, x1 = x1, x0
            y0, y1 = y1, y0
        score = mean_line_support(matrix, {"x0": x0, "y0": y0, "x1": x1, "y1": y1})
        line_dicts.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1, "score": float(score)})

    lines_read_order = sorted(
        line_dicts,
        key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])),
    )
    return det["threshold_start"], (det["mask"] > 0), lines_read_order


def line_y_at_x(line: dict, x: int) -> float:
    """Evaluate interpolated y-coordinate of a line at integer x."""
    x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
    dx = x1 - x0
    if abs(dx) < 1e-8:
        return y0
    t = (x - x0) / dx
    return y0 + t * (y1 - y0)


def line_length(line: dict) -> float:
    """Euclidean length of a fitted line segment."""
    return float(math.hypot(line["x1"] - line["x0"], line["y1"] - line["y0"]))


def mean_line_support(matrix: np.ndarray, line: dict) -> float:
    """
    Mean matrix intensity sampled along a fitted line.
    Used as a robustness check before using a line for block alignment.
    """
    n_ref, n_pred = matrix.shape
    x_start = int(max(0, math.ceil(min(line["x0"], line["x1"]))))
    x_end = int(min(n_pred - 1, math.floor(max(line["x0"], line["x1"]))))
    if x_end < x_start:
        return 0.0

    vals = []
    for x in range(x_start, x_end + 1):
        y_idx = int(np.clip(round(line_y_at_x(line, x)), 0, n_ref - 1))
        vals.append(float(matrix[y_idx, x]))
    if not vals:
        return 0.0
    return float(np.mean(vals))


def filter_lines_for_alignment(lines_read_order: list[dict], matrix: np.ndarray) -> list[dict]:
    """
    Keep only robust lines for block shifting. This suppresses short/noisy lines
    that otherwise cause large wrong permutations.
    """
    if not lines_read_order:
        return []

    # Dynamic thresholds based on current graph scale/distribution.
    max_score = max(float(ln["score"]) for ln in lines_read_order)
    min_len = max(8.0, 0.08 * float(min(matrix.shape)))
    support_floor = float(np.percentile(matrix, 75))

    kept = []
    for ln in lines_read_order:
        ln2 = dict(ln)
        ln2["length"] = line_length(ln2)
        ln2["support"] = mean_line_support(matrix, ln2)

        if ln2["length"] < min_len:
            continue
        if ln2["score"] < 0.06 * max_score:
            continue
        if ln2["support"] < support_floor:
            continue
        kept.append(ln2)

    if not kept:
        # Always keep one best line to avoid empty downstream mapping.
        best = max(lines_read_order, key=lambda ln: float(ln["score"]))
        best2 = dict(best)
        best2["length"] = line_length(best2)
        best2["support"] = mean_line_support(matrix, best2)
        return [best2]

    return sorted(kept, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))


def map_columns_to_ref_rows(matrix: np.ndarray, lines_read_order: list[dict]):
    """
    Build x->y mapping from line endpoints.
    If multiple lines vote for same x, keep the y with higher local score matrix[y, x].
    Note: unused in the current pipeline; retained for alternative strategies.
    """
    n_ref, n_pred = matrix.shape
    # For each prediction column x, we estimate best reference row y.
    mapped_y = np.full(n_pred, np.nan, dtype=float)
    # Track which line produced the winning vote per column.
    mapped_line_id = np.full(n_pred, -1, dtype=int)

    # candidates[x] holds all possible (y,row_source_line,local_strength) votes.
    candidates = [[] for _ in range(n_pred)]

    for lid, ln in enumerate(lines_read_order):
        x_start = int(math.ceil(min(ln["x0"], ln["x1"])))
        x_end = int(math.floor(max(ln["x0"], ln["x1"])))
        x_start = max(0, x_start)
        x_end = min(n_pred - 1, x_end)
        if x_end < x_start:
            continue

        for x in range(x_start, x_end + 1):
            y_float = line_y_at_x(ln, x)
            y_idx = int(np.clip(round(y_float), 0, n_ref - 1))
            local_score = float(matrix[y_idx, x])
            candidates[x].append((y_idx, lid, local_score))

    for x in range(n_pred):
        if not candidates[x]:
            continue
        # Resolve conflicts by highest local matrix score.
        best = max(candidates[x], key=lambda c: c[2])
        mapped_y[x] = float(best[0])
        mapped_line_id[x] = int(best[1])

    return mapped_y, mapped_line_id


def fill_unmapped_rows(mapped_y: np.ndarray, n_ref: int):
    """
    Fill missing target rows for columns that had no line vote.

    Priority:
    1) interpolate between nearest known left/right anchors;
    2) otherwise extrapolate using global diagonal slope;
    3) clip to valid [0, n_ref-1].
    Note: unused in the current pipeline; designed to complement map_columns_to_ref_rows.
    """
    n_pred = mapped_y.shape[0]
    out = mapped_y.copy()
    known = np.where(~np.isnan(out))[0]

    if known.size == 0:
        # No detected mapping at all: fall back to a monotonic diagonal prior.
        if n_pred == 1:
            out[0] = 0.0
        else:
            out[:] = np.linspace(0.0, float(n_ref - 1), n_pred)
        return out

    for x in range(n_pred):
        if not np.isnan(out[x]):
            continue

        left_known = known[known < x]
        right_known = known[known > x]

        if left_known.size > 0 and right_known.size > 0:
            lx = int(left_known[-1])
            rx = int(right_known[0])
            if rx == lx:
                out[x] = out[lx]
            else:
                t = (x - lx) / (rx - lx)
                out[x] = (1.0 - t) * out[lx] + t * out[rx]
        elif left_known.size > 0:
            lx = int(left_known[-1])
            slope = (n_ref - 1) / max(n_pred - 1, 1)
            out[x] = out[lx] + (x - lx) * slope
        else:
            rx = int(right_known[0])
            slope = (n_ref - 1) / max(n_pred - 1, 1)
            out[x] = out[rx] - (rx - x) * slope

    return np.clip(out, 0.0, float(n_ref - 1))


def build_pred_blocks(pred_text: str, n_pred: int, stride: int):
    """
    Build exact movable prediction blocks aligned to matrix x-axis indices.
    Block j spans [j*stride, (j+1)*stride), last block spans to end.
    """
    starts = [j * stride for j in range(n_pred)]
    blocks = []
    for j, s in enumerate(starts):
        # Last block consumes the remaining tail of prediction text.
        e = starts[j + 1] if (j + 1) < n_pred else len(pred_text)
        # Clamp to safe boundaries for short strings.
        s = min(s, len(pred_text))
        e = min(max(e, s), len(pred_text))
        blocks.append(pred_text[s:e])
    return blocks


def assemble_from_order_with_stride(pred_text: str, order: list[int], n_pred: int, stride: int) -> str:
    """
    Assemble text from overlapping sliding-window segments without duplicating overlaps.
    - The first segment is taken fully.
    - Each subsequent segment contributes only its last `stride` characters.
    """
    blocks = build_pred_blocks(pred_text, n_pred=n_pred, stride=stride)
    if not order:
        return ""
    # Take full first block.
    out = [blocks[order[0]]]
    for idx in order[1:]:
        block = blocks[idx]
        if stride <= 0 or len(block) <= stride:
            out.append(block)
        else:
            out.append(block[-stride:])
    return "".join(out)

def assemble_from_order(pred_text: str, order: list[int], n_pred: int) -> str:
    """Rebuild prediction by concatenating fixed blocks in provided order."""
    blocks = build_pred_blocks(pred_text, n_pred=n_pred, stride=WINDOW_STRIDE)
    return "".join(blocks[idx] for idx in order)


def local_window_order(target_rows: np.ndarray, mapped_mask: np.ndarray, window_size: int) -> list[int]:
    """
    Safer local reorder:
    - only reorder mapped blocks
    - only inside local windows
    Note: unused in the current pipeline; kept for experimentation.
    """
    n = int(target_rows.shape[0])
    order = list(range(n))

    # Only permute within local windows to avoid catastrophic global shuffles.
    for s in range(0, n, window_size):
        e = min(n, s + window_size)
        window_indices = order[s:e]

        # Unmapped blocks stay in-place; mapped blocks get row-based ordering.
        mapped_positions = [k for k, idx in enumerate(window_indices) if bool(mapped_mask[idx])]
        mapped_indices = [window_indices[k] for k in mapped_positions]
        mapped_sorted = sorted(mapped_indices, key=lambda idx: (float(target_rows[idx]), idx))

        replaced = list(window_indices)
        for k, idx in zip(mapped_positions, mapped_sorted):
            replaced[k] = idx
        order[s:e] = replaced

    return order


def segment_order_from_line_ids(mapped_line_id: np.ndarray) -> list[int]:
    """
    Segment-wise line order:
    concatenate line-assigned blocks by line-id order, then append unassigned.
    Note: unused in the current pipeline; kept for experimentation.
    """
    n = int(mapped_line_id.shape[0])
    # Positive line ids correspond to detected/assigned alignment traces.
    line_ids = sorted({int(v) for v in mapped_line_id if int(v) >= 0})

    used = set()
    order = []
    for lid in line_ids:
        idxs = [i for i in range(n) if int(mapped_line_id[i]) == lid]
        for i in idxs:
            if i not in used:
                used.add(i)
                order.append(i)

    for i in range(n):
        if i not in used:
            order.append(i)
    return order


def align_prediction_by_line_endpoints(
    pred_text: str,
    ref_text: str,
    matrix: np.ndarray,
    lines_read_order: list[dict],
):
    """
    Reorder prediction blocks using dense-matrices style lines:
    - Lines are ordered top-to-bottom by their y position.
    - Blocks are assigned to lines by x-range overlap.
    - If multiple lines overlap a block, the topmost line (smallest y) wins.
    - Unassigned blocks are appended in original order.
    - Assembly respects overlap (stride) to avoid duplicating text.
    """
    n_ref, n_pred = matrix.shape

    # Precompute per-line y interpolation for candidate assignment.
    def line_y_at_x(line: dict, x: int) -> float:
        x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
        dx = x1 - x0
        if abs(dx) < 1e-8:
            return y0
        t = (x - x0) / dx
        return y0 + t * (y1 - y0)

    # Sort lines by top-to-bottom (y-axis).
    lines_sorted = sorted(
        lines_read_order,
        key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])),
    )

    # Assign each block index to a line based on x overlap.
    block_to_line = {}
    for x in range(n_pred):
        candidates = []
        for lid, ln in enumerate(lines_sorted):
            x_min = int(math.floor(min(ln["x0"], ln["x1"])))
            x_max = int(math.ceil(max(ln["x0"], ln["x1"])))
            if x < x_min or x > x_max:
                continue
            y_est = line_y_at_x(ln, x)
            candidates.append((lid, abs(y_est)))
        if not candidates:
            continue
        # Choose the topmost line at this x (smallest y).
        candidates.sort(key=lambda t: t[1])
        block_to_line[x] = candidates[0][0]

    # Build ordered block list: per line (top->bottom), left->right within line.
    order = []
    used = set()
    for lid in range(len(lines_sorted)):
        xs = [x for x, ln_id in block_to_line.items() if ln_id == lid]
        xs.sort()
        for x in xs:
            if x not in used:
                used.add(x)
                order.append(x)

    # Append any unassigned blocks in original order.
    for x in range(n_pred):
        if x not in used:
            order.append(x)

    adjusted_pred = assemble_from_order_with_stride(pred_text, order, n_pred=n_pred, stride=WINDOW_STRIDE)

    return {
        "adjusted_pred": adjusted_pred,
        "order": order,
        "strategy": "line_order_top_to_bottom",
        "candidate_scores": {},
        "mapped_y": None,
        "mapped_line_id": None,
        "target_rows": None,
        "n_pred_blocks": n_pred,
    }


def graph_stem_to_safe_basename(graph_filename: str) -> str:
    """
    Convert numbered graph filename to the safe base id used in scores entries.
    Example: 0016_newseye-fin_..._graph.png -> newseye-fin_...
    """
    stem = Path(graph_filename).stem
    m = re.match(r"^\d{4}_(.+)_graph$", stem)
    if not m:
        # Fallback for non-prefixed graph names: strip trailing "_graph" if present.
        if stem.endswith("_graph"):
            return stem[: -len("_graph")]
        return stem
    return m.group(1)


def main():
    # Target identifiers expected in scores.pkl after safe_name normalization.
    target_safe_names = {graph_stem_to_safe_basename(name) for name in TARGET_GRAPHS}

    # Load only the requested cases from the pickle stream.
    items_by_safe = {}
    with open(SCORES_PKL, "rb") as f:
        while True:
            try:
                item = pickle.load(f)
            except EOFError:
                break
            sname = safe_name(Path(item["fname"]).name)
            if sname in target_safe_names:
                items_by_safe[sname] = item
                if len(items_by_safe) == len(target_safe_names):
                    break

    missing = [s for s in target_safe_names if s not in items_by_safe]
    if missing:
        raise RuntimeError(f"Missing target items in scores.pkl: {missing}")

    summary = []

    for graph_name in TARGET_GRAPHS:
        safe_base = graph_stem_to_safe_basename(graph_name)
        item = items_by_safe[safe_base]

        matrix = safe_matrix(item["scores"])
        pred = item["pred"]
        ref = item["ref"]

        # Line extraction + robust filtering before text-block shifting.
        thr, mask, lines_read_order = detect_lines_dense_style_for_alignment(matrix)
        lines_for_alignment = filter_lines_for_alignment(lines_read_order, matrix)
        aligned = align_prediction_by_line_endpoints(
            pred_text=pred,
            ref_text=ref,
            matrix=matrix,
            lines_read_order=lines_for_alignment,
        )
        adjusted_pred = aligned["adjusted_pred"]

        before_nls = normalized_levenshtein_similarity(pred, ref)
        after_nls = normalized_levenshtein_similarity(adjusted_pred, ref)

        case_id = Path(graph_name).stem
        out_txt = OUTPUT_DIR / f"{case_id}_adjusted_pred.txt"
        out_report = OUTPUT_DIR / f"{case_id}_alignment_report.json"

        # Save adjusted prediction text exactly as assembled from chosen order.
        out_txt.write_text(adjusted_pred, encoding="utf-8")

        # Compact line endpoint report in requested reading order.
        line_report = []
        for lid, ln in enumerate(lines_for_alignment):
            line_report.append(
                {
                    "line_id": lid,
                    "x0": round(float(ln["x0"]), 4),
                    "y0": round(float(ln["y0"]), 4),
                    "x1": round(float(ln["x1"]), 4),
                    "y1": round(float(ln["y1"]), 4),
                    "score": round(float(ln["score"]), 6),
                    "length": round(float(ln.get("length", line_length(ln))), 4),
                    "support": round(float(ln.get("support", mean_line_support(matrix, ln))), 6),
                }
            )

        mapped_y = aligned.get("mapped_y")
        if mapped_y is None:
            mapped_count = 0
        else:
            mapped_count = int(np.sum(~np.isnan(mapped_y)))
        # Case-level report with both geometry and text-metric outcomes.
        report = {
            "graph": graph_name,
            "file_name": Path(item["fname"]).name,
            "matrix_shape": list(matrix.shape),
            "threshold_p88": thr,
            "line_count_raw": len(lines_read_order),
            "line_count_used": len(lines_for_alignment),
            "lines_read_order": line_report,
            "selected_shift_strategy": aligned["strategy"],
            "candidate_scores": aligned["candidate_scores"],
            "mapped_pred_block_count": mapped_count,
            "total_pred_block_count": int(matrix.shape[1]),
            "before_normalized_levenshtein_similarity": before_nls,
            "after_normalized_levenshtein_similarity": after_nls,
            "delta": after_nls - before_nls,
            "adjusted_pred_path": str(out_txt),
        }

        out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        summary.append(report)

    # Save one summary file covering all target cases.
    summary_path = OUTPUT_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Saved outputs to:", OUTPUT_DIR)
    for r in summary:
        print(
            f"{r['graph']}: before={r['before_normalized_levenshtein_similarity']:.6f}, "
            f"after={r['after_normalized_levenshtein_similarity']:.6f}, "
            f"delta={r['delta']:.6f}, lines_used={r['line_count_used']}, strategy={r['selected_shift_strategy']}"
        )


if __name__ == "__main__":
    main()
