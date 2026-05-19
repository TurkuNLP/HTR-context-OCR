import math

import numpy as np

from alignment_utils.line_geometry_support import line_length, line_y_at_x, mean_line_support

DEFAULT_ABS_MIN_LEN = 6.0
DEFAULT_MIN_IOU_THRESHOLD = 0.035

__all__ = [
    "DEFAULT_ABS_MIN_LEN",
    "DEFAULT_MIN_IOU_THRESHOLD",
    "analyze_line_filtering",
    "filter_lines_for_alignment_by_ownership",
]

# Build the empty per-column mapping used when no final guides survive.
def _empty_assignment(n_pred: int) -> dict[str, np.ndarray]:
    # Match the v1 return shape so downstream scripts do not need to special-case v2.1.
    return {
        "mapped_y": np.full(n_pred, np.nan, dtype=float),
        "mapped_line_id": np.full(n_pred, -1, dtype=int),
    }


# Clamp a line's x-span to valid matrix columns.
def _line_x_bounds(line: dict, n_pred: int) -> tuple[int, int] | None:
    # An empty prediction axis means the line cannot cover any columns.
    if n_pred <= 0:
        return None

    # Convert the geometric span into inclusive integer matrix bounds.
    x_min = max(0, int(math.floor(min(line["x0"], line["x1"]))))
    x_max = min(n_pred - 1, int(math.ceil(max(line["x0"], line["x1"]))))
    if x_max < x_min:
        return None
    return x_min, x_max


# Expand a per-column y-path into the set of covered reference rows.
def _ref_segments_from_path(x_to_y: dict[int, int]) -> set[int]:
    # No sampled path means no reference coverage either.
    if not x_to_y:
        return set()

    rows: set[int] = set()
    prev_y: int | None = None

    # Fill the rows touched at each x and bridge small vertical jumps between neighboring x samples.
    for x in sorted(x_to_y):
        cur_y = int(x_to_y[x])
        rows.add(cur_y)
        if prev_y is not None:
            y0, y1 = sorted((prev_y, cur_y))
            for y in range(y0, y1 + 1):
                rows.add(int(y))
        prev_y = cur_y

    return rows


# Fit one representative straight line through a selected x->y path.
def _fit_line_from_path(
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    *,
    fallback_line: dict | None = None,
) -> dict:
    # Preserve a simple fallback geometry when the path is empty.
    if not x_to_y:
        line = {} if fallback_line is None else dict(fallback_line)
        line.setdefault("x0", 0.0)
        line.setdefault("y0", 0.0)
        line.setdefault("x1", 0.0)
        line.setdefault("y1", 0.0)
        line["length"] = line_length(line)
        line["support"] = mean_line_support(matrix, line) if matrix.size else 0.0
        line["score"] = float(line.get("score", line.get("support", 0.0)))
        return line

    xs = np.asarray(sorted(x_to_y), dtype=float)
    ys = np.asarray([float(x_to_y[int(x)]) for x in xs], dtype=float)
    weights = np.asarray([max(float(x_to_score[int(x)]), 1e-6) for x in xs], dtype=float)

    # Fit a weighted straight line so stronger local matrix cells influence the orientation more.
    if len(xs) == 1 or np.allclose(xs, xs[0]):
        x0 = x1 = float(xs[0])
        y0 = y1 = float(ys[0])
    else:
        slope, intercept = np.polyfit(xs, ys, deg=1, w=weights)
        x0 = float(xs.min())
        x1 = float(xs.max())
        y0 = float((slope * x0) + intercept)
        y1 = float((slope * x1) + intercept)

    line = {} if fallback_line is None else dict(fallback_line)
    # Replace any previous geometry with the coverage-driven representative segment.
    line["x0"] = x0
    line["y0"] = y0
    line["x1"] = x1
    line["y1"] = y1
    line["length"] = line_length(line)
    line["support"] = mean_line_support(matrix, line) if matrix.size else 0.0
    line["score"] = float(line["support"])
    return line


# Build one coverage object from an x->y path and its local matrix scores.
def _coverage_from_path(
    *,
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    fallback_line: dict | None = None,
    source_raw_line_ids: list[int] | None = None,
) -> dict:
    # Normalize the discrete path into sorted prediction coverage and derived reference coverage.
    pred_segments = set(int(x) for x in x_to_y)
    ref_segments = _ref_segments_from_path(x_to_y)
    line = _fit_line_from_path(x_to_y, x_to_score, matrix, fallback_line=fallback_line)

    y_values = [int(x_to_y[x]) for x in sorted(x_to_y)]
    total_score = float(sum(float(v) for v in x_to_score.values()))
    mean_score = float(total_score / len(x_to_score)) if x_to_score else 0.0
    anchor_y = float(np.median(y_values)) if y_values else float(min(line["y0"], line["y1"]))

    # Keep both the representative line and the actual text coverage so later logic can stay coverage-first.
    pred_min = min(pred_segments) if pred_segments else 0
    pred_max = max(pred_segments) if pred_segments else -1
    ref_min = min(ref_segments) if ref_segments else 0
    ref_max = max(ref_segments) if ref_segments else -1

    return {
        "line": line,
        "pred_segments": pred_segments,
        "ref_segments": ref_segments,
        "pred_min": int(pred_min),
        "pred_max": int(pred_max),
        "ref_min": int(ref_min),
        "ref_max": int(ref_max),
        "x_to_y": {int(k): int(v) for k, v in x_to_y.items()},
        "x_to_score": {int(k): float(v) for k, v in x_to_score.items()},
        "total_score": total_score,
        "mean_score": mean_score,
        "anchor_y": anchor_y,
        "source_raw_line_ids": sorted(int(v) for v in (source_raw_line_ids or [])),
    }


# Convert one raw detected line into a coverage object over the matrix grid.
def _build_line_coverage(line: dict, matrix: np.ndarray) -> dict | None:
    # Empty matrices cannot support any useful coverage path.
    if matrix.size == 0:
        return None

    n_ref, n_pred = matrix.shape
    bounds = _line_x_bounds(line, n_pred)
    if bounds is None:
        return None

    x_to_y: dict[int, int] = {}
    x_to_score: dict[int, float] = {}

    # Sample the line at every covered prediction column and keep the exact local matrix score.
    for x in range(bounds[0], bounds[1] + 1):
        y_idx = int(np.clip(round(line_y_at_x(line, x)), 0, n_ref - 1))
        x_to_y[int(x)] = int(y_idx)
        x_to_score[int(x)] = float(matrix[y_idx, x])

    if not x_to_y:
        return None

    return _coverage_from_path(
        x_to_y=x_to_y,
        x_to_score=x_to_score,
        matrix=matrix,
        fallback_line=line,
        source_raw_line_ids=[int(line.get("raw_line_id", -1))] if "raw_line_id" in line else [],
    )


# Compute the exact set IoU used by the v2.1 overlap rule.
def _set_iou(values_a: set[int], values_b: set[int]) -> float:
    # Two empty sets have no meaningful overlap for the merge decision.
    union = values_a | values_b
    if not union:
        return 0.0
    return float(len(values_a & values_b) / len(union))


# Summarize the full x/y IoU relationship between two coverages.
def _segments_bounds_disjoint(min_a: int, max_a: int, min_b: int, max_b: int) -> bool:
    """Cheap axis-aligned disjoint test for integer segment ids."""
    return int(max_a) < int(min_b) or int(max_b) < int(min_a)


def _coverage_iou_stats(cov_a: dict, cov_b: dict, *, min_iou_threshold: float) -> dict:
    pred_a = cov_a["pred_segments"]
    pred_b = cov_b["pred_segments"]
    ref_a = cov_a["ref_segments"]
    ref_b = cov_b["ref_segments"]

    pred_disjoint = _segments_bounds_disjoint(
        int(cov_a.get("pred_min", 0)),
        int(cov_a.get("pred_max", -1)),
        int(cov_b.get("pred_min", 0)),
        int(cov_b.get("pred_max", -1)),
    )
    ref_disjoint = _segments_bounds_disjoint(
        int(cov_a.get("ref_min", 0)),
        int(cov_a.get("ref_max", -1)),
        int(cov_b.get("ref_min", 0)),
        int(cov_b.get("ref_max", -1)),
    )

    # Fast path: when one axis is provably disjoint by bounds, IoU on that axis is
    # exactly zero and the pair cannot merge. We still preserve exact stats for the
    # other axis so debug output remains informative.
    if pred_disjoint and ref_disjoint:
        x_iou = 0.0
        y_iou = 0.0
        shared_pred: set[int] = set()
        shared_ref: set[int] = set()
        union_pred_count = int(len(pred_a) + len(pred_b))
        union_ref_count = int(len(ref_a) + len(ref_b))
    elif pred_disjoint:
        x_iou = 0.0
        shared_pred = set()
        union_pred_count = int(len(pred_a) + len(pred_b))

        shared_ref = ref_a & ref_b
        union_ref = ref_a | ref_b
        union_ref_count = int(len(union_ref))
        y_iou = _set_iou(ref_a, ref_b)
    elif ref_disjoint:
        y_iou = 0.0
        shared_ref = set()
        union_ref_count = int(len(ref_a) + len(ref_b))

        shared_pred = pred_a & pred_b
        union_pred = pred_a | pred_b
        union_pred_count = int(len(union_pred))
        x_iou = _set_iou(pred_a, pred_b)
    else:
        shared_pred = pred_a & pred_b
        shared_ref = ref_a & ref_b
        union_pred = pred_a | pred_b
        union_ref = ref_a | ref_b
        union_pred_count = int(len(union_pred))
        union_ref_count = int(len(union_ref))
        x_iou = _set_iou(pred_a, pred_b)
        y_iou = _set_iou(ref_a, ref_b)

    min_iou = float(min(x_iou, y_iou))

    # Store enough detail to explain exactly why a pair did or did not become a merge edge.
    return {
        "raw_line_ids_a": sorted(int(v) for v in cov_a.get("source_raw_line_ids", [])),
        "raw_line_ids_b": sorted(int(v) for v in cov_b.get("source_raw_line_ids", [])),
        "shared_pred_count": int(len(shared_pred)),
        "union_pred_count": int(union_pred_count),
        "shared_ref_count": int(len(shared_ref)),
        "union_ref_count": int(union_ref_count),
        "shared_pred_segments": [int(v) for v in sorted(shared_pred)],
        "shared_ref_segments": [int(v) for v in sorted(shared_ref)],
        "x_iou": float(x_iou),
        "y_iou": float(y_iou),
        "min_iou": min_iou,
        "min_iou_threshold": float(min_iou_threshold),
        "merge_candidate": bool(min_iou > float(min_iou_threshold)),
    }


# Decide whether two coverages overlap strongly enough under the true-IoU rule.
def _coverages_overlap(cov_a: dict, cov_b: dict, *, min_iou_threshold: float) -> tuple[bool, dict]:
    # A pair merges only when both x and y IoUs exist and the weaker axis still clears the threshold.
    stats = _coverage_iou_stats(cov_a, cov_b, min_iou_threshold=min_iou_threshold)
    return bool(stats["merge_candidate"]), stats


# Choose which coverage contributes a given prediction segment.
def _local_path_key(cov: dict, x: int):
    y = int(cov["x_to_y"][x])
    # Prefer the stronger local cell, then the stronger overall coverage, with stable tie-breakers.
    return (
        float(cov["x_to_score"][x]),
        float(cov.get("total_score", 0.0)),
        float(cov.get("mean_score", 0.0)),
        -float(y),
    )


# Merge one connected overlap component into one coverage object.
def _merge_component(component_coverages: list[dict], matrix: np.ndarray) -> dict:
    # A singleton component already represents one final merged coverage.
    if len(component_coverages) == 1:
        return dict(component_coverages[0])

    merged_x_to_y: dict[int, int] = {}
    merged_x_to_score: dict[int, float] = {}
    all_x = sorted({x for cov in component_coverages for x in cov["x_to_y"]})
    fallback_cov = max(
        component_coverages,
        key=lambda cov: (
            float(cov.get("total_score", 0.0)),
            float(cov.get("mean_score", 0.0)),
            int(len(cov.get("pred_segments", ()))),
            float(cov["line"].get("support", 0.0)),
            float(cov["line"].get("length", 0.0)),
        ),
    )

    # For each covered prediction segment, keep the locally stronger path sample across the whole component.
    for x in all_x:
        candidates = [cov for cov in component_coverages if x in cov["x_to_y"]]
        best_cov = max(candidates, key=lambda cov: _local_path_key(cov, x))
        merged_x_to_y[int(x)] = int(best_cov["x_to_y"][x])
        merged_x_to_score[int(x)] = float(best_cov["x_to_score"][x])

    source_raw_line_ids = sorted(
        {int(raw_id) for cov in component_coverages for raw_id in cov.get("source_raw_line_ids", []) if int(raw_id) >= 0}
    )

    # Recompute one representative line from the merged text coverage rather than from the raw geometry.
    return _coverage_from_path(
        x_to_y=merged_x_to_y,
        x_to_score=merged_x_to_score,
        matrix=matrix,
        fallback_line=fallback_cov["line"],
        source_raw_line_ids=source_raw_line_ids,
    )


# Build connected components from an explicit adjacency graph.
def _components_from_adjacency(adjacency: dict[int, set[int]]) -> list[list[int]]:
    # No nodes means no components either.
    if not adjacency:
        return []

    components: list[list[int]] = []
    seen: set[int] = set()

    # Walk the overlap graph so any transitive true-IoU edge lands in one merge component.
    for start in sorted(adjacency):
        if start in seen:
            continue

        stack = [int(start)]
        seen.add(int(start))
        component: list[int] = []

        while stack:
            idx = stack.pop()
            component.append(int(idx))
            for other in sorted(adjacency[idx]):
                if other in seen:
                    continue
                seen.add(int(other))
                stack.append(int(other))

        components.append(sorted(component))

    return components


# Group overlapping coverages into connected components under the true-IoU rule.
def _coverage_components(
    coverages: list[dict],
    *,
    min_iou_threshold: float,
) -> tuple[list[list[int]], list[dict]]:
    # No coverages means no overlap graph either.
    if not coverages:
        return [], []

    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(coverages))}
    pairwise_stats: list[dict] = []

    # Evaluate every pair once so we can both build components and save the exact IoU diagnostics.
    for idx in range(len(coverages)):
        for other in range(idx + 1, len(coverages)):
            overlaps, stats = _coverages_overlap(
                coverages[idx],
                coverages[other],
                min_iou_threshold=min_iou_threshold,
            )
            stats["coverage_index_a"] = int(idx)
            stats["coverage_index_b"] = int(other)
            pairwise_stats.append(stats)
            if not overlaps:
                continue
            adjacency[idx].add(int(other))
            adjacency[other].add(int(idx))

    return _components_from_adjacency(adjacency), pairwise_stats


# Convert one coverage object into a compact debug-friendly summary.
def _coverage_debug_summary(cov: dict, *, coverage_index: int | None = None) -> dict:
    line = cov["line"]
    summary = {
        "source_raw_line_ids": sorted(int(v) for v in cov.get("source_raw_line_ids", [])),
        "pred_segment_count": int(len(cov.get("pred_segments", ()))),
        "ref_segment_count": int(len(cov.get("ref_segments", ()))),
        "total_score": float(cov.get("total_score", 0.0)),
        "mean_score": float(cov.get("mean_score", 0.0)),
        "anchor_y": float(cov.get("anchor_y", 0.0)),
        "x0": float(line.get("x0", 0.0)),
        "y0": float(line.get("y0", 0.0)),
        "x1": float(line.get("x1", 0.0)),
        "y1": float(line.get("y1", 0.0)),
        "length": float(line.get("length", 0.0)),
        "support": float(line.get("support", 0.0)),
        "score": float(line.get("score", 0.0)),
    }
    if coverage_index is not None:
        summary["coverage_index"] = int(coverage_index)
    return summary


# Normalize the raw Hough lines into a credible v2.1 candidate set.
def _prepare_candidates(lines: list[dict], matrix: np.ndarray, *, abs_min_len: float) -> list[dict]:
    # Empty inputs stay empty so the caller can keep the standard fallback behavior.
    if not lines:
        return []

    max_score = max(float(ln.get("score", 0.0)) for ln in lines)
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0
    candidates: list[dict] = []

    for raw_line_id, ln in enumerate(lines):
        ln2 = dict(ln)
        # Recompute geometry and support so all candidates are judged on the same baseline.
        ln2["raw_line_id"] = int(raw_line_id)
        ln2["length"] = line_length(ln2)
        ln2["support"] = mean_line_support(matrix, ln2)

        # Keep only obviously credible candidates before the coverage-based merge stage.
        if ln2["length"] < float(abs_min_len):
            continue
        if max_score > 0 and float(ln2.get("score", 0.0)) < 0.06 * max_score:
            continue
        if ln2["support"] < support_floor:
            continue
        candidates.append(ln2)

    # Preserve the old "keep the single best raw line" fallback when everything fails the coarse gates.
    if not candidates:
        best_raw_line_id, best = max(
            enumerate(lines),
            key=lambda pair: float(pair[1].get("score", 0.0)),
        )
        best2 = dict(best)
        best2["raw_line_id"] = int(best_raw_line_id)
        best2["length"] = line_length(best2)
        best2["support"] = mean_line_support(matrix, best2)
        candidates = [best2]

    return sorted(candidates, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))


# Analyze the full v2.1 filtering pipeline and expose detailed IoU/debug state.
def analyze_line_filtering(
    lines: list[dict],
    matrix: np.ndarray,
    *,
    abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
) -> dict:
    # Keep the analysis self-contained so callers can save a full debug bundle without duplicating logic.
    candidates = _prepare_candidates(lines, matrix, abs_min_len=abs_min_len)
    coverages = [cov for cov in (_build_line_coverage(ln, matrix) for ln in candidates) if cov is not None]
    components, pairwise_iou = _coverage_components(
        coverages,
        min_iou_threshold=min_iou_threshold,
    )
    merged_coverages = [_merge_component([coverages[idx] for idx in component], matrix) for component in components]

    return {
        "candidate_lines": [dict(ln) for ln in candidates],
        "candidate_coverages": [
            _coverage_debug_summary(cov, coverage_index=idx) for idx, cov in enumerate(coverages)
        ],
        "pairwise_iou": pairwise_iou,
        "components": [
            {
                "component_index": int(component_index),
                "coverage_indices": [int(idx) for idx in component],
                "source_raw_line_ids": sorted(
                    {
                        int(raw_id)
                        for idx in component
                        for raw_id in coverages[idx].get("source_raw_line_ids", [])
                        if int(raw_id) >= 0
                    }
                ),
            }
            for component_index, component in enumerate(components)
        ],
        "merged_coverages": [
            _coverage_debug_summary(cov, coverage_index=idx) for idx, cov in enumerate(merged_coverages)
        ],
        "merged_coverage_objects": merged_coverages,
    }


# Assign each prediction column to the strongest surviving coverage.
def _compute_final_assignment(coverages: list[dict], matrix: np.ndarray) -> dict[str, np.ndarray]:
    n_ref, n_pred = matrix.shape
    assignment = _empty_assignment(n_pred)
    mapped_y = assignment["mapped_y"]
    mapped_line_id = assignment["mapped_line_id"]

    # Compete only among the surviving coverages, using the local matrix score at each x.
    for x in range(n_pred):
        best = None
        best_key = None

        for lid, cov in enumerate(coverages):
            if x not in cov["x_to_y"]:
                continue

            y_idx = int(cov["x_to_y"][x])
            key = _local_path_key(cov, x) + (
                float(cov["line"].get("length", 0.0)),
                -float(lid),
            )
            if best_key is None or key > best_key:
                best_key = key
                best = (lid, y_idx)

        if best is None:
            continue

        lid, y_idx = best
        mapped_line_id[x] = int(lid)
        mapped_y[x] = float(np.clip(y_idx, 0, n_ref - 1))

    return assignment


# Convert the merged coverages into final output lines and per-column assignment.
def _finalize_outputs(
    coverages: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
    current = sorted(
        coverages,
        key=lambda cov: (
            float(cov.get("anchor_y", 0.0)),
            min(cov["line"]["x0"], cov["line"]["x1"]),
        ),
    )
    if not current:
        return [], _empty_assignment(n_pred)

    # Recompute until every surviving representative line owns at least one prediction segment.
    while True:
        assignment = _compute_final_assignment(current, matrix)
        mapped_line_id = np.asarray(assignment["mapped_line_id"], dtype=int)
        keep_ids = [lid for lid in range(len(current)) if np.any(mapped_line_id == lid)]
        if len(keep_ids) == len(current):
            break
        current = [current[lid] for lid in keep_ids]
        current = sorted(
            current,
            key=lambda cov: (
                float(cov.get("anchor_y", 0.0)),
                min(cov["line"]["x0"], cov["line"]["x1"]),
            ),
        )
        if not current:
            return [], _empty_assignment(n_pred)

    mapped_y = np.asarray(assignment["mapped_y"], dtype=float)
    mapped_line_id = np.asarray(assignment["mapped_line_id"], dtype=int)
    final_lines: list[dict] = []

    # Re-express the merged text coverage in the same fields that current reports already expect.
    for lid, cov in enumerate(current):
        line = dict(cov["line"])
        owned_cols = [int(x) for x in np.flatnonzero(mapped_line_id == lid)]
        owned_scores = [float(cov["x_to_score"][x]) for x in owned_cols if x in cov["x_to_score"]]
        owned_rows = [
            int(np.clip(round(float(mapped_y[x])), 0, mask_bool.shape[0] - 1))
            for x in owned_cols
            if mask_bool.shape[0] > 0
        ]

        span_cols = max(1, len(cov["pred_segments"]))
        owned_mask_hits = 0
        if mask_bool.shape[0] > 0 and mask_bool.shape[1] > 0:
            for x, y in zip(owned_cols, owned_rows):
                if 0 <= x < mask_bool.shape[1] and 0 <= y < mask_bool.shape[0]:
                    owned_mask_hits += int(bool(mask_bool[y, x]))

        line["source_raw_line_ids"] = sorted(int(v) for v in cov.get("source_raw_line_ids", []))
        line["owned_cols"] = int(len(owned_cols))
        line["owned_fraction"] = float(len(owned_cols) / span_cols)
        line["owned_score_mean"] = float(np.mean(owned_scores)) if owned_scores else 0.0
        line["owned_score_sum"] = float(np.sum(owned_scores)) if owned_scores else 0.0
        line["owned_mask_hits"] = int(owned_mask_hits)
        line["owned_mask_fraction"] = float(owned_mask_hits / len(owned_cols)) if owned_cols else 0.0
        line["anchor_y"] = (
            float(np.median(owned_rows))
            if owned_rows
            else float(cov.get("anchor_y", min(line["y0"], line["y1"])))
        )
        final_lines.append(line)

    # Keep the final line order in stable reading order.
    final_lines = sorted(
        final_lines,
        key=lambda ln: (float(ln.get("anchor_y", min(ln["y0"], ln["y1"]))), min(ln["x0"], ln["x1"])),
    )
    return final_lines, assignment


# Filter lines using true IoU over prediction/reference coverage rather than the v1 ownership geometry.
def filter_lines_for_alignment_by_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    **_ignored,
):
    # Keep the minimum public surface needed by v2.1 while still accepting old v1-style keyword calls.
    if not lines:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    if matrix.size == 0:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    # Preserve the same shape check as v1 even though v2.1 uses the mask only for output statistics.
    if mask_bool.shape != matrix.shape:
        raise ValueError(f"mask_bool shape {mask_bool.shape} does not match matrix shape {matrix.shape}")

    analysis = analyze_line_filtering(
        lines,
        matrix,
        abs_min_len=abs_min_len,
        min_iou_threshold=min_iou_threshold,
    )
    return _finalize_outputs(list(analysis["merged_coverage_objects"]), matrix, mask_bool)
