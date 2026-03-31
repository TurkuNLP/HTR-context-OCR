import math
import numpy as np
from line_filtering import line_length, line_y_at_x, mean_line_support

DEFAULT_ABS_MIN_LEN = 6.0

__all__ = [
    "DEFAULT_ABS_MIN_LEN",
    "filter_lines_for_alignment_by_ownership",
]


# Build the empty per-column mapping used when no final guides survive.
def _empty_assignment(n_pred: int) -> dict[str, np.ndarray]:
    # Match the v1 return shape so downstream scripts do not need to special-case v2.
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
    return {
        "line": line,
        "pred_segments": pred_segments,
        "ref_segments": ref_segments,
        "x_to_y": {int(k): int(v) for k, v in x_to_y.items()},
        "x_to_score": {int(k): float(v) for k, v in x_to_score.items()},
        "total_score": total_score,
        "mean_score": mean_score,
        "anchor_y": anchor_y,
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
    )


# Decide whether two coverages overlap on both prediction and reference text coverage.
def _coverages_overlap(cov_a: dict, cov_b: dict) -> bool:
    # The v2 candidate rule is simply positive overlap on both text axes.
    shared_pred = cov_a["pred_segments"] & cov_b["pred_segments"]
    if not shared_pred:
        return False
    shared_ref = cov_a["ref_segments"] & cov_b["ref_segments"]
    return bool(shared_ref)


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

    # Recompute one representative line from the merged text coverage rather than from the raw geometry.
    return _coverage_from_path(
        x_to_y=merged_x_to_y,
        x_to_score=merged_x_to_score,
        matrix=matrix,
        fallback_line=fallback_cov["line"],
    )


# Group overlapping coverages into connected components.
def _coverage_components(coverages: list[dict]) -> list[list[int]]:
    # No coverages means no overlap graph either.
    if not coverages:
        return []

    components: list[list[int]] = []
    seen: set[int] = set()

    # Walk the implicit overlap graph so any transitive overlap lands in one merge component.
    for start in range(len(coverages)):
        if start in seen:
            continue

        stack = [int(start)]
        seen.add(int(start))
        component: list[int] = []

        while stack:
            idx = stack.pop()
            component.append(int(idx))

            for other in range(len(coverages)):
                if other in seen:
                    continue
                if _coverages_overlap(coverages[idx], coverages[other]):
                    seen.add(int(other))
                    stack.append(int(other))

        components.append(sorted(component))

    return components


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


# Normalize the raw Hough lines into a credible v2 candidate set.
def _prepare_candidates(lines: list[dict], matrix: np.ndarray, *, abs_min_len: float) -> list[dict]:
    # Empty inputs stay empty so the caller can keep the standard fallback behavior.
    if not lines:
        return []

    max_score = max(float(ln.get("score", 0.0)) for ln in lines)
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0
    candidates: list[dict] = []

    for ln in lines:
        ln2 = dict(ln)
        # Recompute geometry and support so all candidates are judged on the same baseline.
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
        best = max(lines, key=lambda ln: float(ln.get("score", 0.0)))
        best2 = dict(best)
        best2["length"] = line_length(best2)
        best2["support"] = mean_line_support(matrix, best2)
        candidates = [best2]

    return sorted(candidates, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))


# Filter lines using coverage overlap on prediction/reference segments rather than v1 ownership geometry.
def filter_lines_for_alignment_by_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    **_ignored,
):
    # Keep only the minimum public surface needed by v2 while still accepting old v1-style keyword calls.
    if not lines:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    if matrix.size == 0:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    # Preserve the same shape check as v1 even though v2 uses the mask only for output statistics.
    if mask_bool.shape != matrix.shape:
        raise ValueError(f"mask_bool shape {mask_bool.shape} does not match matrix shape {matrix.shape}")

    candidates = _prepare_candidates(lines, matrix, abs_min_len=abs_min_len)
    coverages = [cov for cov in (_build_line_coverage(ln, matrix) for ln in candidates) if cov is not None]
    if not coverages:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    # Merge all raw lines that belong to the same text-coverage component.
    components = _coverage_components(coverages)
    merged_coverages = [_merge_component([coverages[idx] for idx in component], matrix) for component in components]
    return _finalize_outputs(merged_coverages, matrix, mask_bool)
