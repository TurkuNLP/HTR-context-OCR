import math

import numpy as np


__all__ = [
    "line_y_at_x",
    "line_length",
    "mean_line_support",
    "filter_lines_for_alignment",
    "filter_lines_for_alignment_by_ownership",
]


# Interpolate a line's y-position at a specific prediction-column x.
def line_y_at_x(line: dict, x: int) -> float:
    # Work with the segment endpoints directly so every caller uses the same interpolation rule.
    x0, y0, x1, y1 = line["x0"], line["y0"], line["x1"], line["y1"]
    dx = x1 - x0
    # Treat vertical or nearly vertical lines as constant-y to avoid unstable division.
    if abs(dx) < 1e-8:
        return y0
    # Convert x into a segment-relative interpolation factor and evaluate the matching y.
    t = (x - x0) / dx
    return y0 + t * (y1 - y0)


# Measure the Euclidean length of a line segment.
def line_length(line: dict) -> float:
    # Use the endpoint distance so length comparisons stay geometry-based.
    return float(math.hypot(line["x1"] - line["x0"], line["y1"] - line["y0"]))


# Average the matrix values sampled directly along a line.
def mean_line_support(matrix: np.ndarray, line: dict) -> float:
    # Empty matrices cannot provide any support signal.
    if matrix.size == 0:
        return 0.0

    n_ref, n_pred = matrix.shape
    # Clamp the sampled x-range to the valid prediction-column interval.
    x_start = int(max(0, math.ceil(min(line["x0"], line["x1"]))))
    x_end = int(min(n_pred - 1, math.floor(max(line["x0"], line["x1"]))))
    if x_end < x_start:
        return 0.0

    vals = []
    # Sample each covered column at the line's interpolated y-position.
    for x in range(x_start, x_end + 1):
        y_idx = int(np.clip(round(line_y_at_x(line, x)), 0, n_ref - 1))
        vals.append(float(matrix[y_idx, x]))
    # Return the mean sampled support so long and short lines stay comparable.
    return float(np.mean(vals)) if vals else 0.0


# Clamp a line's x-span to valid prediction-column bounds.
def _line_x_bounds(line: dict, n_pred: int):
    # No prediction columns means no usable span.
    if n_pred <= 0:
        return None
    # Convert the geometric span into inclusive matrix-column bounds.
    x_min = max(0, int(math.floor(min(line["x0"], line["x1"]))))
    x_max = min(n_pred - 1, int(math.ceil(max(line["x0"], line["x1"]))))
    if x_max < x_min:
        return None
    return x_min, x_max


# Check whether a candidate line is supported by the active mask near a sampled point.
def _mask_hit_at_x(mask_bool: np.ndarray, x: int, y_idx: int, radius: int) -> bool:
    # Empty masks cannot confirm support anywhere.
    if mask_bool.size == 0:
        return False

    n_ref, n_pred = mask_bool.shape
    # Ignore out-of-bounds columns before slicing the mask.
    if x < 0 or x >= n_pred:
        return False

    # Search a small vertical neighborhood to tolerate rounding and local detector jitter.
    rad = max(0, int(radius))
    y0 = max(0, int(y_idx) - rad)
    y1 = min(n_ref - 1, int(y_idx) + rad)
    return bool(np.any(mask_bool[y0 : y1 + 1, x]))


# Rank competing lines when assigning one prediction column to one owner.
def _candidate_line_key(line: dict, *, local_score: float, y_idx: int, mask_hit: bool, lid: int):
    # Prefer mask-supported, high-scoring, stronger global lines, with stable tie-breakers.
    return (
        int(mask_hit),
        float(local_score),
        float(line.get("support", 0.0)),
        float(line.get("score", 0.0)),
        -float(y_idx),
        float(line.get("length", 0.0)),
        -float(lid),
    )


# Assign each prediction column to the strongest supported line and collect stats.
def _compute_line_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    mask_radius: int = 1,
) -> dict:
    n_ref, n_pred = matrix.shape
    # Initialize per-column outputs with "unassigned" sentinels.
    mapped_y = np.full(n_pred, np.nan, dtype=float)
    mapped_line_id = np.full(n_pred, -1, dtype=int)

    # Precompute each line's usable x-range and allocate per-line ownership accumulators.
    bounds = [_line_x_bounds(ln, n_pred) for ln in lines]
    stats = []
    for lid, ln in enumerate(lines):
        bound = bounds[lid]
        span_cols = 0 if bound is None else (bound[1] - bound[0] + 1)
        stats.append(
            {
                "x_start": None if bound is None else int(bound[0]),
                "x_end": None if bound is None else int(bound[1]),
                "span_cols": int(max(span_cols, 0)),
                "owned_x": [],
                "owned_y": [],
                "owned_scores": [],
                "owned_mask_hits": 0,
                "anchor_y": float(min(ln["y0"], ln["y1"])),
            }
        )

    # Let the candidate lines compete for each prediction column independently.
    for x in range(n_pred):
        best = None
        best_key = None

        for lid, ln in enumerate(lines):
            bound = bounds[lid]
            if bound is None or x < bound[0] or x > bound[1]:
                continue

            # Sample the line at this column and score the proposed ownership point.
            y_est = line_y_at_x(ln, x)
            y_idx = int(np.clip(round(y_est), 0, n_ref - 1))
            local_score = float(matrix[y_idx, x])
            mask_hit = _mask_hit_at_x(mask_bool, x, y_idx, mask_radius)
            key = _candidate_line_key(ln, local_score=local_score, y_idx=y_idx, mask_hit=mask_hit, lid=lid)

            if best_key is None or key > best_key:
                best_key = key
                best = (lid, y_idx, local_score, mask_hit)

        # Leave the column unowned if no line covered it.
        if best is None:
            continue

        lid, y_idx, local_score, mask_hit = best
        # Even the winning line must still be confirmed by the active Hough mask.
        if not mask_hit:
            continue

        # Record the winning line and accumulate its owned evidence.
        mapped_y[x] = float(y_idx)
        mapped_line_id[x] = int(lid)
        stats[lid]["owned_x"].append(int(x))
        stats[lid]["owned_y"].append(int(y_idx))
        stats[lid]["owned_scores"].append(float(local_score))
        stats[lid]["owned_mask_hits"] += int(mask_hit)

    # Collapse the raw ownership traces into summary statistics used by later filters.
    for stat in stats:
        owned_cols = len(stat["owned_x"])
        span_cols = int(stat["span_cols"])
        stat["owned_cols"] = int(owned_cols)
        stat["owned_fraction"] = float(owned_cols / span_cols) if span_cols > 0 else 0.0
        stat["owned_score_mean"] = float(np.mean(stat["owned_scores"])) if stat["owned_scores"] else 0.0
        stat["owned_score_sum"] = float(np.sum(stat["owned_scores"])) if stat["owned_scores"] else 0.0
        stat["owned_mask_fraction"] = (
            float(stat["owned_mask_hits"] / owned_cols) if owned_cols > 0 else 0.0
        )
        # Use the median owned y as a stable anchor for top-to-bottom ordering.
        if stat["owned_y"]:
            stat["anchor_y"] = float(np.median(stat["owned_y"]))

    return {
        "mapped_y": mapped_y,
        "mapped_line_id": mapped_line_id,
        "stats": stats,
    }


# Attach ownership statistics back onto each line dictionary.
def _decorate_lines_with_ownership(lines: list[dict], stats: list[dict]) -> list[dict]:
    out = []
    for lid, ln in enumerate(lines):
        stat = stats[lid]
        # Copy the original geometry first so callers keep a line-shaped dictionary.
        ln2 = dict(ln)
        # Add the ownership-derived measurements used by downstream ranking and filtering.
        ln2["owned_cols"] = int(stat.get("owned_cols", 0))
        ln2["owned_fraction"] = float(stat.get("owned_fraction", 0.0))
        ln2["owned_score_mean"] = float(stat.get("owned_score_mean", 0.0))
        ln2["owned_score_sum"] = float(stat.get("owned_score_sum", 0.0))
        ln2["owned_mask_hits"] = int(stat.get("owned_mask_hits", 0))
        ln2["owned_mask_fraction"] = float(stat.get("owned_mask_fraction", 0.0))
        ln2["anchor_y"] = float(stat.get("anchor_y", min(ln2["y0"], ln2["y1"])))
        out.append(ln2)
    return out


# Convert a line segment into its direction angle in degrees.
def _line_angle_deg(line: dict) -> float:
    # Express the segment orientation in a common angle space for overlap checks.
    return float(math.degrees(math.atan2(line["y1"] - line["y0"], line["x1"] - line["x0"])))


# Measure the smallest angular difference between two line directions.
def _line_angle_difference_deg(line_a: dict, line_b: dict) -> float:
    # Start with the absolute direction difference.
    angle_diff = abs(_line_angle_deg(line_a) - _line_angle_deg(line_b))
    # Fold obtuse differences back into the smaller equivalent acute angle.
    if angle_diff > 90.0:
        angle_diff = 180.0 - angle_diff
    return float(angle_diff)


# Return the ordered x-span covered by a line segment.
def _line_x_span(line: dict) -> tuple[float, float]:
    # Normalize the endpoint order so later overlap math can assume x0 <= x1.
    return float(min(line["x0"], line["x1"])), float(max(line["x0"], line["x1"]))


# Measure x-overlap relative to the shorter of two line spans.
def _line_x_overlap_ratio(line_a: dict, line_b: dict) -> float:
    ax0, ax1 = _line_x_span(line_a)
    bx0, bx1 = _line_x_span(line_b)
    # Compare only the shared x-range between the two line spans.
    overlap = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    if overlap <= 0.0:
        return 0.0
    # Normalize by the shorter span so containment-like cases approach 1.0.
    shorter = max(1e-8, min(ax1 - ax0, bx1 - bx0))
    return float(overlap / shorter)


# Measure vertical separation where two lines overlap in x.
def _line_mean_y_separation_over_overlap(line_a: dict, line_b: dict) -> float:
    ax0, ax1 = _line_x_span(line_a)
    bx0, bx1 = _line_x_span(line_b)
    # Restrict the comparison to the integer columns shared by both lines.
    x_start = int(max(math.ceil(ax0), math.ceil(bx0)))
    x_end = int(min(math.floor(ax1), math.floor(bx1)))
    if x_end < x_start:
        return float("inf")

    vals = []
    # Sample both lines over the overlap and measure their row separation.
    for x in range(x_start, x_end + 1):
        vals.append(abs(line_y_at_x(line_a, x) - line_y_at_x(line_b, x)))
    return float(np.mean(vals)) if vals else float("inf")


# Collapse a line's owned evidence into one comparison score.
def _line_total_owned_support(line: dict) -> float:
    # Prefer the explicit sum when it is already available from ownership statistics.
    if "owned_score_sum" in line:
        return float(line.get("owned_score_sum", 0.0))
    # Otherwise reconstruct a comparable total from mean support times owned width.
    return float(line.get("owned_score_mean", 0.0)) * float(line.get("owned_cols", 0))


# Build the sort key that prefers stronger owned lines first.
def _line_quality_key(line: dict):
    # Order by total owned evidence first, then use coverage and geometry as tie-breakers.
    return (
        _line_total_owned_support(line),
        int(line.get("owned_cols", 0)),
        float(line.get("owned_fraction", 0.0)),
        float(line.get("support", 0.0)),
        float(line.get("score", 0.0)),
        float(line.get("length", 0.0)),
    )


# Fit one merged segment through two overlapping same-ridge lines.
def _merge_two_overlapping_lines(line_a: dict, line_b: dict, matrix: np.ndarray) -> dict:
    # Fit the merged line against all four endpoints so complementary fragments become one segment.
    points = np.asarray(
        [
            [float(line_a["x0"]), float(line_a["y0"])],
            [float(line_a["x1"]), float(line_a["y1"])],
            [float(line_b["x0"]), float(line_b["y0"])],
            [float(line_b["x1"]), float(line_b["y1"])],
        ],
        dtype=float,
    )
    x_vals = points[:, 0]
    y_vals = points[:, 1]

    # Expand the merged span to cover the union of both line fragments.
    x0 = float(np.min(x_vals))
    x1 = float(np.max(x_vals))
    if abs(x1 - x0) < 1e-8:
        y0 = float(np.min(y_vals))
        y1 = float(np.max(y_vals))
    else:
        # Re-fit one best straight line across the combined endpoints.
        slope, intercept = np.polyfit(x_vals, y_vals, deg=1)
        y0 = float((slope * x0) + intercept)
        y1 = float((slope * x1) + intercept)

    # Recompute the merged line's support metrics so later ranking uses current geometry.
    merged = {"x0": x0, "y0": y0, "x1": x1, "y1": y1}
    merged["length"] = line_length(merged)
    merged["support"] = mean_line_support(matrix, merged)
    merged["score"] = float(merged["support"])
    return merged


# Build a rounded geometric signature for convergence checks.
def _line_signature(line: dict) -> tuple[float, float, float, float]:
    # Round geometry so tiny floating-point changes do not prevent fixed-point detection.
    return (
        round(float(line["x0"]), 3),
        round(float(line["y0"]), 3),
        round(float(line["x1"]), 3),
        round(float(line["y1"]), 3),
    )


# Merge or suppress overlapping lines after an ownership pass.
def _resolve_overlapping_lines_with_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    mask_radius: int,
    overlap_angle_tol_deg: float,
    overlap_min_x_ratio: float,
    overlap_max_mean_y_sep: float,
    overlap_containment_ratio: float,
    max_rounds: int = 4,
):
    # Start from ownership-decorated lines so overlap decisions can use current evidence.
    current = [dict(ln) for ln in lines]
    assignment = _compute_line_ownership(current, matrix, mask_bool, mask_radius=mask_radius)
    current = _decorate_lines_with_ownership(current, assignment["stats"])
    current = sorted(
        current,
        key=lambda ln: (float(ln.get("anchor_y", min(ln["y0"], ln["y1"]))), min(ln["x0"], ln["x1"])),
    )

    # Iterate a few times because merging one pair can change ownership for the next pair.
    for _ in range(max_rounds):
        prev_signature = tuple(_line_signature(ln) for ln in current)
        # Resolve from strongest to weakest so high-evidence lines claim duplicates first.
        ordered = sorted(current, key=_line_quality_key, reverse=True)
        resolved: list[dict] = []
        changed = False

        for line in ordered:
            matched = False
            for idx, kept in enumerate(resolved):
                # Reject pairs that are too different in direction to represent one ridge.
                angle_diff = _line_angle_difference_deg(kept, line)
                if angle_diff > float(overlap_angle_tol_deg):
                    continue

                # Require a meaningful shared x-span before treating the pair as overlapping.
                overlap_ratio = _line_x_overlap_ratio(kept, line)
                if overlap_ratio < float(overlap_min_x_ratio):
                    continue

                # Reject nearby but vertically separated lines that likely belong to different text lines.
                mean_y_sep = _line_mean_y_separation_over_overlap(kept, line)
                if mean_y_sep > float(overlap_max_mean_y_sep):
                    continue

                matched = True
                changed = True
                # If one line is mostly contained in the other, keep only the stronger resolved line.
                if overlap_ratio >= float(overlap_containment_ratio):
                    break

                # Otherwise merge complementary fragments into one longer fitted replacement.
                resolved[idx] = _merge_two_overlapping_lines(kept, line, matrix)
                break

            # Keep unmatched lines as separate survivors for this round.
            if not matched:
                resolved.append(dict(line))

        # Recompute ownership after each round so later comparisons use updated coverage.
        resolved = sorted(resolved, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))
        assignment = _compute_line_ownership(resolved, matrix, mask_bool, mask_radius=mask_radius)
        current = _decorate_lines_with_ownership(resolved, assignment["stats"])
        current = sorted(
            current,
            key=lambda ln: (float(ln.get("anchor_y", min(ln["y0"], ln["y1"]))), min(ln["x0"], ln["x1"])),
        )
        if not changed:
            break
        cur_signature = tuple(_line_signature(ln) for ln in current)
        if cur_signature == prev_signature:
            break

    # Return one last ownership pass so callers receive consistent final stats and mappings.
    assignment = _compute_line_ownership(current, matrix, mask_bool, mask_radius=mask_radius)
    current = _decorate_lines_with_ownership(current, assignment["stats"])
    current = sorted(
        current,
        key=lambda ln: (float(ln.get("anchor_y", min(ln["y0"], ln["y1"]))), min(ln["x0"], ln["x1"])),
    )
    return current, assignment


# Keep only robust line candidates for the basic alignment path.
def filter_lines_for_alignment(
    lines: list[dict],
    matrix: np.ndarray,
    *,
    min_len_ratio: float = 0.08,
) -> list[dict]:
    # The basic path falls back to an empty result immediately when nothing was detected.
    if not lines:
        return []

    # Derive simple global thresholds from the detected line set and score matrix.
    max_score = max(float(ln.get("score", 0.0)) for ln in lines)
    min_dim = min(matrix.shape) if matrix.size > 0 else 1
    min_len = max(8.0, float(min_len_ratio) * float(min_dim))
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0

    kept = []
    for ln in lines:
        ln2 = dict(ln)
        # Recompute geometry and support so every candidate is judged with the same metrics.
        ln2["length"] = line_length(ln2)
        ln2["support"] = mean_line_support(matrix, ln2)

        # Drop short, globally weak, or low-support lines before alignment ordering.
        if ln2["length"] < min_len:
            continue
        if max_score > 0 and float(ln2.get("score", 0.0)) < 0.06 * max_score:
            continue
        if ln2["support"] < support_floor:
            continue
        kept.append(ln2)

    # If every candidate failed, keep the single best-scoring raw line as a fallback.
    if not kept:
        best = max(lines, key=lambda ln: float(ln.get("score", 0.0)))
        best2 = dict(best)
        best2["length"] = line_length(best2)
        best2["support"] = mean_line_support(matrix, best2)
        return [best2]

    # Sort surviving lines into stable reading order.
    return sorted(kept, key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])))


# Keep only robust line candidates using ownership and overlap resolution.
def filter_lines_for_alignment_by_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    abs_min_len: float = 8.0,                # Minimum raw line length in matrix cells; 8 keeps obvious tiny fragments out without removing typical short real lines.
    mask_radius: int = 1,                    # Vertical tolerance when checking mask support at each x; 1 allows small rounding noise while staying locally strict.
    min_owned_cols: int = 6,                 # A line must win at least this many prediction columns; 6 is low enough for short true lines but removes near-empty survivors.
    min_owned_fraction: float = 0.12,        # A line must own at least 12% of its claimed x-span; permissive enough for fragmented lines, but rejects lines that explain only a sliver.
    overlap_angle_tol_deg: float = 5.0,      # Lines must be within 5 degrees to count as same-ridge candidates; this allows small fit noise without merging clearly different slopes.
    overlap_min_x_ratio: float = 0.5,        # At least half of the shorter line must overlap in x before duplicate handling starts; this is conservative and is the gate that currently lets lines 5 and 6 both survive.
    overlap_max_mean_y_sep: float = 3.0,     # Mean vertical separation over the shared x-range must stay within 3 rows; this avoids collapsing nearby parallel text lines.
    overlap_containment_ratio: float = 0.85, # If 85% of the shorter line is overlapped, treat it as mostly contained and drop it instead of fitting a merged replacement.
):
    # Return empty ownership arrays when there are no candidate lines at all.
    if not lines:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], {
            "mapped_y": np.full(n_pred, np.nan, dtype=float),
            "mapped_line_id": np.full(n_pred, -1, dtype=int),
        }

    # Preserve the same empty-shape fallback when the score matrix itself is empty.
    if matrix.size == 0:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], {
            "mapped_y": np.full(n_pred, np.nan, dtype=float),
            "mapped_line_id": np.full(n_pred, -1, dtype=int),
        }

    # Ownership only makes sense when the Hough support mask matches the matrix geometry.
    if mask_bool.shape != matrix.shape:
        raise ValueError(
            f"mask_bool shape {mask_bool.shape} does not match matrix shape {matrix.shape}"
        )

    # Build the same global score and support gates used to remove obviously weak candidates.
    max_score = max(float(ln.get("score", 0.0)) for ln in lines)
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0

    candidates = []
    for ln in lines:
        ln2 = dict(ln)
        # Normalize each raw line into a candidate with explicit geometry and support metrics.
        ln2["length"] = line_length(ln2)
        ln2["support"] = mean_line_support(matrix, ln2)

        # Remove short, weak, or globally unsupported lines before ownership competition.
        if ln2["length"] < float(abs_min_len):
            continue
        if max_score > 0 and float(ln2.get("score", 0.0)) < 0.06 * max_score:
            continue
        if ln2["support"] < support_floor:
            continue
        candidates.append(ln2)

    # If everything was rejected, keep the best raw line so downstream code still has one guide.
    if not candidates:
        best = max(lines, key=lambda ln: float(ln.get("score", 0.0)))
        best2 = dict(best)
        best2["length"] = line_length(best2)
        best2["support"] = mean_line_support(matrix, best2)
        candidates = [best2]

    # Compute ownership once so each candidate can be judged by how much matrix territory it really explains.
    initial_assignment = _compute_line_ownership(
        candidates,
        matrix,
        mask_bool,
        mask_radius=mask_radius,
    )
    initial_stats = initial_assignment["stats"]

    kept = []
    for lid, ln in enumerate(candidates):
        stat = initial_stats[lid]
        # Keep only lines that win enough columns and own a meaningful fraction of their claimed span.
        if (
            int(stat.get("owned_cols", 0)) >= int(min_owned_cols)
            and float(stat.get("owned_fraction", 0.0)) >= float(min_owned_fraction)
        ):
            kept.append(dict(ln))

    # If every line failed the ownership thresholds, keep the best-owned fallback candidate.
    if not kept:
        best_lid = max(
            range(len(candidates)),
            key=lambda lid: (
                int(initial_stats[lid].get("owned_cols", 0)),
                float(initial_stats[lid].get("owned_fraction", 0.0)),
                int(initial_stats[lid].get("owned_mask_hits", 0)),
                float(candidates[lid].get("score", 0.0)),
                float(candidates[lid].get("support", 0.0)),
                float(candidates[lid].get("length", 0.0)),
                -float(initial_stats[lid].get("anchor_y", 0.0)),
            ),
        )
        kept = [dict(candidates[best_lid])]

    # Decorate the kept lines, resolve overlaps, then recompute ownership on the final geometry.
    kept = sorted(
        kept,
        key=lambda ln: (min(ln["y0"], ln["y1"]), min(ln["x0"], ln["x1"])),
    )
    final_assignment = _compute_line_ownership(
        kept,
        matrix,
        mask_bool,
        mask_radius=mask_radius,
    )
    kept = _decorate_lines_with_ownership(kept, final_assignment["stats"])
    kept, final_assignment = _resolve_overlapping_lines_with_ownership(
        kept,
        matrix,
        mask_bool,
        mask_radius=mask_radius,
        overlap_angle_tol_deg=overlap_angle_tol_deg,
        overlap_min_x_ratio=overlap_min_x_ratio,
        overlap_max_mean_y_sep=overlap_max_mean_y_sep,
        overlap_containment_ratio=overlap_containment_ratio,
    )
    kept = sorted(
        kept,
        key=lambda ln: (float(ln.get("anchor_y", min(ln["y0"], ln["y1"]))), min(ln["x0"], ln["x1"])),
    )
    final_assignment = _compute_line_ownership(
        kept,
        matrix,
        mask_bool,
        mask_radius=mask_radius,
    )
    kept = _decorate_lines_with_ownership(kept, final_assignment["stats"])

    # Return both the final kept lines and the final per-column ownership mapping.
    return kept, {
        "mapped_y": final_assignment["mapped_y"],
        "mapped_line_id": final_assignment["mapped_line_id"],
    }
