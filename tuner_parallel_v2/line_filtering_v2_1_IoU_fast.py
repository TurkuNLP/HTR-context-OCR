from __future__ import annotations

import math

import numpy as np

try:
    from .runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime_paths import ensure_tuner_runtime_paths  # type: ignore

ensure_tuner_runtime_paths()

from line_filtering import line_length, line_y_at_x, mean_line_support
DEFAULT_ABS_MIN_LEN = 6.0
DEFAULT_MIN_IOU_THRESHOLD = 0.035


def _empty_assignment(n_pred: int) -> dict[str, np.ndarray]:
    return {
        "mapped_y": np.full(int(n_pred), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_pred), -1, dtype=int),
    }


def _line_x_bounds(line: dict, n_pred: int) -> tuple[int, int] | None:
    if n_pred <= 0:
        return None

    x_min = max(0, int(math.floor(min(line["x0"], line["x1"]))))
    x_max = min(n_pred - 1, int(math.ceil(max(line["x0"], line["x1"]))))
    if x_max < x_min:
        return None
    return x_min, x_max


def _ref_segments_from_path(x_to_y: dict[int, int]) -> set[int]:
    if not x_to_y:
        return set()

    rows: set[int] = set()
    prev_y: int | None = None
    for x in sorted(x_to_y):
        cur_y = int(x_to_y[x])
        rows.add(cur_y)
        if prev_y is not None:
            y0, y1 = sorted((prev_y, cur_y))
            for y in range(y0, y1 + 1):
                rows.add(int(y))
        prev_y = cur_y
    return rows


def _fit_line_from_path(
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    *,
    fallback_line: dict | None = None,
) -> dict:
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
    line["x0"] = x0
    line["y0"] = y0
    line["x1"] = x1
    line["y1"] = y1
    line["length"] = line_length(line)
    line["support"] = mean_line_support(matrix, line) if matrix.size else 0.0
    line["score"] = float(line["support"])
    return line


def _coverage_from_path(
    *,
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    fallback_line: dict | None = None,
    source_raw_line_ids: list[int] | None = None,
) -> dict:
    pred_segments = set(int(x) for x in x_to_y)
    ref_segments = _ref_segments_from_path(x_to_y)
    line = _fit_line_from_path(x_to_y, x_to_score, matrix, fallback_line=fallback_line)

    y_values = [int(x_to_y[x]) for x in sorted(x_to_y)]
    total_score = float(sum(float(v) for v in x_to_score.values()))
    mean_score = float(total_score / len(x_to_score)) if x_to_score else 0.0
    anchor_y = float(np.median(y_values)) if y_values else float(min(line["y0"], line["y1"]))

    return {
        "line": line,
        "pred_segments": pred_segments,
        "ref_segments": ref_segments,
        "x_to_y": {int(k): int(v) for k, v in x_to_y.items()},
        "x_to_score": {int(k): float(v) for k, v in x_to_score.items()},
        "total_score": total_score,
        "mean_score": mean_score,
        "anchor_y": anchor_y,
        "source_raw_line_ids": sorted(int(v) for v in (source_raw_line_ids or [])),
    }


def _build_line_coverage(line: dict, matrix: np.ndarray) -> dict | None:
    if matrix.size == 0:
        return None

    n_ref, n_pred = matrix.shape
    bounds = _line_x_bounds(line, n_pred)
    if bounds is None:
        return None

    x_to_y: dict[int, int] = {}
    x_to_score: dict[int, float] = {}
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


def _set_iou(values_a: set[int], values_b: set[int]) -> float:
    union = values_a | values_b
    if not union:
        return 0.0
    return float(len(values_a & values_b) / len(union))


def _coverages_overlap(cov_a: dict, cov_b: dict, *, min_iou_threshold: float) -> bool:
    x_iou = _set_iou(cov_a["pred_segments"], cov_b["pred_segments"])
    y_iou = _set_iou(cov_a["ref_segments"], cov_b["ref_segments"])
    return bool(min(x_iou, y_iou) > float(min_iou_threshold))


def _local_path_key(cov: dict, x: int):
    y = int(cov["x_to_y"][x])
    return (
        float(cov["x_to_score"][x]),
        float(cov.get("total_score", 0.0)),
        float(cov.get("mean_score", 0.0)),
        -float(y),
    )


def _merge_component(component_coverages: list[dict], matrix: np.ndarray) -> dict:
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

    for x in all_x:
        candidates = [cov for cov in component_coverages if x in cov["x_to_y"]]
        best_cov = max(candidates, key=lambda cov: _local_path_key(cov, x))
        merged_x_to_y[int(x)] = int(best_cov["x_to_y"][x])
        merged_x_to_score[int(x)] = float(best_cov["x_to_score"][x])

    source_raw_line_ids = sorted(
        {
            int(raw_id)
            for cov in component_coverages
            for raw_id in cov.get("source_raw_line_ids", [])
            if int(raw_id) >= 0
        }
    )

    return _coverage_from_path(
        x_to_y=merged_x_to_y,
        x_to_score=merged_x_to_score,
        matrix=matrix,
        fallback_line=fallback_cov["line"],
        source_raw_line_ids=source_raw_line_ids,
    )


def _components_from_adjacency(adjacency: dict[int, set[int]]) -> list[list[int]]:
    if not adjacency:
        return []

    components: list[list[int]] = []
    seen: set[int] = set()
    for start in sorted(adjacency):
        if start in seen:
            continue

        stack = [int(start)]
        seen.add(int(start))
        component: list[int] = []

        while stack:
            idx = stack.pop()
            component.append(int(idx))
            for other in adjacency[idx]:
                if other in seen:
                    continue
                seen.add(int(other))
                stack.append(int(other))

        components.append(sorted(component))

    return components


def _coverage_components_fast(coverages: list[dict], *, min_iou_threshold: float) -> list[list[int]]:
    if not coverages:
        return []

    adjacency: dict[int, set[int]] = {idx: set() for idx in range(len(coverages))}
    for idx in range(len(coverages)):
        for other in range(idx + 1, len(coverages)):
            if not _coverages_overlap(coverages[idx], coverages[other], min_iou_threshold=min_iou_threshold):
                continue
            adjacency[idx].add(int(other))
            adjacency[other].add(int(idx))

    return _components_from_adjacency(adjacency)


def _prepare_candidates(lines: list[dict], matrix: np.ndarray, *, abs_min_len: float) -> list[dict]:
    if not lines:
        return []

    max_score = max(float(ln.get("score", 0.0)) for ln in lines)
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0
    candidates: list[dict] = []

    for raw_line_id, ln in enumerate(lines):
        ln2 = dict(ln)
        ln2["raw_line_id"] = int(raw_line_id)
        ln2["length"] = line_length(ln2)
        ln2["support"] = mean_line_support(matrix, ln2)

        if ln2["length"] < float(abs_min_len):
            continue
        if max_score > 0 and float(ln2.get("score", 0.0)) < 0.06 * max_score:
            continue
        if ln2["support"] < support_floor:
            continue
        candidates.append(ln2)

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


def _compute_final_assignment(coverages: list[dict], matrix: np.ndarray) -> dict[str, np.ndarray]:
    n_ref, n_pred = matrix.shape
    assignment = _empty_assignment(n_pred)
    mapped_y = assignment["mapped_y"]
    mapped_line_id = assignment["mapped_line_id"]

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

    final_lines = sorted(
        final_lines,
        key=lambda ln: (
            float(ln.get("anchor_y", min(ln["y0"], ln["y1"]))),
            min(ln["x0"], ln["x1"]),
        ),
    )
    return final_lines, assignment


def filter_lines_for_alignment_by_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    **_ignored,
):
    """Fast ownership filter variant: same final outputs, stripped debug bookkeeping."""
    if not lines:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    if matrix.size == 0:
        n_pred = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_pred)

    if mask_bool.shape != matrix.shape:
        raise ValueError(f"mask_bool shape {mask_bool.shape} does not match matrix shape {matrix.shape}")

    candidates = _prepare_candidates(lines, matrix, abs_min_len=abs_min_len)
    coverages = [cov for cov in (_build_line_coverage(ln, matrix) for ln in candidates) if cov is not None]
    components = _coverage_components_fast(coverages, min_iou_threshold=min_iou_threshold)
    merged_coverages = [_merge_component([coverages[idx] for idx in component], matrix) for component in components]

    return _finalize_outputs(list(merged_coverages), matrix, mask_bool)
