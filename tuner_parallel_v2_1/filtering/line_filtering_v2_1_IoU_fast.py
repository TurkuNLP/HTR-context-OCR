from __future__ import annotations

"""Production-style true-IoU line filtering for tuner_parallel_v2.

The tuner keeps its own filtering module because it needs a stable local
implementation that can evolve independently of the report pipeline while still
remaining easy to read months later.

Important design points:
- raw Hough segments are converted into line dictionaries before filtering
- filtering is now the only place that merges overlapping guides
- the overlap graph uses the optimized v2.12 production strategy
- final ownership uses a per-column coverage index instead of scanning every
  coverage for every prediction column
"""

import math

import numpy as np

try:
    from ..cython_accel.optional_filtering import (
        build_coverage_indices_by_prediction_column as _accelerated_build_coverage_indices,
        set_iou as _accelerated_set_iou,
    )
except ImportError:
    try:
        from cython_accel.optional_filtering import (  # type: ignore
            build_coverage_indices_by_prediction_column as _accelerated_build_coverage_indices,
            set_iou as _accelerated_set_iou,
        )
    except ImportError:
        _accelerated_build_coverage_indices = None  # type: ignore
        _accelerated_set_iou = None  # type: ignore

try:
    from ..runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

# Ensure shared helper modules from the project root are importable whether this
# file is used as a package module or executed through a direct script path.
ensure_tuner_runtime_paths()

# Reuse the shared geometry helpers instead of duplicating interpolation and
# support sampling logic inside the tuner.
from line_filtering import line_length, line_y_at_x, mean_line_support

# Keep the current tuner defaults unchanged.
DEFAULT_ABS_MIN_LEN = 6.0
DEFAULT_MIN_IOU_THRESHOLD = 0.035

__all__ = [
    "DEFAULT_ABS_MIN_LEN",
    "DEFAULT_MIN_IOU_THRESHOLD",
    "filter_lines_for_alignment_by_ownership",
]


# Build the empty per-column mapping used when no final guides survive.
def _empty_assignment(n_pred: int) -> dict[str, np.ndarray]:
    """Return the stable empty assignment structure expected downstream."""
    return {
        "mapped_y": np.full(int(n_pred), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_pred), -1, dtype=int),
    }


# Clamp a line's x-span to valid matrix columns.
def _line_x_bounds(line: dict, n_pred: int) -> tuple[int, int] | None:
    """Return the inclusive integer prediction-column bounds for one line."""
    if int(n_pred) <= 0:
        return None

    x_min = max(0, int(math.floor(min(line["x0"], line["x1"]))))
    x_max = min(int(n_pred) - 1, int(math.ceil(max(line["x0"], line["x1"]))))
    if x_max < x_min:
        return None
    return x_min, x_max


# Expand a per-column y-path into the set of covered reference rows.
def _ref_segments_from_path(x_to_y: dict[int, int]) -> set[int]:
    """Expand a sampled x->y path into covered reference-row ids."""
    if not x_to_y:
        return set()

    covered_reference_rows: set[int] = set()
    previous_row: int | None = None

    # Add the sampled row at every x and bridge vertical gaps between
    # neighbouring columns so IoU uses the full covered band.
    for prediction_column in sorted(x_to_y):
        current_row = int(x_to_y[prediction_column])
        covered_reference_rows.add(current_row)
        if previous_row is not None:
            row_start, row_end = sorted((previous_row, current_row))
            for reference_row in range(row_start, row_end + 1):
                covered_reference_rows.add(int(reference_row))
        previous_row = current_row

    return covered_reference_rows


# Fit one representative straight line through a selected x->y path.
def _fit_line_from_path(
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    *,
    fallback_line: dict | None = None,
) -> dict:
    """Fit the representative straight segment used by downstream reports."""
    if not x_to_y:
        fallback_geometry = {} if fallback_line is None else dict(fallback_line)
        fallback_geometry.setdefault("x0", 0.0)
        fallback_geometry.setdefault("y0", 0.0)
        fallback_geometry.setdefault("x1", 0.0)
        fallback_geometry.setdefault("y1", 0.0)
        fallback_geometry["length"] = line_length(fallback_geometry)
        fallback_geometry["support"] = mean_line_support(matrix, fallback_geometry) if matrix.size else 0.0
        fallback_geometry["score"] = float(
            fallback_geometry.get("score", fallback_geometry.get("support", 0.0))
        )
        return fallback_geometry

    sampled_prediction_columns = np.asarray(sorted(x_to_y), dtype=float)
    sampled_reference_rows = np.asarray(
        [float(x_to_y[int(prediction_column)]) for prediction_column in sampled_prediction_columns],
        dtype=float,
    )
    sampled_weights = np.asarray(
        [max(float(x_to_score[int(prediction_column)]), 1e-6) for prediction_column in sampled_prediction_columns],
        dtype=float,
    )

    # Preserve the current weighted straight-line fit exactly.
    if len(sampled_prediction_columns) == 1 or np.allclose(
        sampled_prediction_columns,
        sampled_prediction_columns[0],
    ):
        fitted_x0 = fitted_x1 = float(sampled_prediction_columns[0])
        fitted_y0 = fitted_y1 = float(sampled_reference_rows[0])
    else:
        fitted_slope, fitted_intercept = np.polyfit(
            sampled_prediction_columns,
            sampled_reference_rows,
            deg=1,
            w=sampled_weights,
        )
        fitted_x0 = float(sampled_prediction_columns.min())
        fitted_x1 = float(sampled_prediction_columns.max())
        fitted_y0 = float((fitted_slope * fitted_x0) + fitted_intercept)
        fitted_y1 = float((fitted_slope * fitted_x1) + fitted_intercept)

    representative_line = {} if fallback_line is None else dict(fallback_line)
    representative_line["x0"] = fitted_x0
    representative_line["y0"] = fitted_y0
    representative_line["x1"] = fitted_x1
    representative_line["y1"] = fitted_y1
    representative_line["length"] = line_length(representative_line)
    representative_line["support"] = mean_line_support(matrix, representative_line) if matrix.size else 0.0
    representative_line["score"] = float(representative_line["support"])
    return representative_line


# Build one coverage object from an x->y path and its local matrix scores.
def _coverage_from_path(
    *,
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    fallback_line: dict | None = None,
    source_raw_line_ids: list[int] | None = None,
) -> dict:
    """Normalize one sampled path into the canonical coverage structure."""
    prediction_segments = set(int(prediction_column) for prediction_column in x_to_y)
    reference_segments = _ref_segments_from_path(x_to_y)
    representative_line = _fit_line_from_path(x_to_y, x_to_score, matrix, fallback_line=fallback_line)

    sampled_reference_rows = [int(x_to_y[prediction_column]) for prediction_column in sorted(x_to_y)]
    total_local_score = float(sum(float(score_value) for score_value in x_to_score.values()))
    mean_local_score = float(total_local_score / len(x_to_score)) if x_to_score else 0.0
    anchor_reference_row = (
        float(np.median(sampled_reference_rows))
        if sampled_reference_rows
        else float(min(representative_line["y0"], representative_line["y1"]))
    )

    prediction_min = min(prediction_segments) if prediction_segments else 0
    prediction_max = max(prediction_segments) if prediction_segments else -1
    reference_min = min(reference_segments) if reference_segments else 0
    reference_max = max(reference_segments) if reference_segments else -1

    return {
        "line": representative_line,
        "pred_segments": prediction_segments,
        "ref_segments": reference_segments,
        "pred_min": int(prediction_min),
        "pred_max": int(prediction_max),
        "ref_min": int(reference_min),
        "ref_max": int(reference_max),
        "x_to_y": {
            int(prediction_column): int(reference_row)
            for prediction_column, reference_row in x_to_y.items()
        },
        "x_to_score": {
            int(prediction_column): float(local_score)
            for prediction_column, local_score in x_to_score.items()
        },
        "total_score": total_local_score,
        "mean_score": mean_local_score,
        "anchor_y": anchor_reference_row,
        "source_raw_line_ids": sorted(int(raw_line_id) for raw_line_id in (source_raw_line_ids or [])),
    }


# Convert one raw detected line into a coverage object over the matrix grid.
def _build_line_coverage(line: dict, matrix: np.ndarray) -> dict | None:
    """Project one line onto the score-matrix grid using the current sampling rule."""
    if matrix.size == 0:
        return None

    n_reference_rows, n_prediction_columns = matrix.shape
    x_bounds = _line_x_bounds(line, n_prediction_columns)
    if x_bounds is None:
        return None

    x_to_y: dict[int, int] = {}
    x_to_score: dict[int, float] = {}

    # Sample the line once for every covered prediction column.
    for prediction_column in range(x_bounds[0], x_bounds[1] + 1):
        sampled_reference_row = int(np.clip(round(line_y_at_x(line, prediction_column)), 0, n_reference_rows - 1))
        x_to_y[int(prediction_column)] = int(sampled_reference_row)
        x_to_score[int(prediction_column)] = float(matrix[sampled_reference_row, prediction_column])

    if not x_to_y:
        return None

    raw_line_id = int(line.get("raw_line_id", -1)) if "raw_line_id" in line else -1
    return _coverage_from_path(
        x_to_y=x_to_y,
        x_to_score=x_to_score,
        matrix=matrix,
        fallback_line=line,
        source_raw_line_ids=[raw_line_id] if raw_line_id >= 0 else [],
    )


# Compute the exact set IoU used by the true-IoU overlap rule.
def _set_iou(values_a: set[int], values_b: set[int]) -> float:
    """Return exact set IoU while preserving the current empty-union behavior."""
    if _accelerated_set_iou is not None:
        return float(_accelerated_set_iou(values_a, values_b))

    union_values = values_a | values_b
    if not union_values:
        return 0.0
    return float(len(values_a & values_b) / len(union_values))


# Summarize whether two integer ranges are disjoint or overlapping.
def _segments_bounds_disjoint(min_a: int, max_a: int, min_b: int, max_b: int) -> bool:
    """Return whether two inclusive integer ranges are disjoint."""
    return int(max_a) < int(min_b) or int(max_b) < int(min_a)


# Summarize whether two integer ranges overlap at all.
def _segments_bounds_overlap(min_a: int, max_a: int, min_b: int, max_b: int) -> bool:
    """Return whether two inclusive integer ranges overlap at all."""
    return not _segments_bounds_disjoint(min_a, max_a, min_b, max_b)


# Fast production-only overlap decision that avoids building debug payloads.
def _coverages_merge_candidate(cov_a: dict, cov_b: dict, *, min_iou_threshold: float) -> bool:
    """Return the exact merge decision for production filtering only."""
    if _segments_bounds_disjoint(
        int(cov_a.get("pred_min", 0)),
        int(cov_a.get("pred_max", -1)),
        int(cov_b.get("pred_min", 0)),
        int(cov_b.get("pred_max", -1)),
    ):
        return False

    if _segments_bounds_disjoint(
        int(cov_a.get("ref_min", 0)),
        int(cov_a.get("ref_max", -1)),
        int(cov_b.get("ref_min", 0)),
        int(cov_b.get("ref_max", -1)),
    ):
        return False

    prediction_iou = _set_iou(cov_a["pred_segments"], cov_b["pred_segments"])
    reference_iou = _set_iou(cov_a["ref_segments"], cov_b["ref_segments"])
    return bool(min(prediction_iou, reference_iou) > float(min_iou_threshold))


# Choose which coverage contributes a given prediction column.
def _local_path_key(cov: dict, prediction_column: int):
    """Return the exact local winner key used across merge and assignment stages."""
    sampled_reference_row = int(cov["x_to_y"][prediction_column])
    return (
        float(cov["x_to_score"][prediction_column]),
        float(cov.get("total_score", 0.0)),
        float(cov.get("mean_score", 0.0)),
        -float(sampled_reference_row),
    )


# Merge one connected overlap component into one coverage object.
def _merge_component(component_coverages: list[dict], matrix: np.ndarray) -> dict:
    """Merge one overlap component while preserving per-column local winners."""
    if len(component_coverages) == 1:
        return dict(component_coverages[0])

    fallback_coverage = max(
        component_coverages,
        key=lambda coverage: (
            float(coverage.get("total_score", 0.0)),
            float(coverage.get("mean_score", 0.0)),
            int(len(coverage.get("pred_segments", ()))),
            float(coverage["line"].get("support", 0.0)),
            float(coverage["line"].get("length", 0.0)),
        ),
    )

    best_path_sample_by_column: dict[int, tuple[tuple, int, float]] = {}

    # Keep the first coverage that attains the best local key at each x, just
    # like the production filter does.
    for coverage in component_coverages:
        for prediction_column, sampled_reference_row in coverage["x_to_y"].items():
            local_winner_key = _local_path_key(coverage, int(prediction_column))
            current_best_sample = best_path_sample_by_column.get(int(prediction_column))
            if current_best_sample is None or local_winner_key > current_best_sample[0]:
                best_path_sample_by_column[int(prediction_column)] = (
                    local_winner_key,
                    int(sampled_reference_row),
                    float(coverage["x_to_score"][int(prediction_column)]),
                )

    merged_x_to_y = {
        int(prediction_column): int(best_path_sample_by_column[int(prediction_column)][1])
        for prediction_column in sorted(best_path_sample_by_column)
    }
    merged_x_to_score = {
        int(prediction_column): float(best_path_sample_by_column[int(prediction_column)][2])
        for prediction_column in sorted(best_path_sample_by_column)
    }

    merged_source_raw_line_ids = sorted(
        {
            int(raw_line_id)
            for coverage in component_coverages
            for raw_line_id in coverage.get("source_raw_line_ids", [])
            if int(raw_line_id) >= 0
        }
    )

    return _coverage_from_path(
        x_to_y=merged_x_to_y,
        x_to_score=merged_x_to_score,
        matrix=matrix,
        fallback_line=fallback_coverage["line"],
        source_raw_line_ids=merged_source_raw_line_ids,
    )


# Build connected components from an explicit adjacency graph.
def _components_from_adjacency(adjacency: dict[int, set[int]]) -> list[list[int]]:
    """Return connected components in deterministic index order."""
    if not adjacency:
        return []

    connected_components: list[list[int]] = []
    visited_indices: set[int] = set()

    for start_index in sorted(adjacency):
        if start_index in visited_indices:
            continue

        stack = [int(start_index)]
        visited_indices.add(int(start_index))
        component_indices: list[int] = []

        while stack:
            current_index = stack.pop()
            component_indices.append(int(current_index))
            for adjacent_index in sorted(adjacency[current_index]):
                if adjacent_index in visited_indices:
                    continue
                visited_indices.add(int(adjacent_index))
                stack.append(int(adjacent_index))

        connected_components.append(sorted(component_indices))

    return connected_components


# Generate only coverage pairs that can possibly overlap on both axes.
def _iter_possible_overlap_pairs_for_production(coverages: list[dict]):
    """Yield only pairs whose prediction and reference bounds overlap."""
    sorted_coverage_indices = sorted(
        range(len(coverages)),
        key=lambda coverage_index: (
            int(coverages[coverage_index].get("pred_min", 0)),
            int(coverages[coverage_index].get("pred_max", -1)),
            int(coverages[coverage_index].get("ref_min", 0)),
            int(coverages[coverage_index].get("ref_max", -1)),
            int(coverage_index),
        ),
    )

    active_prediction_overlap_indices: list[int] = []

    for current_index in sorted_coverage_indices:
        current_coverage = coverages[int(current_index)]
        current_prediction_min = int(current_coverage.get("pred_min", 0))
        current_reference_min = int(current_coverage.get("ref_min", 0))
        current_reference_max = int(current_coverage.get("ref_max", -1))

        still_active_indices: list[int] = []
        for active_index in active_prediction_overlap_indices:
            active_coverage = coverages[int(active_index)]
            active_prediction_max = int(active_coverage.get("pred_max", -1))

            # Once prediction bounds are disjoint, the pair can never merge under
            # the exact IoU rule.
            if active_prediction_max < current_prediction_min:
                continue

            still_active_indices.append(int(active_index))

            # The production path keeps only pairs that overlap on both axes
            # before doing exact IoU work.
            if not _segments_bounds_overlap(
                int(active_coverage.get("ref_min", 0)),
                int(active_coverage.get("ref_max", -1)),
                current_reference_min,
                current_reference_max,
            ):
                continue

            left_index = min(int(active_index), int(current_index))
            right_index = max(int(active_index), int(current_index))
            yield left_index, right_index

        still_active_indices.append(int(current_index))
        active_prediction_overlap_indices = still_active_indices


# Production-only overlap graph that avoids the old O(n^2) pair scan.
def _coverage_components_for_production(
    coverages: list[dict],
    *,
    min_iou_threshold: float,
) -> list[list[int]]:
    """Build connected components for the production filtering path."""
    if not coverages:
        return []

    adjacency: dict[int, set[int]] = {coverage_index: set() for coverage_index in range(len(coverages))}

    for left_index, right_index in _iter_possible_overlap_pairs_for_production(coverages):
        if not _coverages_merge_candidate(
            coverages[int(left_index)],
            coverages[int(right_index)],
            min_iou_threshold=min_iou_threshold,
        ):
            continue
        adjacency[int(left_index)].add(int(right_index))
        adjacency[int(right_index)].add(int(left_index))

    return _components_from_adjacency(adjacency)


# Reuse already-available line fields when they are valid, otherwise recompute them.
def _normalize_candidate_line(line: dict, matrix: np.ndarray, *, raw_line_id: int) -> dict:
    """Return one candidate line with normalized filter fields populated."""
    normalized_line = dict(line)
    normalized_line["raw_line_id"] = int(raw_line_id)

    existing_length = normalized_line.get("length")
    if isinstance(existing_length, (int, float)) and np.isfinite(float(existing_length)):
        normalized_line["length"] = float(existing_length)
    else:
        normalized_line["length"] = float(line_length(normalized_line))

    existing_support = normalized_line.get("support")
    if isinstance(existing_support, (int, float)) and np.isfinite(float(existing_support)):
        normalized_line["support"] = float(existing_support)
    else:
        normalized_line["support"] = float(mean_line_support(matrix, normalized_line))

    return normalized_line


# Normalize the raw Hough lines into a credible candidate set.
def _prepare_candidates(lines: list[dict], matrix: np.ndarray, *, abs_min_len: float) -> list[dict]:
    """Apply the unchanged coarse gates before the true-IoU stage."""
    if not lines:
        return []

    max_score = max(float(line.get("score", 0.0)) for line in lines)
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0
    prepared_candidates: list[dict] = []

    for raw_line_id, line in enumerate(lines):
        prepared_line = _normalize_candidate_line(line, matrix, raw_line_id=int(raw_line_id))

        if prepared_line["length"] < float(abs_min_len):
            continue
        if max_score > 0 and float(prepared_line.get("score", 0.0)) < 0.06 * max_score:
            continue
        if prepared_line["support"] < support_floor:
            continue
        prepared_candidates.append(prepared_line)

    # Preserve the exact fallback behavior when every candidate fails the coarse gates.
    if not prepared_candidates:
        best_raw_line_id, best_line = max(
            enumerate(lines),
            key=lambda indexed_line: float(indexed_line[1].get("score", 0.0)),
        )
        prepared_candidates = [
            _normalize_candidate_line(best_line, matrix, raw_line_id=int(best_raw_line_id))
        ]

    return sorted(
        prepared_candidates,
        key=lambda line: (min(line["y0"], line["y1"]), min(line["x0"], line["x1"])),
    )


# Build sampled coverage objects for all prepared candidates.
def _build_candidate_coverages(candidate_lines: list[dict], matrix: np.ndarray) -> list[dict]:
    """Convert all prepared candidate lines into coverage objects."""
    coverage_objects: list[dict] = []
    for candidate_line in candidate_lines:
        coverage_object = _build_line_coverage(candidate_line, matrix)
        if coverage_object is None:
            continue
        coverage_objects.append(coverage_object)
    return coverage_objects


# Lean production filtering path used by the tuner hot loop.
def _run_production_filtering_pipeline(
    lines: list[dict],
    matrix: np.ndarray,
    *,
    abs_min_len: float,
    min_iou_threshold: float,
) -> list[dict]:
    """Run the exact filtering semantics without the old quadratic pair scan."""
    candidate_lines = _prepare_candidates(lines, matrix, abs_min_len=abs_min_len)
    candidate_coverages = _build_candidate_coverages(candidate_lines, matrix)
    component_indices = _coverage_components_for_production(
        candidate_coverages,
        min_iou_threshold=min_iou_threshold,
    )
    return [
        _merge_component([candidate_coverages[coverage_index] for coverage_index in component], matrix)
        for component in component_indices
    ]


# Build a per-column index of which surviving coverages can own that column.
def _build_coverage_indices_by_prediction_column(
    coverages: list[dict],
    n_prediction_columns: int,
) -> list[list[int]]:
    """Index surviving coverages by prediction column for faster ownership scans."""
    if _accelerated_build_coverage_indices is not None:
        return _accelerated_build_coverage_indices(coverages, int(n_prediction_columns))

    coverage_indices_by_prediction_column: list[list[int]] = [
        [] for _ in range(int(n_prediction_columns))
    ]

    for coverage_index, coverage in enumerate(coverages):
        for prediction_column in coverage["x_to_y"]:
            if 0 <= int(prediction_column) < int(n_prediction_columns):
                coverage_indices_by_prediction_column[int(prediction_column)].append(int(coverage_index))

    return coverage_indices_by_prediction_column


# Assign each prediction column to the strongest surviving coverage.
def _compute_final_assignment(coverages: list[dict], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Compute final ownership arrays while preserving current tie-break behavior."""
    n_reference_rows, n_prediction_columns = matrix.shape
    assignment = _empty_assignment(n_prediction_columns)
    mapped_y = assignment["mapped_y"]
    mapped_line_id = assignment["mapped_line_id"]

    coverage_indices_by_prediction_column = _build_coverage_indices_by_prediction_column(
        coverages,
        n_prediction_columns,
    )

    for prediction_column in range(n_prediction_columns):
        best_owner: tuple[int, int] | None = None
        best_owner_key = None

        for coverage_index in coverage_indices_by_prediction_column[prediction_column]:
            coverage = coverages[int(coverage_index)]
            sampled_reference_row = int(coverage["x_to_y"][prediction_column])
            owner_key = _local_path_key(coverage, prediction_column) + (
                float(coverage["line"].get("length", 0.0)),
                -float(coverage_index),
            )
            if best_owner_key is None or owner_key > best_owner_key:
                best_owner_key = owner_key
                best_owner = (int(coverage_index), int(sampled_reference_row))

        if best_owner is None:
            continue

        winning_coverage_index, winning_reference_row = best_owner
        mapped_line_id[prediction_column] = int(winning_coverage_index)
        mapped_y[prediction_column] = float(np.clip(winning_reference_row, 0, n_reference_rows - 1))

    return assignment


# Convert the merged coverages into final output lines and per-column assignment.
def _finalize_outputs(
    coverages: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Finalize surviving lines and ownership arrays in stable reading order."""
    n_prediction_columns = matrix.shape[1] if matrix.ndim == 2 else 0
    current_coverages = sorted(
        coverages,
        key=lambda coverage: (
            float(coverage.get("anchor_y", 0.0)),
            min(coverage["line"]["x0"], coverage["line"]["x1"]),
        ),
    )
    if not current_coverages:
        return [], _empty_assignment(n_prediction_columns)

    # Preserve the iterative prune-until-stable ownership behavior.
    while True:
        assignment = _compute_final_assignment(current_coverages, matrix)
        mapped_line_id = np.asarray(assignment["mapped_line_id"], dtype=int)
        kept_coverage_indices = [
            coverage_index
            for coverage_index in range(len(current_coverages))
            if np.any(mapped_line_id == coverage_index)
        ]
        if len(kept_coverage_indices) == len(current_coverages):
            break
        current_coverages = [current_coverages[coverage_index] for coverage_index in kept_coverage_indices]
        current_coverages = sorted(
            current_coverages,
            key=lambda coverage: (
                float(coverage.get("anchor_y", 0.0)),
                min(coverage["line"]["x0"], coverage["line"]["x1"]),
            ),
        )
        if not current_coverages:
            return [], _empty_assignment(n_prediction_columns)

    mapped_y = np.asarray(assignment["mapped_y"], dtype=float)
    mapped_line_id = np.asarray(assignment["mapped_line_id"], dtype=int)
    final_lines: list[dict] = []

    for line_id, coverage in enumerate(current_coverages):
        final_line = dict(coverage["line"])
        owned_prediction_columns = [
            int(prediction_column)
            for prediction_column in np.flatnonzero(mapped_line_id == line_id)
        ]
        owned_local_scores = [
            float(coverage["x_to_score"][prediction_column])
            for prediction_column in owned_prediction_columns
            if prediction_column in coverage["x_to_score"]
        ]
        owned_reference_rows = [
            int(np.clip(round(float(mapped_y[prediction_column])), 0, mask_bool.shape[0] - 1))
            for prediction_column in owned_prediction_columns
            if mask_bool.shape[0] > 0
        ]

        prediction_span_size = max(1, len(coverage["pred_segments"]))
        owned_mask_hits = 0
        if mask_bool.shape[0] > 0 and mask_bool.shape[1] > 0:
            for prediction_column, reference_row in zip(owned_prediction_columns, owned_reference_rows):
                if 0 <= prediction_column < mask_bool.shape[1] and 0 <= reference_row < mask_bool.shape[0]:
                    owned_mask_hits += int(bool(mask_bool[reference_row, prediction_column]))

        final_line["source_raw_line_ids"] = sorted(
            int(raw_line_id) for raw_line_id in coverage.get("source_raw_line_ids", [])
        )
        final_line["owned_cols"] = int(len(owned_prediction_columns))
        final_line["owned_fraction"] = float(len(owned_prediction_columns) / prediction_span_size)
        final_line["owned_score_mean"] = float(np.mean(owned_local_scores)) if owned_local_scores else 0.0
        final_line["owned_score_sum"] = float(np.sum(owned_local_scores)) if owned_local_scores else 0.0
        final_line["owned_mask_hits"] = int(owned_mask_hits)
        final_line["owned_mask_fraction"] = (
            float(owned_mask_hits / len(owned_prediction_columns))
            if owned_prediction_columns
            else 0.0
        )
        final_line["anchor_y"] = (
            float(np.median(owned_reference_rows))
            if owned_reference_rows
            else float(coverage.get("anchor_y", min(final_line["y0"], final_line["y1"])))
        )
        final_lines.append(final_line)

    final_lines = sorted(
        final_lines,
        key=lambda line: (
            float(line.get("anchor_y", min(line["y0"], line["y1"]))),
            min(line["x0"], line["x1"]),
        ),
    )
    return final_lines, assignment


# Filter lines using true IoU over prediction/reference coverage.
def filter_lines_for_alignment_by_ownership(
    lines: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    abs_min_len: float = DEFAULT_ABS_MIN_LEN,
    min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD,
    **_ignored,
):
    """Run the production-style true-IoU filter and return final lines plus ownership arrays."""
    if not lines:
        n_prediction_columns = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_prediction_columns)

    if matrix.size == 0:
        n_prediction_columns = matrix.shape[1] if matrix.ndim == 2 else 0
        return [], _empty_assignment(n_prediction_columns)

    if mask_bool.shape != matrix.shape:
        raise ValueError(f"mask_bool shape {mask_bool.shape} does not match matrix shape {matrix.shape}")

    merged_coverages = _run_production_filtering_pipeline(
        lines,
        matrix,
        abs_min_len=float(abs_min_len),
        min_iou_threshold=float(min_iou_threshold),
    )
    return _finalize_outputs(list(merged_coverages), matrix, mask_bool)
