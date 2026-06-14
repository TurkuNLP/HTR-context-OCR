from __future__ import annotations

"""Final ownership assignment and final-line export for line filtering."""

import time

import numpy as np

from . import filter_cython_accelerators as accelerators
from .filter_candidate_coverages import local_path_key
from .filter_profile_fields import add_profile_seconds, set_profile_count


def empty_assignment(n_pred: int) -> dict[str, np.ndarray]:
    """Return the stable no-owner assignment arrays expected downstream."""
    return {
        "mapped_y": np.full(int(n_pred), np.nan, dtype=float),
        "mapped_line_id": np.full(int(n_pred), -1, dtype=int),
    }


def build_coverage_indices_by_prediction_column(
    coverages: list[dict],
    n_prediction_columns: int,
) -> list[list[int]]:
    """Index surviving coverages by prediction column for fast ownership scans."""
    if accelerators.accelerated_build_coverage_indices is not None:
        return accelerators.accelerated_build_coverage_indices(coverages, int(n_prediction_columns))

    coverage_indices_by_prediction_column: list[list[int]] = [
        [] for _ in range(int(n_prediction_columns))
    ]

    for coverage_index, coverage in enumerate(coverages):
        for prediction_column in coverage["x_to_y"]:
            if 0 <= int(prediction_column) < int(n_prediction_columns):
                coverage_indices_by_prediction_column[int(prediction_column)].append(int(coverage_index))

    return coverage_indices_by_prediction_column


def compute_final_assignment(coverages: list[dict], matrix: np.ndarray) -> dict[str, np.ndarray]:
    """Assign each prediction column to the strongest surviving coverage.

    The filter intentionally keeps one owner per prediction column.  The winning
    owner gives downstream scoring one mapped reference row and one final line id
    for that prediction column.
    """
    n_reference_rows, n_prediction_columns = matrix.shape
    assignment = empty_assignment(n_prediction_columns)
    mapped_y = assignment["mapped_y"]
    mapped_line_id = assignment["mapped_line_id"]

    coverage_indices_by_prediction_column = build_coverage_indices_by_prediction_column(
        coverages,
        n_prediction_columns,
    )

    if accelerators.accelerated_compute_final_assignment is not None:
        accelerated_assignment = accelerators.accelerated_compute_final_assignment(
            coverages=coverages,
            coverage_indices_by_prediction_column=coverage_indices_by_prediction_column,
            n_reference_rows=int(n_reference_rows),
            n_prediction_columns=int(n_prediction_columns),
        )
        if accelerated_assignment is not None:
            return {
                "mapped_y": np.asarray(accelerated_assignment["mapped_y"], dtype=float),
                "mapped_line_id": np.asarray(accelerated_assignment["mapped_line_id"], dtype=int),
            }

    for prediction_column in range(n_prediction_columns):
        best_owner: tuple[int, int] | None = None
        best_owner_key = None

        for coverage_index in coverage_indices_by_prediction_column[prediction_column]:
            coverage = coverages[int(coverage_index)]
            sampled_reference_row = int(coverage["x_to_y"][prediction_column])
            owner_key = local_path_key(coverage, prediction_column) + (
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


def used_coverage_indices_from_assignment(
    mapped_line_id: np.ndarray,
    *,
    coverage_count: int,
) -> list[int]:
    """Return coverage ids that own at least one prediction column.

    The return order is ascending coverage id.  That matches the old
    ``for coverage_index in range(...): np.any(...)`` rule exactly and keeps
    final line ids deterministic.
    """
    resolved_coverage_count = int(coverage_count)
    if resolved_coverage_count <= 0:
        return []

    mapped_line_id_array = np.asarray(mapped_line_id, dtype=int)
    if mapped_line_id_array.size == 0:
        return []

    valid_owner_mask = (
        (mapped_line_id_array >= 0)
        & (mapped_line_id_array < resolved_coverage_count)
    )
    if not np.any(valid_owner_mask):
        return []

    used_counts_by_coverage_id = np.bincount(
        mapped_line_id_array[valid_owner_mask].astype(np.intp, copy=False),
        minlength=resolved_coverage_count,
    )
    return [
        int(coverage_index)
        for coverage_index in np.flatnonzero(
            used_counts_by_coverage_id[:resolved_coverage_count] > 0
        )
    ]


def final_line_sort_key(line: dict) -> tuple[float, float]:
    """Return the visible reading-order key used for final line output."""
    return (
        float(line.get("anchor_y", min(line["y0"], line["y1"]))),
        min(float(line["x0"]), float(line["x1"])),
    )


def sort_final_lines_and_remap_assignment(
    final_lines: list[dict],
    assignment: dict[str, np.ndarray],
) -> tuple[list[dict], dict[str, np.ndarray]]:
    """Sort final lines and keep column-owner ids aligned with that sorted order.

    ``compute_final_assignment`` stores line ids as positions in the coverage list
    used during ownership assignment.  Once final lines are sorted for plotting
    and reading order, those ids must be rewritten so downstream text scoring
    still compares each line with the columns it actually owns.
    """
    indexed_lines = [(int(line_id), dict(line)) for line_id, line in enumerate(final_lines)]
    sorted_indexed_lines = sorted(indexed_lines, key=lambda item: final_line_sort_key(item[1]))
    old_line_id_to_new_line_id = {
        int(old_line_id): int(new_line_id)
        for new_line_id, (old_line_id, _line) in enumerate(sorted_indexed_lines)
    }

    original_mapped_line_id = np.asarray(assignment.get("mapped_line_id", []), dtype=int)
    remapped_line_id = np.full(original_mapped_line_id.shape, -1, dtype=int)
    for old_line_id, new_line_id in old_line_id_to_new_line_id.items():
        remapped_line_id[original_mapped_line_id == int(old_line_id)] = int(new_line_id)

    remapped_assignment = dict(assignment)
    remapped_assignment["mapped_line_id"] = remapped_line_id
    return [line for _old_line_id, line in sorted_indexed_lines], remapped_assignment


def finalize_outputs(
    coverages: list[dict],
    matrix: np.ndarray,
    mask_bool: np.ndarray,
    *,
    profile: dict | None = None,
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
        set_profile_count(profile, "filter_final_line_count", 0)
        return [], empty_assignment(n_prediction_columns)

    prune_iteration_count = 0
    while True:
        prune_iteration_count += 1
        final_assignment_started_at = time.perf_counter() if profile is not None else 0.0
        assignment = compute_final_assignment(current_coverages, matrix)
        if profile is not None:
            add_profile_seconds(
                profile,
                "filter_final_assignment_seconds",
                float(time.perf_counter() - final_assignment_started_at),
            )
        mapped_line_id = np.asarray(assignment["mapped_line_id"], dtype=int)
        kept_coverage_indices = used_coverage_indices_from_assignment(
            mapped_line_id,
            coverage_count=len(current_coverages),
        )
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
            set_profile_count(profile, "filter_finalize_prune_iteration_count", prune_iteration_count)
            set_profile_count(profile, "filter_final_line_count", 0)
            return [], empty_assignment(n_prediction_columns)

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

    final_lines, assignment = sort_final_lines_and_remap_assignment(final_lines, assignment)
    set_profile_count(profile, "filter_finalize_prune_iteration_count", prune_iteration_count)
    set_profile_count(profile, "filter_final_line_count", len(final_lines))
    return final_lines, assignment


__all__ = [
    "build_coverage_indices_by_prediction_column",
    "compute_final_assignment",
    "empty_assignment",
    "final_line_sort_key",
    "finalize_outputs",
    "sort_final_lines_and_remap_assignment",
    "used_coverage_indices_from_assignment",
]
