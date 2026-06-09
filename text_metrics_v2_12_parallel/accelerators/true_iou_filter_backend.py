"""Optional exact-result backend hooks for the production true-IoU filter.

The true-IoU filter keeps its readable Python implementation as the reference
path. This module isolates two hot helper boundaries so an optional compiled
backend can accelerate them without changing filtering semantics:

1. sampling one geometric line onto score-matrix columns,
2. selecting per-column local winners while merging one overlap component.

Compiled modules are loaded lazily from the external runtime-artifact cache. If
anything is unavailable, the current Python reference logic remains in control.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from accelerators.load_optional_exact_result_cython_backends import (
    load_optional_exact_result_cython_backend_module,
)

__all__ = [
    "USING_COMPILED_TRUE_IOU_FILTER_BACKEND",
    "sample_line_path_with_optional_accelerator",
    "select_component_local_winners_with_optional_accelerator",
]

USING_COMPILED_TRUE_IOU_FILTER_BACKEND = False


# Resolve the optional compiled helper module lazily after startup-time build preparation.
def _load_compiled_true_iou_backend_module():
    """Return the optional compiled true-IoU helper module when available."""
    global USING_COMPILED_TRUE_IOU_FILTER_BACKEND

    compiled_backend_module = load_optional_exact_result_cython_backend_module(
        "true_iou_filter_backend"
    )
    USING_COMPILED_TRUE_IOU_FILTER_BACKEND = compiled_backend_module is not None
    return compiled_backend_module


# Sample one geometric line onto prediction columns using the active backend.
def sample_line_path_with_optional_accelerator(
    line: dict,
    matrix: np.ndarray,
    *,
    line_x_bounds_function: Callable[[dict, int], tuple[int, int] | None],
    line_y_at_x_function: Callable[[dict, float], float],
) -> tuple[dict[int, int], dict[int, float]] | None:
    """Return the sampled x->y and x->score maps for one line.

    The fallback path intentionally mirrors the existing Python logic exactly so
    it remains safe to compare against the optional compiled implementation.
    """
    compiled_backend_module = _load_compiled_true_iou_backend_module()
    if compiled_backend_module is not None:
        compiled_result = compiled_backend_module.sample_line_path_exact(line, matrix)
        if compiled_result is None:
            return None
        sampled_x_to_y, sampled_x_to_score = compiled_result
        return (
            {int(prediction_column): int(reference_row) for prediction_column, reference_row in sampled_x_to_y.items()},
            {int(prediction_column): float(local_score) for prediction_column, local_score in sampled_x_to_score.items()},
        )

    if matrix.size == 0:
        return None

    n_reference_rows, n_prediction_columns = matrix.shape
    x_bounds = line_x_bounds_function(line, n_prediction_columns)
    if x_bounds is None:
        return None

    sampled_x_to_y: dict[int, int] = {}
    sampled_x_to_score: dict[int, float] = {}

    # Sample the line once for every covered prediction column.
    for prediction_column in range(x_bounds[0], x_bounds[1] + 1):
        sampled_reference_row = int(
            np.clip(
                round(line_y_at_x_function(line, prediction_column)),
                0,
                n_reference_rows - 1,
            )
        )
        sampled_x_to_y[int(prediction_column)] = int(sampled_reference_row)
        sampled_x_to_score[int(prediction_column)] = float(matrix[sampled_reference_row, prediction_column])

    if not sampled_x_to_y:
        return None

    return sampled_x_to_y, sampled_x_to_score


# Merge the per-column local winners for one overlap component using the active backend.
def select_component_local_winners_with_optional_accelerator(
    component_coverages: list[dict],
    *,
    local_path_key_function: Callable[[dict, int], tuple],
) -> tuple[dict[int, int], dict[int, float], list[int]]:
    """Return the merged per-column winners for one overlap component.

    The fallback path preserves the current tie-breaking rule exactly: the first
    coverage that attains the best local key at one prediction column stays the
    winner for that column.
    """
    compiled_backend_module = _load_compiled_true_iou_backend_module()
    if compiled_backend_module is not None:
        compiled_result = compiled_backend_module.select_component_local_winners_exact(component_coverages)
        merged_x_to_y, merged_x_to_score, merged_source_raw_line_ids = compiled_result
        return (
            {int(prediction_column): int(reference_row) for prediction_column, reference_row in merged_x_to_y.items()},
            {int(prediction_column): float(local_score) for prediction_column, local_score in merged_x_to_score.items()},
            [int(raw_line_id) for raw_line_id in merged_source_raw_line_ids],
        )

    best_path_sample_by_column: dict[int, tuple[tuple, int, float]] = {}

    # Keep the first coverage that attains the best local key at each x.
    for coverage in component_coverages:
        for prediction_column, sampled_reference_row in coverage["x_to_y"].items():
            local_winner_key = local_path_key_function(coverage, int(prediction_column))
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

    return merged_x_to_y, merged_x_to_score, merged_source_raw_line_ids
