from __future__ import annotations

"""Candidate preparation and coverage-object construction for line filtering.

The public filter receives raw Hough line dictionaries.  This module turns them
into normalized candidate lines and then into coverage objects that describe:

- which prediction columns a candidate crosses;
- which reference rows it samples at those columns;
- the local score-matrix value at each sample;
- compact sets/bounds used by the true-IoU overlap graph.
"""

import numpy as np

from . import filter_cython_accelerators as accelerators
from .filter_geometry_helpers import (
    fit_line_from_path,
    line_length,
    line_x_bounds,
    line_y_at_x,
    mean_line_support_for_filter,
    ref_segments_from_path,
)
from .filter_profile_fields import set_profile_count


def coverage_from_sampled_path(
    *,
    sampled_path: dict,
    matrix: np.ndarray,
    fallback_line: dict | None = None,
    source_raw_line_ids: list[int] | None = None,
) -> dict:
    """Build one coverage object from a path sampled by the Cython helper."""
    x_to_y = {
        int(prediction_column): int(reference_row)
        for prediction_column, reference_row in sampled_path["x_to_y"].items()
    }
    x_to_score = {
        int(prediction_column): float(local_score)
        for prediction_column, local_score in sampled_path["x_to_score"].items()
    }
    prediction_segments = set(int(prediction_column) for prediction_column in sampled_path["pred_segments"])
    reference_segments = set(int(reference_row) for reference_row in sampled_path["ref_segments"])
    representative_line = fit_line_from_path(
        x_to_y,
        x_to_score,
        matrix,
        fallback_line=fallback_line,
    )

    sampled_reference_rows = [
        int(reference_row)
        for reference_row in sampled_path.get("sampled_reference_rows", [])
    ]
    anchor_reference_row = (
        float(np.median(sampled_reference_rows))
        if sampled_reference_rows
        else float(min(representative_line["y0"], representative_line["y1"]))
    )

    return {
        "line": representative_line,
        "pred_segments": prediction_segments,
        "ref_segments": reference_segments,
        "pred_min": int(sampled_path.get("pred_min", min(prediction_segments) if prediction_segments else 0)),
        "pred_max": int(sampled_path.get("pred_max", max(prediction_segments) if prediction_segments else -1)),
        "ref_min": int(sampled_path.get("ref_min", min(reference_segments) if reference_segments else 0)),
        "ref_max": int(sampled_path.get("ref_max", max(reference_segments) if reference_segments else -1)),
        "x_to_y": x_to_y,
        "x_to_score": x_to_score,
        "total_score": float(sampled_path.get("total_score", sum(x_to_score.values()))),
        "mean_score": float(
            sampled_path.get(
                "mean_score",
                (sum(x_to_score.values()) / len(x_to_score)) if x_to_score else 0.0,
            )
        ),
        "anchor_y": anchor_reference_row,
        "source_raw_line_ids": sorted(int(raw_line_id) for raw_line_id in (source_raw_line_ids or [])),
    }


def coverage_from_path(
    *,
    x_to_y: dict[int, int],
    x_to_score: dict[int, float],
    matrix: np.ndarray,
    fallback_line: dict | None = None,
    source_raw_line_ids: list[int] | None = None,
) -> dict:
    """Normalize one Python-sampled path into the canonical coverage structure."""
    prediction_segments = set(int(prediction_column) for prediction_column in x_to_y)
    reference_segments = ref_segments_from_path(x_to_y)
    representative_line = fit_line_from_path(x_to_y, x_to_score, matrix, fallback_line=fallback_line)

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


def build_line_coverage(line: dict, matrix: np.ndarray) -> dict | None:
    """Project one candidate line onto the score-matrix grid."""
    if matrix.size == 0:
        return None

    n_reference_rows, n_prediction_columns = matrix.shape
    x_bounds = line_x_bounds(line, n_prediction_columns)
    if x_bounds is None:
        return None

    if accelerators.accelerated_sample_line_path is not None:
        sampled_path = accelerators.accelerated_sample_line_path(
            matrix,
            x0=float(line["x0"]),
            y0=float(line["y0"]),
            x1=float(line["x1"]),
            y1=float(line["y1"]),
        )
        if sampled_path is not None:
            raw_line_id = int(line.get("raw_line_id", -1)) if "raw_line_id" in line else -1
            return coverage_from_sampled_path(
                sampled_path=sampled_path,
                matrix=matrix,
                fallback_line=line,
                source_raw_line_ids=[raw_line_id] if raw_line_id >= 0 else [],
            )

    x_to_y: dict[int, int] = {}
    x_to_score: dict[int, float] = {}

    for prediction_column in range(x_bounds[0], x_bounds[1] + 1):
        sampled_reference_row = int(np.clip(round(line_y_at_x(line, prediction_column)), 0, n_reference_rows - 1))
        x_to_y[int(prediction_column)] = int(sampled_reference_row)
        x_to_score[int(prediction_column)] = float(matrix[sampled_reference_row, prediction_column])

    if not x_to_y:
        return None

    raw_line_id = int(line.get("raw_line_id", -1)) if "raw_line_id" in line else -1
    return coverage_from_path(
        x_to_y=x_to_y,
        x_to_score=x_to_score,
        matrix=matrix,
        fallback_line=line,
        source_raw_line_ids=[raw_line_id] if raw_line_id >= 0 else [],
    )


def local_path_key(coverage: dict, prediction_column: int):
    """Return the deterministic local winner key for one coverage at one column."""
    sampled_reference_row = int(coverage["x_to_y"][prediction_column])
    return (
        float(coverage["x_to_score"][prediction_column]),
        float(coverage.get("total_score", 0.0)),
        float(coverage.get("mean_score", 0.0)),
        -float(sampled_reference_row),
    )


def normalize_candidate_line(line: dict, matrix: np.ndarray, *, raw_line_id: int) -> dict:
    """Return one raw Hough line with filter fields populated and finite."""
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
        normalized_line["support"] = float(mean_line_support_for_filter(matrix, normalized_line))

    return normalized_line


def prepare_candidate_lines(
    lines: list[dict],
    matrix: np.ndarray,
    *,
    profile: dict | None = None,
) -> list[dict]:
    """Apply the unchanged coarse gates before the true-IoU merge stage.

    These gates are intentionally cheap.  They remove obviously unusable Hough
    lines, but they do not prove that a line is scientifically real.  The later
    true-IoU merge and ownership stages still decide final survival.
    """
    if not lines:
        set_profile_count(profile, "filter_input_line_count", 0)
        set_profile_count(profile, "filter_prepared_candidate_count", 0)
        return []

    set_profile_count(profile, "filter_input_line_count", len(lines))
    max_score = max(float(line.get("score", 0.0)) for line in lines)
    support_floor = float(np.percentile(matrix, 75)) if matrix.size > 0 else 0.0
    prepared_candidates: list[dict] = []

    for raw_line_id, line in enumerate(lines):
        prepared_line = normalize_candidate_line(line, matrix, raw_line_id=int(raw_line_id))

        if max_score > 0 and float(prepared_line.get("score", 0.0)) < 0.06 * max_score:
            continue
        if prepared_line["support"] < support_floor:
            continue
        prepared_candidates.append(prepared_line)

    if not prepared_candidates:
        set_profile_count(profile, "filter_fallback_candidate_used", 1)
        best_raw_line_id, best_line = max(
            enumerate(lines),
            key=lambda indexed_line: float(indexed_line[1].get("score", 0.0)),
        )
        prepared_candidates = [
            normalize_candidate_line(best_line, matrix, raw_line_id=int(best_raw_line_id))
        ]

    sorted_candidates = sorted(
        prepared_candidates,
        key=lambda line: (min(line["y0"], line["y1"]), min(line["x0"], line["x1"])),
    )
    set_profile_count(profile, "filter_prepared_candidate_count", len(sorted_candidates))
    return sorted_candidates


def build_candidate_coverages(candidate_lines: list[dict], matrix: np.ndarray) -> list[dict]:
    """Convert prepared candidate lines into coverage objects."""
    coverage_objects: list[dict] = []
    for candidate_line in candidate_lines:
        coverage_object = build_line_coverage(candidate_line, matrix)
        if coverage_object is None:
            continue
        coverage_objects.append(coverage_object)
    return coverage_objects


__all__ = [
    "build_candidate_coverages",
    "build_line_coverage",
    "coverage_from_path",
    "coverage_from_sampled_path",
    "local_path_key",
    "normalize_candidate_line",
    "prepare_candidate_lines",
]
