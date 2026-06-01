from __future__ import annotations

"""Build binary Hough inputs from strong score-matrix regions."""

import time
from dataclasses import asdict

import numpy as np

from .config import HoughPreprocessingConfig
from .connected_components import active_mask_geometry, dilate_mask, label_connected_components
from .matrix_statistics import summarize_score_matrix


def _empty_boolean_mask(score_matrix: np.ndarray) -> np.ndarray:
    """Return an empty boolean mask with the same shape as the score matrix."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.ndim != 2:
        return np.zeros((0, 0), dtype=bool)
    return np.zeros(matrix.shape, dtype=bool)


def _context_from_masks(
    *,
    hough_input_mask: np.ndarray,
    score_floor_mask: np.ndarray,
    near_peak_score_mask: np.ndarray,
    strong_match_mask: np.ndarray,
    region_of_interest_mask: np.ndarray,
    summary: dict,
    keep_debug_arrays: bool,
) -> dict:
    """Create the dictionary consumed by the existing Hough detector."""
    binary_hough_image = np.asarray(hough_input_mask, dtype=bool).astype(np.uint8)
    context = {
        "hough_image": binary_hough_image,
        "hough_mask_bool": np.asarray(hough_input_mask, dtype=bool),
        "mask": binary_hough_image,
        "threshold_start": float("nan"),
        "preprocessing_mode": "region_of_interest",
        "hough_preprocessing_accepted": bool(summary.get("accepted", False)),
        "hough_preprocessing_rejection_reason": str(summary.get("rejection_reason", "")),
        "hough_preprocessing_summary": dict(summary),
        "region_of_interest_mask_bool": np.asarray(region_of_interest_mask, dtype=bool),
        "strong_match_mask_bool": np.asarray(strong_match_mask, dtype=bool),
        "score_floor_mask_bool": np.asarray(score_floor_mask, dtype=bool),
        "near_peak_score_mask_bool": np.asarray(near_peak_score_mask, dtype=bool),
        # Compatibility aliases keep the existing experiment visualisation code
        # able to inspect these masks without changing its record reader.
        "roi_mask_bool": np.asarray(region_of_interest_mask, dtype=bool),
        "strong_evidence_mask_bool": np.asarray(strong_match_mask, dtype=bool),
        "roi_experiment_rejected": not bool(summary.get("accepted", False)),
        "roi_experiment_rejection_reason": str(summary.get("rejection_reason", "")),
        "roi_experiment_stats": dict(summary),
    }
    if not keep_debug_arrays:
        return context
    context["debug_score_floor_mask"] = np.asarray(score_floor_mask, dtype=bool)
    context["debug_near_peak_score_mask"] = np.asarray(near_peak_score_mask, dtype=bool)
    context["debug_strong_match_mask"] = np.asarray(strong_match_mask, dtype=bool)
    context["debug_region_of_interest_mask"] = np.asarray(region_of_interest_mask, dtype=bool)
    return context


def _build_rejected_context(
    *,
    score_matrix: np.ndarray,
    config: HoughPreprocessingConfig,
    rejection_reason: str,
    started_at: float,
    statistics: dict | None = None,
    keep_debug_arrays: bool = False,
) -> dict:
    """Return a zero-vote Hough context for a rejected matrix."""
    empty_mask = _empty_boolean_mask(score_matrix)
    row_count, column_count = empty_mask.shape
    summary = {
        "accepted": False,
        "rejection_reason": str(rejection_reason),
        "row_count": int(row_count),
        "column_count": int(column_count),
        "matrix_cell_count": int(empty_mask.size),
        "preprocessing_seconds": float(time.perf_counter() - started_at),
        "config": config.as_dict(),
    }
    if statistics is not None:
        summary.update(statistics)
    summary.update(active_mask_geometry(empty_mask))
    return _context_from_masks(
        hough_input_mask=empty_mask,
        score_floor_mask=empty_mask,
        near_peak_score_mask=empty_mask,
        strong_match_mask=empty_mask,
        region_of_interest_mask=empty_mask,
        summary=summary,
        keep_debug_arrays=bool(keep_debug_arrays),
    )


def _component_passes_region_gate(
    *,
    component,
    config: HoughPreprocessingConfig,
) -> bool:
    """Return True when a connected component is large enough to search."""
    return (
        int(component.cell_count) >= int(config.minimum_component_cells)
        and int(component.row_count) >= int(config.minimum_component_rows)
        and int(component.column_count) >= int(config.minimum_component_columns)
    )


def _first_geometry_rejection_reason(
    *,
    geometry: dict[str, int | float],
    kept_component_count: int,
    config: HoughPreprocessingConfig,
) -> str:
    """Return the first reason that makes the final Hough input unusable."""
    if int(kept_component_count) <= 0:
        return "no_line_like_region_of_interest"
    if int(geometry["active_cell_count"]) < int(config.minimum_active_cells):
        return "insufficient_hough_evidence"
    if int(geometry["active_row_count"]) < int(config.minimum_active_rows):
        return "insufficient_active_rows"
    if int(geometry["active_column_count"]) < int(config.minimum_active_columns):
        return "insufficient_active_columns"
    if int(geometry["x_span"]) < int(config.minimum_x_span):
        return "insufficient_x_span"
    if int(geometry["y_span"]) < int(config.minimum_y_span):
        return "insufficient_y_span"
    if float(geometry["active_fraction"]) > float(config.maximum_active_fraction):
        return "ambiguous_or_too_dense"
    return ""


def build_region_of_interest_hough_context(
    score_matrix: np.ndarray,
    *,
    config: HoughPreprocessingConfig | None = None,
    keep_debug_arrays: bool = False,
) -> dict:
    """Build one binary Hough input from locally strong matrix cells."""
    started_at = time.perf_counter()
    preprocessing_config = HoughPreprocessingConfig() if config is None else config
    matrix = np.asarray(score_matrix, dtype=float)

    if matrix.ndim != 2:
        return _build_rejected_context(
            score_matrix=matrix,
            config=preprocessing_config,
            rejection_reason="not_a_two_dimensional_matrix",
            started_at=started_at,
            keep_debug_arrays=bool(keep_debug_arrays),
        )
    if matrix.size == 0 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return _build_rejected_context(
            score_matrix=matrix,
            config=preprocessing_config,
            rejection_reason="empty_matrix",
            started_at=started_at,
            keep_debug_arrays=bool(keep_debug_arrays),
        )

    statistics = summarize_score_matrix(
        matrix,
        median_absolute_deviation_backend=preprocessing_config.median_absolute_deviation_backend,
    )
    statistics_dict = statistics.as_dict()
    if not statistics.has_finite_scores:
        return _build_rejected_context(
            score_matrix=matrix,
            config=preprocessing_config,
            rejection_reason="no_finite_scores",
            started_at=started_at,
            statistics=statistics_dict,
            keep_debug_arrays=bool(keep_debug_arrays),
        )

    adaptive_score_floor = float(
        statistics.score_median
        + float(preprocessing_config.median_absolute_deviation_multiplier)
        * float(statistics.scaled_median_absolute_deviation)
    )
    final_score_floor = float(max(float(preprocessing_config.minimum_score_floor), adaptive_score_floor))
    statistics_dict.update(
        {
            "adaptive_score_floor": adaptive_score_floor,
            "score_floor": final_score_floor,
        }
    )

    if float(statistics.score_maximum) < final_score_floor:
        return _build_rejected_context(
            score_matrix=matrix,
            config=preprocessing_config,
            rejection_reason="no_strong_match_evidence",
            started_at=started_at,
            statistics=statistics_dict,
            keep_debug_arrays=bool(keep_debug_arrays),
        )

    finite_score_matrix = np.where(np.isfinite(matrix), matrix, -np.inf)
    score_floor_mask = finite_score_matrix >= final_score_floor

    row_peak_scores = np.max(finite_score_matrix, axis=1)
    column_peak_scores = np.max(finite_score_matrix, axis=0)
    row_near_peak_mask = finite_score_matrix >= (row_peak_scores[:, None] * float(preprocessing_config.near_peak_ratio))
    column_near_peak_mask = finite_score_matrix >= (column_peak_scores[None, :] * float(preprocessing_config.near_peak_ratio))

    if preprocessing_config.near_peak_margin is not None:
        margin = float(preprocessing_config.near_peak_margin)
        row_near_peak_mask |= finite_score_matrix >= (row_peak_scores[:, None] - margin)
        column_near_peak_mask |= finite_score_matrix >= (column_peak_scores[None, :] - margin)

    finite_cell_mask = np.isfinite(matrix)
    near_peak_score_mask = finite_cell_mask & (row_near_peak_mask | column_near_peak_mask)
    strong_match_mask = score_floor_mask & near_peak_score_mask

    component_labels, components, connected_component_backend_used = label_connected_components(
        strong_match_mask,
        backend=preprocessing_config.connected_component_backend,
    )
    kept_component_labels = {
        int(component.label)
        for component in components
        if _component_passes_region_gate(component=component, config=preprocessing_config)
    }
    if kept_component_labels:
        component_region_mask = np.isin(component_labels, list(kept_component_labels))
    else:
        component_region_mask = np.zeros_like(strong_match_mask, dtype=bool)

    region_of_interest_mask = dilate_mask(
        component_region_mask,
        radius=int(preprocessing_config.region_dilation_radius),
    )
    hough_input_mask = strong_match_mask & region_of_interest_mask
    geometry = active_mask_geometry(hough_input_mask)
    rejection_reason = _first_geometry_rejection_reason(
        geometry=geometry,
        kept_component_count=len(kept_component_labels),
        config=preprocessing_config,
    )

    summary = {
        "accepted": rejection_reason == "",
        "rejection_reason": rejection_reason,
        "row_count": int(matrix.shape[0]),
        "column_count": int(matrix.shape[1]),
        "matrix_cell_count": int(matrix.size),
        "strong_match_cell_count": int(np.count_nonzero(strong_match_mask)),
        "strong_match_fraction": float(np.count_nonzero(strong_match_mask) / matrix.size),
        "region_of_interest_cell_count": int(np.count_nonzero(region_of_interest_mask)),
        "region_of_interest_fraction": float(np.count_nonzero(region_of_interest_mask) / matrix.size),
        "component_count": int(len(components)),
        "connected_component_backend_requested": preprocessing_config.connected_component_backend,
        "connected_component_backend_used": connected_component_backend_used,
        "kept_component_count": int(len(kept_component_labels)),
        "dropped_component_count": int(len(components) - len(kept_component_labels)),
        "preprocessing_seconds": float(time.perf_counter() - started_at),
        "config": preprocessing_config.as_dict(),
        **statistics_dict,
        **geometry,
    }

    return _context_from_masks(
        hough_input_mask=hough_input_mask,
        score_floor_mask=score_floor_mask,
        near_peak_score_mask=near_peak_score_mask,
        strong_match_mask=strong_match_mask,
        region_of_interest_mask=region_of_interest_mask,
        summary=summary,
        keep_debug_arrays=bool(keep_debug_arrays),
    )


def preprocessing_summary_from_hough_context(hough_context: dict | None) -> dict:
    """Extract the stored preprocessing summary from a Hough context."""
    if not isinstance(hough_context, dict):
        return {}
    summary = hough_context.get("hough_preprocessing_summary")
    return dict(summary) if isinstance(summary, dict) else {}
