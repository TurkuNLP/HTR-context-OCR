from __future__ import annotations

"""Hough sweep-grid helpers.

The tuner loops over inclusive integer ranges.  This module centralizes range
expansion and combination counting so the CLI, scheduler, logs, tests, and JSON
summary all describe the same active grid.
"""

from .tuner_config import (
    HoughSweepRanges,
    PARAM_HOUGH_LINE_GAP,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_SEED,
    PARAM_HOUGH_THRESHOLD,
    SUPPORTED_SWEEP_PARAMETERS,
    default_hough_sweep_ranges,
)


def build_sweep_values(active_ranges: HoughSweepRanges | None = None) -> dict[str, list[int]]:
    """Return active inclusive value lists for every tuned Hough parameter."""
    ranges = default_hough_sweep_ranges() if active_ranges is None else active_ranges
    return ranges.values_by_parameter()


def build_fixed_sweep_values() -> dict[str, list[int]]:
    """Compatibility wrapper returning the default full-grid values."""
    return build_sweep_values(default_hough_sweep_ranges())


def combinations_per_document(active_ranges: HoughSweepRanges | None = None) -> int:
    """Return the exact evaluated-combination count for one document."""
    values_by_parameter = build_sweep_values(active_ranges)
    total = 1
    for parameter_name in SUPPORTED_SWEEP_PARAMETERS:
        total *= len(values_by_parameter[parameter_name])
    return int(total)


def combinations_per_threshold_worker(active_ranges: HoughSweepRanges | None = None) -> int:
    """Return the serial work owned by one threshold worker for one document."""
    values_by_parameter = build_sweep_values(active_ranges)
    return int(
        len(values_by_parameter[PARAM_HOUGH_LINE_LENGTH])
        * len(values_by_parameter[PARAM_HOUGH_LINE_GAP])
        * len(values_by_parameter[PARAM_HOUGH_SEED])
    )


def threshold_worker_count_per_document(active_ranges: HoughSweepRanges | None = None) -> int:
    """Return natural threshold-worker count for one fully parallel document."""
    return int(len(build_sweep_values(active_ranges)[PARAM_HOUGH_THRESHOLD]))


def active_grid_summary(active_ranges: HoughSweepRanges | None = None) -> dict[str, dict[str, int]]:
    """Return active grid ranges in the summary JSON schema."""
    ranges = default_hough_sweep_ranges() if active_ranges is None else active_ranges
    return ranges.as_summary_dict()


__all__ = [
    "PARAM_HOUGH_THRESHOLD",
    "build_sweep_values",
    "build_fixed_sweep_values",
    "combinations_per_document",
    "combinations_per_threshold_worker",
    "threshold_worker_count_per_document",
    "active_grid_summary",
]
