"""Self-contained v2.12 metric compatibility layer for the tuner."""

from .line_coverage_arrays import (
    build_line_coverage_arrays_from_bundles,
    build_line_coverage_arrays_from_cached_refref_y,
    build_other_line_coverage_arrays_from_bundle,
    build_refref_y_coverage_array_from_bundle,
    compute_line_coverage_percentage_metrics_from_arrays,
    compute_line_coverage_ratio_metrics_from_arrays,
)
from .line_metric_bundle import (
    accumulate_counts_from_interval_groups,
    build_line_metric_bundle,
    reference_rows_for_levenshtein,
)

__all__ = [
    "accumulate_counts_from_interval_groups",
    "build_line_metric_bundle",
    "reference_rows_for_levenshtein",
    "build_line_coverage_arrays_from_bundles",
    "build_line_coverage_arrays_from_cached_refref_y",
    "build_other_line_coverage_arrays_from_bundle",
    "build_refref_y_coverage_array_from_bundle",
    "compute_line_coverage_percentage_metrics_from_arrays",
    "compute_line_coverage_ratio_metrics_from_arrays",
]
