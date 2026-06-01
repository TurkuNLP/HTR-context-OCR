"""Optional Cython acceleration package for ``tuner_parallel_v2_1``.

The pure-Python tuner remains the source of truth for behavior.  Modules in this
package are optional accelerators and must preserve outputs exactly when they
are compiled and available.
"""

from .optional_line_grouping import (
    cython_line_grouping_available,
    group_owned_columns_by_line,
)
from .optional_filtering import (
    build_coverage_indices_by_prediction_column,
    compute_final_assignment_from_coverages,
    cython_final_assignment_available,
    cython_filtering_helpers_available,
    cython_line_sampling_available,
    mean_line_support_from_endpoints,
    sample_line_path,
    set_iou,
)

__all__ = [
    "build_coverage_indices_by_prediction_column",
    "compute_final_assignment_from_coverages",
    "cython_final_assignment_available",
    "cython_filtering_helpers_available",
    "cython_line_sampling_available",
    "cython_line_grouping_available",
    "group_owned_columns_by_line",
    "mean_line_support_from_endpoints",
    "sample_line_path",
    "set_iou",
]
