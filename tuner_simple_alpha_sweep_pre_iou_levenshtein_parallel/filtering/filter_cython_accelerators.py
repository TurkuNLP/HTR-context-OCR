from __future__ import annotations

"""Optional compiled helpers used by the filtering modules.

The Python modules in ``filtering/`` are the readable reference implementation.
The compiled functions imported here are exact accelerators for narrow inner
loops.  Caller modules import this module, not individual variables, so tests can
temporarily disable one accelerator by monkeypatching the attribute here.
"""

try:
    from ..cython_accel.optional_filtering import (
        build_coverage_indices_by_prediction_column as accelerated_build_coverage_indices,
        compute_final_assignment_from_coverages as accelerated_compute_final_assignment,
        mean_line_support_from_endpoints as accelerated_mean_line_support_from_endpoints,
        sample_line_path as accelerated_sample_line_path,
        set_iou as accelerated_set_iou,
        unique_reference_rows_from_path_slice as accelerated_unique_reference_rows_from_path_slice,
    )
except ImportError:
    try:
        from cython_accel.optional_filtering import (  # type: ignore
            build_coverage_indices_by_prediction_column as accelerated_build_coverage_indices,
            compute_final_assignment_from_coverages as accelerated_compute_final_assignment,
            mean_line_support_from_endpoints as accelerated_mean_line_support_from_endpoints,
            sample_line_path as accelerated_sample_line_path,
            set_iou as accelerated_set_iou,
            unique_reference_rows_from_path_slice as accelerated_unique_reference_rows_from_path_slice,
        )
    except ImportError:
        accelerated_build_coverage_indices = None  # type: ignore
        accelerated_compute_final_assignment = None  # type: ignore
        accelerated_mean_line_support_from_endpoints = None  # type: ignore
        accelerated_sample_line_path = None  # type: ignore
        accelerated_set_iou = None  # type: ignore
        accelerated_unique_reference_rows_from_path_slice = None  # type: ignore


__all__ = [
    "accelerated_build_coverage_indices",
    "accelerated_compute_final_assignment",
    "accelerated_mean_line_support_from_endpoints",
    "accelerated_sample_line_path",
    "accelerated_set_iou",
    "accelerated_unique_reference_rows_from_path_slice",
]
