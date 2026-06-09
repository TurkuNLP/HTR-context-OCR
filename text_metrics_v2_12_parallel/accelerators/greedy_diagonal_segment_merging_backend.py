"""Optional exact-result backend loader for greedy diagonal segment merging.

This module gives the production pipeline one narrow hook where a compiled
implementation can be inserted without changing the surrounding Hough
postprocessing code. The backend itself is optional and lives in the external
runtime-artifact cache. If no compiled backend is available, the exact Python
reference implementation remains authoritative.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from accelerators.load_optional_exact_result_cython_backends import (
    load_optional_exact_result_cython_backend_module,
)

__all__ = [
    "USING_COMPILED_GREEDY_DIAGONAL_SEGMENT_MERGING_BACKEND",
    "merge_diagonal_segments_with_optional_accelerator",
]

USING_COMPILED_GREEDY_DIAGONAL_SEGMENT_MERGING_BACKEND = False


# Resolve the optional compiled merge function lazily after startup-time build preparation.
def _load_compiled_merge_function() -> Callable | None:
    """Return the compiled merge function when one is available."""
    global USING_COMPILED_GREEDY_DIAGONAL_SEGMENT_MERGING_BACKEND

    compiled_backend_module = load_optional_exact_result_cython_backend_module(
        "greedy_diagonal_segment_merging_backend"
    )
    if compiled_backend_module is None:
        USING_COMPILED_GREEDY_DIAGONAL_SEGMENT_MERGING_BACKEND = False
        return None

    compiled_merge_function = getattr(
        compiled_backend_module,
        "merge_diagonal_segments_exact",
        None,
    )
    USING_COMPILED_GREEDY_DIAGONAL_SEGMENT_MERGING_BACKEND = compiled_merge_function is not None
    return compiled_merge_function


# Route merge-diagonal work through the compiled backend when one is available.
def merge_diagonal_segments_with_optional_accelerator(
    raw_hough_segments: list[tuple[tuple[float, float], tuple[float, float]]],
    active_mask: np.ndarray,
    active_mask_points_xy: list[tuple[int, int]],
    *,
    python_reference_merge_function: Callable[
        [
            list[tuple[tuple[float, float], tuple[float, float]]],
            np.ndarray,
            list[tuple[int, int]],
        ],
        list[tuple[tuple[float, float], tuple[float, float]]],
    ],
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Return merged diagonal segments using the optional exact-result backend.

    The public production contract stays intentionally simple:

    - if a compiled exact-result backend exists, use it,
    - otherwise fall back to the Python reference implementation.

    The Python implementation remains the correctness reference.
    """
    compiled_merge_function = _load_compiled_merge_function()
    if compiled_merge_function is None:
        return python_reference_merge_function(
            raw_hough_segments,
            active_mask,
            active_mask_points_xy,
        )

    return compiled_merge_function(
        raw_hough_segments,
        active_mask,
        active_mask_points_xy,
    )
