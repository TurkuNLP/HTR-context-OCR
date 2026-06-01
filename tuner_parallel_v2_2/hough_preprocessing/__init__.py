from __future__ import annotations

"""Score-matrix preprocessing for binary Hough line detection."""

from .config import (
    CONNECTED_COMPONENT_BACKEND_CYTHON,
    CONNECTED_COMPONENT_BACKEND_PYTHON,
    CONNECTED_COMPONENT_BACKEND_SCIPY,
    HoughPreprocessingConfig,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
)
from .region_of_interest import build_region_of_interest_hough_context

__all__ = [
    "CONNECTED_COMPONENT_BACKEND_CYTHON",
    "CONNECTED_COMPONENT_BACKEND_PYTHON",
    "CONNECTED_COMPONENT_BACKEND_SCIPY",
    "HoughPreprocessingConfig",
    "MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY",
    "MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY",
    "build_region_of_interest_hough_context",
]
