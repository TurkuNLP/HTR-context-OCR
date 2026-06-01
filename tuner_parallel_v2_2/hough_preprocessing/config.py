from __future__ import annotations

"""Configuration for score-matrix Hough preprocessing.

The preprocessing step runs once per score matrix.  It decides which matrix
cells are strong enough to vote in the Hough line detector.
"""

from dataclasses import asdict, dataclass


MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY = "manual_numpy"
MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY = "scipy"
SUPPORTED_MEDIAN_ABSOLUTE_DEVIATION_BACKENDS = (
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
)

CONNECTED_COMPONENT_BACKEND_CYTHON = "cython"
CONNECTED_COMPONENT_BACKEND_SCIPY = "scipy"
CONNECTED_COMPONENT_BACKEND_PYTHON = "python"
SUPPORTED_CONNECTED_COMPONENT_BACKENDS = (
    CONNECTED_COMPONENT_BACKEND_CYTHON,
    CONNECTED_COMPONENT_BACKEND_SCIPY,
    CONNECTED_COMPONENT_BACKEND_PYTHON,
)


@dataclass(frozen=True)
class HoughPreprocessingConfig:
    """User-facing controls for Region of Interest Hough preprocessing."""

    minimum_score_floor: float = 20.0
    median_absolute_deviation_multiplier: float = 0.0
    median_absolute_deviation_backend: str = MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY
    near_peak_ratio: float = 0.90
    near_peak_margin: float | None = None
    minimum_component_cells: int = 2
    minimum_component_rows: int = 1
    minimum_component_columns: int = 1
    connected_component_backend: str = CONNECTED_COMPONENT_BACKEND_CYTHON
    region_dilation_radius: int = 1
    minimum_active_cells: int = 5
    minimum_active_rows: int = 2
    minimum_active_columns: int = 2
    minimum_x_span: int = 2
    minimum_y_span: int = 2
    maximum_active_fraction: float = 0.08

    def __post_init__(self) -> None:
        """Reject settings that would make the preprocessing ambiguous."""
        if float(self.minimum_score_floor) < 0.0:
            raise ValueError("minimum_score_floor must be non-negative")
        if float(self.median_absolute_deviation_multiplier) < 0.0:
            raise ValueError("median_absolute_deviation_multiplier must be non-negative")
        if self.median_absolute_deviation_backend not in SUPPORTED_MEDIAN_ABSOLUTE_DEVIATION_BACKENDS:
            raise ValueError(
                "median_absolute_deviation_backend must be one of "
                f"{SUPPORTED_MEDIAN_ABSOLUTE_DEVIATION_BACKENDS!r}"
            )
        if not (0.0 < float(self.near_peak_ratio) <= 1.0):
            raise ValueError("near_peak_ratio must be greater than 0.0 and at most 1.0")
        if self.near_peak_margin is not None and float(self.near_peak_margin) < 0.0:
            raise ValueError("near_peak_margin must be non-negative when provided")
        if int(self.minimum_component_cells) < 1:
            raise ValueError("minimum_component_cells must be at least 1")
        if int(self.minimum_component_rows) < 1:
            raise ValueError("minimum_component_rows must be at least 1")
        if int(self.minimum_component_columns) < 1:
            raise ValueError("minimum_component_columns must be at least 1")
        if self.connected_component_backend not in SUPPORTED_CONNECTED_COMPONENT_BACKENDS:
            raise ValueError(
                "connected_component_backend must be one of "
                f"{SUPPORTED_CONNECTED_COMPONENT_BACKENDS!r}"
            )
        if int(self.region_dilation_radius) < 0:
            raise ValueError("region_dilation_radius must be non-negative")
        if int(self.minimum_active_cells) < 1:
            raise ValueError("minimum_active_cells must be at least 1")
        if int(self.minimum_active_rows) < 1:
            raise ValueError("minimum_active_rows must be at least 1")
        if int(self.minimum_active_columns) < 1:
            raise ValueError("minimum_active_columns must be at least 1")
        if int(self.minimum_x_span) < 1:
            raise ValueError("minimum_x_span must be at least 1")
        if int(self.minimum_y_span) < 1:
            raise ValueError("minimum_y_span must be at least 1")
        if not (0.0 < float(self.maximum_active_fraction) <= 1.0):
            raise ValueError("maximum_active_fraction must be greater than 0.0 and at most 1.0")

    def as_dict(self) -> dict:
        """Return a JSON-friendly copy of the active preprocessing settings."""
        return asdict(self)
