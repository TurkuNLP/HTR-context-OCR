from __future__ import annotations

"""Configuration for score-matrix Hough preprocessing.

The preprocessing step runs once per score matrix.  It decides which matrix
cells are strong enough to define a Region of Interest and which cells are
allowed to vote in the Hough line detector.
"""

from dataclasses import asdict, dataclass


MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY = "manual_numpy"
MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY = "scipy"
SUPPORTED_MEDIAN_ABSOLUTE_DEVIATION_BACKENDS = (
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
)

SCORE_FLOOR_METHOD_MEAN_PLUS_STANDARD_DEVIATION = "mean_plus_standard_deviation"
SCORE_FLOOR_METHOD_MEDIAN_PLUS_SCALED_MEDIAN_ABSOLUTE_DEVIATION = (
    "median_plus_scaled_median_absolute_deviation"
)
SUPPORTED_SCORE_FLOOR_METHODS = (
    SCORE_FLOOR_METHOD_MEAN_PLUS_STANDARD_DEVIATION,
    SCORE_FLOOR_METHOD_MEDIAN_PLUS_SCALED_MEDIAN_ABSOLUTE_DEVIATION,
)

FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST = "roi"
FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST_AND_SCORE_FLOOR = "roi_and_score_floor"
FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST_OR_SCORE_FLOOR = "roi_or_score_floor"
SUPPORTED_FINAL_HOUGH_INPUT_MODES = (
    FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST,
    FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST_AND_SCORE_FLOOR,
    FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST_OR_SCORE_FLOOR,
)

ADAPTIVE_BUDGET_MASK_FINAL_HOUGH_INPUT = "final_hough_input"
ADAPTIVE_BUDGET_MASK_REGION_OF_INTEREST = "region_of_interest"
ADAPTIVE_BUDGET_MASK_STRONG_MATCH = "strong_match"
ADAPTIVE_BUDGET_MASK_COMPONENT_REGION = "component_region"
ADAPTIVE_BUDGET_MASK_SCORE_FLOOR = "score_floor"
SUPPORTED_ADAPTIVE_BUDGET_MASKS = (
    ADAPTIVE_BUDGET_MASK_FINAL_HOUGH_INPUT,
    ADAPTIVE_BUDGET_MASK_REGION_OF_INTEREST,
    ADAPTIVE_BUDGET_MASK_STRONG_MATCH,
    ADAPTIVE_BUDGET_MASK_COMPONENT_REGION,
    ADAPTIVE_BUDGET_MASK_SCORE_FLOOR,
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
    score_floor_method: str = SCORE_FLOOR_METHOD_MEAN_PLUS_STANDARD_DEVIATION
    median_absolute_deviation_multiplier: float = 0.0
    median_absolute_deviation_backend: str = MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY
    near_peak_ratio: float | None = 0.70
    near_peak_margin: float | None = None
    minimum_component_cells: int = 2
    minimum_component_rows: int = 1
    minimum_component_columns: int = 1
    connected_component_backend: str = CONNECTED_COMPONENT_BACKEND_CYTHON
    region_dilation_radius: int = 1
    final_hough_input_mode: str = FINAL_HOUGH_INPUT_MODE_REGION_OF_INTEREST
    adaptive_budget_mask: str = ADAPTIVE_BUDGET_MASK_STRONG_MATCH
    minimum_active_cells: int = 0
    minimum_active_rows: int = 2
    minimum_active_columns: int = 2
    minimum_x_span: int = 2
    minimum_y_span: int = 2
    maximum_active_fraction: float = 1.0
    minimum_matrix_rows: int = 4
    minimum_matrix_columns: int = 4

    def __post_init__(self) -> None:
        """Reject settings that would make preprocessing ambiguous."""
        if float(self.minimum_score_floor) < 0.0:
            raise ValueError("minimum_score_floor must be non-negative")
        if self.score_floor_method not in SUPPORTED_SCORE_FLOOR_METHODS:
            raise ValueError(f"score_floor_method must be one of {SUPPORTED_SCORE_FLOOR_METHODS!r}")
        if float(self.median_absolute_deviation_multiplier) < 0.0:
            raise ValueError("median_absolute_deviation_multiplier must be non-negative")
        if self.median_absolute_deviation_backend not in SUPPORTED_MEDIAN_ABSOLUTE_DEVIATION_BACKENDS:
            raise ValueError(
                "median_absolute_deviation_backend must be one of "
                f"{SUPPORTED_MEDIAN_ABSOLUTE_DEVIATION_BACKENDS!r}"
            )
        if self.near_peak_ratio is not None and not (0.0 < float(self.near_peak_ratio) <= 1.0):
            raise ValueError("near_peak_ratio must be greater than 0.0 and at most 1.0 when provided")
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
        if self.final_hough_input_mode not in SUPPORTED_FINAL_HOUGH_INPUT_MODES:
            raise ValueError(f"final_hough_input_mode must be one of {SUPPORTED_FINAL_HOUGH_INPUT_MODES!r}")
        if self.adaptive_budget_mask not in SUPPORTED_ADAPTIVE_BUDGET_MASKS:
            raise ValueError(f"adaptive_budget_mask must be one of {SUPPORTED_ADAPTIVE_BUDGET_MASKS!r}")
        if int(self.minimum_active_cells) < 0:
            raise ValueError("minimum_active_cells must be non-negative")
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
        if int(self.minimum_matrix_rows) < 0:
            raise ValueError("minimum_matrix_rows must be non-negative")
        if int(self.minimum_matrix_columns) < 0:
            raise ValueError("minimum_matrix_columns must be non-negative")

    def as_dict(self) -> dict:
        """Return a JSON-friendly copy of the active preprocessing settings."""
        return asdict(self)
