from __future__ import annotations

"""Configuration primitives and shared types for ``tuner_parallel_v2``.

This module is the single source of truth for the Hough sweep grid, document
payload shape, and shared parameter names.  Keeping these definitions here makes
it much harder for the CLI, scheduler, CSV exports, plots, and README to drift
apart.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    from ..filtering.line_filtering_v2_1_IoU_fast import DEFAULT_MIN_IOU_THRESHOLD
except ImportError:
    from filtering.line_filtering_v2_1_IoU_fast import DEFAULT_MIN_IOU_THRESHOLD  # type: ignore


LogFn = Callable[[str], None]
MATRIX_CACHE_VERSION = "score_matrix_v2_strict_theta_30_60"

PARAM_HOUGH_THRESHOLD = "hough_threshold"
PARAM_HOUGH_LINE_LENGTH = "hough_line_length"
PARAM_HOUGH_LINE_GAP = "hough_line_gap"
PARAM_HOUGH_SEED = "hough_seed"

SUPPORTED_SWEEP_PARAMETERS: tuple[str, ...] = (
    PARAM_HOUGH_THRESHOLD,
    PARAM_HOUGH_LINE_LENGTH,
    PARAM_HOUGH_LINE_GAP,
    PARAM_HOUGH_SEED,
)

HOUGH_THRESHOLD_MIN = 1
HOUGH_THRESHOLD_MAX = 40
HOUGH_LINE_LENGTH_MIN = 1
HOUGH_LINE_LENGTH_MAX = 50
HOUGH_LINE_GAP_MIN = 1
HOUGH_LINE_GAP_MAX = 30
FIXED_HOUGH_SEED = 1
HOUGH_SEED_MIN = FIXED_HOUGH_SEED
HOUGH_SEED_MAX = FIXED_HOUGH_SEED

DEFAULT_SCORE_INDEX_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "text_metrics_v2_1_parallel" / ".score_index_cache"
)
DEFAULT_TEXT_METRICS_V212_DIR = Path(__file__).resolve().parents[2] / "text_metrics_v2_12_parallel"
DEFAULT_REF_TO_REF_COMBO_CACHE_DIR = (
    Path(__file__).resolve().parents[2] / "results" / "tuner_parallel_v2_cache" / "ref_to_ref_combo_cache_v1"
)


@dataclass(frozen=True)
class InclusiveIntegerRange:
    """Inclusive integer range used by one Hough sweep parameter."""

    minimum: int
    maximum: int
    allow_zero: bool = False

    def __post_init__(self) -> None:
        """Validate sign and ordering constraints for one inclusive range."""
        min_value = int(self.minimum)
        max_value = int(self.maximum)
        lower_bound = 0 if bool(self.allow_zero) else 1
        if min_value < lower_bound or max_value < lower_bound:
            raise ValueError(
                f"Range values must be >= {lower_bound}, got minimum={min_value}, maximum={max_value}"
            )
        if min_value > max_value:
            raise ValueError(f"Range minimum must be <= maximum, got {min_value}>{max_value}")

    def values(self) -> list[int]:
        """Expand the inclusive range into the exact integer values to sweep."""
        return list(range(int(self.minimum), int(self.maximum) + 1))

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-friendly representation for run summaries."""
        return {"min": int(self.minimum), "max": int(self.maximum)}

    def label(self) -> str:
        """Return compact ``MIN..MAX`` text for logs and examples."""
        return f"{int(self.minimum)}..{int(self.maximum)}"


@dataclass(frozen=True)
class HoughSweepRanges:
    """All active inclusive Hough ranges for one tuner run."""

    threshold: InclusiveIntegerRange = InclusiveIntegerRange(HOUGH_THRESHOLD_MIN, HOUGH_THRESHOLD_MAX)
    line_length: InclusiveIntegerRange = InclusiveIntegerRange(HOUGH_LINE_LENGTH_MIN, HOUGH_LINE_LENGTH_MAX)
    line_gap: InclusiveIntegerRange = InclusiveIntegerRange(HOUGH_LINE_GAP_MIN, HOUGH_LINE_GAP_MAX)
    seed: InclusiveIntegerRange = InclusiveIntegerRange(HOUGH_SEED_MIN, HOUGH_SEED_MAX, allow_zero=True)

    def ranges_by_parameter(self) -> dict[str, InclusiveIntegerRange]:
        """Return active range objects keyed by canonical parameter name."""
        return {
            PARAM_HOUGH_THRESHOLD: self.threshold,
            PARAM_HOUGH_LINE_LENGTH: self.line_length,
            PARAM_HOUGH_LINE_GAP: self.line_gap,
            PARAM_HOUGH_SEED: self.seed,
        }

    def values_by_parameter(self) -> dict[str, list[int]]:
        """Return active sweep values keyed by canonical parameter name."""
        return {name: active_range.values() for name, active_range in self.ranges_by_parameter().items()}

    def as_summary_dict(self) -> dict[str, dict[str, int]]:
        """Return active ranges in the summary JSON schema."""
        return {name: active_range.as_dict() for name, active_range in self.ranges_by_parameter().items()}

    def active_grid_label(self) -> str:
        """Return a compact active-grid label for timestamped logs."""
        return (
            f"threshold:{self.threshold.label()},"
            f"line_length:{self.line_length.label()},"
            f"line_gap:{self.line_gap.label()},"
            f"seed:{self.seed.label()}"
        )


def default_hough_sweep_ranges() -> HoughSweepRanges:
    """Return a fresh default range object for the full exhaustive grid."""
    return HoughSweepRanges()


def _range_from_optional_pair(
    *,
    pair: tuple[int, int] | list[int] | None,
    default_range: InclusiveIntegerRange,
    allow_zero: bool,
    parameter_label: str,
) -> InclusiveIntegerRange:
    """Resolve one optional CLI range pair against its default range."""
    if pair is None:
        return default_range
    if len(pair) != 2:
        raise ValueError(f"{parameter_label} range requires exactly two integers: start end")
    return InclusiveIntegerRange(int(pair[0]), int(pair[1]), allow_zero=bool(allow_zero))


def build_hough_sweep_ranges(
    *,
    threshold_range: tuple[int, int] | list[int] | None = None,
    line_length_range: tuple[int, int] | list[int] | None = None,
    line_gap_range: tuple[int, int] | list[int] | None = None,
    seed_range: tuple[int, int] | list[int] | None = None,
) -> HoughSweepRanges:
    """Build active Hough ranges from optional inclusive range overrides."""
    defaults = default_hough_sweep_ranges()
    fixed_seed_range = InclusiveIntegerRange(FIXED_HOUGH_SEED, FIXED_HOUGH_SEED, allow_zero=False)
    return HoughSweepRanges(
        threshold=_range_from_optional_pair(
            pair=threshold_range,
            default_range=defaults.threshold,
            allow_zero=False,
            parameter_label=PARAM_HOUGH_THRESHOLD,
        ),
        line_length=_range_from_optional_pair(
            pair=line_length_range,
            default_range=defaults.line_length,
            allow_zero=False,
            parameter_label=PARAM_HOUGH_LINE_LENGTH,
        ),
        line_gap=_range_from_optional_pair(
            pair=line_gap_range,
            default_range=defaults.line_gap,
            allow_zero=True,
            parameter_label=PARAM_HOUGH_LINE_GAP,
        ),
        # Seed sweep is temporarily disabled so each Hough combination uses the
        # same deterministic grid seed.  The old range resolver is kept below as
        # a commented restoration point because the CLI still accepts seed flags.
        # seed=_range_from_optional_pair(
        #     pair=seed_range,
        #     default_range=defaults.seed,
        #     allow_zero=True,
        #     parameter_label=PARAM_HOUGH_SEED,
        # ),
        seed=fixed_seed_range,
    )


@dataclass(frozen=True)
class HoughBaselineConfig:
    """Base configuration for non-grid parameters and fallback values."""

    hough_threshold: int = 26
    hough_line_length: int = 10
    hough_line_gap: int = 15
    hough_seed: int = FIXED_HOUGH_SEED
    hough_start: float = 2.6
    align_abs_min_len: float = 8.0
    align_min_iou_threshold: float = DEFAULT_MIN_IOU_THRESHOLD


@dataclass
class SweepDocument:
    """Prepared document payload reused across all Hough combinations.

    The v2.12 coverage metric requires both reference-to-prediction and
    reference-to-reference matrix directions.  The matrix and Hough context for
    each direction are prepared once per document and reused by all threshold,
    line-length, line-gap, and seed combinations.
    """

    index: int
    fname: str
    pred: str
    ref: str
    window_size: int
    window_stride: int
    ref_to_pred_matrix: np.ndarray
    ref_to_ref_matrix: np.ndarray
    whole_document_nls: float
    pred_blocks: list[str]
    ref_blocks: list[str]
    ref_to_pred_hough_ctx: dict
    ref_to_ref_hough_ctx: dict

    @property
    def matrix(self) -> np.ndarray:
        """Compatibility alias for the reference-to-prediction score matrix."""
        return self.ref_to_pred_matrix

    @property
    def hough_ctx(self) -> dict:
        """Compatibility alias for the reference-to-prediction Hough context."""
        return self.ref_to_pred_hough_ctx


def fixed_parameter_ranges() -> dict[str, tuple[int, int]]:
    """Return the default inclusive ranges for every swept parameter."""
    return {
        PARAM_HOUGH_THRESHOLD: (HOUGH_THRESHOLD_MIN, HOUGH_THRESHOLD_MAX),
        PARAM_HOUGH_LINE_LENGTH: (HOUGH_LINE_LENGTH_MIN, HOUGH_LINE_LENGTH_MAX),
        PARAM_HOUGH_LINE_GAP: (HOUGH_LINE_GAP_MIN, HOUGH_LINE_GAP_MAX),
        PARAM_HOUGH_SEED: (HOUGH_SEED_MIN, HOUGH_SEED_MAX),
    }


def fixed_values_for(param: str) -> list[int]:
    """Expand the default inclusive range for one parameter into values."""
    ranges = fixed_parameter_ranges()
    if param not in ranges:
        raise KeyError(f"Unsupported parameter {param!r}")
    lo, hi = ranges[param]
    return list(range(int(lo), int(hi) + 1))


__all__ = [
    "DEFAULT_MIN_IOU_THRESHOLD",
    "DEFAULT_SCORE_INDEX_CACHE_DIR",
    "DEFAULT_REF_TO_REF_COMBO_CACHE_DIR",
    "DEFAULT_TEXT_METRICS_V212_DIR",
    "FIXED_HOUGH_SEED",
    "HOUGH_THRESHOLD_MIN",
    "HOUGH_THRESHOLD_MAX",
    "HOUGH_LINE_LENGTH_MIN",
    "HOUGH_LINE_LENGTH_MAX",
    "HOUGH_LINE_GAP_MIN",
    "HOUGH_LINE_GAP_MAX",
    "HOUGH_SEED_MIN",
    "HOUGH_SEED_MAX",
    "HoughBaselineConfig",
    "HoughSweepRanges",
    "InclusiveIntegerRange",
    "LogFn",
    "MATRIX_CACHE_VERSION",
    "PARAM_HOUGH_THRESHOLD",
    "PARAM_HOUGH_LINE_LENGTH",
    "PARAM_HOUGH_LINE_GAP",
    "PARAM_HOUGH_SEED",
    "SUPPORTED_SWEEP_PARAMETERS",
    "SweepDocument",
    "build_hough_sweep_ranges",
    "default_hough_sweep_ranges",
    "fixed_parameter_ranges",
    "fixed_values_for",
]
