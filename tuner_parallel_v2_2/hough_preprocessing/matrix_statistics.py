from __future__ import annotations

"""Score-matrix statistics used by Region of Interest preprocessing."""

from dataclasses import asdict, dataclass

import numpy as np

from .config import (
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY,
    MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY,
)


NORMAL_DISTRIBUTION_MEDIAN_ABSOLUTE_DEVIATION_SCALE = 1.482602218505602


@dataclass(frozen=True)
class ScoreMatrixStatistics:
    """Finite-value summary for one score matrix."""

    finite_value_count: int
    score_minimum: float
    score_mean: float
    score_median: float
    score_maximum: float
    score_standard_deviation: float
    scaled_median_absolute_deviation: float
    median_absolute_deviation_backend: str

    @property
    def has_finite_scores(self) -> bool:
        """Return True when at least one usable score was found."""
        return int(self.finite_value_count) > 0

    def as_dict(self) -> dict:
        """Return a JSON-friendly statistics dictionary."""
        return asdict(self)


def finite_score_values(score_matrix: np.ndarray) -> np.ndarray:
    """Return the finite cells from a score matrix as one float array."""
    matrix = np.asarray(score_matrix, dtype=float)
    if matrix.size == 0:
        return np.asarray([], dtype=float)
    return matrix[np.isfinite(matrix)]


def scaled_median_absolute_deviation(
    values: np.ndarray,
    *,
    median_value: float,
    backend: str,
) -> float:
    """Return Median Absolute Deviation on a normal-distribution scale."""
    finite_values = np.asarray(values, dtype=float)
    if finite_values.size == 0:
        return 0.0

    if backend == MEDIAN_ABSOLUTE_DEVIATION_BACKEND_SCIPY:
        try:
            from scipy.stats import median_abs_deviation
        except ImportError as exc:
            raise RuntimeError(
                "SciPy Median Absolute Deviation was requested, but scipy.stats is not available."
            ) from exc
        return float(median_abs_deviation(finite_values, scale="normal"))

    if backend != MEDIAN_ABSOLUTE_DEVIATION_BACKEND_MANUAL_NUMPY:
        raise ValueError(f"Unsupported Median Absolute Deviation backend: {backend!r}")

    absolute_deviation_from_median = np.abs(finite_values - float(median_value))
    raw_median_absolute_deviation = float(np.median(absolute_deviation_from_median))
    return float(raw_median_absolute_deviation * NORMAL_DISTRIBUTION_MEDIAN_ABSOLUTE_DEVIATION_SCALE)


def summarize_score_matrix(
    score_matrix: np.ndarray,
    *,
    median_absolute_deviation_backend: str,
) -> ScoreMatrixStatistics:
    """Compute finite-score statistics for one matrix."""
    finite_values = finite_score_values(score_matrix)
    if finite_values.size == 0:
        return ScoreMatrixStatistics(
            finite_value_count=0,
            score_minimum=float("nan"),
            score_mean=float("nan"),
            score_median=float("nan"),
            score_maximum=float("nan"),
            score_standard_deviation=0.0,
            scaled_median_absolute_deviation=0.0,
            median_absolute_deviation_backend=str(median_absolute_deviation_backend),
        )

    median_value = float(np.median(finite_values))
    return ScoreMatrixStatistics(
        finite_value_count=int(finite_values.size),
        score_minimum=float(np.min(finite_values)),
        score_mean=float(np.mean(finite_values)),
        score_median=median_value,
        score_maximum=float(np.max(finite_values)),
        score_standard_deviation=float(np.std(finite_values, ddof=0)),
        scaled_median_absolute_deviation=scaled_median_absolute_deviation(
            finite_values,
            median_value=median_value,
            backend=str(median_absolute_deviation_backend),
        ),
        median_absolute_deviation_backend=str(median_absolute_deviation_backend),
    )
