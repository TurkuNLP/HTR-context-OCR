"""Coverage-array subtraction helpers for the report pipeline.

This module is intentionally reduced to the exact functionality used by
``run_text_metrics_report.sh``. It receives prebuilt line bundles, converts them
into per-character coverage arrays once, and computes the percentage metrics
from those arrays without any standalone CLI surface.
"""

from __future__ import annotations

import numpy as np

from line_metric_bundle import accumulate_counts_from_interval_groups

__all__ = [
    "build_line_coverage_arrays_from_bundles",
    "compute_line_coverage_percentage_metrics_from_arrays",
]



def _compute_y_axis_percentage_metrics(y_diff: np.ndarray) -> dict:
    """Compute missing/ok/repetition categories from y-axis subtraction.

    Category semantics are intentionally unchanged:
    - missing: ``y_diff == -1``
    - ok: ``y_diff == 0``
    - repetition: ``y_diff > 0``

    Any other value still raises, preserving the strict behavior the current
    pipeline relies on for debugging data issues.
    """
    total_chars = int(y_diff.size)
    if total_chars == 0:
        return {
            "missing_percent": 0.0,
            "ok_percent": 0.0,
            "repetition_percent": 0.0,
        }

    missing_count = int(np.count_nonzero(y_diff == -1))
    ok_count = int(np.count_nonzero(y_diff == 0))
    repetition_count = int(np.count_nonzero(y_diff > 0))

    covered_count = missing_count + ok_count + repetition_count
    if covered_count != total_chars:
        unknown_count = total_chars - covered_count
        raise ValueError(
            "Found y-axis subtraction values outside defined categories "
            "(-1, 0, >0). "
            f"unknown_count={unknown_count}"
        )

    return {
        "missing_percent": float((missing_count / total_chars) * 100.0),
        "ok_percent": float((ok_count / total_chars) * 100.0),
        "repetition_percent": float((repetition_count / total_chars) * 100.0),
    }



def _compute_x_axis_hallucination_percent(other_x: np.ndarray) -> float:
    """Compute hallucination percentage from prediction-axis zero coverage."""
    total_chars = int(other_x.size)
    if total_chars == 0:
        return 0.0
    hallucination_count = int(np.count_nonzero(other_x == 0))
    return float((hallucination_count / total_chars) * 100.0)



def _interval_groups_from_bundle(bundle: dict, key: str) -> list[list[tuple[int, int]]]:
    """Extract one interval-group field from every line entry in a bundle."""
    groups: list[list[tuple[int, int]]] = []
    for line in bundle.get("lines", []):
        intervals = line.get(key, [])
        groups.append([(int(start), int(end)) for start, end in intervals])
    return groups



def build_line_coverage_arrays_from_bundles(
    *,
    refref_bundle: dict,
    other_bundle: dict,
) -> dict[str, np.ndarray]:
    """Build coverage arrays once and reuse them for metrics and visuals.

    Returns four arrays:
    - ``refref_y``: reference-axis coverage counts for ref->ref lines
    - ``other_y``: reference-axis coverage counts for ref->pred lines
    - ``other_x``: prediction-axis coverage counts for ref->pred lines
    - ``y_diff``: ``other_y - refref_y`` with the current strict semantics
    """
    ref_text_len = int(refref_bundle.get("ref_text_len", 0))
    other_text_len = int(other_bundle.get("other_text_len", 0))

    refref_y = accumulate_counts_from_interval_groups(
        text_len=ref_text_len,
        interval_groups=_interval_groups_from_bundle(refref_bundle, "y_char_intervals_coverage_legacy"),
    )
    other_y = accumulate_counts_from_interval_groups(
        text_len=ref_text_len,
        interval_groups=_interval_groups_from_bundle(other_bundle, "y_char_intervals_coverage_legacy"),
    )
    other_x = accumulate_counts_from_interval_groups(
        text_len=other_text_len,
        interval_groups=_interval_groups_from_bundle(other_bundle, "x_char_intervals_coverage_legacy"),
    )

    if refref_y.shape != other_y.shape:
        raise ValueError(
            f"Reference-axis counts must have same length, got {refref_y.shape[0]} and {other_y.shape[0]}"
        )

    y_diff = np.subtract(other_y, refref_y)
    return {
        "refref_y": np.asarray(refref_y, dtype=np.int32),
        "other_y": np.asarray(other_y, dtype=np.int32),
        "other_x": np.asarray(other_x, dtype=np.int32),
        "y_diff": np.asarray(y_diff, dtype=np.int32),
    }



def compute_line_coverage_percentage_metrics_from_arrays(
    *,
    y_diff: np.ndarray,
    other_x: np.ndarray,
    file_name: str | None = None,
) -> dict:
    """Compute percentage metrics from precomputed coverage arrays."""
    y_diff = np.asarray(y_diff, dtype=np.int32)
    other_x = np.asarray(other_x, dtype=np.int32)

    metrics = _compute_y_axis_percentage_metrics(y_diff)
    metrics["hallucination_percent"] = _compute_x_axis_hallucination_percent(other_x)
    if file_name is not None:
        metrics["file_name"] = str(file_name)
    return metrics
