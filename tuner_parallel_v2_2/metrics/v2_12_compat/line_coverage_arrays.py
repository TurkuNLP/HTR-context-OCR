from __future__ import annotations

"""Build v2.12 coverage arrays and ratio metrics from line bundles.

Source:
`text_metrics_v2_12_parallel/line_coverage_subtract.py`

Copied on: 2026-05-25.
"""

import numpy as np

from .line_metric_bundle import accumulate_counts_from_interval_groups


def _compute_y_axis_ratio_metrics(y_diff: np.ndarray) -> dict:
    """Compute missing/ok/repetition ratios from y-axis subtraction."""
    total_chars = int(y_diff.size)
    if total_chars == 0:
        return {
            "missing_ref_coverage": 0.0,
            "correct_ref_coverage": 0.0,
            "repetition_on_ref": 0.0,
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
        "missing_ref_coverage": float(missing_count / total_chars),
        "correct_ref_coverage": float(ok_count / total_chars),
        "repetition_on_ref": float(repetition_count / total_chars),
    }


def _compute_x_axis_hallucination_ratio(other_x: np.ndarray) -> float:
    """Compute hallucination ratio from prediction-axis zero coverage."""
    total_chars = int(other_x.size)
    if total_chars == 0:
        return 0.0
    hallucination_count = int(np.count_nonzero(other_x == 0))
    return float(hallucination_count / total_chars)


def _interval_groups_from_bundle(bundle: dict, key: str) -> list[list[tuple[int, int]]]:
    """Extract one interval-group field from every line entry in a bundle."""
    groups: list[list[tuple[int, int]]] = []
    for line in bundle.get("lines", []):
        intervals = line.get(key, [])
        groups.append([(int(start), int(end)) for start, end in intervals])
    return groups


def build_refref_y_coverage_array_from_bundle(refref_bundle: dict) -> np.ndarray:
    """Build the reference-self y-axis coverage array from a v2.12 bundle."""
    ref_text_len = int(refref_bundle.get("ref_text_len", 0))
    return accumulate_counts_from_interval_groups(
        text_len=int(ref_text_len),
        interval_groups=_interval_groups_from_bundle(refref_bundle, "y_char_intervals_for_coverage"),
    )


def build_other_line_coverage_arrays_from_bundle(
    *,
    other_bundle: dict,
    ref_text_len: int,
) -> dict[str, np.ndarray]:
    """Build other-side coverage arrays from a v2.12 ref-to-other bundle."""
    other_text_len = int(other_bundle.get("other_text_len", 0))
    other_y = accumulate_counts_from_interval_groups(
        text_len=int(ref_text_len),
        interval_groups=_interval_groups_from_bundle(other_bundle, "y_char_intervals_for_coverage"),
    )
    other_x = accumulate_counts_from_interval_groups(
        text_len=int(other_text_len),
        interval_groups=_interval_groups_from_bundle(other_bundle, "x_char_intervals_for_coverage"),
    )
    return {
        "other_y": np.asarray(other_y, dtype=np.int32),
        "other_x": np.asarray(other_x, dtype=np.int32),
    }


def build_line_coverage_arrays_from_cached_refref_y(
    *,
    refref_y: np.ndarray,
    other_bundle: dict,
) -> dict[str, np.ndarray]:
    """Build coverage arrays from cached ``refref_y`` and one other bundle."""
    refref_y = np.asarray(refref_y, dtype=np.int32)
    other_arrays = build_other_line_coverage_arrays_from_bundle(
        other_bundle=other_bundle,
        ref_text_len=int(refref_y.size),
    )
    other_y = np.asarray(other_arrays["other_y"], dtype=np.int32)
    other_x = np.asarray(other_arrays["other_x"], dtype=np.int32)
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


def build_line_coverage_arrays_from_bundles(
    *,
    refref_bundle: dict,
    other_bundle: dict,
) -> dict[str, np.ndarray]:
    """Build coverage arrays once and reuse them for metrics and visuals."""
    refref_y = build_refref_y_coverage_array_from_bundle(refref_bundle)
    return build_line_coverage_arrays_from_cached_refref_y(
        refref_y=refref_y,
        other_bundle=other_bundle,
    )


def compute_line_coverage_ratio_metrics_from_arrays(
    *,
    y_diff: np.ndarray,
    other_x: np.ndarray,
    file_name: str | None = None,
) -> dict:
    """Compute coverage and hallucination ratios from precomputed arrays."""
    y_diff = np.asarray(y_diff, dtype=np.int32)
    other_x = np.asarray(other_x, dtype=np.int32)

    metrics = _compute_y_axis_ratio_metrics(y_diff)
    metrics["hallucination"] = _compute_x_axis_hallucination_ratio(other_x)
    if file_name is not None:
        metrics["file_name"] = str(file_name)
    return metrics


def compute_line_coverage_percentage_metrics_from_arrays(
    *,
    y_diff: np.ndarray,
    other_x: np.ndarray,
    file_name: str | None = None,
) -> dict:
    """Compute percentage metrics from precomputed coverage arrays."""
    ratio_metrics = compute_line_coverage_ratio_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name=file_name,
    )
    return {
        "missing_percent": float(ratio_metrics["missing_ref_coverage"] * 100.0),
        "ok_percent": float(ratio_metrics["correct_ref_coverage"] * 100.0),
        "repetition_percent": float(ratio_metrics["repetition_on_ref"] * 100.0),
        "hallucination_percent": float(ratio_metrics["hallucination"] * 100.0),
        **({"file_name": str(file_name)} if file_name is not None else {}),
    }


__all__ = [
    "build_line_coverage_arrays_from_bundles",
    "build_line_coverage_arrays_from_cached_refref_y",
    "build_other_line_coverage_arrays_from_bundle",
    "build_refref_y_coverage_array_from_bundle",
    "compute_line_coverage_percentage_metrics_from_arrays",
    "compute_line_coverage_ratio_metrics_from_arrays",
]
