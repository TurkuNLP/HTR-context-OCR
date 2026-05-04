from __future__ import annotations

"""Aggregation helpers for parameter influence curves and document best points."""

import math

import numpy as np

try:
    from .hough_eval import is_finite_along_lines
    from .tuner_config import (
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )
except ImportError:
    from hough_eval import is_finite_along_lines  # type: ignore
    from tuner_config import (  # type: ignore
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )


def aggregate_along_lines(rows: list[dict]) -> dict:
    """Aggregate along-lines NLS statistics across a list of rows."""
    vals = [
        float(row["along_lines_nls"])
        for row in rows
        if row.get("along_lines_nls") is not None and math.isfinite(float(row["along_lines_nls"]))
    ]
    if not vals:
        return {
            "mean_along_lines_nls": None,
            "median_along_lines_nls": None,
            "std_along_lines_nls": None,
            "min_along_lines_nls": None,
            "max_along_lines_nls": None,
            "valid_doc_count": 0,
        }

    arr = np.asarray(vals, dtype=float)
    return {
        "mean_along_lines_nls": float(np.mean(arr)),
        "median_along_lines_nls": float(np.median(arr)),
        "std_along_lines_nls": float(np.std(arr, ddof=0)),
        "min_along_lines_nls": float(np.min(arr)),
        "max_along_lines_nls": float(np.max(arr)),
        "valid_doc_count": int(arr.size),
    }


def best_curve_row(rows: list[dict]) -> dict | None:
    """Pick best curve point by mean along-lines NLS."""
    candidates = [row for row in rows if row.get("mean_along_lines_nls") is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["mean_along_lines_nls"]),
            int(row.get("valid_doc_count", 0)),
            -int(row.get("value", 0)),
        ),
    )


def compact_best_curve_row(best: dict | None) -> dict | None:
    """Return compact summary payload for one curve best point."""
    if best is None:
        return None
    return {
        "value": int(best["value"]),
        "mean_along_lines_nls": best.get("mean_along_lines_nls"),
        "median_along_lines_nls": best.get("median_along_lines_nls"),
        "std_along_lines_nls": best.get("std_along_lines_nls"),
        "valid_doc_count": int(best.get("valid_doc_count", 0)),
        "doc_count": int(best.get("doc_count", 0)),
    }


def build_curve_row(*, parameter: str, value: int, doc_rows: list[dict]) -> dict:
    """Build one aggregated curve row for a parameter value."""
    agg = aggregate_along_lines(doc_rows)
    timing_hough_detect_seconds = float(
        sum(float(doc_row.get("timing_hough_detect_seconds", 0.0)) for doc_row in doc_rows)
    )
    timing_filter_seconds = float(sum(float(doc_row.get("timing_filter_seconds", 0.0)) for doc_row in doc_rows))
    timing_detect_filter_seconds = float(
        sum(float(doc_row.get("timing_detect_filter_seconds", 0.0)) for doc_row in doc_rows)
    )
    timing_build_bundle_seconds = float(
        sum(float(doc_row.get("timing_build_bundle_seconds", 0.0)) for doc_row in doc_rows)
    )
    timing_levenshtein_seconds = float(
        sum(float(doc_row.get("timing_levenshtein_seconds", 0.0)) for doc_row in doc_rows)
    )
    timing_total_seconds = float(sum(float(doc_row.get("timing_total_seconds", 0.0)) for doc_row in doc_rows))
    per_doc_count = max(1, int(len(doc_rows)))

    return {
        "parameter": str(parameter),
        "value": int(value),
        "doc_count": int(len(doc_rows)),
        **agg,
        "timing_hough_detect_seconds": timing_hough_detect_seconds,
        "timing_filter_seconds": timing_filter_seconds,
        "timing_detect_filter_seconds": timing_detect_filter_seconds,
        "timing_build_bundle_seconds": timing_build_bundle_seconds,
        "timing_levenshtein_seconds": timing_levenshtein_seconds,
        "timing_total_seconds": timing_total_seconds,
        "timing_hough_detect_per_doc_seconds": float(timing_hough_detect_seconds / per_doc_count),
        "timing_filter_per_doc_seconds": float(timing_filter_seconds / per_doc_count),
        "timing_detect_filter_per_doc_seconds": float(timing_detect_filter_seconds / per_doc_count),
        "timing_build_bundle_per_doc_seconds": float(timing_build_bundle_seconds / per_doc_count),
        "timing_levenshtein_per_doc_seconds": float(timing_levenshtein_seconds / per_doc_count),
        "timing_total_per_doc_seconds": float(timing_total_seconds / per_doc_count),
        "docs": doc_rows,
    }


def best_doc_nls_mean(best_records: list[dict]) -> float | None:
    """Mean of per-document best along-lines NLS values."""
    vals = [
        float(rec["best"]["along_lines_nls"])
        for rec in best_records
        if rec.get("best") is not None and is_finite_along_lines(rec["best"].get("along_lines_nls"))
    ]
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def point_from_best(*, doc: SweepDocument, best_eval: dict | None, baseline_cfg: HoughBaselineConfig) -> dict:
    """Convert one best-row payload into stable profile-point structure."""
    if best_eval is None or not is_finite_along_lines(best_eval.get("along_lines_nls")):
        return {
            "index": int(doc.index),
            "fname": str(doc.fname),
            "along_lines_nls": None,
            "whole_document_nls": float(doc.whole_document_nls),
            "best_config": None,
        }

    return {
        "index": int(doc.index),
        "fname": str(doc.fname),
        "along_lines_nls": float(best_eval["along_lines_nls"]),
        "whole_document_nls": float(doc.whole_document_nls),
        "line_guided_columns": int(best_eval.get("line_guided_columns", 0)),
        "fallback_columns": int(best_eval.get("fallback_columns", 0)),
        "used_line_count": int(best_eval.get("used_line_count", 0)),
        "timing_hough_detect_seconds": float(best_eval.get("timing_hough_detect_seconds", 0.0)),
        "timing_filter_seconds": float(best_eval.get("timing_filter_seconds", 0.0)),
        "timing_detect_filter_seconds": float(best_eval.get("timing_detect_filter_seconds", 0.0)),
        "timing_build_bundle_seconds": float(best_eval.get("timing_build_bundle_seconds", 0.0)),
        "timing_levenshtein_seconds": float(best_eval.get("timing_levenshtein_seconds", 0.0)),
        "timing_total_seconds": float(best_eval.get("timing_total_seconds", 0.0)),
        "best_config": {
            PARAM_HOUGH_THRESHOLD: int(best_eval.get(PARAM_HOUGH_THRESHOLD, baseline_cfg.hough_threshold)),
            PARAM_HOUGH_LINE_LENGTH: int(best_eval.get(PARAM_HOUGH_LINE_LENGTH, baseline_cfg.hough_line_length)),
            PARAM_HOUGH_LINE_GAP: int(best_eval.get(PARAM_HOUGH_LINE_GAP, baseline_cfg.hough_line_gap)),
            PARAM_HOUGH_SEED: int(best_eval.get(PARAM_HOUGH_SEED, baseline_cfg.hough_seed)),
        },
    }


__all__ = [
    "aggregate_along_lines",
    "best_curve_row",
    "compact_best_curve_row",
    "build_curve_row",
    "best_doc_nls_mean",
    "point_from_best",
]
