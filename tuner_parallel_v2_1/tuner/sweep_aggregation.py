from __future__ import annotations

"""Aggregation helpers for parameter influence curves and document best points."""

import math

import numpy as np

try:
    from .hough_eval import is_finite_tuning_score
    from .tuner_config import (
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )
except ImportError:
    from tuner.hough_eval import is_finite_tuning_score  # type: ignore
    from tuner.tuner_config import (  # type: ignore
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )


AGGREGATED_FLOAT_FIELDS: tuple[str, ...] = (
    "tuning_score",
    "weighted_along_lines_nls",
    "correct_ref_coverage",
    "missing_ref_coverage",
    "repetition_on_ref",
    "hallucination",
)

SUMMED_TIMING_FIELDS: tuple[str, ...] = (
    "timing_hough_detect_ref_to_pred_seconds",
    "timing_filter_ref_to_pred_seconds",
    "timing_hough_detect_ref_to_ref_seconds",
    "timing_filter_ref_to_ref_seconds",
    "timing_hough_detect_seconds",
    "timing_filter_seconds",
    "timing_detect_filter_seconds",
    "timing_build_bundle_seconds",
    "timing_coverage_seconds",
    "timing_levenshtein_seconds",
    "timing_total_seconds",
)


def _finite_values(rows: list[dict], field_name: str) -> list[float]:
    """Collect finite numeric values for one field across rows."""
    values: list[float] = []
    for row in rows:
        value = row.get(field_name)
        if value is None:
            continue
        try:
            numeric_value = float(value)
        except Exception:
            continue
        if math.isfinite(numeric_value):
            values.append(float(numeric_value))
    return values


def aggregate_metric(rows: list[dict], field_name: str) -> dict:
    """Aggregate mean/median/std/min/max for one numeric field."""
    values = _finite_values(rows, field_name)
    if not values:
        return {
            f"mean_{field_name}": None,
            f"median_{field_name}": None,
            f"std_{field_name}": None,
            f"min_{field_name}": None,
            f"max_{field_name}": None,
            f"valid_{field_name}_count": 0,
        }

    arr = np.asarray(values, dtype=float)
    return {
        f"mean_{field_name}": float(np.mean(arr)),
        f"median_{field_name}": float(np.median(arr)),
        f"std_{field_name}": float(np.std(arr, ddof=0)),
        f"min_{field_name}": float(np.min(arr)),
        f"max_{field_name}": float(np.max(arr)),
        f"valid_{field_name}_count": int(arr.size),
    }


def best_curve_row(rows: list[dict]) -> dict | None:
    """Pick the best curve point by mean tuning score."""
    candidates = [row for row in rows if row.get("mean_tuning_score") is not None]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda row: (
            float(row["mean_tuning_score"]),
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
        "mean_tuning_score": best.get("mean_tuning_score"),
        "median_tuning_score": best.get("median_tuning_score"),
        "std_tuning_score": best.get("std_tuning_score"),
        "mean_weighted_along_lines_nls": best.get("mean_weighted_along_lines_nls"),
        "mean_correct_ref_coverage": best.get("mean_correct_ref_coverage"),
        "mean_missing_ref_coverage": best.get("mean_missing_ref_coverage"),
        "mean_repetition_on_ref": best.get("mean_repetition_on_ref"),
        "mean_hallucination": best.get("mean_hallucination"),
        "valid_doc_count": int(best.get("valid_doc_count", 0)),
        "doc_count": int(best.get("doc_count", 0)),
    }


def build_curve_row(*, parameter: str, value: int, doc_rows: list[dict]) -> dict:
    """Build one aggregated curve row for a parameter value."""
    row: dict = {
        "parameter": str(parameter),
        "value": int(value),
        "doc_count": int(len(doc_rows)),
        "docs": doc_rows,
    }

    for field_name in AGGREGATED_FLOAT_FIELDS:
        row.update(aggregate_metric(doc_rows, field_name))

    row["valid_doc_count"] = int(row.get("valid_tuning_score_count", 0))

    per_doc_count = max(1, int(len(doc_rows)))
    for timing_field in SUMMED_TIMING_FIELDS:
        total = float(sum(float(doc_row.get(timing_field, 0.0)) for doc_row in doc_rows))
        row[timing_field] = total
        row[timing_field.replace("seconds", "per_doc_seconds")] = float(total / per_doc_count)

    return row


def best_doc_tuning_score_mean(best_records: list[dict]) -> float | None:
    """Mean of per-document best tuning-score values."""
    values = [
        float(rec["best"]["tuning_score"])
        for rec in best_records
        if rec.get("best") is not None and is_finite_tuning_score(rec["best"].get("tuning_score"))
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def best_doc_nls_mean(best_records: list[dict]) -> float | None:
    """Compatibility mean of per-document best weighted along-lines NLS values."""
    values = [
        float(rec["best"]["weighted_along_lines_nls"])
        for rec in best_records
        if rec.get("best") is not None and rec["best"].get("weighted_along_lines_nls") is not None
    ]
    if not values:
        return None
    return float(sum(values) / len(values))


def _copy_metric_fields(best_eval: dict) -> dict:
    """Copy scalar metric fields from one best evaluation row."""
    fields = [
        "tuning_score",
        "weighted_along_lines_nls",
        "correct_ref_coverage",
        "missing_ref_coverage",
        "repetition_on_ref",
        "hallucination",
        "total_line_length",
    ]
    return {field: best_eval.get(field) for field in fields}


def point_from_best(*, doc: SweepDocument, best_eval: dict | None, baseline_cfg: HoughBaselineConfig) -> dict:
    """Convert one best-row payload into stable profile-point structure."""
    if best_eval is None or not is_finite_tuning_score(best_eval.get("tuning_score")):
        return {
            "index": int(doc.index),
            "fname": str(doc.fname),
            "tuning_score": None,
            "weighted_along_lines_nls": None,
            "whole_document_nls": float(doc.whole_document_nls),
            "best_config": None,
        }

    point = {
        "index": int(doc.index),
        "fname": str(doc.fname),
        "whole_document_nls": float(doc.whole_document_nls),
        **_copy_metric_fields(best_eval),
        "line_guided_columns": int(best_eval.get("line_guided_columns", 0)),
        "fallback_columns": int(best_eval.get("fallback_columns", 0)),
        "used_line_count": int(best_eval.get("used_line_count", 0)),
        "used_line_count_ref_to_ref": int(best_eval.get("used_line_count_ref_to_ref", 0)),
        "raw_line_count": int(best_eval.get("raw_line_count", 0)),
        "raw_line_count_ref_to_ref": int(best_eval.get("raw_line_count_ref_to_ref", 0)),
        "candidate_line_count": int(best_eval.get("candidate_line_count", 0)),
        "candidate_line_count_ref_to_ref": int(best_eval.get("candidate_line_count_ref_to_ref", 0)),
        "best_config": {
            PARAM_HOUGH_THRESHOLD: int(best_eval.get(PARAM_HOUGH_THRESHOLD, baseline_cfg.hough_threshold)),
            PARAM_HOUGH_LINE_LENGTH: int(best_eval.get(PARAM_HOUGH_LINE_LENGTH, baseline_cfg.hough_line_length)),
            PARAM_HOUGH_LINE_GAP: int(best_eval.get(PARAM_HOUGH_LINE_GAP, baseline_cfg.hough_line_gap)),
            PARAM_HOUGH_SEED: int(best_eval.get(PARAM_HOUGH_SEED, baseline_cfg.hough_seed)),
        },
    }

    for timing_field in SUMMED_TIMING_FIELDS:
        point[timing_field] = float(best_eval.get(timing_field, 0.0))

    return point


__all__ = [
    "AGGREGATED_FLOAT_FIELDS",
    "SUMMED_TIMING_FIELDS",
    "aggregate_metric",
    "best_curve_row",
    "compact_best_curve_row",
    "build_curve_row",
    "best_doc_nls_mean",
    "best_doc_tuning_score_mean",
    "point_from_best",
]
