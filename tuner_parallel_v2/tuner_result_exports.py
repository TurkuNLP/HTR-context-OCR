from __future__ import annotations

"""Export helpers for human-readable tuning results and plotting-friendly long CSVs.

This module intentionally contains only serialization/flattening logic so the
sweep engine stays focused on evaluation.
"""

from csv import DictWriter
import json
import math
from pathlib import Path

try:
    from .tuner_config import (
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
    )
except ImportError:
    from tuner_config import (  # type: ignore
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
    )


def _finite_float_or_none(value) -> float | None:
    """Return finite float value or ``None`` when missing/non-finite."""
    if value is None:
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _int_or_default(value, default: int = 0) -> int:
    """Best-effort integer conversion with a stable default fallback."""
    try:
        return int(value)
    except Exception:
        return int(default)


def build_best_params_records(*, best_records: list[dict]) -> list[dict]:
    """Build compact per-document best-parameter records for human inspection."""
    rows: list[dict] = []
    for rec in best_records:
        best = rec.get("best") if isinstance(rec, dict) else None
        if not isinstance(best, dict):
            best = {}

        rows.append(
            {
                "index": _int_or_default(rec.get("index", 0) if isinstance(rec, dict) else 0),
                "fname": str(rec.get("fname", "") if isinstance(rec, dict) else ""),
                "hough_threshold": _int_or_default(best.get(PARAM_HOUGH_THRESHOLD, 0)),
                "hough_line_length": _int_or_default(best.get(PARAM_HOUGH_LINE_LENGTH, 0)),
                "hough_line_gap": _int_or_default(best.get(PARAM_HOUGH_LINE_GAP, 0)),
                "hough_seed": _int_or_default(best.get(PARAM_HOUGH_SEED, 0)),
                "best_along_lines_nls": _finite_float_or_none(best.get("along_lines_nls")),
                "whole_document_nls": _finite_float_or_none(
                    rec.get("whole_document_nls") if isinstance(rec, dict) else None
                ),
            }
        )

    rows.sort(key=lambda row: (int(row["index"]), str(row["fname"])))
    return rows


def write_best_params_json(*, best_records: list[dict], output_json: Path) -> Path:
    """Write compact best-parameter records into one readable JSON file."""
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    records = build_best_params_records(best_records=best_records)
    payload = {
        "doc_count": int(len(records)),
        "records": records,
    }
    output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_json


def build_parameter_influence_rows(*, profile_points: dict[str, dict[int, list[dict]]]) -> list[dict]:
    """Flatten profile points into one long table for all docs and parameters.

    Output rows are one row per (document, parameter, parameter-value), which is
    the ideal format for plotting and downstream slicing.
    """
    rows: list[dict] = []

    for parameter in SUPPORTED_SWEEP_PARAMETERS:
        values_map = profile_points.get(str(parameter), {})
        if not isinstance(values_map, dict):
            continue

        for value in sorted(int(v) for v in values_map.keys()):
            doc_rows = values_map.get(int(value), [])
            if not isinstance(doc_rows, list):
                continue

            for point in doc_rows:
                if not isinstance(point, dict):
                    continue

                along = _finite_float_or_none(point.get("along_lines_nls"))
                whole = _finite_float_or_none(point.get("whole_document_nls"))
                delta = None
                if along is not None and whole is not None:
                    delta = float(along - whole)

                best_cfg = point.get("best_config") if isinstance(point.get("best_config"), dict) else {}

                rows.append(
                    {
                        "doc_index": _int_or_default(point.get("index", 0)),
                        "fname": str(point.get("fname", "")),
                        "parameter": str(parameter),
                        "value": int(value),
                        "along_lines_nls": along,
                        "whole_document_nls": whole,
                        "delta_vs_whole_nls": delta,
                        "line_guided_columns": _int_or_default(point.get("line_guided_columns", 0)),
                        "fallback_columns": _int_or_default(point.get("fallback_columns", 0)),
                        "used_line_count": _int_or_default(point.get("used_line_count", 0)),
                        "timing_hough_detect_seconds": _finite_float_or_none(
                            point.get("timing_hough_detect_seconds")
                        ),
                        "timing_filter_seconds": _finite_float_or_none(point.get("timing_filter_seconds")),
                        "timing_detect_filter_seconds": _finite_float_or_none(
                            point.get("timing_detect_filter_seconds")
                        ),
                        "timing_levenshtein_seconds": _finite_float_or_none(
                            point.get("timing_levenshtein_seconds")
                        ),
                        "timing_total_seconds": _finite_float_or_none(point.get("timing_total_seconds")),
                        "selected_hough_threshold": _int_or_default(best_cfg.get(PARAM_HOUGH_THRESHOLD, 0)),
                        "selected_hough_line_length": _int_or_default(best_cfg.get(PARAM_HOUGH_LINE_LENGTH, 0)),
                        "selected_hough_line_gap": _int_or_default(best_cfg.get(PARAM_HOUGH_LINE_GAP, 0)),
                        "selected_hough_seed": _int_or_default(best_cfg.get(PARAM_HOUGH_SEED, 0)),
                        "is_valid": int(along is not None),
                    }
                )

    rows.sort(key=lambda row: (str(row["parameter"]), int(row["value"]), int(row["doc_index"]), str(row["fname"])))
    return rows


def write_parameter_influence_csv(*, rows: list[dict], output_csv: Path) -> Path:
    """Write flattened parameter-influence rows into one long-format CSV file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "doc_index",
        "fname",
        "parameter",
        "value",
        "along_lines_nls",
        "whole_document_nls",
        "delta_vs_whole_nls",
        "line_guided_columns",
        "fallback_columns",
        "used_line_count",
        "timing_hough_detect_seconds",
        "timing_filter_seconds",
        "timing_detect_filter_seconds",
        "timing_levenshtein_seconds",
        "timing_total_seconds",
        "selected_hough_threshold",
        "selected_hough_line_length",
        "selected_hough_line_gap",
        "selected_hough_seed",
        "is_valid",
    ]

    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})

    return output_csv


__all__ = [
    "build_best_params_records",
    "write_best_params_json",
    "build_parameter_influence_rows",
    "write_parameter_influence_csv",
]
