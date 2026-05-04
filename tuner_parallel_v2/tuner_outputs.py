from __future__ import annotations

from pathlib import Path


def format_csv_float(value) -> str:
    """Format numeric values for CSV while keeping missing values empty."""
    if value is None:
        return ""
    return f"{float(value):.10f}"


def write_parameter_curve_csv(*, rows: list[dict], output_csv: Path) -> None:
    """Write one CSV file for one parameter curve (x=value, y=mean along-lines NLS)."""
    header = [
        "value",
        "mean_along_lines_nls",
        "median_along_lines_nls",
        "std_along_lines_nls",
        "min_along_lines_nls",
        "max_along_lines_nls",
        "valid_doc_count",
        "doc_count",
        "timing_detect_filter_seconds",
        "timing_build_bundle_seconds",
        "timing_levenshtein_seconds",
        "timing_total_seconds",
    ]
    lines = [",".join(header)]

    for row in rows:
        lines.append(
            ",".join(
                [
                    str(int(row.get("value", 0))),
                    format_csv_float(row.get("mean_along_lines_nls")),
                    format_csv_float(row.get("median_along_lines_nls")),
                    format_csv_float(row.get("std_along_lines_nls")),
                    format_csv_float(row.get("min_along_lines_nls")),
                    format_csv_float(row.get("max_along_lines_nls")),
                    str(int(row.get("valid_doc_count", 0))),
                    str(int(row.get("doc_count", 0))),
                    format_csv_float(row.get("timing_detect_filter_seconds")),
                    format_csv_float(row.get("timing_build_bundle_seconds")),
                    format_csv_float(row.get("timing_levenshtein_seconds")),
                    format_csv_float(row.get("timing_total_seconds")),
                ]
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_best_configs_csv(*, best_records: list[dict], output_csv: Path) -> None:
    """Write best parameter combination per document into a single CSV."""
    header = [
        "index",
        "fname",
        "whole_document_nls",
        "best_along_lines_nls",
        "delta_vs_whole_nls",
        "hough_threshold",
        "hough_line_length",
        "hough_line_gap",
        "hough_seed",
        "line_guided_columns",
        "fallback_columns",
        "used_line_count",
        "timing_total_seconds",
        "evaluated_combination_count",
        "doc_grid_seconds",
    ]
    lines = [",".join(header)]

    for rec in best_records:
        best = rec.get("best", {})
        whole = rec.get("whole_document_nls")
        along = best.get("along_lines_nls")
        delta = None
        if whole is not None and along is not None:
            try:
                delta = float(along) - float(whole)
            except Exception:
                delta = None

        lines.append(
            ",".join(
                [
                    str(int(rec.get("index", 0))),
                    str(rec.get("fname", "")),
                    format_csv_float(whole),
                    format_csv_float(along),
                    format_csv_float(delta),
                    str(int(best.get("hough_threshold", 0))),
                    str(int(best.get("hough_line_length", 0))),
                    str(int(best.get("hough_line_gap", 0))),
                    str(int(best.get("hough_seed", 0))),
                    str(int(best.get("line_guided_columns", 0))),
                    str(int(best.get("fallback_columns", 0))),
                    str(int(best.get("used_line_count", 0))),
                    format_csv_float(best.get("timing_total_seconds")),
                    str(int(rec.get("evaluated_combination_count", 0))),
                    format_csv_float(rec.get("doc_grid_seconds")),
                ]
            )
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

