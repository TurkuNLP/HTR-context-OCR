from __future__ import annotations

"""Compact profiling exports for optional tuner diagnostics."""

import csv
from pathlib import Path


def _stable_profile_fieldnames(rows: list[dict]) -> list[str]:
    """Return deterministic CSV columns with identity fields first."""
    preferred_prefix = [
        "doc_index",
        "fname",
        "hough_threshold",
        "hough_line_length",
        "hough_line_gap",
        "hough_seed",
        "matrix_rows_ref_to_pred",
        "matrix_cols_ref_to_pred",
        "matrix_rows_ref_to_ref",
        "matrix_cols_ref_to_ref",
        "is_valid",
        "invalid_reason",
        "invalid_error_message",
        "metric_outcome_reason",
        "tuning_score",
        "weighted_along_lines_nls",
        "correct_ref_coverage",
        "missing_ref_coverage",
        "repetition_on_ref",
        "hallucination",
    ]
    all_fieldnames = sorted({str(field_name) for row in rows for field_name in row})
    prefix = [field_name for field_name in preferred_prefix if field_name in all_fieldnames]
    suffix = [field_name for field_name in all_fieldnames if field_name not in set(prefix)]
    return prefix + suffix


def write_combination_profile_csv(*, rows: list[dict], output_csv: Path) -> Path:
    """Write optional per-combination profiling rows as a scalar CSV file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        output_csv.write_text("", encoding="utf-8")
        return output_csv

    fieldnames = _stable_profile_fieldnames(rows)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return output_csv


__all__ = ["write_combination_profile_csv"]
