from __future__ import annotations

"""CSV export helpers for invalid Hough combinations.

The tuner normally keeps only compact best rows because a full exhaustive grid
can contain millions of combinations per document.  Invalid combinations are
rare but important: if v2.12 coverage rejects a combination, this module writes a
small diagnostic row so the run can continue without hiding the parameter set
that produced the invalid coverage state.
"""

from csv import DictWriter
import json
from pathlib import Path


INVALID_COMBINATION_FIELDNAMES = [
    "doc_index",
    "fname",
    "hough_threshold",
    "hough_line_length",
    "hough_line_gap",
    "hough_seed",
    "invalid_reason",
    "invalid_error_message",
    "coverage_y_diff_size",
    "coverage_y_diff_min",
    "coverage_y_diff_max",
    "coverage_y_diff_le_minus_one_count",
    "coverage_y_diff_lt_minus_one_count",
    "coverage_y_diff_below_minus_one_counts_json",
    "line_guided_columns",
    "fallback_columns",
    "used_line_count",
    "used_line_count_ref_to_ref",
    "raw_line_count",
    "raw_line_count_ref_to_ref",
    "candidate_line_count",
    "candidate_line_count_ref_to_ref",
    "timing_hough_detect_ref_to_pred_seconds",
    "timing_filter_ref_to_pred_seconds",
    "timing_hough_detect_ref_to_ref_seconds",
    "timing_filter_ref_to_ref_seconds",
    "timing_build_bundle_seconds",
    "timing_coverage_seconds",
    "timing_levenshtein_seconds",
    "timing_total_seconds",
]


def _csv_value(value):
    """Return a stable scalar representation for invalid-combination CSV cells."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def write_invalid_combinations_csv(*, rows: list[dict], output_csv: Path) -> Path:
    """Write every captured invalid Hough combination into one CSV file."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=INVALID_COMBINATION_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in INVALID_COMBINATION_FIELDNAMES})
    return output_csv


__all__ = ["INVALID_COMBINATION_FIELDNAMES", "write_invalid_combinations_csv"]
