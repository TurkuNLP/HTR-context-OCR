from __future__ import annotations

"""CSV writers for aggregated tuner outputs."""

from csv import DictWriter
from pathlib import Path


PARAMETER_CURVE_FIELDNAMES = [
    "value",
    "mean_tuning_score",
    "median_tuning_score",
    "std_tuning_score",
    "min_tuning_score",
    "max_tuning_score",
    "mean_alignment_selection_score",
    "median_alignment_selection_score",
    "std_alignment_selection_score",
    "min_alignment_selection_score",
    "max_alignment_selection_score",
    "mean_score_matrix_support",
    "mean_line_guided_fraction",
    "mean_average_weighted_normalised_levenshtein_similarity",
    "mean_correct_ref_coverage",
    "mean_missing_ref_coverage",
    "mean_repetition_on_ref",
    "mean_hallucination",
    "valid_doc_count",
    "doc_count",
    "timing_hough_detect_ref_to_pred_seconds",
    "timing_filter_ref_to_pred_seconds",
    "timing_hough_detect_ref_to_ref_seconds",
    "timing_filter_ref_to_ref_seconds",
    "timing_build_bundle_seconds",
    "timing_line_nls_filter_seconds",
    "timing_coverage_seconds",
    "timing_levenshtein_seconds",
    "timing_total_seconds",
]

BEST_CONFIG_FIELDNAMES = [
    "index",
    "fname",
    "normalised_levenshtein_similarity",
    "metric_outcome_reason",
    "best_tuning_score",
    "selection_objective",
    "alignment_selection_score",
    "score_matrix_support",
    "line_guided_fraction",
    "average_weighted_normalised_levenshtein_similarity",
    "correct_ref_coverage",
    "missing_ref_coverage",
    "repetition_on_ref",
    "hallucination",
    "hough_threshold",
    "hough_line_length",
    "hough_line_gap",
    "hough_seed",
    "line_guided_columns",
    "fallback_columns",
    "used_line_count",
    "used_line_count_ref_to_ref",
    "raw_line_count",
    "raw_line_count_ref_to_ref",
    "candidate_line_count",
    "candidate_line_count_ref_to_ref",
    "timing_total_seconds",
    "evaluated_combination_count",
    "invalid_combination_count",
    "invalid_y_diff_le_minus_one_total",
    "invalid_y_diff_lt_minus_one_total",
    "line_nls_filter_enabled",
    "min_surviving_line_nls",
    "line_nls_filter_input_line_count",
    "line_nls_filter_scored_line_count",
    "line_nls_filter_removed_line_count",
    "line_nls_filter_surviving_line_count",
    "line_nls_filter_removed_column_count",
    "line_nls_filter_surviving_column_count",
    "line_nls_filter_all_lines_removed",
    "line_nls_filter_all_removed_combination_count",
    "doc_grid_seconds",
]

SKIPPED_DOCUMENT_FIELDNAMES = [
    "index",
    "fname",
    "skip_reason",
    "skip_stage",
    "prediction_character_count",
    "prediction_non_whitespace_character_count",
    "reference_character_count",
    "reference_non_whitespace_character_count",
    "ref_to_pred_matrix_rows",
    "ref_to_pred_matrix_cols",
    "ref_to_pred_source",
    "ref_to_pred_matrix_max",
    "preprocessing_matrix_direction",
    "preprocessing_rejection_reason",
    "preprocessing_score_floor",
    "preprocessing_active_cells",
    "preprocessing_active_fraction",
    "preprocessing_seconds",
    "diagnostic_bundle_dir",
    "message",
]


def _csv_value(value):
    """Return a CSV-safe scalar while preserving missing values as empty cells."""
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10f}"
    return value


def write_parameter_curve_csv(*, rows: list[dict], output_csv: Path) -> None:
    """Write one CSV file for one parameter curve."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=PARAMETER_CURVE_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            aliased_row = {
                **row,
                "mean_average_weighted_normalised_levenshtein_similarity": row.get(
                    "mean_weighted_along_lines_nls"
                ),
            }
            writer.writerow({field: _csv_value(aliased_row.get(field)) for field in PARAMETER_CURVE_FIELDNAMES})


def write_best_configs_csv(*, best_records: list[dict], output_csv: Path) -> None:
    """Write the best parameter combination per document into one CSV."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=BEST_CONFIG_FIELDNAMES)
        writer.writeheader()
        for rec in best_records:
            best = rec.get("best", {}) if isinstance(rec, dict) else {}
            row = {
                "index": int(rec.get("index", 0)),
                "fname": str(rec.get("fname", "")),
                "normalised_levenshtein_similarity": rec.get("whole_document_nls"),
                "metric_outcome_reason": best.get("metric_outcome_reason"),
                "best_tuning_score": best.get("tuning_score"),
                "selection_objective": best.get("selection_objective"),
                "alignment_selection_score": best.get("alignment_selection_score"),
                "score_matrix_support": best.get("score_matrix_support"),
                "line_guided_fraction": best.get("line_guided_fraction"),
                "average_weighted_normalised_levenshtein_similarity": best.get("weighted_along_lines_nls"),
                "correct_ref_coverage": best.get("correct_ref_coverage"),
                "missing_ref_coverage": best.get("missing_ref_coverage"),
                "repetition_on_ref": best.get("repetition_on_ref"),
                "hallucination": best.get("hallucination"),
                "hough_threshold": int(best.get("hough_threshold", 0)),
                "hough_line_length": int(best.get("hough_line_length", 0)),
                "hough_line_gap": int(best.get("hough_line_gap", 0)),
                "hough_seed": int(best.get("hough_seed", 0)),
                "line_guided_columns": int(best.get("line_guided_columns", 0)),
                "fallback_columns": int(best.get("fallback_columns", 0)),
                "used_line_count": int(best.get("used_line_count", 0)),
                "used_line_count_ref_to_ref": int(best.get("used_line_count_ref_to_ref", 0)),
                "raw_line_count": int(best.get("raw_line_count", 0)),
                "raw_line_count_ref_to_ref": int(best.get("raw_line_count_ref_to_ref", 0)),
                "candidate_line_count": int(best.get("candidate_line_count", 0)),
                "candidate_line_count_ref_to_ref": int(best.get("candidate_line_count_ref_to_ref", 0)),
                "timing_total_seconds": best.get("timing_total_seconds"),
                "evaluated_combination_count": int(rec.get("evaluated_combination_count", 0)),
                "invalid_combination_count": int(rec.get("invalid_combination_count", 0)),
                "invalid_y_diff_le_minus_one_total": int(rec.get("invalid_y_diff_le_minus_one_total", 0)),
                "invalid_y_diff_lt_minus_one_total": int(rec.get("invalid_y_diff_lt_minus_one_total", 0)),
                "line_nls_filter_enabled": int(bool(best.get("line_nls_filter_enabled", False))),
                "min_surviving_line_nls": best.get("min_surviving_line_nls"),
                "line_nls_filter_input_line_count": int(best.get("line_nls_filter_input_line_count", 0) or 0),
                "line_nls_filter_scored_line_count": int(best.get("line_nls_filter_scored_line_count", 0) or 0),
                "line_nls_filter_removed_line_count": int(best.get("line_nls_filter_removed_line_count", 0) or 0),
                "line_nls_filter_surviving_line_count": int(best.get("line_nls_filter_surviving_line_count", 0) or 0),
                "line_nls_filter_removed_column_count": int(best.get("line_nls_filter_removed_column_count", 0) or 0),
                "line_nls_filter_surviving_column_count": int(best.get("line_nls_filter_surviving_column_count", 0) or 0),
                "line_nls_filter_all_lines_removed": int(bool(best.get("line_nls_filter_all_lines_removed", False))),
                "line_nls_filter_all_removed_combination_count": int(
                    rec.get("line_nls_filter_all_removed_combination_count", 0) or 0
                ),
                "doc_grid_seconds": rec.get("doc_grid_seconds"),
            }
            writer.writerow({field: _csv_value(row.get(field)) for field in BEST_CONFIG_FIELDNAMES})


def write_skipped_documents_csv(*, skipped_records: list[dict], output_csv: Path) -> None:
    """Write documents intentionally skipped before the Hough sweep."""
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = DictWriter(fh, fieldnames=SKIPPED_DOCUMENT_FIELDNAMES)
        writer.writeheader()
        for record in skipped_records:
            writer.writerow({field: _csv_value(record.get(field)) for field in SKIPPED_DOCUMENT_FIELDNAMES})


__all__ = [
    "PARAMETER_CURVE_FIELDNAMES",
    "BEST_CONFIG_FIELDNAMES",
    "SKIPPED_DOCUMENT_FIELDNAMES",
    "write_parameter_curve_csv",
    "write_best_configs_csv",
    "write_skipped_documents_csv",
]
