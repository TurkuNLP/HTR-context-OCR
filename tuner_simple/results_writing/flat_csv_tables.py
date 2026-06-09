from __future__ import annotations

"""Flat CSV and JSON writers for the simple tuner."""

from collections import defaultdict
from csv import DictWriter
import json
from pathlib import Path
from typing import Any

# Compute or store DOCUMENT_RESULT_FIELDNAMES so later code can reuse this named value clearly.
DOCUMENT_RESULT_FIELDNAMES = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "document_index",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "fname",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "main_language",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "document_type",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "matrix_source_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "matrix_source_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "matrix_load_reason_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "matrix_load_reason_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "row_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "column_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "ref_to_ref_row_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "ref_to_ref_column_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_floor_alpha",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_mean_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_standard_deviation_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_floor_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "active_cell_count_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "active_fraction_ref_to_pred",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_mean_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_standard_deviation_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "score_floor_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "active_cell_count_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "active_fraction_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "hough_threshold",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "hough_line_length",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "hough_line_gap",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "hough_seed",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "align_min_iou_threshold",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "min_surviving_line_nls",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "raw_line_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "candidate_line_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "used_line_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "raw_line_count_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "candidate_line_count_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "used_line_count_ref_to_ref",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "document_normalised_levenshtein",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "weighted_along_lines_normalised_levenshtein",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "correct_ref_coverage",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "missing_ref_coverage",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "repetition_on_reference",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "hallucination",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_invalid_reason",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_invalid_error_message",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_y_diff_size",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_y_diff_min",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_y_diff_max",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_y_diff_le_minus_one_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_y_diff_lt_minus_one_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "coverage_y_diff_below_minus_one_counts_json",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_matrix_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_preprocessing_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_hough_detect_ref_to_pred_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_filter_ref_to_pred_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_hough_detect_ref_to_ref_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_filter_ref_to_ref_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_coverage_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_levenshtein_seconds",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "timing_total_seconds",
]

# Compute or store DOCUMENT_TABLE_FIELDNAMES so later code can reuse this named value clearly.
DOCUMENT_TABLE_FIELDNAMES = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "document_index",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "fname",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "main_language",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "document_type",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "reference_text_length",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "prediction_text_length",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "reference_window_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "prediction_window_count",
]

# Compute or store SKIPPED_FIELDNAMES so later code can reuse this named value clearly.
SKIPPED_FIELDNAMES = DOCUMENT_TABLE_FIELDNAMES + [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "skip_stage",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "skip_reason",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "row_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "column_count",
]

# Compute or store SUMMARY_FIELDNAMES so later code can reuse this named value clearly.
SUMMARY_FIELDNAMES = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "main_language",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "document_type",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "runfile_document_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "loadable_document_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "loaded_document_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "skipped_document_count",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "mean_document_normalised_levenshtein",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "mean_weighted_along_lines_normalised_levenshtein",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "mean_correct_ref_coverage",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "mean_missing_ref_coverage",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "mean_repetition_on_reference",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "mean_hallucination",
]


# Define the csv_value function; its body below performs one named step of the pipeline.
def csv_value(value: Any) -> Any:
    """Return a stable CSV scalar while keeping missing values empty."""
    # Check whether value is None; the indented block handles that specific case.
    if value is None:
        # Return this computed value to the caller so the next pipeline stage can use it.
        return ""
    # Check whether isinstance(value, float); the indented block handles that specific case.
    if isinstance(value, float):
        # Return this computed value to the caller so the next pipeline stage can use it.
        return f"{value:.10f}"
    # Return this computed value to the caller so the next pipeline stage can use it.
    return value


# Define the write_csv function; its body below performs one named step of the pipeline.
def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write one CSV with a fixed header, even when there are no rows."""
    # Build a Path object so filesystem locations are handled consistently across the pipeline.
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # Open or manage Path(path).open("w", encoding="utf-8", newline="") as handle with automatic cleanup after the indented block finishes.
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        # Compute or store writer so later code can reuse this named value clearly.
        writer = DictWriter(handle, fieldnames=fieldnames)
        # Execute this statement as the next small step in the surrounding pipeline logic.
        writer.writeheader()
        # Iterate over row in rows so each item is processed with the same logic.
        for row in rows:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            writer.writerow({field: csv_value(row.get(field)) for field in fieldnames})


# Define the mean_of_present function; its body below performs one named step of the pipeline.
def mean_of_present(rows: list[dict], field: str) -> float | None:
    """Return the mean of numeric present values for one result column."""
    # Compute or store values: list[float] so later code can reuse this named value clearly.
    values: list[float] = []
    # Iterate over row in rows so each item is processed with the same logic.
    for row in rows:
        # Compute or store value so later code can reuse this named value clearly.
        value = row.get(field)
        # Check whether value is None or value == ""; the indented block handles that specific case.
        if value is None or value == "":
            # Skip the rest of this loop iteration and move to the next item.
            continue
        # Define the try field so this data object records that value explicitly.
        try:
            # Add this item to the list that is accumulating results for later output.
            values.append(float(value))
        # Catch the matching failure type and turn it into explicit handling instead of crashing silently.
        except Exception:
            # Skip the rest of this loop iteration and move to the next item.
            continue
    # Return this computed value to the caller so the next pipeline stage can use it.
    return None if not values else float(sum(values) / len(values))


# Define the build_document_type_summary_rows function; its body below performs one named step of the pipeline.
def build_document_type_summary_rows(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the runfile_rows field so this data object records that value explicitly.
    runfile_rows: list[dict],
    # Define the loadable_rows field so this data object records that value explicitly.
    loadable_rows: list[dict],
    # Define the loaded_rows field so this data object records that value explicitly.
    loaded_rows: list[dict],
    # Define the skipped_rows field so this data object records that value explicitly.
    skipped_rows: list[dict],
    # Define the result_rows field so this data object records that value explicitly.
    result_rows: list[dict],
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> list[dict]:
    """Aggregate document counts and six metric means by language and type."""
    # Compute or store keys so later code can reuse this named value clearly.
    keys = sorted({
        # Execute this statement as the next small step in the surrounding pipeline logic.
        (row.get("main_language", ""), row.get("document_type", ""))
        # Iterate over row in runfile_rows + loadable_rows + loaded_rows + skipped_rows + result_rows so each item is processed with the same logic.
        for row in runfile_rows + loadable_rows + loaded_rows + skipped_rows + result_rows
    # Execute this statement as the next small step in the surrounding pipeline logic.
    })
    # Compute or store rows_by_key so later code can reuse this named value clearly.
    rows_by_key = defaultdict(list)
    # Iterate over row in result_rows so each item is processed with the same logic.
    for row in result_rows:
        # Add this item to the list that is accumulating results for later output.
        rows_by_key[(row.get("main_language", ""), row.get("document_type", ""))].append(row)

    # Define the count_for function; its body below performs one named step of the pipeline.
    def count_for(source_rows: list[dict], key: tuple[str, str]) -> int:
        # Return this computed value to the caller so the next pipeline stage can use it.
        return sum(1 for row in source_rows if (row.get("main_language", ""), row.get("document_type", "")) == key)

    # Compute or store summary_rows: list[dict] so later code can reuse this named value clearly.
    summary_rows: list[dict] = []
    # Iterate over key in keys so each item is processed with the same logic.
    for key in keys:
        # Compute or store language, document_type so later code can reuse this named value clearly.
        language, document_type = key
        # Compute or store metric_rows so later code can reuse this named value clearly.
        metric_rows = rows_by_key[key]
        # Add this item to the list that is accumulating results for later output.
        summary_rows.append(
            # Start a multi-line collection so related values can be listed clearly.
            {
                # Add the main_language field to the surrounding dictionary so it appears in outputs or returned metadata.
                "main_language": language,
                # Add the document_type field to the surrounding dictionary so it appears in outputs or returned metadata.
                "document_type": document_type,
                # Add the runfile_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
                "runfile_document_count": count_for(runfile_rows, key),
                # Add the loadable_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
                "loadable_document_count": count_for(loadable_rows, key),
                # Add the loaded_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
                "loaded_document_count": count_for(loaded_rows, key),
                # Add the skipped_document_count field to the surrounding dictionary so it appears in outputs or returned metadata.
                "skipped_document_count": count_for(skipped_rows, key),
                # Add the mean_document_normalised_levenshtein field to the surrounding dictionary so it appears in outputs or returned metadata.
                "mean_document_normalised_levenshtein": mean_of_present(metric_rows, "document_normalised_levenshtein"),
                # Add the mean_weighted_along_lines_normalised_levenshtein field to the surrounding dictionary so it appears in outputs or returned metadata.
                "mean_weighted_along_lines_normalised_levenshtein": mean_of_present(metric_rows, "weighted_along_lines_normalised_levenshtein"),
                # Add the mean_correct_ref_coverage field to the surrounding dictionary so it appears in outputs or returned metadata.
                "mean_correct_ref_coverage": mean_of_present(metric_rows, "correct_ref_coverage"),
                # Add the mean_missing_ref_coverage field to the surrounding dictionary so it appears in outputs or returned metadata.
                "mean_missing_ref_coverage": mean_of_present(metric_rows, "missing_ref_coverage"),
                # Add the mean_repetition_on_reference field to the surrounding dictionary so it appears in outputs or returned metadata.
                "mean_repetition_on_reference": mean_of_present(metric_rows, "repetition_on_reference"),
                # Add the mean_hallucination field to the surrounding dictionary so it appears in outputs or returned metadata.
                "mean_hallucination": mean_of_present(metric_rows, "hallucination"),
            }
        )
    # Return this computed value to the caller so the next pipeline stage can use it.
    return summary_rows


# Define the write_all_flat_outputs function; its body below performs one named step of the pipeline.
def write_all_flat_outputs(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the output_dir field; it stores the directory where CSV, JSON, and optional plot files will be written.
    output_dir: Path,
    # Define the runfile_rows field so this data object records that value explicitly.
    runfile_rows: list[dict],
    # Define the loadable_rows field so this data object records that value explicitly.
    loadable_rows: list[dict],
    # Define the loaded_rows field so this data object records that value explicitly.
    loaded_rows: list[dict],
    # Define the skipped_rows field so this data object records that value explicitly.
    skipped_rows: list[dict],
    # Define the result_rows field so this data object records that value explicitly.
    result_rows: list[dict],
    # Define the run_summary field so this data object records that value explicitly.
    run_summary: dict,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> dict[str, str]:
    """Write every run-level table directly under the user-selected output directory."""
    # Compute or store output_dir so later code can reuse this named value clearly.
    output_dir = Path(output_dir)
    # Ensure the target directory exists before later code tries to write files into it.
    output_dir.mkdir(parents=True, exist_ok=True)
    # Compute or store summary_rows so later code can reuse this named value clearly.
    summary_rows = build_document_type_summary_rows(
        # Pass the runfile_rows argument into the surrounding call so the callee receives that setting explicitly.
        runfile_rows=runfile_rows,
        # Pass the loadable_rows argument into the surrounding call so the callee receives that setting explicitly.
        loadable_rows=loadable_rows,
        # Pass the loaded_rows argument into the surrounding call so the callee receives that setting explicitly.
        loaded_rows=loaded_rows,
        # Pass the skipped_rows argument into the surrounding call so the callee receives that setting explicitly.
        skipped_rows=skipped_rows,
        # Pass the result_rows argument into the surrounding call so the callee receives that setting explicitly.
        result_rows=result_rows,
    )
    # Compute or store paths so later code can reuse this named value clearly.
    paths = {
        # Add the best_combination_per_document_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "best_combination_per_document_csv": output_dir / "best_combination_per_document.csv",
        # Add the compact_combination_metrics_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "compact_combination_metrics_csv": output_dir / "compact_combination_metrics.csv",
        # Add the document_type_summary_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "document_type_summary_csv": output_dir / "document_type_summary.csv",
        # Add the loadable_documents_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "loadable_documents_csv": output_dir / "loadable_documents.csv",
        # Add the loaded_documents_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "loaded_documents_csv": output_dir / "loaded_documents.csv",
        # Add the runfile_documents_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "runfile_documents_csv": output_dir / "runfile_documents.csv",
        # Add the skipped_documents_csv field to the surrounding dictionary so it appears in outputs or returned metadata.
        "skipped_documents_csv": output_dir / "skipped_documents.csv",
        # Add the run_summary_json field to the surrounding dictionary so it appears in outputs or returned metadata.
        "run_summary_json": output_dir / "run_summary.json",
    }
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["best_combination_per_document_csv"], result_rows, DOCUMENT_RESULT_FIELDNAMES)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["compact_combination_metrics_csv"], result_rows, DOCUMENT_RESULT_FIELDNAMES)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["document_type_summary_csv"], summary_rows, SUMMARY_FIELDNAMES)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["loadable_documents_csv"], loadable_rows, DOCUMENT_TABLE_FIELDNAMES)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["loaded_documents_csv"], loaded_rows, DOCUMENT_TABLE_FIELDNAMES)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["runfile_documents_csv"], runfile_rows, DOCUMENT_TABLE_FIELDNAMES)
    # Execute this statement as the next small step in the surrounding pipeline logic.
    write_csv(paths["skipped_documents_csv"], skipped_rows, SKIPPED_FIELDNAMES)
    # Write serialized output to disk so the run can be inspected after the process exits.
    paths["run_summary_json"].write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Return this computed value to the caller so the next pipeline stage can use it.
    return {key: str(value) for key, value in paths.items()}


__all__ = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "DOCUMENT_RESULT_FIELDNAMES",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "DOCUMENT_TABLE_FIELDNAMES",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "SKIPPED_FIELDNAMES",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "SUMMARY_FIELDNAMES",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "write_all_flat_outputs",
]
