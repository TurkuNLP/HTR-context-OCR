from __future__ import annotations

"""Line-level text filtering and weighted along-line Levenshtein for tuner_simple."""

from dataclasses import dataclass
import time

import numpy as np

from tuner_simple.scoring.levenshtein import normalized_levenshtein_similarity


@dataclass(frozen=True)
class WeightedAlongLinesResult:
    """Weighted text similarity for the final surviving alignment lines."""

    # Store the weighted mean of surviving line similarities, or None when no line has text coverage.
    weighted_along_lines_nls: float | None
    # Store how many final lines contributed to the weighted mean.
    scored_line_count: int
    # Store how many prediction columns were covered by those scored lines.
    covered_column_count: int


@dataclass(frozen=True)
class LineTextRecord:
    """Text-similarity evidence for one final line."""

    # Store the line id before any line-level text filtering is applied.
    line_id: int
    # Store the normalized Levenshtein similarity between text covered by this line.
    normalized_levenshtein_similarity: float
    # Store how many prediction columns this line owns.
    covered_column_count: int


@dataclass(frozen=True)
class LineTextFilterResult:
    """Filtered line ownership and the text score calculated for surviving lines."""

    # Store the filtered line result after weak text lines have been removed.
    filtered_result: dict
    # Store the weighted along-lines score computed from the surviving text lines.
    weighted_result: WeightedAlongLinesResult | None
    # Store whether the user enabled the line-level text filter.
    filter_enabled: bool
    # Store how many lines entered the line-level text filter.
    input_line_count: int
    # Store how many lines received a text score.
    scored_line_count: int
    # Store how many lines were removed by the line-level text filter.
    removed_line_count: int
    # Store how many lines survived the line-level text filter.
    surviving_line_count: int
    # Store how many previously assigned prediction columns were removed with rejected lines.
    removed_column_count: int
    # Store how many prediction columns remain assigned after text filtering.
    surviving_column_count: int
    # Store whether every input line was removed.
    all_lines_removed: bool
    # Store how long the line-level text filtering took.
    seconds: float


def safe_window_at(windows: list[str], index: int) -> str:
    """Return one text window or an empty string when the index is outside the list."""
    # Reject negative indices because they mean the line assignment did not point to a valid row.
    if int(index) < 0:
        return ""
    # Reject indices beyond the available text-window list.
    if int(index) >= len(windows):
        return ""
    # Return the selected text window.
    return str(windows[int(index)])


def line_owned_columns(column_assignment: dict, line_id: int) -> list[int]:
    """Return prediction-column indices owned by one final line."""
    # Convert the assignment array into integer ids so comparisons are stable.
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    # Return every column where the final line id matches the requested line.
    return [int(column_index) for column_index in np.flatnonzero(mapped_line_id == int(line_id))]


def line_reference_rows(column_assignment: dict, owned_columns: list[int], reference_window_count: int) -> list[int]:
    """Return rounded reference-row indices mapped from one line's owned prediction columns."""
    # Convert mapped y coordinates into a float array because unassigned entries are NaN.
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    # Store valid rounded reference rows in prediction-column order.
    rows: list[int] = []
    # Inspect each prediction column owned by the line.
    for column_index in owned_columns:
        # Skip columns outside the mapped_y array defensively.
        if int(column_index) < 0 or int(column_index) >= mapped_y.size:
            continue
        # Read the floating reference-row estimate for this prediction column.
        y_value = float(mapped_y[int(column_index)])
        # Skip unassigned or invalid y values.
        if not np.isfinite(y_value):
            continue
        # Round to the nearest reference-window row because text windows are discrete.
        row_index = int(round(y_value))
        # Keep only rows that exist in the reference window list.
        if 0 <= row_index < int(reference_window_count):
            rows.append(int(row_index))
    # Return the valid row sequence.
    return rows


def compute_line_text_record(
    *,
    line_id: int,
    column_assignment: dict,
    reference_windows: list[str],
    prediction_windows: list[str],
) -> LineTextRecord | None:
    """Compute one line's text similarity from the windows it owns."""
    # Identify the prediction columns owned by this line.
    owned_columns = line_owned_columns(column_assignment, int(line_id))
    # A line with no prediction columns has no text evidence.
    if not owned_columns:
        return None
    # Convert owned prediction columns into their mapped reference rows.
    reference_rows = line_reference_rows(column_assignment, owned_columns, len(reference_windows))
    # If no valid reference rows exist, this line cannot be scored as a text alignment.
    if not reference_rows:
        return None
    # Build reference text in the same column order as the line assignments.
    reference_text = " ".join(safe_window_at(reference_windows, row_index) for row_index in reference_rows)
    # Build prediction text from the prediction windows owned by this line.
    prediction_text = " ".join(safe_window_at(prediction_windows, column_index) for column_index in owned_columns)
    # Compute the normalized similarity between the two line-level text strings.
    similarity = normalized_levenshtein_similarity(
        prediction_text,
        reference_text,
    )
    # Return a compact record used by filtering and weighted averaging.
    return LineTextRecord(
        line_id=int(line_id),
        normalized_levenshtein_similarity=float(similarity),
        covered_column_count=int(len(owned_columns)),
    )


def compute_line_text_records(
    *,
    lines_used: list[dict],
    column_assignment: dict,
    reference_windows: list[str],
    prediction_windows: list[str],
) -> list[LineTextRecord]:
    """Compute text-similarity records for all current final lines."""
    # Store scored records in final-line order.
    records: list[LineTextRecord] = []
    # Score every final line by its current line id.
    for line_id, _line_record in enumerate(lines_used):
        # Compute the line text score from the columns currently assigned to this line.
        record = compute_line_text_record(
            line_id=int(line_id),
            column_assignment=column_assignment,
            reference_windows=reference_windows,
            prediction_windows=prediction_windows,
            )
        # Keep only lines that have enough assignment data to score.
        if record is not None:
            records.append(record)
    # Return all scored line records.
    return records


def weighted_result_from_records(records: list[LineTextRecord]) -> WeightedAlongLinesResult:
    """Return a column-count weighted average of line text similarities."""
    # Sum the column weights for every scored line.
    total_weight = sum(max(0, int(record.covered_column_count)) for record in records)
    # If no scored line covers columns, the weighted score is undefined.
    if total_weight <= 0:
        return WeightedAlongLinesResult(weighted_along_lines_nls=None, scored_line_count=0, covered_column_count=0)
    # Compute the weighted sum of each line similarity by its covered-column count.
    weighted_sum = sum(
        float(record.normalized_levenshtein_similarity) * float(max(0, int(record.covered_column_count)))
        for record in records
    )
    # Divide by total covered columns to get the final weighted along-lines score.
    return WeightedAlongLinesResult(
        weighted_along_lines_nls=float(weighted_sum / float(total_weight)),
        scored_line_count=int(len(records)),
        covered_column_count=int(total_weight),
    )


def prune_assignment_to_kept_lines(
    *,
    filtered_result: dict,
    kept_original_line_ids: list[int],
    line_score_by_original_id: dict[int, LineTextRecord],
) -> dict:
    """Return a filtered-result dictionary containing only selected line ids."""
    # Read original final lines before text filtering.
    original_lines = list(filtered_result.get("lines_used", []))
    # Read the original column assignment arrays.
    original_assignment = filtered_result.get("column_assignment", {})
    # Convert mapped y values to a float array so rejected columns can become NaN.
    original_mapped_y = np.asarray(original_assignment.get("mapped_y", []), dtype=float)
    # Convert original line ids to an integer array for masking.
    original_mapped_line_id = np.asarray(original_assignment.get("mapped_line_id", []), dtype=int)
    # Store the pruned line dictionaries after compacting ids.
    pruned_lines: list[dict] = []
    # Map original line ids to new compact line ids.
    original_to_new_line_id: dict[int, int] = {}
    # Keep requested lines in their original order.
    for new_line_id, original_line_id in enumerate(kept_original_line_ids):
        # Copy the original line record so text-filter metadata is local to this result.
        line_copy = dict(original_lines[int(original_line_id)])
        # Store the line-level similarity used by the filter.
        line_copy["line_nls"] = float(line_score_by_original_id[int(original_line_id)].normalized_levenshtein_similarity)
        # Add the copied line to the pruned final line list.
        pruned_lines.append(line_copy)
        # Store the original-to-new id mapping for assignment rewriting.
        original_to_new_line_id[int(original_line_id)] = int(new_line_id)
    # Start every prediction column as unassigned after pruning.
    pruned_mapped_line_id = np.full(original_mapped_line_id.shape, -1, dtype=int)
    # Start every prediction column y coordinate as unknown after pruning.
    pruned_mapped_y = np.full(original_mapped_y.shape, np.nan, dtype=float)
    # Rewrite only columns whose original line survived.
    for original_line_id, new_line_id in original_to_new_line_id.items():
        # Identify columns owned by this original line id.
        owned_column_mask = original_mapped_line_id == int(original_line_id)
        # Assign those columns to the new compact line id.
        pruned_mapped_line_id[owned_column_mask] = int(new_line_id)
        # Preserve the original mapped y coordinates for those surviving columns.
        pruned_mapped_y[owned_column_mask] = original_mapped_y[owned_column_mask]
    # Return a copy of the filtering result with updated final lines and assignments.
    return {
        **filtered_result,
        "lines_used": pruned_lines,
        "column_assignment": {"mapped_y": pruned_mapped_y, "mapped_line_id": pruned_mapped_line_id},
    }


def filter_lines_by_minimum_normalised_levenshtein(
    *,
    filtered_result: dict,
    reference_windows: list[str],
    prediction_windows: list[str],
    reference_window_count: int,
    minimum_line_nls: float | None,
) -> LineTextFilterResult:
    """Remove final lines whose own text similarity is below the user floor."""
    # Mark the beginning of line-text filtering for timing output.
    started_at = time.perf_counter()
    # Copy current final lines so this function never mutates the caller's list in place.
    original_lines = list(filtered_result.get("lines_used", []))
    # Read current column assignments.
    original_assignment = filtered_result.get("column_assignment", {})
    # Count how many prediction columns were assigned before the text filter.
    original_mapped_line_id = np.asarray(original_assignment.get("mapped_line_id", []), dtype=int)
    # Store the number of originally assigned columns for removal statistics.
    original_guided_column_count = int(np.count_nonzero(original_mapped_line_id >= 0))
    # Compute line-level similarity records regardless of whether pruning is enabled, because the weighted score needs them.
    line_records = compute_line_text_records(
        lines_used=original_lines,
        column_assignment=original_assignment,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
    )
    # Store line scores by their original line id for quick filter decisions.
    record_by_line_id = {int(record.line_id): record for record in line_records}
    # If the user disabled the filter, keep all original lines and report the weighted score from all scored records.
    if minimum_line_nls is None:
        # Build the weighted score from all current line records.
        weighted_result = weighted_result_from_records(line_records)
        # Return the original filtered result unchanged.
        return LineTextFilterResult(
            filtered_result=filtered_result,
            weighted_result=weighted_result,
            filter_enabled=False,
            input_line_count=int(len(original_lines)),
            scored_line_count=int(len(line_records)),
            removed_line_count=0,
            surviving_line_count=int(len(original_lines)),
            removed_column_count=0,
            surviving_column_count=int(original_guided_column_count),
            all_lines_removed=False,
            seconds=float(time.perf_counter() - started_at),
        )
    # Select only scored lines whose text similarity reaches the configured minimum.
    kept_original_line_ids = [
        int(line_id)
        for line_id in range(len(original_lines))
        if int(line_id) in record_by_line_id
        and float(record_by_line_id[int(line_id)].normalized_levenshtein_similarity) >= float(minimum_line_nls)
    ]
    # Rewrite final lines and assignments to contain only the kept line ids.
    pruned_filtered_result = prune_assignment_to_kept_lines(
        filtered_result=filtered_result,
        kept_original_line_ids=kept_original_line_ids,
        line_score_by_original_id=record_by_line_id,
    )
    # Keep records in the same order as the surviving original line ids.
    surviving_records = [record_by_line_id[int(line_id)] for line_id in kept_original_line_ids if int(line_id) in record_by_line_id]
    # Compute the weighted score using only surviving lines.
    weighted_result = weighted_result_from_records(surviving_records)
    # Count how many prediction columns are still assigned after pruning.
    pruned_mapped_line_id = np.asarray(pruned_filtered_result.get("column_assignment", {}).get("mapped_line_id", []), dtype=int)
    # Store surviving column count for output statistics.
    surviving_guided_column_count = int(np.count_nonzero(pruned_mapped_line_id >= 0))
    # Compute how many assigned columns were removed with rejected lines.
    removed_column_count = max(0, int(original_guided_column_count - surviving_guided_column_count))
    # Return the complete text-filter summary.
    return LineTextFilterResult(
        filtered_result=pruned_filtered_result,
        weighted_result=weighted_result,
        filter_enabled=True,
        input_line_count=int(len(original_lines)),
        scored_line_count=int(len(line_records)),
        removed_line_count=int(len(original_lines) - len(kept_original_line_ids)),
        surviving_line_count=int(len(kept_original_line_ids)),
        removed_column_count=int(removed_column_count),
        surviving_column_count=int(surviving_guided_column_count),
        all_lines_removed=bool(len(original_lines) > 0 and len(kept_original_line_ids) == 0),
        seconds=float(time.perf_counter() - started_at),
    )


def compute_weighted_along_lines_from_payload(
    *,
    reference_windows: list[str],
    prediction_windows: list[str],
    lines_used: list[dict],
    compact_payload: dict,
) -> WeightedAlongLinesResult:
    """Compute weighted along-line similarity from local assignment payload data."""
    # Read the assignment dictionary from the local compact payload.
    column_assignment = compact_payload.get("column_assignment", {})
    # Compute one line-level similarity record for every final line.
    records = compute_line_text_records(
        lines_used=lines_used,
        column_assignment=column_assignment,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
    )
    # Convert records into the weighted along-lines score.
    return weighted_result_from_records(records)


# Declare the public helpers that other tuner_simple modules may import.
__all__ = [
    "LineTextFilterResult",
    "WeightedAlongLinesResult",
    "compute_weighted_along_lines_from_payload",
    "filter_lines_by_minimum_normalised_levenshtein",
]
