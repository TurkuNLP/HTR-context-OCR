from __future__ import annotations

"""Line-level text filtering and weighted along-line Levenshtein for tuner_simple_alpha_sweep_pre_iou_levenshtein."""

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.levenshtein import normalized_levenshtein_similarity


@dataclass(frozen=True)
class WeightedAlongLinesResult:
    """Weighted and unweighted text similarity for final surviving alignment lines."""

    # Store the line-length weighted mean, or None when no final line has usable text evidence.
    weighted_along_lines_nls: float | None
    # Store the simple arithmetic mean across scored lines so audits can compare weighted and unweighted evidence.
    unweighted_along_lines_nls: float | None = None
    # Store how many final lines contributed a valid line-level text score.
    scored_line_count: int = 0
    # Store the total geometric length of scored lines, because v2.2 weights line text by line length.
    total_line_length: float = 0.0
    # Store covered prediction columns only for old tuner_simple test compatibility; this is not the v2.2 weight.
    covered_column_count: int = 0


@dataclass(frozen=True)
class LineTextRecord:
    """Text-similarity evidence for one final line before optional line pruning."""

    # Store the line id before text filtering renumbers surviving lines.
    line_id: int
    # Store the normalized Levenshtein similarity between prediction text and reference text for this line.
    normalized_levenshtein_similarity: float
    # Store the geometric line length used as the weighted along-line averaging weight.
    line_length: float
    # Store how many prediction windows this line owns after true-IoU filtering.
    owned_prediction_column_count: int
    # Store how many reference windows are compared to the owned prediction windows.
    mapped_reference_row_count: int


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


def clamp_unit_interval(value) -> float:
    """Return a finite numeric value clipped into the closed interval [0, 1]."""
    # Try to convert the incoming value to a float because metric code may receive NumPy scalars.
    try:
        numeric_value = float(value)
    except Exception:
        return 0.0
    # Treat NaN and infinity as zero-quality evidence so they cannot pass a text gate by accident.
    if not math.isfinite(numeric_value):
        return 0.0
    # Keep valid similarities inside the probability-like unit interval used by the metrics.
    return max(0.0, min(1.0, numeric_value))


def euclidean_line_length(line_record: dict) -> float:
    """Return the finite positive geometric length for one final filtered line."""
    # Reuse a cached length when the Hough filtering stage already stored one on the line record.
    existing_length = line_record.get("length") if isinstance(line_record, dict) else None
    try:
        cached_length = float(existing_length)
    except Exception:
        cached_length = float("nan")
    # Accept the cached value only when it is finite and positive.
    if math.isfinite(cached_length) and cached_length > 0.0:
        return float(cached_length)

    # Fall back to endpoint geometry for line records that do not carry a length field.
    try:
        x0 = float(line_record.get("x0", 0.0))
        y0 = float(line_record.get("y0", 0.0))
        x1 = float(line_record.get("x1", 0.0))
        y1 = float(line_record.get("y1", 0.0))
    except Exception:
        return 0.0

    # Measure the straight-line distance between the two endpoints in matrix-cell coordinates.
    computed_length = math.hypot(x1 - x0, y1 - y0)
    # Reject zero-length or invalid segments because they cannot be meaningful text-line evidence.
    if not math.isfinite(computed_length) or computed_length <= 0.0:
        return 0.0
    return float(computed_length)


def join_text_windows_without_separators(windows: Sequence[str], indices: Sequence[int]) -> str:
    """Concatenate valid text windows in the exact order used by v2.2 line scoring."""
    # Store the valid window count once so each index check is cheap and explicit.
    window_count = len(windows)
    # Join with an empty separator because sliding windows are already slices of contiguous source text.
    return "".join(str(windows[int(index)]) for index in indices if 0 <= int(index) < int(window_count))


def ordered_unique(values: Sequence[int]) -> list[int]:
    """Return integers in first-seen order while removing duplicates."""
    # Store output values in the order the line encountered them across prediction columns.
    ordered_values: list[int] = []
    # Track values already emitted so repeated windows do not duplicate the same reference row.
    seen_values: set[int] = set()
    # Visit values in their original order because monotonic line text should preserve column order.
    for value in values:
        # Normalize NumPy integer types and Python integers to the same dictionary key type.
        integer_value = int(value)
        # Skip values that were already emitted for this line.
        if integer_value in seen_values:
            continue
        # Keep the first occurrence of this value.
        ordered_values.append(integer_value)
        # Remember that the value has already been used.
        seen_values.add(integer_value)
    # Return the de-duplicated sequence.
    return ordered_values


def sequence_is_non_decreasing(values: Sequence[int]) -> bool:
    """Return True when values never move backward."""
    # Compare every neighboring pair and require each later value to be at least the previous value.
    return all(int(left_value) <= int(right_value) for left_value, right_value in zip(values, values[1:]))


def reference_rows_for_mapped_columns(
    owned_prediction_columns: Sequence[int],
    mapped_y: np.ndarray,
    reference_window_count: int,
) -> tuple[list[int], bool]:
    """Return the reference-row sequence used for line-level Levenshtein."""
    # A matrix with no reference windows cannot provide reference text for a line.
    if int(reference_window_count) <= 0:
        return [], False

    # Convert each owned prediction column into a rounded, clipped reference-row id.
    mapped_reference_rows = [
        int(np.clip(round(float(mapped_y[int(column_index)])), 0, int(reference_window_count) - 1))
        for column_index in owned_prediction_columns
        if 0 <= int(column_index) < int(mapped_y.shape[0]) and np.isfinite(mapped_y[int(column_index)])
    ]
    # If no owned column maps to a valid y coordinate, the line cannot be text-scored.
    if not mapped_reference_rows:
        return [], False

    # Remove repeated reference rows while preserving the line's left-to-right order.
    unique_reference_rows = ordered_unique(mapped_reference_rows)
    # Preserve the line order when it is already monotonic along the reference axis.
    if sequence_is_non_decreasing(unique_reference_rows):
        return unique_reference_rows, False

    # Sort non-monotonic rows so text comparison still reads reference text in document order.
    return sorted(set(unique_reference_rows)), True


def reference_rows_for_levenshtein(
    owned_prediction_columns: Sequence[int],
    mapped_y: np.ndarray,
    reference_window_count: int,
) -> tuple[list[int], bool]:
    """Compatibility wrapper matching the v2.2 helper name and behavior."""
    # Delegate to the local implementation copied from the v2.12-compatible semantics.
    return reference_rows_for_mapped_columns(
        owned_prediction_columns=owned_prediction_columns,
        mapped_y=mapped_y,
        reference_window_count=int(reference_window_count),
    )


def line_owned_columns(column_assignment: dict, line_id: int) -> list[int]:
    """Return prediction-column indices owned by one final line."""
    # Convert the assignment array into integer ids so comparisons are stable.
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    # Return every column where the final line id matches the requested line.
    return [int(column_index) for column_index in np.flatnonzero(mapped_line_id == int(line_id))]


def compute_line_text_record(
    *,
    line_id: int,
    line_record: dict,
    column_assignment: dict,
    reference_windows: Sequence[str],
    prediction_windows: Sequence[str],
    reference_window_count: int,
) -> LineTextRecord | None:
    """Compute one line's v2.2-compatible text similarity from the windows it owns."""
    # Identify the prediction columns owned by this final line after true-IoU filtering.
    owned_prediction_columns = line_owned_columns(column_assignment, int(line_id))
    # A line with no prediction columns has no text evidence and cannot pass a text-quality threshold.
    if not owned_prediction_columns:
        return None

    # Read the mapped reference y coordinate for every prediction column.
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    # Convert owned prediction columns into the reference rows used by v2.12-compatible Levenshtein text order.
    reference_rows_for_line, _rows_reordered_for_monotonicity = reference_rows_for_levenshtein(
        owned_prediction_columns,
        mapped_y,
        int(reference_window_count),
    )
    # A line without mapped reference rows has no reference text to compare against.
    if not reference_rows_for_line:
        return None

    # Concatenate prediction windows without separator because the windows are overlapping text slices.
    prediction_line_text = join_text_windows_without_separators(prediction_windows, owned_prediction_columns)
    # Concatenate mapped reference windows using the same no-separator rule as v2.2.
    reference_line_text = join_text_windows_without_separators(reference_windows, reference_rows_for_line)
    # Compute the normalized Levenshtein similarity for this line using tuner_simple's RapidFuzz implementation.
    line_score = float(normalized_levenshtein_similarity(prediction_line_text, reference_line_text))
    # Read or compute the geometric length used by v2.2 as the weighted along-line averaging weight.
    line_length = euclidean_line_length(line_record)

    # Reject invalid line scores or degenerate line geometry before the record can influence filtering or metrics.
    if not math.isfinite(line_score) or line_length <= 0.0:
        return None

    # Return the compact record consumed by the line text gate and along-line weighted average.
    return LineTextRecord(
        line_id=int(line_id),
        normalized_levenshtein_similarity=clamp_unit_interval(line_score),
        line_length=float(line_length),
        owned_prediction_column_count=int(len(owned_prediction_columns)),
        mapped_reference_row_count=int(len(reference_rows_for_line)),
    )


def compute_line_text_records(
    *,
    lines_used: list[dict],
    column_assignment: dict,
    reference_windows: Sequence[str],
    prediction_windows: Sequence[str],
    reference_window_count: int,
) -> list[LineTextRecord]:
    """Compute v2.2-compatible text-similarity records for all current final lines."""
    # Store scored records in final-line order.
    records: list[LineTextRecord] = []
    # Score every final line by the id it has before optional text pruning.
    for line_id, line_record in enumerate(lines_used):
        # Compute line text evidence from the same ownership arrays used for final metrics.
        record = compute_line_text_record(
            line_id=int(line_id),
            line_record=line_record,
            column_assignment=column_assignment,
            reference_windows=reference_windows,
            prediction_windows=prediction_windows,
            reference_window_count=int(reference_window_count),
        )
        # Keep only lines that have both prediction text, reference text, and positive geometry.
        if record is not None:
            records.append(record)
    # Return all scored line records.
    return records


def weighted_result_from_records(records: Sequence[LineTextRecord]) -> WeightedAlongLinesResult:
    """Return v2.2-compatible line-length weighted and unweighted line text scores."""
    # Keep only records whose similarity is finite and whose geometric length can be used as a positive weight.
    valid_records = [
        record
        for record in records
        if math.isfinite(float(record.normalized_levenshtein_similarity)) and float(record.line_length) > 0.0
    ]
    # If no line has valid text and positive length, the along-line score is undefined.
    if not valid_records:
        return WeightedAlongLinesResult(
            weighted_along_lines_nls=None,
            unweighted_along_lines_nls=None,
            scored_line_count=0,
            total_line_length=0.0,
            covered_column_count=0,
        )

    # Sum all line lengths once because v2.2 uses geometric line length as the averaging weight.
    total_line_length = sum(float(record.line_length) for record in valid_records)
    # Compute the weighted text score where longer accepted lines contribute proportionally more evidence.
    weighted_sum = sum(
        clamp_unit_interval(record.normalized_levenshtein_similarity) * float(record.line_length)
        for record in valid_records
    )
    # Compute the unweighted mean for diagnostic parity with v2.2.
    unweighted_mean = sum(clamp_unit_interval(record.normalized_levenshtein_similarity) for record in valid_records) / len(valid_records)
    # Sum owned prediction columns only for compatibility with existing tuner_simple tests and reports.
    covered_column_count = sum(int(record.owned_prediction_column_count) for record in valid_records)

    # Return both along-line scores and their evidence counts.
    return WeightedAlongLinesResult(
        weighted_along_lines_nls=clamp_unit_interval(weighted_sum / float(total_line_length)),
        unweighted_along_lines_nls=clamp_unit_interval(unweighted_mean),
        scored_line_count=int(len(valid_records)),
        total_line_length=float(total_line_length),
        covered_column_count=int(covered_column_count),
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

    # Keep requested lines in their original order so plot ordering remains stable.
    for new_line_id, original_line_id in enumerate(kept_original_line_ids):
        # Copy the original line record so text-filter metadata is local to this result.
        line_copy = dict(original_lines[int(original_line_id)])
        # Store the line-level similarity used by the filter for plots and CSV diagnostics.
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
    reference_windows: Sequence[str],
    prediction_windows: Sequence[str],
    reference_window_count: int,
    minimum_line_nls: float | None,
) -> LineTextFilterResult:
    """Remove final lines whose v2.2-compatible line-level text similarity is too low."""
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

    # Compute line-level similarity records exactly where v2.2 computes them: after true-IoU geometry filtering.
    line_records = compute_line_text_records(
        lines_used=original_lines,
        column_assignment=original_assignment,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
        reference_window_count=int(reference_window_count),
    )
    # Store line scores by their original line id for quick filter decisions.
    record_by_line_id = {int(record.line_id): record for record in line_records}

    # If the user disabled the text filter, keep all original lines and report weighted score from all scored records.
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
    reference_windows: Sequence[str],
    prediction_windows: Sequence[str],
    lines_used: list[dict],
    compact_payload: dict,
) -> WeightedAlongLinesResult:
    """Compute weighted along-line similarity from final lines and ownership arrays."""
    # Read the assignment dictionary from the local compact payload.
    column_assignment = compact_payload.get("column_assignment", {})
    # Read the number of reference windows stored when the scoring payload was built.
    reference_window_count = int(compact_payload.get("reference_window_count", len(reference_windows)) or 0)
    # Compute one v2.2-compatible line-level similarity record for every final line.
    records = compute_line_text_records(
        lines_used=lines_used,
        column_assignment=column_assignment,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
        reference_window_count=int(reference_window_count),
    )
    # Convert records into the v2.2-compatible weighted along-lines score.
    return weighted_result_from_records(records)


__all__ = [
    "LineTextFilterResult",
    "LineTextRecord",
    "WeightedAlongLinesResult",
    "clamp_unit_interval",
    "compute_line_text_record",
    "compute_line_text_records",
    "compute_weighted_along_lines_from_payload",
    "euclidean_line_length",
    "filter_lines_by_minimum_normalised_levenshtein",
    "reference_rows_for_levenshtein",
    "weighted_result_from_records",
]
