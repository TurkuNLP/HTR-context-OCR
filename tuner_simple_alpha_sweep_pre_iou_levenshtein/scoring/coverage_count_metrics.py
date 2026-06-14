from __future__ import annotations

"""V2.12-compatible character coverage counts for the public alignment metrics."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CoverageCountMetricResult:
    """Coverage metrics plus diagnostics from reference-axis count subtraction."""

    correct_ref_coverage: float | None
    missing_ref_coverage: float | None
    repetition_on_reference: float | None
    hallucination: float | None
    invalid_reason: str | None
    invalid_error_message: str | None
    diagnostics: dict


def clamp_unit_interval(value: float) -> float:
    """Return a finite value clipped into the interval from zero to one."""
    value = float(value)
    if not np.isfinite(value):
        return 0.0
    return max(0.0, min(1.0, value))


def number_of_windows_for_text_length(text_length: int, window_size: int, window_stride: int) -> int:
    """Return how many sliding text windows exist for one text length."""
    text_length = int(text_length)
    window_size = int(window_size)
    window_stride = int(window_stride)
    if text_length < window_size:
        return 0
    return ((text_length - window_size) // window_stride) + 1


def normalized_line_endpoints(line_record: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return one line as ``((x0, y0), (x1, y1))`` using matrix coordinates."""
    try:
        return (
            (float(line_record["x0"]), float(line_record["y0"])),
            (float(line_record["x1"]), float(line_record["y1"])),
        )
    except KeyError as exc:
        raise ValueError(f"Line dictionary is missing endpoint field: {exc}") from exc


def line_window_ids_from_endpoint(
    line_record: dict,
    *,
    x_window_count: int,
    y_window_count: int,
) -> tuple[list[int], list[int]]:
    """Sample a line segment and return the window ids it crosses on both axes."""
    if int(x_window_count) <= 0 and int(y_window_count) <= 0:
        return [], []

    (x0, y0), (x1, y1) = normalized_line_endpoints(line_record)
    step_count = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    step_count = max(step_count, 1)

    sampled_x_positions = np.rint(np.linspace(x0, x1, step_count)).astype(int)
    sampled_y_positions = np.rint(np.linspace(y0, y1, step_count)).astype(int)

    x_window_ids: list[int] = []
    y_window_ids: list[int] = []
    if int(x_window_count) > 0:
        sampled_x_positions = np.clip(sampled_x_positions, 0, int(x_window_count) - 1)
        x_window_ids = sorted(set(int(value) for value in sampled_x_positions.tolist()))
    if int(y_window_count) > 0:
        sampled_y_positions = np.clip(sampled_y_positions, 0, int(y_window_count) - 1)
        y_window_ids = sorted(set(int(value) for value in sampled_y_positions.tolist()))
    return x_window_ids, y_window_ids


def window_ids_to_merged_character_intervals(
    window_ids: list[int],
    *,
    text_length: int,
    window_size: int,
    window_stride: int,
) -> list[tuple[int, int]]:
    """Convert text-window ids into merged character intervals ``[start, end)``."""
    if not window_ids or int(text_length) <= 0:
        return []

    raw_intervals: list[tuple[int, int]] = []
    for window_id in sorted(set(int(value) for value in window_ids)):
        start = int(window_id) * int(window_stride)
        end = min(start + int(window_size), int(text_length))
        if start >= int(text_length) or end <= start:
            continue
        raw_intervals.append((int(start), int(end)))
    if not raw_intervals:
        return []

    merged_intervals: list[tuple[int, int]] = [raw_intervals[0]]
    for start, end in raw_intervals[1:]:
        previous_start, previous_end = merged_intervals[-1]
        if start <= previous_end:
            merged_intervals[-1] = (previous_start, max(previous_end, end))
        else:
            merged_intervals.append((start, end))
    return merged_intervals


def accumulate_character_counts(
    *,
    text_length: int,
    interval_groups: list[list[tuple[int, int]]],
) -> np.ndarray:
    """Return one per-character count array from many line interval groups."""
    if int(text_length) <= 0:
        return np.zeros(0, dtype=np.int32)

    difference_array = np.zeros(int(text_length) + 1, dtype=np.int64)
    for intervals in interval_groups:
        for start, end in intervals:
            interval_start = max(0, min(int(start), int(text_length)))
            interval_end = max(0, min(int(end), int(text_length)))
            if interval_end <= interval_start:
                continue
            difference_array[interval_start] += 1
            difference_array[interval_end] -= 1
    return np.cumsum(difference_array[:-1], dtype=np.int64).astype(np.int32)


def build_coverage_line_entries(scoring_payload: dict) -> list[dict]:
    """Build the compact line interval entries used by v2.12 coverage counts."""
    reference_text_length = int(scoring_payload.get("reference_text_length", 0))
    other_text_length = int(scoring_payload.get("other_text_length", 0))
    window_size = int(scoring_payload.get("window_size", 1))
    window_stride = int(scoring_payload.get("window_stride", 1))
    reference_window_count = number_of_windows_for_text_length(reference_text_length, window_size, window_stride)
    other_window_count = number_of_windows_for_text_length(other_text_length, window_size, window_stride)

    line_entries: list[dict] = []
    for line_id, line_record in enumerate(scoring_payload.get("lines_used", []) or []):
        x_window_ids, y_window_ids = line_window_ids_from_endpoint(
            line_record,
            x_window_count=other_window_count,
            y_window_count=reference_window_count,
        )
        x_intervals = window_ids_to_merged_character_intervals(
            x_window_ids,
            text_length=other_text_length,
            window_size=window_size,
            window_stride=window_stride,
        )
        y_intervals = window_ids_to_merged_character_intervals(
            y_window_ids,
            text_length=reference_text_length,
            window_size=window_size,
            window_stride=window_stride,
        )
        line_entries.append(
            {
                "line_id": int(line_id),
                "x_char_intervals_for_coverage": x_intervals,
                "y_char_intervals_for_coverage": y_intervals,
            }
        )
    return line_entries


def interval_groups_from_line_entries(line_entries: list[dict], key: str) -> list[list[tuple[int, int]]]:
    """Read one interval-list field from each line entry."""
    interval_groups: list[list[tuple[int, int]]] = []
    for line_entry in line_entries:
        intervals = line_entry.get(key, [])
        interval_groups.append([(int(start), int(end)) for start, end in intervals])
    return interval_groups


def reference_axis_counts_from_payload(scoring_payload: dict) -> np.ndarray:
    """Return per-reference-character counts from the lines in one payload."""
    line_entries = build_coverage_line_entries(scoring_payload)
    return accumulate_character_counts(
        text_length=int(scoring_payload.get("reference_text_length", 0)),
        interval_groups=interval_groups_from_line_entries(line_entries, "y_char_intervals_for_coverage"),
    )


def prediction_axis_counts_from_payload(scoring_payload: dict) -> np.ndarray:
    """Return per-prediction-character counts from the ref-to-pred lines."""
    line_entries = build_coverage_line_entries(scoring_payload)
    return accumulate_character_counts(
        text_length=int(scoring_payload.get("other_text_length", 0)),
        interval_groups=interval_groups_from_line_entries(line_entries, "x_char_intervals_for_coverage"),
    )


def reference_self_counts_with_minimum_diagonal(
    ref_to_ref_scoring_payload: dict | None,
    *,
    reference_axis_template: np.ndarray,
) -> np.ndarray:
    """Return ref-to-ref counts where every reference character covers itself at least once."""
    minimum_self_counts = np.ones_like(reference_axis_template, dtype=np.int32)
    if ref_to_ref_scoring_payload is None:
        return minimum_self_counts

    observed_self_counts = reference_axis_counts_from_payload(ref_to_ref_scoring_payload)
    if observed_self_counts.shape != minimum_self_counts.shape:
        return observed_self_counts

    # Preserve repeated self-coverage above one, but never let missing Hough evidence erase the identity baseline.
    return np.maximum(observed_self_counts, minimum_self_counts).astype(np.int32, copy=False)


def y_difference_diagnostics(y_difference: np.ndarray) -> dict:
    """Return the same compact reference-axis subtraction diagnostics as v2.2."""
    y_difference = np.asarray(y_difference, dtype=np.int32)
    if y_difference.size == 0:
        return {
            "coverage_y_diff_size": 0,
            "coverage_y_diff_min": None,
            "coverage_y_diff_max": None,
            "coverage_y_diff_le_minus_one_count": 0,
            "coverage_y_diff_lt_minus_one_count": 0,
            "coverage_y_diff_below_minus_one_counts_json": {},
        }

    unique_values, unique_counts = np.unique(y_difference, return_counts=True)
    below_minus_one_counts = {
        str(int(value)): int(count)
        for value, count in zip(unique_values, unique_counts)
        if int(value) < -1
    }
    return {
        "coverage_y_diff_size": int(y_difference.size),
        "coverage_y_diff_min": int(np.min(y_difference)),
        "coverage_y_diff_max": int(np.max(y_difference)),
        "coverage_y_diff_le_minus_one_count": int(np.count_nonzero(y_difference <= -1)),
        "coverage_y_diff_lt_minus_one_count": int(np.count_nonzero(y_difference < -1)),
        "coverage_y_diff_below_minus_one_counts_json": below_minus_one_counts,
    }


def compute_coverage_count_metrics(
    *,
    ref_to_pred_scoring_payload: dict,
    ref_to_ref_scoring_payload: dict | None,
) -> CoverageCountMetricResult:
    """Compute public coverage metrics with the v2.2/v2.12 count-subtraction rule."""
    other_y = reference_axis_counts_from_payload(ref_to_pred_scoring_payload)
    other_x = prediction_axis_counts_from_payload(ref_to_pred_scoring_payload)

    refref_y = reference_self_counts_with_minimum_diagonal(
        ref_to_ref_scoring_payload,
        reference_axis_template=other_y,
    )

    if refref_y.shape != other_y.shape:
        raise ValueError(
            "Reference-axis count arrays must have the same length, "
            f"got {refref_y.shape[0]} and {other_y.shape[0]}"
        )

    y_difference = np.subtract(other_y, refref_y)
    diagnostics = y_difference_diagnostics(y_difference)
    if int(diagnostics["coverage_y_diff_lt_minus_one_count"]) > 0:
        return CoverageCountMetricResult(
            correct_ref_coverage=None,
            missing_ref_coverage=None,
            repetition_on_reference=None,
            hallucination=None,
            invalid_reason="coverage_y_diff_below_minus_one",
            invalid_error_message=(
                "Found reference-axis subtraction values below -1; "
                "ref-to-ref repeated reference coverage more often than ref-to-pred explained."
            ),
            diagnostics=diagnostics,
        )

    reference_character_count = int(y_difference.size)
    if reference_character_count <= 0:
        missing_ref_coverage = 0.0
        correct_ref_coverage = 0.0
        repetition_on_reference = 0.0
    else:
        missing_ref_coverage = np.count_nonzero(y_difference == -1) / float(reference_character_count)
        correct_ref_coverage = np.count_nonzero(y_difference == 0) / float(reference_character_count)
        repetition_on_reference = np.count_nonzero(y_difference > 0) / float(reference_character_count)

    prediction_character_count = int(other_x.size)
    if prediction_character_count <= 0:
        hallucination = 0.0
    else:
        hallucination = np.count_nonzero(other_x == 0) / float(prediction_character_count)

    return CoverageCountMetricResult(
        correct_ref_coverage=clamp_unit_interval(correct_ref_coverage),
        missing_ref_coverage=clamp_unit_interval(missing_ref_coverage),
        repetition_on_reference=clamp_unit_interval(repetition_on_reference),
        hallucination=clamp_unit_interval(hallucination),
        invalid_reason=None,
        invalid_error_message=None,
        diagnostics=diagnostics,
    )


__all__ = [
    "CoverageCountMetricResult",
    "compute_coverage_count_metrics",
    "reference_self_counts_with_minimum_diagonal",
]
