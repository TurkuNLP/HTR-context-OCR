from __future__ import annotations

"""Text-score raw falling Hough segments before true-IoU merging."""

from dataclasses import dataclass
import math
import time
from typing import Any, Sequence

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein.alignment.hough_segment_endpoint_records import (
    line_records_from_raw_hough_segments,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.filtering.filter_candidate_coverages import build_line_coverage
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.levenshtein import normalized_levenshtein_similarity
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.line_text_similarity import compute_line_text_record

# Fast-path accelerators: call sample_line_path directly so coverage_from_sampled_path,
# build_single_raw_line_assignment, and compute_line_text_record are all bypassed.
try:
    from tuner_simple_alpha_sweep_pre_iou_levenshtein.filtering import (
        filter_cython_accelerators as _filter_accel,
    )
    _fast_sample_line_path = _filter_accel.accelerated_sample_line_path
    _fast_unique_rows = getattr(_filter_accel, "accelerated_unique_reference_rows_from_path_slice", None)
except ImportError:
    _fast_sample_line_path = None
    _fast_unique_rows = None

RawHoughSegment = tuple[tuple[float, float], tuple[float, float]]


@dataclass(frozen=True)
class RawHoughLineTextFilterResult:
    """Result of filtering raw falling Hough segments by line-level text similarity."""

    filtered_segments: list[RawHoughSegment]
    line_score_records: list[dict[str, Any]]
    filter_enabled: bool
    threshold: float | None
    input_line_count: int
    surviving_line_count: int
    removed_line_count: int
    score_minimum: float | None
    score_maximum: float | None
    score_mean: float | None
    seconds: float


def build_single_raw_line_assignment(coverage: dict, prediction_column_count: int) -> dict[str, np.ndarray]:
    """Assign every sampled prediction column to one temporary raw line."""

    mapped_y = np.full(int(prediction_column_count), np.nan, dtype=float)
    mapped_line_id = np.full(int(prediction_column_count), -1, dtype=int)
    for prediction_column, reference_row in coverage.get("x_to_y", {}).items():
        prediction_column_index = int(prediction_column)
        if 0 <= prediction_column_index < int(prediction_column_count):
            mapped_y[prediction_column_index] = float(reference_row)
            mapped_line_id[prediction_column_index] = 0
    return {"mapped_y": mapped_y, "mapped_line_id": mapped_line_id}


def summarize_finite_scores(scores: Sequence[float]) -> tuple[float | None, float | None, float | None]:
    """Return minimum, maximum, and mean for finite raw-line scores."""

    finite_scores = [float(score) for score in scores if math.isfinite(float(score))]
    if not finite_scores:
        return None, None, None
    return min(finite_scores), max(finite_scores), sum(finite_scores) / float(len(finite_scores))


def filter_raw_hough_segments_by_line_levenshtein(
    *,
    score_matrix: np.ndarray,
    raw_segments: Sequence[RawHoughSegment],
    reference_windows: Sequence[str],
    prediction_windows: Sequence[str],
    reference_window_count: int,
    minimum_line_nls: float | None,
) -> RawHoughLineTextFilterResult:
    """Remove raw falling Hough segments whose own text similarity is too weak."""

    started_at = time.perf_counter()
    matrix = np.asarray(score_matrix, dtype=float)
    original_segments = list(raw_segments)
    input_line_count = int(len(original_segments))

    if minimum_line_nls is None:
        return RawHoughLineTextFilterResult(
            filtered_segments=original_segments,
            line_score_records=[],
            filter_enabled=False,
            threshold=None,
            input_line_count=input_line_count,
            surviving_line_count=input_line_count,
            removed_line_count=0,
            score_minimum=None,
            score_maximum=None,
            score_mean=None,
            seconds=float(time.perf_counter() - started_at),
        )

    if matrix.ndim != 2 or matrix.size == 0 or not original_segments:
        return RawHoughLineTextFilterResult(
            filtered_segments=[],
            line_score_records=[],
            filter_enabled=True,
            threshold=float(minimum_line_nls),
            input_line_count=input_line_count,
            surviving_line_count=0,
            removed_line_count=input_line_count,
            score_minimum=None,
            score_maximum=None,
            score_mean=None,
            seconds=float(time.perf_counter() - started_at),
        )

    prediction_column_count = int(matrix.shape[1])
    converted_line_records = line_records_from_raw_hough_segments(matrix, original_segments)
    kept_segments: list[RawHoughSegment] = []
    audit_records: list[dict[str, Any]] = []
    scored_values: list[float] = []
    n_pred_windows = len(prediction_windows)
    n_ref_windows = int(reference_window_count)

    for raw_line_id, line_record in enumerate(converted_line_records):
        line_score = None
        owned_prediction_column_count = 0
        mapped_reference_row_count = 0

        if _fast_sample_line_path is not None:
            # Fast path: call Cython sampler directly, bypass coverage_from_sampled_path,
            # build_single_raw_line_assignment, and compute_line_text_record entirely.
            sampled_path = _fast_sample_line_path(
                matrix,
                x0=float(line_record["x0"]),
                y0=float(line_record["y0"]),
                x1=float(line_record["x1"]),
                y1=float(line_record["y1"]),
            )
            if sampled_path is not None:
                pred_min = int(sampled_path["pred_min"])
                pred_max = int(sampled_path["pred_max"])
                sampled_rows = sampled_path["sampled_reference_rows"]
                col_start = max(0, pred_min)
                col_end = min(n_pred_windows - 1, pred_max)
                if col_start <= col_end and sampled_rows:
                    owned_count = col_end - col_start + 1
                    row_start_idx = col_start - pred_min
                    if _fast_unique_rows is not None:
                        unique_rows = _fast_unique_rows(
                            sampled_rows,
                            row_start_idx,
                            row_start_idx + owned_count,
                            n_ref_windows,
                        )
                    else:
                        rows_slice = sampled_rows[row_start_idx:row_start_idx + owned_count]
                        seen: set[int] = set()
                        unique_rows = []
                        for r in rows_slice:
                            ri = int(r)
                            if 0 <= ri < n_ref_windows and ri not in seen:
                                seen.add(ri)
                                unique_rows.append(ri)
                    if unique_rows:
                        pred_text = "".join(
                            str(prediction_windows[c]) for c in range(col_start, col_end + 1)
                        )
                        ref_text = "".join(str(reference_windows[r]) for r in unique_rows)
                        raw_nls = float(normalized_levenshtein_similarity(pred_text, ref_text))
                        if math.isfinite(raw_nls):
                            line_score = max(0.0, min(1.0, raw_nls))
                            owned_prediction_column_count = owned_count
                            mapped_reference_row_count = len(unique_rows)
                            scored_values.append(float(line_score))
        else:
            # Slow path: build full coverage object, then use generic text-record helpers.
            line_record_with_id = dict(line_record)
            line_record_with_id["raw_line_id"] = int(raw_line_id)
            coverage = build_line_coverage(line_record_with_id, matrix)
            if coverage is not None:
                temporary_assignment = build_single_raw_line_assignment(
                    coverage,
                    prediction_column_count=prediction_column_count,
                )
                text_record = compute_line_text_record(
                    line_id=0,
                    line_record=coverage["line"],
                    column_assignment=temporary_assignment,
                    reference_windows=reference_windows,
                    prediction_windows=prediction_windows,
                    reference_window_count=int(reference_window_count),
                )
                if text_record is not None:
                    line_score = float(text_record.normalized_levenshtein_similarity)
                    owned_prediction_column_count = int(text_record.owned_prediction_column_count)
                    mapped_reference_row_count = int(text_record.mapped_reference_row_count)
                    scored_values.append(float(line_score))

        line_passed = line_score is not None and float(line_score) >= float(minimum_line_nls)
        if line_passed:
            kept_segments.append(original_segments[int(raw_line_id)])

        audit_records.append(
            {
                "raw_line_id": int(raw_line_id),
                "segment": original_segments[int(raw_line_id)],
                "line_nls": line_score,
                "passed": bool(line_passed),
                "owned_prediction_column_count": int(owned_prediction_column_count),
                "mapped_reference_row_count": int(mapped_reference_row_count),
            }
        )

    score_minimum, score_maximum, score_mean = summarize_finite_scores(scored_values)
    return RawHoughLineTextFilterResult(
        filtered_segments=kept_segments,
        line_score_records=audit_records,
        filter_enabled=True,
        threshold=float(minimum_line_nls),
        input_line_count=input_line_count,
        surviving_line_count=int(len(kept_segments)),
        removed_line_count=int(input_line_count - len(kept_segments)),
        score_minimum=score_minimum,
        score_maximum=score_maximum,
        score_mean=score_mean,
        seconds=float(time.perf_counter() - started_at),
    )


__all__ = [
    "RawHoughLineTextFilterResult",
    "filter_raw_hough_segments_by_line_levenshtein",
]
