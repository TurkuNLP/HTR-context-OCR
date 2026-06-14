from __future__ import annotations

"""Numerical equivalence test: fast path vs slow path for the pre-IoU text filter."""

import numpy as np
import pytest

from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.raw_hough_line_text_filter import (
    _fast_sample_line_path,
    _fast_unique_rows,
    build_single_raw_line_assignment,
    filter_raw_hough_segments_by_line_levenshtein,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.filtering.filter_candidate_coverages import build_line_coverage
from tuner_simple_alpha_sweep_pre_iou_levenshtein.alignment.hough_segment_endpoint_records import (
    line_records_from_raw_hough_segments,
)
from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.line_text_similarity import compute_line_text_record
from tuner_simple_alpha_sweep_pre_iou_levenshtein.cython_accel.optional_filtering import (
    unique_reference_rows_from_path_slice,
)


def make_diagonal_matrix(n_ref: int = 20, n_pred: int = 30, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.3, 0.7, size=(n_ref, n_pred)).astype(np.float64)
    for i in range(n_pred):
        row = min(n_ref - 1, i * (n_ref - 1) // (n_pred - 1))
        matrix[row, i] = 0.95
    return matrix


def slow_path_score(
    segment,
    matrix: np.ndarray,
    reference_windows,
    prediction_windows,
    reference_window_count: int,
):
    """Compute NLS through the original slow path for one segment."""
    line_records = line_records_from_raw_hough_segments(matrix, [segment])
    if not line_records:
        return None, 0, 0
    line_record = dict(line_records[0])
    line_record["raw_line_id"] = 0
    coverage = build_line_coverage(line_record, matrix)
    if coverage is None:
        return None, 0, 0
    assignment = build_single_raw_line_assignment(coverage, prediction_column_count=matrix.shape[1])
    text_record = compute_line_text_record(
        line_id=0,
        line_record=coverage["line"],
        column_assignment=assignment,
        reference_windows=reference_windows,
        prediction_windows=prediction_windows,
        reference_window_count=reference_window_count,
    )
    if text_record is None:
        return None, 0, 0
    return (
        float(text_record.normalized_levenshtein_similarity),
        int(text_record.owned_prediction_column_count),
        int(text_record.mapped_reference_row_count),
    )


def test_cython_accelerators_available():
    assert _fast_sample_line_path is not None, "fast sampler should be available"
    assert _fast_unique_rows is not None, "fast dedup should be available"


def test_unique_reference_rows_basic():
    # Basic deduplication preserving first-seen order
    result = unique_reference_rows_from_path_slice([5, 3, 5, 2, 3, 8], 0, 6, 10)
    assert result == [5, 3, 2, 8]


def test_unique_reference_rows_slice():
    result = unique_reference_rows_from_path_slice([5, 3, 5, 2, 3, 8], 1, 4, 10)
    assert result == [3, 5, 2]


def test_unique_reference_rows_out_of_range():
    # Rows >= reference_window_count must be skipped
    result = unique_reference_rows_from_path_slice([0, 99, 2, 1], 0, 4, 5)
    assert result == [0, 2, 1]


def test_unique_reference_rows_empty():
    assert unique_reference_rows_from_path_slice([], 0, 0, 10) == []
    assert unique_reference_rows_from_path_slice([1, 2, 3], 3, 3, 10) == []
    assert unique_reference_rows_from_path_slice([1, 2, 3], 0, 0, 0) == []


def test_fast_path_matches_slow_path_on_diagonal():
    """Fast path must produce identical NLS, owned count, and ref row count as slow path."""
    matrix = make_diagonal_matrix(n_ref=20, n_pred=30)
    n_ref_windows, n_pred_windows = matrix.shape
    ref_windows = [f"reference_window_{i}" for i in range(n_ref_windows)]
    pred_windows = [f"prediction_window_{i}" for i in range(n_pred_windows)]

    segments = [
        ((0.0, 0.0), (29.0, 19.0)),
        ((5.0, 3.0), (20.0, 12.0)),
        ((10.0, 6.0), (29.0, 14.0)),
    ]

    for segment in segments:
        slow_nls, slow_owned, slow_ref = slow_path_score(
            segment, matrix, ref_windows, pred_windows, n_ref_windows
        )
        fast_result = filter_raw_hough_segments_by_line_levenshtein(
            score_matrix=matrix,
            raw_segments=[segment],
            reference_windows=ref_windows,
            prediction_windows=pred_windows,
            reference_window_count=n_ref_windows,
            minimum_line_nls=0.0,
        )
        fast_record = fast_result.line_score_records[0]
        fast_nls = fast_record["line_nls"]
        fast_owned = fast_record["owned_prediction_column_count"]
        fast_ref_rows = fast_record["mapped_reference_row_count"]

        assert fast_owned == slow_owned, (
            f"Owned count mismatch for {segment}: fast={fast_owned} slow={slow_owned}"
        )
        assert fast_ref_rows == slow_ref, (
            f"Ref row count mismatch for {segment}: fast={fast_ref_rows} slow={slow_ref}"
        )
        if slow_nls is not None and fast_nls is not None:
            assert abs(fast_nls - slow_nls) < 1e-9, (
                f"NLS mismatch for {segment}: fast={fast_nls} slow={slow_nls}"
            )


def test_fast_path_filter_passes_good_segments():
    matrix = make_diagonal_matrix(n_ref=20, n_pred=30)
    n_ref_windows, n_pred_windows = matrix.shape
    ref_windows = [f"ref_{i}" for i in range(n_ref_windows)]
    pred_windows = [f"pred_{i}" for i in range(n_pred_windows)]

    good_segment = ((0.0, 0.0), (29.0, 19.0))
    result = filter_raw_hough_segments_by_line_levenshtein(
        score_matrix=matrix,
        raw_segments=[good_segment],
        reference_windows=ref_windows,
        prediction_windows=pred_windows,
        reference_window_count=n_ref_windows,
        minimum_line_nls=0.5,
    )
    assert result.filter_enabled is True
    # NLS of identical-text windows should be 1.0 (ref_i vs pred_i — different strings)
    # but the filter should at least produce a score
    assert result.line_score_records[0]["line_nls"] is not None


def test_fast_path_no_threshold_returns_all():
    matrix = make_diagonal_matrix()
    segments = [((0.0, 0.0), (15.0, 10.0)), ((5.0, 3.0), (20.0, 12.0))]
    ref_windows = ["word"] * 20
    pred_windows = ["word"] * 30
    result = filter_raw_hough_segments_by_line_levenshtein(
        score_matrix=matrix,
        raw_segments=segments,
        reference_windows=ref_windows,
        prediction_windows=pred_windows,
        reference_window_count=20,
        minimum_line_nls=None,
    )
    assert result.filter_enabled is False
    assert result.surviving_line_count == len(segments)
