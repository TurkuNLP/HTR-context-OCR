from __future__ import annotations

"""Timing comparison: fast Cython path vs slow Python path for pre-IoU text filter."""

import time

import numpy as np

from tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.raw_hough_line_text_filter import (
    filter_raw_hough_segments_by_line_levenshtein,
)
import tuner_simple_alpha_sweep_pre_iou_levenshtein.scoring.raw_hough_line_text_filter as _filt_mod


def _make_segments_and_matrix(n_ref: int, n_pred: int, n_segments: int, seed: int = 7):
    rng = np.random.default_rng(seed)
    matrix = rng.uniform(0.3, 0.9, size=(n_ref, n_pred)).astype(np.float64)
    for i in range(n_pred):
        row = min(n_ref - 1, i * (n_ref - 1) // (n_pred - 1))
        matrix[row, i] = 0.95
    segments = [
        (
            (float(rng.integers(0, n_pred // 2)), float(rng.integers(0, n_ref // 2))),
            (float(rng.integers(n_pred // 2, n_pred)), float(rng.integers(n_ref // 2, n_ref))),
        )
        for _ in range(n_segments)
    ]
    return matrix, segments


def test_fast_path_is_faster_than_slow_path():
    n_ref, n_pred, n_seg = 80, 200, 100
    matrix, segments = _make_segments_and_matrix(n_ref, n_pred, n_seg)
    ref_windows = [f"ref_{i}" for i in range(n_ref)]
    pred_windows = [f"pred_{i}" for i in range(n_pred)]
    kwargs = dict(
        score_matrix=matrix,
        raw_segments=segments,
        reference_windows=ref_windows,
        prediction_windows=pred_windows,
        reference_window_count=n_ref,
        minimum_line_nls=0.0,
    )

    # Warm up
    for _ in range(3):
        filter_raw_hough_segments_by_line_levenshtein(**kwargs)

    n_reps = 30

    # Fast path (Cython active)
    t0 = time.perf_counter()
    for _ in range(n_reps):
        filter_raw_hough_segments_by_line_levenshtein(**kwargs)
    fast_ms = (time.perf_counter() - t0) / n_reps * 1000

    # Slow path (disable fast sampler)
    orig = _filt_mod._fast_sample_line_path
    _filt_mod._fast_sample_line_path = None
    try:
        t0 = time.perf_counter()
        for _ in range(n_reps):
            filter_raw_hough_segments_by_line_levenshtein(**kwargs)
        slow_ms = (time.perf_counter() - t0) / n_reps * 1000
    finally:
        _filt_mod._fast_sample_line_path = orig

    speedup = slow_ms / fast_ms
    print(
        f"\n  fast={fast_ms:.2f}ms  slow={slow_ms:.2f}ms  "
        f"speedup={speedup:.1f}x  ({n_seg} segments)"
    )
    assert speedup >= 1.5, f"Expected at least 1.5x speedup, got {speedup:.2f}x"
