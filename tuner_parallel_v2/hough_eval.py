from __future__ import annotations

"""Per-combination Hough evaluation and ranking helpers."""

import math
import time

import numpy as np

try:
    from .along_lines_fast import compute_along_lines_similarity
    from .line_alignment_pipeline_fast import detect_lines_only_from_hough_ctx, filter_lines_after_hough
    from .tuner_config import (
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )
except ImportError:
    from along_lines_fast import compute_along_lines_similarity  # type: ignore
    from line_alignment_pipeline_fast import detect_lines_only_from_hough_ctx, filter_lines_after_hough  # type: ignore
    from tuner_config import (  # type: ignore
        HoughBaselineConfig,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SweepDocument,
    )

from levenshtein_metric import normalized_levenshtein_similarity


def is_finite_along_lines(value) -> bool:
    """Check if along-lines NLS is a finite numeric value."""
    if value is None:
        return False
    try:
        return bool(math.isfinite(float(value)))
    except Exception:
        return False


def evaluation_rank_key(row: dict) -> tuple[float, int, int, int, int, int, int]:
    """Ranking key for choosing best evaluation (primary: along-lines NLS)."""
    return (
        float(row.get("along_lines_nls", float("-inf"))),
        int(row.get("line_guided_columns", 0)),
        -int(row.get("fallback_columns", 0)),
        -int(row.get(PARAM_HOUGH_THRESHOLD, 0)),
        -int(row.get(PARAM_HOUGH_LINE_LENGTH, 0)),
        -int(row.get(PARAM_HOUGH_LINE_GAP, 0)),
        -int(row.get(PARAM_HOUGH_SEED, 0)),
    )


def pick_better_eval(current: dict | None, candidate: dict | None) -> dict | None:
    """Return the better of two evaluation rows according to deterministic ranking."""
    if candidate is None:
        return current
    can_ok = is_finite_along_lines(candidate.get("along_lines_nls"))
    if current is None:
        return candidate if can_ok else None

    cur_ok = is_finite_along_lines(current.get("along_lines_nls"))
    if not can_ok:
        return current
    if not cur_ok:
        return candidate
    if evaluation_rank_key(candidate) > evaluation_rank_key(current):
        return candidate
    return current


def evaluate_single_combination(
    *,
    doc: SweepDocument,
    cfg: HoughBaselineConfig,
    levenshtein_backend: str,
) -> dict:
    """Evaluate one (threshold, line_length, line_gap, seed) configuration."""
    eval_started_at = time.perf_counter()

    t_detect = time.perf_counter()
    det = detect_lines_only_from_hough_ctx(
        hough_ctx=doc.hough_ctx,
        seed=int(cfg.hough_seed) + int(doc.index),
        threshold=int(cfg.hough_threshold),
        line_length=int(cfg.hough_line_length),
        line_gap=int(cfg.hough_line_gap),
    )
    detect_seconds = time.perf_counter() - t_detect

    t_filter = time.perf_counter()
    filtered = filter_lines_after_hough(
        matrix=doc.matrix,
        det_result=det,
        align_abs_min_len=float(cfg.align_abs_min_len),
        align_min_iou_threshold=float(cfg.align_min_iou_threshold),
    )
    filter_seconds = time.perf_counter() - t_filter

    column_assignment = filtered["column_assignment"]
    mapped_line_id = np.asarray(column_assignment["mapped_line_id"], dtype=int)
    mapped_y = np.asarray(column_assignment["mapped_y"], dtype=float)

    t_lev = time.perf_counter()
    similarity_fn = lambda pred_line, ref_line: normalized_levenshtein_similarity(
        pred_line,
        ref_line,
        backend=levenshtein_backend,
    )
    along_lines_nls, used_line_count = compute_along_lines_similarity(
        ref_blocks=doc.ref_blocks,
        pred_blocks=doc.pred_blocks,
        mapped_line_id=mapped_line_id,
        mapped_y=mapped_y,
        line_count_hint=len(filtered["lines_used"]),
        similarity_fn=similarity_fn,
    )
    levenshtein_seconds = time.perf_counter() - t_lev

    line_guided_columns = int(np.sum(mapped_line_id >= 0)) if mapped_line_id.size else 0
    fallback_columns = int(np.sum(mapped_line_id < 0)) if mapped_line_id.size else 0

    detect_filter_seconds = float(detect_seconds + filter_seconds)

    return {
        "along_lines_nls": None if along_lines_nls is None else float(along_lines_nls),
        "line_count": int(used_line_count),
        "used_line_count": int(len(filtered["lines_used"])),
        "line_guided_columns": int(line_guided_columns),
        "fallback_columns": int(fallback_columns),
        "raw_line_count": int(len(det.get("raw_lines", []))),
        "merged_line_count": int(len(det.get("merged_lines", []))),
        "threshold_start": float(det.get("threshold_start", float("nan"))),
        "timing_hough_detect_seconds": float(detect_seconds),
        "timing_filter_seconds": float(filter_seconds),
        "timing_detect_filter_seconds": float(detect_filter_seconds),
        "timing_build_bundle_seconds": 0.0,
        "timing_levenshtein_seconds": float(levenshtein_seconds),
        "timing_total_seconds": float(time.perf_counter() - eval_started_at),
    }


__all__ = [
    "is_finite_along_lines",
    "evaluation_rank_key",
    "pick_better_eval",
    "evaluate_single_combination",
]
