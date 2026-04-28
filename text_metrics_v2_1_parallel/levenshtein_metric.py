"""Levenshtein metrics for whole-document and line-guided comparisons.

This module now uses a single exact C-backed backend (rapidfuzz).
The payload schema is intentionally preserved.
"""

from __future__ import annotations

import numpy as np

from shared.ordered_sequence_helpers import (
    reference_rows_for_mapped_columns as _reference_rows_for_mapped_columns,
)

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein
except Exception:
    _RapidfuzzLevenshtein = None


BACKEND_C = "c"
SUPPORTED_BACKENDS = (BACKEND_C,)


def _levenshtein_distance_c(source: str, target: str) -> int:
    """Compute exact Levenshtein distance with rapidfuzz C backend."""
    if _RapidfuzzLevenshtein is None:
        raise RuntimeError(
            "Levenshtein backend 'c' requested, but rapidfuzz is unavailable in this environment."
        )
    return int(_RapidfuzzLevenshtein.distance(source, target))


def levenshtein_distance(source: str, target: str, *, backend: str = BACKEND_C) -> int:
    """Dispatch to the supported Levenshtein backend.

    Only backend ``c`` is supported in v2_1_parallel cleanup mode.
    """
    if backend != BACKEND_C:
        raise ValueError(
            f"Unsupported Levenshtein backend {backend!r}. Allowed: {SUPPORTED_BACKENDS!r}"
        )
    return _levenshtein_distance_c(source, target)


def normalized_levenshtein_similarity(
    predicted_text: str,
    gold_text: str,
    *,
    backend: str = BACKEND_C,
) -> float:
    """Return normalized Levenshtein similarity in ``[0, 1]``."""
    denom = max(len(predicted_text), len(gold_text))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(predicted_text, gold_text, backend=backend) / denom)


def build_stride_blocks(text: str, n_blocks: int, stride: int) -> list[str]:
    """Split text into contiguous stride-based blocks."""
    starts = [j * stride for j in range(n_blocks)]
    blocks = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if (j + 1) < n_blocks else len(text)
        start = min(start, len(text))
        end = min(max(end, start), len(text))
        blocks.append(text[start:end])
    return blocks


def _reference_rows_for_line(owned_cols: list[int], mapped_y: np.ndarray, n_ref: int) -> tuple[list[int], bool]:
    """Compute mapped reference rows for one line from ownership arrays."""
    return _reference_rows_for_mapped_columns(owned_cols, mapped_y, int(n_ref))


def _build_line_similarity_reports(
    *,
    other_text: str,
    ref_text: str,
    lines_used: list[dict],
    column_assignment: dict,
    window_stride: int,
    n_ref: int,
    n_other: int,
    backend: str,
) -> list[dict]:
    """Build per-line similarity reports from raw column-assignment arrays."""
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    if mapped_y.shape != (n_other,) or mapped_line_id.shape != (n_other,):
        raise ValueError(
            "column_assignment must provide mapped_y and mapped_line_id arrays with shape "
            f"({n_other},), got {mapped_y.shape} and {mapped_line_id.shape}"
        )

    other_blocks = build_stride_blocks(other_text, n_blocks=n_other, stride=window_stride)
    ref_blocks = build_stride_blocks(ref_text, n_blocks=n_ref, stride=window_stride)

    rows: list[dict] = []
    for lid, line in enumerate(lines_used):
        owned_cols = [int(x) for x in np.flatnonzero(mapped_line_id == lid)]
        if not owned_cols:
            continue

        ref_rows, ref_rows_reordered = _reference_rows_for_line(owned_cols, mapped_y, n_ref=n_ref)
        other_line_text = "".join(other_blocks[x] for x in owned_cols if 0 <= x < len(other_blocks))
        ref_line_text = "".join(ref_blocks[y] for y in ref_rows if 0 <= y < len(ref_blocks))
        score = normalized_levenshtein_similarity(other_line_text, ref_line_text, backend=backend)

        rows.append(
            {
                "line_id": int(lid),
                "normalized_levenshtein_similarity": float(score),
                "pred_text": other_line_text,
                "ref_text": ref_line_text,
                "pred_char_len": int(len(other_line_text)),
                "ref_char_len": int(len(ref_line_text)),
                "owned_column_count": int(len(owned_cols)),
                "pred_column_start": int(owned_cols[0]),
                "pred_column_end": int(owned_cols[-1]),
                "mapped_ref_row_count": int(len(ref_rows)),
                "mapped_ref_row_start": None if not ref_rows else int(ref_rows[0]),
                "mapped_ref_row_end": None if not ref_rows else int(ref_rows[-1]),
                "mapped_ref_rows": ref_rows,
                "ref_rows_reordered_for_monotonicity": bool(ref_rows_reordered),
                "x0": float(line.get("x0", 0.0)),
                "y0": float(line.get("y0", 0.0)),
                "x1": float(line.get("x1", 0.0)),
                "y1": float(line.get("y1", 0.0)),
                "score": float(line.get("score", 0.0)),
                "length": float(line.get("length", 0.0)),
                "support": float(line.get("support", 0.0)),
                "owned_cols": int(line.get("owned_cols", len(owned_cols))),
                "owned_fraction": float(line.get("owned_fraction", 0.0)),
                "owned_score_mean": float(line.get("owned_score_mean", 0.0)),
                "owned_mask_hits": int(line.get("owned_mask_hits", 0)),
                "owned_mask_fraction": float(line.get("owned_mask_fraction", 0.0)),
                "anchor_y": float(line.get("anchor_y", min(line.get("y0", 0.0), line.get("y1", 0.0)))),
            }
        )

    return rows


def _build_line_similarity_reports_from_bundle(
    *,
    other_text: str,
    ref_text: str,
    lines_used: list[dict],
    bundle: dict,
    backend: str,
) -> list[dict]:
    """Build per-line similarity reports from precomputed line bundles."""
    n_other = int(bundle.get("n_other_windows", 0))
    n_ref = int(bundle.get("n_ref_windows", 0))
    stride = int(bundle.get("window_stride", 1))

    other_blocks = build_stride_blocks(other_text, n_blocks=n_other, stride=stride)
    ref_blocks = build_stride_blocks(ref_text, n_blocks=n_ref, stride=stride)
    lines_by_id = {int(i): ln for i, ln in enumerate(lines_used)}

    rows: list[dict] = []
    for entry in bundle.get("lines", []):
        lid = int(entry.get("line_id", -1))
        owned_cols = [int(v) for v in entry.get("x_window_ids_owned", [])]
        ref_rows = [int(v) for v in entry.get("y_window_ids_for_levenshtein", [])]
        if not owned_cols:
            continue

        line = lines_by_id.get(lid, {})
        other_line_text = "".join(other_blocks[x] for x in owned_cols if 0 <= x < len(other_blocks))
        ref_line_text = "".join(ref_blocks[y] for y in ref_rows if 0 <= y < len(ref_blocks))
        score = normalized_levenshtein_similarity(other_line_text, ref_line_text, backend=backend)

        rows.append(
            {
                "line_id": int(lid),
                "normalized_levenshtein_similarity": float(score),
                "pred_text": other_line_text,
                "ref_text": ref_line_text,
                "pred_char_len": int(len(other_line_text)),
                "ref_char_len": int(len(ref_line_text)),
                "owned_column_count": int(len(owned_cols)),
                "pred_column_start": int(owned_cols[0]),
                "pred_column_end": int(owned_cols[-1]),
                "mapped_ref_row_count": int(len(ref_rows)),
                "mapped_ref_row_start": None if not ref_rows else int(ref_rows[0]),
                "mapped_ref_row_end": None if not ref_rows else int(ref_rows[-1]),
                "mapped_ref_rows": ref_rows,
                "ref_rows_reordered_for_monotonicity": bool(entry.get("y_rows_reordered_for_monotonicity", False)),
                "x0": float(line.get("x0", 0.0)),
                "y0": float(line.get("y0", 0.0)),
                "x1": float(line.get("x1", 0.0)),
                "y1": float(line.get("y1", 0.0)),
                "score": float(line.get("score", 0.0)),
                "length": float(line.get("length", 0.0)),
                "support": float(line.get("support", 0.0)),
                "owned_cols": int(line.get("owned_cols", len(owned_cols))),
                "owned_fraction": float(line.get("owned_fraction", 0.0)),
                "owned_score_mean": float(line.get("owned_score_mean", 0.0)),
                "owned_mask_hits": int(line.get("owned_mask_hits", 0)),
                "owned_mask_fraction": float(line.get("owned_mask_fraction", 0.0)),
                "anchor_y": float(line.get("anchor_y", min(line.get("y0", 0.0), line.get("y1", 0.0)))),
            }
        )

    return rows


def compute_levenshtein_metrics(
    *,
    ref_text: str,
    other_text: str,
    lines_used: list[dict],
    column_assignment: dict,
    n_ref: int,
    n_other: int,
    window_stride: int,
    backend: str = BACKEND_C,
) -> dict:
    """Compute whole-document and line-guided Levenshtein metrics."""
    whole = float(normalized_levenshtein_similarity(other_text, ref_text, backend=backend))
    line_reports = _build_line_similarity_reports(
        other_text=other_text,
        ref_text=ref_text,
        lines_used=lines_used,
        column_assignment=column_assignment,
        window_stride=window_stride,
        n_ref=n_ref,
        n_other=n_other,
        backend=backend,
    )

    line_scores = [row["normalized_levenshtein_similarity"] for row in line_reports]
    along_lines = None if not line_scores else float(sum(line_scores) / len(line_scores))

    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    if mapped_line_id.shape != (n_other,):
        raise ValueError(
            f"column_assignment['mapped_line_id'] shape must be ({n_other},), got {mapped_line_id.shape}"
        )

    return {
        "whole_document_normalized_levenshtein_similarity": whole,
        "document_normalized_levenshtein_similarity_along_lines": along_lines,
        "line_count": int(len(line_reports)),
        "line_guided_columns": int(np.sum(mapped_line_id >= 0)),
        "fallback_columns": int(np.sum(mapped_line_id < 0)),
        "lines": line_reports,
        "levenshtein_backend": backend,
    }


def compute_levenshtein_metrics_from_bundle(
    *,
    ref_text: str,
    other_text: str,
    lines_used: list[dict],
    bundle: dict,
    backend: str = BACKEND_C,
) -> dict:
    """Compute Levenshtein metrics from a precomputed line bundle."""
    whole = float(normalized_levenshtein_similarity(other_text, ref_text, backend=backend))
    line_reports = _build_line_similarity_reports_from_bundle(
        other_text=other_text,
        ref_text=ref_text,
        lines_used=lines_used,
        bundle=bundle,
        backend=backend,
    )

    line_scores = [row["normalized_levenshtein_similarity"] for row in line_reports]
    along_lines = None if not line_scores else float(sum(line_scores) / len(line_scores))

    return {
        "whole_document_normalized_levenshtein_similarity": whole,
        "document_normalized_levenshtein_similarity_along_lines": along_lines,
        "line_count": int(len(line_reports)),
        "line_guided_columns": int(bundle.get("line_guided_columns", 0)),
        "fallback_columns": int(bundle.get("fallback_columns", 0)),
        "lines": line_reports,
        "levenshtein_backend": backend,
    }
