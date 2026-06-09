"""Levenshtein metrics for whole-document and line-guided comparisons.

This module now exposes only the bundle-based metric path used by the report
pipeline. The exact metric formulas and output fields stay unchanged.
"""

from __future__ import annotations

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein
except Exception:
    _RapidfuzzLevenshtein = None


# Keep the backend label as fixed provenance in reports.
LEVENSHTEIN_BACKEND = "c"



def _levenshtein_distance_c(source: str, target: str) -> int:
    """Compute exact Levenshtein distance with the rapidfuzz C backend."""
    if _RapidfuzzLevenshtein is None:
        raise RuntimeError(
            "Levenshtein backend 'c' is unavailable because rapidfuzz is not installed in this environment."
        )
    return int(_RapidfuzzLevenshtein.distance(source, target))



def levenshtein_distance(source: str, target: str) -> int:
    """Return exact Levenshtein distance using the fixed C-backed implementation."""
    return _levenshtein_distance_c(source, target)



def normalized_levenshtein_similarity(predicted_text: str, gold_text: str) -> float:
    """Return normalized Levenshtein similarity in ``[0, 1]``.

    The normalization formula is preserved exactly:
    ``1 - distance / max(len(pred), len(gold))``.
    """
    denom = max(len(predicted_text), len(gold_text))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(predicted_text, gold_text) / denom)



def build_stride_blocks(text: str, n_blocks: int, stride: int) -> list[str]:
    """Split text into contiguous stride-based blocks used by line metrics."""
    starts = [j * stride for j in range(n_blocks)]
    blocks = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if (j + 1) < n_blocks else len(text)
        start = min(start, len(text))
        end = min(max(end, start), len(text))
        blocks.append(text[start:end])
    return blocks



def _build_line_similarity_reports_from_bundle(
    *,
    other_text: str,
    ref_text: str,
    lines_used: list[dict],
    bundle: dict,
) -> list[dict]:
    """Build per-line similarity reports from a precomputed line bundle.

    The bundle already contains the exact x-window ownership and the reference
    y-window order chosen for Levenshtein. That means we can compute the metric
    rows without reusing the older raw column-assignment path.
    """
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
        score = normalized_levenshtein_similarity(other_line_text, ref_line_text)

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



def compute_levenshtein_metrics_from_bundle(
    *,
    ref_text: str,
    other_text: str,
    lines_used: list[dict],
    bundle: dict,
) -> dict:
    """Compute whole-document and line-guided Levenshtein metrics.

    This is the only public Levenshtein entry point used by the report pipeline.
    """
    whole = float(normalized_levenshtein_similarity(other_text, ref_text))
    line_reports = _build_line_similarity_reports_from_bundle(
        other_text=other_text,
        ref_text=ref_text,
        lines_used=lines_used,
        bundle=bundle,
    )

    line_scores = [row["normalized_levenshtein_similarity"] for row in line_reports]
    weighted_line_score_sum = 0.0
    weighted_line_length_sum = 0.0
    for row in line_reports:
        # Use the final filtered geometric line length as the same weighting
        # signal used by the tuner-side weighted along-lines objective.
        line_length = float(row.get("length", 0.0))
        # Ignore non-positive lengths so a degenerate line cannot skew the mean.
        if line_length <= 0.0:
            continue
        weighted_line_score_sum += float(row["normalized_levenshtein_similarity"]) * line_length
        weighted_line_length_sum += line_length
    along_lines = None if not line_scores else float(sum(line_scores) / len(line_scores))
    weighted_along_lines = (
        None
        if weighted_line_length_sum <= 0.0
        else float(weighted_line_score_sum / weighted_line_length_sum)
    )

    return {
        "whole_document_normalized_levenshtein_similarity": whole,
        "document_normalized_levenshtein_similarity_along_lines": along_lines,
        "document_weighted_normalized_levenshtein_similarity_along_lines": weighted_along_lines,
        "line_count": int(len(line_reports)),
        "line_guided_columns": int(bundle.get("line_guided_columns", 0)),
        "fallback_columns": int(bundle.get("fallback_columns", 0)),
        "lines": line_reports,
        "levenshtein_backend": LEVENSHTEIN_BACKEND,
    }
