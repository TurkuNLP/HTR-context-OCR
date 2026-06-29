#!/usr/bin/env python3
"""Shared helpers for the "small document" (<10×10 matrix) tools.

These documents were skipped by the pipeline before Hough line detection, so they have no
detected lines. We fit ONE straight diagonal to the model's own Levenshtein alignment and draw
it on the score-matrix heatmap as a **visual aid only** — no metric is derived from it. All the
small-document metrics are character-level and computed in ``score_small_documents.py``.

Used by ``score_small_documents.py`` (only ``load_score_pkl_records``, for the prediction gate)
and ``plot_small_documents.py`` (``fit_alignment_line`` for the drawn diagonal).

Diagonal fit: anchors from ``equal``+``replace`` opcodes, then PCA / total least squares
(axis-agnostic, survives extreme aspect ratios).
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np

# Window stride the pipeline used for the w50s35 dev split. Small-doc score records do not store
# it, so callers fall back to this value when converting character indices into matrix windows.
DEFAULT_WINDOW_STRIDE = 35


# --------------------------------------------------------------------------- #
# Loading the progressively-pickled score dump
# --------------------------------------------------------------------------- #
def load_score_pkl_records(pkl_path: Path) -> dict[str, dict]:
    """Scan a score-matrix pickle and return one record per document, keyed by basename.

    The file is *progressively* pickled (records appended over time), so we read objects
    until EOF. Each record holds at least ``{'fname', 'scores', 'ref', 'pred'}``. If a
    document appears more than once the last record wins.
    """
    records: dict[str, dict] = {}
    with open(pkl_path, "rb") as handle:
        while True:
            try:
                item = pickle.load(handle)
            except EOFError:
                break
            if not isinstance(item, dict):
                continue
            fname = Path(str(item.get("fname", ""))).name
            if fname:
                records[fname] = item
    return records


# --------------------------------------------------------------------------- #
# Step 1 — turn the Levenshtein opcodes into window-space anchor points
# --------------------------------------------------------------------------- #
def alignment_anchor_points(ops: list[tuple], stride: int) -> list[tuple[float, float]]:
    """Project ``equal``/``replace`` opcodes onto matrix-window coordinates.

    Each returned point is ``(x, y) = (prediction_window, reference_window)`` — the same axes
    as the score matrix (rows = reference, columns = prediction). We sample roughly one point
    per window inside each aligned block, so longer aligned blocks contribute more points and
    naturally pull the fitted line harder (length weighting for free). ``insert``/``delete``
    opcodes are gaps in one text and would bend the straight line, so they are excluded.

    Opcode tuple is ``(tag, ref_start, ref_end, pred_start, pred_end)``.
    """
    step = max(1, int(stride))
    points: list[tuple[float, float]] = []
    for tag, ref_start, ref_end, pred_start, pred_end in ops:
        if tag not in ("equal", "replace"):
            continue
        ref_span = ref_end - ref_start
        if ref_span <= 0:
            continue
        pred_span = pred_end - pred_start
        # one reference char per window, plus the block's last char so the line reaches the end
        ref_chars = list(range(ref_start, ref_end, step))
        if ref_chars[-1] != ref_end - 1:
            ref_chars.append(ref_end - 1)
        for ref_char in ref_chars:
            # map this reference char to its aligned prediction char (proportional inside block)
            fraction = (ref_char - ref_start) / ref_span
            pred_char = pred_start + fraction * pred_span
            points.append((pred_char / step, ref_char / step))  # (x, y) in window units
    return points


# --------------------------------------------------------------------------- #
# Step 2 — fit a single straight diagonal with PCA (total least squares)
# --------------------------------------------------------------------------- #
def _clamp(value: float, upper: int) -> float:
    """Clamp a window coordinate to the valid range [0, upper-1]."""
    return min(max(value, 0.0), upper - 1)


def fit_alignment_line(
    ops: list[tuple], stride: int, n_rows: int, n_cols: int
) -> tuple[float, float, float, float] | None:
    """Fit ONE diagonal to the alignment anchors via PCA and return its endpoints.

    Returns ``(x0, y0, x1, y1)`` in window coordinates, or ``None`` when a diagonal is
    undefined: a 1-D / single-window matrix, or fewer than two anchor points. ``None`` tells
    callers to fall back to character-level metrics and to draw no line.
    """
    if n_rows < 2 or n_cols < 2:
        return None  # no room for a 2-D diagonal
    points = alignment_anchor_points(ops, stride)
    if len(points) < 2:
        return None

    cloud = np.asarray(points, dtype=float)          # shape (N, 2): columns are [x, y]
    centroid = cloud.mean(axis=0)
    centered = cloud - centroid
    # Principal direction = eigenvector of the largest eigenvalue of the 2×2 covariance.
    # eigh returns eigenvalues ascending, so the last column is the principal axis.
    covariance = np.cov(centered, rowvar=False)
    direction = np.linalg.eigh(covariance)[1][:, -1]
    # Endpoints = centroid plus the extreme projections of the anchors onto that direction.
    projections = centered @ direction
    start = centroid + projections.min() * direction
    end = centroid + projections.max() * direction
    return (
        _clamp(start[0], n_cols), _clamp(start[1], n_rows),
        _clamp(end[0], n_cols),   _clamp(end[1], n_rows),
    )
