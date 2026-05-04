from __future__ import annotations

import numpy as np


def build_stride_blocks(text: str, n_blocks: int, stride: int) -> list[str]:
    """Build stride-based text blocks once and reuse them across evaluations."""
    starts = [int(j) * int(stride) for j in range(int(n_blocks))]
    blocks: list[str] = []
    for j, start in enumerate(starts):
        end = starts[j + 1] if (j + 1) < len(starts) else len(text)
        start = min(start, len(text))
        end = min(max(end, start), len(text))
        blocks.append(text[start:end])
    return blocks


def _ordered_unique(values: list[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue in seen:
            continue
        seen.add(ivalue)
        out.append(ivalue)
    return out


def _is_non_decreasing(values: list[int]) -> bool:
    return all(a <= b for a, b in zip(values, values[1:]))


def reference_rows_for_line(owned_cols: list[int], mapped_y: np.ndarray, n_ref: int) -> list[int]:
    """Map owned prediction columns to reference rows with the same monotonicity rule as metrics."""
    if int(n_ref) <= 0:
        return []

    rows = [
        int(np.clip(round(float(mapped_y[x])), 0, int(n_ref) - 1))
        for x in owned_cols
        if 0 <= int(x) < mapped_y.shape[0] and np.isfinite(mapped_y[x])
    ]
    if not rows:
        return []

    unique_rows = _ordered_unique(rows)
    if _is_non_decreasing(unique_rows):
        return unique_rows
    return sorted(set(unique_rows))


def compute_along_lines_similarity(
    *,
    ref_blocks: list[str],
    pred_blocks: list[str],
    mapped_line_id: np.ndarray,
    mapped_y: np.ndarray,
    line_count_hint: int,
    similarity_fn,
) -> tuple[float | None, int]:
    """Compute mean line-level normalized Levenshtein similarity without building verbose reports."""
    n_other = int(mapped_line_id.shape[0])
    n_ref = int(len(ref_blocks))

    line_scores: list[float] = []
    for lid in range(max(0, int(line_count_hint))):
        owned_cols = [int(x) for x in np.flatnonzero(mapped_line_id == int(lid))]
        if not owned_cols:
            continue

        ref_rows = reference_rows_for_line(owned_cols, mapped_y, n_ref=n_ref)
        if not ref_rows:
            continue

        pred_line_text = "".join(pred_blocks[x] for x in owned_cols if 0 <= x < n_other)
        ref_line_text = "".join(ref_blocks[y] for y in ref_rows if 0 <= y < n_ref)

        score = float(similarity_fn(pred_line_text, ref_line_text))
        line_scores.append(score)

    if not line_scores:
        return None, 0

    return float(sum(line_scores) / len(line_scores)), int(len(line_scores))
