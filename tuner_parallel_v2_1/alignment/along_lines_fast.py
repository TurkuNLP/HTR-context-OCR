from __future__ import annotations

import numpy as np

try:
    from ..cython_accel.optional_line_grouping import group_owned_columns_by_line
except ImportError:
    try:
        from cython_accel.optional_line_grouping import group_owned_columns_by_line  # type: ignore
    except ImportError:
        group_owned_columns_by_line = None  # type: ignore

try:
    from ..metrics.levenshtein_compat import normalized_levenshtein_similarity
except ImportError:
    from metrics.levenshtein_compat import normalized_levenshtein_similarity  # type: ignore


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
    similarity_fn=None,
    levenshtein_backend: str | None = None,
) -> tuple[float | None, int]:
    """Compute mean line-level normalized Levenshtein similarity without building verbose reports."""
    n_other = int(mapped_line_id.shape[0])
    n_ref = int(len(ref_blocks))

    # Group columns once in prediction-column order.  This preserves the exact
    # order produced by repeatedly calling np.flatnonzero(mapped_line_id == lid)
    # while avoiding one full mapped_line_id scan per possible line id.
    if group_owned_columns_by_line is None:
        owned_columns_by_line: list[list[int]] = [[] for _ in range(max(0, int(line_count_hint)))]
        for column_index, raw_line_id in enumerate(mapped_line_id):
            line_id = int(raw_line_id)
            if 0 <= line_id < len(owned_columns_by_line):
                owned_columns_by_line[line_id].append(int(column_index))
    else:
        owned_columns_by_line = group_owned_columns_by_line(mapped_line_id, int(line_count_hint))

    line_scores: list[float] = []
    for owned_cols in owned_columns_by_line:
        if not owned_cols:
            continue

        ref_rows = reference_rows_for_line(owned_cols, mapped_y, n_ref=n_ref)
        if not ref_rows:
            continue

        pred_line_text = "".join(pred_blocks[x] for x in owned_cols if 0 <= x < n_other)
        ref_line_text = "".join(ref_blocks[y] for y in ref_rows if 0 <= y < n_ref)

        if similarity_fn is None:
            score = float(
                normalized_levenshtein_similarity(
                    pred_line_text,
                    ref_line_text,
                    backend=str(levenshtein_backend),
                )
            )
        else:
            score = float(similarity_fn(pred_line_text, ref_line_text))
        line_scores.append(score)

    if not line_scores:
        return None, 0

    return float(sum(line_scores) / len(line_scores)), int(len(line_scores))


def compute_along_lines_similarity_with_backend(
    *,
    ref_blocks: list[str],
    pred_blocks: list[str],
    mapped_line_id: np.ndarray,
    mapped_y: np.ndarray,
    line_count_hint: int,
    levenshtein_backend: str,
) -> tuple[float | None, int]:
    """Compute along-lines similarity using one named Levenshtein backend.

    This wrapper avoids allocating a new lambda for every Hough combination
    while preserving the same normalized Levenshtein function and backend.
    """
    return compute_along_lines_similarity(
        ref_blocks=ref_blocks,
        pred_blocks=pred_blocks,
        mapped_line_id=mapped_line_id,
        mapped_y=mapped_y,
        line_count_hint=int(line_count_hint),
        levenshtein_backend=str(levenshtein_backend),
    )
