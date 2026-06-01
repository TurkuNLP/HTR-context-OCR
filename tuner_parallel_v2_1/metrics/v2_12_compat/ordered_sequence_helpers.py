from __future__ import annotations

"""Ordered sequence helpers copied from v2.12 for tuner-local metric semantics.

Source:
`text_metrics_v2_12_parallel/shared/ordered_sequence_helpers.py`

Copied on: 2026-05-25.
"""

import numpy as np


def ordered_unique(values: list[int]) -> list[int]:
    """Return values in first-seen order without duplicates."""
    out: list[int] = []
    seen: set[int] = set()
    for value in values:
        ivalue = int(value)
        if ivalue in seen:
            continue
        out.append(ivalue)
        seen.add(ivalue)
    return out


def is_non_decreasing(values: list[int]) -> bool:
    """Return True when the sequence is monotonic non-decreasing."""
    return all(a <= b for a, b in zip(values, values[1:]))


def reference_rows_for_mapped_columns(
    owned_cols: list[int],
    mapped_y: np.ndarray,
    n_ref_windows: int,
) -> tuple[list[int], bool]:
    """Build ordered reference-row ids for one line from mapped columns."""
    if int(n_ref_windows) <= 0:
        return [], False

    rows = [
        int(np.clip(round(float(mapped_y[x])), 0, int(n_ref_windows) - 1))
        for x in owned_cols
        if 0 <= int(x) < mapped_y.shape[0] and np.isfinite(mapped_y[x])
    ]
    if not rows:
        return [], False

    unique_rows = ordered_unique(rows)
    if is_non_decreasing(unique_rows):
        return unique_rows, False

    return sorted(set(unique_rows)), True


__all__ = [
    "is_non_decreasing",
    "ordered_unique",
    "reference_rows_for_mapped_columns",
]
