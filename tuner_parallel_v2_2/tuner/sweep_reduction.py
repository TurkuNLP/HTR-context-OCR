from __future__ import annotations

"""Best-row reduction helpers for exhaustive Hough sweeps.

Threshold workers return compact best-row maps.  These helpers merge those maps
with the single shared ranking rule from :mod:`tuner.hough_eval`, keeping
tie-breaking behavior in one place.
"""

from .hough_eval import DEFAULT_SELECTION_OBJECTIVE, pick_better_eval


def empty_best_value_map(values: list[int]) -> dict[int, dict | None]:
    """Return a deterministic value-to-best-row mapping initialized to ``None``."""
    return {int(value): None for value in values}


def merge_best_value_payloads(
    target: dict[int, dict | None],
    source: dict[int, dict | None],
    *,
    selection_objective: str = DEFAULT_SELECTION_OBJECTIVE,
) -> None:
    """Merge source best rows into target using the selected tuner objective."""
    for value, candidate_row in source.items():
        normalized_value = int(value)
        target[normalized_value] = pick_better_eval(
            target[normalized_value],
            candidate_row,
            selection_objective=selection_objective,
        )


__all__ = [
    "empty_best_value_map",
    "merge_best_value_payloads",
]

