from __future__ import annotations

"""Runfile item selection for Hough tuning.

This module keeps lightweight `outputs.json` filtering separate from matrix
loading.  Selecting runfile records does not build score matrices, Hough
contexts, or text blocks, so it is safe to call before starting the RAM-heavy
document preparation stream.
"""

from collections.abc import Iterable
from pathlib import Path

try:
    from ..runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

# Make shared project helpers such as ``runfile_records.py`` importable when
# this module is loaded directly as ``matrices.runfile_selection``.
ensure_tuner_runtime_paths()

from runfile_records import load_run_items, same_file


def select_run_items_for_tuning(
    *,
    runfile_json: Path,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    selection_index_range: tuple[int, int] | None = None,
) -> list[dict]:
    """Select lightweight runfile records that should enter the tuner.

    ``selection_index_range`` is zero-based and inclusive.  It is applied after
    optional target-name filtering and after the optional ``max_items`` cap, so
    large runs can be split safely as chunks of the exact same selected set.
    The original runfile ``item["index"]`` value is preserved for reproducible
    Hough seeding and output provenance.
    """
    if max_items is not None and int(max_items) <= 0:
        raise ValueError("max_items must be positive when provided")
    if selection_index_range is not None:
        selection_start, selection_end = (int(selection_index_range[0]), int(selection_index_range[1]))
        if selection_start < 0 or selection_end < 0:
            raise ValueError("selection_index_range values must be non-negative")
        if selection_start > selection_end:
            raise ValueError("selection_index_range start must be <= end")
    if not Path(runfile_json).exists():
        raise FileNotFoundError(f"Missing runfile JSON: {runfile_json}")

    targets = [str(v) for v in (target_fnames or []) if str(v).strip()]
    selected_items: list[dict] = []

    for item in load_run_items(Path(runfile_json)):
        fname = str(item["fname"])
        if targets and not any(same_file(fname, target) for target in targets):
            continue
        if max_items is not None and len(selected_items) >= int(max_items):
            break
        selected_items.append(dict(item))

    matched_before_range_count = int(len(selected_items))
    if selection_index_range is not None:
        selection_start, selection_end = (int(selection_index_range[0]), int(selection_index_range[1]))
        # Python slices are end-exclusive, while the public CLI range is
        # inclusive because that is easier to reason about when splitting jobs.
        selected_items = selected_items[selection_start : selection_end + 1]

    if targets and matched_before_range_count == 0:
        raise KeyError(f"None of target_fnames were found in runfile: {targets!r}")
    if not selected_items:
        if selection_index_range is not None:
            selection_start, selection_end = (int(selection_index_range[0]), int(selection_index_range[1]))
            raise RuntimeError(
                "No documents selected for parameter sweep after applying "
                f"selection_index_range={selection_start}..{selection_end} "
                f"to {matched_before_range_count} pre-range selected documents."
            )
        raise RuntimeError("No documents selected for parameter sweep.")

    return selected_items


__all__ = ["select_run_items_for_tuning"]
