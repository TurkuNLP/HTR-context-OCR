from __future__ import annotations

"""Local read-only indexing for streamed score-matrix pickle files."""

from dataclasses import dataclass
import pickle
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ScoreIndexLoadResult:
    """The in-memory filename-to-offset index for one score pickle stream."""

    # Store the lookup table keyed by basename, with byte offsets that can reload one document record later.
    index_by_fname: dict[str, dict]
    # Store how this index was created so output logs can explain the loading path.
    source: str


@dataclass(frozen=True)
class ScoreMatrixLoadResult:
    """The result of trying to load one validated matrix from a score pickle stream."""

    # Store the loaded numeric matrix, or None when validation failed and fallback computation is needed.
    matrix: np.ndarray | None
    # Store a short source label for audit output.
    source_desc: str
    # Store the validation failure reason, or None when loading succeeded.
    reason: str | None


def count_windows_from_text_length(text_length: int, *, window_size: int, window_stride: int) -> int:
    """Return how many sliding windows exist for a text length and window geometry."""
    # Convert all values to integers so arithmetic and comparisons are stable even when callers pass NumPy scalars.
    text_length = int(text_length)
    # Convert window_size for the same reason.
    window_size = int(window_size)
    # Convert window_stride for the same reason.
    window_stride = int(window_stride)
    # Text shorter than one full window has zero score-matrix windows.
    if text_length < window_size:
        # Return zero because no complete window can be created.
        return 0
    # Count the first full window plus every stride-sized step that still leaves a complete window.
    return int(((text_length - window_size) // window_stride) + 1)


def coerce_score_matrix(raw_scores: object) -> np.ndarray:
    """Convert a raw pickle score field into a two-dimensional float matrix."""
    # Convert lists, tuples, and existing arrays into one NumPy array with floating-point values.
    matrix = np.asarray(raw_scores, dtype=float)
    # Require two dimensions because Hough expects rows for reference windows and columns for prediction windows.
    if matrix.ndim != 2:
        # Raise a clear error that the caller can turn into a fallback reason.
        raise ValueError(f"score matrix must be two-dimensional, got shape {tuple(matrix.shape)!r}")
    # Return a contiguous copy so downstream numeric code sees a predictable array layout.
    return np.ascontiguousarray(matrix, dtype=float)


def build_score_stream_index(scores_pkl: Path) -> dict[str, dict]:
    """Scan a streamed pickle once and remember where each document record starts."""
    # Normalize the path at the boundary of this helper.
    scores_pkl = Path(scores_pkl)
    # Store filename-to-offset records in a dictionary for direct document lookup later.
    lookup: dict[str, dict] = {}
    # Open the stream in binary mode because pickle offsets are byte offsets.
    with scores_pkl.open("rb") as handle:
        # Count records in stream order so logs and debug output can refer to the original position.
        stream_index = 0
        # Keep reading pickle objects until EOFError signals the end of the stream.
        while True:
            # Remember the byte offset before loading the next record so we can seek back to it later.
            offset = int(handle.tell())
            # Attempt to load exactly one pickle object from the stream.
            try:
                # Load one record; score pickle files are streams of dict records, not one giant dictionary.
                item = pickle.load(handle)
            # EOFError is the normal end-of-file signal for a pickle stream.
            except EOFError:
                # Stop scanning once there are no more records.
                break
            # Ignore malformed non-dictionary records instead of crashing the whole index build.
            if not isinstance(item, dict):
                # Advance the stream counter even for ignored records so positions remain faithful to the file.
                stream_index += 1
                # Continue with the next pickle object.
                continue
            # Use only the basename because runfile document names and pickle records may contain different directory prefixes.
            fname = Path(str(item.get("fname", f"item_{stream_index:04d}"))).name
            # Duplicate filenames would make direct lookup ambiguous, so fail loudly.
            if fname in lookup:
                # Raise an explicit error because silently picking one duplicate would corrupt document matching.
                raise ValueError(f"Duplicate fname in score pickle stream: {fname!r}")
            # Store only lightweight metadata and the byte offset, not the full score matrix.
            lookup[fname] = {
                "stream_index": int(stream_index),
                "offset": int(offset),
                "fname": str(fname),
                "has_ref": "ref" in item,
                "has_pred": "pred" in item,
                "ref": str(item.get("ref", "")) if "ref" in item else None,
                "pred": str(item.get("pred", "")) if "pred" in item else None,
            }
            # Advance the record counter after storing this record.
            stream_index += 1
    # Return the lightweight lookup table for later document-level loading.
    return lookup


def load_score_stream_index_readonly(scores_pkl: Path) -> ScoreIndexLoadResult:
    """Build a read-only in-memory index for one score pickle stream."""
    # Build the index without writing cache files because tuner_simple keeps output structure minimal.
    index_by_fname = build_score_stream_index(Path(scores_pkl))
    # Return the index with an explicit source label for logging and audit tables.
    return ScoreIndexLoadResult(index_by_fname=index_by_fname, source="in_memory_build")


def load_score_item_by_offset(scores_pkl: Path, offset: int) -> dict:
    """Seek to one pickle record and load exactly that document item."""
    # Open the pickle stream in binary mode so seeking uses byte offsets.
    with Path(scores_pkl).open("rb") as handle:
        # Move directly to the offset captured during index construction.
        handle.seek(int(offset))
        # Load exactly one pickle object from that offset.
        item = pickle.load(handle)
    # Validate the record shape before returning it to matrix-specific code.
    if not isinstance(item, dict):
        # Raise a clear error because the index promised a dictionary record at this offset.
        raise ValueError(f"Expected dict record at offset {offset} in {scores_pkl}")
    # Return the raw record for strict text and shape validation.
    return item


def load_matrix_from_scores_pkl_index_readonly(
    *,
    scores_pkl: Path,
    score_index_by_fname: dict[str, dict],
    fname: str,
    expected_ref_text: str,
    expected_pred_text: str,
    window_size: int,
    window_stride: int,
) -> ScoreMatrixLoadResult:
    """Load and strictly validate one score matrix from a pre-indexed pickle stream."""
    # Normalize the requested document name to match the index key style.
    base_name = Path(str(fname)).name
    # Look up the document metadata and byte offset captured during index construction.
    index_item = score_index_by_fname.get(base_name)
    # Missing index entries mean the caller must compute a fallback matrix from text.
    if index_item is None:
        # Return a structured failure reason instead of raising.
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="index_miss")
    # Try to seek directly to the document record.
    try:
        # Load the raw pickle record from the stored byte offset.
        raw_record = load_score_item_by_offset(Path(scores_pkl), int(index_item["offset"]))
    # Any offset or pickle error should trigger fallback computation for this document.
    except Exception as exc:
        # Return the exception representation as the audit reason.
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason=f"offset_read_error:{exc!r}")
    # Score records must include both reference and comparison text for strict validation.
    if "ref" not in raw_record or "pred" not in raw_record:
        # Return a structured reason so the fallback path can log exactly what happened.
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="missing_ref_or_pred")
    # Normalize the stored reference text to a string before comparison.
    raw_ref_text = str(raw_record.get("ref", ""))
    # Normalize the stored comparison text to a string before comparison.
    raw_pred_text = str(raw_record.get("pred", ""))
    # Reject records whose reference text does not match the runfile text.
    if raw_ref_text != str(expected_ref_text):
        # Return a mismatch reason because using the matrix would compare the wrong text.
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="ref_text_mismatch")
    # Reject records whose prediction/self text does not match the expected comparison text.
    if raw_pred_text != str(expected_pred_text):
        # Return a mismatch reason because using the matrix would compare the wrong text.
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="pred_text_mismatch")
    # Try to coerce the raw scores into a numeric two-dimensional matrix.
    try:
        # Convert the matrix only after text validation succeeds.
        matrix = coerce_score_matrix(raw_record.get("scores"))
    # Any malformed score array should trigger fallback computation.
    except Exception as exc:
        # Return the coercion error as a readable reason.
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason=f"coerce_error:{exc!r}")
    # Compute the shape implied by the validated reference text and window settings.
    expected_shape = (
        count_windows_from_text_length(len(raw_ref_text), window_size=int(window_size), window_stride=int(window_stride)),
        count_windows_from_text_length(len(raw_pred_text), window_size=int(window_size), window_stride=int(window_stride)),
    )
    # Reject matrices whose shape does not match the text-window geometry requested for this run.
    if tuple(matrix.shape) != tuple(expected_shape):
        # Return the expected and actual shapes to make debugging straightforward.
        return ScoreMatrixLoadResult(
            matrix=None,
            source_desc="scores_pkl",
            reason=f"shape_mismatch:got={tuple(matrix.shape)} expected={tuple(expected_shape)}",
        )
    # Return the validated matrix to the caller.
    return ScoreMatrixLoadResult(matrix=matrix, source_desc="scores_pkl", reason=None)


# Declare the public helpers that other tuner_simple modules may import.
__all__ = [
    "ScoreIndexLoadResult",
    "ScoreMatrixLoadResult",
    "build_score_stream_index",
    "coerce_score_matrix",
    "count_windows_from_text_length",
    "load_matrix_from_scores_pkl_index_readonly",
    "load_score_item_by_offset",
    "load_score_stream_index_readonly",
]
