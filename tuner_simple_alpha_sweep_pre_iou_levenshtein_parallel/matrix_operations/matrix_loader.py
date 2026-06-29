from __future__ import annotations

"""Load score matrices from `.pkl` files, with Levenshtein fallback computation."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .matrix_fallback_computation import compute_levenshtein_score_matrix

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.matrix_operations.score_pkl_index import (
    load_matrix_from_scores_pkl_index_readonly,
    load_score_stream_index_readonly,
)

# Compute or store LogFn so later code can reuse this named value clearly.
LogFn = Callable[[str], None]


# Ask Python to generate common data-container methods for the class defined next.
@dataclass(frozen=True)
# Define the ScoreMatrixIndexBundle class, which groups related state and behavior for this part of the pipeline.
class ScoreMatrixIndexBundle:
    """Run-level indexes for both score-matrix streams."""

    # Define the ref_to_pred_index field so this data object records that value explicitly.
    ref_to_pred_index: dict[str, dict]
    # Define the ref_to_ref_index field so this data object records that value explicitly.
    ref_to_ref_index: dict[str, dict]
    # Define the ref_to_pred_index_source field so this data object records that value explicitly.
    ref_to_pred_index_source: str
    # Define the ref_to_ref_index_source field so this data object records that value explicitly.
    ref_to_ref_index_source: str


# Ask Python to generate common data-container methods for the class defined next.
@dataclass(frozen=True)
# Define the LoadedScoreMatrix class, which groups related state and behavior for this part of the pipeline.
class LoadedScoreMatrix:
    """One matrix plus provenance information for audit tables."""

    # Define the matrix field so this data object records that value explicitly.
    matrix: np.ndarray
    # Define the source field so this data object records that value explicitly.
    source: str
    # Define the reason field so this data object records that value explicitly.
    reason: str | None


# Define the build_score_matrix_indexes function; its body below performs one named step of the pipeline.
def build_score_matrix_indexes(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the ref_to_pred_scores_pkl field so this data object records that value explicitly.
    ref_to_pred_scores_pkl: Path,
    # Define the ref_to_ref_scores_pkl field so this data object records that value explicitly.
    ref_to_ref_scores_pkl: Path,
    # Define the log field so this data object records that value explicitly.
    log: LogFn,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> ScoreMatrixIndexBundle:
    """Build read-only score-stream indexes once for the whole run."""
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[matrix-index] building ref-to-pred index: {ref_to_pred_scores_pkl}")
    # Compute or store ref_to_pred_result so later code can reuse this named value clearly.
    ref_to_pred_result = load_score_stream_index_readonly(Path(ref_to_pred_scores_pkl))
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        # Provide this literal text value to the surrounding path, message, or argument definition.
        "[matrix-index] ref-to-pred index ready "
        # Compute or store f"source so later code can reuse this named value clearly.
        f"source={ref_to_pred_result.source} records={len(ref_to_pred_result.index_by_fname)}"
    )

    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[matrix-index] building ref-to-ref index: {ref_to_ref_scores_pkl}")
    # Compute or store ref_to_ref_result so later code can reuse this named value clearly.
    ref_to_ref_result = load_score_stream_index_readonly(Path(ref_to_ref_scores_pkl))
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(
        # Provide this literal text value to the surrounding path, message, or argument definition.
        "[matrix-index] ref-to-ref index ready "
        # Compute or store f"source so later code can reuse this named value clearly.
        f"source={ref_to_ref_result.source} records={len(ref_to_ref_result.index_by_fname)}"
    )

    # Return this computed value to the caller so the next pipeline stage can use it.
    return ScoreMatrixIndexBundle(
        # Pass the ref_to_pred_index argument into the surrounding call so the callee receives that setting explicitly.
        ref_to_pred_index=ref_to_pred_result.index_by_fname,
        # Pass the ref_to_ref_index argument into the surrounding call so the callee receives that setting explicitly.
        ref_to_ref_index=ref_to_ref_result.index_by_fname,
        # Pass the ref_to_pred_index_source argument into the surrounding call so the callee receives that setting explicitly.
        ref_to_pred_index_source=str(ref_to_pred_result.source),
        # Pass the ref_to_ref_index_source argument into the surrounding call so the callee receives that setting explicitly.
        ref_to_ref_index_source=str(ref_to_ref_result.source),
    )


# Define the _load_or_compute_matrix function; its body below performs one named step of the pipeline.
def _load_or_compute_matrix(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the scores_pkl field so this data object records that value explicitly.
    scores_pkl: Path,
    # Define the score_index_by_fname field so this data object records that value explicitly.
    score_index_by_fname: dict[str, dict],
    # Define the fname field; it stores the document filename used to match runfile records to score matrices.
    fname: str,
    # Define the expected_ref_text field so this data object records that value explicitly.
    expected_ref_text: str,
    # Define the expected_other_text field so this data object records that value explicitly.
    expected_other_text: str,
    # Define the window_size field; it stores the number of text characters represented by one score-matrix window.
    window_size: int,
    # Define the window_stride field; it stores how many characters the sliding window moves between neighboring matrix cells.
    window_stride: int,
    # Define the fallback_label field so this data object records that value explicitly.
    fallback_label: str,
    # Define the log field so this data object records that value explicitly.
    log: LogFn,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> LoadedScoreMatrix:
    """Use a precomputed matrix when valid, otherwise compute a Levenshtein matrix."""
    # Compute or store load_result so later code can reuse this named value clearly.
    load_result = load_matrix_from_scores_pkl_index_readonly(
        # Pass the scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
        scores_pkl=Path(scores_pkl),
        # Pass the score_index_by_fname argument into the surrounding call so the callee receives that setting explicitly.
        score_index_by_fname=score_index_by_fname,
        # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
        fname=str(fname),
        # Pass the expected_ref_text argument into the surrounding call so the callee receives that setting explicitly.
        expected_ref_text=str(expected_ref_text),
        # Pass the expected_pred_text argument into the surrounding call so the callee receives that setting explicitly.
        expected_pred_text=str(expected_other_text),
        # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
        window_size=int(window_size),
        # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
        window_stride=int(window_stride),
    )
    # Check whether load_result.matrix is not None; the indented block handles that specific case.
    if load_result.matrix is not None:
        # Return this computed value to the caller so the next pipeline stage can use it.
        return LoadedScoreMatrix(matrix=np.asarray(load_result.matrix, dtype=float), source="scores_pkl", reason=None)

    # Compute or store reason so later code can reuse this named value clearly.
    reason = str(load_result.reason or "unknown_pkl_load_failure")
    # Write a progress message so long runs are understandable from terminal or Slurm output.
    log(f"[matrix-load] {fallback_label} fallback for {fname}: {reason}")
    # Compute or store computed_matrix so later code can reuse this named value clearly.
    computed_matrix = compute_levenshtein_score_matrix(
        # Pass reference_text into the surrounding call; this supplies the normalized reference transcription for this document.
        reference_text=str(expected_ref_text),
        # Pass the other_text argument into the surrounding call so the callee receives that setting explicitly.
        other_text=str(expected_other_text),
        # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
        window_size=int(window_size),
        # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
        window_stride=int(window_stride),
    )
    # Return this computed value to the caller so the next pipeline stage can use it.
    return LoadedScoreMatrix(
        # Pass the matrix argument into the surrounding call so the callee receives that setting explicitly.
        matrix=computed_matrix,
        # Pass the source argument into the surrounding call so the callee receives that setting explicitly.
        source="computed_missing_or_invalid_pkl",
        # Pass the reason argument into the surrounding call so the callee receives that setting explicitly.
        reason=reason,
    )


# Define the load_or_compute_ref_to_pred_matrix function; its body below performs one named step of the pipeline.
def load_or_compute_ref_to_pred_matrix(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the scores_pkl field so this data object records that value explicitly.
    scores_pkl: Path,
    # Define the score_index_by_fname field so this data object records that value explicitly.
    score_index_by_fname: dict[str, dict],
    # Define the fname field; it stores the document filename used to match runfile records to score matrices.
    fname: str,
    # Define the reference_text field; it stores the normalized reference transcription for this document.
    reference_text: str,
    # Define the prediction_text field; it stores the normalized model prediction for this document.
    prediction_text: str,
    # Define the window_size field; it stores the number of text characters represented by one score-matrix window.
    window_size: int,
    # Define the window_stride field; it stores how many characters the sliding window moves between neighboring matrix cells.
    window_stride: int,
    # Define the log field so this data object records that value explicitly.
    log: LogFn,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> LoadedScoreMatrix:
    """Load or compute the reference-to-prediction score matrix."""
    # Return this computed value to the caller so the next pipeline stage can use it.
    return _load_or_compute_matrix(
        # Pass the scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
        scores_pkl=scores_pkl,
        # Pass the score_index_by_fname argument into the surrounding call so the callee receives that setting explicitly.
        score_index_by_fname=score_index_by_fname,
        # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
        fname=fname,
        # Pass the expected_ref_text argument into the surrounding call so the callee receives that setting explicitly.
        expected_ref_text=reference_text,
        # Pass the expected_other_text argument into the surrounding call so the callee receives that setting explicitly.
        expected_other_text=prediction_text,
        # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
        window_size=window_size,
        # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
        window_stride=window_stride,
        # Pass the fallback_label argument into the surrounding call so the callee receives that setting explicitly.
        fallback_label="ref_to_pred",
        # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
        log=log,
    )


# Define the load_or_compute_ref_to_ref_matrix function; its body below performs one named step of the pipeline.
def load_or_compute_ref_to_ref_matrix(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the scores_pkl field so this data object records that value explicitly.
    scores_pkl: Path,
    # Define the score_index_by_fname field so this data object records that value explicitly.
    score_index_by_fname: dict[str, dict],
    # Define the fname field; it stores the document filename used to match runfile records to score matrices.
    fname: str,
    # Define the reference_text field; it stores the normalized reference transcription for this document.
    reference_text: str,
    # Define the window_size field; it stores the number of text characters represented by one score-matrix window.
    window_size: int,
    # Define the window_stride field; it stores how many characters the sliding window moves between neighboring matrix cells.
    window_stride: int,
    # Define the log field so this data object records that value explicitly.
    log: LogFn,
# Execute this statement as the next small step in the surrounding pipeline logic.
) -> LoadedScoreMatrix:
    """Load or compute the reference-to-reference score matrix."""
    # Return this computed value to the caller so the next pipeline stage can use it.
    return _load_or_compute_matrix(
        # Pass the scores_pkl argument into the surrounding call so the callee receives that setting explicitly.
        scores_pkl=scores_pkl,
        # Pass the score_index_by_fname argument into the surrounding call so the callee receives that setting explicitly.
        score_index_by_fname=score_index_by_fname,
        # Pass fname into the surrounding call; this supplies the document filename used to match runfile records to score matrices.
        fname=fname,
        # Pass the expected_ref_text argument into the surrounding call so the callee receives that setting explicitly.
        expected_ref_text=reference_text,
        # Pass the expected_other_text argument into the surrounding call so the callee receives that setting explicitly.
        expected_other_text=reference_text,
        # Pass window_size into the surrounding call; this supplies the number of text characters represented by one score-matrix window.
        window_size=window_size,
        # Pass window_stride into the surrounding call; this supplies how many characters the sliding window moves between neighboring matrix cells.
        window_stride=window_stride,
        # Pass the fallback_label argument into the surrounding call so the callee receives that setting explicitly.
        fallback_label="ref_to_ref",
        # Pass the log argument into the surrounding call so the callee receives that setting explicitly.
        log=log,
    )


__all__ = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "LoadedScoreMatrix",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "ScoreMatrixIndexBundle",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "build_score_matrix_indexes",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "load_or_compute_ref_to_pred_matrix",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "load_or_compute_ref_to_ref_matrix",
]
