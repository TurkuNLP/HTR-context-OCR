from __future__ import annotations

"""Levenshtein fallback score-matrix computation for missing `.pkl` records."""

import numpy as np

from .matrix_shape import sliding_text_windows

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.scoring.levenshtein import normalized_levenshtein_similarity


# Define the compute_levenshtein_score_matrix function; its body below performs one named step of the pipeline.
def compute_levenshtein_score_matrix(
    # Pass this value into the surrounding multi-line call or collection.
    *,
    # Define the reference_text field; it stores the normalized reference transcription for this document.
    reference_text: str,
    # Define the other_text field so this data object records that value explicitly.
    other_text: str,
    # Define the window_size field; it stores the number of text characters represented by one score-matrix window.
    window_size: int,
    # Define the window_stride field; it stores how many characters the sliding window moves between neighboring matrix cells.
    window_stride: int,
# Use NumPy here because matrix operations should run on compact numeric arrays.
) -> np.ndarray:
    """Compute a `0..100` Levenshtein similarity matrix from text windows."""
    # Compute or store reference_windows so later code can reuse this named value clearly.
    reference_windows = sliding_text_windows(reference_text, window_size=window_size, window_stride=window_stride)
    # Compute or store other_windows so later code can reuse this named value clearly.
    other_windows = sliding_text_windows(other_text, window_size=window_size, window_stride=window_stride)
    # Use NumPy here because matrix operations should run on compact numeric arrays.
    matrix = np.zeros((len(reference_windows), len(other_windows)), dtype=float)

    # Iterate over reference_index, reference_window in enumerate(reference_windows) so each item is processed with the same logic.
    for reference_index, reference_window in enumerate(reference_windows):
        # Iterate over other_index, other_window in enumerate(other_windows) so each item is processed with the same logic.
        for other_index, other_window in enumerate(other_windows):
            # Compute or store similarity so later code can reuse this named value clearly.
            similarity = normalized_levenshtein_similarity(
                # Pass this value into the surrounding multi-line call or collection.
                other_window,
                # Pass this value into the surrounding multi-line call or collection.
                reference_window,
            )
            # Compute or store matrix[reference_index, other_index] so later code can reuse this named value clearly.
            matrix[reference_index, other_index] = 100.0 * float(similarity)

    # Return this computed value to the caller so the next pipeline stage can use it.
    return matrix


__all__ = ["compute_levenshtein_score_matrix"]
