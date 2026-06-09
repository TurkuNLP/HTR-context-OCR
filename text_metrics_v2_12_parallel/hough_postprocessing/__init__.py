"""Hough postprocessing helpers for the text-metrics pipeline.

This package contains post-Hough geometry logic that is separate from the raw
probabilistic Hough transform itself. Keeping this code isolated makes the
current merged-Hough default path easier to understand, easier to profile, and
much easier to remove in a later version when the raw-Hough handoff becomes the
only supported behavior.
"""

from .greedy_diagonal_segment_merging import merge_diagonal_segments

__all__ = ["merge_diagonal_segments"]
