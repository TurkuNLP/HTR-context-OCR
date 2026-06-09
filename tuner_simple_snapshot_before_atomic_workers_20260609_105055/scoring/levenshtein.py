from __future__ import annotations

"""RapidFuzz-only Levenshtein helpers used by tuner_simple."""

from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein


def levenshtein_distance(source_text: str, target_text: str) -> int:
    """Return exact Levenshtein edit distance using RapidFuzz."""
    # Normalize both inputs to strings so the distance function always receives text.
    source_text = str(source_text)
    # Normalize the target in the same way as the source.
    target_text = str(target_text)
    # Delegate the actual edit-distance calculation to RapidFuzz's compiled implementation.
    return int(_RapidfuzzLevenshtein.distance(source_text, target_text))


def normalized_levenshtein_similarity(predicted_text: str, reference_text: str) -> float:
    """Return Levenshtein similarity in the unit interval, where 1.0 means identical text."""
    # Normalize the prediction to a string before measuring its length and distance.
    predicted_text = str(predicted_text)
    # Normalize the reference to a string before measuring its length and distance.
    reference_text = str(reference_text)
    # Use the longer string length so a complete mismatch has a similarity near zero.
    denominator = max(len(predicted_text), len(reference_text))
    # Two empty strings are perfectly equal, so return the maximum similarity.
    if denominator == 0:
        # Return 1.0 because there are no edits needed between two empty strings.
        return 1.0
    # Compute the exact edit distance with RapidFuzz.
    edit_distance = levenshtein_distance(predicted_text, reference_text)
    # Convert distance into similarity and clamp tiny floating-point edge cases into the valid interval.
    return max(0.0, min(1.0, 1.0 - (float(edit_distance) / float(denominator))))


# Declare the public helpers that other tuner_simple modules may import.
__all__ = ["levenshtein_distance", "normalized_levenshtein_similarity"]
