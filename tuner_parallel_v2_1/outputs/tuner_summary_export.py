from __future__ import annotations

"""Build the public JSON view for tuner summaries.

The tuner still uses a few compact internal metric names while ranking Hough
parameter combinations.  This module converts those internal names into the
human-facing names used in report files, without changing any evaluated values
or the best-combination selection logic.
"""

from typing import Any


# These keys are internal aliases or deprecated report fields.  They are safe to
# hide from the public summary because the same information is either unused in
# reports or exposed under the clearer public metric names below.
PUBLIC_SUMMARY_DROPPED_KEYS: frozenset[str] = frozenset(
    {
        "unweighted_along_lines_nls",
        "mean_unweighted_along_lines_nls",
        "median_unweighted_along_lines_nls",
        "std_unweighted_along_lines_nls",
        "min_unweighted_along_lines_nls",
        "max_unweighted_along_lines_nls",
        "valid_unweighted_along_lines_nls_count",
        "along_lines_nls",
        "correct_ref_coverage_percent",
        "missing_ref_percent",
        "missing_ref_coverage_percent",
        "repetition_on_ref_percent",
        "hallucination_percent",
        "one_minus_hallucination",
        "mean_one_minus_hallucination",
        "median_one_minus_hallucination",
        "std_one_minus_hallucination",
        "min_one_minus_hallucination",
        "max_one_minus_hallucination",
        "valid_one_minus_hallucination_count",
    }
)


# Public summaries should use the exact metric wording requested for reports,
# while the scorer can keep short internal keys in the hot path.
PUBLIC_SUMMARY_KEY_RENAMES: dict[str, str] = {
    "whole_document_nls": "normalised_levenshtein_similarity",
    "whole_document_nls_seconds": "normalised_levenshtein_similarity_seconds",
    "weighted_along_lines_nls": "average_weighted_normalised_levenshtein_similarity",
    "mean_weighted_along_lines_nls": "mean_average_weighted_normalised_levenshtein_similarity",
    "median_weighted_along_lines_nls": "median_average_weighted_normalised_levenshtein_similarity",
    "std_weighted_along_lines_nls": "std_average_weighted_normalised_levenshtein_similarity",
    "min_weighted_along_lines_nls": "min_average_weighted_normalised_levenshtein_similarity",
    "max_weighted_along_lines_nls": "max_average_weighted_normalised_levenshtein_similarity",
    "valid_weighted_along_lines_nls_count": "valid_average_weighted_normalised_levenshtein_similarity_count",
    "mean_best_weighted_along_lines_across_docs": (
        "mean_best_average_weighted_normalised_levenshtein_similarity_across_docs"
    ),
}


def _public_summary_value(value: Any) -> Any:
    """Recursively convert one JSON-compatible value into its public form."""
    if isinstance(value, dict):
        public_dict: dict[str, Any] = {}
        for raw_key, raw_child in value.items():
            # JSON object keys are strings in the final file, so normalize here
            # before checking drop/rename rules.
            key = str(raw_key)
            if key in PUBLIC_SUMMARY_DROPPED_KEYS:
                continue

            public_key = PUBLIC_SUMMARY_KEY_RENAMES.get(key, key)
            public_child = _public_summary_value(raw_child)

            # If a future caller already provides the public key directly, keep
            # that explicit value rather than overwriting it with a legacy alias.
            if public_key in public_dict and public_key != key:
                continue
            public_dict[public_key] = public_child

        return public_dict

    if isinstance(value, list):
        return [_public_summary_value(child) for child in value]

    return value


def build_public_tuner_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Return a report-safe summary without mutating the internal payload."""
    public_summary = _public_summary_value(summary)
    if not isinstance(public_summary, dict):
        raise TypeError("Expected tuner summary to convert into a JSON object")
    return public_summary


__all__ = [
    "PUBLIC_SUMMARY_DROPPED_KEYS",
    "PUBLIC_SUMMARY_KEY_RENAMES",
    "build_public_tuner_summary",
]
