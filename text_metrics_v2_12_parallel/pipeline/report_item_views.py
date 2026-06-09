"""Public and internal report-item views for text-metrics outputs.

The pipeline keeps a richer internal per-document result object for progress
logging, run-level aggregation, debug output, and optional visuals. This module
also defines the smaller public view that is written into ``report.json`` when
``--debug`` is not enabled, together with the smaller internal success-spool
view used to reduce non-debug JSONL overhead.
"""

from __future__ import annotations


# Resolve the final full-document NLS after the pipeline's current realignment flow.
def _resolve_after_realignment_nls(internal_item: dict) -> float:
    """Return the final full-document NLS after the current realignment flow."""
    stored_after_value = internal_item.get("after_normalized_levenshtein_similarity")
    if stored_after_value is not None:
        return float(stored_after_value)

    before_value = float(internal_item["normalized_levenshtein_before"])
    along_lines_value = internal_item.get("average_normalized_levenshtein_along_lines")
    if along_lines_value is None:
        return float(before_value)
    return float(along_lines_value)


# Build the smaller internal success payload used by non-debug JSONL spools.
def build_internal_non_debug_success_spool_item(internal_item: dict) -> dict:
    """Project one internal success item into the minimal non-debug spool view."""
    return {
        "fname": str(internal_item["fname"]),
        "normalized_levenshtein_before": float(internal_item["normalized_levenshtein_before"]),
        "average_normalized_levenshtein_along_lines": (
            None
            if internal_item.get("average_normalized_levenshtein_along_lines") is None
            else float(internal_item["average_normalized_levenshtein_along_lines"])
        ),
        "average_weighted_normalized_levenshtein_along_lines": (
            None
            if internal_item.get("average_weighted_normalized_levenshtein_along_lines") is None
            else float(internal_item["average_weighted_normalized_levenshtein_along_lines"])
        ),
        "after_normalized_levenshtein_similarity": float(_resolve_after_realignment_nls(internal_item)),
        "correct_ref_coverage": float(internal_item["correct_ref_coverage"]),
        "missing_ref_coverage": float(internal_item["missing_ref_coverage"]),
        "repetition_on_ref": float(internal_item["repetition_on_ref"]),
        "hallucination": float(internal_item["hallucination"]),
    }


# Build the compact public report item shown in non-debug report.json outputs.
def build_public_non_debug_report_item(internal_item: dict) -> dict:
    """Project one internal success item into the compact non-debug report view."""
    file_name = str(internal_item["fname"])
    document_normalized_levenshtein = float(internal_item["normalized_levenshtein_before"])
    average_weighted_normalized_levenshtein_along_lines = internal_item.get(
        "average_weighted_normalized_levenshtein_along_lines"
    )

    return {
        "fname": file_name,
        "normalised_levenshtein_similarity": float(document_normalized_levenshtein),
        "average_weighted_normalised_levenshtein_similarity": (
            None
            if average_weighted_normalized_levenshtein_along_lines is None
            else float(average_weighted_normalized_levenshtein_along_lines)
        ),
        "correct_ref_coverage": float(internal_item["correct_ref_coverage"]),
        "missing_ref_coverage": float(internal_item["missing_ref_coverage"]),
        "repetition_on_ref": float(internal_item["repetition_on_ref"]),
        "hallucination": float(internal_item["hallucination"]),
    }
