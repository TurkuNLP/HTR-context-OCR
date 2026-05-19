from __future__ import annotations

"""Alignment-quality scoring helpers for Hough parameter tuning.

This module computes only the tuner objective components.  Coverage and
hallucination ratios come from v2.12 through ``v2_12_metric_adapter``; this file
only clamps those exact ratios and combines them with weighted along-lines
Levenshtein.
"""

from dataclasses import dataclass
import math
from typing import Sequence

try:
    from ..cython_accel.optional_line_grouping import weighted_mean_from_scores_and_lengths
except ImportError:
    from cython_accel.optional_line_grouping import weighted_mean_from_scores_and_lengths  # type: ignore

try:
    from .levenshtein_compat import normalized_levenshtein_similarity
except ImportError:
    from metrics.levenshtein_compat import normalized_levenshtein_similarity  # type: ignore


@dataclass(frozen=True)
class WeightedAlongLinesResult:
    """Compact result of line-level text scoring for one Hough combination."""

    weighted_along_lines_nls: float | None
    unweighted_along_lines_nls: float | None
    scored_line_count: int
    total_line_length: float


def clamp_unit_interval(value) -> float:
    """Clamp a finite numeric value into the closed interval ``[0, 1]``."""
    try:
        numeric_value = float(value)
    except Exception:
        return 0.0
    if not math.isfinite(numeric_value):
        return 0.0
    return max(0.0, min(1.0, numeric_value))


def euclidean_line_length(line: dict) -> float:
    """Return the finite positive Euclidean length for one final filtered line."""
    existing_length = line.get("length") if isinstance(line, dict) else None
    try:
        length = float(existing_length)
    except Exception:
        length = float("nan")
    if math.isfinite(length) and length > 0.0:
        return float(length)

    try:
        x0 = float(line.get("x0", 0.0))
        y0 = float(line.get("y0", 0.0))
        x1 = float(line.get("x1", 0.0))
        y1 = float(line.get("y1", 0.0))
    except Exception:
        return 0.0

    computed_length = math.hypot(x1 - x0, y1 - y0)
    if not math.isfinite(computed_length) or computed_length <= 0.0:
        return 0.0
    return float(computed_length)


def _join_blocks_by_indices(blocks: Sequence[str], indices: Sequence[int]) -> str:
    """Concatenate valid text blocks in the exact order provided by the bundle."""
    block_count = len(blocks)
    return "".join(str(blocks[int(index)]) for index in indices if 0 <= int(index) < block_count)


def compute_weighted_along_lines_similarity_from_bundle(
    *,
    ref_blocks: Sequence[str],
    other_blocks: Sequence[str],
    lines_used: list[dict],
    bundle: dict,
    levenshtein_backend: str,
) -> WeightedAlongLinesResult:
    """Compute weighted and unweighted along-lines NLS from a v2.12 bundle.

    The v2.12 bundle already contains the final ownership and reference-row order
    used for line-level Levenshtein.  Reusing it avoids rescanning ownership
    arrays and keeps the text-order semantics aligned with v2.12.
    """
    lines_by_id = {int(line_index): line for line_index, line in enumerate(lines_used)}
    line_scores: list[float] = []
    line_lengths: list[float] = []

    for line_entry in bundle.get("lines", []):
        line_id = int(line_entry.get("line_id", -1))
        owned_prediction_columns = [int(value) for value in line_entry.get("x_window_ids_owned", [])]
        mapped_reference_rows = [int(value) for value in line_entry.get("y_window_ids_for_levenshtein", [])]

        # A line without prediction ownership cannot provide line-level text.
        if not owned_prediction_columns or not mapped_reference_rows:
            continue

        prediction_line_text = _join_blocks_by_indices(other_blocks, owned_prediction_columns)
        reference_line_text = _join_blocks_by_indices(ref_blocks, mapped_reference_rows)
        line_score = float(
            normalized_levenshtein_similarity(
                prediction_line_text,
                reference_line_text,
                backend=str(levenshtein_backend),
            )
        )
        line_length = euclidean_line_length(lines_by_id.get(line_id, {}))

        if not math.isfinite(line_score):
            continue
        if line_length <= 0.0:
            continue

        line_scores.append(clamp_unit_interval(line_score))
        line_lengths.append(float(line_length))

    if not line_scores:
        return WeightedAlongLinesResult(
            weighted_along_lines_nls=None,
            unweighted_along_lines_nls=None,
            scored_line_count=0,
            total_line_length=0.0,
        )

    weighted_mean = weighted_mean_from_scores_and_lengths(line_scores, line_lengths)
    unweighted_mean = float(sum(line_scores) / len(line_scores))

    return WeightedAlongLinesResult(
        weighted_along_lines_nls=None if weighted_mean is None else clamp_unit_interval(weighted_mean),
        unweighted_along_lines_nls=clamp_unit_interval(unweighted_mean),
        scored_line_count=int(len(line_scores)),
        total_line_length=float(sum(line_lengths)),
    )


def normalize_v212_coverage_metrics(raw_ratio_metrics: dict) -> dict:
    """Clamp v2.12 coverage metrics that are already expressed as ratios.

    V2.12 now exposes coverage as direct ``0..1`` ratios, so the tuner no longer
    needs to multiply or divide by ``100``.  The fallback reads the old percent
    keys only to keep historical summaries/tests importable; production runs use
    the ratio keys from ``compute_line_coverage_ratio_metrics_from_arrays``.
    """
    if "correct_ref_coverage" in raw_ratio_metrics:
        correct_ref_coverage = clamp_unit_interval(raw_ratio_metrics.get("correct_ref_coverage", 0.0))
        missing_ref_coverage = clamp_unit_interval(raw_ratio_metrics.get("missing_ref_coverage", 0.0))
        repetition_on_ref = clamp_unit_interval(raw_ratio_metrics.get("repetition_on_ref", 0.0))
        hallucination = clamp_unit_interval(raw_ratio_metrics.get("hallucination", 0.0))
    else:
        correct_ref_coverage = clamp_unit_interval(float(raw_ratio_metrics.get("ok_percent", 0.0)) / 100.0)
        missing_ref_coverage = clamp_unit_interval(float(raw_ratio_metrics.get("missing_percent", 0.0)) / 100.0)
        repetition_on_ref = clamp_unit_interval(float(raw_ratio_metrics.get("repetition_percent", 0.0)) / 100.0)
        hallucination = clamp_unit_interval(float(raw_ratio_metrics.get("hallucination_percent", 0.0)) / 100.0)

    return {
        "correct_ref_coverage": float(correct_ref_coverage),
        "missing_ref_coverage": float(missing_ref_coverage),
        "repetition_on_ref": float(repetition_on_ref),
        "hallucination": float(hallucination),
    }


def compute_harmonic_tuning_score(
    *,
    weighted_along_lines_nls,
    correct_ref_coverage,
    hallucination,
) -> float:
    """Compute the final harmonic tuner objective in ``[0, 1]``.

    A zero-quality component makes the harmonic score zero.  This avoids division
    by zero and ensures combinations with no meaningful alignment cannot win
    because of one strong component alone.
    """
    weighted_nls = clamp_unit_interval(weighted_along_lines_nls)
    coverage = clamp_unit_interval(correct_ref_coverage)
    hallucination_rate = clamp_unit_interval(hallucination)
    non_hallucination = clamp_unit_interval(1.0 - hallucination_rate)

    if weighted_nls <= 0.0 or coverage <= 0.0 or non_hallucination <= 0.0:
        return 0.0

    score = 3.0 / ((1.0 / weighted_nls) + (1.0 / coverage) + (1.0 / non_hallucination))
    return clamp_unit_interval(score)


__all__ = [
    "WeightedAlongLinesResult",
    "clamp_unit_interval",
    "compute_harmonic_tuning_score",
    "compute_weighted_along_lines_similarity_from_bundle",
    "euclidean_line_length",
    "normalize_v212_coverage_metrics",
]
