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

import numpy as np

try:
    from ..cython_accel.optional_line_grouping import weighted_mean_from_scores_and_lengths
except ImportError:
    from cython_accel.optional_line_grouping import weighted_mean_from_scores_and_lengths  # type: ignore

try:
    from .levenshtein_compat import normalized_levenshtein_similarity
    from .v2_12_compat.line_metric_bundle import reference_rows_for_levenshtein
except ImportError:
    from metrics.levenshtein_compat import normalized_levenshtein_similarity  # type: ignore
    from metrics.v2_12_compat.line_metric_bundle import reference_rows_for_levenshtein  # type: ignore


@dataclass(frozen=True)
class WeightedAlongLinesResult:
    """Compact result of line-level text scoring for one Hough combination."""

    weighted_along_lines_nls: float | None
    unweighted_along_lines_nls: float | None
    scored_line_count: int
    total_line_length: float


@dataclass(frozen=True)
class LineLevelSimilarityRecord:
    """Exact text-similarity score for one final line.

    ``line_id`` is the index of the line in the current ``lines_used`` list.
    The caller may later renumber surviving lines, so this record intentionally
    keeps the original id that was scored.
    """

    line_id: int
    normalized_levenshtein_similarity: float
    line_length: float
    owned_prediction_column_count: int
    mapped_reference_row_count: int


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


def weighted_along_lines_result_from_line_similarity_records(
    line_similarity_records: Sequence[LineLevelSimilarityRecord],
) -> WeightedAlongLinesResult:
    """Return weighted/unweighted along-line NLS from already-scored lines."""
    line_scores = [
        clamp_unit_interval(record.normalized_levenshtein_similarity)
        for record in line_similarity_records
        if math.isfinite(float(record.normalized_levenshtein_similarity))
        and float(record.line_length) > 0.0
    ]
    line_lengths = [
        float(record.line_length)
        for record in line_similarity_records
        if math.isfinite(float(record.normalized_levenshtein_similarity))
        and float(record.line_length) > 0.0
    ]

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


def compute_line_level_similarity_records_from_assignment(
    *,
    ref_blocks: Sequence[str],
    other_blocks: Sequence[str],
    lines_used: list[dict],
    column_assignment: dict,
    n_ref_windows: int,
    levenshtein_backend: str,
) -> list[LineLevelSimilarityRecord]:
    """Compute exact line-level NLS for final lines and final ownership arrays.

    This helper is used by the optional post-filter text-quality gate.  It uses
    the same ownership arrays and the same ``reference_rows_for_levenshtein``
    helper as the compact v2.12 scoring payload, but it avoids building coverage
    intervals before weak lines have been removed.
    """
    mapped_y = np.asarray(column_assignment.get("mapped_y", []), dtype=float)
    mapped_line_id = np.asarray(column_assignment.get("mapped_line_id", []), dtype=int)
    scored_records: list[LineLevelSimilarityRecord] = []

    for line_id, line in enumerate(lines_used):
        owned_prediction_columns = [
            int(prediction_column)
            for prediction_column in np.flatnonzero(mapped_line_id == int(line_id))
        ]
        reference_rows_for_line, _rows_reordered_for_monotonicity = reference_rows_for_levenshtein(
            owned_prediction_columns,
            mapped_y,
            int(n_ref_windows),
        )

        # A line without owned prediction text or mapped reference text cannot
        # satisfy a minimum text-similarity threshold.
        if not owned_prediction_columns or not reference_rows_for_line:
            continue

        prediction_line_text = _join_blocks_by_indices(other_blocks, owned_prediction_columns)
        reference_line_text = _join_blocks_by_indices(ref_blocks, reference_rows_for_line)
        line_score = float(
            normalized_levenshtein_similarity(
                prediction_line_text,
                reference_line_text,
                backend=str(levenshtein_backend),
            )
        )
        line_length = euclidean_line_length(line)

        if not math.isfinite(line_score) or line_length <= 0.0:
            continue

        scored_records.append(
            LineLevelSimilarityRecord(
                line_id=int(line_id),
                normalized_levenshtein_similarity=clamp_unit_interval(line_score),
                line_length=float(line_length),
                owned_prediction_column_count=int(len(owned_prediction_columns)),
                mapped_reference_row_count=int(len(reference_rows_for_line)),
            )
        )

    return scored_records


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
    return _compute_weighted_along_lines_similarity_from_line_entries(
        ref_blocks=ref_blocks,
        other_blocks=other_blocks,
        lines_used=lines_used,
        line_entries=list(bundle.get("lines", [])),
        levenshtein_backend=str(levenshtein_backend),
    )


def compute_weighted_along_lines_similarity_from_compact_payload(
    *,
    ref_blocks: Sequence[str],
    other_blocks: Sequence[str],
    lines_used: list[dict],
    compact_payload: dict,
    levenshtein_backend: str,
) -> WeightedAlongLinesResult:
    """Compute along-lines NLS from the compact hot-loop scoring payload."""
    return _compute_weighted_along_lines_similarity_from_line_entries(
        ref_blocks=ref_blocks,
        other_blocks=other_blocks,
        lines_used=lines_used,
        line_entries=list(compact_payload.get("lines", [])),
        levenshtein_backend=str(levenshtein_backend),
    )


def _compute_weighted_along_lines_similarity_from_line_entries(
    *,
    ref_blocks: Sequence[str],
    other_blocks: Sequence[str],
    lines_used: list[dict],
    line_entries: list[dict],
    levenshtein_backend: str,
) -> WeightedAlongLinesResult:
    """Compute line-level text similarity from canonical line-entry records.

    Both the full v2.12 bundle and the compact scorer payload expose the same
    two fields needed here: owned prediction columns and mapped reference rows
    for Levenshtein.  Keeping this calculation in one helper prevents the full
    and compact payload paths from drifting apart.
    """
    lines_by_id = {int(line_index): line for line_index, line in enumerate(lines_used)}
    line_similarity_records: list[LineLevelSimilarityRecord] = []

    for line_entry in line_entries:
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

        line_similarity_records.append(
            LineLevelSimilarityRecord(
                line_id=int(line_id),
                normalized_levenshtein_similarity=clamp_unit_interval(line_score),
                line_length=float(line_length),
                owned_prediction_column_count=int(len(owned_prediction_columns)),
                mapped_reference_row_count=int(len(mapped_reference_rows)),
            )
        )

    return weighted_along_lines_result_from_line_similarity_records(line_similarity_records)


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



def compute_balanced_harmonic_mean(unit_interval_values: Sequence[float]) -> float:
    """Return the harmonic mean of positive values already measured in ``[0, 1]``.

    The harmonic mean is intentionally strict: one zero-quality component makes
    the whole score zero.  That behavior is useful for tuner objectives because
    it prevents one excellent signal from hiding another missing signal.
    """
    cleaned_values = [clamp_unit_interval(value) for value in unit_interval_values]
    if not cleaned_values or any(value <= 0.0 for value in cleaned_values):
        return 0.0
    denominator = sum(1.0 / value for value in cleaned_values)
    if denominator <= 0.0 or not math.isfinite(denominator):
        return 0.0
    return clamp_unit_interval(float(len(cleaned_values)) / float(denominator))


def compute_line_guided_fraction(*, line_guided_columns, fallback_columns) -> float:
    """Return the fraction of prediction columns explained by detected lines.

    ``line_guided_columns`` are prediction windows assigned through surviving
    Hough lines.  ``fallback_columns`` are prediction windows that had to be
    handled without a surviving line.  A value near one means the geometry is
    actually doing the alignment work.
    """
    try:
        guided_count = max(0.0, float(line_guided_columns))
    except Exception:
        guided_count = 0.0
    try:
        fallback_count = max(0.0, float(fallback_columns))
    except Exception:
        fallback_count = 0.0
    total_columns = guided_count + fallback_count
    if total_columns <= 0.0:
        return 0.0
    return clamp_unit_interval(guided_count / total_columns)


def compute_score_matrix_support_from_lines(lines_used: Sequence[dict]) -> float:
    """Return average score-matrix support for final surviving lines.

    Each final line carries ``owned_score_mean`` from the score matrix cells that
    the final assignment gave to that line.  The score matrix is measured on the
    familiar 0..100 percentage scale, so this helper converts it to 0..1 before
    it is combined with other tuner signals.
    """
    weighted_support_sum = 0.0
    support_weight_sum = 0.0

    for line in lines_used:
        if not isinstance(line, dict):
            continue
        try:
            support_percent = float(line.get("owned_score_mean", 0.0))
        except Exception:
            continue
        if not math.isfinite(support_percent):
            continue

        try:
            owned_column_count = float(line.get("owned_cols", 0.0))
        except Exception:
            owned_column_count = 0.0
        line_weight = owned_column_count if owned_column_count > 0.0 else euclidean_line_length(line)
        if line_weight <= 0.0 or not math.isfinite(line_weight):
            line_weight = 1.0

        weighted_support_sum += clamp_unit_interval(support_percent / 100.0) * float(line_weight)
        support_weight_sum += float(line_weight)

    if support_weight_sum <= 0.0:
        return 0.0
    return clamp_unit_interval(weighted_support_sum / support_weight_sum)


def compute_alignment_evidence_selection_score(
    *,
    weighted_along_lines_nls,
    score_matrix_support,
    line_guided_fraction,
    hallucination,
) -> float:
    """Score how strongly the matrix and final geometry support an alignment.

    This score is a selection objective only.  It does not replace the final
    scientific metrics, and it does not hide repetition or missing-reference
    penalties from the exported result.  It is useful when the Hough winner
    should prefer a matrix-supported repeated line over a geometrically neat but
    less faithful alternative.
    """
    non_hallucination = clamp_unit_interval(1.0 - clamp_unit_interval(hallucination))
    return compute_balanced_harmonic_mean(
        [
            clamp_unit_interval(weighted_along_lines_nls),
            clamp_unit_interval(score_matrix_support),
            clamp_unit_interval(line_guided_fraction),
            non_hallucination,
        ]
    )

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


def compute_non_hallucination_weighted_tuning_score(
    *,
    weighted_along_lines_nls,
    correct_ref_coverage,
    hallucination,
) -> float:
    """Compute the harmonic tuner score with double weight on non-hallucination."""
    weighted_nls = clamp_unit_interval(weighted_along_lines_nls)
    coverage = clamp_unit_interval(correct_ref_coverage)
    hallucination_rate = clamp_unit_interval(hallucination)
    non_hallucination = clamp_unit_interval(1.0 - hallucination_rate)

    if weighted_nls <= 0.0 or coverage <= 0.0 or non_hallucination <= 0.0:
        return 0.0

    score = 4.0 / ((1.0 / weighted_nls) + (1.0 / coverage) + (2.0 / non_hallucination))
    return clamp_unit_interval(score)


__all__ = [
    "LineLevelSimilarityRecord",
    "WeightedAlongLinesResult",
    "clamp_unit_interval",
    "compute_alignment_evidence_selection_score",
    "compute_balanced_harmonic_mean",
    "compute_harmonic_tuning_score",
    "compute_non_hallucination_weighted_tuning_score",
    "compute_line_guided_fraction",
    "compute_line_level_similarity_records_from_assignment",
    "compute_weighted_along_lines_similarity_from_bundle",
    "compute_score_matrix_support_from_lines",
    "compute_weighted_along_lines_similarity_from_compact_payload",
    "euclidean_line_length",
    "normalize_v212_coverage_metrics",
    "weighted_along_lines_result_from_line_similarity_records",
]
