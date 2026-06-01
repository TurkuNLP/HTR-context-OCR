from __future__ import annotations

"""Equivalence tests for the tuner-local v2.12 metric compatibility layer."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tuner_parallel_v2_2.metrics.v2_12_metric_adapter import (
    get_external_v212_metric_functions,
    get_v212_metric_functions,
)
from tuner_parallel_v2_2.metrics.alignment_quality_score import (
    compute_weighted_along_lines_similarity_from_bundle,
    compute_weighted_along_lines_similarity_from_compact_payload,
)
from tuner_parallel_v2_2.metrics.v2_12_compat.line_coverage_arrays import (
    build_line_coverage_arrays_from_bundles,
    build_line_coverage_arrays_from_cached_refref_y,
    build_refref_y_coverage_array_from_bundle,
)
from tuner_parallel_v2_2.metrics.v2_12_compat.line_metric_bundle import (
    build_compact_line_scoring_payload,
    build_line_metric_bundle,
)


EXTERNAL_V212_DIR = Path(__file__).resolve().parents[2] / "text_metrics_v2_12_parallel"


def _assert_same_payload(actual: Any, expected: Any) -> None:
    """Recursively compare v2.12 payloads containing arrays and plain objects."""
    if isinstance(expected, np.ndarray):
        np.testing.assert_array_equal(np.asarray(actual), expected)
        return
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in sorted(expected):
            _assert_same_payload(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for actual_item, expected_item in zip(actual, expected):
            _assert_same_payload(actual_item, expected_item)
        return
    if isinstance(expected, float):
        assert float(actual) == pytest.approx(float(expected), abs=0.0, rel=0.0)
        return
    assert actual == expected


def _empty_bundle_inputs() -> dict:
    """Return an empty-line case with fallback columns only."""
    return {
        "lines_used": [],
        "column_assignment": {
            "mapped_y": np.full(5, np.nan, dtype=float),
            "mapped_line_id": np.full(5, -1, dtype=int),
        },
        "n_ref_windows": 5,
        "n_other_windows": 5,
        "ref_text_len": 210,
        "other_text_len": 210,
        "window_size": 50,
        "window_stride": 35,
    }


def _diagonal_bundle_inputs() -> dict:
    """Return a single clean diagonal line case."""
    return {
        "lines_used": [
            {"x0": 0, "y0": 0, "x1": 5, "y1": 5, "support": 90.0, "length": 7.0},
        ],
        "column_assignment": {
            "mapped_y": np.asarray([0, 1, 2, 3, 4, 5], dtype=float),
            "mapped_line_id": np.zeros(6, dtype=int),
        },
        "n_ref_windows": 6,
        "n_other_windows": 6,
        "ref_text_len": 260,
        "other_text_len": 260,
        "window_size": 50,
        "window_stride": 35,
    }


def _two_line_reordered_bundle_inputs() -> dict:
    """Return a multi-line case that exercises monotonicity reordering."""
    return {
        "lines_used": [
            {"x0": 0, "y0": 3, "x1": 2, "y1": 1, "support": 72.0, "length": 4.0},
            {"x0": 3, "y0": 2, "x1": 6, "y1": 5, "support": 83.0, "length": 5.0},
        ],
        "column_assignment": {
            "mapped_y": np.asarray([3, 2, 1, 2, 3, 4, 5], dtype=float),
            "mapped_line_id": np.asarray([0, 0, 0, 1, 1, 1, 1], dtype=int),
        },
        "n_ref_windows": 7,
        "n_other_windows": 7,
        "ref_text_len": 310,
        "other_text_len": 310,
        "window_size": 50,
        "window_stride": 35,
    }


def _metric_functions_pair():
    """Return local and external v2.12 functions, skipping if external is absent."""
    if not EXTERNAL_V212_DIR.is_dir():
        pytest.skip(f"external v2.12 directory not available: {EXTERNAL_V212_DIR}")
    return get_v212_metric_functions(), get_external_v212_metric_functions(EXTERNAL_V212_DIR)


def test_default_v212_adapter_uses_tuner_local_compat_package() -> None:
    """The normal hot-loop adapter should no longer depend on the external tree."""
    functions = get_v212_metric_functions()

    assert "tuner_parallel_v2_2" in str(functions.text_metrics_v212_dir)
    assert "v2_12_compat" in str(functions.text_metrics_v212_dir)
    assert functions.line_metric_bundle_path.name == "line_metric_bundle.py"
    assert functions.line_coverage_subtract_path.name == "line_coverage_arrays.py"


@pytest.mark.parametrize(
    "bundle_inputs",
    [
        _empty_bundle_inputs(),
        _diagonal_bundle_inputs(),
        _two_line_reordered_bundle_inputs(),
    ],
)
def test_local_v212_bundle_matches_external_v212(bundle_inputs: dict) -> None:
    """Local bundle construction must match the historical v2.12 implementation."""
    local, external = _metric_functions_pair()

    local_bundle = local.build_line_metric_bundle(**bundle_inputs)
    external_bundle = external.build_line_metric_bundle(**bundle_inputs)

    _assert_same_payload(local_bundle, external_bundle)


def test_local_v212_coverage_arrays_match_external_v212() -> None:
    """Coverage arrays and cached-refref path must match external v2.12."""
    local, external = _metric_functions_pair()
    refref_inputs = _diagonal_bundle_inputs()
    other_inputs = _two_line_reordered_bundle_inputs()

    local_refref_bundle = local.build_line_metric_bundle(**refref_inputs)
    external_refref_bundle = external.build_line_metric_bundle(**refref_inputs)
    local_other_bundle = local.build_line_metric_bundle(**other_inputs)
    external_other_bundle = external.build_line_metric_bundle(**other_inputs)

    local_arrays = local.build_line_coverage_arrays_from_bundles(
        refref_bundle=local_refref_bundle,
        other_bundle=local_other_bundle,
    )
    external_arrays = external.build_line_coverage_arrays_from_bundles(
        refref_bundle=external_refref_bundle,
        other_bundle=external_other_bundle,
    )
    _assert_same_payload(local_arrays, external_arrays)

    local_refref_y = local.build_refref_y_coverage_array_from_bundle(refref_bundle=local_refref_bundle)
    external_refref_y = external.build_refref_y_coverage_array_from_bundle(refref_bundle=external_refref_bundle)
    np.testing.assert_array_equal(local_refref_y, external_refref_y)

    local_cached_arrays = local.build_line_coverage_arrays_from_cached_refref_y(
        refref_y=local_refref_y,
        other_bundle=local_other_bundle,
    )
    external_cached_arrays = external.build_line_coverage_arrays_from_cached_refref_y(
        refref_y=external_refref_y,
        other_bundle=external_other_bundle,
    )
    _assert_same_payload(local_cached_arrays, external_cached_arrays)


def test_local_v212_ratio_and_percentage_metrics_match_external_v212() -> None:
    """Local ratio and percentage coverage metrics must match external v2.12."""
    local, external = _metric_functions_pair()
    y_diff = np.asarray([-1, -1, 0, 0, 0, 1, 2], dtype=np.int32)
    other_x = np.asarray([0, 1, 1, 0, 2, 3, 0], dtype=np.int32)

    local_ratio = local.compute_line_coverage_ratio_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name="synthetic.jpeg",
    )
    external_ratio = external.compute_line_coverage_ratio_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name="synthetic.jpeg",
    )
    _assert_same_payload(local_ratio, external_ratio)

    local_percentage = local.compute_line_coverage_percentage_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name="synthetic.jpeg",
    )
    external_percentage = external.compute_line_coverage_percentage_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name="synthetic.jpeg",
    )
    _assert_same_payload(local_percentage, external_percentage)


def test_local_v212_invalid_y_diff_behavior_matches_external_v212() -> None:
    """Invalid y-diff categories must raise in the same condition as v2.12."""
    local, external = _metric_functions_pair()
    y_diff = np.asarray([-2, 0, 1], dtype=np.int32)
    other_x = np.asarray([1, 1, 1], dtype=np.int32)

    with pytest.raises(ValueError, match="outside defined categories"):
        local.compute_line_coverage_ratio_metrics_from_arrays(y_diff=y_diff, other_x=other_x)
    with pytest.raises(ValueError, match="outside defined categories"):
        external.compute_line_coverage_ratio_metrics_from_arrays(y_diff=y_diff, other_x=other_x)


@pytest.mark.parametrize(
    "bundle_inputs",
    [
        _empty_bundle_inputs(),
        _diagonal_bundle_inputs(),
        _two_line_reordered_bundle_inputs(),
    ],
)
def test_compact_scoring_payload_matches_full_bundle_scoring_fields(bundle_inputs: dict) -> None:
    """The compact hot-loop payload must be a subset-equivalent full bundle."""
    full_bundle = build_line_metric_bundle(**bundle_inputs)
    compact_payload = build_compact_line_scoring_payload(**bundle_inputs)

    assert compact_payload["line_guided_columns"] == full_bundle["line_guided_columns"]
    assert compact_payload["fallback_columns"] == full_bundle["fallback_columns"]
    assert compact_payload["ref_text_len"] == full_bundle["ref_text_len"]
    assert compact_payload["other_text_len"] == full_bundle["other_text_len"]
    assert len(compact_payload["lines"]) == len(full_bundle["lines"])

    for compact_line, full_line in zip(compact_payload["lines"], full_bundle["lines"]):
        assert compact_line["line_id"] == full_line["line_id"]
        assert compact_line["x_window_ids_owned"] == full_line["x_window_ids_owned"]
        assert compact_line["y_window_ids_for_levenshtein"] == full_line["y_window_ids_for_levenshtein"]
        assert compact_line["x_char_intervals_for_coverage"] == full_line["x_char_intervals_for_coverage"]
        assert compact_line["y_char_intervals_for_coverage"] == full_line["y_char_intervals_for_coverage"]


def test_compact_scoring_payload_matches_full_bundle_metrics() -> None:
    """Compact payload scoring must match the full bundle path exactly."""
    refref_inputs = _diagonal_bundle_inputs()
    other_inputs = _two_line_reordered_bundle_inputs()

    full_refref_bundle = build_line_metric_bundle(**refref_inputs)
    compact_refref_payload = build_compact_line_scoring_payload(**refref_inputs)
    full_other_bundle = build_line_metric_bundle(**other_inputs)
    compact_other_payload = build_compact_line_scoring_payload(**other_inputs)

    full_refref_y = build_refref_y_coverage_array_from_bundle(full_refref_bundle)
    compact_refref_y = build_refref_y_coverage_array_from_bundle(compact_refref_payload)
    np.testing.assert_array_equal(compact_refref_y, full_refref_y)

    full_arrays = build_line_coverage_arrays_from_bundles(
        refref_bundle=full_refref_bundle,
        other_bundle=full_other_bundle,
    )
    compact_arrays = build_line_coverage_arrays_from_cached_refref_y(
        refref_y=compact_refref_y,
        other_bundle=compact_other_payload,
    )
    _assert_same_payload(compact_arrays, full_arrays)

    ref_blocks = [f"r{i:02d}" for i in range(20)]
    other_blocks = [f"o{i:02d}" for i in range(20)]
    full_weighted = compute_weighted_along_lines_similarity_from_bundle(
        ref_blocks=ref_blocks,
        other_blocks=other_blocks,
        lines_used=other_inputs["lines_used"],
        bundle=full_other_bundle,
        levenshtein_backend="c",
    )
    compact_weighted = compute_weighted_along_lines_similarity_from_compact_payload(
        ref_blocks=ref_blocks,
        other_blocks=other_blocks,
        lines_used=other_inputs["lines_used"],
        compact_payload=compact_other_payload,
        levenshtein_backend="c",
    )

    assert compact_weighted.weighted_along_lines_nls == pytest.approx(
        full_weighted.weighted_along_lines_nls,
        abs=0.0,
        rel=0.0,
    )
    assert compact_weighted.unweighted_along_lines_nls == pytest.approx(
        full_weighted.unweighted_along_lines_nls,
        abs=0.0,
        rel=0.0,
    )
    assert compact_weighted.scored_line_count == full_weighted.scored_line_count
    assert compact_weighted.total_line_length == pytest.approx(
        full_weighted.total_line_length,
        abs=0.0,
        rel=0.0,
    )
