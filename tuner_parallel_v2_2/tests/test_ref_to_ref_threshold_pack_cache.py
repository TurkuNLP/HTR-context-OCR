from __future__ import annotations

"""Tests for the exact ref_to_ref threshold-pack cache."""

from pathlib import Path
import json
from types import SimpleNamespace

import numpy as np

from tuner_parallel_v2_2.cache.ref_to_ref_combo_cache import (
    _THRESHOLD_PACK_CACHE_SCHEMA_VERSION_V2,
    _cache_file_path,
    _cache_key_from_metadata,
    build_ref_to_ref_threshold_pack_metadata,
    RefToRefCombinationCache,
)


def _dummy_sweep_document() -> SimpleNamespace:
    """Return the minimal document shape required by cache metadata."""
    return SimpleNamespace(
        index=3,
        fname="dummy_document.jpeg",
        ref="abcdef",
        ref_to_ref_matrix=np.eye(4, dtype=float),
        window_size=50,
        window_stride=35,
    )


def _payload_for(line_length: int, line_gap: int, seed: int) -> dict:
    """Return a deterministic tiny payload that mirrors evaluator output."""
    payload_value = int(line_length * 100 + line_gap * 10 + seed)
    return {
        "refref_y": np.asarray([payload_value, payload_value + 1, payload_value + 2], dtype=np.int32),
        "line_guided_columns": 2,
        "fallback_columns": 1,
        "raw_line_count": 4,
        "skimage_raw_line_count_before_direction_filter": 5,
        "direction_rejected_line_count": 1,
        "candidate_line_count": 3,
        "used_line_count": 2,
        "threshold_start": 2.4,
        "timing_hough_detect_seconds": 0.01,
        "timing_filter_seconds": 0.02,
        "timing_build_bundle_seconds": 0.03,
        "timing_direction_total_seconds": 0.06,
    }


def test_auto_writes_one_complete_threshold_pack_and_read_only_reuses_it(tmp_path: Path) -> None:
    """Auto writes one pack per threshold, and read-only serves exact payloads."""
    doc = _dummy_sweep_document()
    line_lengths = [5, 6]
    line_gaps = [0, 1]
    seeds = [1]

    auto_cache = RefToRefCombinationCache(cache_dir=tmp_path, mode="auto")
    auto_session = auto_cache.begin_threshold(
        doc=doc,
        hough_threshold=10,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )

    for line_length in line_lengths:
        for line_gap in line_gaps:
            auto_session.get_or_compute(
                doc=doc,
                hough_threshold=10,
                hough_line_length=line_length,
                hough_line_gap=line_gap,
                hough_seed=1,
                align_abs_min_len=8.0,
                align_min_iou_threshold=0.035,
                compute_payload=lambda line_length=line_length, line_gap=line_gap: _payload_for(line_length, line_gap, 1),
            )
    auto_session.close()

    cache_files = sorted(tmp_path.glob("*/*.npz"))
    assert len(cache_files) == 1
    assert auto_cache.stats.as_dict()["threshold_pack_writes"] == 1
    with np.load(cache_files[0], allow_pickle=False) as compact_cache_payload:
        assert "refref_y_unique_rows" in compact_cache_payload.files
        assert "refref_y_row_index_by_combination" in compact_cache_payload.files
        assert "refref_y_by_combination" not in compact_cache_payload.files

    read_only_cache = RefToRefCombinationCache(cache_dir=tmp_path, mode="read-only")
    read_only_session = read_only_cache.begin_threshold(
        doc=doc,
        hough_threshold=10,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )
    cached_payload = read_only_session.get_or_compute(
        doc=doc,
        hough_threshold=10,
        hough_line_length=6,
        hough_line_gap=1,
        hough_seed=1,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
        compute_payload=lambda: (_ for _ in ()).throw(AssertionError("read-only cache should not compute")),
    )

    assert cached_payload["ref_to_ref_cache_hit"] is True
    assert cached_payload["timing_hough_detect_seconds"] == 0.0
    assert cached_payload["direction_rejected_line_count"] == 1
    assert cached_payload["skimage_raw_line_count_before_direction_filter"] == 5
    assert cached_payload["refref_y"].tolist() == _payload_for(6, 1, 1)["refref_y"].tolist()
    assert read_only_cache.stats.as_dict()["threshold_pack_hits"] == 1
    assert read_only_cache.stats.as_dict()["hits"] == 1


def test_read_only_can_reuse_v2_threshold_pack_cache(tmp_path: Path) -> None:
    """The compact v3 reader still accepts exact v2 threshold-pack files."""
    doc = _dummy_sweep_document()
    line_lengths = [5, 6]
    line_gaps = [0, 1]
    seeds = [1]
    sorted_keys = [(line_length, line_gap, 1) for line_length in line_lengths for line_gap in line_gaps]
    metadata = build_ref_to_ref_threshold_pack_metadata(
        doc=doc,
        hough_threshold=10,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
        cache_schema_version=_THRESHOLD_PACK_CACHE_SCHEMA_VERSION_V2,
    )
    cache_path = _cache_file_path(tmp_path, _cache_key_from_metadata(metadata))
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        cache_path,
        metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
        combination_keys=np.asarray(sorted_keys, dtype=np.int32),
        refref_y_by_combination=np.stack(
            [_payload_for(line_length, line_gap, seed)["refref_y"] for line_length, line_gap, seed in sorted_keys],
            axis=0,
        ),
        line_guided_columns_by_combination=np.asarray([2, 2, 2, 2], dtype=np.int64),
        fallback_columns_by_combination=np.asarray([1, 1, 1, 1], dtype=np.int64),
        raw_line_count_by_combination=np.asarray([4, 4, 4, 4], dtype=np.int64),
        skimage_raw_line_count_before_direction_filter_by_combination=np.asarray([5, 5, 5, 5], dtype=np.int64),
        direction_rejected_line_count_by_combination=np.asarray([1, 1, 1, 1], dtype=np.int64),
        candidate_line_count_by_combination=np.asarray([3, 3, 3, 3], dtype=np.int64),
        used_line_count_by_combination=np.asarray([2, 2, 2, 2], dtype=np.int64),
        threshold_start_by_combination=np.asarray([2.4, 2.4, 2.4, 2.4], dtype=np.float64),
    )

    read_only_cache = RefToRefCombinationCache(cache_dir=tmp_path, mode="read-only")
    read_only_session = read_only_cache.begin_threshold(
        doc=doc,
        hough_threshold=10,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )
    cached_payload = read_only_session.get_or_compute(
        doc=doc,
        hough_threshold=10,
        hough_line_length=5,
        hough_line_gap=1,
        hough_seed=1,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
        compute_payload=lambda: (_ for _ in ()).throw(AssertionError("v2 cache should be reused")),
    )

    assert cached_payload["ref_to_ref_cache_hit"] is True
    assert cached_payload["refref_y"].tolist() == _payload_for(5, 1, 1)["refref_y"].tolist()


def test_document_session_writes_one_document_pack_after_thresholds_finish(tmp_path: Path) -> None:
    """Production-style sessions write one cache file for the whole document."""
    doc = _dummy_sweep_document()
    thresholds = [10, 11]
    line_lengths = [5, 6]
    line_gaps = [0, 1]
    seeds = [1]

    auto_cache = RefToRefCombinationCache(cache_dir=tmp_path, mode="auto")
    document_session = auto_cache.begin_document(
        doc=doc,
        threshold_values=thresholds,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )

    for threshold in thresholds:
        threshold_session = document_session.begin_threshold(
            doc=doc,
            hough_threshold=threshold,
            line_length_values=line_lengths,
            line_gap_values=line_gaps,
            seed_values=seeds,
            align_abs_min_len=8.0,
            align_min_iou_threshold=0.035,
        )
        for line_length in line_lengths:
            for line_gap in line_gaps:
                threshold_session.get_or_compute(
                    doc=doc,
                    hough_threshold=threshold,
                    hough_line_length=line_length,
                    hough_line_gap=line_gap,
                    hough_seed=1,
                    align_abs_min_len=8.0,
                    align_min_iou_threshold=0.035,
                    compute_payload=lambda line_length=line_length, line_gap=line_gap: _payload_for(
                        line_length,
                        line_gap,
                        1,
                    ),
                )
        threshold_session.close()

    document_session.submit_completed_document_write()
    auto_cache.close()

    cache_files = sorted(tmp_path.glob("*/*.npz"))
    assert len(cache_files) == 1
    assert auto_cache.stats.as_dict()["document_pack_writes"] == 1
    assert auto_cache.stats.as_dict()["threshold_pack_writes"] == 0
    with np.load(cache_files[0], allow_pickle=False) as compact_cache_payload:
        metadata = json.loads(str(compact_cache_payload["metadata_json"].item()))
        assert metadata["cache_schema_version"] == "ref_to_ref_document_pack_cache_v2_literal_seed"
        assert compact_cache_payload["combination_keys"].shape[1] == 4


def test_read_only_document_session_reuses_document_pack(tmp_path: Path) -> None:
    """Read-only production-style sessions serve exact payloads from one document pack."""
    doc = _dummy_sweep_document()
    thresholds = [10]
    line_lengths = [5, 6]
    line_gaps = [0, 1]
    seeds = [1]

    auto_cache = RefToRefCombinationCache(cache_dir=tmp_path, mode="auto")
    document_session = auto_cache.begin_document(
        doc=doc,
        threshold_values=thresholds,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )
    threshold_session = document_session.begin_threshold(
        doc=doc,
        hough_threshold=10,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )
    for line_length in line_lengths:
        for line_gap in line_gaps:
            threshold_session.get_or_compute(
                doc=doc,
                hough_threshold=10,
                hough_line_length=line_length,
                hough_line_gap=line_gap,
                hough_seed=1,
                align_abs_min_len=8.0,
                align_min_iou_threshold=0.035,
                compute_payload=lambda line_length=line_length, line_gap=line_gap: _payload_for(line_length, line_gap, 1),
            )
    threshold_session.close()
    document_session.submit_completed_document_write()
    auto_cache.close()

    read_only_cache = RefToRefCombinationCache(cache_dir=tmp_path, mode="read-only")
    read_only_document_session = read_only_cache.begin_document(
        doc=doc,
        threshold_values=thresholds,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )
    read_only_threshold_session = read_only_document_session.begin_threshold(
        doc=doc,
        hough_threshold=10,
        line_length_values=line_lengths,
        line_gap_values=line_gaps,
        seed_values=seeds,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
    )
    cached_payload = read_only_threshold_session.get_or_compute(
        doc=doc,
        hough_threshold=10,
        hough_line_length=6,
        hough_line_gap=1,
        hough_seed=1,
        align_abs_min_len=8.0,
        align_min_iou_threshold=0.035,
        compute_payload=lambda: (_ for _ in ()).throw(AssertionError("document cache should be reused")),
    )

    assert cached_payload["ref_to_ref_cache_hit"] is True
    assert cached_payload["timing_hough_detect_seconds"] == 0.0
    assert cached_payload["refref_y"].tolist() == _payload_for(6, 1, 1)["refref_y"].tolist()
    assert read_only_cache.stats.as_dict()["document_pack_hits"] == 1
    assert read_only_cache.stats.as_dict()["hits"] == 1
