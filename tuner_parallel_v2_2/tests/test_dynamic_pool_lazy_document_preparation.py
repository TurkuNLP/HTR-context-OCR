from __future__ import annotations

"""Regression tests for dynamic-pool document claiming.

The dynamic scheduler must claim documents only when a local document slot asks
for work.  A previous eager list conversion in document preparation consumed the
entire dynamic iterator and moved every document into claimed/ immediately.
"""

import json
from pathlib import Path

from tuner_parallel_v2_2.dynamic_pool.document_pool import (
    DocumentLeasePool,
    initialize_document_pool,
    iter_claimed_selected_run_items_from_pool,
)
from tuner_parallel_v2_2.hough_preprocessing import HoughPreprocessingConfig
from tuner_parallel_v2_2.matrices.document_prep import iter_prepared_documents_from_items


def _count_document_state_files(pool_dir: Path, state_name: str) -> int:
    """Return how many document JSON files are currently in one pool state."""
    return len(list((pool_dir / state_name).glob("document_*.json")))


def _tiny_selected_run_item(index: int) -> dict:
    """Build one tiny runfile-like item that is cheap to prepare in a test."""
    tiny_reference = f"reference text {index}"
    tiny_prediction = f"prediction text {index}"
    return {
        "index": int(index),
        "fname": f"tiny_document_{index:03d}.jpeg",
        "ref": tiny_reference,
        "pred": tiny_prediction,
    }


def test_dynamic_pool_preparation_claims_only_one_document_per_next(tmp_path: Path) -> None:
    """The first prepared-document request must not claim the whole pool."""
    selected_run_items = [_tiny_selected_run_item(index) for index in range(3)]
    pool_dir = tmp_path / "document_pool"

    initialize_document_pool(
        pool_dir=pool_dir,
        selected_run_items=selected_run_items,
        runfile_json=tmp_path / "outputs.json",
        max_items=len(selected_run_items),
    )
    document_pool = DocumentLeasePool(pool_dir=pool_dir, worker_id="worker_lazy_test")
    active_leases_by_document_index: dict[int, object] = {}

    lazy_claimed_items = iter_claimed_selected_run_items_from_pool(
        document_pool=document_pool,
        selected_run_items=selected_run_items,
        active_lease_by_document_index=active_leases_by_document_index,
    )
    prepared_documents = iter_prepared_documents_from_items(
        selected_run_items=lazy_claimed_items,
        window_size=4,
        window_stride=2,
        levenshtein_backend="python",
        hough_preprocessing_config=HoughPreprocessingConfig(
            minimum_score_floor=0.0,
            maximum_active_fraction=1.0,
            minimum_active_cells=1,
            minimum_active_rows=1,
            minimum_active_columns=1,
            minimum_x_span=1,
            minimum_y_span=1,
        ),
        disable_pkl_matrix_source=True,
        raise_when_no_documents_selected=False,
    )

    assert _count_document_state_files(pool_dir, "available") == 3
    assert _count_document_state_files(pool_dir, "claimed") == 0

    first_prepared_document = next(prepared_documents)

    assert first_prepared_document.fname == "tiny_document_000.jpeg"
    assert _count_document_state_files(pool_dir, "available") == 2
    assert _count_document_state_files(pool_dir, "claimed") == 1
    assert sorted(active_leases_by_document_index) == [0]

    second_prepared_document = next(prepared_documents)

    assert second_prepared_document.fname == "tiny_document_001.jpeg"
    assert _count_document_state_files(pool_dir, "available") == 1
    assert _count_document_state_files(pool_dir, "claimed") == 2
    assert sorted(active_leases_by_document_index) == [0, 1]


def test_dynamic_pool_empty_iterator_can_end_without_worker_failure(tmp_path: Path) -> None:
    """A late-starting dynamic worker should stop cleanly when the pool is empty."""
    pool_dir = tmp_path / "document_pool"

    initialize_document_pool(
        pool_dir=pool_dir,
        selected_run_items=[],
        runfile_json=tmp_path / "outputs.json",
        max_items=0,
    )
    document_pool = DocumentLeasePool(pool_dir=pool_dir, worker_id="worker_empty_test")
    active_leases_by_document_index: dict[int, object] = {}

    lazy_claimed_items = iter_claimed_selected_run_items_from_pool(
        document_pool=document_pool,
        selected_run_items=[],
        active_lease_by_document_index=active_leases_by_document_index,
    )
    prepared_documents = iter_prepared_documents_from_items(
        selected_run_items=lazy_claimed_items,
        window_size=4,
        window_stride=2,
        levenshtein_backend="python",
        hough_preprocessing_config=HoughPreprocessingConfig(
            minimum_score_floor=0.0,
            maximum_active_fraction=1.0,
            minimum_active_cells=1,
            minimum_active_rows=1,
            minimum_active_columns=1,
            minimum_x_span=1,
            minimum_y_span=1,
        ),
        disable_pkl_matrix_source=True,
        raise_when_no_documents_selected=False,
    )

    assert list(prepared_documents) == []
    assert active_leases_by_document_index == {}

def test_skipped_document_preparation_writes_diagnostic_bundle(tmp_path: Path) -> None:
    """Skipped documents should leave metadata for later visualisation."""
    timing: dict = {}
    diagnostic_bundle_root = tmp_path / "combination_bundles"
    prepared_documents = iter_prepared_documents_from_items(
        selected_run_items=[
            {
                "index": 42,
                "fname": "empty_prediction.jpeg",
                "ref": "reference text",
                "pred": "",
            }
        ],
        window_size=4,
        window_stride=2,
        levenshtein_backend="python",
        disable_pkl_matrix_source=True,
        raise_when_no_documents_selected=False,
        skip_diagnostic_bundle_dir=diagnostic_bundle_root,
        timing_out=timing,
    )

    assert list(prepared_documents) == []
    skipped_record = timing["skipped_documents"][0]
    diagnostic_bundle_dir = Path(skipped_record["diagnostic_bundle_dir"])
    metadata_path = diagnostic_bundle_dir / "document_metadata.json"

    assert diagnostic_bundle_dir.parent == diagnostic_bundle_root
    assert metadata_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["record_format"] == "skipped_document_diagnostic"
    assert metadata["skip_record"]["skip_reason"] == "no_prediction_text"

