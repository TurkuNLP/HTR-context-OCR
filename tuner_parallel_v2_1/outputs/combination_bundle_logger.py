from __future__ import annotations

"""Optional per-document binary bundles for Hough tuner visualization.

The normal tuner output keeps compact aggregate metrics.  This writer is only
constructed when a caller explicitly asks for visualization bundles, and it
records only the geometry that ``tools/language_hough_parameter_metric_analysis.py``
currently consumes: raw ref-to-pred Hough lines, final surviving ref-to-pred
lines, scalar metrics, and reusable score-matrix assets.

This module is deliberately observational.  It does not rank combinations, does
not mutate evaluator payloads, and does not change final tuner metrics.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
import time

import numpy as np

try:
    from .combination_bundle_records import PICKLE_STREAM_SUFFIX, serialize_pickle_stream_record
except ImportError:
    from outputs.combination_bundle_records import PICKLE_STREAM_SUFFIX, serialize_pickle_stream_record  # type: ignore

try:
    from ..tuner.tuner_config import LogFn, SweepDocument
except ImportError:
    from tuner.tuner_config import LogFn, SweepDocument  # type: ignore


SCHEMA_VERSION = "tuner_combination_bundle_v2"
VALID_BUNDLE_SCOPES = {"none", "all", "valid-only", "invalid-only", "winner-only"}
BUNDLE_FILE_BUFFER_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_PENDING_DOCUMENT_WRITES = 2


def _no_log(_: str) -> None:
    """No-op logger used when the caller does not provide a log hook."""
    return


def _safe_path_component(value: str, *, max_length: int = 120) -> str:
    """Return a stable filesystem-safe path component for one document name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not cleaned:
        cleaned = "document"
    if len(cleaned) <= int(max_length):
        return cleaned
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, int(max_length) - len(digest) - 1)
    return f"{cleaned[:prefix_length]}_{digest}"


def _combination_is_valid(eval_row: dict) -> bool:
    """Return True when the evaluator marked this row as valid and scored it."""
    if not bool(eval_row.get("is_valid", True)):
        return False
    score = eval_row.get("tuning_score")
    if score is None:
        return False
    try:
        return bool(math.isfinite(float(score)))
    except Exception:
        return False


class CombinationBundleLogger:
    """Build combination records in memory and write completed documents later."""

    def __init__(
        self,
        *,
        root_dir: Path,
        scope: str,
        include_candidate_lines: bool,
        shard_index: int | None = None,
        selection_index_range: tuple[int, int] | None = None,
        log_fn: LogFn | None = None,
        max_pending_document_writes: int = DEFAULT_MAX_PENDING_DOCUMENT_WRITES,
    ) -> None:
        """Create an optional bundle logger.

        ``scope`` decides which combinations are emitted.  The all/valid/invalid
        modes build threshold-local record lists, while ``winner-only`` is
        filled by the scheduler after it knows the best row for a document.
        Disk I/O remains outside the combination hot loop in every mode.
        """
        normalized_scope = str(scope).strip().lower()
        if normalized_scope not in VALID_BUNDLE_SCOPES:
            raise ValueError(
                f"Unsupported combination bundle scope {scope!r}; "
                f"expected one of {sorted(VALID_BUNDLE_SCOPES)!r}"
            )

        self.root_dir = Path(root_dir)
        self.scope = normalized_scope
        # The current visualization intentionally ignores candidate lines.  The
        # flag is retained for CLI compatibility, but the lean v2 bundle schema
        # does not write candidate geometry because it is not consumed anywhere.
        self.include_candidate_lines = bool(include_candidate_lines)
        self.shard_index = None if shard_index is None else int(shard_index)
        self.selection_index_range = selection_index_range
        self.log = _no_log if log_fn is None else log_fn
        self.max_pending_document_writes = max(1, int(max_pending_document_writes))

        self._lock = Lock()
        self._writer_executor: ThreadPoolExecutor | None = None
        self._pending_document_write_futures: list[Future] = []
        self._records_written = 0
        self._payload_bytes_written = 0
        self._pickle_serialization_seconds = 0.0
        self._file_write_seconds = 0.0
        self._asset_write_seconds = 0.0
        self._document_write_seconds = 0.0

        if self.scope != "none":
            self.root_dir.mkdir(parents=True, exist_ok=True)
            self._writer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="combination_bundle_document_writer",
            )

    def close(self) -> None:
        """Wait for all completed-document bundle writes and log final totals."""
        self._wait_for_all_pending_document_writes()
        if self._writer_executor is not None:
            self._writer_executor.shutdown(wait=True)
            self._writer_executor = None

        records_written = int(self._records_written)
        payload_bytes_written = int(self._payload_bytes_written)
        pickle_serialization_seconds = float(self._pickle_serialization_seconds)
        file_write_seconds = float(self._file_write_seconds)
        asset_write_seconds = float(self._asset_write_seconds)
        document_write_seconds = float(self._document_write_seconds)

        if records_written > 0:
            self.log(
                "[combination-bundles-summary] "
                f"records={records_written} payload_bytes={payload_bytes_written} "
                f"pickle_serialization_s={pickle_serialization_seconds:.3f} "
                f"file_write_s={file_write_seconds:.3f} asset_write_s={asset_write_seconds:.3f} "
                f"document_write_s={document_write_seconds:.3f} "
                f"format=pickle_stream schema={SCHEMA_VERSION}"
            )

    def should_write(self, eval_row: dict) -> bool:
        """Return True when the configured scope wants this evaluated row."""
        if self.scope == "none":
            return False
        if self.scope == "winner-only":
            return False
        is_valid = _combination_is_valid(eval_row)
        if self.scope == "all":
            return True
        if self.scope == "valid-only":
            return is_valid
        if self.scope == "invalid-only":
            return not is_valid
        return False

    def is_winner_only(self) -> bool:
        """Return True when only the document-winning geometry should be saved."""
        return self.scope == "winner-only"

    def document_dir(self, doc: SweepDocument) -> Path:
        """Return the bundle directory for one document."""
        safe_name = _safe_path_component(doc.fname)
        return self.root_dir / f"document_{int(doc.index):06d}_{safe_name}"

    def _document_metadata(self, doc: SweepDocument) -> dict:
        """Build the stable metadata JSON written beside one document bundle."""
        return {
            "schema_version": SCHEMA_VERSION,
            "record_format": "pickle_stream",
            "document": {
                "index": int(doc.index),
                "fname": str(doc.fname),
                "ref_text_len": int(len(doc.ref)),
                "pred_text_len": int(len(doc.pred)),
                "window_size": int(doc.window_size),
                "window_stride": int(doc.window_stride),
                "ref_to_pred_matrix_shape": [int(value) for value in doc.ref_to_pred_matrix.shape],
                "ref_to_ref_matrix_shape": [int(value) for value in doc.ref_to_ref_matrix.shape],
                "whole_document_nls": float(doc.whole_document_nls),
            },
            "score_matrices": {
                "ref_to_pred": "ref_to_pred_score_matrix.npy",
                "ref_to_ref": "ref_to_ref_score_matrix.npy",
            },
            "threshold_record_pattern": f"threshold_XXX{PICKLE_STREAM_SUFFIX}",
        }

    def _ref_to_pred_visualization_payload(self, payload: dict) -> dict:
        """Extract the exact ref-to-pred geometry consumed by visualization."""
        det = payload.get("det", {}) if isinstance(payload, dict) else {}
        filtered = payload.get("filtered", {}) if isinstance(payload, dict) else {}
        return {
            "hough_detection": {
                "threshold_start": det.get("threshold_start"),
                "raw_lines": det.get("raw_lines", []),
            },
            "filtering": {
                "lines_used": filtered.get("lines_used", []),
            },
        }

    def build_combination_record(
        self,
        *,
        doc: SweepDocument,
        hough_threshold: int,
        hough_line_length: int,
        hough_line_gap: int,
        hough_seed: int,
        align_abs_min_len: float,
        align_min_iou_threshold: float,
        eval_row: dict,
        ref_to_pred_payload: dict,
        ref_to_ref_payload: dict,
        force: bool = False,
    ) -> dict | None:
        """Build one lean combination record without touching the filesystem."""
        if not bool(force) and not self.should_write(eval_row):
            return None

        # The ref_to_ref payload is intentionally not written: the current
        # language visualization only displays the ref_to_ref score matrix with
        # no Hough overlays, and metrics already contain the scalar counts.
        return {
            "schema_version": SCHEMA_VERSION,
            "record_format": "pickle_stream",
            "shard": {
                "shard_index": self.shard_index,
                "selection_index_start": (
                    None if self.selection_index_range is None else int(self.selection_index_range[0])
                ),
                "selection_index_end": (
                    None if self.selection_index_range is None else int(self.selection_index_range[1])
                ),
            },
            "document": {
                "index": int(doc.index),
                "fname": str(doc.fname),
                "window_size": int(doc.window_size),
                "window_stride": int(doc.window_stride),
                "whole_document_nls": float(doc.whole_document_nls),
            },
            "hough_parameters": {
                "hough_threshold": int(hough_threshold),
                "hough_line_length": int(hough_line_length),
                "hough_line_gap": int(hough_line_gap),
                "hough_seed": int(hough_seed),
                "effective_hough_seed": int(hough_seed) + int(doc.index),
                "align_abs_min_len": float(align_abs_min_len),
                "align_min_iou_threshold": float(align_min_iou_threshold),
            },
            "metrics": eval_row,
            "ref_to_pred": self._ref_to_pred_visualization_payload(ref_to_pred_payload),
        }

    def submit_completed_document(
        self,
        *,
        doc: SweepDocument,
        records_by_threshold: dict[int, list[dict]],
    ) -> None:
        """Queue one completed document bundle for background writing."""
        if self.scope == "none" or self._writer_executor is None:
            return

        # Drop empty threshold lists before submitting so documents with no
        # emitted records do not create empty bundle directories.
        non_empty_records_by_threshold = {
            int(threshold): list(records)
            for threshold, records in records_by_threshold.items()
            if records
        }
        if not non_empty_records_by_threshold:
            return

        future = self._writer_executor.submit(
            self._write_completed_document,
            doc,
            non_empty_records_by_threshold,
        )
        with self._lock:
            self._pending_document_write_futures.append(future)
        self._wait_until_pending_document_writes_are_bounded()

    def _wait_until_pending_document_writes_are_bounded(self) -> None:
        """Keep at most a small number of completed document buckets in memory."""
        while True:
            with self._lock:
                pending_futures = [future for future in self._pending_document_write_futures if not future.done()]
                completed_futures = [
                    future for future in self._pending_document_write_futures if future.done()
                ]
                self._pending_document_write_futures = pending_futures

            for completed_future in completed_futures:
                # Surface writer failures immediately instead of hiding them
                # until the final visualization job tries to read missing files.
                completed_future.result()

            if len(pending_futures) <= self.max_pending_document_writes:
                return

            done_futures, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
            for done_future in done_futures:
                done_future.result()

    def _wait_for_all_pending_document_writes(self) -> None:
        """Wait until every submitted document bundle has been written."""
        while True:
            with self._lock:
                pending_futures = list(self._pending_document_write_futures)
                self._pending_document_write_futures.clear()
            if not pending_futures:
                return
            for future in pending_futures:
                future.result()

    def _write_completed_document(
        self,
        doc: SweepDocument,
        records_by_threshold: dict[int, list[dict]],
    ) -> None:
        """Write one document directory after all its combinations finish."""
        document_started_at = time.perf_counter()
        document_dir = self.document_dir(doc)
        document_dir.mkdir(parents=True, exist_ok=True)

        asset_started_at = time.perf_counter()
        (document_dir / "document_metadata.json").write_text(
            json.dumps(self._document_metadata(doc), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        np.save(document_dir / "ref_to_pred_score_matrix.npy", np.asarray(doc.ref_to_pred_matrix))
        np.save(document_dir / "ref_to_ref_score_matrix.npy", np.asarray(doc.ref_to_ref_matrix))
        asset_write_seconds = time.perf_counter() - asset_started_at

        records_written = 0
        payload_bytes_written = 0
        pickle_serialization_seconds = 0.0
        file_write_seconds = 0.0

        for threshold in sorted(int(value) for value in records_by_threshold):
            record_path = document_dir / f"threshold_{int(threshold):03d}{PICKLE_STREAM_SUFFIX}"
            records_for_threshold = records_by_threshold[int(threshold)]

            with record_path.open("wb", buffering=BUNDLE_FILE_BUFFER_BYTES) as file_handle:
                for record in records_for_threshold:
                    serialization_started_at = time.perf_counter()
                    encoded_record = serialize_pickle_stream_record(record)
                    pickle_serialization_seconds += time.perf_counter() - serialization_started_at

                    write_started_at = time.perf_counter()
                    file_handle.write(encoded_record)
                    file_write_seconds += time.perf_counter() - write_started_at
                    records_written += 1
                    payload_bytes_written += int(len(encoded_record))

        with self._lock:
            self._records_written += int(records_written)
            self._payload_bytes_written += int(payload_bytes_written)
            self._pickle_serialization_seconds += float(pickle_serialization_seconds)
            self._file_write_seconds += float(file_write_seconds)
            self._asset_write_seconds += float(asset_write_seconds)
            self._document_write_seconds += float(time.perf_counter() - document_started_at)


__all__ = [
    "BUNDLE_FILE_BUFFER_BYTES",
    "CombinationBundleLogger",
    "DEFAULT_MAX_PENDING_DOCUMENT_WRITES",
    "SCHEMA_VERSION",
    "VALID_BUNDLE_SCOPES",
]
