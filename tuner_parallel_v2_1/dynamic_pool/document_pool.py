from __future__ import annotations

"""Atomic file-backed document leasing for dynamic Slurm scheduling.

This module deliberately stores *only* scheduling metadata.  The pool knows
which document index/name is available, claimed, done, or failed; it does not
store tuning scores, best combinations, metric rows, or visualisation bundles.
Those outputs remain owned by the existing tuner code paths.
"""

from collections.abc import Callable, Iterable, Iterator
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any


LogFn = Callable[[str], None]


AVAILABLE_DIR_NAME = "available"
CLAIMED_DIR_NAME = "claimed"
DONE_DIR_NAME = "done"
FAILED_DIR_NAME = "failed"
EVENTS_FILE_NAME = "events.jsonl"
SELECTED_DOCUMENTS_FILE_NAME = "selected_documents.jsonl"
POOL_MANIFEST_FILE_NAME = "document_pool_manifest.json"


def _utc_timestamp() -> str:
    """Return an ISO timestamp that is stable across Slurm worker machines."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _safe_worker_id(worker_id: str) -> str:
    """Return a worker id that is safe to embed in pool state filenames."""
    cleaned_worker_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(worker_id).strip())
    return cleaned_worker_id or f"pid_{os.getpid()}"


def _document_state_filename(pool_ordinal: int) -> str:
    """Return the canonical scheduling filename for one selected document."""
    return f"document_{int(pool_ordinal):06d}.json"


def _write_json_atomically(output_path: Path, payload: dict[str, Any]) -> None:
    """Write JSON via a same-directory temporary file and atomic replace."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp.{os.getpid()}")
    temporary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary_path, output_path)


def _append_json_line_safely(events_path: Path, payload: dict[str, Any]) -> None:
    """Append one best-effort JSONL event without letting logging break work."""
    try:
        events_path.parent.mkdir(parents=True, exist_ok=True)
        with events_path.open("a", encoding="utf-8") as events_handle:
            events_handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except OSError:
        # Event logging is useful for monitoring, but scheduling correctness is
        # provided by atomic file moves.  A transient event-log write problem
        # should not stop an otherwise valid document claim.
        return


def _read_json_file(input_path: Path) -> dict[str, Any]:
    """Load one small scheduling JSON file."""
    with Path(input_path).open("r", encoding="utf-8") as input_handle:
        payload = json.load(input_handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {input_path}")
    return payload


def _document_payload_from_selected_item(*, selected_item: dict[str, Any], pool_ordinal: int) -> dict[str, Any]:
    """Build the minimal scheduling payload for one selected runfile item."""
    return {
        "pool_ordinal": int(pool_ordinal),
        "runfile_index": int(selected_item["index"]),
        "fname": Path(str(selected_item["fname"])).name,
    }


@dataclass(frozen=True)
class DocumentLease:
    """One document claimed by one worker from the shared pool."""

    pool_ordinal: int
    runfile_index: int
    fname: str
    worker_id: str
    claimed_path: Path
    claimed_at: str

    def scheduling_payload(self) -> dict[str, Any]:
        """Return scheduling-only metadata for status files and event logs."""
        payload = asdict(self)
        payload["claimed_path"] = str(self.claimed_path)
        return payload


@dataclass(frozen=True)
class DocumentPoolInitializationSummary:
    """Small human-readable summary returned after pool creation."""

    pool_dir: Path
    selected_document_count: int
    available_dir: Path
    selected_documents_jsonl: Path
    manifest_json: Path

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary for the launcher log."""
        return {
            "pool_dir": str(self.pool_dir),
            "selected_document_count": int(self.selected_document_count),
            "available_dir": str(self.available_dir),
            "selected_documents_jsonl": str(self.selected_documents_jsonl),
            "manifest_json": str(self.manifest_json),
        }


def initialize_document_pool(
    *,
    pool_dir: Path,
    selected_run_items: Iterable[dict[str, Any]],
    runfile_json: Path,
    max_items: int | None,
    log_fn: LogFn | None = None,
) -> DocumentPoolInitializationSummary:
    """Create a fresh scheduling-only pool from selected runfile records.

    The initializer fails when an existing pool is already present.  That is
    intentional: silently reusing old available/claimed/done files could skip
    documents or mix two different parameter runs.
    """
    log = (lambda _message: None) if log_fn is None else log_fn
    pool_dir = Path(pool_dir)
    available_dir = pool_dir / AVAILABLE_DIR_NAME
    claimed_dir = pool_dir / CLAIMED_DIR_NAME
    done_dir = pool_dir / DONE_DIR_NAME
    failed_dir = pool_dir / FAILED_DIR_NAME
    selected_documents_jsonl = pool_dir / SELECTED_DOCUMENTS_FILE_NAME
    manifest_json = pool_dir / POOL_MANIFEST_FILE_NAME

    existing_state_paths = [
        path
        for status_dir in (available_dir, claimed_dir, done_dir, failed_dir)
        if status_dir.exists()
        for path in status_dir.glob("*.json")
    ]
    if selected_documents_jsonl.exists() or manifest_json.exists() or existing_state_paths:
        raise FileExistsError(
            "Dynamic document pool already exists. Use a fresh --output-dir "
            f"or remove the old pool deliberately before relaunching: {pool_dir}"
        )

    for status_dir in (available_dir, claimed_dir, done_dir, failed_dir):
        status_dir.mkdir(parents=True, exist_ok=True)

    selected_count = 0
    with selected_documents_jsonl.open("w", encoding="utf-8") as selected_handle:
        for pool_ordinal, selected_item in enumerate(selected_run_items):
            payload = _document_payload_from_selected_item(
                selected_item=selected_item,
                pool_ordinal=int(pool_ordinal),
            )
            selected_handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
            _write_json_atomically(available_dir / _document_state_filename(int(pool_ordinal)), payload)
            selected_count += 1

    manifest_payload = {
        "schema_version": "tuner_dynamic_document_pool_v1",
        "created_at": _utc_timestamp(),
        "runfile_json": str(Path(runfile_json)),
        "max_items": None if max_items is None else int(max_items),
        "selected_document_count": int(selected_count),
        "note": "Scheduling metadata only; tuner metrics remain in shard output files.",
    }
    _write_json_atomically(manifest_json, manifest_payload)

    summary = DocumentPoolInitializationSummary(
        pool_dir=pool_dir,
        selected_document_count=int(selected_count),
        available_dir=available_dir,
        selected_documents_jsonl=selected_documents_jsonl,
        manifest_json=manifest_json,
    )
    log(f"[dynamic-pool] initialized documents={selected_count} pool={pool_dir}")
    return summary


class DocumentLeasePool:
    """Shared file-backed queue used by all dynamic Slurm workers."""

    def __init__(self, *, pool_dir: Path, worker_id: str, log_fn: LogFn | None = None) -> None:
        """Create a worker-local view of the shared scheduling pool."""
        self.pool_dir = Path(pool_dir)
        self.worker_id = _safe_worker_id(worker_id)
        self.log = (lambda _message: None) if log_fn is None else log_fn
        self.available_dir = self.pool_dir / AVAILABLE_DIR_NAME
        self.claimed_dir = self.pool_dir / CLAIMED_DIR_NAME
        self.done_dir = self.pool_dir / DONE_DIR_NAME
        self.failed_dir = self.pool_dir / FAILED_DIR_NAME
        self.events_path = self.pool_dir / EVENTS_FILE_NAME

        for required_dir in (self.available_dir, self.claimed_dir, self.done_dir, self.failed_dir):
            if not required_dir.is_dir():
                raise FileNotFoundError(f"Dynamic document pool is missing directory: {required_dir}")

    def _event(self, event_name: str, payload: dict[str, Any]) -> None:
        """Write one scheduling event for lightweight live monitoring."""
        event_payload = {
            "event": str(event_name),
            "timestamp": _utc_timestamp(),
            "worker_id": self.worker_id,
            **payload,
        }
        _append_json_line_safely(self.events_path, event_payload)

    def claim_next_available_document(self) -> DocumentLease | None:
        """Atomically claim one available document, or return None if empty."""
        for available_path in sorted(self.available_dir.glob("document_*.json")):
            claimed_path = self.claimed_dir / (
                f"{available_path.stem}.{self.worker_id}.pid_{os.getpid()}.json"
            )
            try:
                # Same-filesystem rename is atomic: exactly one worker can move
                # this available file into claimed state.
                available_path.rename(claimed_path)
            except FileNotFoundError:
                # Another worker won the race for this file; try the next one.
                continue
            except OSError:
                # A transient filesystem race should not kill the worker while
                # other available documents may still exist.
                continue

            payload = _read_json_file(claimed_path)
            claimed_at = _utc_timestamp()
            lease = DocumentLease(
                pool_ordinal=int(payload["pool_ordinal"]),
                runfile_index=int(payload["runfile_index"]),
                fname=str(payload["fname"]),
                worker_id=self.worker_id,
                claimed_path=claimed_path,
                claimed_at=claimed_at,
            )
            claimed_payload = {
                **payload,
                "worker_id": self.worker_id,
                "hostname": socket.gethostname(),
                "pid": int(os.getpid()),
                "claimed_at": claimed_at,
            }
            _write_json_atomically(claimed_path, claimed_payload)
            self._event("claim", lease.scheduling_payload())
            self.log(
                f"[dynamic-pool-claim] worker={self.worker_id} "
                f"pool_ordinal={lease.pool_ordinal} runfile_index={lease.runfile_index} fname={lease.fname}"
            )
            return lease

        self._event("empty", {"message": "no available document files remain"})
        return None

    def mark_lease_done(self, lease: DocumentLease) -> None:
        """Move one completed lease into done state after tuner outputs exist."""
        done_path = self.done_dir / _document_state_filename(lease.pool_ordinal)
        done_payload = {
            **lease.scheduling_payload(),
            "completed_at": _utc_timestamp(),
            "status": "done",
        }
        _write_json_atomically(done_path, done_payload)
        try:
            Path(lease.claimed_path).unlink()
        except FileNotFoundError:
            pass
        self._event("done", done_payload)
        self.log(
            f"[dynamic-pool-done] worker={self.worker_id} "
            f"pool_ordinal={lease.pool_ordinal} runfile_index={lease.runfile_index} fname={lease.fname}"
        )

    def mark_lease_failed(self, lease: DocumentLease, *, reason: str) -> None:
        """Move one lease into failed state without storing tuner metrics."""
        failed_path = self.failed_dir / _document_state_filename(lease.pool_ordinal)
        failed_payload = {
            **lease.scheduling_payload(),
            "failed_at": _utc_timestamp(),
            "status": "failed",
            "reason": str(reason),
        }
        _write_json_atomically(failed_path, failed_payload)
        try:
            Path(lease.claimed_path).unlink()
        except FileNotFoundError:
            pass
        self._event("failed", failed_payload)
        self.log(
            f"[dynamic-pool-failed] worker={self.worker_id} "
            f"pool_ordinal={lease.pool_ordinal} runfile_index={lease.runfile_index} "
            f"fname={lease.fname} reason={reason}"
        )


def iter_claimed_selected_run_items_from_pool(
    *,
    document_pool: DocumentLeasePool,
    selected_run_items: list[dict[str, Any]],
    active_lease_by_document_index: dict[int, DocumentLease],
    log_fn: LogFn | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield selected runfile items by claiming one free document at a time.

    The existing tuner scheduler pulls from this iterator only when it has an
    open document slot.  Therefore a Slurm worker claims exactly one replacement
    document immediately after one active document completes.
    """
    log = (lambda _message: None) if log_fn is None else log_fn

    while True:
        lease = document_pool.claim_next_available_document()
        if lease is None:
            log("[dynamic-pool-empty] no more free documents for this worker")
            return

        if lease.pool_ordinal < 0 or lease.pool_ordinal >= len(selected_run_items):
            document_pool.mark_lease_failed(
                lease,
                reason=(
                    "pool ordinal is outside the selected run item list; "
                    "the pool and runfile selection do not match"
                ),
            )
            raise IndexError(
                f"Dynamic pool ordinal {lease.pool_ordinal} is outside selected item count {len(selected_run_items)}"
            )

        selected_run_item = dict(selected_run_items[int(lease.pool_ordinal)])
        document_index = int(selected_run_item["index"])
        if document_index in active_lease_by_document_index:
            document_pool.mark_lease_failed(
                lease,
                reason=f"runfile index {document_index} is already active in this worker",
            )
            raise RuntimeError(f"Duplicate active dynamic-pool lease for runfile index {document_index}")

        # Keep the lease in memory until the document finishes.  The completion
        # callback removes it from this map and finalizes it after normal tuner
        # outputs have been written.
        active_lease_by_document_index[document_index] = lease
        yield selected_run_item


def exit_with_error(message: str) -> None:
    """Print one CLI error and exit with a non-zero status."""
    print(f"[dynamic-pool-error] {message}", file=sys.stderr)
    raise SystemExit(2)
