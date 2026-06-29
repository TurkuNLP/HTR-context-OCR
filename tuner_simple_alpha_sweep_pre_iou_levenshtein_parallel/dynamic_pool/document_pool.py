from __future__ import annotations

import json
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


POOL_STATE_DIRECTORIES = ("available", "claimed", "done", "failed")


@dataclass(frozen=True)
class DocumentLease:
    """Describe one document file that has been claimed by one worker."""

    pool_ordinal: int
    document_index: int
    filename: str
    available_path: Path
    claimed_path: Path
    payload: dict[str, Any]


class DocumentPool:
    """Coordinate document claims through atomic file moves on the shared filesystem."""

    def __init__(self, pool_dir: Path) -> None:
        self.pool_dir = Path(pool_dir)
        self.available_dir = self.pool_dir / "available"
        self.claimed_dir = self.pool_dir / "claimed"
        self.done_dir = self.pool_dir / "done"
        self.failed_dir = self.pool_dir / "failed"
        self.events_path = self.pool_dir / "events.jsonl"

    def ensure_state_directories(self) -> None:
        """Create every state directory before workers try to claim documents."""

        self.pool_dir.mkdir(parents=True, exist_ok=True)
        for state_directory_name in POOL_STATE_DIRECTORIES:
            state_directory_path = self.pool_dir / state_directory_name
            state_directory_path.mkdir(parents=True, exist_ok=True)

    def write_event(self, event_type: str, payload: Mapping[str, Any]) -> None:
        """Append a small JSON event so the run can be audited after failures."""

        event_record = {
            "event_type": event_type,
            "timestamp_unix_seconds": time.time(),
            "hostname": socket.gethostname(),
            "process_id": os.getpid(),
            **dict(payload),
        }
        with self.events_path.open("a", encoding="utf-8") as event_handle:
            event_handle.write(json.dumps(event_record, sort_keys=True) + "\n")

    def claim_next_available_document(self, worker_id: str) -> DocumentLease | None:
        """Claim the lowest available document by renaming its JSON file into claimed/."""

        self.ensure_state_directories()
        for available_path in sorted(self.available_dir.glob("document_*.json")):
            claimed_path = self.claimed_dir / f"{available_path.stem}__worker_{safe_path_token(worker_id)}__pid_{os.getpid()}.json"
            try:
                available_path.rename(claimed_path)
            except FileNotFoundError:
                continue
            except OSError:
                continue
            payload = read_json_file(claimed_path)
            lease = DocumentLease(
                pool_ordinal=int(payload["pool_ordinal"]),
                document_index=int(payload["document_index"]),
                filename=str(payload["filename"]),
                available_path=available_path,
                claimed_path=claimed_path,
                payload=dict(payload),
            )
            self.write_event(
                "claimed",
                {
                    "worker_id": worker_id,
                    "pool_ordinal": lease.pool_ordinal,
                    "document_index": lease.document_index,
                    "filename": lease.filename,
                    "claimed_path": str(claimed_path),
                },
            )
            return lease
        return None

    def mark_done(self, lease: DocumentLease, worker_id: str) -> Path:
        """Move a claimed document to done/ after its CSV row is safely written."""

        done_path = self.done_dir / done_filename_for_lease(lease)
        lease.claimed_path.rename(done_path)
        self.write_event(
            "done",
            {
                "worker_id": worker_id,
                "pool_ordinal": lease.pool_ordinal,
                "document_index": lease.document_index,
                "filename": lease.filename,
                "done_path": str(done_path),
            },
        )
        return done_path

    def mark_failed(self, lease: DocumentLease, worker_id: str, reason: str) -> Path:
        """Move a claimed document to failed/ when the worker cannot finish it."""

        failed_path = self.failed_dir / failed_filename_for_lease(lease)
        lease.claimed_path.rename(failed_path)
        self.write_event(
            "failed",
            {
                "worker_id": worker_id,
                "pool_ordinal": lease.pool_ordinal,
                "document_index": lease.document_index,
                "filename": lease.filename,
                "failed_path": str(failed_path),
                "reason": reason,
            },
        )
        return failed_path

    def requeue_claimed_documents(self) -> int:
        """Move documents left in claimed/ back to available/ before a resumed run."""

        moved_count = 0
        self.ensure_state_directories()
        for claimed_path in sorted(self.claimed_dir.glob("document_*.json")):
            payload = read_json_file(claimed_path)
            available_path = self.available_dir / document_filename(int(payload["pool_ordinal"]))
            claimed_path.rename(available_path)
            moved_count += 1
        if moved_count:
            self.write_event("requeued_claimed", {"document_count": moved_count})
        return moved_count

    def requeue_failed_documents(self) -> int:
        """Move documents in failed/ back to available/ when the user wants retries."""

        moved_count = 0
        self.ensure_state_directories()
        for failed_path in sorted(self.failed_dir.glob("document_*.json")):
            payload = read_json_file(failed_path)
            available_path = self.available_dir / document_filename(int(payload["pool_ordinal"]))
            failed_path.rename(available_path)
            moved_count += 1
        if moved_count:
            self.write_event("requeued_failed", {"document_count": moved_count})
        return moved_count

    def state_counts(self) -> dict[str, int]:
        """Count documents in every state directory for status logging."""

        self.ensure_state_directories()
        counts: dict[str, int] = {}
        for state_directory_name in POOL_STATE_DIRECTORIES:
            state_directory_path = self.pool_dir / state_directory_name
            counts[state_directory_name] = sum(1 for _ in state_directory_path.glob("document_*.json"))
        return counts


def document_filename(pool_ordinal: int) -> str:
    """Return the stable filename used by all pool state directories."""

    return f"document_{int(pool_ordinal):06d}.json"


def done_filename_for_lease(lease: DocumentLease) -> str:
    """Return the final done/ filename while preserving the document order number."""

    return document_filename(lease.pool_ordinal)


def failed_filename_for_lease(lease: DocumentLease) -> str:
    """Return the final failed/ filename while preserving the document order number."""

    return document_filename(lease.pool_ordinal)


def safe_path_token(raw_value: str) -> str:
    """Convert a worker identifier into a conservative filename fragment."""

    cleaned_characters = []
    for character in str(raw_value):
        if character.isalnum() or character in ("-", "_"):
            cleaned_characters.append(character)
        else:
            cleaned_characters.append("_")
    cleaned_value = "".join(cleaned_characters).strip("_")
    return cleaned_value or "worker"


def read_json_file(path: Path) -> dict[str, Any]:
    """Load one small JSON file from the pool."""

    with Path(path).open("r", encoding="utf-8") as input_handle:
        return dict(json.load(input_handle))


def write_json_file_atomically(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON through a temporary file so workers never see a half-written file."""

    destination_path = Path(path)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination_path.with_name(f".{destination_path.name}.tmp.{os.getpid()}")
    with temporary_path.open("w", encoding="utf-8") as output_handle:
        json.dump(dict(payload), output_handle, indent=2, sort_keys=True)
        output_handle.write("\n")
    temporary_path.replace(destination_path)
