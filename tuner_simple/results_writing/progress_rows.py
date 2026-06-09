from __future__ import annotations

import os
import socket
import time
from typing import Any, Mapping

from tuner_simple.dynamic_pool.document_pool import DocumentLease
from tuner_simple.results_writing.flat_csv_tables import DOCUMENT_RESULT_FIELDNAMES, DOCUMENT_TABLE_FIELDNAMES, SKIPPED_FIELDNAMES
from tuner_simple.serial_runner.document_runner import DocumentRunResult


PROGRESS_SCHEMA_VERSION = "tuner_simple_dynamic_pool_v1"


PROGRESS_CONTROL_FIELDNAMES = [
    "progress_schema_version",
    "pool_ordinal",
    "worker_id",
    "attempt_id",
    "slurm_job_id",
    "hostname",
    "process_id",
    "status",
    "completed_at_unix_seconds",
    "document_elapsed_seconds",
    "panel_path",
    "has_runfile_row",
    "has_loadable_row",
    "has_loaded_row",
    "has_skipped_row",
    "has_result_row",
]


def ordered_unique(fieldnames: list[str]) -> list[str]:
    seen_fieldnames: set[str] = set()
    unique_fieldnames: list[str] = []
    for fieldname in fieldnames:
        if fieldname in seen_fieldnames:
            continue
        seen_fieldnames.add(fieldname)
        unique_fieldnames.append(fieldname)
    return unique_fieldnames


PROGRESS_FIELDNAMES = ordered_unique(
    PROGRESS_CONTROL_FIELDNAMES + DOCUMENT_TABLE_FIELDNAMES + SKIPPED_FIELDNAMES + DOCUMENT_RESULT_FIELDNAMES
)


def build_progress_row(
    *,
    lease: DocumentLease,
    document_result: DocumentRunResult,
    worker_id: str,
    attempt_counter: int,
    panel_path: str | None,
    document_elapsed_seconds: float,
) -> dict[str, Any]:
    progress_row: dict[str, Any] = {
        "progress_schema_version": PROGRESS_SCHEMA_VERSION,
        "pool_ordinal": int(lease.pool_ordinal),
        "worker_id": str(worker_id),
        "attempt_id": f"{worker_id}.{os.getpid()}.{int(lease.pool_ordinal)}.{int(attempt_counter)}",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hostname": socket.gethostname(),
        "process_id": int(os.getpid()),
        "status": "processed" if document_result.result_row is not None else "skipped",
        "completed_at_unix_seconds": float(time.time()),
        "document_elapsed_seconds": float(document_elapsed_seconds),
        "panel_path": str(panel_path or ""),
        "has_runfile_row": "1",
        "has_loadable_row": "1" if document_result.loadable_row is not None else "0",
        "has_loaded_row": "1" if document_result.loaded_row is not None else "0",
        "has_skipped_row": "1" if document_result.skipped_row is not None else "0",
        "has_result_row": "1" if document_result.result_row is not None else "0",
    }
    for source_row in (
        document_result.loadable_row,
        document_result.loaded_row,
        document_result.skipped_row,
        document_result.result_row,
    ):
        if source_row is None:
            continue
        progress_row.update(dict(source_row))
    return progress_row


def truthy_csv_flag(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def extract_fields(source_row: Mapping[str, Any], fieldnames: list[str]) -> dict[str, Any]:
    return {fieldname: source_row.get(fieldname, "") for fieldname in fieldnames}


def progress_row_to_runfile_row(progress_row: Mapping[str, Any]) -> dict[str, Any]:
    return extract_fields(progress_row, DOCUMENT_TABLE_FIELDNAMES)


def progress_row_to_loadable_row(progress_row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not truthy_csv_flag(progress_row.get("has_loadable_row")):
        return None
    return extract_fields(progress_row, DOCUMENT_TABLE_FIELDNAMES)


def progress_row_to_loaded_row(progress_row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not truthy_csv_flag(progress_row.get("has_loaded_row")):
        return None
    return extract_fields(progress_row, DOCUMENT_TABLE_FIELDNAMES)


def progress_row_to_skipped_row(progress_row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not truthy_csv_flag(progress_row.get("has_skipped_row")):
        return None
    return extract_fields(progress_row, SKIPPED_FIELDNAMES)


def progress_row_to_result_row(progress_row: Mapping[str, Any]) -> dict[str, Any] | None:
    if not truthy_csv_flag(progress_row.get("has_result_row")):
        return None
    return extract_fields(progress_row, DOCUMENT_RESULT_FIELDNAMES)
