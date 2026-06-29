from __future__ import annotations

"""Run one dynamic tuner worker that claims documents from a shared pool."""

import gc
import os
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.config.pipeline_config import PipelineConfig
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.document_selection.runfile_loader import RunfileDocument, load_runfile_documents
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.dynamic_pool.document_pool import DocumentLease, DocumentPool
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.matrix_operations.matrix_loader import build_score_matrix_indexes
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.plotting.atomic_panel_writer import render_atomic_document_panel
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.results_writing.locked_csv_bucket import LockedCsvBucketWriter
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.results_writing.progress_rows import PROGRESS_FIELDNAMES, build_progress_row
from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.serial_runner.document_runner import process_one_document


@dataclass
class CompletedDocumentForFlush:
    """Keep one completed lease together with the progress row that must be written."""

    lease: DocumentLease
    progress_row: dict[str, Any]


def atomic_output_dir_from_config(config: PipelineConfig) -> Path:
    """Return the directory where dynamic progress files are written."""

    if config.atomic_output_dir is not None:
        return Path(config.atomic_output_dir)
    return Path(config.output_dir)


def worker_id_from_config(config: PipelineConfig) -> str:
    """Return a stable worker label for logs, CSV rows, and pool events."""

    if config.dynamic_worker_id:
        return str(config.dynamic_worker_id)
    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if slurm_job_id and slurm_array_task_id:
        return f"slurm_{slurm_job_id}_{slurm_array_task_id}"
    if slurm_job_id:
        return f"slurm_{slurm_job_id}"
    return f"local_{socket.gethostname()}_{os.getpid()}"


def build_document_lookup(runfile_documents: list[RunfileDocument]) -> dict[int, RunfileDocument]:
    """Index runfile documents by original runfile position."""

    document_by_index: dict[int, RunfileDocument] = {}
    for document in runfile_documents:
        document_by_index[int(document.document_index)] = document
    return document_by_index


def should_flush_completed_bucket(
    *,
    pending_documents: list[CompletedDocumentForFlush],
    last_flush_time: float,
    config: PipelineConfig,
) -> bool:
    """Decide whether the worker should write its current progress bucket."""

    if not pending_documents:
        return False
    if len(pending_documents) >= int(config.result_bucket_size):
        return True
    if time.perf_counter() - float(last_flush_time) >= float(config.result_bucket_seconds):
        return True
    return False


def flush_completed_documents(
    *,
    pending_documents: list[CompletedDocumentForFlush],
    csv_writer: LockedCsvBucketWriter,
    document_pool: DocumentPool,
    worker_id: str,
    log,
) -> int:
    """Write completed rows to CSV, then move their leases from claimed/ to done/."""

    if not pending_documents:
        return 0
    progress_rows = [completed.progress_row for completed in pending_documents]
    log(f"[dynamic-worker] flush start rows={len(progress_rows)}")
    written_count = csv_writer.append_rows(progress_rows)
    for completed in pending_documents:
        document_pool.mark_done(completed.lease, worker_id=worker_id)
    pending_documents.clear()
    log(f"[dynamic-worker] flush done rows={written_count}")
    return int(written_count)


def run_atomic_document_worker(config: PipelineConfig, *, log) -> dict[str, Any]:
    """Process dynamically claimed documents until the shared pool is empty."""

    if config.dynamic_document_pool_dir is None:
        raise ValueError("dynamic worker mode requires --dynamic-document-pool-dir")
    worker_started_at = time.perf_counter()
    worker_id = worker_id_from_config(config)
    output_dir = atomic_output_dir_from_config(config)
    progress_csv_path = output_dir / "progress_csv" / "document_completion_attempts.csv"
    progress_lock_path = output_dir / "locks" / "document_completion_attempts.lock"
    panel_directory_name = "document_panels" if str(config.plot_mode) == "stitched-language-and-document-grids" else ".temporary_document_panels"
    panel_root_dir = output_dir / "plots" / panel_directory_name
    document_pool = DocumentPool(Path(config.dynamic_document_pool_dir))
    csv_writer = LockedCsvBucketWriter(csv_path=progress_csv_path, lock_path=progress_lock_path, fieldnames=PROGRESS_FIELDNAMES)
    pending_documents: list[CompletedDocumentForFlush] = []
    last_flush_time = time.perf_counter()
    processed_count = 0
    skipped_count = 0
    failed_claim_count = 0
    attempted_count = 0

    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"[dynamic-worker] started worker_id={worker_id}")
    log(f"[dynamic-worker] pool_dir={config.dynamic_document_pool_dir}")
    log(f"[dynamic-worker] output_dir={output_dir}")
    log(f"[dynamic-worker] progress_csv={progress_csv_path}")
    log(f"[dynamic-worker] result_bucket_size={int(config.result_bucket_size)}")
    log(f"[dynamic-worker] result_bucket_seconds={float(config.result_bucket_seconds):.6f}")

    runfile_started_at = time.perf_counter()
    log("[dynamic-worker] runfile load start")
    all_runfile_documents = load_runfile_documents(config.runfile_json)
    document_by_index = build_document_lookup(all_runfile_documents)
    log(
        f"[dynamic-worker] runfile load done documents={len(all_runfile_documents)} "
        f"seconds={time.perf_counter() - runfile_started_at:.6f}"
    )

    index_started_at = time.perf_counter()
    log("[dynamic-worker] matrix index build start")
    indexes = build_score_matrix_indexes(
        ref_to_pred_scores_pkl=config.scores_pkl_ref_to_pred,
        ref_to_ref_scores_pkl=config.scores_pkl_ref_to_ref,
        log=log,
    )
    log(f"[dynamic-worker] matrix index build done seconds={time.perf_counter() - index_started_at:.6f}")

    try:
        while True:
            lease = document_pool.claim_next_available_document(worker_id=worker_id)
            if lease is None:
                log("[dynamic-worker] no available documents remain")
                break
            attempted_count += 1
            document_started_at = time.perf_counter()
            log(
                f"[dynamic-worker] claimed pool_ordinal={lease.pool_ordinal} "
                f"document_index={lease.document_index} filename={lease.filename}"
            )
            document = document_by_index.get(int(lease.document_index))
            if document is None:
                failed_claim_count += 1
                document_pool.mark_failed(lease, worker_id=worker_id, reason="document_index_not_found_in_runfile")
                log(f"[dynamic-worker] failed missing runfile document_index={lease.document_index}")
                continue
            document_result = process_one_document(
                document=document,
                config=config,
                indexes=indexes,
                log=log,
                keep_plot_payload=config.plot_mode != "none",
            )
            panel_path = None
            if config.plot_mode != "none" and document_result.plot_payload is not None:
                plot_started_at = time.perf_counter()
                log(f"[dynamic-worker] panel render start document={document.fname}")
                try:
                    panel_path = render_atomic_document_panel(
                        plot_payload=document_result.plot_payload,
                        panel_root_dir=panel_root_dir,
                        saved_figure_dpi=int(config.saved_figure_dpi),
                        show_line_ids=bool(config.show_line_ids),
                    )
                    log(
                        f"[dynamic-worker] panel render done document={document.fname} "
                        f"path={panel_path} seconds={time.perf_counter() - plot_started_at:.6f}"
                    )
                except Exception as plot_error:
                    log(f"[dynamic-worker] panel render failed document={document.fname} error={repr(plot_error)}")
            document_elapsed_seconds = time.perf_counter() - document_started_at
            progress_row = build_progress_row(
                lease=lease,
                document_result=document_result,
                worker_id=worker_id,
                attempt_counter=attempted_count,
                panel_path=str(panel_path) if panel_path is not None else None,
                alpha_sweep_pickle_path=document_result.alpha_sweep_pickle_path,
                document_elapsed_seconds=document_elapsed_seconds,
            )
            pending_documents.append(CompletedDocumentForFlush(lease=lease, progress_row=progress_row))
            if document_result.result_row is not None:
                processed_count += 1
            if document_result.skipped_row is not None:
                skipped_count += 1
            log(
                f"[dynamic-worker] document finished filename={document.fname} "
                f"processed={document_result.result_row is not None} "
                f"skipped={document_result.skipped_row is not None} "
                f"seconds={document_elapsed_seconds:.6f}"
            )
            del document_result
            gc.collect()
            if should_flush_completed_bucket(pending_documents=pending_documents, last_flush_time=last_flush_time, config=config):
                flush_completed_documents(
                    pending_documents=pending_documents,
                    csv_writer=csv_writer,
                    document_pool=document_pool,
                    worker_id=worker_id,
                    log=log,
                )
                last_flush_time = time.perf_counter()
    finally:
        flush_completed_documents(
            pending_documents=pending_documents,
            csv_writer=csv_writer,
            document_pool=document_pool,
            worker_id=worker_id,
            log=log,
        )

    state_counts = document_pool.state_counts()
    worker_elapsed_seconds = time.perf_counter() - worker_started_at
    worker_summary = {
        "worker_id": worker_id,
        "attempted_document_count": int(attempted_count),
        "processed_document_count": int(processed_count),
        "skipped_document_count": int(skipped_count),
        "failed_claim_count": int(failed_claim_count),
        "elapsed_seconds": float(worker_elapsed_seconds),
        "progress_csv": str(progress_csv_path),
        "pool_state_counts": state_counts,
    }
    log(f"[dynamic-worker] finished summary={worker_summary}")
    return worker_summary
