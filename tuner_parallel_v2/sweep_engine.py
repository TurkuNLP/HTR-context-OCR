from __future__ import annotations

"""Core exhaustive sweep engine for per-document Hough tuning.

Parallelization model:
1) Optional document-level concurrency (multiple docs in flight).
2) Per-document threshold-level concurrency (one threshold chunk per worker).
3) Inside each threshold chunk, line_length x line_gap combinations are serial.

This keeps deterministic ranking/merging while allowing better CPU utilization on
multi-document runs.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from threading import Lock
import time

try:
    from .hough_eval import evaluate_single_combination, pick_better_eval
    from .sweep_aggregation import point_from_best
    from .tuner_config import (
        HoughBaselineConfig,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        SweepDocument,
    )
except ImportError:
    from hough_eval import evaluate_single_combination, pick_better_eval  # type: ignore
    from sweep_aggregation import point_from_best  # type: ignore
    from tuner_config import (  # type: ignore
        HoughBaselineConfig,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        SweepDocument,
    )


def _evaluate_threshold(
    *,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold: int,
    line_length_values: list[int],
    line_gap_values: list[int],
) -> dict:
    """Evaluate all (line_length, line_gap) combinations for one threshold."""
    started_at = time.perf_counter()

    best_threshold: dict | None = None
    best_by_line_length: dict[int, dict | None] = {int(v): None for v in line_length_values}
    best_by_line_gap: dict[int, dict | None] = {int(v): None for v in line_gap_values}

    eval_count = 0
    for ln in line_length_values:
        for gp in line_gap_values:
            cfg = replace(
                baseline_cfg,
                hough_threshold=int(threshold),
                hough_line_length=int(ln),
                hough_line_gap=int(gp),
            )

            eval_row = evaluate_single_combination(
                doc=doc,
                cfg=cfg,
                levenshtein_backend=levenshtein_backend,
            )
            eval_row[PARAM_HOUGH_THRESHOLD] = int(threshold)
            eval_row[PARAM_HOUGH_LINE_LENGTH] = int(ln)
            eval_row[PARAM_HOUGH_LINE_GAP] = int(gp)
            eval_row[PARAM_HOUGH_SEED] = int(cfg.hough_seed)

            eval_count += 1
            best_threshold = pick_better_eval(best_threshold, eval_row)
            best_by_line_length[int(ln)] = pick_better_eval(best_by_line_length[int(ln)], eval_row)
            best_by_line_gap[int(gp)] = pick_better_eval(best_by_line_gap[int(gp)], eval_row)

    return {
        "threshold": int(threshold),
        "best_threshold": best_threshold,
        "best_by_line_length": best_by_line_length,
        "best_by_line_gap": best_by_line_gap,
        "eval_count": int(eval_count),
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }


def tune_single_document(
    *,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    workers: int,
    log_fn: LogFn,
) -> dict:
    """Run exhaustive fixed nested-grid tuning for one document."""
    doc_started_at = time.perf_counter()

    best_overall: dict | None = None
    best_by_threshold: dict[int, dict | None] = {int(v): None for v in threshold_values}
    best_by_line_length: dict[int, dict | None] = {int(v): None for v in line_length_values}
    best_by_line_gap: dict[int, dict | None] = {int(v): None for v in line_gap_values}

    eval_count = 0
    requested_workers = max(1, int(workers))
    use_parallel_thresholds = requested_workers > 1 and len(threshold_values) > 1

    if use_parallel_thresholds:
        with ThreadPoolExecutor(max_workers=requested_workers) as executor:
            futures = {
                executor.submit(
                    _evaluate_threshold,
                    doc=doc,
                    baseline_cfg=baseline_cfg,
                    levenshtein_backend=levenshtein_backend,
                    threshold=int(th),
                    line_length_values=line_length_values,
                    line_gap_values=line_gap_values,
                ): int(th)
                for th in threshold_values
            }

            for future in as_completed(futures):
                th = futures[future]
                payload = future.result()
                eval_count += int(payload["eval_count"])

                best_th = payload["best_threshold"]
                best_overall = pick_better_eval(best_overall, best_th)
                best_by_threshold[int(th)] = pick_better_eval(best_by_threshold[int(th)], best_th)

                for ln, row in payload["best_by_line_length"].items():
                    best_by_line_length[int(ln)] = pick_better_eval(best_by_line_length[int(ln)], row)
                for gp, row in payload["best_by_line_gap"].items():
                    best_by_line_gap[int(gp)] = pick_better_eval(best_by_line_gap[int(gp)], row)

                log_fn(
                    f"[doc-loop] fname={doc.fname} threshold={th} done evals={eval_count} "
                    f"elapsed_s={float(payload['elapsed_seconds']):.3f} mode=thread"
                )
    else:
        for th in threshold_values:
            payload = _evaluate_threshold(
                doc=doc,
                baseline_cfg=baseline_cfg,
                levenshtein_backend=levenshtein_backend,
                threshold=int(th),
                line_length_values=line_length_values,
                line_gap_values=line_gap_values,
            )
            eval_count += int(payload["eval_count"])

            best_th = payload["best_threshold"]
            best_overall = pick_better_eval(best_overall, best_th)
            best_by_threshold[int(th)] = pick_better_eval(best_by_threshold[int(th)], best_th)

            for ln, row in payload["best_by_line_length"].items():
                best_by_line_length[int(ln)] = pick_better_eval(best_by_line_length[int(ln)], row)
            for gp, row in payload["best_by_line_gap"].items():
                best_by_line_gap[int(gp)] = pick_better_eval(best_by_line_gap[int(gp)], row)

            log_fn(
                f"[doc-loop] fname={doc.fname} threshold={th} done evals={eval_count} "
                f"elapsed_s={float(payload['elapsed_seconds']):.3f} mode=serial"
            )

    doc_grid_seconds = float(time.perf_counter() - doc_started_at)

    if best_overall is None:
        best_payload = {
            "along_lines_nls": None,
            PARAM_HOUGH_THRESHOLD: int(baseline_cfg.hough_threshold),
            PARAM_HOUGH_LINE_LENGTH: int(baseline_cfg.hough_line_length),
            PARAM_HOUGH_LINE_GAP: int(baseline_cfg.hough_line_gap),
            PARAM_HOUGH_SEED: int(baseline_cfg.hough_seed),
        }
    else:
        best_payload = dict(best_overall)

    doc_best_record = {
        "index": int(doc.index),
        "fname": str(doc.fname),
        "whole_document_nls": float(doc.whole_document_nls),
        "best": best_payload,
        "evaluated_combination_count": int(eval_count),
        "doc_grid_seconds": float(doc_grid_seconds),
    }

    profile_points = {
        PARAM_HOUGH_THRESHOLD: {
            v: point_from_best(doc=doc, best_eval=best_by_threshold[v], baseline_cfg=baseline_cfg)
            for v in threshold_values
        },
        PARAM_HOUGH_LINE_LENGTH: {
            v: point_from_best(doc=doc, best_eval=best_by_line_length[v], baseline_cfg=baseline_cfg)
            for v in line_length_values
        },
        PARAM_HOUGH_LINE_GAP: {
            v: point_from_best(doc=doc, best_eval=best_by_line_gap[v], baseline_cfg=baseline_cfg)
            for v in line_gap_values
        },
    }

    return {
        "doc_best_record": doc_best_record,
        "profile_points": profile_points,
        "doc_grid_seconds": float(doc_grid_seconds),
        "evaluated_combination_count": int(eval_count),
    }




def _log_doc_started(*, doc: SweepDocument, progress_state: dict[str, int], progress_lock, log_fn: LogFn) -> None:
    """Log document-level start progress for visibility during long sweeps."""
    with progress_lock:
        progress_state["started"] += 1
        started = int(progress_state["started"])
        completed = int(progress_state["completed"])
        total = int(progress_state["total"])

    in_progress = max(0, started - completed)
    pending_start = max(0, total - started)
    log_fn(
        f"[doc-start] fname={doc.fname} index={doc.index} "
        f"started={started}/{total} in_progress={in_progress} "
        f"completed={completed}/{total} pending_start={pending_start}"
    )


def _log_doc_completed(
    *,
    doc: SweepDocument,
    tuned: dict,
    progress_state: dict[str, int],
    progress_lock,
    log_fn: LogFn,
) -> None:
    """Log document-level completion progress and remaining in-flight count."""
    with progress_lock:
        progress_state["completed"] += 1
        started = int(progress_state["started"])
        completed = int(progress_state["completed"])
        total = int(progress_state["total"])

    in_progress = max(0, started - completed)
    pending_start = max(0, total - started)
    log_fn(
        f"[doc-done] fname={doc.fname} index={doc.index} "
        f"doc_grid_s={float(tuned.get('doc_grid_seconds', 0.0)):.3f} "
        f"in_progress={in_progress} completed={completed}/{total} "
        f"pending_start={pending_start}"
    )


def _run_one_document_with_start_logging(
    *,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    workers: int,
    progress_state: dict[str, int],
    progress_lock,
    log_fn: LogFn,
) -> dict:
    """Wrapper used in doc-parallel mode to emit per-document start progress."""
    _log_doc_started(doc=doc, progress_state=progress_state, progress_lock=progress_lock, log_fn=log_fn)
    return tune_single_document(
        doc=doc,
        baseline_cfg=baseline_cfg,
        levenshtein_backend=str(levenshtein_backend),
        threshold_values=threshold_values,
        line_length_values=line_length_values,
        line_gap_values=line_gap_values,
        workers=workers,
        log_fn=log_fn,
    )


def run_document_sweeps(
    *,
    docs: list[SweepDocument],
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    workers: int,
    doc_workers: int,
    log_fn: LogFn,
) -> dict:
    """Run per-document exhaustive sweeps and collect profile points.

    `workers` controls threshold-level workers inside one document.
    `doc_workers` controls how many documents are processed concurrently.
    """
    profile_points: dict[str, dict[int, list[dict]]] = {
        PARAM_HOUGH_THRESHOLD: {v: [] for v in threshold_values},
        PARAM_HOUGH_LINE_LENGTH: {v: [] for v in line_length_values},
        PARAM_HOUGH_LINE_GAP: {v: [] for v in line_gap_values},
    }
    doc_best_records: list[dict] = []

    grid_eval_started_at = time.perf_counter()
    doc_grid_seconds_total = 0.0

    threshold_workers = max(1, int(workers))
    requested_doc_workers = max(1, int(doc_workers))
    use_parallel_docs = requested_doc_workers > 1 and len(docs) > 1

    tuned_by_doc_index: dict[int, dict] = {}

    # Shared counters for document-level progress visibility.
    progress_lock = Lock()
    progress_state: dict[str, int] = {
        "total": int(len(docs)),
        "started": 0,
        "completed": 0,
    }

    if use_parallel_docs:
        log_fn(
            f"[doc-parallel] enabled doc_workers={requested_doc_workers} "
            f"threshold_workers_per_doc={threshold_workers} docs={len(docs)}"
        )
        with ThreadPoolExecutor(max_workers=requested_doc_workers) as executor:
            futures = {
                executor.submit(
                    _run_one_document_with_start_logging,
                    doc=doc,
                    baseline_cfg=baseline_cfg,
                    levenshtein_backend=str(levenshtein_backend),
                    threshold_values=threshold_values,
                    line_length_values=line_length_values,
                    line_gap_values=line_gap_values,
                    workers=threshold_workers,
                    progress_state=progress_state,
                    progress_lock=progress_lock,
                    log_fn=log_fn,
                ): doc
                for doc in docs
            }

            for future in as_completed(futures):
                doc = futures[future]
                tuned = future.result()
                tuned_by_doc_index[int(doc.index)] = tuned
                _log_doc_completed(
                    doc=doc,
                    tuned=tuned,
                    progress_state=progress_state,
                    progress_lock=progress_lock,
                    log_fn=log_fn,
                )
    else:
        mode = "serial" if len(docs) <= 1 else "serial_docs"
        log_fn(
            f"[doc-parallel] {mode} doc_workers={requested_doc_workers} "
            f"threshold_workers_per_doc={threshold_workers} docs={len(docs)}"
        )
        for doc in docs:
            _log_doc_started(doc=doc, progress_state=progress_state, progress_lock=progress_lock, log_fn=log_fn)
            tuned = tune_single_document(
                doc=doc,
                baseline_cfg=baseline_cfg,
                levenshtein_backend=str(levenshtein_backend),
                threshold_values=threshold_values,
                line_length_values=line_length_values,
                line_gap_values=line_gap_values,
                workers=threshold_workers,
                log_fn=log_fn,
            )
            tuned_by_doc_index[int(doc.index)] = tuned
            _log_doc_completed(
                doc=doc,
                tuned=tuned,
                progress_state=progress_state,
                progress_lock=progress_lock,
                log_fn=log_fn,
            )

    # Aggregate in original document order for deterministic outputs.
    for doc in docs:
        tuned = tuned_by_doc_index[int(doc.index)]

        doc_best_record = tuned["doc_best_record"]
        doc_best_records.append(doc_best_record)
        doc_grid_seconds_total += float(tuned["doc_grid_seconds"])

        best_nls = doc_best_record["best"].get("along_lines_nls")
        best_nls_str = "None" if best_nls is None else f"{float(best_nls):.6f}"
        log_fn(
            f"[doc-best] fname={doc.fname} best_along_lines={best_nls_str} "
            f"th={doc_best_record['best'].get(PARAM_HOUGH_THRESHOLD)} "
            f"len={doc_best_record['best'].get(PARAM_HOUGH_LINE_LENGTH)} "
            f"gap={doc_best_record['best'].get(PARAM_HOUGH_LINE_GAP)} "
            f"seed={doc_best_record['best'].get(PARAM_HOUGH_SEED)} "
            f"combos={doc_best_record['evaluated_combination_count']} "
            f"doc_grid_s={doc_best_record['doc_grid_seconds']:.3f}"
        )

        per_doc_profile = tuned["profile_points"]
        for param in SUPPORTED_SWEEP_PARAMETERS:
            for value, point in per_doc_profile[param].items():
                profile_points[param][int(value)].append(point)

    grid_eval_seconds = float(time.perf_counter() - grid_eval_started_at)

    return {
        "profile_points": profile_points,
        "doc_best_records": doc_best_records,
        "grid_eval_seconds": grid_eval_seconds,
        "doc_grid_seconds_total": float(doc_grid_seconds_total),
    }


__all__ = [
    "tune_single_document",
    "run_document_sweeps",
]
