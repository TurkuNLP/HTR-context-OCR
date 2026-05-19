"""Execute document tasks and record envelopes for sequential/parallel runs.

The execution flow is intentionally unified: the same scheduler path is used for
all worker counts. ``workers=1`` simply creates a one-process pool.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import time
import traceback
from pathlib import Path

from parallelisation.record_parallel_progress import (
    maybe_emit_inflight_straggler_log,
    record_completed_envelope,
)
from pipeline.process_single_document_metrics import process_item

KIND_REF_TO_PRED = "ref_to_pred"
KIND_REF_TO_REF = "ref_to_ref"
KIND_REF_TO_ADJUSTED_PRED = "ref_to_adjusted_pred"


def build_worker_args_payload(args: argparse.Namespace) -> dict:
    """Build a minimal pickle-friendly argument payload for worker tasks.

    Only parameters that truly vary per run are carried into the worker process.
    Levenshtein backend selection is intentionally absent because the pipeline now
    uses one fixed exact C-backed implementation.
    """
    return {
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed),
        "hough_start": float(args.hough_start),
        "align_abs_min_len": float(args.align_abs_min_len),
        "align_min_iou_threshold": float(args.align_min_iou_threshold),
    }


def build_effective_worker_args_payload(
    *,
    base_args_payload: dict,
    item_fname: str,
    per_doc_hough_params_by_fname: dict[str, dict[str, int]] | None,
) -> dict:
    """Build worker args for one item by applying optional per-doc Hough overrides.

    Metric logic remains unchanged. This function only determines which parameter
    values are passed into the existing pipeline for a single document.
    """
    effective = dict(base_args_payload)
    if not per_doc_hough_params_by_fname:
        return effective

    fname = Path(str(item_fname)).name
    override = per_doc_hough_params_by_fname.get(fname)
    if not override:
        return effective

    # Override only the tuned Hough knobs. Non-tuned parameters keep global values.
    for key in ("hough_threshold", "hough_line_length", "hough_line_gap", "hough_seed"):
        if key in override:
            effective[key] = int(override[key])

    return effective


def build_item_score_index_subset(
    score_index_by_kind: dict[str, dict[str, dict]],
    *,
    item_fname: str,
) -> dict[str, dict[str, dict]]:
    """Extract only per-document score-index entries needed by one task."""
    base_name = Path(str(item_fname)).name
    subset: dict[str, dict[str, dict]] = {}
    for kind in (KIND_REF_TO_PRED, KIND_REF_TO_REF, KIND_REF_TO_ADJUSTED_PRED):
        lookup = score_index_by_kind.get(kind, {})
        hit = lookup.get(base_name)
        subset[kind] = {} if hit is None else {base_name: hit}
    return subset


def build_skip_envelope(*, seq_id: int, item_index: int, item_fname: str) -> dict:
    """Build a standard envelope for empty-prediction skipped documents."""
    return {
        "seq_id": int(seq_id),
        "status": "skipped_empty_prediction",
        "index": int(item_index),
        "fname": str(item_fname),
        "reason": "empty_prediction_text",
        "timings_seconds": {"total_item_s": 0.0},
    }


def run_non_empty_item_task(
    *,
    seq_id: int,
    item: dict,
    args_payload: dict,
    visual_output_dir_str: str,
    score_index_by_kind_item: dict[str, dict[str, dict]],
    scores_pkl_paths_by_kind_raw: dict[str, str | None],
    visuals_enabled: bool,
    debug_enabled: bool,
) -> dict:
    """Run one non-empty document and return a success/failed envelope."""
    item_index = int(item["index"])
    item_fname = Path(str(item["fname"])).name

    worker_args = argparse.Namespace(**dict(args_payload))
    visual_output_dir = Path(visual_output_dir_str)
    scores_pkl_paths_by_kind = {
        kind: (None if raw is None else Path(raw))
        for kind, raw in scores_pkl_paths_by_kind_raw.items()
    }

    item_start = time.perf_counter() if debug_enabled else None
    try:
        result = process_item(
            item,
            worker_args,
            visual_output_dir,
            score_index_by_kind=score_index_by_kind_item,
            scores_pkl_paths_by_kind=scores_pkl_paths_by_kind,
            visuals_enabled=bool(visuals_enabled),
            debug_enabled=bool(debug_enabled),
        )
        return {
            "seq_id": int(seq_id),
            "status": "success",
            "index": int(item_index),
            "fname": str(item_fname),
            "result": result,
        }
    except Exception as exc:
        elapsed = 0.0 if item_start is None else float(time.perf_counter() - item_start)
        return {
            "seq_id": int(seq_id),
            "status": "failed",
            "index": int(item_index),
            "fname": str(item_fname),
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
            "timings_seconds": {"total_item_s": elapsed},
        }


def build_failed_envelope_from_executor_exception(
    *,
    seq_id: int,
    item_index: int,
    item_fname: str,
    exc: Exception,
    start_time: float | None,
    debug_enabled: bool,
) -> dict:
    """Convert an executor-level exception into the standard failed envelope."""
    elapsed = 0.0 if (not debug_enabled or start_time is None) else float(time.perf_counter() - start_time)
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return {
        "seq_id": int(seq_id),
        "status": "failed",
        "index": int(item_index),
        "fname": str(item_fname),
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "traceback": tb,
        "timings_seconds": {"total_item_s": elapsed},
    }


def consume_done_futures(
    *,
    done_futures: set[cf.Future],
    in_flight: dict[cf.Future, dict],
    success_f,
    skipped_f,
    failed_f,
    timing_f,
    state: dict,
    debug_enabled: bool,
) -> None:
    """Consume completed futures and persist envelopes immediately."""
    for done_fut in done_futures:
        meta = in_flight.pop(done_fut)
        try:
            envelope = done_fut.result()
        except Exception as exc:
            envelope = build_failed_envelope_from_executor_exception(
                seq_id=int(meta["seq_id"]),
                item_index=int(meta["item_index"]),
                item_fname=str(meta["item_fname"]),
                exc=exc,
                start_time=meta.get("start_time"),
                debug_enabled=bool(debug_enabled),
            )

        record_completed_envelope(
            envelope=envelope,
            success_f=success_f,
            skipped_f=skipped_f,
            failed_f=failed_f,
            timing_f=timing_f,
            state=state,
        )


def execute_document_tasks(
    *,
    selected_items: list[dict],
    workers: int,
    worker_args_payload: dict,
    visual_output_dir: Path,
    score_index_by_kind: dict[str, dict[str, dict]],
    scores_pkl_paths_by_kind_raw: dict[str, str | None],
    visuals_enabled: bool,
    debug_enabled: bool,
    success_f,
    skipped_f,
    failed_f,
    timing_f,
    state: dict,
    per_doc_hough_params_by_fname: dict[str, dict[str, int]] | None = None,
) -> None:
    """Execute all selected document tasks via one shared scheduler implementation.

    The same dispatch logic is used for all worker counts to avoid drift between
    a "sequential" branch and a "parallel" branch. With ``workers=1`` this
    scheduler still behaves sequentially because at most one future is in flight.
    """
    monitor_interval_s = 30.0
    monitor_last_emit_s = time.perf_counter()

    with cf.ProcessPoolExecutor(max_workers=int(workers)) as executor:
        in_flight: dict[cf.Future, dict] = {}

        for seq_id, item in enumerate(selected_items):
            item_index = int(item["index"])
            item_fname = Path(str(item["fname"])).name
            pred_text = str(item["pred"])

            # Empty predictions are skipped synchronously to avoid unnecessary
            # pool dispatch and to preserve existing skip semantics.
            if pred_text == "":
                envelope = build_skip_envelope(
                    seq_id=seq_id,
                    item_index=item_index,
                    item_fname=item_fname,
                )
                record_completed_envelope(
                    envelope=envelope,
                    success_f=success_f,
                    skipped_f=skipped_f,
                    failed_f=failed_f,
                    timing_f=timing_f,
                    state=state,
                )
                continue

            item_score_index_subset = build_item_score_index_subset(
                score_index_by_kind,
                item_fname=item_fname,
            )
            effective_worker_args_payload = build_effective_worker_args_payload(
                base_args_payload=worker_args_payload,
                item_fname=item_fname,
                per_doc_hough_params_by_fname=per_doc_hough_params_by_fname,
            )

            future_start = time.perf_counter()
            fut = executor.submit(
                run_non_empty_item_task,
                seq_id=seq_id,
                item=item,
                args_payload=effective_worker_args_payload,
                visual_output_dir_str=str(visual_output_dir),
                score_index_by_kind_item=item_score_index_subset,
                scores_pkl_paths_by_kind_raw=scores_pkl_paths_by_kind_raw,
                visuals_enabled=visuals_enabled,
                debug_enabled=debug_enabled,
            )
            in_flight[fut] = {
                "seq_id": int(seq_id),
                "item_index": int(item_index),
                "item_fname": str(item_fname),
                "start_time": future_start,
            }

            # Throttle submissions: keep at most <workers> in-flight tasks.
            while len(in_flight) >= int(workers):
                timeout_s = max(
                    0.1,
                    monitor_interval_s - (time.perf_counter() - monitor_last_emit_s),
                )
                done, _ = cf.wait(
                    set(in_flight.keys()),
                    timeout=timeout_s,
                    return_when=cf.FIRST_COMPLETED,
                )
                if not done:
                    monitor_last_emit_s = maybe_emit_inflight_straggler_log(
                        in_flight=in_flight,
                        last_emit_s=monitor_last_emit_s,
                        interval_s=monitor_interval_s,
                        top_n=3,
                    )
                    continue

                consume_done_futures(
                    done_futures=set(done),
                    in_flight=in_flight,
                    success_f=success_f,
                    skipped_f=skipped_f,
                    failed_f=failed_f,
                    timing_f=timing_f,
                    state=state,
                    debug_enabled=debug_enabled,
                )

        # Drain all remaining in-flight work.
        while in_flight:
            timeout_s = max(
                0.1,
                monitor_interval_s - (time.perf_counter() - monitor_last_emit_s),
            )
            done, _ = cf.wait(
                set(in_flight.keys()),
                timeout=timeout_s,
                return_when=cf.FIRST_COMPLETED,
            )
            if not done:
                monitor_last_emit_s = maybe_emit_inflight_straggler_log(
                    in_flight=in_flight,
                    last_emit_s=monitor_last_emit_s,
                    interval_s=monitor_interval_s,
                    top_n=3,
                )
                continue

            consume_done_futures(
                done_futures=set(done),
                in_flight=in_flight,
                success_f=success_f,
                skipped_f=skipped_f,
                failed_f=failed_f,
                timing_f=timing_f,
                state=state,
                debug_enabled=debug_enabled,
            )
