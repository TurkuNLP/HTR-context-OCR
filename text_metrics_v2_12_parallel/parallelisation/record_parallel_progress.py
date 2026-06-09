"""Runtime progress recording for completed document envelopes."""

from __future__ import annotations

import time

from parallelisation.write_parallel_report_files import write_jsonl_line
from pipeline.report_item_views import build_internal_non_debug_success_spool_item


# Persist one completed envelope and update global counters.
def record_completed_envelope(
    *,
    envelope: dict,
    success_f,
    skipped_f,
    failed_f,
    timing_f,
    state: dict,
) -> None:
    """Persist one completed envelope and update global counters."""
    seq_id = int(envelope["seq_id"])
    status = str(envelope.get("status", ""))

    if status == "skipped_empty_prediction":
        skipped_payload = {
            "seq_id": seq_id,
            "index": int(envelope["index"]),
            "fname": str(envelope["fname"]),
            "reason": str(envelope.get("reason", "empty_prediction_text")),
        }
        write_jsonl_line(skipped_f, skipped_payload)
        state["skipped_count"] += 1
        state["completed_count"] += 1

        if timing_f is not None:
            from debug.run_timing_telemetry import write_skipped_timing_entry

            write_skipped_timing_entry(
                timing_handle=timing_f,
                seq_id=seq_id,
                envelope=envelope,
                state=state,
            )

        print(
            f"[S seq={seq_id}] {skipped_payload['fname']} | "
            f"skipped: {skipped_payload['reason']} | "
            f"done={state['completed_count']}"
        )
        return

    if status == "failed":
        failed_payload = {
            "seq_id": seq_id,
            "index": int(envelope["index"]),
            "fname": str(envelope["fname"]),
            "error_type": str(envelope.get("error_type", "RuntimeError")),
            "error_message": str(envelope.get("error_message", "unknown error")),
            "traceback": str(envelope.get("traceback", "")),
        }
        write_jsonl_line(failed_f, failed_payload)
        state["failed_count"] += 1
        state["completed_count"] += 1

        if timing_f is not None:
            from debug.run_timing_telemetry import write_failed_timing_entry

            write_failed_timing_entry(
                timing_handle=timing_f,
                seq_id=seq_id,
                envelope=envelope,
                state=state,
            )

        print(
            f"[X seq={seq_id}] {failed_payload['fname']} | "
            f"failed: {failed_payload['error_type']}: {failed_payload['error_message']} | "
            f"done={state['completed_count']}"
        )
        return

    if status != "success":
        raise ValueError(f"Unknown processing status envelope: {status!r}")

    full_success_result = dict(envelope["result"])
    timings = dict(full_success_result.pop("__timing", {})) if bool(state["debug_enabled"]) else {}
    success_spool_item = (
        dict(full_success_result)
        if bool(state["debug_enabled"])
        else build_internal_non_debug_success_spool_item(full_success_result)
    )

    write_jsonl_line(
        success_f,
        {
            "seq_id": seq_id,
            **success_spool_item,
        },
    )

    state["success_count"] += 1
    state["completed_count"] += 1

    before_val = float(full_success_result["normalized_levenshtein_before"])
    state["sum_before"] += before_val
    state["count_before"] += 1

    along_val = full_success_result.get("average_normalized_levenshtein_along_lines")
    if along_val is not None:
        state["sum_along"] += float(along_val)
        state["count_along"] += 1

    if timing_f is not None:
        from debug.run_timing_telemetry import write_success_timing_entry

        write_success_timing_entry(
            timing_handle=timing_f,
            seq_id=seq_id,
            result_payload=full_success_result,
            timings_seconds=timings,
            state=state,
        )

    print(
        f"[{state['success_count']} seq={seq_id}] {full_success_result['fname']} | "
        f"before={full_success_result['normalized_levenshtein_before']:.6f} "
        f"along={full_success_result['average_normalized_levenshtein_along_lines']} "
        f"correct_ref={full_success_result['correct_ref_coverage']:.4f} "
        f"missing_ref={full_success_result['missing_ref_coverage']:.4f} "
        f"repetition_ref={full_success_result['repetition_on_ref']:.4f} "
        f"hallucination={full_success_result['hallucination']:.4f} "
        f"| done={state['completed_count']}"
    )


# Emit periodic in-flight diagnostics so long-running items remain visible.
def maybe_emit_inflight_straggler_log(
    *,
    in_flight: dict,
    last_emit_s: float,
    interval_s: float,
    top_n: int = 3,
) -> float:
    """Emit periodic in-flight diagnostics and return updated timestamp."""
    now = time.perf_counter()
    if now - float(last_emit_s) < float(interval_s):
        return float(last_emit_s)

    if not in_flight:
        return float(now)

    rows = []
    for meta in in_flight.values():
        start_time = float(meta.get("start_time", now))
        rows.append(
            (
                float(now - start_time),
                int(meta.get("seq_id", -1)),
                int(meta.get("item_index", -1)),
                str(meta.get("item_fname", "<unknown>")),
            )
        )

    rows.sort(reverse=True)
    shown = rows[: max(1, int(top_n))]
    formatted = "; ".join(
        f"seq={seq_id} idx={item_index} elapsed={elapsed:.1f}s fname={fname}"
        for elapsed, seq_id, item_index, fname in shown
    )
    print(f"[monitor] in_flight={len(in_flight)} slowest: {formatted}")
    return float(now)
