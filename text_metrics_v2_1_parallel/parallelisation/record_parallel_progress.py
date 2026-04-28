"""Runtime progress recording for completed document envelopes."""

from __future__ import annotations

import time

from parallelisation.write_parallel_report_files import write_jsonl_line


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
            write_jsonl_line(
                timing_f,
                {
                    "seq_id": seq_id,
                    "index": int(envelope["index"]),
                    "fname": str(envelope["fname"]),
                    "status": "skipped_empty_prediction",
                    "reason": skipped_payload["reason"],
                    "timings_seconds": {"total_item_s": 0.0},
                },
            )
            state["timing_count"] += 1

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
            timings_seconds = dict(envelope.get("timings_seconds", {}))
            if "total_item_s" not in timings_seconds:
                timings_seconds["total_item_s"] = 0.0
            write_jsonl_line(
                timing_f,
                {
                    "seq_id": seq_id,
                    "index": int(envelope["index"]),
                    "fname": str(envelope["fname"]),
                    "status": "failed",
                    "error_type": failed_payload["error_type"],
                    "error_message": failed_payload["error_message"],
                    "timings_seconds": timings_seconds,
                },
            )
            state["timing_count"] += 1

        print(
            f"[X seq={seq_id}] {failed_payload['fname']} | "
            f"failed: {failed_payload['error_type']}: {failed_payload['error_message']} | "
            f"done={state['completed_count']}"
        )
        return

    if status != "success":
        raise ValueError(f"Unknown processing status envelope: {status!r}")

    res = dict(envelope["result"])
    timings = dict(res.pop("__timing", {})) if bool(state["debug_enabled"]) else {}

    write_jsonl_line(
        success_f,
        {
            "seq_id": seq_id,
            **res,
        },
    )

    state["success_count"] += 1
    state["completed_count"] += 1

    before_val = float(res["normalized_levenshtein_before"])
    state["sum_before"] += before_val
    state["count_before"] += 1

    along_val = res.get("average_normalized_levenshtein_along_lines")
    if along_val is not None:
        state["sum_along"] += float(along_val)
        state["count_along"] += 1

    if timing_f is not None:
        write_jsonl_line(
            timing_f,
            {
                "seq_id": seq_id,
                "index": int(res["index"]),
                "fname": str(res["fname"]),
                "status": "success",
                "timings_seconds": timings,
            },
        )
        state["timing_count"] += 1

    print(
        f"[{state['success_count']} seq={seq_id}] {res['fname']} | "
        f"before={res['normalized_levenshtein_before']:.6f} "
        f"along={res['average_normalized_levenshtein_along_lines']} "
        f"ok={res['ok_percent']:.4f} missing={res['missing_percent']:.4f} "
        f"repetition={res['repetition_percent']:.4f} hallucination={res['hallucination_percent']:.4f} "
        f"| done={state['completed_count']}"
    )


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
