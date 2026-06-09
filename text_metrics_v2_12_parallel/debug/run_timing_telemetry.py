"""Run-level timing telemetry helpers used only when --debug is enabled."""

from __future__ import annotations

from pathlib import Path

from parallelisation.write_parallel_report_files import (
    create_temp_jsonl,
    sort_jsonl_by_seq,
    write_jsonl_line,
    write_payload_with_items_stream,
)


# Create the unsorted timing spool used during a debug-enabled run.
def create_debug_timing_spool_file(output_dir: Path) -> Path:
    """Create the temporary unsorted timing JSONL spool for one debug run."""
    # Create the spool file in the same output directory as the main reports.
    return create_temp_jsonl(Path(output_dir), prefix="report_timing_unsorted_")


# Write one skipped-item timing entry into the timing spool.
def write_skipped_timing_entry(
    *,
    timing_handle,
    seq_id: int,
    envelope: dict,
    state: dict,
) -> None:
    """Write timing telemetry for one skipped-empty-prediction envelope."""
    # Build the exact skipped timing payload shape used by the existing pipeline.
    timing_payload = {
        "seq_id": int(seq_id),
        "index": int(envelope["index"]),
        "fname": str(envelope["fname"]),
        "status": "skipped_empty_prediction",
        "reason": str(envelope.get("reason", "empty_prediction_text")),
        "timings_seconds": {"total_item_s": 0.0},
    }
    # Append the JSONL record to the timing spool.
    write_jsonl_line(timing_handle, timing_payload)
    # Increase the run-level timing-entry counter.
    state["timing_count"] += 1


# Write one failed-item timing entry into the timing spool.
def write_failed_timing_entry(
    *,
    timing_handle,
    seq_id: int,
    envelope: dict,
    state: dict,
) -> None:
    """Write timing telemetry for one failed envelope."""
    # Copy the executor-provided timings so we can normalize the payload safely.
    timings_seconds = dict(envelope.get("timings_seconds", {}))
    # Ensure the payload always contains a total-item duration field.
    if "total_item_s" not in timings_seconds:
        timings_seconds["total_item_s"] = 0.0
    # Build the exact failed timing payload shape used by the existing pipeline.
    timing_payload = {
        "seq_id": int(seq_id),
        "index": int(envelope["index"]),
        "fname": str(envelope["fname"]),
        "status": "failed",
        "error_type": str(envelope.get("error_type", "RuntimeError")),
        "error_message": str(envelope.get("error_message", "unknown error")),
        "timings_seconds": timings_seconds,
    }
    # Append the JSONL record to the timing spool.
    write_jsonl_line(timing_handle, timing_payload)
    # Increase the run-level timing-entry counter.
    state["timing_count"] += 1


# Write one successful-item timing entry into the timing spool.
def write_success_timing_entry(
    *,
    timing_handle,
    seq_id: int,
    result_payload: dict,
    timings_seconds: dict,
    state: dict,
) -> None:
    """Write timing telemetry for one successful result payload."""
    # Build the exact success timing payload shape used by the existing pipeline.
    timing_payload = {
        "seq_id": int(seq_id),
        "index": int(result_payload["index"]),
        "fname": str(result_payload["fname"]),
        "status": "success",
        "timings_seconds": dict(timings_seconds),
    }
    # Append the JSONL record to the timing spool.
    write_jsonl_line(timing_handle, timing_payload)
    # Increase the run-level timing-entry counter.
    state["timing_count"] += 1


# Finalize the sorted timing report JSON file at the end of a debug-enabled run.
def write_final_debug_timing_report(
    *,
    output_dir: Path,
    timing_spool_unsorted: Path,
    matched: int,
    attempted: int,
    state: dict,
    args,
    available_cpus: int,
    hough_params_meta: dict,
    levenshtein_backend: str,
) -> tuple[Path, Path]:
    """Sort the timing spool and write the final report_timings.json payload."""
    # Create the sorted temporary timing spool next to the final reports.
    timing_spool_sorted = create_temp_jsonl(Path(output_dir), prefix="report_timing_sorted_")
    # Sort the unsorted timing spool by sequence id to preserve stable output order.
    sort_jsonl_by_seq(input_path=Path(timing_spool_unsorted), output_path=timing_spool_sorted)

    # Build the exact run-level timing metadata payload used by the existing pipeline.
    timing_meta = {
        "count": int(state["timing_count"]),
        "matched_count": int(matched),
        "attempted_count": int(attempted),
        "successful_count": int(state["success_count"]),
        "skipped_empty_prediction_count": int(state["skipped_count"]),
        "failed_count": int(state["failed_count"]),
        "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
        "scores_pkl_root": None if args.scores_pkl_root is None else str(args.scores_pkl_root),
        "window_size": int(args.window_size),
        "window_stride": int(args.window_stride),
        "hough_threshold": int(args.hough_threshold),
        "hough_line_length": int(args.hough_line_length),
        "hough_line_gap": int(args.hough_line_gap),
        "hough_seed": int(args.hough_seed),
        "hough_start": float(args.hough_start),
        "hough_handoff_mode": str(args.hough_handoff_mode),
        "align_abs_min_len": float(args.align_abs_min_len),
        "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
        "levenshtein_backend": str(levenshtein_backend),
        "workers": int(args.workers),
        "available_cpus": int(available_cpus),
        "hough_params_per_document_json": hough_params_meta["json_path"],
        "hough_params_selection_mode": hough_params_meta["selection_mode"],
        "hough_params_strict": bool(hough_params_meta["strict"]),
        "hough_params_record_count": int(hough_params_meta["record_count"]),
    }
    # Choose the final report_timings.json output path.
    output_timing_path = Path(output_dir) / "report_timings.json"
    # Stream the sorted timing entries into the final pretty-printed JSON report.
    write_payload_with_items_stream(
        output_path=output_timing_path,
        metadata=timing_meta,
        items_jsonl_path=timing_spool_sorted,
        strip_item_keys=("seq_id",),
    )
    # Return both the final report path and the sorted spool so the caller can clean it up.
    return output_timing_path, timing_spool_sorted
