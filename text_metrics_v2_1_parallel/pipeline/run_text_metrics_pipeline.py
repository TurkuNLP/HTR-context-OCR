"""Top-level orchestration for text_metrics_report pipeline."""

from __future__ import annotations

from pathlib import Path

from parallelisation.execute_document_tasks import (
    build_worker_args_payload,
    execute_document_tasks,
)
from parallelisation.write_parallel_report_files import (
    compute_success_averages_from_sorted_jsonl,
    create_temp_jsonl,
    sort_jsonl_by_seq,
    write_payload_with_items_stream,
)
from pipeline.parse_text_metrics_report_args import (
    parse_text_metrics_report_args,
    validate_text_metrics_report_args,
    validate_workers_or_raise,
)
from pipeline.resolve_text_metrics_input_sources import (
    KIND_REF_TO_ADJUSTED_PRED,
    KIND_REF_TO_PRED,
    KIND_REF_TO_REF,
    build_selected_items,
    load_items_from_sources,
    resolve_scores_pkl_paths,
    select_run_items_source_kind,
)
from score_stream_index import build_score_stream_index_cached


def _cleanup_temp_spools(paths: list[Path | None]) -> None:
    """Best-effort cleanup for temporary JSONL spool files."""
    for path in paths:
        if path is None:
            continue
        try:
            Path(path).unlink(missing_ok=True)
        except Exception:
            pass


def main() -> None:
    """Run the full text-metrics pipeline and write report artifacts."""
    args = parse_text_metrics_report_args()
    scores_pkl_paths_by_kind = resolve_scores_pkl_paths(args)
    validate_text_metrics_report_args(args, scores_pkl_paths_by_kind=scores_pkl_paths_by_kind)

    available_cpus = validate_workers_or_raise(int(args.workers))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    visuals_enabled = not bool(args.skip_visuals)
    debug_enabled = bool(args.debug)
    visual_output_dir = args.output_dir

    score_index_by_kind: dict[str, dict[str, dict]] = {}
    score_index_cache_root = Path(__file__).resolve().parent.parent / ".score_index_cache"
    for kind, path in scores_pkl_paths_by_kind.items():
        if path is None:
            score_index_by_kind[kind] = {}
            continue
        score_index_by_kind[kind] = build_score_stream_index_cached(Path(path), score_index_cache_root)

    items_source_kind = select_run_items_source_kind(
        runfile_json=args.runfile_json,
        score_index_by_kind=score_index_by_kind,
    )
    run_items = load_items_from_sources(
        items_source_kind=items_source_kind,
        runfile_json=args.runfile_json,
        score_index_by_kind=score_index_by_kind,
        scores_pkl_paths_by_kind=scores_pkl_paths_by_kind,
    )

    base_selected_items, base_matched, base_attempted = build_selected_items(
        run_items,
        target_fname=args.target_fname,
        max_items=args.max_items,
    )

    selected_items = list(base_selected_items)
    matched = int(base_matched)
    attempted = int(base_attempted)

    per_doc_hough_params_by_fname: dict[str, dict[str, int]] | None = None
    hough_params_meta = {
        "enabled": False,
        "json_path": None,
        "selection_mode": None,
        "strict": False,
        "record_count": 0,
        "doc_count_field": None,
        "selected_before_count": int(len(base_selected_items)),
        "selected_after_count": int(len(base_selected_items)),
        "selected_with_params_count": 0,
        "selected_missing_params_count": 0,
        "json_missing_in_run_items_count": 0,
        "selected_missing_params_sample": [],
        "json_missing_in_run_items_sample": [],
    }

    # Lazy import: per-document Hough JSON handling is loaded only when the user
    # explicitly enables this mode. Default pipeline runs are unaffected.
    if args.hough_params_per_document_json is not None:
        from pipeline.load_hough_params_per_document import (
            apply_hough_params_selection,
            load_hough_params_per_document_json,
        )

        loaded_hough_params = load_hough_params_per_document_json(
            json_path=Path(args.hough_params_per_document_json),
        )
        per_doc_hough_params_by_fname = dict(loaded_hough_params["params_by_fname"])

        hough_selection = apply_hough_params_selection(
            run_items=run_items,
            selected_items=base_selected_items,
            params_by_fname=per_doc_hough_params_by_fname,
            selection_mode=str(args.hough_params_selection_mode),
            strict=bool(args.hough_params_strict),
        )
        selected_items = list(hough_selection["selected_items"])

        matched = int(len(selected_items))
        attempted = int(len(selected_items))

        hough_params_meta = {
            "enabled": True,
            "json_path": str(args.hough_params_per_document_json),
            "selection_mode": str(args.hough_params_selection_mode),
            "strict": bool(args.hough_params_strict),
            "record_count": int(loaded_hough_params["record_count"]),
            "doc_count_field": loaded_hough_params["doc_count_field"],
            "selected_before_count": int(len(base_selected_items)),
            "selected_after_count": int(len(selected_items)),
            "selected_with_params_count": int(hough_selection["stats"]["selected_with_params_count"]),
            "selected_missing_params_count": int(hough_selection["stats"]["selected_missing_params_count"]),
            "json_missing_in_run_items_count": int(hough_selection["stats"]["json_missing_in_run_items_count"]),
            "selected_missing_params_sample": list(hough_selection["stats"]["selected_missing_params_sample"]),
            "json_missing_in_run_items_sample": list(hough_selection["stats"]["json_missing_in_run_items_sample"]),
        }

    print(f"[run] workers={int(args.workers)} available_cpus={available_cpus}")
    print(f"[run] matched={matched} attempted={attempted}")
    if bool(hough_params_meta["enabled"]):
        print(
            "[run] hough_params_per_document="
            f"{hough_params_meta['json_path']} "
            f"mode={hough_params_meta['selection_mode']} "
            f"strict={hough_params_meta['strict']} "
            f"records={hough_params_meta['record_count']} "
            f"selected_after={hough_params_meta['selected_after_count']}"
        )

    success_spool_unsorted = create_temp_jsonl(args.output_dir, prefix="report_success_unsorted_")
    skipped_spool_unsorted = create_temp_jsonl(args.output_dir, prefix="report_skipped_unsorted_")
    failed_spool_unsorted = create_temp_jsonl(args.output_dir, prefix="report_failed_unsorted_")
    timing_spool_unsorted = create_temp_jsonl(args.output_dir, prefix="report_timing_unsorted_") if debug_enabled else None

    success_spool_sorted: Path | None = None
    skipped_spool_sorted: Path | None = None
    failed_spool_sorted: Path | None = None
    timing_spool_sorted: Path | None = None

    state = {
        "debug_enabled": bool(debug_enabled),
        "success_count": 0,
        "skipped_count": 0,
        "failed_count": 0,
        "completed_count": 0,
        "timing_count": 0,
        "sum_before": 0.0,
        "count_before": 0,
        "sum_along": 0.0,
        "count_along": 0,
    }

    worker_args_payload = build_worker_args_payload(args)
    scores_pkl_paths_by_kind_raw = {
        kind: (None if path is None else str(path))
        for kind, path in scores_pkl_paths_by_kind.items()
    }

    try:
        with (
            success_spool_unsorted.open("w", encoding="utf-8") as success_f,
            skipped_spool_unsorted.open("w", encoding="utf-8") as skipped_f,
            failed_spool_unsorted.open("w", encoding="utf-8") as failed_f,
        ):
            timing_f = timing_spool_unsorted.open("w", encoding="utf-8") if timing_spool_unsorted is not None else None
            try:
                execute_document_tasks(
                    selected_items=selected_items,
                    workers=int(args.workers),
                    worker_args_payload=worker_args_payload,
                    visual_output_dir=visual_output_dir,
                    score_index_by_kind=score_index_by_kind,
                    scores_pkl_paths_by_kind_raw=scores_pkl_paths_by_kind_raw,
                    visuals_enabled=bool(visuals_enabled),
                    debug_enabled=bool(debug_enabled),
                    success_f=success_f,
                    skipped_f=skipped_f,
                    failed_f=failed_f,
                    timing_f=timing_f,
                    state=state,
                    per_doc_hough_params_by_fname=per_doc_hough_params_by_fname,
                )
            finally:
                if timing_f is not None:
                    timing_f.close()

        if int(state["completed_count"]) != int(attempted):
            raise RuntimeError(
                f"Internal completion mismatch: completed={state['completed_count']} attempted={attempted}"
            )

        # Preserve original target-existence semantics regardless of optional
        # per-document Hough selection mode.
        if args.target_fname is not None and int(base_matched) == 0:
            raise KeyError(f"Target file not found in provided input items: {args.target_fname!r}")

        success_spool_sorted = create_temp_jsonl(args.output_dir, prefix="report_success_sorted_")
        skipped_spool_sorted = create_temp_jsonl(args.output_dir, prefix="report_skipped_sorted_")
        failed_spool_sorted = create_temp_jsonl(args.output_dir, prefix="report_failed_sorted_")

        sort_jsonl_by_seq(input_path=success_spool_unsorted, output_path=success_spool_sorted)
        sort_jsonl_by_seq(input_path=skipped_spool_unsorted, output_path=skipped_spool_sorted)
        sort_jsonl_by_seq(input_path=failed_spool_unsorted, output_path=failed_spool_sorted)

        avg_before, avg_along, count_before_sorted, count_along_sorted = compute_success_averages_from_sorted_jsonl(
            success_sorted_jsonl_path=success_spool_sorted,
        )
        if int(count_before_sorted) != int(state["success_count"]):
            raise RuntimeError(
                "Internal average-count mismatch: "
                f"success_count={state['success_count']} count_before_sorted={count_before_sorted}"
            )
        if int(count_along_sorted) != int(state["count_along"]):
            raise RuntimeError(
                "Internal along-count mismatch: "
                f"count_along_state={state['count_along']} count_along_sorted={count_along_sorted}"
            )

        report_meta = {
            "count": int(state["success_count"]),
            "matched_count": int(matched),
            "attempted_count": int(attempted),
            "skipped_empty_prediction_count": int(state["skipped_count"]),
            "failed_count": int(state["failed_count"]),
            "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
            "scores_pkl": None if scores_pkl_paths_by_kind[KIND_REF_TO_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_PRED]),
            "scores_pkl_ref_to_pred": None if scores_pkl_paths_by_kind[KIND_REF_TO_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_PRED]),
            "scores_pkl_ref_to_ref": None if scores_pkl_paths_by_kind[KIND_REF_TO_REF] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_REF]),
            "scores_pkl_ref_to_adjusted_pred": None if scores_pkl_paths_by_kind[KIND_REF_TO_ADJUSTED_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_ADJUSTED_PRED]),
            "scores_pkl_root": None if args.scores_pkl_root is None else str(args.scores_pkl_root),
            "visuals_enabled": bool(visuals_enabled),
            "window_size": int(args.window_size),
            "window_stride": int(args.window_stride),
            "hough_threshold": int(args.hough_threshold),
            "hough_line_length": int(args.hough_line_length),
            "hough_line_gap": int(args.hough_line_gap),
            "hough_seed": int(args.hough_seed),
            "hough_start": float(args.hough_start),
            "align_abs_min_len": float(args.align_abs_min_len),
            "line_filter_version": "v2_1_true_iou",
            "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
            "levenshtein_backend": str(args.levenshtein_backend),
            "debug": bool(args.debug),
            "workers": int(args.workers),
            "available_cpus": int(available_cpus),
            "run_average_normalized_levenshtein_before": avg_before,
            "run_average_normalized_levenshtein_along_lines": avg_along,
            "hough_params_per_document_json": hough_params_meta["json_path"],
            "hough_params_selection_mode": hough_params_meta["selection_mode"],
            "hough_params_strict": bool(hough_params_meta["strict"]),
            "hough_params_record_count": int(hough_params_meta["record_count"]),
            "hough_params_doc_count_field": hough_params_meta["doc_count_field"],
            "hough_params_selected_before_count": int(hough_params_meta["selected_before_count"]),
            "hough_params_selected_after_count": int(hough_params_meta["selected_after_count"]),
            "hough_params_selected_with_params_count": int(hough_params_meta["selected_with_params_count"]),
            "hough_params_selected_missing_params_count": int(hough_params_meta["selected_missing_params_count"]),
            "hough_params_json_missing_in_run_items_count": int(hough_params_meta["json_missing_in_run_items_count"]),
            "hough_params_selected_missing_params_sample": list(hough_params_meta["selected_missing_params_sample"]),
            "hough_params_json_missing_in_run_items_sample": list(hough_params_meta["json_missing_in_run_items_sample"]),
        }
        out_report = args.output_dir / "report.json"
        write_payload_with_items_stream(
            output_path=out_report,
            metadata=report_meta,
            items_jsonl_path=success_spool_sorted,
            strip_item_keys=("seq_id",),
        )

        skipped_meta = {
            "count": int(state["skipped_count"]),
            "matched_count": int(matched),
            "attempted_count": int(attempted),
            "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
            "scores_pkl_ref_to_pred": None if scores_pkl_paths_by_kind[KIND_REF_TO_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_PRED]),
            "workers": int(args.workers),
            "hough_params_per_document_json": hough_params_meta["json_path"],
            "hough_params_selection_mode": hough_params_meta["selection_mode"],
            "hough_params_strict": bool(hough_params_meta["strict"]),
        }
        out_skipped = args.output_dir / "report_skipped_empty_prediction.json"
        write_payload_with_items_stream(
            output_path=out_skipped,
            metadata=skipped_meta,
            items_jsonl_path=skipped_spool_sorted,
            strip_item_keys=("seq_id",),
        )

        failed_meta = {
            "count": int(state["failed_count"]),
            "matched_count": int(matched),
            "attempted_count": int(attempted),
            "runfile_json": None if args.runfile_json is None else str(args.runfile_json),
            "scores_pkl_ref_to_pred": None if scores_pkl_paths_by_kind[KIND_REF_TO_PRED] is None else str(scores_pkl_paths_by_kind[KIND_REF_TO_PRED]),
            "workers": int(args.workers),
            "hough_params_per_document_json": hough_params_meta["json_path"],
            "hough_params_selection_mode": hough_params_meta["selection_mode"],
            "hough_params_strict": bool(hough_params_meta["strict"]),
        }
        out_failed = args.output_dir / "report_failed_items.json"
        write_payload_with_items_stream(
            output_path=out_failed,
            metadata=failed_meta,
            items_jsonl_path=failed_spool_sorted,
            strip_item_keys=("seq_id",),
        )

        out_timing = None
        if debug_enabled and timing_spool_unsorted is not None:
            timing_spool_sorted = create_temp_jsonl(args.output_dir, prefix="report_timing_sorted_")
            sort_jsonl_by_seq(input_path=timing_spool_unsorted, output_path=timing_spool_sorted)

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
                "align_abs_min_len": float(args.align_abs_min_len),
                "line_filter_min_iou_threshold": float(args.align_min_iou_threshold),
                "levenshtein_backend": str(args.levenshtein_backend),
                "workers": int(args.workers),
                "available_cpus": int(available_cpus),
                "hough_params_per_document_json": hough_params_meta["json_path"],
                "hough_params_selection_mode": hough_params_meta["selection_mode"],
                "hough_params_strict": bool(hough_params_meta["strict"]),
                "hough_params_record_count": int(hough_params_meta["record_count"]),
            }
            out_timing = args.output_dir / "report_timings.json"
            write_payload_with_items_stream(
                output_path=out_timing,
                metadata=timing_meta,
                items_jsonl_path=timing_spool_sorted,
                strip_item_keys=("seq_id",),
            )

        print()
        print(f"Matched items: {matched}")
        print(f"Attempted items: {attempted}")
        print(f"Successful items: {int(state['success_count'])}")
        print(f"Skipped empty prediction items: {int(state['skipped_count'])}")
        print(f"Failed items: {int(state['failed_count'])}")
        if avg_before is None:
            print("Run avg normalized levenshtein before: <none>")
        else:
            print(f"Run avg normalized levenshtein before: {avg_before:.6f}")
        if avg_along is None:
            print("Run avg normalized levenshtein along lines: <none>")
        else:
            print(f"Run avg normalized levenshtein along lines: {avg_along:.6f}")
        print(f"Report: {out_report}")
        print(f"Skipped empty prediction report: {out_skipped}")
        print(f"Failed items report: {out_failed}")
        if out_timing is not None:
            print(f"Timings: {out_timing}")

    finally:
        _cleanup_temp_spools(
            [
                success_spool_unsorted,
                skipped_spool_unsorted,
                failed_spool_unsorted,
                timing_spool_unsorted,
                success_spool_sorted,
                skipped_spool_sorted,
                failed_spool_sorted,
                timing_spool_sorted,
            ]
        )
