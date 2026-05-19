from __future__ import annotations

"""Document and threshold scheduling for exhaustive Hough tuning.

Parallelization model:
1) Optional document-level concurrency (multiple docs in flight).
2) Per-document threshold-level concurrency (one threshold chunk per worker).
3) Inside each threshold chunk, line_length x line_gap combinations are serial
   while the Hough seed is temporarily fixed to one deterministic value.

This keeps deterministic ranking/aggregation while allowing better CPU
utilization on multi-document runs.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass, field
from threading import Lock
import time
from typing import Callable, Iterable

try:
    from .hough_eval import evaluate_single_combination_values, pick_better_eval
    from .sweep_aggregation import point_from_best
    from .sweep_reduction import empty_best_value_map, merge_best_value_payloads
    from .tuner_config import (
        FIXED_HOUGH_SEED,
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
    from tuner.hough_eval import evaluate_single_combination_values, pick_better_eval  # type: ignore
    from tuner.sweep_aggregation import point_from_best  # type: ignore
    from tuner.sweep_reduction import empty_best_value_map, merge_best_value_payloads  # type: ignore
    from tuner.tuner_config import (  # type: ignore
        FIXED_HOUGH_SEED,
        HoughBaselineConfig,
        LogFn,
        PARAM_HOUGH_LINE_GAP,
        PARAM_HOUGH_LINE_LENGTH,
        PARAM_HOUGH_SEED,
        PARAM_HOUGH_THRESHOLD,
        SUPPORTED_SWEEP_PARAMETERS,
        SweepDocument,
    )

def _invalid_record_from_eval_row(*, doc: SweepDocument, eval_row: dict) -> dict:
    """Build one compact CSV-ready record for an invalid Hough combination."""
    return {
        "doc_index": int(doc.index),
        "fname": str(doc.fname),
        PARAM_HOUGH_THRESHOLD: int(eval_row.get(PARAM_HOUGH_THRESHOLD, 0)),
        PARAM_HOUGH_LINE_LENGTH: int(eval_row.get(PARAM_HOUGH_LINE_LENGTH, 0)),
        PARAM_HOUGH_LINE_GAP: int(eval_row.get(PARAM_HOUGH_LINE_GAP, 0)),
        PARAM_HOUGH_SEED: int(eval_row.get(PARAM_HOUGH_SEED, 0)),
        "invalid_reason": eval_row.get("invalid_reason"),
        "invalid_error_message": eval_row.get("invalid_error_message"),
        "coverage_y_diff_size": int(eval_row.get("coverage_y_diff_size", 0) or 0),
        "coverage_y_diff_min": eval_row.get("coverage_y_diff_min"),
        "coverage_y_diff_max": eval_row.get("coverage_y_diff_max"),
        "coverage_y_diff_le_minus_one_count": int(eval_row.get("coverage_y_diff_le_minus_one_count", 0) or 0),
        "coverage_y_diff_lt_minus_one_count": int(eval_row.get("coverage_y_diff_lt_minus_one_count", 0) or 0),
        "coverage_y_diff_below_minus_one_counts_json": eval_row.get("coverage_y_diff_below_minus_one_counts_json", {}),
        "line_guided_columns": int(eval_row.get("line_guided_columns", 0) or 0),
        "fallback_columns": int(eval_row.get("fallback_columns", 0) or 0),
        "used_line_count": int(eval_row.get("used_line_count", 0) or 0),
        "used_line_count_ref_to_ref": int(eval_row.get("used_line_count_ref_to_ref", 0) or 0),
        "raw_line_count": int(eval_row.get("raw_line_count", 0) or 0),
        "raw_line_count_ref_to_ref": int(eval_row.get("raw_line_count_ref_to_ref", 0) or 0),
        "candidate_line_count": int(eval_row.get("candidate_line_count", 0) or 0),
        "candidate_line_count_ref_to_ref": int(eval_row.get("candidate_line_count_ref_to_ref", 0) or 0),
        "timing_hough_detect_ref_to_pred_seconds": float(eval_row.get("timing_hough_detect_ref_to_pred_seconds", 0.0) or 0.0),
        "timing_filter_ref_to_pred_seconds": float(eval_row.get("timing_filter_ref_to_pred_seconds", 0.0) or 0.0),
        "timing_hough_detect_ref_to_ref_seconds": float(eval_row.get("timing_hough_detect_ref_to_ref_seconds", 0.0) or 0.0),
        "timing_filter_ref_to_ref_seconds": float(eval_row.get("timing_filter_ref_to_ref_seconds", 0.0) or 0.0),
        "timing_build_bundle_seconds": float(eval_row.get("timing_build_bundle_seconds", 0.0) or 0.0),
        "timing_coverage_seconds": float(eval_row.get("timing_coverage_seconds", 0.0) or 0.0),
        "timing_levenshtein_seconds": float(eval_row.get("timing_levenshtein_seconds", 0.0) or 0.0),
        "timing_total_seconds": float(eval_row.get("timing_total_seconds", 0.0) or 0.0),
    }


@dataclass
class _DocumentSweepAccumulator:
    """Mutable best-row accumulator for one document's full Hough grid.

    The scheduler can evaluate thresholds serially, per-document in parallel, or
    through a global threshold queue.  This object keeps the reduction logic in
    one place so all scheduler shapes produce the same document-level payload.
    """

    doc: SweepDocument
    threshold_values: list[int]
    line_length_values: list[int]
    line_gap_values: list[int]
    seed_values: list[int]
    started_at: float = field(default_factory=time.perf_counter)
    best_overall: dict | None = None
    best_by_threshold: dict[int, dict | None] = field(default_factory=dict)
    best_by_line_length: dict[int, dict | None] = field(default_factory=dict)
    best_by_line_gap: dict[int, dict | None] = field(default_factory=dict)
    best_by_seed: dict[int, dict | None] = field(default_factory=dict)
    invalid_combination_records: list[dict] = field(default_factory=list)
    invalid_combination_count: int = 0
    invalid_y_diff_le_minus_one_total: int = 0
    invalid_y_diff_lt_minus_one_total: int = 0
    eval_count: int = 0
    completed_threshold_count: int = 0

    @classmethod
    def create(
        cls,
        *,
        doc: SweepDocument,
        threshold_values: list[int],
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
    ) -> "_DocumentSweepAccumulator":
        """Create an empty accumulator with every swept value initialized."""
        return cls(
            doc=doc,
            threshold_values=[int(value) for value in threshold_values],
            line_length_values=[int(value) for value in line_length_values],
            line_gap_values=[int(value) for value in line_gap_values],
            seed_values=[int(value) for value in seed_values],
            best_by_threshold=empty_best_value_map(threshold_values),
            best_by_line_length=empty_best_value_map(line_length_values),
            best_by_line_gap=empty_best_value_map(line_gap_values),
            best_by_seed=empty_best_value_map(seed_values),
        )

    def merge_threshold_payload(self, payload: dict) -> None:
        """Merge one completed threshold payload into this document state."""
        threshold = int(payload["threshold"])
        self.eval_count += int(payload["eval_count"])
        self.invalid_combination_count += int(payload.get("invalid_combination_count", 0))
        self.invalid_y_diff_le_minus_one_total += int(payload.get("invalid_y_diff_le_minus_one_total", 0))
        self.invalid_y_diff_lt_minus_one_total += int(payload.get("invalid_y_diff_lt_minus_one_total", 0))
        self.invalid_combination_records.extend(payload.get("invalid_records", []))

        best_threshold = payload["best_threshold"]
        self.best_overall = pick_better_eval(self.best_overall, best_threshold)
        self.best_by_threshold[threshold] = pick_better_eval(
            self.best_by_threshold[threshold],
            best_threshold,
        )
        merge_best_value_payloads(self.best_by_line_length, payload["best_by_line_length"])
        merge_best_value_payloads(self.best_by_line_gap, payload["best_by_line_gap"])
        merge_best_value_payloads(self.best_by_seed, payload["best_by_seed"])
        self.completed_threshold_count += 1

    def is_complete(self) -> bool:
        """Return True when every threshold value has been merged."""
        return int(self.completed_threshold_count) >= int(len(self.threshold_values))

    def build_tuned_payload(self, *, baseline_cfg: HoughBaselineConfig) -> dict:
        """Build the stable per-document result payload used by all schedulers."""
        doc_grid_seconds = float(time.perf_counter() - self.started_at)

        # Preserve a stable fallback payload even when all evaluated rows are invalid.
        if self.best_overall is None:
            best_payload = {
                "tuning_score": None,
                "weighted_along_lines_nls": None,
                PARAM_HOUGH_THRESHOLD: int(baseline_cfg.hough_threshold),
                PARAM_HOUGH_LINE_LENGTH: int(baseline_cfg.hough_line_length),
                PARAM_HOUGH_LINE_GAP: int(baseline_cfg.hough_line_gap),
                PARAM_HOUGH_SEED: int(baseline_cfg.hough_seed),
            }
        else:
            best_payload = dict(self.best_overall)

        doc_best_record = {
            "index": int(self.doc.index),
            "fname": str(self.doc.fname),
            "whole_document_nls": float(self.doc.whole_document_nls),
            "best": best_payload,
            "evaluated_combination_count": int(self.eval_count),
            "invalid_combination_count": int(self.invalid_combination_count),
            "invalid_y_diff_le_minus_one_total": int(self.invalid_y_diff_le_minus_one_total),
            "invalid_y_diff_lt_minus_one_total": int(self.invalid_y_diff_lt_minus_one_total),
            "doc_grid_seconds": float(doc_grid_seconds),
        }

        # Build one influence curve per swept parameter from the best row seen at
        # each value after optimizing over the other three parameters.
        profile_points = {
            PARAM_HOUGH_THRESHOLD: {
                value: point_from_best(doc=self.doc, best_eval=self.best_by_threshold[value], baseline_cfg=baseline_cfg)
                for value in self.threshold_values
            },
            PARAM_HOUGH_LINE_LENGTH: {
                value: point_from_best(doc=self.doc, best_eval=self.best_by_line_length[value], baseline_cfg=baseline_cfg)
                for value in self.line_length_values
            },
            PARAM_HOUGH_LINE_GAP: {
                value: point_from_best(doc=self.doc, best_eval=self.best_by_line_gap[value], baseline_cfg=baseline_cfg)
                for value in self.line_gap_values
            },
            PARAM_HOUGH_SEED: {
                value: point_from_best(doc=self.doc, best_eval=self.best_by_seed[value], baseline_cfg=baseline_cfg)
                for value in self.seed_values
            },
        }

        return {
            "doc_best_record": doc_best_record,
            "profile_points": profile_points,
            "doc_grid_seconds": float(doc_grid_seconds),
            "evaluated_combination_count": int(self.eval_count),
            "invalid_combination_records": self.invalid_combination_records,
            "invalid_combination_count": int(self.invalid_combination_count),
            "invalid_y_diff_le_minus_one_total": int(self.invalid_y_diff_le_minus_one_total),
            "invalid_y_diff_lt_minus_one_total": int(self.invalid_y_diff_lt_minus_one_total),
        }


# Evaluate all line-length and line-gap combinations for one fixed seed/threshold.
def _evaluate_threshold(
    *,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold: int,
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    ref_to_ref_cache,
    log_fn: LogFn,
    combination_bundle_logger=None,
) -> dict:
    """Evaluate all combinations for one threshold value."""
    started_at = time.perf_counter()
    active_seed_values = [int(FIXED_HOUGH_SEED)]
    combinations_for_threshold = int(len(line_length_values) * len(line_gap_values) * len(active_seed_values))
    log_fn(
        f"[threshold-worker-start] fname={doc.fname} threshold={int(threshold)} "
        f"combinations={combinations_for_threshold} line_length_count={len(line_length_values)} "
        f"line_gap_count={len(line_gap_values)} seed_count={len(active_seed_values)}"
    )

    best_threshold: dict | None = None
    best_by_line_length = empty_best_value_map(line_length_values)
    best_by_line_gap = empty_best_value_map(line_gap_values)
    best_by_seed = empty_best_value_map(active_seed_values)
    invalid_records: list[dict] = []
    invalid_combination_count = 0
    invalid_y_diff_le_minus_one_total = 0
    invalid_y_diff_lt_minus_one_total = 0

    eval_count = 0
    progress_log_interval = 200
    progress_chunk_started_at = time.perf_counter()
    for line_length in line_length_values:
        for line_gap in line_gap_values:
            # Seed sweep is temporarily disabled.  The previous loop is kept
            # here, commented out, so restoring seed search is a small diff.
            # for seed in seed_values:
            seed = int(FIXED_HOUGH_SEED)
            eval_row = evaluate_single_combination_values(
                doc=doc,
                hough_threshold=int(threshold),
                hough_line_length=int(line_length),
                hough_line_gap=int(line_gap),
                hough_seed=int(seed),
                align_abs_min_len=float(baseline_cfg.align_abs_min_len),
                align_min_iou_threshold=float(baseline_cfg.align_min_iou_threshold),
                levenshtein_backend=str(levenshtein_backend),
                ref_to_ref_cache=ref_to_ref_cache,
                combination_bundle_logger=combination_bundle_logger,
            )
            eval_row[PARAM_HOUGH_THRESHOLD] = int(threshold)
            eval_row[PARAM_HOUGH_LINE_LENGTH] = int(line_length)
            eval_row[PARAM_HOUGH_LINE_GAP] = int(line_gap)
            eval_row[PARAM_HOUGH_SEED] = int(seed)

            eval_count += 1
            if not bool(eval_row.get("is_valid", True)):
                invalid_combination_count += 1
                invalid_y_diff_le_minus_one_total += int(
                    eval_row.get("coverage_y_diff_le_minus_one_count", 0) or 0
                )
                invalid_y_diff_lt_minus_one_total += int(
                    eval_row.get("coverage_y_diff_lt_minus_one_count", 0) or 0
                )
                invalid_record = _invalid_record_from_eval_row(doc=doc, eval_row=eval_row)
                invalid_records.append(invalid_record)
                if invalid_combination_count == 1 or invalid_combination_count % 200 == 0:
                    log_fn(
                        f"[threshold-worker-invalid] fname={doc.fname} threshold={int(threshold)} "
                        f"invalid_count={invalid_combination_count} completed={eval_count}/{combinations_for_threshold} "
                        f"reason={invalid_record.get('invalid_reason')} "
                        f"y_diff_min={invalid_record.get('coverage_y_diff_min')} "
                        f"y_diff_max={invalid_record.get('coverage_y_diff_max')} "
                        f"y_diff_le_minus_one={invalid_record.get('coverage_y_diff_le_minus_one_count')} "
                        f"y_diff_lt_minus_one={invalid_record.get('coverage_y_diff_lt_minus_one_count')} "
                        f"line_length={int(line_length)} line_gap={int(line_gap)} seed={int(seed)}"
                    )
            if eval_count % progress_log_interval == 0:
                progress_now = time.perf_counter()
                progress_chunk_seconds = float(progress_now - progress_chunk_started_at)
                progress_total_seconds = float(progress_now - started_at)
                progress_seconds_per_combination = progress_chunk_seconds / float(progress_log_interval)
                log_fn(
                    f"[threshold-worker-progress] fname={doc.fname} threshold={int(threshold)} "
                    f"completed={eval_count}/{combinations_for_threshold} "
                    f"chunk_combinations={progress_log_interval} "
                    f"chunk_seconds={progress_chunk_seconds:.3f} "
                    f"seconds_per_combination={progress_seconds_per_combination:.6f} "
                    f"total_elapsed_s={progress_total_seconds:.3f} "
                    f"line_length={int(line_length)} line_gap={int(line_gap)} seed={int(seed)}"
                )
                progress_chunk_started_at = progress_now

            best_threshold = pick_better_eval(best_threshold, eval_row)
            best_by_line_length[int(line_length)] = pick_better_eval(
                best_by_line_length[int(line_length)],
                eval_row,
            )
            best_by_line_gap[int(line_gap)] = pick_better_eval(
                best_by_line_gap[int(line_gap)],
                eval_row,
            )
            best_by_seed[int(seed)] = pick_better_eval(best_by_seed[int(seed)], eval_row)

    return {
        "threshold": int(threshold),
        "best_threshold": best_threshold,
        "best_by_line_length": best_by_line_length,
        "best_by_line_gap": best_by_line_gap,
        "best_by_seed": best_by_seed,
        "eval_count": int(eval_count),
        "invalid_combination_count": int(invalid_combination_count),
        "invalid_y_diff_le_minus_one_total": int(invalid_y_diff_le_minus_one_total),
        "invalid_y_diff_lt_minus_one_total": int(invalid_y_diff_lt_minus_one_total),
        "invalid_records": invalid_records,
        "elapsed_seconds": float(time.perf_counter() - started_at),
    }


# Run the full exhaustive grid for one document.
def tune_single_document(
    *,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    workers: int,
    ref_to_ref_cache=None,
    log_fn: LogFn,
    combination_bundle_logger=None,
) -> dict:
    """Run exhaustive fixed nested-grid tuning for one document."""
    accumulator = _DocumentSweepAccumulator.create(
        doc=doc,
        threshold_values=threshold_values,
        line_length_values=line_length_values,
        line_gap_values=line_gap_values,
        seed_values=seed_values,
    )
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
                    threshold=int(threshold),
                    line_length_values=line_length_values,
                    line_gap_values=line_gap_values,
                    seed_values=seed_values,
                    ref_to_ref_cache=ref_to_ref_cache,
                    log_fn=log_fn,
                    combination_bundle_logger=combination_bundle_logger,
                ): int(threshold)
                for threshold in threshold_values
            }

            for future in as_completed(futures):
                threshold = futures[future]
                payload = future.result()
                accumulator.merge_threshold_payload(payload)

                log_fn(
                    f"[doc-loop] fname={doc.fname} threshold={threshold} done evals={accumulator.eval_count} "
                    f"elapsed_s={float(payload['elapsed_seconds']):.3f} mode=thread"
                )
    else:
        for threshold in threshold_values:
            payload = _evaluate_threshold(
                doc=doc,
                baseline_cfg=baseline_cfg,
                levenshtein_backend=levenshtein_backend,
                threshold=int(threshold),
                line_length_values=line_length_values,
                line_gap_values=line_gap_values,
                seed_values=seed_values,
                ref_to_ref_cache=ref_to_ref_cache,
                log_fn=log_fn,
                combination_bundle_logger=combination_bundle_logger,
            )
            accumulator.merge_threshold_payload(payload)

            log_fn(
                f"[doc-loop] fname={doc.fname} threshold={threshold} done evals={accumulator.eval_count} "
                f"elapsed_s={float(payload['elapsed_seconds']):.3f} mode=serial"
            )

    return accumulator.build_tuned_payload(baseline_cfg=baseline_cfg)


# Log document-level start progress for visibility during long sweeps.
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


# Log document-level completion progress and remaining in-flight count.
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


def _run_document_sweeps_with_global_threshold_queue(
    *,
    doc_iter,
    resolved_total_docs: int,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    threshold_workers: int,
    doc_workers: int,
    ref_to_ref_cache,
    progress_state: dict[str, int],
    progress_lock,
    log_fn: LogFn,
    combination_bundle_logger=None,
    on_document_completed: Callable[[SweepDocument, dict], None] | None = None,
) -> dict[int, dict]:
    """Run threshold tasks from all in-flight documents through one executor.

    The old document-parallel shape created one document future, and each
    document future owned its own threshold executor.  That preserved results,
    but it left threads idle whenever a document had only a few slow thresholds
    remaining.  This global queue keeps the exact same threshold evaluator and
    reducer, while letting any free worker immediately take the next threshold
    from any in-flight document.
    """
    requested_doc_workers = max(1, int(doc_workers))
    requested_threshold_workers = max(1, int(threshold_workers))
    global_threshold_workers = max(1, requested_doc_workers * requested_threshold_workers)
    tuned_by_selection_order: dict[int, dict] = {}
    future_to_document_state: dict = {}
    active_document_states: dict[int, _DocumentSweepAccumulator] = {}
    next_selection_order = 0
    active_document_count = 0

    log_fn(
        f"[doc-threshold-queue] enabled doc_workers={requested_doc_workers} "
        f"threshold_workers_per_doc={requested_threshold_workers} "
        f"global_threshold_workers={global_threshold_workers} docs={resolved_total_docs}"
    )

    with ThreadPoolExecutor(max_workers=global_threshold_workers) as executor:

        def submit_next_document() -> bool:
            """Prepare one document and submit one global task per threshold."""
            nonlocal next_selection_order, active_document_count
            if active_document_count >= requested_doc_workers:
                return False

            try:
                next_doc = next(doc_iter)
            except StopIteration:
                return False

            selection_order = int(next_selection_order)
            next_selection_order += 1
            active_document_count += 1

            _log_doc_started(
                doc=next_doc,
                progress_state=progress_state,
                progress_lock=progress_lock,
                log_fn=log_fn,
            )
            accumulator = _DocumentSweepAccumulator.create(
                doc=next_doc,
                threshold_values=threshold_values,
                line_length_values=line_length_values,
                line_gap_values=line_gap_values,
                seed_values=seed_values,
            )
            active_document_states[selection_order] = accumulator

            for threshold in threshold_values:
                future = executor.submit(
                    _evaluate_threshold,
                    doc=next_doc,
                    baseline_cfg=baseline_cfg,
                    levenshtein_backend=str(levenshtein_backend),
                    threshold=int(threshold),
                    line_length_values=line_length_values,
                    line_gap_values=line_gap_values,
                    seed_values=seed_values,
                    ref_to_ref_cache=ref_to_ref_cache,
                    log_fn=log_fn,
                    combination_bundle_logger=combination_bundle_logger,
                )
                future_to_document_state[future] = (selection_order, int(threshold), accumulator)

            return True

        for _ in range(requested_doc_workers):
            if not submit_next_document():
                break

        while future_to_document_state:
            done_futures, _ = wait(
                future_to_document_state.keys(),
                return_when=FIRST_COMPLETED,
            )
            for future in done_futures:
                selection_order, threshold, accumulator = future_to_document_state.pop(future)
                payload = future.result()
                accumulator.merge_threshold_payload(payload)
                log_fn(
                    f"[doc-loop] fname={accumulator.doc.fname} threshold={threshold} "
                    f"done evals={accumulator.eval_count} "
                    f"elapsed_s={float(payload['elapsed_seconds']):.3f} mode=global-threshold"
                )

                if not accumulator.is_complete():
                    continue

                tuned = accumulator.build_tuned_payload(baseline_cfg=baseline_cfg)
                tuned_by_selection_order[int(selection_order)] = tuned
                _log_doc_completed(
                    doc=accumulator.doc,
                    tuned=tuned,
                    progress_state=progress_state,
                    progress_lock=progress_lock,
                    log_fn=log_fn,
                )
                if on_document_completed is not None:
                    # Dynamic-pool workers use this callback to release the
                    # local document slot immediately.  The callback must not
                    # write tuner metrics; normal summary/CSV/bundle exports
                    # remain owned by the existing output pipeline.
                    on_document_completed(accumulator.doc, tuned)

                # Drop the completed SweepDocument reference immediately so its
                # score matrices and Hough contexts can be reclaimed while the
                # global queue continues with later documents.
                active_document_states.pop(int(selection_order), None)
                active_document_count -= 1
                del accumulator

                while active_document_count < requested_doc_workers:
                    if not submit_next_document():
                        break

    return tuned_by_selection_order


# Run per-document exhaustive sweeps and collect profile points.
def run_document_sweeps(
    *,
    docs: Iterable[SweepDocument],
    total_docs: int | None = None,
    baseline_cfg: HoughBaselineConfig,
    levenshtein_backend: str,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    workers: int,
    doc_workers: int,
    ref_to_ref_cache=None,
    log_fn: LogFn,
    combination_bundle_logger=None,
    on_document_completed: Callable[[SweepDocument, dict], None] | None = None,
) -> dict:
    """Run per-document exhaustive sweeps while retaining only compact results.

    ``workers`` controls threshold-level workers inside one document.
    ``doc_workers`` controls how many documents are processed concurrently.

    ``docs`` may be a list or a streaming iterator.  The streaming mode is the
    RAM-safe path used by the main runner: only in-flight documents keep their
    score matrices and Hough contexts alive.
    """
    profile_points: dict[str, dict[int, list[dict]]] = {
        PARAM_HOUGH_THRESHOLD: {value: [] for value in threshold_values},
        PARAM_HOUGH_LINE_LENGTH: {value: [] for value in line_length_values},
        PARAM_HOUGH_LINE_GAP: {value: [] for value in line_gap_values},
        PARAM_HOUGH_SEED: {value: [] for value in seed_values},
    }
    doc_best_records: list[dict] = []
    invalid_combination_records: list[dict] = []

    grid_eval_started_at = time.perf_counter()
    doc_grid_seconds_total = 0.0

    threshold_workers = max(1, int(workers))
    requested_doc_workers = max(1, int(doc_workers))
    doc_iter = iter(docs)
    if total_docs is None:
        try:
            resolved_total_docs = int(len(docs))  # type: ignore[arg-type]
        except TypeError:
            resolved_total_docs = 0
    else:
        resolved_total_docs = int(total_docs)
    use_parallel_docs = requested_doc_workers > 1 and (resolved_total_docs != 1)

    tuned_by_selection_order: dict[int, dict] = {}

    # Shared counters for document-level progress visibility.
    progress_lock = Lock()
    progress_state: dict[str, int] = {
        "total": int(resolved_total_docs),
        "started": 0,
        "completed": 0,
    }

    scheduler_mode = "global_threshold_queue" if use_parallel_docs else "serial_documents"

    if use_parallel_docs:
        tuned_by_selection_order = _run_document_sweeps_with_global_threshold_queue(
            doc_iter=doc_iter,
            resolved_total_docs=resolved_total_docs,
            baseline_cfg=baseline_cfg,
            levenshtein_backend=str(levenshtein_backend),
            threshold_values=threshold_values,
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            threshold_workers=threshold_workers,
            doc_workers=requested_doc_workers,
            ref_to_ref_cache=ref_to_ref_cache,
            progress_state=progress_state,
            progress_lock=progress_lock,
            log_fn=log_fn,
            combination_bundle_logger=combination_bundle_logger,
            on_document_completed=on_document_completed,
        )
    else:
        mode = "serial" if resolved_total_docs <= 1 else "serial_docs"
        log_fn(
            f"[doc-parallel] {mode} doc_workers={requested_doc_workers} "
            f"threshold_workers_per_doc={threshold_workers} docs={resolved_total_docs}"
        )
        for selection_order, doc in enumerate(doc_iter):
            _log_doc_started(doc=doc, progress_state=progress_state, progress_lock=progress_lock, log_fn=log_fn)
            tuned = tune_single_document(
                doc=doc,
                baseline_cfg=baseline_cfg,
                levenshtein_backend=str(levenshtein_backend),
                threshold_values=threshold_values,
                line_length_values=line_length_values,
                line_gap_values=line_gap_values,
                seed_values=seed_values,
                workers=threshold_workers,
                ref_to_ref_cache=ref_to_ref_cache,
                log_fn=log_fn,
                combination_bundle_logger=combination_bundle_logger,
            )
            tuned_by_selection_order[int(selection_order)] = tuned
            _log_doc_completed(
                doc=doc,
                tuned=tuned,
                progress_state=progress_state,
                progress_lock=progress_lock,
                log_fn=log_fn,
            )
            if on_document_completed is not None:
                # Keep serial mode behavior equivalent to the global-threshold
                # queue: completion is reported as soon as one document is
                # fully tuned, not after the whole run ends.
                on_document_completed(doc, tuned)
            del doc

    # Aggregate in original selection order for deterministic CSV/JSON output.
    for selection_order in sorted(tuned_by_selection_order):
        tuned = tuned_by_selection_order[int(selection_order)]

        doc_best_record = tuned["doc_best_record"]
        doc_best_records.append(doc_best_record)
        doc_grid_seconds_total += float(tuned["doc_grid_seconds"])
        invalid_combination_records.extend(tuned.get("invalid_combination_records", []))

        best_score = doc_best_record["best"].get("tuning_score")
        best_score_str = "None" if best_score is None else f"{float(best_score):.6f}"
        best_weighted = doc_best_record["best"].get("weighted_along_lines_nls")
        best_weighted_str = "None" if best_weighted is None else f"{float(best_weighted):.6f}"
        log_fn(
            f"[doc-best] fname={doc_best_record['fname']} best_tuning_score={best_score_str} "
            f"best_weighted_along_lines={best_weighted_str} "
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

        # The tuned payload has now been reduced into compact aggregate lists.
        del tuned_by_selection_order[int(selection_order)]

    grid_eval_seconds = float(time.perf_counter() - grid_eval_started_at)

    return {
        "profile_points": profile_points,
        "doc_best_records": doc_best_records,
        "invalid_combination_records": invalid_combination_records,
        "invalid_combination_count": int(len(invalid_combination_records)),
        "invalid_y_diff_le_minus_one_total": int(
            sum(int(row.get("coverage_y_diff_le_minus_one_count", 0) or 0) for row in invalid_combination_records)
        ),
        "invalid_y_diff_lt_minus_one_total": int(
            sum(int(row.get("coverage_y_diff_lt_minus_one_count", 0) or 0) for row in invalid_combination_records)
        ),
        "grid_eval_seconds": grid_eval_seconds,
        "doc_grid_seconds_total": float(doc_grid_seconds_total),
        "scheduler_mode": str(scheduler_mode),
    }


__all__ = [
    "tune_single_document",
    "run_document_sweeps",
]
