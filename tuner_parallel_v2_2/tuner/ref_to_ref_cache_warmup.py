from __future__ import annotations

"""Reference-self document-pack cache warm-up for exhaustive Hough tuning.

This module fills the exact ``ref_to_ref`` document-pack cache without evaluating
the prediction side.  A warm production workflow can therefore run:

1. warm the reference-self cache with ``--ref-to-ref-cache-warm-only``;
2. run the real tuner with ``--ref-to-ref-cache-mode read-only``.

The warm-up path calls the same low-level reference-self payload builder used by
the normal evaluator, so it cannot drift from production scoring semantics.
"""

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Lock
import time
from typing import Iterable

try:
    from .hough_eval import compute_reference_self_payload_for_combination
    from .tuner_config import (
        HoughBaselineConfig,
        LogFn,
        SweepDocument,
    )
except ImportError:
    from tuner.hough_eval import compute_reference_self_payload_for_combination  # type: ignore
    from tuner.tuner_config import HoughBaselineConfig, LogFn, SweepDocument  # type: ignore


@dataclass
class RefToRefWarmupSummary:
    """Compact summary of a cache warm-up run."""

    document_count: int
    threshold_task_count: int
    combination_count: int
    elapsed_seconds: float

    def as_dict(self) -> dict:
        """Return a JSON-friendly representation for the final summary file."""
        return {
            "document_count": int(self.document_count),
            "threshold_task_count": int(self.threshold_task_count),
            "combination_count": int(self.combination_count),
            "elapsed_seconds": float(self.elapsed_seconds),
        }


def _warm_one_threshold(
    *,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    threshold: int,
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    ref_to_ref_cache,
    log_fn: LogFn,
) -> dict:
    """Warm every line-length/gap/seed entry for one threshold and document."""
    started_at = time.perf_counter()
    warmed_combination_count = 0
    threshold = int(threshold)
    total_for_threshold = int(len(line_length_values) * len(line_gap_values) * len(seed_values))
    log_fn(
        f"[ref-to-ref-warm-threshold-start] fname={doc.fname} threshold={threshold} "
        f"combinations={total_for_threshold}"
    )

    threshold_ref_to_ref_cache = ref_to_ref_cache
    if ref_to_ref_cache is not None and hasattr(ref_to_ref_cache, "begin_threshold"):
        # Warm-up uses the same document-level cache view as production.  The
        # actual payload builder remains shared, so warming cannot drift from
        # the evaluator's ref_to_ref behavior.
        threshold_ref_to_ref_cache = ref_to_ref_cache.begin_threshold(
            doc=doc,
            hough_threshold=int(threshold),
            line_length_values=[int(value) for value in line_length_values],
            line_gap_values=[int(value) for value in line_gap_values],
            seed_values=[int(value) for value in seed_values],
            align_abs_min_len=float(baseline_cfg.align_abs_min_len),
            align_min_iou_threshold=float(baseline_cfg.align_min_iou_threshold),
        )

    try:
        for line_length in line_length_values:
            for line_gap in line_gap_values:
                for seed in seed_values:
                    line_length = int(line_length)
                    line_gap = int(line_gap)
                    seed = int(seed)

                    def compute_payload() -> dict:
                        """Compute one exact cache payload on miss."""
                        return compute_reference_self_payload_for_combination(
                            doc=doc,
                            hough_threshold=threshold,
                            hough_line_length=line_length,
                            hough_line_gap=line_gap,
                            hough_seed=seed,
                            align_abs_min_len=float(baseline_cfg.align_abs_min_len),
                            align_min_iou_threshold=float(baseline_cfg.align_min_iou_threshold),
                        )

                    threshold_ref_to_ref_cache.get_or_compute(
                        doc=doc,
                        hough_threshold=threshold,
                        hough_line_length=line_length,
                        hough_line_gap=line_gap,
                        hough_seed=seed,
                        align_abs_min_len=float(baseline_cfg.align_abs_min_len),
                        align_min_iou_threshold=float(baseline_cfg.align_min_iou_threshold),
                        compute_payload=compute_payload,
                    )
                    warmed_combination_count += 1
    finally:
        if threshold_ref_to_ref_cache is not ref_to_ref_cache and hasattr(threshold_ref_to_ref_cache, "close"):
            threshold_ref_to_ref_cache.close()

    elapsed_seconds = float(time.perf_counter() - started_at)
    log_fn(
        f"[ref-to-ref-warm-threshold-done] fname={doc.fname} threshold={threshold} "
        f"combinations={warmed_combination_count} elapsed_s={elapsed_seconds:.3f}"
    )
    return {
        "document_index": int(doc.index),
        "fname": str(doc.fname),
        "threshold": int(threshold),
        "combination_count": int(warmed_combination_count),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _begin_ref_to_ref_document_cache_if_available(
    *,
    ref_to_ref_cache,
    doc: SweepDocument,
    baseline_cfg: HoughBaselineConfig,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
):
    """Create one document-level cache session when the cache supports it."""
    if ref_to_ref_cache is None or not hasattr(ref_to_ref_cache, "begin_document"):
        return ref_to_ref_cache
    return ref_to_ref_cache.begin_document(
        doc=doc,
        threshold_values=[int(value) for value in threshold_values],
        line_length_values=[int(value) for value in line_length_values],
        line_gap_values=[int(value) for value in line_gap_values],
        seed_values=[int(value) for value in seed_values],
        align_abs_min_len=float(baseline_cfg.align_abs_min_len),
        align_min_iou_threshold=float(baseline_cfg.align_min_iou_threshold),
    )


def warm_ref_to_ref_cache_for_documents(
    *,
    docs: Iterable[SweepDocument],
    total_docs: int,
    baseline_cfg: HoughBaselineConfig,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    workers: int,
    doc_workers: int,
    ref_to_ref_cache,
    log_fn: LogFn,
) -> RefToRefWarmupSummary:
    """Warm reference-self cache entries for a document stream.

    The scheduler keeps at most ``doc_workers`` prepared documents resident in
    memory, while one global threshold executor keeps workers busy across those
    active documents.  This mirrors the production scheduler shape without
    touching any ref-to-pred or final metric logic.
    """
    if ref_to_ref_cache is None or not bool(getattr(ref_to_ref_cache, "enabled", False)):
        raise ValueError("Reference-self warm-up requires ref_to_ref_cache mode 'auto' or 'read-only'.")

    started_at = time.perf_counter()
    doc_iter = iter(docs)
    requested_doc_workers = max(1, int(doc_workers))
    requested_threshold_workers = max(1, int(workers))
    global_threshold_workers = max(1, requested_doc_workers * requested_threshold_workers)
    active_document_count = 0
    next_selection_order = 0
    threshold_task_count = 0
    combination_count = 0
    future_to_state: dict = {}
    active_docs_by_order: dict[int, dict] = {}
    progress_lock = Lock()
    completed_doc_count = 0

    log_fn(
        f"[ref-to-ref-warm-start] docs={int(total_docs)} doc_workers={requested_doc_workers} "
        f"threshold_workers_per_doc={requested_threshold_workers} "
        f"global_threshold_workers={global_threshold_workers}"
    )

    with ThreadPoolExecutor(max_workers=global_threshold_workers) as executor:

        def submit_next_document() -> bool:
            """Prepare one document's threshold tasks if capacity is available."""
            nonlocal active_document_count, next_selection_order, threshold_task_count
            if active_document_count >= requested_doc_workers:
                return False

            try:
                doc = next(doc_iter)
            except StopIteration:
                return False

            selection_order = int(next_selection_order)
            next_selection_order += 1
            active_document_count += 1
            ref_to_ref_document_cache = _begin_ref_to_ref_document_cache_if_available(
                ref_to_ref_cache=ref_to_ref_cache,
                doc=doc,
                baseline_cfg=baseline_cfg,
                threshold_values=threshold_values,
                line_length_values=line_length_values,
                line_gap_values=line_gap_values,
                seed_values=seed_values,
            )
            active_docs_by_order[selection_order] = {
                "doc": doc,
                "ref_to_ref_document_cache": ref_to_ref_document_cache,
                "remaining_thresholds": int(len(threshold_values)),
                "combination_count": 0,
                "started_at": time.perf_counter(),
            }
            log_fn(
                f"[ref-to-ref-warm-doc-start] fname={doc.fname} index={doc.index} "
                f"started={next_selection_order}/{int(total_docs)} active_docs={active_document_count}"
            )

            for threshold in threshold_values:
                future = executor.submit(
                    _warm_one_threshold,
                    doc=doc,
                    baseline_cfg=baseline_cfg,
                    threshold=int(threshold),
                    line_length_values=line_length_values,
                    line_gap_values=line_gap_values,
                    seed_values=seed_values,
                    ref_to_ref_cache=ref_to_ref_document_cache,
                    log_fn=log_fn,
                )
                future_to_state[future] = (selection_order, int(threshold))
                threshold_task_count += 1

            return True

        for _ in range(requested_doc_workers):
            if not submit_next_document():
                break

        while future_to_state:
            done_futures, _ = wait(
                future_to_state.keys(),
                return_when=FIRST_COMPLETED,
            )
            for future in done_futures:
                selection_order, threshold = future_to_state.pop(future)
                payload = future.result()
                active_state = active_docs_by_order[int(selection_order)]
                active_state["remaining_thresholds"] -= 1
                active_state["combination_count"] += int(payload.get("combination_count", 0))
                combination_count += int(payload.get("combination_count", 0))

                if int(active_state["remaining_thresholds"]) > 0:
                    continue

                doc = active_state["doc"]
                ref_to_ref_document_cache = active_state.get("ref_to_ref_document_cache")
                elapsed = float(time.perf_counter() - float(active_state["started_at"]))
                with progress_lock:
                    completed_doc_count += 1
                    completed = int(completed_doc_count)
                log_fn(
                    f"[ref-to-ref-warm-doc-done] fname={doc.fname} index={doc.index} "
                    f"combinations={int(active_state['combination_count'])} "
                    f"elapsed_s={elapsed:.3f} completed={completed}/{int(total_docs)}"
                )

                active_docs_by_order.pop(int(selection_order), None)
                active_document_count -= 1
                del doc

                while active_document_count < requested_doc_workers:
                    if not submit_next_document():
                        break

                if ref_to_ref_document_cache is not None and hasattr(
                    ref_to_ref_document_cache,
                    "submit_completed_document_write",
                ):
                    # Queue the finished document cache after starting the next
                    # document so warm-up workers keep consuming threshold work.
                    ref_to_ref_document_cache.submit_completed_document_write()

    elapsed_seconds = float(time.perf_counter() - started_at)
    log_fn(
        f"[ref-to-ref-warm-done] docs={completed_doc_count} "
        f"threshold_tasks={threshold_task_count} combinations={combination_count} "
        f"elapsed_s={elapsed_seconds:.3f}"
    )
    return RefToRefWarmupSummary(
        document_count=int(completed_doc_count),
        threshold_task_count=int(threshold_task_count),
        combination_count=int(combination_count),
        elapsed_seconds=float(elapsed_seconds),
    )


__all__ = [
    "RefToRefWarmupSummary",
    "warm_ref_to_ref_cache_for_documents",
]
