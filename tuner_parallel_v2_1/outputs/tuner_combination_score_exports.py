from __future__ import annotations

"""Compressed scalar score-table writer for every evaluated Hough combination.

The tuner has two different per-combination output needs:

* scalar analysis needs scores, counts, timings, and validity flags;
* visual diagnostics need geometry for only selected combinations.

This module owns the scalar path.  It writes one gzip-compressed CSV row per
evaluated combination and deliberately refuses to store raw lines, final line
geometry, text snippets, matrix arrays, or bundle payloads.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from csv import DictWriter
import gzip
import math
from pathlib import Path
from threading import Lock
import time
from typing import Any


BASE_COMBINATION_SCORE_FIELDNAMES = [
    "doc_index",
    "fname",
    "whole_document_nls",
    "hough_threshold",
    "hough_line_length",
    "hough_line_gap",
    "hough_seed",
    "matrix_rows_ref_to_pred",
    "matrix_cols_ref_to_pred",
    "matrix_rows_ref_to_ref",
    "matrix_cols_ref_to_ref",
    "is_valid",
    "invalid_reason",
    "invalid_error_message",
    "metric_outcome_reason",
    "tuning_score",
    "weighted_along_lines_nls",
    "correct_ref_coverage",
    "missing_ref_coverage",
    "repetition_on_ref",
    "hallucination",
    "line_count",
    "used_line_count",
    "used_line_count_ref_to_ref",
    "line_guided_columns",
    "fallback_columns",
    "raw_line_count",
    "raw_line_count_ref_to_ref",
    "skimage_raw_line_count_before_direction_filter",
    "skimage_raw_line_count_before_direction_filter_ref_to_ref",
    "direction_rejected_line_count",
    "direction_rejected_line_count_ref_to_ref",
    "candidate_line_count",
    "candidate_line_count_ref_to_ref",
    "threshold_start",
    "threshold_start_ref_to_ref",
    "coverage_y_diff_size",
    "coverage_y_diff_min",
    "coverage_y_diff_max",
    "coverage_y_diff_le_minus_one_count",
    "coverage_y_diff_lt_minus_one_count",
    "line_nls_filter_enabled",
    "min_surviving_line_nls",
    "line_nls_filter_input_line_count",
    "line_nls_filter_scored_line_count",
    "line_nls_filter_removed_line_count",
    "line_nls_filter_surviving_line_count",
    "line_nls_filter_removed_column_count",
    "line_nls_filter_surviving_column_count",
    "line_nls_filter_all_lines_removed",
    "timing_hough_detect_ref_to_pred_seconds",
    "timing_filter_ref_to_pred_seconds",
    "timing_hough_detect_ref_to_ref_seconds",
    "timing_filter_ref_to_ref_seconds",
    "timing_hough_detect_seconds",
    "timing_filter_seconds",
    "timing_detect_filter_seconds",
    "timing_build_bundle_seconds",
    "timing_line_nls_filter_seconds",
    "timing_coverage_seconds",
    "timing_levenshtein_seconds",
    "timing_total_seconds",
]

FILTER_PROFILE_TIMING_FIELDNAMES = [
    "timing_filter_prepare_candidates_ref_to_pred_seconds",
    "timing_filter_build_candidate_coverages_ref_to_pred_seconds",
    "timing_filter_possible_pair_generation_ref_to_pred_seconds",
    "timing_filter_exact_iou_ref_to_pred_seconds",
    "timing_filter_component_build_ref_to_pred_seconds",
    "timing_filter_merge_components_ref_to_pred_seconds",
    "timing_filter_final_assignment_ref_to_pred_seconds",
    "timing_filter_finalize_outputs_ref_to_pred_seconds",
    "timing_filter_total_profiled_ref_to_pred_seconds",
    "timing_filter_prepare_candidates_ref_to_ref_seconds",
    "timing_filter_build_candidate_coverages_ref_to_ref_seconds",
    "timing_filter_possible_pair_generation_ref_to_ref_seconds",
    "timing_filter_exact_iou_ref_to_ref_seconds",
    "timing_filter_component_build_ref_to_ref_seconds",
    "timing_filter_merge_components_ref_to_ref_seconds",
    "timing_filter_final_assignment_ref_to_ref_seconds",
    "timing_filter_finalize_outputs_ref_to_ref_seconds",
    "timing_filter_total_profiled_ref_to_ref_seconds",
]

FILTER_PROFILE_COUNT_FIELDNAMES = [
    "filter_input_line_count_ref_to_pred",
    "filter_prepared_candidate_count_ref_to_pred",
    "filter_candidate_coverage_count_ref_to_pred",
    "filter_possible_overlap_pair_count_ref_to_pred",
    "filter_merge_edge_count_ref_to_pred",
    "filter_component_count_ref_to_pred",
    "filter_merged_coverage_count_ref_to_pred",
    "filter_finalize_prune_iteration_count_ref_to_pred",
    "filter_final_line_count_ref_to_pred",
    "filter_fallback_candidate_used_ref_to_pred",
    "filter_input_line_count_ref_to_ref",
    "filter_prepared_candidate_count_ref_to_ref",
    "filter_candidate_coverage_count_ref_to_ref",
    "filter_possible_overlap_pair_count_ref_to_ref",
    "filter_merge_edge_count_ref_to_ref",
    "filter_component_count_ref_to_ref",
    "filter_merged_coverage_count_ref_to_ref",
    "filter_finalize_prune_iteration_count_ref_to_ref",
    "filter_final_line_count_ref_to_ref",
    "filter_fallback_candidate_used_ref_to_ref",
]

COMBINATION_SCORE_FIELDNAMES = [
    *BASE_COMBINATION_SCORE_FIELDNAMES,
    *FILTER_PROFILE_TIMING_FIELDNAMES,
    *FILTER_PROFILE_COUNT_FIELDNAMES,
]

DEFAULT_MAX_PENDING_SCORE_WRITES = 4


def _csv_scalar_value(value: Any) -> Any:
    """Return one stable CSV scalar, leaving missing values as empty cells."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        return f"{float(value):.10f}" if math.isfinite(float(value)) else ""
    return value


class CombinationScoreTableWriter:
    """Write compact combination-score rows in completed-document batches."""

    def __init__(
        self,
        *,
        output_csv_gz: Path,
        max_pending_writes: int = DEFAULT_MAX_PENDING_SCORE_WRITES,
    ) -> None:
        """Create the compressed score table and write its header immediately."""
        self.output_csv_gz = Path(output_csv_gz)
        self.output_csv_gz.parent.mkdir(parents=True, exist_ok=True)
        self.max_pending_writes = max(1, int(max_pending_writes))

        self._lock = Lock()
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="combination_score_table_writer")
        self._pending_futures: list[Future] = []
        self._row_count = 0
        self._write_seconds = 0.0
        self._closed = False

        self._file_handle = gzip.open(self.output_csv_gz, mode="wt", encoding="utf-8", newline="")
        self._writer = DictWriter(self._file_handle, fieldnames=COMBINATION_SCORE_FIELDNAMES)
        self._writer.writeheader()

    def submit_document_rows(self, rows: list[dict]) -> None:
        """Queue one completed document's score rows for sequential writing."""
        if self._closed:
            raise RuntimeError("Cannot write combination scores after the writer has been closed")
        if not rows:
            return

        # Copy the list container so callers can immediately clear their
        # document accumulator and release references to evaluated rows.
        future = self._executor.submit(self._write_rows, list(rows))
        with self._lock:
            self._pending_futures.append(future)
        self._wait_until_pending_writes_are_bounded()

    def _write_rows(self, rows: list[dict]) -> None:
        """Write one completed-document row batch to the gzip CSV stream."""
        started_at = time.perf_counter()
        for row in rows:
            self._writer.writerow(
                {
                    field_name: _csv_scalar_value(row.get(field_name))
                    for field_name in COMBINATION_SCORE_FIELDNAMES
                }
            )
        with self._lock:
            self._row_count += int(len(rows))
            self._write_seconds += float(time.perf_counter() - started_at)

    def _wait_until_pending_writes_are_bounded(self) -> None:
        """Surface writer errors and bound queued completed-document batches."""
        while True:
            with self._lock:
                pending_futures = [future for future in self._pending_futures if not future.done()]
                completed_futures = [future for future in self._pending_futures if future.done()]
                self._pending_futures = pending_futures

            for completed_future in completed_futures:
                completed_future.result()

            if len(pending_futures) <= self.max_pending_writes:
                return

            done_futures, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
            for done_future in done_futures:
                done_future.result()

    def close(self) -> None:
        """Wait for queued writes, then close the gzip stream exactly once."""
        if self._closed:
            return

        while True:
            with self._lock:
                pending_futures = list(self._pending_futures)
                self._pending_futures.clear()
            if not pending_futures:
                break
            for future in pending_futures:
                future.result()

        self._executor.shutdown(wait=True)
        self._file_handle.close()
        self._closed = True

    def summary(self) -> dict:
        """Return JSON-ready writer statistics for the run summary."""
        with self._lock:
            return {
                "enabled": True,
                "csv_gz_path": str(self.output_csv_gz),
                "row_count": int(self._row_count),
                "write_seconds": float(self._write_seconds),
                "field_count": int(len(COMBINATION_SCORE_FIELDNAMES)),
                "format": "csv.gz",
            }


__all__ = [
    "COMBINATION_SCORE_FIELDNAMES",
    "CombinationScoreTableWriter",
    "DEFAULT_MAX_PENDING_SCORE_WRITES",
]
