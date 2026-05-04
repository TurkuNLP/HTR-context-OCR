from __future__ import annotations

"""Document loading/preparation for fixed-grid Hough tuning."""

import time
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    from .along_lines_fast import build_stride_blocks
    from .line_alignment_pipeline_fast import precompute_hough_context
    from .matrix_sources import (
        categorize_pkl_load_failure,
        load_matrix_from_cache,
        matrix_cache_key,
        matrix_cache_path,
        save_matrix_to_cache,
        should_enable_scores_pkl_source,
    )
    from .pkl_index_readonly import (
        load_matrix_from_scores_pkl_index_readonly,
        load_score_stream_index_readonly,
    )
    from .tuner_config import LogFn, SweepDocument
except ImportError:
    from along_lines_fast import build_stride_blocks  # type: ignore
    from line_alignment_pipeline_fast import precompute_hough_context  # type: ignore
    from matrix_sources import (  # type: ignore
        categorize_pkl_load_failure,
        load_matrix_from_cache,
        matrix_cache_key,
        matrix_cache_path,
        save_matrix_to_cache,
        should_enable_scores_pkl_source,
    )
    from pkl_index_readonly import (  # type: ignore
        load_matrix_from_scores_pkl_index_readonly,
        load_score_stream_index_readonly,
    )
    from tuner_config import LogFn, SweepDocument  # type: ignore

from levenshtein_metric import BACKEND_C, SUPPORTED_BACKENDS, normalized_levenshtein_similarity
from runfile_records import load_run_items, same_file
from score_matrix_builder import compute_score_matrix


def _no_log(_: str) -> None:
    return


def ensure_backend_available(backend: str) -> None:
    """Validate backend and ensure C-backed rapidfuzz path is usable when requested."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend!r}. Allowed: {SUPPORTED_BACKENDS!r}")
    if backend == BACKEND_C:
        try:
            _ = normalized_levenshtein_similarity("a", "b", backend=BACKEND_C)
        except Exception as exc:
            raise RuntimeError(
                "Levenshtein backend 'c' was requested but is not available. "
                "Install rapidfuzz in the runtime environment."
            ) from exc


def _build_doc_blocks(pred: str, ref: str, matrix: np.ndarray, window_stride: int) -> tuple[list[str], list[str]]:
    """Precompute stride blocks once per document and reuse in all evaluations."""
    if matrix.ndim != 2:
        return [], []
    n_ref, n_pred = int(matrix.shape[0]), int(matrix.shape[1])
    pred_blocks = build_stride_blocks(pred, n_blocks=n_pred, stride=int(window_stride))
    ref_blocks = build_stride_blocks(ref, n_blocks=n_ref, stride=int(window_stride))
    return pred_blocks, ref_blocks


def load_documents(
    *,
    runfile_json: Path,
    window_size: int,
    window_stride: int,
    hough_start: float,
    levenshtein_backend: str,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    timing_out: dict | None = None,
    log_fn: LogFn | None = None,
) -> list[SweepDocument]:
    """Load runfile documents using layered matrix sources and collect timings.

    Matrix source priority per document:
    1) tuner-local npz cache (if enabled and hit)
    2) read-only scores.pkl via prebuilt index (if configured and valid)
    3) on-demand matrix recomputation
    """
    ensure_backend_available(levenshtein_backend)
    log = _no_log if log_fn is None else log_fn
    load_started_at = time.perf_counter()

    if int(window_size) <= 0 or int(window_stride) <= 0:
        raise ValueError("window_size and window_stride must be positive")
    if max_items is not None and int(max_items) <= 0:
        raise ValueError("max_items must be positive when provided")
    if not runfile_json.exists():
        raise FileNotFoundError(f"Missing runfile JSON: {runfile_json}")
    if matrix_cache_dir is not None and Path(matrix_cache_dir).exists() and not Path(matrix_cache_dir).is_dir():
        raise NotADirectoryError(f"matrix_cache_dir is not a directory: {matrix_cache_dir}")

    targets = [str(v) for v in (target_fnames or []) if str(v).strip()]
    run_items = load_run_items(runfile_json)
    cache_dir = None if matrix_cache_dir is None else Path(matrix_cache_dir)

    cache_hits = 0
    cache_misses = 0
    cache_stores = 0
    cache_load_errors = 0
    cache_store_errors = 0

    matrix_total_seconds = 0.0
    matrix_compute_seconds = 0.0
    matrix_cache_load_seconds = 0.0
    matrix_cache_store_seconds = 0.0
    matrix_pkl_load_seconds = 0.0
    pkl_index_prepare_seconds = 0.0

    matrix_source_npz_hits = 0
    matrix_source_pkl_hits = 0
    matrix_source_computed = 0

    pkl_index_source = "disabled"
    pkl_index_entry_count = 0
    pkl_index_cache_file_used: str | None = None
    pkl_index_load_failures = 0
    pkl_lookup_misses = 0
    pkl_ref_text_mismatch_count = 0
    pkl_pred_text_mismatch_count = 0
    pkl_shape_mismatch_count = 0
    pkl_other_failure_count = 0

    whole_document_nls_seconds = 0.0
    block_build_seconds = 0.0
    hough_context_precompute_seconds = 0.0

    pkl_enabled, pkl_disabled_reason = should_enable_scores_pkl_source(
        scores_pkl_ref_to_pred=scores_pkl_ref_to_pred,
        window_size=int(window_size),
        window_stride=int(window_stride),
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        log_fn=log,
    )

    pkl_scores_path = None if scores_pkl_ref_to_pred is None else Path(scores_pkl_ref_to_pred)
    pkl_index_by_fname: dict[str, dict] | None = None

    if pkl_enabled and pkl_scores_path is not None:
        t_idx = time.perf_counter()
        try:
            index_result = load_score_stream_index_readonly(
                scores_pkl=Path(pkl_scores_path),
                explicit_cache_file=None if score_index_cache_file is None else Path(score_index_cache_file),
                cache_dir=None if score_index_cache_dir is None else Path(score_index_cache_dir),
                log_fn=log,
            )
            pkl_index_prepare_seconds += time.perf_counter() - t_idx
            pkl_index_by_fname = index_result.index_by_fname
            pkl_index_source = str(index_result.source)
            pkl_index_entry_count = int(len(index_result.index_by_fname))
            pkl_index_cache_file_used = (
                None if index_result.cache_file is None else str(index_result.cache_file)
            )
            log(
                f"[pkl-index] enabled source={pkl_index_source} entries={pkl_index_entry_count} "
                f"cache_file={pkl_index_cache_file_used}"
            )
        except Exception as exc:
            pkl_index_prepare_seconds += time.perf_counter() - t_idx
            pkl_index_load_failures += 1
            pkl_enabled = False
            pkl_disabled_reason = f"index_load_error:{exc!r}"
            log(f"[pkl-index] disabled due to index load error: {exc!r}")
    else:
        log(f"[pkl-index] disabled reason={pkl_disabled_reason}")

    docs: list[SweepDocument] = []
    matched = 0

    for item in run_items:
        fname = str(item["fname"])
        if targets and not any(same_file(fname, t) for t in targets):
            continue

        matched += 1
        if max_items is not None and len(docs) >= int(max_items):
            break

        pred = str(item["pred"])
        ref = str(item["ref"])
        matrix_started_at = time.perf_counter()

        matrix: np.ndarray | None = None
        matrix_source = "unknown"

        if cache_dir is not None:
            cache_key = matrix_cache_key(
                ref_text=ref,
                pred_text=pred,
                window_size=int(window_size),
                window_stride=int(window_stride),
            )
            cache_path = matrix_cache_path(cache_dir=cache_dir, cache_key=cache_key)
            if cache_path.exists():
                try:
                    t0 = time.perf_counter()
                    matrix = load_matrix_from_cache(cache_path=cache_path)
                    matrix_cache_load_seconds += time.perf_counter() - t0
                    cache_hits += 1
                    matrix_source = "npz_hit"
                    matrix_source_npz_hits += 1
                except Exception as exc:
                    cache_load_errors += 1
                    cache_misses += 1
                    log(f"[cache] read_error fname={Path(fname).name} path={cache_path} err={exc!r}")
            else:
                cache_misses += 1

        if matrix is None and pkl_enabled and pkl_scores_path is not None and pkl_index_by_fname is not None:
            t0 = time.perf_counter()
            pkl_result = load_matrix_from_scores_pkl_index_readonly(
                scores_pkl=Path(pkl_scores_path),
                score_index_by_fname=pkl_index_by_fname,
                fname=Path(fname).name,
                expected_ref_text=ref,
                expected_pred_text=pred,
                window_size=int(window_size),
                window_stride=int(window_stride),
            )
            matrix_pkl_load_seconds += time.perf_counter() - t0

            if pkl_result.matrix is not None:
                matrix = pkl_result.matrix
                matrix_source = "scores_pkl"
                matrix_source_pkl_hits += 1

                if cache_dir is not None:
                    cache_key = matrix_cache_key(
                        ref_text=ref,
                        pred_text=pred,
                        window_size=int(window_size),
                        window_stride=int(window_stride),
                    )
                    cache_path = matrix_cache_path(cache_dir=cache_dir, cache_key=cache_key)
                    if not cache_path.exists():
                        try:
                            t_store = time.perf_counter()
                            save_matrix_to_cache(cache_path=cache_path, matrix=matrix)
                            matrix_cache_store_seconds += time.perf_counter() - t_store
                            cache_stores += 1
                        except Exception as exc:
                            cache_store_errors += 1
                            log(f"[cache] write_error fname={Path(fname).name} path={cache_path} err={exc!r}")
            else:
                category = categorize_pkl_load_failure(pkl_result.reason)
                if category == "index_miss":
                    pkl_lookup_misses += 1
                elif category == "ref_text_mismatch":
                    pkl_ref_text_mismatch_count += 1
                elif category == "pred_text_mismatch":
                    pkl_pred_text_mismatch_count += 1
                elif category == "shape_mismatch":
                    pkl_shape_mismatch_count += 1
                else:
                    pkl_other_failure_count += 1

        if matrix is None:
            t0 = time.perf_counter()
            matrix = compute_score_matrix(ref, pred, window_size=int(window_size), window_stride=int(window_stride))
            matrix_compute_seconds += time.perf_counter() - t0
            matrix_source = "computed"
            matrix_source_computed += 1

            if cache_dir is not None:
                cache_key = matrix_cache_key(
                    ref_text=ref,
                    pred_text=pred,
                    window_size=int(window_size),
                    window_stride=int(window_stride),
                )
                cache_path = matrix_cache_path(cache_dir=cache_dir, cache_key=cache_key)
                try:
                    t_store = time.perf_counter()
                    save_matrix_to_cache(cache_path=cache_path, matrix=matrix)
                    matrix_cache_store_seconds += time.perf_counter() - t_store
                    cache_stores += 1
                except Exception as exc:
                    cache_store_errors += 1
                    log(f"[cache] write_error fname={Path(fname).name} path={cache_path} err={exc!r}")

        matrix_total_seconds += time.perf_counter() - matrix_started_at

        t_nls = time.perf_counter()
        whole_nls = float(normalized_levenshtein_similarity(pred, ref, backend=levenshtein_backend))
        whole_document_nls_seconds += time.perf_counter() - t_nls

        t_blocks = time.perf_counter()
        pred_blocks, ref_blocks = _build_doc_blocks(pred, ref, matrix, window_stride=int(window_stride))
        block_build_seconds += time.perf_counter() - t_blocks

        t_ctx = time.perf_counter()
        hough_ctx = precompute_hough_context(matrix, start_init=float(hough_start))
        hough_context_precompute_seconds += time.perf_counter() - t_ctx

        docs.append(
            SweepDocument(
                index=int(item["index"]),
                fname=Path(fname).name,
                pred=pred,
                ref=ref,
                matrix=matrix,
                whole_document_nls=whole_nls,
                pred_blocks=pred_blocks,
                ref_blocks=ref_blocks,
                hough_ctx=hough_ctx,
            )
        )
        log(
            f"[load] {len(docs)} fname={Path(fname).name} matrix_shape={matrix.shape} "
            f"whole_nls={whole_nls:.6f} matrix_source={matrix_source}"
        )

    if targets and matched == 0:
        raise KeyError(f"None of target_fnames were found in runfile: {targets!r}")
    if not docs:
        raise RuntimeError("No documents selected for parameter sweep.")

    if cache_dir is not None:
        log(
            f"[cache] dir={cache_dir} hits={cache_hits} misses={cache_misses} stores={cache_stores} "
            f"read_errors={cache_load_errors} write_errors={cache_store_errors}"
        )

    load_total_seconds = time.perf_counter() - load_started_at
    log(
        f"[timing] load_total_s={load_total_seconds:.3f} matrix_total_s={matrix_total_seconds:.3f} "
        f"matrix_compute_s={matrix_compute_seconds:.3f} matrix_cache_load_s={matrix_cache_load_seconds:.3f} "
        f"matrix_cache_store_s={matrix_cache_store_seconds:.3f} matrix_pkl_load_s={matrix_pkl_load_seconds:.3f} "
        f"pkl_index_prepare_s={pkl_index_prepare_seconds:.3f} whole_doc_nls_s={whole_document_nls_seconds:.3f} "
        f"precompute_blocks_s={block_build_seconds:.3f} hough_ctx_precompute_s={hough_context_precompute_seconds:.3f}"
    )

    if timing_out is not None:
        timing_out.clear()
        timing_out.update(
            {
                "load_documents_total_seconds": float(load_total_seconds),
                "matrix_total_seconds": float(matrix_total_seconds),
                "matrix_compute_seconds": float(matrix_compute_seconds),
                "matrix_cache_load_seconds": float(matrix_cache_load_seconds),
                "matrix_cache_store_seconds": float(matrix_cache_store_seconds),
                "matrix_pkl_load_seconds": float(matrix_pkl_load_seconds),
                "pkl_index_prepare_seconds": float(pkl_index_prepare_seconds),
                "whole_document_nls_seconds": float(whole_document_nls_seconds),
                "precompute_blocks_seconds": float(block_build_seconds),
                "hough_context_precompute_seconds": float(hough_context_precompute_seconds),
                "cache_hits": int(cache_hits),
                "cache_misses": int(cache_misses),
                "cache_stores": int(cache_stores),
                "cache_read_errors": int(cache_load_errors),
                "cache_write_errors": int(cache_store_errors),
                "matrix_source_npz_hits": int(matrix_source_npz_hits),
                "matrix_source_pkl_hits": int(matrix_source_pkl_hits),
                "matrix_source_computed": int(matrix_source_computed),
                "scores_pkl_enabled": bool(pkl_enabled),
                "scores_pkl_disabled_reason": pkl_disabled_reason,
                "scores_pkl_path": None if pkl_scores_path is None else str(pkl_scores_path),
                "scores_pkl_index_source": str(pkl_index_source),
                "scores_pkl_index_entry_count": int(pkl_index_entry_count),
                "scores_pkl_index_cache_file_used": pkl_index_cache_file_used,
                "scores_pkl_index_load_failures": int(pkl_index_load_failures),
                "scores_pkl_lookup_misses": int(pkl_lookup_misses),
                "scores_pkl_ref_text_mismatch_count": int(pkl_ref_text_mismatch_count),
                "scores_pkl_pred_text_mismatch_count": int(pkl_pred_text_mismatch_count),
                "scores_pkl_shape_mismatch_count": int(pkl_shape_mismatch_count),
                "scores_pkl_other_failure_count": int(pkl_other_failure_count),
            }
        )

    return docs


__all__ = [
    "ensure_backend_available",
    "load_documents",
]
