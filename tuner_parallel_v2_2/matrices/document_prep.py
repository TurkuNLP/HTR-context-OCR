from __future__ import annotations

"""Document selection and preparation for Hough tuning.

The expensive document-level artifacts are built once per selected document and
reused for every Hough parameter combination:

- reference-to-prediction score matrix
- reference-to-reference score matrix
- whole-document normalized Levenshtein
- stride text blocks for line-level Levenshtein
- one precomputed Hough context per matrix direction

The streaming iterator is the RAM-safe path used by the main tuner: only active
documents hold matrices in memory.
"""

from dataclasses import dataclass, field
import json
import re
from itertools import chain
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

try:
    from ..runtime.runtime_paths import ensure_tuner_runtime_paths
except ImportError:
    from runtime.runtime_paths import ensure_tuner_runtime_paths  # type: ignore

# Document preparation imports shared score-matrix helpers from the project root,
# so bootstrap paths before importing those helper modules.
ensure_tuner_runtime_paths()

try:
    from ..alignment.along_lines_fast import build_stride_blocks
    from ..hough_preprocessing import HoughPreprocessingConfig, build_region_of_interest_hough_context
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
    from .runfile_selection import select_run_items_for_tuning
    from ..tuner.tuner_config import LogFn, SweepDocument
except ImportError:
    from alignment.along_lines_fast import build_stride_blocks  # type: ignore
    from hough_preprocessing import HoughPreprocessingConfig, build_region_of_interest_hough_context  # type: ignore
    from matrices.matrix_sources import (  # type: ignore
        categorize_pkl_load_failure,
        load_matrix_from_cache,
        matrix_cache_key,
        matrix_cache_path,
        save_matrix_to_cache,
        should_enable_scores_pkl_source,
    )
    from matrices.pkl_index_readonly import (  # type: ignore
        load_matrix_from_scores_pkl_index_readonly,
        load_score_stream_index_readonly,
    )
    from matrices.runfile_selection import select_run_items_for_tuning  # type: ignore
    from tuner.tuner_config import LogFn, SweepDocument  # type: ignore

try:
    from ..metrics.levenshtein_compat import BACKEND_C, SUPPORTED_BACKENDS, normalized_levenshtein_similarity
except ImportError:
    from metrics.levenshtein_compat import BACKEND_C, SUPPORTED_BACKENDS, normalized_levenshtein_similarity  # type: ignore

from score_matrix_builder import coerce_score_matrix, compute_score_matrix


@dataclass
class PklSourceState:
    """Read-only score-stream state for one matrix direction."""

    label: str
    enabled: bool = False
    disabled_reason: str | None = None
    scores_pkl_path: Path | None = None
    index_by_fname: dict[str, dict] | None = None
    index_source: str = "disabled"
    index_entry_count: int = 0
    index_cache_file_used: str | None = None
    index_prepare_seconds: float = 0.0
    index_load_failures: int = 0


@dataclass
class MatrixLoadTelemetry:
    """Counters and timings for one matrix direction."""

    label: str
    matrix_total_seconds: float = 0.0
    matrix_compute_seconds: float = 0.0
    matrix_cache_load_seconds: float = 0.0
    matrix_cache_store_seconds: float = 0.0
    matrix_pkl_load_seconds: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    cache_stores: int = 0
    cache_read_errors: int = 0
    cache_write_errors: int = 0
    matrix_source_npz_hits: int = 0
    matrix_source_pkl_hits: int = 0
    matrix_source_computed: int = 0
    pkl_lookup_misses: int = 0
    pkl_ref_text_mismatch_count: int = 0
    pkl_pred_text_mismatch_count: int = 0
    pkl_shape_mismatch_count: int = 0
    pkl_other_failure_count: int = 0
    source_counts: dict[str, int] = field(default_factory=dict)

    def note_source(self, source_name: str) -> None:
        """Increment the count for the matrix source used by one document."""
        self.source_counts[source_name] = int(self.source_counts.get(source_name, 0)) + 1

    def as_dict(self) -> dict:
        """Return JSON-friendly telemetry for summary output."""
        return {
            "label": self.label,
            "matrix_total_seconds": float(self.matrix_total_seconds),
            "matrix_compute_seconds": float(self.matrix_compute_seconds),
            "matrix_cache_load_seconds": float(self.matrix_cache_load_seconds),
            "matrix_cache_store_seconds": float(self.matrix_cache_store_seconds),
            "matrix_pkl_load_seconds": float(self.matrix_pkl_load_seconds),
            "cache_hits": int(self.cache_hits),
            "cache_misses": int(self.cache_misses),
            "cache_stores": int(self.cache_stores),
            "cache_read_errors": int(self.cache_read_errors),
            "cache_write_errors": int(self.cache_write_errors),
            "matrix_source_npz_hits": int(self.matrix_source_npz_hits),
            "matrix_source_pkl_hits": int(self.matrix_source_pkl_hits),
            "matrix_source_computed": int(self.matrix_source_computed),
            "pkl_lookup_misses": int(self.pkl_lookup_misses),
            "pkl_ref_text_mismatch_count": int(self.pkl_ref_text_mismatch_count),
            "pkl_pred_text_mismatch_count": int(self.pkl_pred_text_mismatch_count),
            "pkl_shape_mismatch_count": int(self.pkl_shape_mismatch_count),
            "pkl_other_failure_count": int(self.pkl_other_failure_count),
            "source_counts": dict(self.source_counts),
        }


def _no_log(_: str) -> None:
    """Default no-op logger used when callers do not provide a log hook."""
    return


def ensure_backend_available(backend: str) -> None:
    """Validate backend and ensure rapidfuzz C path is usable when requested."""
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend!r}. Allowed: {SUPPORTED_BACKENDS!r}")
    if backend == BACKEND_C:
        try:
            _ = normalized_levenshtein_similarity("a", "b", backend=BACKEND_C)
        except Exception as exc:
            raise RuntimeError(
                "Levenshtein backend 'c' was requested but is not available. "
                "Load the pytorch module or install rapidfuzz in the runtime environment."
            ) from exc


def _build_doc_blocks(pred: str, ref: str, matrix: np.ndarray, window_stride: int) -> tuple[list[str], list[str]]:
    """Precompute stride blocks once per document for ref-to-pred scoring."""
    if matrix.ndim != 2:
        return [], []
    n_ref, n_pred = int(matrix.shape[0]), int(matrix.shape[1])
    pred_blocks = build_stride_blocks(pred, n_blocks=n_pred, stride=int(window_stride))
    ref_blocks = build_stride_blocks(ref, n_blocks=n_ref, stride=int(window_stride))
    return pred_blocks, ref_blocks


def _prediction_text_is_empty(prediction_text: str) -> bool:
    """Return True when a selected runfile item has no usable prediction text."""
    return len(str(prediction_text).strip()) == 0


def _finite_matrix_maximum(matrix: np.ndarray) -> float:
    """Return the finite maximum cell value, or NaN when no finite cell exists."""
    if matrix.size == 0:
        return float("nan")
    finite_mask = np.isfinite(matrix)
    if not bool(finite_mask.any()):
        return float("nan")
    return float(np.nanmax(matrix))


def _safe_document_path_component(value: str, *, max_length: int = 120) -> str:
    """Return a readable path component for a document diagnostic directory."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not cleaned:
        cleaned = "document"
    return cleaned[: int(max_length)]


def _matrix_shape_for_metadata(matrix: np.ndarray | None) -> list[int]:
    """Return a JSON-friendly matrix shape, or an empty list when absent."""
    if matrix is None:
        return []
    return [int(value) for value in np.asarray(matrix).shape]


def _context_summary_without_arrays(hough_context: dict | None) -> dict:
    """Return preprocessing summary values without storing large arrays in JSON."""
    if not isinstance(hough_context, dict):
        return {}
    summary = hough_context.get("hough_preprocessing_summary", {})
    return dict(summary) if isinstance(summary, dict) else {}


def _mask_array_from_context(hough_context: dict | None, key: str, shape: tuple[int, int]) -> np.ndarray:
    """Return one boolean mask from a Hough context, or an empty mask with matrix shape."""
    mask = hough_context.get(key) if isinstance(hough_context, dict) else None
    if mask is None:
        return np.zeros(shape, dtype=bool)
    mask_array = np.asarray(mask, dtype=bool)
    if tuple(mask_array.shape) != tuple(shape):
        return np.zeros(shape, dtype=bool)
    return mask_array


def _save_diagnostic_matrix(document_dir: Path, filename: str, matrix: np.ndarray | None) -> str | None:
    """Save one matrix for skipped-document diagnostics when it exists."""
    if matrix is None:
        return None
    np.save(Path(document_dir) / filename, np.asarray(matrix))
    return filename


def _save_diagnostic_hough_masks(
    *,
    document_dir: Path,
    direction_name: str,
    matrix: np.ndarray | None,
    hough_context: dict | None,
) -> dict[str, str]:
    """Save binary preprocessing masks for one skipped matrix direction."""
    if matrix is None or np.asarray(matrix).ndim != 2:
        return {}

    matrix_shape = tuple(int(value) for value in np.asarray(matrix).shape)
    saved_masks: dict[str, str] = {}
    mask_specs = [
        ("hough_input_mask", "hough_mask_bool"),
        ("region_of_interest_mask", "region_of_interest_mask_bool"),
        ("strong_match_mask", "strong_match_mask_bool"),
    ]
    for filename_part, context_key in mask_specs:
        mask_filename = f"{direction_name}_{filename_part}.npy"
        np.save(
            Path(document_dir) / mask_filename,
            _mask_array_from_context(hough_context, context_key, matrix_shape),
        )
        saved_masks[f"{direction_name}_{filename_part}"] = mask_filename
    return saved_masks


def _write_skipped_document_diagnostic_bundle(
    *,
    skip_diagnostic_bundle_dir: Path | None,
    item: dict,
    fname: str,
    pred: str,
    ref: str,
    skip_record: dict,
    ref_to_pred_matrix: np.ndarray | None = None,
    ref_to_ref_matrix: np.ndarray | None = None,
    ref_to_pred_hough_ctx: dict | None = None,
    ref_to_ref_hough_ctx: dict | None = None,
) -> Path | None:
    """Write score matrices and preprocessing masks for a skipped document."""
    if skip_diagnostic_bundle_dir is None:
        return None

    document_index = int(item.get("index", -1))
    document_dir = Path(skip_diagnostic_bundle_dir) / (
        f"document_{document_index:06d}_{_safe_document_path_component(Path(fname).name)}"
    )
    document_dir.mkdir(parents=True, exist_ok=True)

    score_matrices: dict[str, str] = {}
    ref_to_pred_matrix_name = _save_diagnostic_matrix(
        document_dir,
        "ref_to_pred_score_matrix.npy",
        ref_to_pred_matrix,
    )
    if ref_to_pred_matrix_name is not None:
        score_matrices["ref_to_pred"] = ref_to_pred_matrix_name
    ref_to_ref_matrix_name = _save_diagnostic_matrix(
        document_dir,
        "ref_to_ref_score_matrix.npy",
        ref_to_ref_matrix,
    )
    if ref_to_ref_matrix_name is not None:
        score_matrices["ref_to_ref"] = ref_to_ref_matrix_name

    hough_preprocessing = {
        "ref_to_pred_summary": _context_summary_without_arrays(ref_to_pred_hough_ctx),
        "ref_to_ref_summary": _context_summary_without_arrays(ref_to_ref_hough_ctx),
    }
    hough_preprocessing.update(
        _save_diagnostic_hough_masks(
            document_dir=document_dir,
            direction_name="ref_to_pred",
            matrix=ref_to_pred_matrix,
            hough_context=ref_to_pred_hough_ctx,
        )
    )
    hough_preprocessing.update(
        _save_diagnostic_hough_masks(
            document_dir=document_dir,
            direction_name="ref_to_ref",
            matrix=ref_to_ref_matrix,
            hough_context=ref_to_ref_hough_ctx,
        )
    )

    metadata = {
        "schema_version": "tuner_skipped_document_diagnostic_v1",
        "record_format": "skipped_document_diagnostic",
        "document": {
            "index": document_index,
            "fname": Path(fname).name,
            "ref_text_len": int(len(ref)),
            "pred_text_len": int(len(pred)),
            "window_size": int(item.get("window_size", 0) or 0),
            "window_stride": int(item.get("window_stride", 0) or 0),
            "ref_to_pred_matrix_shape": _matrix_shape_for_metadata(ref_to_pred_matrix),
            "ref_to_ref_matrix_shape": _matrix_shape_for_metadata(ref_to_ref_matrix),
            "whole_document_nls": None,
        },
        "skip_record": dict(skip_record),
        "score_matrices": score_matrices,
        "hough_preprocessing": hough_preprocessing,
        "threshold_record_pattern": None,
    }
    (document_dir / "document_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return document_dir


def _attach_skipped_document_diagnostic_bundle(
    *,
    skip_record: dict,
    skip_diagnostic_bundle_dir: Path | None,
    item: dict,
    fname: str,
    pred: str,
    ref: str,
    ref_to_pred_matrix: np.ndarray | None = None,
    ref_to_ref_matrix: np.ndarray | None = None,
    ref_to_pred_hough_ctx: dict | None = None,
    ref_to_ref_hough_ctx: dict | None = None,
) -> dict:
    """Add a diagnostic bundle path to a skipped-record dictionary when possible."""
    diagnostic_bundle_path = _write_skipped_document_diagnostic_bundle(
        skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
        item=item,
        fname=fname,
        pred=pred,
        ref=ref,
        skip_record=skip_record,
        ref_to_pred_matrix=ref_to_pred_matrix,
        ref_to_ref_matrix=ref_to_ref_matrix,
        ref_to_pred_hough_ctx=ref_to_pred_hough_ctx,
        ref_to_ref_hough_ctx=ref_to_ref_hough_ctx,
    )
    if diagnostic_bundle_path is not None:
        skip_record = dict(skip_record)
        skip_record["diagnostic_bundle_dir"] = str(diagnostic_bundle_path)
    return skip_record


def _empty_prediction_skip_record(*, item: dict, fname: str, pred: str, ref: str) -> dict:
    """Build one stable CSV/JSON record for a document skipped before tuning."""
    return {
        "index": int(item.get("index", -1)),
        "fname": Path(fname).name,
        "skip_reason": "no_prediction_text",
        "skip_stage": "document_preparation",
        "prediction_character_count": int(len(pred)),
        "prediction_non_whitespace_character_count": int(len(pred.strip())),
        "reference_character_count": int(len(ref)),
        "reference_non_whitespace_character_count": int(len(ref.strip())),
        "ref_to_pred_matrix_rows": None,
        "ref_to_pred_matrix_cols": None,
        "ref_to_pred_source": None,
        "ref_to_pred_matrix_max": None,
        "message": "Skipped before matrix loading because the selected runfile prediction text is empty.",
    }


def _no_prediction_windows_skip_record(
    *,
    item: dict,
    fname: str,
    pred: str,
    ref: str,
    ref_to_pred_matrix: np.ndarray,
    ref_to_pred_source: str,
) -> dict:
    """Build a skip record for a score matrix with no prediction columns."""
    return {
        "index": int(item.get("index", -1)),
        "fname": Path(fname).name,
        "skip_reason": "no_ref_to_pred_prediction_windows",
        "skip_stage": "document_preparation",
        "prediction_character_count": int(len(pred)),
        "prediction_non_whitespace_character_count": int(len(pred.strip())),
        "reference_character_count": int(len(ref)),
        "reference_non_whitespace_character_count": int(len(ref.strip())),
        "ref_to_pred_matrix_rows": int(ref_to_pred_matrix.shape[0]) if ref_to_pred_matrix.ndim >= 1 else 0,
        "ref_to_pred_matrix_cols": int(ref_to_pred_matrix.shape[1]) if ref_to_pred_matrix.ndim >= 2 else 0,
        "ref_to_pred_source": str(ref_to_pred_source),
        "ref_to_pred_matrix_max": _finite_matrix_maximum(ref_to_pred_matrix),
        "message": (
            "Skipped after loading the ref_to_pred matrix because it has zero prediction columns, "
            "so no prediction-side Hough lines can be evaluated."
        ),
    }


def _hough_preprocessing_skip_record(
    *,
    item: dict,
    fname: str,
    pred: str,
    ref: str,
    matrix: np.ndarray,
    matrix_source: str,
    matrix_direction: str,
    hough_context: dict,
) -> dict:
    """Build a skip record when preprocessing leaves no usable Hough input."""
    preprocessing_summary = hough_context.get("hough_preprocessing_summary", {})
    if not isinstance(preprocessing_summary, dict):
        preprocessing_summary = {}
    matrix_maximum = _finite_matrix_maximum(matrix)
    rejection_reason = str(hough_context.get("hough_preprocessing_rejection_reason", "unknown"))
    return {
        "index": int(item.get("index", -1)),
        "fname": Path(fname).name,
        "skip_reason": f"{matrix_direction}_hough_preprocessing_rejected",
        "skip_stage": "hough_preprocessing",
        "prediction_character_count": int(len(pred)),
        "prediction_non_whitespace_character_count": int(len(pred.strip())),
        "reference_character_count": int(len(ref)),
        "reference_non_whitespace_character_count": int(len(ref.strip())),
        "ref_to_pred_matrix_rows": int(matrix.shape[0]) if matrix.ndim >= 1 else 0,
        "ref_to_pred_matrix_cols": int(matrix.shape[1]) if matrix.ndim >= 2 else 0,
        "ref_to_pred_source": str(matrix_source),
        "ref_to_pred_matrix_max": matrix_maximum,
        "preprocessing_matrix_direction": str(matrix_direction),
        "preprocessing_rejection_reason": rejection_reason,
        "preprocessing_score_floor": preprocessing_summary.get("score_floor"),
        "preprocessing_active_cells": preprocessing_summary.get("active_cell_count"),
        "preprocessing_active_fraction": preprocessing_summary.get("active_fraction"),
        "preprocessing_seconds": preprocessing_summary.get("preprocessing_seconds"),
        "message": (
            "Skipped after Hough preprocessing because the score matrix did not produce "
            "a usable binary Region of Interest input."
        ),
    }


def _record_skipped_document(
    *,
    skip_record: dict,
    skipped_documents: list[dict],
    skipped_reason_counts: dict[str, int],
    log_fn: LogFn,
    on_document_skipped: Callable[[dict], None] | None,
) -> None:
    """Store, log, and optionally report one skipped document."""
    skipped_documents.append(skip_record)
    skip_reason = str(skip_record["skip_reason"])
    skipped_reason_counts[skip_reason] = int(skipped_reason_counts.get(skip_reason, 0)) + 1
    matrix_cols = skip_record.get("ref_to_pred_matrix_cols")
    matrix_cols_text = "" if matrix_cols is None else f" ref_to_pred_cols={int(matrix_cols)}"
    matrix_maximum = skip_record.get("ref_to_pred_matrix_max")
    matrix_maximum_text = "" if matrix_maximum is None else f" ref_to_pred_max={float(matrix_maximum):.6f}"
    log_fn(
        f"[skip-document] fname={skip_record['fname']} index={int(skip_record['index'])} "
        f"reason={skip_reason} pred_chars={int(skip_record['prediction_character_count'])} "
        f"ref_chars={int(skip_record['reference_character_count'])}{matrix_cols_text}{matrix_maximum_text}"
    )
    if on_document_skipped is not None:
        on_document_skipped(dict(skip_record))


def _prepare_readonly_pkl_source(
    *,
    label: str,
    scores_pkl_path: Path | None,
    score_index_cache_file: Path | None,
    score_index_cache_dir: Path | None,
    window_size: int,
    window_stride: int,
    disable_pkl_matrix_source: bool,
    log_fn: LogFn,
) -> PklSourceState:
    """Prepare one read-only pkl source and its index once per run."""
    state = PklSourceState(label=str(label), scores_pkl_path=None if scores_pkl_path is None else Path(scores_pkl_path))
    pkl_enabled, disabled_reason = should_enable_scores_pkl_source(
        scores_pkl_ref_to_pred=scores_pkl_path,
        window_size=int(window_size),
        window_stride=int(window_stride),
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        log_fn=log_fn,
    )
    state.enabled = bool(pkl_enabled)
    state.disabled_reason = disabled_reason

    if not pkl_enabled or scores_pkl_path is None:
        log_fn(f"[pkl-index:{label}] disabled reason={disabled_reason}")
        return state

    started_at = time.perf_counter()
    try:
        index_result = load_score_stream_index_readonly(
            scores_pkl=Path(scores_pkl_path),
            explicit_cache_file=None if score_index_cache_file is None else Path(score_index_cache_file),
            cache_dir=None if score_index_cache_dir is None else Path(score_index_cache_dir),
            log_fn=log_fn,
        )
        state.index_prepare_seconds = float(time.perf_counter() - started_at)
        state.index_by_fname = index_result.index_by_fname
        state.index_source = str(index_result.source)
        state.index_entry_count = int(len(index_result.index_by_fname))
        state.index_cache_file_used = None if index_result.cache_file is None else str(index_result.cache_file)
        log_fn(
            f"[pkl-index:{label}] enabled source={state.index_source} entries={state.index_entry_count} "
            f"cache_file={state.index_cache_file_used}"
        )
    except Exception as exc:
        state.index_prepare_seconds = float(time.perf_counter() - started_at)
        state.index_load_failures += 1
        state.enabled = False
        state.disabled_reason = f"index_load_error:{exc!r}"
        log_fn(f"[pkl-index:{label}] disabled due to index load error: {exc!r}")

    return state


def _record_pkl_failure(*, telemetry: MatrixLoadTelemetry, reason: str | None) -> None:
    """Classify one pkl matrix-load miss into stable telemetry counters."""
    category = categorize_pkl_load_failure(reason)
    if category == "index_miss":
        telemetry.pkl_lookup_misses += 1
    elif category == "ref_text_mismatch":
        telemetry.pkl_ref_text_mismatch_count += 1
    elif category == "pred_text_mismatch":
        telemetry.pkl_pred_text_mismatch_count += 1
    elif category == "shape_mismatch":
        telemetry.pkl_shape_mismatch_count += 1
    else:
        telemetry.pkl_other_failure_count += 1


def _load_or_compute_score_matrix(
    *,
    label: str,
    fname: str,
    ref_text: str,
    other_text: str,
    window_size: int,
    window_stride: int,
    cache_dir: Path | None,
    pkl_source: PklSourceState,
    telemetry: MatrixLoadTelemetry,
    log_fn: LogFn,
) -> tuple[np.ndarray, str]:
    """Load one matrix through cache, read-only pkl, then clean computation."""
    matrix_started_at = time.perf_counter()
    matrix: np.ndarray | None = None
    matrix_source = "unknown"

    if cache_dir is not None:
        cache_key = matrix_cache_key(
            ref_text=ref_text,
            pred_text=other_text,
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        cache_path = matrix_cache_path(cache_dir=cache_dir, cache_key=cache_key)
        if cache_path.exists():
            try:
                t0 = time.perf_counter()
                matrix = load_matrix_from_cache(cache_path=cache_path)
                telemetry.matrix_cache_load_seconds += time.perf_counter() - t0
                telemetry.cache_hits += 1
                telemetry.matrix_source_npz_hits += 1
                matrix_source = "npz_hit"
            except Exception as exc:
                telemetry.cache_read_errors += 1
                telemetry.cache_misses += 1
                log_fn(f"[cache:{label}] read_error fname={Path(fname).name} path={cache_path} err={exc!r}")
        else:
            telemetry.cache_misses += 1

    if matrix is None and pkl_source.enabled and pkl_source.scores_pkl_path is not None and pkl_source.index_by_fname is not None:
        t0 = time.perf_counter()
        pkl_result = load_matrix_from_scores_pkl_index_readonly(
            scores_pkl=Path(pkl_source.scores_pkl_path),
            score_index_by_fname=pkl_source.index_by_fname,
            fname=Path(fname).name,
            expected_ref_text=ref_text,
            expected_pred_text=other_text,
            window_size=int(window_size),
            window_stride=int(window_stride),
        )
        telemetry.matrix_pkl_load_seconds += time.perf_counter() - t0

        if pkl_result.matrix is not None:
            matrix = pkl_result.matrix
            matrix_source = "scores_pkl"
            telemetry.matrix_source_pkl_hits += 1

            if cache_dir is not None:
                cache_key = matrix_cache_key(
                    ref_text=ref_text,
                    pred_text=other_text,
                    window_size=int(window_size),
                    window_stride=int(window_stride),
                )
                cache_path = matrix_cache_path(cache_dir=cache_dir, cache_key=cache_key)
                if not cache_path.exists():
                    try:
                        t_store = time.perf_counter()
                        save_matrix_to_cache(cache_path=cache_path, matrix=matrix)
                        telemetry.matrix_cache_store_seconds += time.perf_counter() - t_store
                        telemetry.cache_stores += 1
                    except Exception as exc:
                        telemetry.cache_write_errors += 1
                        log_fn(f"[cache:{label}] write_error fname={Path(fname).name} path={cache_path} err={exc!r}")
        else:
            _record_pkl_failure(telemetry=telemetry, reason=pkl_result.reason)

    if matrix is None:
        t0 = time.perf_counter()
        matrix = compute_score_matrix(ref_text, other_text, window_size=int(window_size), window_stride=int(window_stride))
        telemetry.matrix_compute_seconds += time.perf_counter() - t0
        matrix_source = "computed"
        telemetry.matrix_source_computed += 1

        if cache_dir is not None:
            cache_key = matrix_cache_key(
                ref_text=ref_text,
                pred_text=other_text,
                window_size=int(window_size),
                window_stride=int(window_stride),
            )
            cache_path = matrix_cache_path(cache_dir=cache_dir, cache_key=cache_key)
            try:
                t_store = time.perf_counter()
                save_matrix_to_cache(cache_path=cache_path, matrix=matrix)
                telemetry.matrix_cache_store_seconds += time.perf_counter() - t_store
                telemetry.cache_stores += 1
            except Exception as exc:
                telemetry.cache_write_errors += 1
                log_fn(f"[cache:{label}] write_error fname={Path(fname).name} path={cache_path} err={exc!r}")

    telemetry.matrix_total_seconds += time.perf_counter() - matrix_started_at
    telemetry.note_source(matrix_source)

    prepared_matrix = coerce_score_matrix(matrix, source_desc=f"prepared_document:{label}:{Path(fname).name}")
    return prepared_matrix, matrix_source


def iter_prepared_documents_from_items(
    *,
    selected_run_items: Iterable[dict],
    window_size: int,
    window_stride: int,
    levenshtein_backend: str,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    scores_pkl_ref_to_ref: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_file_ref_to_ref: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    prepare_ref_to_pred_artifacts: bool = True,
    raise_when_no_documents_selected: bool = True,
    timing_out: dict | None = None,
    on_document_skipped: Callable[[dict], None] | None = None,
    skip_diagnostic_bundle_dir: Path | None = None,
    hough_preprocessing_config: HoughPreprocessingConfig | None = None,
    log_fn: LogFn | None = None,
) -> Iterator[SweepDocument]:
    """Yield prepared documents using layered matrix sources and timings."""
    ensure_backend_available(levenshtein_backend)
    log = _no_log if log_fn is None else log_fn
    load_started_at = time.perf_counter()
    active_hough_preprocessing_config = (
        HoughPreprocessingConfig() if hough_preprocessing_config is None else hough_preprocessing_config
    )

    if int(window_size) <= 0 or int(window_stride) <= 0:
        raise ValueError("window_size and window_stride must be positive")
    if matrix_cache_dir is not None and Path(matrix_cache_dir).exists() and not Path(matrix_cache_dir).is_dir():
        raise NotADirectoryError(f"matrix_cache_dir is not a directory: {matrix_cache_dir}")
    selected_item_iterator = iter(selected_run_items)
    try:
        # Pull exactly one item as a lookahead.  This keeps dynamic-pool runs
        # lazy: one scheduler request claims one document, not the whole pool.
        first_selected_item = dict(next(selected_item_iterator))
    except StopIteration:
        if bool(raise_when_no_documents_selected):
            raise RuntimeError("No documents selected for parameter sweep.")
        return

    cache_dir = None if matrix_cache_dir is None else Path(matrix_cache_dir)
    ref_to_pred_telemetry = MatrixLoadTelemetry(label="ref_to_pred")
    ref_to_ref_telemetry = MatrixLoadTelemetry(label="ref_to_ref")

    if bool(prepare_ref_to_pred_artifacts):
        ref_to_pred_pkl_source = _prepare_readonly_pkl_source(
            label="ref_to_pred",
            scores_pkl_path=scores_pkl_ref_to_pred,
            score_index_cache_file=score_index_cache_file,
            score_index_cache_dir=score_index_cache_dir,
            window_size=int(window_size),
            window_stride=int(window_stride),
            disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
            log_fn=log,
        )
    else:
        # Warm-cache-only runs never inspect the prediction matrix.  Keeping the
        # pkl source disabled here avoids index loading, matrix loading, and
        # Hough-context preparation for artifacts that cannot affect ref_to_ref.
        ref_to_pred_pkl_source = PklSourceState(
            label="ref_to_pred",
            enabled=False,
            disabled_reason="skipped_for_ref_to_ref_cache_warmup",
            scores_pkl_path=None if scores_pkl_ref_to_pred is None else Path(scores_pkl_ref_to_pred),
        )
        log("[pkl-index:ref_to_pred] disabled reason=skipped_for_ref_to_ref_cache_warmup")
    ref_to_ref_pkl_source = _prepare_readonly_pkl_source(
        label="ref_to_ref",
        scores_pkl_path=scores_pkl_ref_to_ref,
        score_index_cache_file=score_index_cache_file_ref_to_ref,
        score_index_cache_dir=score_index_cache_dir,
        window_size=int(window_size),
        window_stride=int(window_stride),
        disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
        log_fn=log,
    )

    whole_document_nls_seconds = 0.0
    block_build_seconds = 0.0
    hough_context_ref_to_pred_seconds = 0.0
    hough_context_ref_to_ref_seconds = 0.0
    prepared_count = 0
    selected_count_seen = 0
    skipped_documents: list[dict] = []
    skipped_reason_counts: dict[str, int] = {}

    selected_items_to_prepare = chain(
        (first_selected_item,),
        # Convert each later item only when the scheduler asks for another
        # prepared document.  Dynamic-pool claims therefore stay slot-by-slot.
        (dict(item) for item in selected_item_iterator),
    )

    for item in selected_items_to_prepare:
        fname = str(item["fname"])
        pred = str(item["pred"])
        ref = str(item["ref"])
        selected_count_seen += 1

        if _prediction_text_is_empty(pred):
            skip_record = _empty_prediction_skip_record(item=item, fname=fname, pred=pred, ref=ref)
            skip_record = _attach_skipped_document_diagnostic_bundle(
                skip_record=skip_record,
                skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
                item=item,
                fname=fname,
                pred=pred,
                ref=ref,
            )
            _record_skipped_document(
                skip_record=skip_record,
                skipped_documents=skipped_documents,
                skipped_reason_counts=skipped_reason_counts,
                log_fn=log,
                on_document_skipped=on_document_skipped,
            )
            continue

        if bool(prepare_ref_to_pred_artifacts):
            ref_to_pred_matrix, ref_to_pred_source = _load_or_compute_score_matrix(
                label="ref_to_pred",
                fname=fname,
                ref_text=ref,
                other_text=pred,
                window_size=int(window_size),
                window_stride=int(window_stride),
                cache_dir=cache_dir,
                pkl_source=ref_to_pred_pkl_source,
                telemetry=ref_to_pred_telemetry,
                log_fn=log,
            )
            if ref_to_pred_matrix.ndim != 2 or int(ref_to_pred_matrix.shape[1]) <= 0:
                skip_record = _no_prediction_windows_skip_record(
                    item=item,
                    fname=fname,
                    pred=pred,
                    ref=ref,
                    ref_to_pred_matrix=ref_to_pred_matrix,
                    ref_to_pred_source=ref_to_pred_source,
                )
                skip_record = _attach_skipped_document_diagnostic_bundle(
                    skip_record=skip_record,
                    skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
                    item=item,
                    fname=fname,
                    pred=pred,
                    ref=ref,
                    ref_to_pred_matrix=ref_to_pred_matrix,
                )
                _record_skipped_document(
                    skip_record=skip_record,
                    skipped_documents=skipped_documents,
                    skipped_reason_counts=skipped_reason_counts,
                    log_fn=log,
                    on_document_skipped=on_document_skipped,
                )
                continue
        else:
            # The warm-up path only needs the reference-self matrix/context.
            # A zero-sized placeholder satisfies the shared SweepDocument shape
            # without allocating prediction-side data that will never be read.
            ref_to_pred_matrix = np.zeros((0, 0), dtype=float)
            ref_to_pred_source = "skipped_for_ref_to_ref_cache_warmup"
        ref_to_ref_matrix, ref_to_ref_source = _load_or_compute_score_matrix(
            label="ref_to_ref",
            fname=fname,
            ref_text=ref,
            other_text=ref,
            window_size=int(window_size),
            window_stride=int(window_stride),
            cache_dir=cache_dir,
            pkl_source=ref_to_ref_pkl_source,
            telemetry=ref_to_ref_telemetry,
            log_fn=log,
        )

        if bool(prepare_ref_to_pred_artifacts):
            t_nls = time.perf_counter()
            whole_nls = float(normalized_levenshtein_similarity(pred, ref, backend=levenshtein_backend))
            whole_document_nls_seconds += time.perf_counter() - t_nls
        else:
            whole_nls = 0.0

        if bool(prepare_ref_to_pred_artifacts):
            t_blocks = time.perf_counter()
            pred_blocks, ref_blocks = _build_doc_blocks(pred, ref, ref_to_pred_matrix, window_stride=int(window_stride))
            block_build_seconds += time.perf_counter() - t_blocks
        else:
            pred_blocks, ref_blocks = [], []

        if bool(prepare_ref_to_pred_artifacts):
            t_ctx = time.perf_counter()
            ref_to_pred_hough_ctx = build_region_of_interest_hough_context(
                ref_to_pred_matrix,
                config=active_hough_preprocessing_config,
                keep_debug_arrays=False,
            )
            hough_context_ref_to_pred_seconds += time.perf_counter() - t_ctx
            if not bool(ref_to_pred_hough_ctx.get("hough_preprocessing_accepted", False)):
                skip_record = _hough_preprocessing_skip_record(
                    item=item,
                    fname=fname,
                    pred=pred,
                    ref=ref,
                    matrix=ref_to_pred_matrix,
                    matrix_source=ref_to_pred_source,
                    matrix_direction="ref_to_pred",
                    hough_context=ref_to_pred_hough_ctx,
                )
                skip_record = _attach_skipped_document_diagnostic_bundle(
                    skip_record=skip_record,
                    skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
                    item=item,
                    fname=fname,
                    pred=pred,
                    ref=ref,
                    ref_to_pred_matrix=ref_to_pred_matrix,
                    ref_to_pred_hough_ctx=ref_to_pred_hough_ctx,
                )
                _record_skipped_document(
                    skip_record=skip_record,
                    skipped_documents=skipped_documents,
                    skipped_reason_counts=skipped_reason_counts,
                    log_fn=log,
                    on_document_skipped=on_document_skipped,
                )
                continue
        else:
            ref_to_pred_hough_ctx = {}

        t_ctx = time.perf_counter()
        ref_to_ref_hough_ctx = build_region_of_interest_hough_context(
            ref_to_ref_matrix,
            config=active_hough_preprocessing_config,
            keep_debug_arrays=False,
        )
        hough_context_ref_to_ref_seconds += time.perf_counter() - t_ctx
        if not bool(ref_to_ref_hough_ctx.get("hough_preprocessing_accepted", False)):
            skip_record = _hough_preprocessing_skip_record(
                item=item,
                fname=fname,
                pred=pred,
                ref=ref,
                matrix=ref_to_ref_matrix,
                matrix_source=ref_to_ref_source,
                matrix_direction="ref_to_ref",
                hough_context=ref_to_ref_hough_ctx,
            )
            skip_record = _attach_skipped_document_diagnostic_bundle(
                skip_record=skip_record,
                skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
                item=item,
                fname=fname,
                pred=pred,
                ref=ref,
                ref_to_pred_matrix=ref_to_pred_matrix,
                ref_to_ref_matrix=ref_to_ref_matrix,
                ref_to_pred_hough_ctx=ref_to_pred_hough_ctx,
                ref_to_ref_hough_ctx=ref_to_ref_hough_ctx,
            )
            _record_skipped_document(
                skip_record=skip_record,
                skipped_documents=skipped_documents,
                skipped_reason_counts=skipped_reason_counts,
                log_fn=log,
                on_document_skipped=on_document_skipped,
            )
            continue

        prepared_count += 1
        prepared_document = SweepDocument(
            index=int(item["index"]),
            fname=Path(fname).name,
            window_size=int(window_size),
            window_stride=int(window_stride),
            pred=pred,
            ref=ref,
            ref_to_pred_matrix=ref_to_pred_matrix,
            ref_to_ref_matrix=ref_to_ref_matrix,
            whole_document_nls=whole_nls,
            pred_blocks=pred_blocks,
            ref_blocks=ref_blocks,
            ref_to_pred_hough_ctx=ref_to_pred_hough_ctx,
            ref_to_ref_hough_ctx=ref_to_ref_hough_ctx,
        )
        log(
            f"[load] {prepared_count} fname={Path(fname).name} "
            f"ref_to_pred_shape={ref_to_pred_matrix.shape} ref_to_ref_shape={ref_to_ref_matrix.shape} "
            f"whole_nls={whole_nls:.6f} ref_to_pred_source={ref_to_pred_source} "
            f"ref_to_ref_source={ref_to_ref_source}"
        )
        yield prepared_document

    if cache_dir is not None:
        log(
            f"[cache:ref_to_pred] dir={cache_dir} hits={ref_to_pred_telemetry.cache_hits} "
            f"misses={ref_to_pred_telemetry.cache_misses} stores={ref_to_pred_telemetry.cache_stores} "
            f"read_errors={ref_to_pred_telemetry.cache_read_errors} write_errors={ref_to_pred_telemetry.cache_write_errors}"
        )
        log(
            f"[cache:ref_to_ref] dir={cache_dir} hits={ref_to_ref_telemetry.cache_hits} "
            f"misses={ref_to_ref_telemetry.cache_misses} stores={ref_to_ref_telemetry.cache_stores} "
            f"read_errors={ref_to_ref_telemetry.cache_read_errors} write_errors={ref_to_ref_telemetry.cache_write_errors}"
        )

    load_total_seconds = time.perf_counter() - load_started_at
    log(
        f"[timing] load_total_s={load_total_seconds:.3f} "
        f"ref_to_pred_matrix_total_s={ref_to_pred_telemetry.matrix_total_seconds:.3f} "
        f"ref_to_ref_matrix_total_s={ref_to_ref_telemetry.matrix_total_seconds:.3f} "
        f"whole_doc_nls_s={whole_document_nls_seconds:.3f} precompute_blocks_s={block_build_seconds:.3f} "
        f"hough_ctx_ref_to_pred_s={hough_context_ref_to_pred_seconds:.3f} "
        f"hough_ctx_ref_to_ref_s={hough_context_ref_to_ref_seconds:.3f}"
    )
    if skipped_documents:
        log(
            f"[skip-summary] skipped_documents={len(skipped_documents)} "
            f"prepared_documents={prepared_count} reasons={dict(skipped_reason_counts)}"
        )

    if timing_out is not None:
        timing_out.clear()
        timing_out.update(
            {
                "load_documents_total_seconds": float(load_total_seconds),
                "selected_document_records_seen": int(selected_count_seen),
                "prepared_document_count": int(prepared_count),
                "skipped_document_count": int(len(skipped_documents)),
                "skipped_document_reason_counts": dict(skipped_reason_counts),
                "skipped_documents": list(skipped_documents),
                "hough_preprocessing_config": active_hough_preprocessing_config.as_dict(),
                "whole_document_nls_seconds": float(whole_document_nls_seconds),
                "precompute_blocks_seconds": float(block_build_seconds),
                "hough_context_ref_to_pred_seconds": float(hough_context_ref_to_pred_seconds),
                "hough_context_ref_to_ref_seconds": float(hough_context_ref_to_ref_seconds),
                "ref_to_pred_matrix": ref_to_pred_telemetry.as_dict(),
                "ref_to_ref_matrix": ref_to_ref_telemetry.as_dict(),
                "scores_pkl_ref_to_pred": {
                    "enabled": bool(ref_to_pred_pkl_source.enabled),
                    "disabled_reason": ref_to_pred_pkl_source.disabled_reason,
                    "path": None if ref_to_pred_pkl_source.scores_pkl_path is None else str(ref_to_pred_pkl_source.scores_pkl_path),
                    "index_source": str(ref_to_pred_pkl_source.index_source),
                    "index_entry_count": int(ref_to_pred_pkl_source.index_entry_count),
                    "index_cache_file_used": ref_to_pred_pkl_source.index_cache_file_used,
                    "index_prepare_seconds": float(ref_to_pred_pkl_source.index_prepare_seconds),
                    "index_load_failures": int(ref_to_pred_pkl_source.index_load_failures),
                },
                "scores_pkl_ref_to_ref": {
                    "enabled": bool(ref_to_ref_pkl_source.enabled),
                    "disabled_reason": ref_to_ref_pkl_source.disabled_reason,
                    "path": None if ref_to_ref_pkl_source.scores_pkl_path is None else str(ref_to_ref_pkl_source.scores_pkl_path),
                    "index_source": str(ref_to_ref_pkl_source.index_source),
                    "index_entry_count": int(ref_to_ref_pkl_source.index_entry_count),
                    "index_cache_file_used": ref_to_ref_pkl_source.index_cache_file_used,
                    "index_prepare_seconds": float(ref_to_ref_pkl_source.index_prepare_seconds),
                    "index_load_failures": int(ref_to_ref_pkl_source.index_load_failures),
                },
            }
        )

    return


def load_documents(
    *,
    runfile_json: Path,
    window_size: int,
    window_stride: int,
    levenshtein_backend: str,
    matrix_cache_dir: Path | None = None,
    scores_pkl_ref_to_pred: Path | None = None,
    scores_pkl_ref_to_ref: Path | None = None,
    score_index_cache_file: Path | None = None,
    score_index_cache_file_ref_to_ref: Path | None = None,
    score_index_cache_dir: Path | None = None,
    disable_pkl_matrix_source: bool = False,
    prepare_ref_to_pred_artifacts: bool = True,
    target_fnames: Iterable[str] | None = None,
    max_items: int | None = None,
    selection_index_range: tuple[int, int] | None = None,
    timing_out: dict | None = None,
    on_document_skipped: Callable[[dict], None] | None = None,
    skip_diagnostic_bundle_dir: Path | None = None,
    hough_preprocessing_config: HoughPreprocessingConfig | None = None,
    log_fn: LogFn | None = None,
) -> list[SweepDocument]:
    """Compatibility loader returning all prepared documents as a list."""
    selected_items = select_run_items_for_tuning(
        runfile_json=Path(runfile_json),
        target_fnames=target_fnames,
        max_items=max_items,
        selection_index_range=selection_index_range,
    )
    return list(
        iter_prepared_documents_from_items(
            selected_run_items=selected_items,
            window_size=int(window_size),
            window_stride=int(window_stride),
            levenshtein_backend=str(levenshtein_backend),
            matrix_cache_dir=matrix_cache_dir,
            scores_pkl_ref_to_pred=scores_pkl_ref_to_pred,
            scores_pkl_ref_to_ref=scores_pkl_ref_to_ref,
            score_index_cache_file=score_index_cache_file,
            score_index_cache_file_ref_to_ref=score_index_cache_file_ref_to_ref,
            score_index_cache_dir=score_index_cache_dir,
            disable_pkl_matrix_source=bool(disable_pkl_matrix_source),
            prepare_ref_to_pred_artifacts=bool(prepare_ref_to_pred_artifacts),
            timing_out=timing_out,
            on_document_skipped=on_document_skipped,
            skip_diagnostic_bundle_dir=skip_diagnostic_bundle_dir,
            hough_preprocessing_config=hough_preprocessing_config,
            log_fn=log_fn,
        )
    )


__all__ = [
    "ensure_backend_available",
    "iter_prepared_documents_from_items",
    "load_documents",
    "select_run_items_for_tuning",
]
