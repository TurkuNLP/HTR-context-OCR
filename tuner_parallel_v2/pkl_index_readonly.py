from __future__ import annotations

"""Read-only helpers for reusing precomputed score-stream indexes and matrices.

This module is intentionally standalone and *never writes* to index cache paths.
It supports three safe operations:
1) Resolve an existing index cache file path for a scores.pkl stream.
2) Load and validate index payload from cache if present.
3) Fall back to building an in-memory index (no persistence) when cache is absent/invalid.
"""

from dataclasses import dataclass
import hashlib
import pickle
import re
from pathlib import Path
from typing import Callable

import numpy as np

from runfile_records import safe_name
from score_matrix_builder import coerce_score_matrix
from score_stream_index import build_score_stream_index, load_score_item_by_offset

LogFn = Callable[[str], None]


def _no_log(_: str) -> None:
    return


@dataclass(frozen=True)
class ScoreIndexLoadResult:
    """Container describing how a score-stream index was acquired."""

    index_by_fname: dict[str, dict]
    source: str
    cache_file: Path | None


def build_expected_index_cache_file(scores_pkl: Path, cache_dir: Path) -> Path:
    """Build expected index-cache filename matching score_stream_index cache naming.

    The naming is intentionally identical to text_metrics_v2_1_parallel so existing
    cache files can be reused directly in read-only mode.
    """
    resolved = str(scores_pkl.resolve())
    path_hash = hashlib.sha1(resolved.encode("utf-8")).hexdigest()[:16]
    return Path(cache_dir) / f"{safe_name(scores_pkl.name)}.{path_hash}.index.pkl"


def parse_ws_st_from_scores_pkl_name(scores_pkl: Path) -> tuple[int | None, int | None]:
    """Parse ws/st parameters from score pkl filename when present.

    Example supported pattern: scores_reference_prediction_ws50_st35.pkl
    """
    m = re.search(r"_ws(\d+)_st(\d+)(?:\.|$)", str(scores_pkl.name))
    if m is None:
        return None, None
    return int(m.group(1)), int(m.group(2))


def resolve_existing_index_cache_file(
    *,
    scores_pkl: Path,
    explicit_cache_file: Path | None,
    cache_dir: Path | None,
) -> Path | None:
    """Resolve existing cache file path without creating or mutating anything."""
    if explicit_cache_file is not None:
        explicit = Path(explicit_cache_file)
        return explicit if explicit.exists() else None

    if cache_dir is None:
        return None

    candidate = build_expected_index_cache_file(Path(scores_pkl), Path(cache_dir))
    return candidate if candidate.exists() else None


def _validate_cached_index_payload(*, payload: object, scores_pkl: Path) -> dict[str, dict] | None:
    """Validate cached index payload against the target score stream metadata.

    Returns the index dict if valid, otherwise None.
    """
    if not isinstance(payload, dict):
        return None

    index_obj = payload.get("index", {})
    if not isinstance(index_obj, dict):
        return None

    try:
        stat = Path(scores_pkl).stat()
    except OSError:
        return None

    expected_path = str(Path(scores_pkl).resolve())
    cache_path = str(payload.get("scores_pkl_path", ""))
    cache_size = int(payload.get("scores_pkl_size", -1))
    cache_mtime_ns = int(payload.get("scores_pkl_mtime_ns", -1))

    if cache_path != expected_path:
        return None
    if cache_size != int(stat.st_size):
        return None
    if cache_mtime_ns != int(stat.st_mtime_ns):
        return None

    return index_obj


def load_score_stream_index_readonly(
    *,
    scores_pkl: Path,
    explicit_cache_file: Path | None = None,
    cache_dir: Path | None = None,
    log_fn: LogFn | None = None,
) -> ScoreIndexLoadResult:
    """Load score-stream index in read-only mode.

    Resolution order:
    1) explicit cache file (if provided and exists)
    2) derived cache file in cache_dir (if provided and exists)
    3) in-memory index build from scores_pkl stream (no file writes)
    """
    log = _no_log if log_fn is None else log_fn
    scores_pkl = Path(scores_pkl)

    cache_file = resolve_existing_index_cache_file(
        scores_pkl=scores_pkl,
        explicit_cache_file=explicit_cache_file,
        cache_dir=cache_dir,
    )

    if cache_file is not None:
        try:
            with open(cache_file, "rb") as fh:
                payload = pickle.load(fh)
            index_obj = _validate_cached_index_payload(payload=payload, scores_pkl=scores_pkl)
            if index_obj is not None:
                return ScoreIndexLoadResult(
                    index_by_fname=index_obj,
                    source="index_cache",
                    cache_file=Path(cache_file),
                )
            log(
                f"[pkl-index] cache_invalid path={cache_file} "
                "(metadata mismatch or malformed payload); falling back to in-memory index build"
            )
        except Exception as exc:
            log(f"[pkl-index] cache_read_error path={cache_file} err={exc!r}; falling back to in-memory build")

    # Read-only fallback: build in memory only.
    index_obj = build_score_stream_index(scores_pkl)
    return ScoreIndexLoadResult(index_by_fname=index_obj, source="in_memory_build", cache_file=cache_file)


@dataclass(frozen=True)
class ScoreMatrixLoadResult:
    """Result of attempting matrix load from scores.pkl by index/offset."""

    matrix: np.ndarray | None
    source_desc: str
    reason: str | None


def _num_windows_for_text_len(text_len: int, window_size: int, window_stride: int) -> int:
    """Return sliding-window count for a text length and ws/stride settings."""
    text_len = int(text_len)
    window_size = int(window_size)
    window_stride = int(window_stride)
    if text_len < window_size:
        return 0
    return ((text_len - window_size) // window_stride) + 1


def load_matrix_from_scores_pkl_index_readonly(
    *,
    scores_pkl: Path,
    score_index_by_fname: dict[str, dict],
    fname: str,
    expected_ref_text: str,
    expected_pred_text: str,
    window_size: int,
    window_stride: int,
) -> ScoreMatrixLoadResult:
    """Load one matrix from score stream using preloaded index with strict validation.

    Validation includes:
    - fname exists in index
    - item has ref/pred fields
    - item ref/pred exactly match expected runfile texts
    - matrix shape matches expected window geometry for current ws/stride
    """
    base_name = Path(str(fname)).name
    index_item = score_index_by_fname.get(base_name)
    if index_item is None:
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="index_miss")

    try:
        raw = load_score_item_by_offset(Path(scores_pkl), int(index_item["offset"]))
    except Exception as exc:
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason=f"offset_read_error:{exc!r}")

    if not isinstance(raw, dict):
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="record_not_dict")

    if "ref" not in raw or "pred" not in raw:
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="missing_ref_or_pred")

    raw_ref = str(raw.get("ref", ""))
    raw_pred = str(raw.get("pred", ""))
    if raw_ref != str(expected_ref_text):
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="ref_text_mismatch")
    if raw_pred != str(expected_pred_text):
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason="pred_text_mismatch")

    try:
        matrix = coerce_score_matrix(raw.get("scores"), source_desc=f"{scores_pkl}:{base_name}")
    except Exception as exc:
        return ScoreMatrixLoadResult(matrix=None, source_desc="scores_pkl", reason=f"coerce_error:{exc!r}")

    expected_shape = (
        int(_num_windows_for_text_len(len(raw_ref), int(window_size), int(window_stride))),
        int(_num_windows_for_text_len(len(raw_pred), int(window_size), int(window_stride))),
    )
    if tuple(matrix.shape) != expected_shape:
        return ScoreMatrixLoadResult(
            matrix=None,
            source_desc="scores_pkl",
            reason=f"shape_mismatch:got={tuple(matrix.shape)} expected={expected_shape}",
        )

    return ScoreMatrixLoadResult(matrix=matrix, source_desc="scores_pkl", reason=None)
