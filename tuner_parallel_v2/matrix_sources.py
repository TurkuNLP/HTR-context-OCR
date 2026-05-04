from __future__ import annotations

"""Matrix source helpers: local npz cache and read-only scores.pkl index source."""

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

try:
    from .pkl_index_readonly import parse_ws_st_from_scores_pkl_name
    from .tuner_config import MATRIX_CACHE_VERSION, LogFn
except ImportError:
    from pkl_index_readonly import parse_ws_st_from_scores_pkl_name  # type: ignore
    from tuner_config import MATRIX_CACHE_VERSION, LogFn  # type: ignore


def matrix_cache_key(*, ref_text: str, pred_text: str, window_size: int, window_stride: int) -> str:
    """Build stable cache key for a score matrix."""
    h = hashlib.sha256()
    h.update(str(MATRIX_CACHE_VERSION).encode("utf-8"))
    h.update(b"|")
    h.update(f"ws={int(window_size)}|st={int(window_stride)}|".encode("utf-8"))
    h.update(str(ref_text).encode("utf-8"))
    h.update(b"\0")
    h.update(str(pred_text).encode("utf-8"))
    return h.hexdigest()


def matrix_cache_path(*, cache_dir: Path, cache_key: str) -> Path:
    """Map cache key to cache file path."""
    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.npz"


def load_matrix_from_cache(*, cache_path: Path) -> np.ndarray:
    """Read a cached matrix from disk and sanitize NaN/Inf values."""
    with np.load(cache_path, allow_pickle=False) as payload:
        if "matrix" not in payload:
            raise KeyError(f"Missing 'matrix' key in cache file: {cache_path}")
        matrix = np.asarray(payload["matrix"], dtype=float)
    if matrix.ndim != 2:
        raise ValueError(f"Cached matrix is not 2D: {cache_path} shape={matrix.shape!r}")
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)


def save_matrix_to_cache(*, cache_path: Path, matrix: np.ndarray) -> None:
    """Atomically persist a score matrix to local npz cache."""
    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{cache_path.name}.tmp.",
        suffix=".npz",
        dir=str(cache_path.parent),
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        np.savez_compressed(tmp_path, matrix=np.asarray(matrix, dtype=float))
        os.replace(str(tmp_path), str(cache_path))
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def should_enable_scores_pkl_source(
    *,
    scores_pkl_ref_to_pred: Path | None,
    window_size: int,
    window_stride: int,
    disable_pkl_matrix_source: bool,
    log_fn: LogFn,
) -> tuple[bool, str | None]:
    """Decide whether read-only score-stream matrix loading should be enabled."""
    if disable_pkl_matrix_source:
        return False, "disabled_by_flag"
    if scores_pkl_ref_to_pred is None:
        return False, "not_configured"

    scores_pkl_ref_to_pred = Path(scores_pkl_ref_to_pred)
    if not scores_pkl_ref_to_pred.exists():
        raise FileNotFoundError(f"scores_pkl_ref_to_pred path does not exist: {scores_pkl_ref_to_pred}")

    parsed_ws, parsed_st = parse_ws_st_from_scores_pkl_name(scores_pkl_ref_to_pred)
    if parsed_ws is not None and parsed_st is not None:
        if int(parsed_ws) != int(window_size) or int(parsed_st) != int(window_stride):
            log_fn(
                "[pkl-source] disabled due to ws/st mismatch: "
                f"pkl_name_ws={parsed_ws} pkl_name_st={parsed_st} "
                f"runtime_ws={int(window_size)} runtime_st={int(window_stride)}"
            )
            return False, "window_mismatch"

    return True, None


def categorize_pkl_load_failure(reason: str | None) -> str:
    """Map matrix load failure reason into stable telemetry categories."""
    if reason is None:
        return "none"
    if reason == "index_miss":
        return "index_miss"
    if reason == "ref_text_mismatch":
        return "ref_text_mismatch"
    if reason == "pred_text_mismatch":
        return "pred_text_mismatch"
    if reason.startswith("shape_mismatch"):
        return "shape_mismatch"
    return "other"


__all__ = [
    "matrix_cache_key",
    "matrix_cache_path",
    "load_matrix_from_cache",
    "save_matrix_to_cache",
    "should_enable_scores_pkl_source",
    "categorize_pkl_load_failure",
]
