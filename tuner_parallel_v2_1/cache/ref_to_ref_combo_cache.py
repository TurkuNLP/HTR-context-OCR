from __future__ import annotations

"""Exact persistent cache for reference-self Hough/filter coverage artifacts.

The tuner evaluates many ``(threshold, line_length, line_gap, seed)``
combinations.  For a fixed document and fixed combination, the reference-self
branch is deterministic and independent of the prediction text.  This module
caches the exact v2.12 ``refref_y`` coverage baseline plus the metadata needed
by the existing result rows, so warm tuner runs can evaluate only the
reference-to-prediction branch in the hot combination loop.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import tempfile
from threading import Lock
import time

import numpy as np

try:
    import skimage
except Exception:  # pragma: no cover - only absent in broken environments.
    skimage = None  # type: ignore

try:
    from ..tuner.tuner_config import SweepDocument
except ImportError:
    from tuner.tuner_config import SweepDocument  # type: ignore


REF_TO_REF_CACHE_MODE_OFF = "off"
REF_TO_REF_CACHE_MODE_AUTO = "auto"
REF_TO_REF_CACHE_MODE_READ_ONLY = "read-only"
SUPPORTED_REF_TO_REF_CACHE_MODES = (
    REF_TO_REF_CACHE_MODE_OFF,
    REF_TO_REF_CACHE_MODE_AUTO,
    REF_TO_REF_CACHE_MODE_READ_ONLY,
)

_CACHE_SCHEMA_VERSION = "ref_to_ref_combo_cache_v1"


@dataclass
class RefToRefCombinationCacheStats:
    """Thread-safe counters summarized in the final tuner JSON."""

    enabled: bool
    mode: str
    cache_dir: str | None
    hits: int = 0
    misses: int = 0
    writes: int = 0
    read_errors: int = 0
    write_errors: int = 0
    read_seconds: float = 0.0
    write_seconds: float = 0.0
    metadata_mismatches: int = 0
    _lock: Lock = field(default_factory=Lock, repr=False)

    def as_dict(self) -> dict:
        """Return a JSON-friendly snapshot of the current counters."""
        with self._lock:
            return {
                "enabled": bool(self.enabled),
                "mode": str(self.mode),
                "cache_dir": self.cache_dir,
                "hits": int(self.hits),
                "misses": int(self.misses),
                "writes": int(self.writes),
                "read_errors": int(self.read_errors),
                "write_errors": int(self.write_errors),
                "metadata_mismatches": int(self.metadata_mismatches),
                "read_seconds": float(self.read_seconds),
                "write_seconds": float(self.write_seconds),
            }

    def add(self, field_name: str, amount: float | int = 1) -> None:
        """Increment one counter while protecting concurrent threshold workers."""
        with self._lock:
            setattr(self, field_name, getattr(self, field_name) + amount)


def _sha256_text(value: str) -> str:
    """Hash text content with a stable UTF-8 encoding."""
    digest = hashlib.sha256()
    digest.update(str(value).encode("utf-8"))
    return digest.hexdigest()


def _sha256_numpy_array(array: np.ndarray) -> str:
    """Hash a NumPy array's dtype, shape, and raw value bytes exactly."""
    normalized_array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(normalized_array.dtype).encode("utf-8"))
    digest.update(b"|")
    digest.update(json.dumps(tuple(int(v) for v in normalized_array.shape)).encode("utf-8"))
    digest.update(b"|")
    digest.update(normalized_array.view(np.uint8))
    return digest.hexdigest()


def _ref_to_ref_matrix_hash_for_document(doc: SweepDocument) -> str:
    """Return a per-document matrix hash, computing it only once per process."""
    cached_hash = getattr(doc, "_ref_to_ref_matrix_sha256", None)
    if cached_hash is not None:
        return str(cached_hash)
    matrix_hash = _sha256_numpy_array(doc.ref_to_ref_matrix)
    setattr(doc, "_ref_to_ref_matrix_sha256", matrix_hash)
    return matrix_hash


def _stable_json_bytes(payload: dict) -> bytes:
    """Serialize a dictionary into deterministic UTF-8 JSON bytes."""
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _cache_key_from_metadata(metadata: dict) -> str:
    """Hash cache metadata into the final cache key."""
    return hashlib.sha256(_stable_json_bytes(metadata)).hexdigest()


def _cache_file_path(cache_dir: Path, cache_key: str) -> Path:
    """Map one cache key into a two-level shard path."""
    return Path(cache_dir) / cache_key[:2] / f"{cache_key}.npz"


def build_ref_to_ref_cache_metadata(
    *,
    doc: SweepDocument,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> dict:
    """Build strict provenance metadata for one reference-self cache entry."""
    effective_hough_seed = int(hough_seed) + int(doc.index)
    return {
        "cache_schema_version": _CACHE_SCHEMA_VERSION,
        "document_index": int(doc.index),
        "file_name": str(doc.fname),
        "reference_text_sha256": _sha256_text(doc.ref),
        "ref_to_ref_matrix_sha256": _ref_to_ref_matrix_hash_for_document(doc),
        "ref_to_ref_matrix_shape": [int(v) for v in doc.ref_to_ref_matrix.shape],
        "window_size": int(doc.window_size),
        "window_stride": int(doc.window_stride),
        "hough_threshold": int(hough_threshold),
        "hough_line_length": int(hough_line_length),
        "hough_line_gap": int(hough_line_gap),
        "hough_seed": int(hough_seed),
        "effective_hough_seed": int(effective_hough_seed),
        "align_abs_min_len": float(align_abs_min_len),
        "align_min_iou_threshold": float(align_min_iou_threshold),
        "skimage_version": None if skimage is None else str(skimage.__version__),
        "semantics": "v2_12_raw_or_merged_hough_independent_ref_to_ref_true_iou",
    }


class RefToRefCombinationCache:
    """Read/write exact reference-self artifacts for tuner combinations."""

    def __init__(self, *, cache_dir: Path | None, mode: str) -> None:
        """Create a cache object for one tuner run."""
        normalized_mode = str(mode)
        if normalized_mode not in SUPPORTED_REF_TO_REF_CACHE_MODES:
            raise ValueError(
                "Unsupported ref_to_ref cache mode: "
                f"{mode!r}; expected one of {SUPPORTED_REF_TO_REF_CACHE_MODES!r}"
            )
        self.mode = normalized_mode
        self.cache_dir = None if cache_dir is None else Path(cache_dir)
        self.enabled = self.mode != REF_TO_REF_CACHE_MODE_OFF
        if self.enabled and self.cache_dir is None:
            raise ValueError("ref_to_ref cache mode requires a cache_dir unless mode is 'off'")
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.stats = RefToRefCombinationCacheStats(
            enabled=bool(self.enabled),
            mode=str(self.mode),
            cache_dir=None if self.cache_dir is None else str(self.cache_dir),
        )

    def _read_payload(self, *, cache_path: Path, expected_metadata: dict) -> dict | None:
        """Read one cache entry and reject it unless metadata matches exactly."""
        started_at = time.perf_counter()
        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                metadata_json = str(payload["metadata_json"].item())
                actual_metadata = json.loads(metadata_json)
                if actual_metadata != expected_metadata:
                    self.stats.add("metadata_mismatches")
                    return None
                loaded = {
                    "refref_y": np.asarray(payload["refref_y"], dtype=np.int32),
                    "line_guided_columns": int(payload["line_guided_columns"].item()),
                    "fallback_columns": int(payload["fallback_columns"].item()),
                    "raw_line_count": int(payload["raw_line_count"].item()),
                    "candidate_line_count": int(payload["candidate_line_count"].item()),
                    "used_line_count": int(payload["used_line_count"].item()),
                    "threshold_start": float(payload["threshold_start"].item()),
                    "metadata": actual_metadata,
                }
        except Exception:
            self.stats.add("read_errors")
            return None
        finally:
            self.stats.add("read_seconds", time.perf_counter() - started_at)
        return loaded

    def _write_payload(self, *, cache_path: Path, metadata: dict, payload: dict) -> None:
        """Atomically write one exact reference-self cache entry."""
        started_at = time.perf_counter()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{cache_path.name}.tmp.",
            suffix=".npz",
            dir=str(cache_path.parent),
        )
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            np.savez_compressed(
                tmp_path,
                metadata_json=np.asarray(json.dumps(metadata, ensure_ascii=False, sort_keys=True)),
                refref_y=np.asarray(payload["refref_y"], dtype=np.int32),
                line_guided_columns=np.asarray(int(payload["line_guided_columns"]), dtype=np.int64),
                fallback_columns=np.asarray(int(payload["fallback_columns"]), dtype=np.int64),
                raw_line_count=np.asarray(int(payload["raw_line_count"]), dtype=np.int64),
                candidate_line_count=np.asarray(int(payload["candidate_line_count"]), dtype=np.int64),
                used_line_count=np.asarray(int(payload["used_line_count"]), dtype=np.int64),
                threshold_start=np.asarray(float(payload["threshold_start"]), dtype=np.float64),
            )
            os.replace(str(tmp_path), str(cache_path))
            self.stats.add("writes")
        except Exception:
            self.stats.add("write_errors")
            raise
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            self.stats.add("write_seconds", time.perf_counter() - started_at)

    def get_or_compute(
        self,
        *,
        doc: SweepDocument,
        hough_threshold: int,
        hough_line_length: int,
        hough_line_gap: int,
        hough_seed: int,
        align_abs_min_len: float,
        align_min_iou_threshold: float,
        compute_payload: Callable[[], dict],
    ) -> dict:
        """Return one reference-self payload, using cache when configured."""
        if not self.enabled:
            return compute_payload()

        metadata = build_ref_to_ref_cache_metadata(
            doc=doc,
            hough_threshold=int(hough_threshold),
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
        )
        cache_key = _cache_key_from_metadata(metadata)
        cache_path = _cache_file_path(Path(self.cache_dir), cache_key)

        if cache_path.exists():
            cached_payload = self._read_payload(cache_path=cache_path, expected_metadata=metadata)
            if cached_payload is not None:
                self.stats.add("hits")
                return {
                    **cached_payload,
                    "bundle": None,
                    "timing_hough_detect_seconds": 0.0,
                    "timing_filter_seconds": 0.0,
                    "timing_build_bundle_seconds": 0.0,
                    "timing_direction_total_seconds": 0.0,
                    "ref_to_ref_cache_hit": True,
                    "ref_to_ref_cache_path": str(cache_path),
                }

        self.stats.add("misses")
        if self.mode == REF_TO_REF_CACHE_MODE_READ_ONLY:
            raise FileNotFoundError(f"Missing ref_to_ref cache entry for key {cache_key}: {cache_path}")

        computed_payload = compute_payload()
        self._write_payload(cache_path=cache_path, metadata=metadata, payload=computed_payload)
        return {
            **computed_payload,
            "metadata": metadata,
            "ref_to_ref_cache_hit": False,
            "ref_to_ref_cache_path": str(cache_path),
        }
