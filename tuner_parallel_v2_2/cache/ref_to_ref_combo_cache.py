from __future__ import annotations

"""Exact persistent cache for reference-self Hough/filter coverage artifacts.

The tuner evaluates many ``(threshold, line_length, line_gap, seed)``
combinations.  For a fixed document and fixed combination, the reference-self
branch is deterministic and independent of the prediction text.

The production scheduler uses the document-pack cache implemented here: one
cache file stores every threshold/line-length/line-gap/seed payload for one
document and one active Hough grid.  Threshold-pack readers remain available so
older caches are still reusable, but new production writes happen after a whole
document completes instead of inside threshold-worker hot paths.
"""

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import itertools
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

# The literal-seed schemas separate new Hough geometry from older cache files
# whose random stream was offset by the document index.  Metadata comparison
# still protects readers from accidentally reusing stale threshold packs.
_COMBINATION_CACHE_SCHEMA_VERSION = "ref_to_ref_combo_cache_v2_literal_seed"
_THRESHOLD_PACK_CACHE_SCHEMA_VERSION_V2 = "ref_to_ref_threshold_pack_cache_v2"
_THRESHOLD_PACK_CACHE_SCHEMA_VERSION = "ref_to_ref_threshold_pack_cache_v4_literal_seed"
_DOCUMENT_PACK_CACHE_SCHEMA_VERSION = "ref_to_ref_document_pack_cache_v2_literal_seed"
DEFAULT_MAX_PENDING_DOCUMENT_CACHE_WRITES = 2


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
    threshold_pack_hits: int = 0
    threshold_pack_misses: int = 0
    threshold_pack_writes: int = 0
    threshold_pack_read_errors: int = 0
    threshold_pack_write_errors: int = 0
    threshold_pack_metadata_mismatches: int = 0
    document_pack_hits: int = 0
    document_pack_misses: int = 0
    document_pack_writes: int = 0
    document_pack_read_errors: int = 0
    document_pack_write_errors: int = 0
    document_pack_metadata_mismatches: int = 0
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
                "threshold_pack_hits": int(self.threshold_pack_hits),
                "threshold_pack_misses": int(self.threshold_pack_misses),
                "threshold_pack_writes": int(self.threshold_pack_writes),
                "threshold_pack_read_errors": int(self.threshold_pack_read_errors),
                "threshold_pack_write_errors": int(self.threshold_pack_write_errors),
                "threshold_pack_metadata_mismatches": int(self.threshold_pack_metadata_mismatches),
                "document_pack_hits": int(self.document_pack_hits),
                "document_pack_misses": int(self.document_pack_misses),
                "document_pack_writes": int(self.document_pack_writes),
                "document_pack_read_errors": int(self.document_pack_read_errors),
                "document_pack_write_errors": int(self.document_pack_write_errors),
                "document_pack_metadata_mismatches": int(self.document_pack_metadata_mismatches),
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


def _combination_key(*, hough_line_length: int, hough_line_gap: int, hough_seed: int) -> tuple[int, int, int]:
    """Return the compact key used inside one threshold-pack cache file."""
    return (int(hough_line_length), int(hough_line_gap), int(hough_seed))


def _document_combination_key(
    *,
    hough_threshold: int,
    hough_line_length: int,
    hough_line_gap: int,
    hough_seed: int,
) -> tuple[int, int, int, int]:
    """Return the compact key used inside one document-pack cache file."""
    return (int(hough_threshold), int(hough_line_length), int(hough_line_gap), int(hough_seed))


def _expected_threshold_pack_keys(
    *,
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
) -> set[tuple[int, int, int]]:
    """Return every combination key that should exist in a complete threshold pack."""
    return {
        _combination_key(hough_line_length=line_length, hough_line_gap=line_gap, hough_seed=seed)
        for line_length, line_gap, seed in itertools.product(line_length_values, line_gap_values, seed_values)
    }


def _expected_document_pack_keys(
    *,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
) -> set[tuple[int, int, int, int]]:
    """Return every combination key that should exist in a complete document pack."""
    return {
        _document_combination_key(
            hough_threshold=threshold,
            hough_line_length=line_length,
            hough_line_gap=line_gap,
            hough_seed=seed,
        )
        for threshold, line_length, line_gap, seed in itertools.product(
            threshold_values,
            line_length_values,
            line_gap_values,
            seed_values,
        )
    }


def _smallest_exact_integer_dtype(array: np.ndarray) -> np.dtype:
    """Return the smallest NumPy integer dtype that preserves every value."""
    values = np.asarray(array)
    if values.size == 0:
        return np.dtype(np.uint8)

    minimum_value = int(values.min())
    maximum_value = int(values.max())
    if minimum_value >= 0:
        for dtype in (np.uint8, np.uint16, np.uint32, np.uint64):
            if maximum_value <= int(np.iinfo(dtype).max):
                return np.dtype(dtype)
    else:
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            limits = np.iinfo(dtype)
            if minimum_value >= int(limits.min) and maximum_value <= int(limits.max):
                return np.dtype(dtype)
    return np.dtype(np.int64)


def _as_compact_integer_array(values) -> np.ndarray:
    """Store integer arrays with the smallest exact dtype to reduce cache size."""
    integer_array = np.asarray(values)
    return np.asarray(integer_array, dtype=_smallest_exact_integer_dtype(integer_array))


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
    """Build strict provenance metadata for one legacy reference-self cache entry."""
    effective_hough_seed = int(hough_seed)
    return {
        "cache_schema_version": _COMBINATION_CACHE_SCHEMA_VERSION,
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
        "semantics": "v2_12_falling_hough_true_iou_reference_self_exact_literal_seed",
    }


def build_ref_to_ref_threshold_pack_metadata(
    *,
    doc: SweepDocument,
    hough_threshold: int,
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    align_abs_min_len: float,
    align_min_iou_threshold: float,
    cache_schema_version: str = _THRESHOLD_PACK_CACHE_SCHEMA_VERSION,
) -> dict:
    """Build strict provenance metadata for one threshold-pack cache file."""
    normalized_seed_values = [int(seed) for seed in seed_values]
    cache_schema_version = str(cache_schema_version)
    semantics = (
        "v2_12_falling_hough_true_iou_reference_self_exact_threshold_pack_legacy_offset_seed"
        if cache_schema_version == _THRESHOLD_PACK_CACHE_SCHEMA_VERSION_V2
        else "v2_12_falling_hough_true_iou_reference_self_exact_threshold_pack_compact_literal_seed"
    )
    return {
        "cache_schema_version": cache_schema_version,
        "document_index": int(doc.index),
        "file_name": str(doc.fname),
        "reference_text_sha256": _sha256_text(doc.ref),
        "ref_to_ref_matrix_sha256": _ref_to_ref_matrix_hash_for_document(doc),
        "ref_to_ref_matrix_shape": [int(v) for v in doc.ref_to_ref_matrix.shape],
        "window_size": int(doc.window_size),
        "window_stride": int(doc.window_stride),
        "hough_threshold": int(hough_threshold),
        "hough_line_length_values": [int(value) for value in line_length_values],
        "hough_line_gap_values": [int(value) for value in line_gap_values],
        "hough_seed_values": normalized_seed_values,
        "effective_hough_seed_values": [int(seed) for seed in normalized_seed_values],
        "align_abs_min_len": float(align_abs_min_len),
        "align_min_iou_threshold": float(align_min_iou_threshold),
        "skimage_version": None if skimage is None else str(skimage.__version__),
        "semantics": semantics,
    }


def build_ref_to_ref_document_pack_metadata(
    *,
    doc: SweepDocument,
    threshold_values: list[int],
    line_length_values: list[int],
    line_gap_values: list[int],
    seed_values: list[int],
    align_abs_min_len: float,
    align_min_iou_threshold: float,
) -> dict:
    """Build strict provenance metadata for one document-level cache file."""
    normalized_seed_values = [int(seed) for seed in seed_values]
    return {
        "cache_schema_version": _DOCUMENT_PACK_CACHE_SCHEMA_VERSION,
        "document_index": int(doc.index),
        "file_name": str(doc.fname),
        "reference_text_sha256": _sha256_text(doc.ref),
        "ref_to_ref_matrix_sha256": _ref_to_ref_matrix_hash_for_document(doc),
        "ref_to_ref_matrix_shape": [int(v) for v in doc.ref_to_ref_matrix.shape],
        "window_size": int(doc.window_size),
        "window_stride": int(doc.window_stride),
        "hough_threshold_values": [int(value) for value in threshold_values],
        "hough_line_length_values": [int(value) for value in line_length_values],
        "hough_line_gap_values": [int(value) for value in line_gap_values],
        "hough_seed_values": normalized_seed_values,
        "effective_hough_seed_values": [int(seed) for seed in normalized_seed_values],
        "align_abs_min_len": float(align_abs_min_len),
        "align_min_iou_threshold": float(align_min_iou_threshold),
        "skimage_version": None if skimage is None else str(skimage.__version__),
        "semantics": "v2_12_falling_hough_true_iou_reference_self_exact_document_pack_compact_literal_seed",
    }


class RefToRefThresholdCacheSession:
    """Serve exact ref-to-ref payloads for one document and one threshold.

    This is the compatibility path for direct ``begin_threshold`` callers.  The
    production scheduler uses ``RefToRefDocumentCacheSession`` so normal runs
    write one cache file per completed document rather than one file per
    threshold.
    """

    def __init__(
        self,
        *,
        parent_cache: "RefToRefCombinationCache",
        doc: SweepDocument,
        hough_threshold: int,
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
        align_abs_min_len: float,
        align_min_iou_threshold: float,
    ) -> None:
        """Create a threshold-local cache view for one worker."""
        self.parent_cache = parent_cache
        self.doc = doc
        self.hough_threshold = int(hough_threshold)
        self.line_length_values = [int(value) for value in line_length_values]
        self.line_gap_values = [int(value) for value in line_gap_values]
        self.seed_values = [int(value) for value in seed_values]
        self.align_abs_min_len = float(align_abs_min_len)
        self.align_min_iou_threshold = float(align_min_iou_threshold)
        self.enabled = bool(parent_cache.enabled)
        self.mode = str(parent_cache.mode)
        self._lock = Lock()
        self._computed_payloads_by_key: dict[tuple[int, int, int], dict] = {}
        self._cached_payloads_by_key: dict[tuple[int, int, int], dict] = {}
        self._expected_keys = _expected_threshold_pack_keys(
            line_length_values=self.line_length_values,
            line_gap_values=self.line_gap_values,
            seed_values=self.seed_values,
        )
        self.metadata: dict | None = None
        self.cache_path: Path | None = None
        self.loaded_cache_path: Path | None = None

        if not self.enabled:
            return

        # New writes use the compact v3 metadata/key.  If that exact v3 file is
        # absent, the parent helper also tries older v2 threshold-pack metadata.
        self.metadata, self.cache_path, self._cached_payloads_by_key, self.loaded_cache_path = (
            parent_cache._read_threshold_pack_payloads_with_legacy_fallback(
                doc=doc,
                hough_threshold=int(hough_threshold),
                line_length_values=self.line_length_values,
                line_gap_values=self.line_gap_values,
                seed_values=self.seed_values,
                align_abs_min_len=float(align_abs_min_len),
                align_min_iou_threshold=float(align_min_iou_threshold),
            )
        )

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
        """Return one exact reference-self payload from the pack or compute it."""
        if not self.enabled:
            return compute_payload()

        key = _combination_key(
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
        )

        with self._lock:
            cached_payload = self._cached_payloads_by_key.get(key)
            computed_payload = self._computed_payloads_by_key.get(key)

        if cached_payload is not None:
            self.parent_cache.stats.add("hits")
            return self.parent_cache._payload_from_cached_values(
                cached_payload,
                cache_path=self.loaded_cache_path or self.cache_path,
                cache_hit=True,
            )

        if computed_payload is not None:
            # A duplicate request inside one threshold worker should not happen,
            # but returning the already computed payload keeps the session safe.
            self.parent_cache.stats.add("hits")
            return computed_payload

        self.parent_cache.stats.add("misses")
        if self.mode == REF_TO_REF_CACHE_MODE_READ_ONLY:
            raise FileNotFoundError(
                "Missing ref_to_ref threshold-pack cache entry for "
                f"threshold={int(hough_threshold)} line_length={int(hough_line_length)} "
                f"line_gap={int(hough_line_gap)} seed={int(hough_seed)} path={self.cache_path}"
            )

        fresh_payload = compute_payload()
        with self._lock:
            self._computed_payloads_by_key[key] = fresh_payload
        return {
            **fresh_payload,
            "metadata": self.metadata,
            "ref_to_ref_cache_hit": False,
            "ref_to_ref_cache_path": None if self.cache_path is None else str(self.cache_path),
        }

    def close(self) -> None:
        """Persist one complete threshold pack when auto mode computed entries.

        This compatibility method is intentionally kept for one-off callers that
        still use ``begin_threshold`` directly.  The production scheduler uses
        ``begin_document`` and writes one document pack after the document ends.
        """
        if not self.enabled or self.mode != REF_TO_REF_CACHE_MODE_AUTO:
            return
        if self.cache_path is None or self.metadata is None:
            return

        with self._lock:
            if not self._computed_payloads_by_key:
                return
            payloads_to_write = dict(self._cached_payloads_by_key)
            payloads_to_write.update(self._computed_payloads_by_key)

        # Avoid writing partial threshold packs because the metadata declares the
        # full active threshold grid.  A partial pack could make a later read-only
        # run fail halfway through a threshold, which is harder to diagnose.
        if not self._expected_keys.issubset(set(payloads_to_write)):
            return

        self.parent_cache._write_threshold_pack_payload(
            cache_path=self.cache_path,
            metadata=self.metadata,
            payloads_by_key={key: payloads_to_write[key] for key in self._expected_keys},
        )


class RefToRefDocumentThresholdCacheSession:
    """Threshold-local view backed by one document-level cache session.

    The threshold worker keeps newly computed payloads in a small local bucket
    while it evaluates line-length/line-gap combinations.  ``close()`` only
    moves those payloads into the owning document session; no filesystem write
    happens on the threshold worker.
    """

    def __init__(self, *, document_session: "RefToRefDocumentCacheSession", hough_threshold: int) -> None:
        """Create a lightweight threshold view for one document cache session."""
        self.document_session = document_session
        self.parent_cache = document_session.parent_cache
        self.hough_threshold = int(hough_threshold)
        self.enabled = bool(document_session.enabled)
        self.mode = str(document_session.mode)
        self._lock = Lock()
        self._computed_payloads_by_key: dict[tuple[int, int, int], dict] = {}

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
        """Return one cached payload or compute it without writing to disk."""
        if not self.enabled:
            return compute_payload()

        threshold = int(hough_threshold)
        short_key = _combination_key(
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
        )
        full_key = _document_combination_key(
            hough_threshold=threshold,
            hough_line_length=int(hough_line_length),
            hough_line_gap=int(hough_line_gap),
            hough_seed=int(hough_seed),
        )

        cached_payload, cache_path = self.document_session.get_cached_payload(full_key=full_key)
        if cached_payload is not None:
            self.parent_cache.stats.add("hits")
            return self.parent_cache._payload_from_cached_values(
                cached_payload,
                cache_path=cache_path,
                cache_hit=True,
            )

        with self._lock:
            computed_payload = self._computed_payloads_by_key.get(short_key)
        if computed_payload is not None:
            self.parent_cache.stats.add("hits")
            return computed_payload

        self.parent_cache.stats.add("misses")
        if self.mode == REF_TO_REF_CACHE_MODE_READ_ONLY:
            raise FileNotFoundError(
                "Missing ref_to_ref document-pack cache entry for "
                f"threshold={threshold} line_length={int(hough_line_length)} "
                f"line_gap={int(hough_line_gap)} seed={int(hough_seed)} "
                f"path={self.document_session.cache_path}"
            )

        fresh_payload = compute_payload()
        with self._lock:
            self._computed_payloads_by_key[short_key] = fresh_payload
        return {
            **fresh_payload,
            "metadata": self.document_session.metadata,
            "ref_to_ref_cache_hit": False,
            "ref_to_ref_cache_path": None if self.document_session.cache_path is None else str(self.document_session.cache_path),
        }

    def close(self) -> None:
        """Move threshold-local computed payloads into the document session."""
        if not self.enabled or self.mode != REF_TO_REF_CACHE_MODE_AUTO:
            return
        with self._lock:
            if not self._computed_payloads_by_key:
                return
            computed_payloads = dict(self._computed_payloads_by_key)
            self._computed_payloads_by_key.clear()
        self.document_session.record_computed_threshold_payloads(
            hough_threshold=int(self.hough_threshold),
            payloads_by_short_key=computed_payloads,
        )


class RefToRefDocumentCacheSession:
    """Serve and collect exact ref-to-ref payloads for one whole document."""

    def __init__(
        self,
        *,
        parent_cache: "RefToRefCombinationCache",
        doc: SweepDocument,
        threshold_values: list[int],
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
        align_abs_min_len: float,
        align_min_iou_threshold: float,
    ) -> None:
        """Create one cache session that spans the active grid for one document."""
        self.parent_cache = parent_cache
        self.doc = doc
        self.threshold_values = [int(value) for value in threshold_values]
        self.line_length_values = [int(value) for value in line_length_values]
        self.line_gap_values = [int(value) for value in line_gap_values]
        self.seed_values = [int(value) for value in seed_values]
        self.align_abs_min_len = float(align_abs_min_len)
        self.align_min_iou_threshold = float(align_min_iou_threshold)
        self.enabled = bool(parent_cache.enabled)
        self.mode = str(parent_cache.mode)
        self._lock = Lock()
        self._cached_payloads_by_full_key: dict[tuple[int, int, int, int], dict] = {}
        self._computed_payloads_by_full_key: dict[tuple[int, int, int, int], dict] = {}
        self._cache_paths_by_full_key: dict[tuple[int, int, int, int], Path] = {}
        self._legacy_thresholds_loaded: set[int] = set()
        self._expected_keys = _expected_document_pack_keys(
            threshold_values=self.threshold_values,
            line_length_values=self.line_length_values,
            line_gap_values=self.line_gap_values,
            seed_values=self.seed_values,
        )
        self.metadata: dict | None = None
        self.cache_path: Path | None = None
        self.loaded_document_cache_path: Path | None = None

        if not self.enabled:
            return

        self.metadata = build_ref_to_ref_document_pack_metadata(
            doc=doc,
            threshold_values=self.threshold_values,
            line_length_values=self.line_length_values,
            line_gap_values=self.line_gap_values,
            seed_values=self.seed_values,
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
        )
        cache_key = _cache_key_from_metadata(self.metadata)
        self.cache_path = _cache_file_path(Path(parent_cache.cache_dir), cache_key)
        self._cached_payloads_by_full_key = parent_cache._read_document_pack_payload(
            cache_path=self.cache_path,
            expected_metadata=self.metadata,
        )
        if self._cached_payloads_by_full_key:
            self.loaded_document_cache_path = self.cache_path
            self._cache_paths_by_full_key = {
                key: self.cache_path for key in self._cached_payloads_by_full_key
            }

    def begin_threshold(
        self,
        *,
        doc: SweepDocument,
        hough_threshold: int,
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
        align_abs_min_len: float,
        align_min_iou_threshold: float,
    ) -> RefToRefDocumentThresholdCacheSession:
        """Return a no-I/O threshold view for the evaluator hot path."""
        return RefToRefDocumentThresholdCacheSession(
            document_session=self,
            hough_threshold=int(hough_threshold),
        )

    def get_cached_payload(
        self,
        *,
        full_key: tuple[int, int, int, int],
    ) -> tuple[dict | None, Path | None]:
        """Return a cached payload and its source path, loading legacy packs lazily."""
        with self._lock:
            cached_payload = self._cached_payloads_by_full_key.get(full_key)
            if cached_payload is not None:
                return cached_payload, self._cache_paths_by_full_key.get(full_key, self.cache_path)

        threshold = int(full_key[0])
        self._load_legacy_threshold_pack_once(hough_threshold=threshold)
        with self._lock:
            cached_payload = self._cached_payloads_by_full_key.get(full_key)
            if cached_payload is None:
                return None, None
            return cached_payload, self._cache_paths_by_full_key.get(full_key, self.cache_path)

    def _load_legacy_threshold_pack_once(self, *, hough_threshold: int) -> None:
        """Load an older threshold-pack cache for one threshold at most once."""
        threshold = int(hough_threshold)
        with self._lock:
            if threshold in self._legacy_thresholds_loaded:
                return
            self._legacy_thresholds_loaded.add(threshold)

        _, _, payloads_by_short_key, loaded_cache_path = (
            self.parent_cache._read_threshold_pack_payloads_with_legacy_fallback(
                doc=self.doc,
                hough_threshold=threshold,
                line_length_values=self.line_length_values,
                line_gap_values=self.line_gap_values,
                seed_values=self.seed_values,
                align_abs_min_len=float(self.align_abs_min_len),
                align_min_iou_threshold=float(self.align_min_iou_threshold),
            )
        )
        if not payloads_by_short_key or loaded_cache_path is None:
            return

        with self._lock:
            for short_key, payload in payloads_by_short_key.items():
                full_key = _document_combination_key(
                    hough_threshold=threshold,
                    hough_line_length=int(short_key[0]),
                    hough_line_gap=int(short_key[1]),
                    hough_seed=int(short_key[2]),
                )
                self._cached_payloads_by_full_key[full_key] = payload
                self._cache_paths_by_full_key[full_key] = loaded_cache_path

    def record_computed_threshold_payloads(
        self,
        *,
        hough_threshold: int,
        payloads_by_short_key: dict[tuple[int, int, int], dict],
    ) -> None:
        """Store finished threshold payloads for the final document cache write."""
        with self._lock:
            for short_key, payload in payloads_by_short_key.items():
                full_key = _document_combination_key(
                    hough_threshold=int(hough_threshold),
                    hough_line_length=int(short_key[0]),
                    hough_line_gap=int(short_key[1]),
                    hough_seed=int(short_key[2]),
                )
                self._computed_payloads_by_full_key[full_key] = payload

    def submit_completed_document_write(self) -> None:
        """Queue one complete document cache write and release session memory."""
        try:
            if not self.enabled or self.mode != REF_TO_REF_CACHE_MODE_AUTO:
                return
            if self.cache_path is None or self.metadata is None:
                return

            with self._lock:
                payloads_to_write = dict(self._cached_payloads_by_full_key)
                payloads_to_write.update(self._computed_payloads_by_full_key)

            # Only complete document packs are written.  Partial cache files make
            # read-only reruns fail halfway through a document, which is painful
            # for long research sweeps to diagnose.
            if not self._expected_keys.issubset(set(payloads_to_write)):
                return

            # If the current document pack was already read and no new payloads
            # were computed, there is nothing to rewrite.
            if self.loaded_document_cache_path is not None and not self._computed_payloads_by_full_key:
                return

            self.parent_cache.submit_document_pack_write(
                cache_path=self.cache_path,
                metadata=self.metadata,
                payloads_by_key={key: payloads_to_write[key] for key in self._expected_keys},
            )
        finally:
            self.clear_in_memory_payloads()

    def clear_in_memory_payloads(self) -> None:
        """Release cached/computed arrays after the owning document is complete."""
        with self._lock:
            self._cached_payloads_by_full_key.clear()
            self._computed_payloads_by_full_key.clear()
            self._cache_paths_by_full_key.clear()
            self._legacy_thresholds_loaded.clear()


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
        self._writer_lock = Lock()
        self._pending_document_write_futures: list[Future] = []
        self._document_writer_executor: ThreadPoolExecutor | None = None
        self.max_pending_document_cache_writes = int(DEFAULT_MAX_PENDING_DOCUMENT_CACHE_WRITES)
        if self.enabled and self.mode == REF_TO_REF_CACHE_MODE_AUTO:
            # One writer keeps disk compression serialized and predictable.  The
            # scheduler queues finished-document cache writes after starting the
            # next document, so CPU-heavy combination work stays in the workers.
            self._document_writer_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="ref_to_ref_document_cache_writer",
            )

    def close(self) -> None:
        """Wait for all background document-pack writes before reading stats."""
        self._wait_for_all_pending_document_writes()
        if self._document_writer_executor is not None:
            self._document_writer_executor.shutdown(wait=True)
            self._document_writer_executor = None

    def begin_document(
        self,
        *,
        doc: SweepDocument,
        threshold_values: list[int],
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
        align_abs_min_len: float,
        align_min_iou_threshold: float,
    ) -> RefToRefDocumentCacheSession:
        """Return a document-level cache session for production scheduling."""
        return RefToRefDocumentCacheSession(
            parent_cache=self,
            doc=doc,
            threshold_values=threshold_values,
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
        )

    def begin_threshold(
        self,
        *,
        doc: SweepDocument,
        hough_threshold: int,
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
        align_abs_min_len: float,
        align_min_iou_threshold: float,
    ) -> RefToRefThresholdCacheSession:
        """Return a threshold-pack session for compatibility callers."""
        return RefToRefThresholdCacheSession(
            parent_cache=self,
            doc=doc,
            hough_threshold=int(hough_threshold),
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
        )

    def submit_document_pack_write(
        self,
        *,
        cache_path: Path,
        metadata: dict,
        payloads_by_key: dict[tuple[int, int, int, int], dict],
    ) -> None:
        """Queue one finished document cache pack for background writing."""
        if not self.enabled or self.mode != REF_TO_REF_CACHE_MODE_AUTO:
            return
        if self._document_writer_executor is None:
            return

        future = self._document_writer_executor.submit(
            self._write_document_pack_payload,
            cache_path=cache_path,
            metadata=metadata,
            payloads_by_key=payloads_by_key,
        )
        with self._writer_lock:
            self._pending_document_write_futures.append(future)
        self._wait_until_pending_document_writes_are_bounded()

    def _wait_until_pending_document_writes_are_bounded(self) -> None:
        """Keep only a small number of completed-document cache buckets in RAM."""
        while True:
            with self._writer_lock:
                pending_futures = [future for future in self._pending_document_write_futures if not future.done()]
                completed_futures = [future for future in self._pending_document_write_futures if future.done()]
                self._pending_document_write_futures = pending_futures

            for completed_future in completed_futures:
                # Fail fast if the writer could not serialize a cache file.
                completed_future.result()

            if len(pending_futures) <= self.max_pending_document_cache_writes:
                return

            done_futures, _ = wait(pending_futures, return_when=FIRST_COMPLETED)
            for done_future in done_futures:
                done_future.result()

    def _wait_for_all_pending_document_writes(self) -> None:
        """Wait until every queued document cache write has finished."""
        while True:
            with self._writer_lock:
                pending_futures = list(self._pending_document_write_futures)
                self._pending_document_write_futures.clear()
            if not pending_futures:
                return
            for future in pending_futures:
                future.result()

    def _read_threshold_pack_payloads_with_legacy_fallback(
        self,
        *,
        doc: SweepDocument,
        hough_threshold: int,
        line_length_values: list[int],
        line_gap_values: list[int],
        seed_values: list[int],
        align_abs_min_len: float,
        align_min_iou_threshold: float,
    ) -> tuple[dict, Path, dict[tuple[int, int, int], dict], Path | None]:
        """Read compact v3 threshold packs, then fallback to older v2 packs."""
        metadata = build_ref_to_ref_threshold_pack_metadata(
            doc=doc,
            hough_threshold=int(hough_threshold),
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
        )
        cache_key = _cache_key_from_metadata(metadata)
        cache_path = _cache_file_path(Path(self.cache_dir), cache_key)
        payloads_by_key = self._read_threshold_pack_payload(
            cache_path=cache_path,
            expected_metadata=metadata,
        )
        if payloads_by_key:
            return metadata, cache_path, payloads_by_key, cache_path

        v2_metadata = build_ref_to_ref_threshold_pack_metadata(
            doc=doc,
            hough_threshold=int(hough_threshold),
            line_length_values=line_length_values,
            line_gap_values=line_gap_values,
            seed_values=seed_values,
            align_abs_min_len=float(align_abs_min_len),
            align_min_iou_threshold=float(align_min_iou_threshold),
            cache_schema_version=_THRESHOLD_PACK_CACHE_SCHEMA_VERSION_V2,
        )
        v2_cache_key = _cache_key_from_metadata(v2_metadata)
        v2_cache_path = _cache_file_path(Path(self.cache_dir), v2_cache_key)
        v2_payloads_by_key = self._read_threshold_pack_payload(
            cache_path=v2_cache_path,
            expected_metadata=v2_metadata,
        )
        if v2_payloads_by_key:
            return metadata, cache_path, v2_payloads_by_key, v2_cache_path

        return metadata, cache_path, {}, None

    def _payload_from_cached_values(self, cached_payload: dict, *, cache_path: Path | None, cache_hit: bool) -> dict:
        """Attach evaluator-compatible cache-hit fields to cached scalar arrays."""
        return {
            **cached_payload,
            "bundle": None,
            "timing_hough_detect_seconds": 0.0,
            "timing_filter_seconds": 0.0,
            "timing_build_bundle_seconds": 0.0,
            "timing_direction_total_seconds": 0.0,
            "ref_to_ref_cache_hit": bool(cache_hit),
            "ref_to_ref_cache_path": None if cache_path is None else str(cache_path),
        }

    def _read_payload(self, *, cache_path: Path, expected_metadata: dict) -> dict | None:
        """Read one legacy cache entry and reject it unless metadata matches exactly."""
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
                    "skimage_raw_line_count_before_direction_filter": int(
                        payload["skimage_raw_line_count_before_direction_filter"].item()
                        if "skimage_raw_line_count_before_direction_filter" in payload.files
                        else payload["raw_line_count"].item()
                    ),
                    "direction_rejected_line_count": int(
                        payload["direction_rejected_line_count"].item()
                        if "direction_rejected_line_count" in payload.files
                        else 0
                    ),
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

    def _read_threshold_pack_payload(self, *, cache_path: Path, expected_metadata: dict) -> dict[tuple[int, int, int], dict]:
        """Read one threshold-pack cache file into a key-to-payload mapping."""
        started_at = time.perf_counter()
        if not Path(cache_path).exists():
            self.stats.add("threshold_pack_misses")
            return {}

        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                metadata_json = str(payload["metadata_json"].item())
                actual_metadata = json.loads(metadata_json)
                if actual_metadata != expected_metadata:
                    self.stats.add("threshold_pack_metadata_mismatches")
                    self.stats.add("threshold_pack_misses")
                    return {}

                combination_keys = np.asarray(payload["combination_keys"], dtype=np.int32)
                if "refref_y_by_combination" in payload.files:
                    # v2 cache files stored a full coverage row for every
                    # combination.  Keep the read path so older cache files are
                    # still exact and immediately reusable.
                    refref_y_by_combination = np.asarray(payload["refref_y_by_combination"], dtype=np.int32)
                    unique_refref_y_rows = None
                    refref_y_row_indices = None
                else:
                    # v3 cache files store each distinct coverage row once and
                    # map every combination to one of those rows.
                    unique_refref_y_rows = np.asarray(payload["refref_y_unique_rows"], dtype=np.int32)
                    refref_y_row_indices = np.asarray(payload["refref_y_row_index_by_combination"], dtype=np.int64)
                    refref_y_by_combination = None
                line_guided_columns = np.asarray(payload["line_guided_columns_by_combination"], dtype=np.int64)
                fallback_columns = np.asarray(payload["fallback_columns_by_combination"], dtype=np.int64)
                raw_line_count = np.asarray(payload["raw_line_count_by_combination"], dtype=np.int64)
                skimage_raw_line_count = np.asarray(
                    payload["skimage_raw_line_count_before_direction_filter_by_combination"]
                    if "skimage_raw_line_count_before_direction_filter_by_combination" in payload.files
                    else payload["raw_line_count_by_combination"],
                    dtype=np.int64,
                )
                direction_rejected_line_count = np.asarray(
                    payload["direction_rejected_line_count_by_combination"]
                    if "direction_rejected_line_count_by_combination" in payload.files
                    else np.zeros_like(raw_line_count),
                    dtype=np.int64,
                )
                candidate_line_count = np.asarray(payload["candidate_line_count_by_combination"], dtype=np.int64)
                used_line_count = np.asarray(payload["used_line_count_by_combination"], dtype=np.int64)
                threshold_start = np.asarray(payload["threshold_start_by_combination"], dtype=np.float64)

                payloads_by_key: dict[tuple[int, int, int], dict] = {}
                for row_index, key_row in enumerate(combination_keys):
                    key = (int(key_row[0]), int(key_row[1]), int(key_row[2]))
                    if refref_y_by_combination is not None:
                        refref_y = np.asarray(refref_y_by_combination[int(row_index)], dtype=np.int32)
                    else:
                        unique_row_index = int(refref_y_row_indices[int(row_index)])
                        refref_y = np.asarray(unique_refref_y_rows[unique_row_index], dtype=np.int32)
                    payloads_by_key[key] = {
                        "refref_y": refref_y,
                        "line_guided_columns": int(line_guided_columns[int(row_index)]),
                        "fallback_columns": int(fallback_columns[int(row_index)]),
                        "raw_line_count": int(raw_line_count[int(row_index)]),
                        "skimage_raw_line_count_before_direction_filter": int(skimage_raw_line_count[int(row_index)]),
                        "direction_rejected_line_count": int(direction_rejected_line_count[int(row_index)]),
                        "candidate_line_count": int(candidate_line_count[int(row_index)]),
                        "used_line_count": int(used_line_count[int(row_index)]),
                        "threshold_start": float(threshold_start[int(row_index)]),
                        "metadata": actual_metadata,
                    }
        except Exception:
            self.stats.add("threshold_pack_read_errors")
            self.stats.add("threshold_pack_misses")
            return {}
        finally:
            self.stats.add("read_seconds", time.perf_counter() - started_at)

        self.stats.add("threshold_pack_hits")
        return payloads_by_key

    def _read_document_pack_payload(
        self,
        *,
        cache_path: Path,
        expected_metadata: dict,
    ) -> dict[tuple[int, int, int, int], dict]:
        """Read one document-pack cache file into a full-key payload mapping."""
        started_at = time.perf_counter()
        if not Path(cache_path).exists():
            self.stats.add("document_pack_misses")
            return {}

        try:
            with np.load(cache_path, allow_pickle=False) as payload:
                metadata_json = str(payload["metadata_json"].item())
                actual_metadata = json.loads(metadata_json)
                if actual_metadata != expected_metadata:
                    self.stats.add("document_pack_metadata_mismatches")
                    self.stats.add("document_pack_misses")
                    return {}

                combination_keys = np.asarray(payload["combination_keys"], dtype=np.int32)
                unique_refref_y_rows = np.asarray(payload["refref_y_unique_rows"], dtype=np.int32)
                refref_y_row_indices = np.asarray(payload["refref_y_row_index_by_combination"], dtype=np.int64)
                line_guided_columns = np.asarray(payload["line_guided_columns_by_combination"], dtype=np.int64)
                fallback_columns = np.asarray(payload["fallback_columns_by_combination"], dtype=np.int64)
                raw_line_count = np.asarray(payload["raw_line_count_by_combination"], dtype=np.int64)
                skimage_raw_line_count = np.asarray(
                    payload["skimage_raw_line_count_before_direction_filter_by_combination"]
                    if "skimage_raw_line_count_before_direction_filter_by_combination" in payload.files
                    else payload["raw_line_count_by_combination"],
                    dtype=np.int64,
                )
                direction_rejected_line_count = np.asarray(
                    payload["direction_rejected_line_count_by_combination"]
                    if "direction_rejected_line_count_by_combination" in payload.files
                    else np.zeros_like(raw_line_count),
                    dtype=np.int64,
                )
                candidate_line_count = np.asarray(payload["candidate_line_count_by_combination"], dtype=np.int64)
                used_line_count = np.asarray(payload["used_line_count_by_combination"], dtype=np.int64)
                threshold_start = np.asarray(payload["threshold_start_by_combination"], dtype=np.float64)

                payloads_by_key: dict[tuple[int, int, int, int], dict] = {}
                for row_index, key_row in enumerate(combination_keys):
                    key = (int(key_row[0]), int(key_row[1]), int(key_row[2]), int(key_row[3]))
                    unique_row_index = int(refref_y_row_indices[int(row_index)])
                    payloads_by_key[key] = {
                        "refref_y": np.asarray(unique_refref_y_rows[unique_row_index], dtype=np.int32),
                        "line_guided_columns": int(line_guided_columns[int(row_index)]),
                        "fallback_columns": int(fallback_columns[int(row_index)]),
                        "raw_line_count": int(raw_line_count[int(row_index)]),
                        "skimage_raw_line_count_before_direction_filter": int(skimage_raw_line_count[int(row_index)]),
                        "direction_rejected_line_count": int(direction_rejected_line_count[int(row_index)]),
                        "candidate_line_count": int(candidate_line_count[int(row_index)]),
                        "used_line_count": int(used_line_count[int(row_index)]),
                        "threshold_start": float(threshold_start[int(row_index)]),
                        "metadata": actual_metadata,
                    }
        except Exception:
            self.stats.add("document_pack_read_errors")
            self.stats.add("document_pack_misses")
            return {}
        finally:
            self.stats.add("read_seconds", time.perf_counter() - started_at)

        self.stats.add("document_pack_hits")
        return payloads_by_key

    def _write_payload(self, *, cache_path: Path, metadata: dict, payload: dict) -> None:
        """Atomically write one legacy exact reference-self cache entry."""
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
                skimage_raw_line_count_before_direction_filter=np.asarray(
                    int(payload.get("skimage_raw_line_count_before_direction_filter", payload["raw_line_count"])),
                    dtype=np.int64,
                ),
                direction_rejected_line_count=np.asarray(
                    int(payload.get("direction_rejected_line_count", 0)),
                    dtype=np.int64,
                ),
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

    def _write_threshold_pack_payload(
        self,
        *,
        cache_path: Path,
        metadata: dict,
        payloads_by_key: dict[tuple[int, int, int], dict],
    ) -> None:
        """Atomically write one exact threshold-pack cache file."""
        started_at = time.perf_counter()
        sorted_keys = sorted(payloads_by_key)
        if not sorted_keys:
            return

        refref_y_rows = [np.asarray(payloads_by_key[key]["refref_y"], dtype=np.int32) for key in sorted_keys]
        refref_y_by_combination = np.stack(refref_y_rows, axis=0)
        # Many reference-self threshold packs produce the exact same coverage
        # baseline for every line_length/line_gap combination.  Store distinct
        # rows once and keep a row index per combination; reading reconstructs
        # the exact same arrays expected by the evaluator.
        unique_refref_y_rows, refref_y_row_indices = np.unique(
            refref_y_by_combination,
            axis=0,
            return_inverse=True,
        )

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
                combination_keys=_as_compact_integer_array(sorted_keys),
                refref_y_unique_rows=_as_compact_integer_array(unique_refref_y_rows),
                refref_y_row_index_by_combination=_as_compact_integer_array(refref_y_row_indices),
                line_guided_columns_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["line_guided_columns"]) for key in sorted_keys]
                ),
                fallback_columns_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["fallback_columns"]) for key in sorted_keys]
                ),
                raw_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["raw_line_count"]) for key in sorted_keys]
                ),
                skimage_raw_line_count_before_direction_filter_by_combination=_as_compact_integer_array(
                    [
                        int(
                            payloads_by_key[key].get(
                                "skimage_raw_line_count_before_direction_filter",
                                payloads_by_key[key]["raw_line_count"],
                            )
                        )
                        for key in sorted_keys
                    ]
                ),
                direction_rejected_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key].get("direction_rejected_line_count", 0)) for key in sorted_keys]
                ),
                candidate_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["candidate_line_count"]) for key in sorted_keys]
                ),
                used_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["used_line_count"]) for key in sorted_keys]
                ),
                threshold_start_by_combination=np.asarray(
                    [float(payloads_by_key[key]["threshold_start"]) for key in sorted_keys], dtype=np.float64
                ),
            )
            os.replace(str(tmp_path), str(cache_path))
            self.stats.add("threshold_pack_writes")
        except Exception:
            self.stats.add("threshold_pack_write_errors")
            raise
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
            self.stats.add("write_seconds", time.perf_counter() - started_at)

    def _write_document_pack_payload(
        self,
        *,
        cache_path: Path,
        metadata: dict,
        payloads_by_key: dict[tuple[int, int, int, int], dict],
    ) -> None:
        """Atomically write one exact document-pack cache file."""
        started_at = time.perf_counter()
        sorted_keys = sorted(payloads_by_key)
        if not sorted_keys:
            return

        refref_y_rows = [np.asarray(payloads_by_key[key]["refref_y"], dtype=np.int32) for key in sorted_keys]
        refref_y_by_combination = np.stack(refref_y_rows, axis=0)
        # Reference-self coverage rows are often identical across many Hough
        # combinations.  Storing distinct rows once keeps the cache smaller
        # without changing the exact arrays reconstructed at read time.
        unique_refref_y_rows, refref_y_row_indices = np.unique(
            refref_y_by_combination,
            axis=0,
            return_inverse=True,
        )

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
                combination_keys=_as_compact_integer_array(sorted_keys),
                refref_y_unique_rows=_as_compact_integer_array(unique_refref_y_rows),
                refref_y_row_index_by_combination=_as_compact_integer_array(refref_y_row_indices),
                line_guided_columns_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["line_guided_columns"]) for key in sorted_keys]
                ),
                fallback_columns_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["fallback_columns"]) for key in sorted_keys]
                ),
                raw_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["raw_line_count"]) for key in sorted_keys]
                ),
                skimage_raw_line_count_before_direction_filter_by_combination=_as_compact_integer_array(
                    [
                        int(
                            payloads_by_key[key].get(
                                "skimage_raw_line_count_before_direction_filter",
                                payloads_by_key[key]["raw_line_count"],
                            )
                        )
                        for key in sorted_keys
                    ]
                ),
                direction_rejected_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key].get("direction_rejected_line_count", 0)) for key in sorted_keys]
                ),
                candidate_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["candidate_line_count"]) for key in sorted_keys]
                ),
                used_line_count_by_combination=_as_compact_integer_array(
                    [int(payloads_by_key[key]["used_line_count"]) for key in sorted_keys]
                ),
                threshold_start_by_combination=np.asarray(
                    [float(payloads_by_key[key]["threshold_start"]) for key in sorted_keys], dtype=np.float64
                ),
            )
            os.replace(str(tmp_path), str(cache_path))
            self.stats.add("document_pack_writes")
        except Exception:
            self.stats.add("document_pack_write_errors")
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
        """Return one reference-self payload, using the legacy direct cache path.

        Normal scheduler code uses ``begin_threshold`` instead.  This method is
        retained so direct unit tests or one-off callers still behave as before.
        """
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
                return self._payload_from_cached_values(cached_payload, cache_path=cache_path, cache_hit=True)

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


__all__ = [
    "REF_TO_REF_CACHE_MODE_AUTO",
    "REF_TO_REF_CACHE_MODE_OFF",
    "REF_TO_REF_CACHE_MODE_READ_ONLY",
    "SUPPORTED_REF_TO_REF_CACHE_MODES",
    "RefToRefCombinationCache",
    "RefToRefCombinationCacheStats",
    "RefToRefDocumentCacheSession",
    "RefToRefDocumentThresholdCacheSession",
    "RefToRefThresholdCacheSession",
    "build_ref_to_ref_cache_metadata",
    "build_ref_to_ref_document_pack_metadata",
    "build_ref_to_ref_threshold_pack_metadata",
]
