from __future__ import annotations

"""Optional per-combination JSONL bundles for Hough tuner visualization.

The normal tuner output keeps compact aggregate metrics.  This writer is only
constructed when a caller explicitly asks for per-combination bundles, and it
records the geometry needed to build visuals after the tuner has started:
raw Hough segments, post-Hough candidate records, final filtered lines, and
column ownership arrays.

This module is deliberately observational.  It does not rank combinations, does
not mutate evaluator payloads, and does not change the final tuner metrics.
"""

from dataclasses import asdict, is_dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from threading import Lock
from typing import Any, TextIO

import numpy as np

try:
    from ..tuner.tuner_config import LogFn, SweepDocument
except ImportError:
    from tuner.tuner_config import LogFn, SweepDocument  # type: ignore


SCHEMA_VERSION = "churro_tuner_combination_bundle_v1"
VALID_BUNDLE_SCOPES = {"none", "all", "valid-only", "invalid-only"}


def _no_log(_: str) -> None:
    """No-op logger used when the caller does not provide a log hook."""
    return


def _safe_path_component(value: str, *, max_length: int = 120) -> str:
    """Return a stable filesystem-safe path component for one document name."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    if not cleaned:
        cleaned = "document"
    if len(cleaned) <= int(max_length):
        return cleaned
    digest = hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:12]
    prefix_length = max(1, int(max_length) - len(digest) - 1)
    return f"{cleaned[:prefix_length]}_{digest}"


def json_safe(value: Any) -> Any:
    """Convert NumPy-heavy tuner payloads into strict JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        as_float = float(value)
        return as_float if math.isfinite(as_float) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, dict):
        return {str(json_safe(key)): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, set):
        return [json_safe(item) for item in sorted(value)]
    if is_dataclass(value):
        return json_safe(asdict(value))
    return str(value)


def _combination_is_valid(eval_row: dict) -> bool:
    """Return True when the evaluator marked this row as valid and scored it."""
    if not bool(eval_row.get("is_valid", True)):
        return False
    score = eval_row.get("tuning_score")
    if score is None:
        return False
    try:
        return bool(math.isfinite(float(score)))
    except Exception:
        return False


class CombinationBundleLogger:
    """Thread-safe writer for one JSONL record per evaluated combination."""

    def __init__(
        self,
        *,
        root_dir: Path,
        scope: str,
        include_candidate_lines: bool,
        shard_index: int | None = None,
        selection_index_range: tuple[int, int] | None = None,
        log_fn: LogFn | None = None,
    ) -> None:
        """Create an optional bundle logger.

        ``scope`` decides which combinations are emitted.  Normal tuner runs
        pass no logger at all, so this class is absent from the hot path unless
        visualization/debug bundle output is explicitly requested.
        """
        normalized_scope = str(scope).strip().lower()
        if normalized_scope not in VALID_BUNDLE_SCOPES:
            raise ValueError(
                f"Unsupported combination bundle scope {scope!r}; "
                f"expected one of {sorted(VALID_BUNDLE_SCOPES)!r}"
            )

        self.root_dir = Path(root_dir)
        self.scope = normalized_scope
        self.include_candidate_lines = bool(include_candidate_lines)
        self.shard_index = None if shard_index is None else int(shard_index)
        self.selection_index_range = selection_index_range
        self.log = _no_log if log_fn is None else log_fn

        self._lock = Lock()
        self._document_asset_keys: set[tuple[int, str]] = set()
        self._open_files: dict[Path, TextIO] = {}

        if self.scope != "none":
            self.root_dir.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        """Flush and close every JSONL handle opened by this writer."""
        with self._lock:
            open_files = list(self._open_files.values())
            self._open_files.clear()

        for file_handle in open_files:
            file_handle.flush()
            file_handle.close()

    def should_write(self, eval_row: dict) -> bool:
        """Return True when the configured scope wants this evaluated row."""
        if self.scope == "none":
            return False
        is_valid = _combination_is_valid(eval_row)
        if self.scope == "all":
            return True
        if self.scope == "valid-only":
            return is_valid
        if self.scope == "invalid-only":
            return not is_valid
        return False

    def document_dir(self, doc: SweepDocument) -> Path:
        """Return the bundle directory for one document."""
        safe_name = _safe_path_component(doc.fname)
        return self.root_dir / f"document_{int(doc.index):06d}_{safe_name}"

    def ensure_document_assets(self, doc: SweepDocument) -> Path:
        """Write document metadata and reusable score matrices once."""
        document_dir = self.document_dir(doc)
        asset_key = (int(doc.index), str(doc.fname))

        with self._lock:
            if asset_key in self._document_asset_keys:
                return document_dir

            document_dir.mkdir(parents=True, exist_ok=True)
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "document": {
                    "index": int(doc.index),
                    "fname": str(doc.fname),
                    "ref_text_len": int(len(doc.ref)),
                    "pred_text_len": int(len(doc.pred)),
                    "window_size": int(doc.window_size),
                    "window_stride": int(doc.window_stride),
                    "ref_to_pred_matrix_shape": [int(value) for value in doc.ref_to_pred_matrix.shape],
                    "ref_to_ref_matrix_shape": [int(value) for value in doc.ref_to_ref_matrix.shape],
                    "whole_document_nls": float(doc.whole_document_nls),
                },
                "score_matrices": {
                    "ref_to_pred": "ref_to_pred_score_matrix.npy",
                    "ref_to_ref": "ref_to_ref_score_matrix.npy",
                },
                "threshold_jsonl_pattern": "threshold_XXX.jsonl",
            }
            (document_dir / "document_metadata.json").write_text(
                json.dumps(metadata, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            np.save(document_dir / "ref_to_pred_score_matrix.npy", np.asarray(doc.ref_to_pred_matrix))
            np.save(document_dir / "ref_to_ref_score_matrix.npy", np.asarray(doc.ref_to_ref_matrix))

            self._document_asset_keys.add(asset_key)
            return document_dir

    def _open_threshold_file(self, jsonl_path: Path) -> TextIO:
        """Return a line-buffered JSONL file handle for one threshold file."""
        with self._lock:
            file_handle = self._open_files.get(jsonl_path)
            if file_handle is not None:
                return file_handle

            jsonl_path.parent.mkdir(parents=True, exist_ok=True)
            file_handle = jsonl_path.open("a", encoding="utf-8", buffering=1)
            self._open_files[jsonl_path] = file_handle
            return file_handle

    def _direction_payload(self, payload: dict) -> dict:
        """Extract visualization-relevant fields from one matrix-direction payload."""
        det = payload.get("det", {}) if isinstance(payload, dict) else {}
        filtered = payload.get("filtered", {}) if isinstance(payload, dict) else {}
        filtering_payload = {
            "lines_used": filtered.get("lines_used", []),
            "column_assignment": filtered.get("column_assignment", {}),
        }
        if self.include_candidate_lines:
            filtering_payload["lines_for_filtering"] = filtered.get("lines_for_filtering", [])

        return {
            "hough_detection": {
                "threshold_start": det.get("threshold_start"),
                "raw_lines": det.get("raw_lines", []),
                "candidate_segments": det.get("candidate_segments", []),
            },
            "filtering": filtering_payload,
        }

    def write_combination(
        self,
        *,
        doc: SweepDocument,
        hough_threshold: int,
        hough_line_length: int,
        hough_line_gap: int,
        hough_seed: int,
        align_abs_min_len: float,
        align_min_iou_threshold: float,
        eval_row: dict,
        ref_to_pred_payload: dict,
        ref_to_ref_payload: dict,
    ) -> None:
        """Write one combination record containing metrics plus geometry."""
        if not self.should_write(eval_row):
            return

        document_dir = self.ensure_document_assets(doc)
        threshold = int(hough_threshold)
        jsonl_path = document_dir / f"threshold_{threshold:03d}.jsonl"

        record = {
            "schema_version": SCHEMA_VERSION,
            "shard": {
                "shard_index": self.shard_index,
                "selection_index_start": (
                    None if self.selection_index_range is None else int(self.selection_index_range[0])
                ),
                "selection_index_end": (
                    None if self.selection_index_range is None else int(self.selection_index_range[1])
                ),
            },
            "document": {
                "index": int(doc.index),
                "fname": str(doc.fname),
                "window_size": int(doc.window_size),
                "window_stride": int(doc.window_stride),
                "whole_document_nls": float(doc.whole_document_nls),
            },
            "hough_parameters": {
                "hough_threshold": int(hough_threshold),
                "hough_line_length": int(hough_line_length),
                "hough_line_gap": int(hough_line_gap),
                "hough_seed": int(hough_seed),
                "effective_hough_seed": int(hough_seed) + int(doc.index),
                "align_abs_min_len": float(align_abs_min_len),
                "align_min_iou_threshold": float(align_min_iou_threshold),
            },
            "metrics": eval_row,
            "ref_to_pred": self._direction_payload(ref_to_pred_payload),
            "ref_to_ref": self._direction_payload(ref_to_ref_payload),
        }

        file_handle = self._open_threshold_file(jsonl_path)
        file_handle.write(json.dumps(json_safe(record), ensure_ascii=False, sort_keys=True, allow_nan=False))
        file_handle.write("\n")


__all__ = [
    "CombinationBundleLogger",
    "SCHEMA_VERSION",
    "VALID_BUNDLE_SCOPES",
    "json_safe",
]
