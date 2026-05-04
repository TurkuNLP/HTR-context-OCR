"""Load and apply per-document Hough parameter overrides.

This module is intentionally isolated from metric computation code.
It only validates and prepares document-level Hough parameters coming
from tuner outputs like ``best_params_per_document.json``.

Expected JSON shape:
{
  "doc_count": <int>,
  "records": [
    {
      "fname": "...",
      "hough_threshold": <int>,
      "hough_line_length": <int>,
      "hough_line_gap": <int>,
      "hough_seed": <int>,
      ...
    }
  ]
}
"""

from __future__ import annotations

import json
from pathlib import Path

SELECTION_MODE_ALL_SELECTED_DOCS = "all_selected_docs"
SELECTION_MODE_ONLY_JSON_DOCS = "only_json_docs"
SUPPORTED_SELECTION_MODES = (
    SELECTION_MODE_ALL_SELECTED_DOCS,
    SELECTION_MODE_ONLY_JSON_DOCS,
)


def _normalized_fname(value: str) -> str:
    """Normalize file identifier into stable basename used across the pipeline."""
    return Path(str(value)).name


def _require_int_field(*, record: dict, key: str, record_index: int) -> int:
    """Read one integer field from a JSON record with a clear validation error."""
    if key not in record:
        raise ValueError(f"Missing field in best-params record #{record_index}: {key!r}")
    try:
        return int(record[key])
    except Exception as exc:
        raise ValueError(
            f"Invalid integer field in best-params record #{record_index}: {key!r}={record.get(key)!r}"
        ) from exc


def load_hough_params_per_document_json(*, json_path: Path) -> dict:
    """Load and validate per-document Hough parameters from one JSON file.

    Returns a payload containing:
    - ``params_by_fname``: basename -> dict of validated Hough ints
    - ``record_fnames``: sorted list of document basenames present in JSON
    - ``record_count``: number of records in JSON
    - ``doc_count_field``: optional integer value from JSON top-level ``doc_count``
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Missing per-document Hough params JSON: {path}")

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(
            f"Expected top-level JSON object in per-document Hough params file, got: {type(payload).__name__}"
        )

    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("Expected 'records' list in per-document Hough params JSON")

    doc_count_field_raw = payload.get("doc_count")
    doc_count_field = None
    if doc_count_field_raw is not None:
        try:
            doc_count_field = int(doc_count_field_raw)
        except Exception as exc:
            raise ValueError(f"Invalid 'doc_count' value in per-document Hough params JSON: {doc_count_field_raw!r}") from exc

    params_by_fname: dict[str, dict[str, int]] = {}
    for idx, raw_record in enumerate(records):
        if not isinstance(raw_record, dict):
            raise ValueError(f"Expected dict record at records[{idx}], got: {type(raw_record).__name__}")

        fname_raw = raw_record.get("fname")
        if fname_raw is None or str(fname_raw).strip() == "":
            raise ValueError(f"Missing/empty 'fname' in best-params record #{idx}")
        fname = _normalized_fname(str(fname_raw))

        if fname in params_by_fname:
            raise ValueError(f"Duplicate filename in per-document Hough params JSON: {fname!r}")

        hough_threshold = _require_int_field(record=raw_record, key="hough_threshold", record_index=idx)
        hough_line_length = _require_int_field(record=raw_record, key="hough_line_length", record_index=idx)
        hough_line_gap = _require_int_field(record=raw_record, key="hough_line_gap", record_index=idx)
        hough_seed = _require_int_field(record=raw_record, key="hough_seed", record_index=idx)

        if hough_threshold <= 0:
            raise ValueError(
                f"Invalid hough_threshold in best-params record #{idx} ({fname!r}): {hough_threshold}. Must be > 0."
            )
        if hough_line_length <= 0:
            raise ValueError(
                f"Invalid hough_line_length in best-params record #{idx} ({fname!r}): {hough_line_length}. Must be > 0."
            )
        if hough_line_gap < 0:
            raise ValueError(
                f"Invalid hough_line_gap in best-params record #{idx} ({fname!r}): {hough_line_gap}. Must be >= 0."
            )

        params_by_fname[fname] = {
            "hough_threshold": int(hough_threshold),
            "hough_line_length": int(hough_line_length),
            "hough_line_gap": int(hough_line_gap),
            "hough_seed": int(hough_seed),
        }

    record_fnames = sorted(params_by_fname.keys())

    return {
        "params_by_fname": params_by_fname,
        "record_fnames": record_fnames,
        "record_count": int(len(record_fnames)),
        "doc_count_field": doc_count_field,
    }


def apply_hough_params_selection(
    *,
    run_items: list[dict],
    selected_items: list[dict],
    params_by_fname: dict[str, dict[str, int]],
    selection_mode: str,
    strict: bool,
) -> dict:
    """Apply per-document selection/validation using prepared Hough parameter map.

    The function preserves existing pipeline behavior by default:
    - ``all_selected_docs``: keep current selected items; JSON entries override
      only matching documents.
    - ``only_json_docs``: process only selected items that exist in the JSON map.

    Strict mode enforces stronger consistency checks and raises on mismatches.
    """
    mode = str(selection_mode)
    if mode not in SUPPORTED_SELECTION_MODES:
        raise ValueError(
            f"Unsupported hough-params selection mode: {mode!r}. "
            f"Supported: {SUPPORTED_SELECTION_MODES!r}"
        )

    run_names = {_normalized_fname(item.get("fname", "")) for item in run_items}
    selected_names = [_normalized_fname(item.get("fname", "")) for item in selected_items]
    params_names = set(params_by_fname.keys())

    json_missing_in_run_items = sorted(name for name in params_names if name not in run_names)
    if strict and json_missing_in_run_items:
        raise KeyError(
            "Per-document Hough params JSON contains filenames not present in pipeline input source. "
            f"missing_count={len(json_missing_in_run_items)} missing_sample={json_missing_in_run_items[:10]}"
        )

    selected_missing_params = sorted(name for name in set(selected_names) if name not in params_names)
    if strict and mode == SELECTION_MODE_ALL_SELECTED_DOCS and selected_missing_params:
        raise KeyError(
            "Strict mode requires per-document Hough params for every selected document in "
            f"mode={SELECTION_MODE_ALL_SELECTED_DOCS!r}. "
            f"missing_count={len(selected_missing_params)} missing_sample={selected_missing_params[:10]}"
        )

    if mode == SELECTION_MODE_ONLY_JSON_DOCS:
        selected_items_out = [
            item for item in selected_items if _normalized_fname(item.get("fname", "")) in params_by_fname
        ]
    else:
        selected_items_out = list(selected_items)

    selected_out_names = {_normalized_fname(item.get("fname", "")) for item in selected_items_out}
    selected_with_params_count = sum(1 for name in selected_out_names if name in params_by_fname)

    return {
        "selected_items": selected_items_out,
        "stats": {
            "selection_mode": mode,
            "strict": bool(strict),
            "json_record_count": int(len(params_by_fname)),
            "selected_before_count": int(len(selected_items)),
            "selected_after_count": int(len(selected_items_out)),
            "selected_with_params_count": int(selected_with_params_count),
            "selected_missing_params_count": int(len(selected_missing_params)),
            "json_missing_in_run_items_count": int(len(json_missing_in_run_items)),
            "selected_missing_params_sample": selected_missing_params[:10],
            "json_missing_in_run_items_sample": json_missing_in_run_items[:10],
        },
    }
