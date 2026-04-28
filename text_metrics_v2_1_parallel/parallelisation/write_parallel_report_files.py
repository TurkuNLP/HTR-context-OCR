"""Helpers for temporary JSONL spools and final report file writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def create_temp_jsonl(output_dir: Path, *, prefix: str) -> Path:
    """Create one temporary JSONL spool file under ``output_dir``."""
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".jsonl", dir=str(output_dir))
    os.close(fd)
    return Path(tmp_path)


def write_jsonl_line(handle, payload: dict) -> None:
    """Write one JSON object as a single JSONL line."""
    handle.write(json.dumps(payload, ensure_ascii=False))
    handle.write("\n")


def sort_jsonl_by_seq(
    *,
    input_path: Path,
    output_path: Path,
    seq_key: str = "seq_id",
) -> int:
    """Sort a JSONL spool by sequence id and write sorted JSONL output."""
    records: list[tuple[int, dict]] = []
    with Path(input_path).open("r", encoding="utf-8") as inp:
        for line_no, raw_line in enumerate(inp, start=1):
            line = raw_line.strip()
            if not line:
                continue

            obj = json.loads(line)
            if seq_key not in obj:
                raise KeyError(f"Missing '{seq_key}' in JSONL line {line_no} of {input_path}")

            records.append((int(obj[seq_key]), obj))

    records.sort(key=lambda pair: pair[0])

    with Path(output_path).open("w", encoding="utf-8") as out:
        for _, obj in records:
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")

    return len(records)


def compute_success_averages_from_sorted_jsonl(
    *,
    success_sorted_jsonl_path: Path,
) -> tuple[float | None, float | None, int, int]:
    """Compute run-level averages from the sorted success spool."""
    sum_before = 0.0
    count_before = 0
    sum_along = 0.0
    count_along = 0

    with Path(success_sorted_jsonl_path).open("r", encoding="utf-8") as inp:
        for raw_line in inp:
            line = raw_line.strip()
            if not line:
                continue

            item = json.loads(line)
            sum_before += float(item["normalized_levenshtein_before"])
            count_before += 1

            along = item.get("average_normalized_levenshtein_along_lines")
            if along is not None:
                sum_along += float(along)
                count_along += 1

    avg_before = None if count_before <= 0 else float(sum_before / count_before)
    avg_along = None if count_along <= 0 else float(sum_along / count_along)
    return avg_before, avg_along, int(count_before), int(count_along)


def _to_pretty_indented_json(obj: dict, *, base_indent: int) -> str:
    """Render one object as pretty JSON text with an outer indentation offset."""
    raw = json.dumps(obj, ensure_ascii=False, indent=2)
    pad = " " * int(base_indent)
    return "\n".join(pad + line if line else line for line in raw.splitlines())


def write_payload_with_items_stream(
    *,
    output_path: Path,
    metadata: dict,
    items_jsonl_path: Path,
    strip_item_keys: tuple[str, ...] = (),
) -> None:
    """Write final report JSON while streaming item payloads from JSONL.

    The resulting JSON is pretty-printed for easier human reading without
    changing the numeric values or schema keys.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        out.write("{\n")

        meta_items = list(metadata.items())
        for _, (key, value) in enumerate(meta_items):
            key_json = json.dumps(str(key), ensure_ascii=False)
            value_json = json.dumps(value, ensure_ascii=False, indent=2)
            value_json = value_json.replace("\n", "\n  ")
            out.write(f"  {key_json}: {value_json}")
            out.write(",\n")

        out.write('  "items": [\n')
        first = True
        with Path(items_jsonl_path).open("r", encoding="utf-8") as inp:
            for raw_line in inp:
                line = raw_line.strip()
                if not line:
                    continue

                item_obj = json.loads(line)
                for key in strip_item_keys:
                    item_obj.pop(key, None)

                if not first:
                    out.write(",\n")
                out.write(_to_pretty_indented_json(item_obj, base_indent=4))
                first = False

        out.write("\n  ]\n")
        out.write("}\n")
