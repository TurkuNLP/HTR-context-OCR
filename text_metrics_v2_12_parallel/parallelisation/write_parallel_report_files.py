"""Helpers for temporary JSONL spools and final report file writing."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path


# Create one temporary JSONL spool file underneath the chosen output directory.
def create_temp_jsonl(output_dir: Path, *, prefix: str) -> Path:
    """Create one temporary JSONL spool file under ``output_dir``."""
    # Ask the operating system for a unique temporary file path in the output directory.
    fd, tmp_path = tempfile.mkstemp(prefix=prefix, suffix=".jsonl", dir=str(output_dir))
    # Close the low-level file descriptor immediately because callers reopen by path.
    os.close(fd)
    # Return the path object so the caller can stream JSONL records into it.
    return Path(tmp_path)


# Append one JSON object to a JSONL spool file.
def write_jsonl_line(handle, payload: dict) -> None:
    """Write one JSON object as a single JSONL line."""
    # Serialize the object without changing any field names or numeric values.
    handle.write(json.dumps(payload, ensure_ascii=False))
    # Terminate the record with a newline so the file stays valid JSONL.
    handle.write("\n")


# Sort one JSONL spool by its sequence id field.
def sort_jsonl_by_seq(
    *,
    input_path: Path,
    output_path: Path,
    seq_key: str = "seq_id",
) -> int:
    """Sort a JSONL spool by sequence id and write sorted JSONL output."""
    # Collect ``(sequence_id, object)`` pairs before sorting them in memory.
    records: list[tuple[int, dict]] = []
    # Read every JSONL line from the temporary spool.
    with Path(input_path).open("r", encoding="utf-8") as inp:
        for line_no, raw_line in enumerate(inp, start=1):
            # Drop trailing whitespace so blank lines are easy to skip.
            line = raw_line.strip()
            if not line:
                continue

            # Parse the JSON object from this JSONL line.
            obj = json.loads(line)
            # Fail early if the expected sequence-id field is missing.
            if seq_key not in obj:
                raise KeyError(f"Missing '{seq_key}' in JSONL line {line_no} of {input_path}")

            # Store the parsed object together with its numeric sequence id.
            records.append((int(obj[seq_key]), obj))

    # Restore stable output order by sorting on the recorded sequence id.
    records.sort(key=lambda pair: pair[0])

    # Stream the sorted objects back out as JSONL.
    with Path(output_path).open("w", encoding="utf-8") as out:
        for _, obj in records:
            out.write(json.dumps(obj, ensure_ascii=False))
            out.write("\n")

    # Return the number of written records for sanity checks upstream.
    return len(records)


# Compute run-level success averages from the sorted success spool.
def compute_success_averages_from_sorted_jsonl(
    *,
    success_sorted_jsonl_path: Path,
) -> tuple[float | None, float | None, int, int]:
    """Compute run-level averages from the sorted success spool."""
    # Accumulate the baseline document-level NLS values.
    sum_before = 0.0
    count_before = 0
    # Accumulate the along-lines NLS values only when they are present.
    sum_along = 0.0
    count_along = 0

    # Read each success item from the sorted JSONL spool.
    with Path(success_sorted_jsonl_path).open("r", encoding="utf-8") as inp:
        for raw_line in inp:
            line = raw_line.strip()
            if not line:
                continue

            # Parse the stored internal success item.
            item = json.loads(line)
            # Add the baseline full-document NLS into the running total.
            sum_before += float(item["normalized_levenshtein_before"])
            count_before += 1

            # Add the along-lines NLS only when the item actually has one.
            along = item.get("average_normalized_levenshtein_along_lines")
            if along is not None:
                sum_along += float(along)
                count_along += 1

    # Convert the running totals into optional averages.
    avg_before = None if count_before <= 0 else float(sum_before / count_before)
    avg_along = None if count_along <= 0 else float(sum_along / count_along)
    return avg_before, avg_along, int(count_before), int(count_along)


# Pretty-print one item object while indenting it inside the outer JSON payload.
def _to_pretty_indented_json(obj: dict, *, base_indent: int) -> str:
    """Render one object as pretty JSON text with an outer indentation offset."""
    # Pretty-print the object first with ordinary JSON indentation.
    raw = json.dumps(obj, ensure_ascii=False, indent=2)
    # Build the indentation prefix required by the outer report structure.
    pad = " " * int(base_indent)
    # Re-indent every rendered line so nested item objects align under ``items``.
    return "\n".join(pad + line if line else line for line in raw.splitlines())


# Stream the final JSON payload while optionally projecting each item into a public view.
def write_payload_with_items_stream(
    *,
    output_path: Path,
    metadata: dict,
    items_jsonl_path: Path,
    strip_item_keys: tuple[str, ...] = (),
    item_transform: Callable[[dict], dict] | None = None,
) -> None:
    """Write final report JSON while streaming item payloads from JSONL.

    The resulting JSON is pretty-printed for easier human reading without
    changing the numeric values. Callers may optionally project each internal
    item into a smaller public view before it is written.
    """
    # Normalize the output path and ensure the parent directory exists.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Stream the final report as one pretty JSON file.
    with output_path.open("w", encoding="utf-8") as out:
        out.write("{\n")

        # Write all top-level metadata entries first.
        meta_items = list(metadata.items())
        for _, (key, value) in enumerate(meta_items):
            key_json = json.dumps(str(key), ensure_ascii=False)
            value_json = json.dumps(value, ensure_ascii=False, indent=2)
            value_json = value_json.replace("\n", "\n  ")
            out.write(f"  {key_json}: {value_json}")
            out.write(",\n")

        # Start the streamed items array.
        out.write('  "items": [\n')
        first = True
        with Path(items_jsonl_path).open("r", encoding="utf-8") as inp:
            for raw_line in inp:
                # Skip blank lines in the sorted JSONL spool.
                line = raw_line.strip()
                if not line:
                    continue

                # Parse the stored internal item object.
                item_obj = json.loads(line)
                # Remove any spool-only helper keys first.
                for key in strip_item_keys:
                    item_obj.pop(key, None)
                # Project the internal item into a public view when requested.
                if item_transform is not None:
                    item_obj = item_transform(item_obj)

                # Add a comma between item objects after the first one.
                if not first:
                    out.write(",\n")
                # Pretty-print this item object into the output stream.
                out.write(_to_pretty_indented_json(item_obj, base_indent=4))
                first = False

        # Close the streamed items array and the outer payload.
        out.write("\n  ]\n")
        out.write("}\n")
