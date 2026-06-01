from __future__ import annotations

"""Shared record helpers for tuner combination-bundle files.

The tuner writes one lightweight record for every evaluated Hough combination
when ``--with-visuals`` is enabled.  The records are intentionally observational:
they are never used for scoring, ranking, or metric computation during the tune
itself.  Keeping the read/write helpers here lets the hot writer and the later
visualization reader use exactly the same binary-stream format without copying
format logic into multiple scripts.
"""

from dataclasses import asdict, is_dataclass
import math
import pickle
from pathlib import Path
from typing import Any, BinaryIO, Iterator

import numpy as np

PICKLE_STREAM_SUFFIX = ".pklstream"
JSONL_SUFFIX = ".jsonl"
GZIP_JSONL_SUFFIX = ".jsonl.gz"
PICKLE_PROTOCOL = pickle.HIGHEST_PROTOCOL


class IncompletePickleStreamRecordError(RuntimeError):
    """Raised when a requested pickle-stream record is not present."""


def json_safe(value: Any) -> Any:
    """Convert NumPy-heavy tuner payloads into strict JSON-safe values.

    The binary bundle writer does not need this conversion, but maintenance
    tools and optional JSON exports do.  Keeping this conversion in one shared
    helper avoids subtly different JSON representations in different scripts.
    """
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


def serialize_pickle_stream_record(record: dict[str, Any]) -> bytes:
    """Serialize one combination record using the repository-wide protocol.

    The caller writes the returned bytes to disk.  Returning bytes instead of
    writing directly lets the writer measure serialization time and byte volume
    without duplicating pickle details at every call site.
    """
    return pickle.dumps(record, protocol=PICKLE_PROTOCOL)


def write_pickle_stream_record(file_handle: BinaryIO, record: dict[str, Any]) -> int:
    """Append one pickled combination record and return the written byte count."""
    encoded_record = serialize_pickle_stream_record(record)
    file_handle.write(encoded_record)
    return int(len(encoded_record))


def iter_pickle_stream_records(record_path: Path) -> Iterator[tuple[dict[str, Any], int]]:
    """Yield ``(record, one_based_record_number)`` from a pickle-stream file.

    Pickle streams do not have line numbers.  The visualization code still needs
    a stable pointer back to the selected best combination, so the one-based
    record number is used in exactly the same role that JSONL line numbers used
    before.
    """
    with Path(record_path).open("rb") as pickle_stream_handle:
        record_number = 0
        while True:
            try:
                record = pickle.load(pickle_stream_handle)
            except EOFError:
                break
            record_number += 1
            yield record, int(record_number)


def read_pickle_stream_record_at_position(record_path: Path, target_record_number: int) -> dict[str, Any]:
    """Read one pickled record by its one-based stream position."""
    for record, record_number in iter_pickle_stream_records(Path(record_path)):
        if int(record_number) == int(target_record_number):
            return record
    raise IncompletePickleStreamRecordError(
        f"Could not find pickle-stream record {int(target_record_number)} in {record_path}"
    )


__all__ = [
    "GZIP_JSONL_SUFFIX",
    "IncompletePickleStreamRecordError",
    "JSONL_SUFFIX",
    "PICKLE_PROTOCOL",
    "PICKLE_STREAM_SUFFIX",
    "iter_pickle_stream_records",
    "json_safe",
    "read_pickle_stream_record_at_position",
    "serialize_pickle_stream_record",
    "write_pickle_stream_record",
]
