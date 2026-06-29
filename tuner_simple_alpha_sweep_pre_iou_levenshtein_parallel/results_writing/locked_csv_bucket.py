from __future__ import annotations

import csv
import fcntl
import os
from pathlib import Path
from typing import Mapping, Sequence

from tuner_simple_alpha_sweep_pre_iou_levenshtein_parallel.results_writing.flat_csv_tables import csv_value


class LockedCsvBucketWriter:
    """Append buffered rows to one CSV while holding a filesystem lock."""

    def __init__(self, *, csv_path: Path, lock_path: Path, fieldnames: Sequence[str]) -> None:
        self.csv_path = Path(csv_path)
        self.lock_path = Path(lock_path)
        self.fieldnames = list(fieldnames)

    def append_rows(self, rows: Sequence[Mapping[str, object]]) -> int:
        """Append rows and return how many rows reached disk."""

        if not rows:
            return 0
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            try:
                write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
                with self.csv_path.open("a", encoding="utf-8", newline="") as csv_handle:
                    writer = csv.DictWriter(csv_handle, fieldnames=self.fieldnames)
                    if write_header:
                        writer.writeheader()
                    for row in rows:
                        writer.writerow({field: csv_value(row.get(field)) for field in self.fieldnames})
                    csv_handle.flush()
                    os.fsync(csv_handle.fileno())
            finally:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
        return len(rows)
