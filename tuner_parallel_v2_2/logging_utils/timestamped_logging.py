from __future__ import annotations

"""Timestamped console logging helpers for the Hough tuner.

The tuner is usually executed through Slurm, where stdout and stderr are the
main observability tools during long multi-document runs.  This module keeps
logging deliberately small and dependency-free:

- every emitted line gets a local wall-clock timestamp
- callers can keep using a simple ``Callable[[str], None]`` logger interface
- no scoring logic, output schemas, or timing fields depend on this module

The helper is intentionally not based on Python's global ``logging`` module.
The tuner already passes lightweight log callables through the hot orchestration
paths, and keeping that shape avoids global logger configuration surprises when
the code is imported from notebooks, one-off scripts, or Slurm wrappers.
"""

from collections.abc import Callable
from datetime import datetime
from typing import TextIO
import sys


# A log sink accepts one already formatted message.
LogSink = Callable[[str], None]


def current_local_timestamp() -> str:
    """Return the current local wall-clock timestamp used in tuner logs.

    The timestamp format is fixed so Slurm logs can be sorted, searched, and
    compared across runs without depending on locale-specific date formatting.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def format_timestamped_message(message: object, *, timestamp: str | None = None) -> str:
    """Return ``message`` prefixed with a stable local timestamp.

    ``message`` is accepted as ``object`` rather than only ``str`` because this
    mirrors the permissive behavior of ``print`` and keeps logging robust if a
    caller passes a path, number, or exception object directly.
    """
    rendered_message = str(message)
    rendered_timestamp = current_local_timestamp() if timestamp is None else str(timestamp)
    return f"[{rendered_timestamp}] {rendered_message}"


def build_timestamped_logger(sink: LogSink | None = None) -> LogSink:
    """Build a callable logger that timestamps every message before emitting it.

    Parameters
    ----------
    sink:
        Optional destination callable.  ``print`` is used by default so the
        logger works naturally with Slurm stdout capture.

    Returns
    -------
    Callable[[str], None]
        A lightweight logger compatible with the existing tuner ``log_fn`` API.
    """
    selected_sink = print if sink is None else sink

    def log_with_timestamp(message: str) -> None:
        """Emit one timestamped message through the selected sink."""
        selected_sink(format_timestamped_message(message))

    return log_with_timestamp


def write_timestamped_line(message: object, *, stream: TextIO | None = None) -> None:
    """Write one timestamped line to a text stream and flush it immediately.

    This helper is useful for very small scripts that do not need a reusable
    callable logger but still want deterministic timestamp formatting.
    """
    selected_stream = sys.stdout if stream is None else stream
    selected_stream.write(format_timestamped_message(message) + "\n")
    selected_stream.flush()


__all__ = [
    "LogSink",
    "build_timestamped_logger",
    "current_local_timestamp",
    "format_timestamped_message",
    "write_timestamped_line",
]
