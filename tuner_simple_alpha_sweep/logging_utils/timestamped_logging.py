from __future__ import annotations

"""Small timestamped logging helpers for terminal and Slurm output."""

from collections.abc import Callable
from datetime import datetime
from typing import TextIO
import sys

# A log sink accepts one already formatted message.
# Compute or store LogSink so later code can reuse this named value clearly.
LogSink = Callable[[str], None]


# Define the current_local_timestamp function; its body below performs one named step of the pipeline.
def current_local_timestamp() -> str:
    """Return a stable local timestamp for every emitted log line."""
    # Return this computed value to the caller so the next pipeline stage can use it.
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Define the format_timestamped_message function; its body below performs one named step of the pipeline.
def format_timestamped_message(message: object, *, timestamp: str | None = None) -> str:
    """Prefix one message with a timestamp while accepting any printable value."""
    # Compute or store rendered_message so later code can reuse this named value clearly.
    rendered_message = str(message)
    # Compute or store rendered_timestamp so later code can reuse this named value clearly.
    rendered_timestamp = current_local_timestamp() if timestamp is None else str(timestamp)
    # Return this computed value to the caller so the next pipeline stage can use it.
    return f"[{rendered_timestamp}] {rendered_message}"


# Define the build_timestamped_logger function; its body below performs one named step of the pipeline.
def build_timestamped_logger(sink: LogSink | None = None) -> LogSink:
    """Return a lightweight logger callable used by the serial runner."""
    # Define the log_with_timestamp function; its body below performs one named step of the pipeline.
    def log_with_timestamp(message: str) -> None:
        # Flush standard output immediately so Slurm logs can be followed while the job is still running.
        if sink is None:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            print(format_timestamped_message(message), flush=True)
        # Forward the formatted message to a custom sink when tests or callers provide one.
        else:
            # Execute this statement as the next small step in the surrounding pipeline logic.
            sink(format_timestamped_message(message))

    # Return this computed value to the caller so the next pipeline stage can use it.
    return log_with_timestamp


# Define the write_timestamped_line function; its body below performs one named step of the pipeline.
def write_timestamped_line(message: object, *, stream: TextIO | None = None) -> None:
    """Write one timestamped line and flush immediately."""
    # Compute or store selected_stream so later code can reuse this named value clearly.
    selected_stream = sys.stdout if stream is None else stream
    # Write serialized output to disk so the run can be inspected after the process exits.
    selected_stream.write(format_timestamped_message(message) + "\n")
    # Execute this statement as the next small step in the surrounding pipeline logic.
    selected_stream.flush()


__all__ = [
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "LogSink",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "build_timestamped_logger",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "current_local_timestamp",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "format_timestamped_message",
    # Provide this literal text value to the surrounding path, message, or argument definition.
    "write_timestamped_line",
]
