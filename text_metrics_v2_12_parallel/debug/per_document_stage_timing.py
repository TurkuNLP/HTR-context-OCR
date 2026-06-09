"""Per-document stage-timing helpers used only when --debug is enabled."""

from __future__ import annotations

import time


# Start one stage timer and return the current high-resolution timestamp.
def start_debug_stage_timing() -> float:
    """Return a high-resolution timestamp for one debug-timed stage."""
    # Read the current monotonic high-resolution clock value.
    return time.perf_counter()


# Finish one stage timer and write the elapsed duration into the timings dict.
def finish_debug_stage_timing(
    timings: dict[str, float],
    *,
    key: str,
    start_time: float | None,
) -> None:
    """Store elapsed seconds for one debug-timed stage when a start exists."""
    # Do nothing when the caller has no valid start timestamp.
    if start_time is None:
        return
    # Compute the elapsed duration using the same high-resolution clock.
    elapsed_seconds = float(time.perf_counter() - start_time)
    # Store the elapsed duration under the requested timing key.
    timings[str(key)] = elapsed_seconds
