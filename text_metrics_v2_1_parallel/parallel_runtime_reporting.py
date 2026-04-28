"""Compatibility wrapper for runtime reporting helpers."""

from __future__ import annotations

from parallelisation.record_parallel_progress import (
    maybe_emit_inflight_straggler_log,
    record_completed_envelope,
)

__all__ = [
    "maybe_emit_inflight_straggler_log",
    "record_completed_envelope",
]
