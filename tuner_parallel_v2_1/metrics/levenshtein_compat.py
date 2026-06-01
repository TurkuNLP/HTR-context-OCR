from __future__ import annotations

"""Local Levenshtein helpers for ``tuner_parallel_v2_1``.

This module intentionally mirrors the C-backed Levenshtein logic from
`text_metrics_v2_12_parallel/levenshtein_metric.py` so the tuner does not
depend on whichever shared `levenshtein_metric` module happens to be first on
`sys.path`.

Why keep it local:
- the tuner needs a stable backend-aware API (`backend=`)
- the shared projects expose different `levenshtein_metric` signatures
- importing from one local module removes runtime ambiguity
"""

try:
    from rapidfuzz.distance import Levenshtein as _RapidfuzzLevenshtein
except Exception:
    _RapidfuzzLevenshtein = None


# Stable backend labels used throughout the tuner API.
BACKEND_PYTHON = "python"
BACKEND_C = "c"
SUPPORTED_BACKENDS = (BACKEND_PYTHON, BACKEND_C)


# Pure-Python fallback retained for backwards-compatible CLI support.
def _levenshtein_distance_python(source: str, target: str) -> int:
    """Compute Levenshtein distance with a simple dynamic-programming fallback."""
    if source == target:
        return 0
    if len(source) == 0:
        return len(target)
    if len(target) == 0:
        return len(source)

    previous = list(range(len(target) + 1))
    for i, source_char in enumerate(source, start=1):
        current = [i]
        for j, target_char in enumerate(target, start=1):
            substitution_cost = 0 if source_char == target_char else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


# Exact C-backed path copied in spirit from text_metrics_v2_12_parallel.
def _levenshtein_distance_c(source: str, target: str) -> int:
    """Compute exact Levenshtein distance with the rapidfuzz C backend."""
    if _RapidfuzzLevenshtein is None:
        raise RuntimeError(
            "Levenshtein backend 'c' is unavailable because rapidfuzz is not installed in this environment."
        )
    return int(_RapidfuzzLevenshtein.distance(source, target))


# Public backend-aware distance helper.
def levenshtein_distance(source: str, target: str, *, backend: str = BACKEND_PYTHON) -> int:
    """Return exact Levenshtein distance using the requested backend."""
    if backend == BACKEND_PYTHON:
        return _levenshtein_distance_python(source, target)
    if backend == BACKEND_C:
        return _levenshtein_distance_c(source, target)
    raise ValueError(f"Unsupported Levenshtein backend {backend!r}. Allowed: {SUPPORTED_BACKENDS!r}")


# Public normalized similarity helper used by the tuner.
def normalized_levenshtein_similarity(
    predicted_text: str,
    gold_text: str,
    *,
    backend: str = BACKEND_PYTHON,
) -> float:
    """Return normalized Levenshtein similarity in ``[0, 1]``."""
    denom = max(len(predicted_text), len(gold_text))
    if denom == 0:
        return 1.0
    return 1.0 - (levenshtein_distance(predicted_text, gold_text, backend=backend) / denom)


__all__ = [
    "BACKEND_PYTHON",
    "BACKEND_C",
    "SUPPORTED_BACKENDS",
    "levenshtein_distance",
    "normalized_levenshtein_similarity",
]
