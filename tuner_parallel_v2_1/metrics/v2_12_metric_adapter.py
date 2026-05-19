from __future__ import annotations

"""Read-only adapter for metric functions from ``text_metrics_v2_12_parallel``.

The tuner must reuse v2.12 coverage and hallucination semantics exactly, but it
must not edit that directory.  This adapter imports the needed v2.12 functions
once, verifies their source paths, and exposes clear tuner-local wrappers.
"""

from dataclasses import dataclass
import importlib
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType
from typing import Callable

try:
    from ..tuner.tuner_config import DEFAULT_TEXT_METRICS_V212_DIR
except ImportError:
    from tuner.tuner_config import DEFAULT_TEXT_METRICS_V212_DIR  # type: ignore


@dataclass(frozen=True)
class V212MetricFunctions:
    """Resolved v2.12 metric callables and provenance paths."""

    build_line_metric_bundle: Callable
    build_line_coverage_arrays_from_bundles: Callable
    build_line_coverage_arrays_from_cached_refref_y: Callable
    build_refref_y_coverage_array_from_bundle: Callable
    compute_line_coverage_ratio_metrics_from_arrays: Callable
    compute_line_coverage_percentage_metrics_from_arrays: Callable
    line_metric_bundle_path: Path
    line_coverage_subtract_path: Path
    text_metrics_v212_dir: Path


_CONFIG_LOCK = Lock()
_CONFIGURED_V212_DIR: Path | None = None
_CACHED_FUNCTIONS: V212MetricFunctions | None = None

# These top-level module names are used by v2.12 scripts.  They conflict with
# similarly named v2.1 modules on PYTHONPATH, so the adapter imports them inside a
# temporary sys.path/sys.modules isolation block.
_IMPORT_MODULE_NAMES = (
    "line_metric_bundle",
    "line_coverage_subtract",
    "shared",
    "shared.ordered_sequence_helpers",
    "shared.project_line_to_text_windows",
)


def _module_file(module: ModuleType) -> Path:
    """Return a resolved source path for an imported module."""
    raw_path = getattr(module, "__file__", None)
    if raw_path is None:
        raise RuntimeError(f"Imported module {module.__name__!r} has no __file__ path")
    return Path(raw_path).resolve()


def _ensure_module_under_directory(*, module: ModuleType, expected_dir: Path) -> Path:
    """Validate that an imported module came from the requested v2.12 tree."""
    module_path = _module_file(module)
    expected_root = Path(expected_dir).resolve()
    try:
        module_path.relative_to(expected_root)
    except ValueError as exc:
        raise RuntimeError(
            f"Expected module {module.__name__!r} to load from {expected_root}, but got {module_path}"
        ) from exc
    return module_path


def configure_text_metrics_v212_dir(text_metrics_v212_dir: Path | None) -> None:
    """Configure the read-only v2.12 metric directory used by future imports."""
    global _CONFIGURED_V212_DIR, _CACHED_FUNCTIONS
    with _CONFIG_LOCK:
        resolved = DEFAULT_TEXT_METRICS_V212_DIR if text_metrics_v212_dir is None else Path(text_metrics_v212_dir)
        resolved = resolved.resolve()
        if not resolved.exists() or not resolved.is_dir():
            raise NotADirectoryError(f"text_metrics_v2_12_parallel directory not found: {resolved}")
        if _CONFIGURED_V212_DIR != resolved:
            _CONFIGURED_V212_DIR = resolved
            _CACHED_FUNCTIONS = None


def _import_v212_modules_isolated(text_metrics_v212_dir: Path) -> tuple[ModuleType, ModuleType]:
    """Import v2.12 modules while protecting the rest of the process imports."""
    resolved_dir = Path(text_metrics_v212_dir).resolve()
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.get(name) for name in _IMPORT_MODULE_NAMES}

    try:
        # Remove conflicting already-imported modules so importlib cannot reuse a
        # v2.1 module with the same top-level name.
        for module_name in _IMPORT_MODULE_NAMES:
            sys.modules.pop(module_name, None)

        # Prepend v2.12 for this isolated import window.  Its own modules use
        # top-level imports such as ``from shared...``.
        sys.path.insert(0, str(resolved_dir))

        line_metric_bundle = importlib.import_module("line_metric_bundle")
        line_coverage_subtract = importlib.import_module("line_coverage_subtract")

        _ensure_module_under_directory(module=line_metric_bundle, expected_dir=resolved_dir)
        _ensure_module_under_directory(module=line_coverage_subtract, expected_dir=resolved_dir)
        return line_metric_bundle, line_coverage_subtract
    finally:
        # Restore sys.path and previous modules so the tuner continues using its
        # normal v2.1/project imports after the v2.12 functions have been captured.
        sys.path[:] = saved_path
        for module_name in _IMPORT_MODULE_NAMES:
            previous_module = saved_modules[module_name]
            if previous_module is None:
                sys.modules.pop(module_name, None)
            else:
                sys.modules[module_name] = previous_module


def get_v212_metric_functions(text_metrics_v212_dir: Path | None = None) -> V212MetricFunctions:
    """Return cached v2.12 metric functions, importing them on first use."""
    global _CACHED_FUNCTIONS
    if text_metrics_v212_dir is not None or _CONFIGURED_V212_DIR is None:
        configure_text_metrics_v212_dir(text_metrics_v212_dir)

    with _CONFIG_LOCK:
        if _CACHED_FUNCTIONS is not None:
            return _CACHED_FUNCTIONS

        resolved_dir = DEFAULT_TEXT_METRICS_V212_DIR if _CONFIGURED_V212_DIR is None else _CONFIGURED_V212_DIR
        line_metric_bundle, line_coverage_subtract = _import_v212_modules_isolated(resolved_dir)

        functions = V212MetricFunctions(
            build_line_metric_bundle=getattr(line_metric_bundle, "build_line_metric_bundle"),
            build_line_coverage_arrays_from_bundles=getattr(
                line_coverage_subtract,
                "build_line_coverage_arrays_from_bundles",
            ),
            build_line_coverage_arrays_from_cached_refref_y=getattr(
                line_coverage_subtract,
                "build_line_coverage_arrays_from_cached_refref_y",
            ),
            build_refref_y_coverage_array_from_bundle=getattr(
                line_coverage_subtract,
                "build_refref_y_coverage_array_from_bundle",
            ),
            compute_line_coverage_ratio_metrics_from_arrays=getattr(
                line_coverage_subtract,
                "compute_line_coverage_ratio_metrics_from_arrays",
            ),
            compute_line_coverage_percentage_metrics_from_arrays=getattr(
                line_coverage_subtract,
                "compute_line_coverage_percentage_metrics_from_arrays",
            ),
            line_metric_bundle_path=_module_file(line_metric_bundle),
            line_coverage_subtract_path=_module_file(line_coverage_subtract),
            text_metrics_v212_dir=Path(resolved_dir).resolve(),
        )
        _CACHED_FUNCTIONS = functions
        return functions


def build_v212_line_metric_bundle(**kwargs) -> dict:
    """Build a v2.12 line metric bundle using the read-only source function."""
    return get_v212_metric_functions().build_line_metric_bundle(**kwargs)


def build_v212_line_coverage_arrays_from_bundles(*, refref_bundle: dict, other_bundle: dict) -> dict:
    """Build v2.12 coverage arrays from reference-self and ref-to-pred bundles."""
    return get_v212_metric_functions().build_line_coverage_arrays_from_bundles(
        refref_bundle=refref_bundle,
        other_bundle=other_bundle,
    )


def build_v212_refref_y_coverage_array_from_bundle(*, refref_bundle: dict):
    """Build the v2.12 reference-self y-axis coverage baseline array."""
    return get_v212_metric_functions().build_refref_y_coverage_array_from_bundle(
        refref_bundle=refref_bundle,
    )


def build_v212_line_coverage_arrays_from_cached_refref_y(*, refref_y, other_bundle: dict) -> dict:
    """Build v2.12 coverage arrays from cached reference-self y coverage."""
    return get_v212_metric_functions().build_line_coverage_arrays_from_cached_refref_y(
        refref_y=refref_y,
        other_bundle=other_bundle,
    )


def compute_v212_line_coverage_percentage_metrics_from_arrays(
    *,
    y_diff,
    other_x,
    file_name: str | None = None,
) -> dict:
    """Compute v2.12 missing/ok/repetition/hallucination percentages."""
    return get_v212_metric_functions().compute_line_coverage_percentage_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name=file_name,
    )


def compute_v212_line_coverage_ratio_metrics_from_arrays(
    *,
    y_diff,
    other_x,
    file_name: str | None = None,
) -> dict:
    """Compute v2.12 coverage metrics directly as ``0..1`` ratios."""
    return get_v212_metric_functions().compute_line_coverage_ratio_metrics_from_arrays(
        y_diff=y_diff,
        other_x=other_x,
        file_name=file_name,
    )


__all__ = [
    "V212MetricFunctions",
    "configure_text_metrics_v212_dir",
    "get_v212_metric_functions",
    "build_v212_line_metric_bundle",
    "build_v212_line_coverage_arrays_from_bundles",
    "build_v212_line_coverage_arrays_from_cached_refref_y",
    "build_v212_refref_y_coverage_array_from_bundle",
    "compute_v212_line_coverage_percentage_metrics_from_arrays",
    "compute_v212_line_coverage_ratio_metrics_from_arrays",
]
