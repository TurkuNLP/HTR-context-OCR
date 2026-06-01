from __future__ import annotations

"""Adapter for the tuner's v2.12 metric compatibility functions.

The tuner now carries the required v2.12 metric semantics locally in
``metrics.v2_12_compat``.  This adapter remains the single public boundary used
by the hot loop, so evaluator code does not need to know whether the metric
implementation is local or loaded from the historical external tree.

The external ``text_metrics_v2_12_parallel`` loader is retained only for
equivalence tests and audits.
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
    from .v2_12_compat import line_coverage_arrays as local_line_coverage_arrays
    from .v2_12_compat import line_metric_bundle as local_line_metric_bundle
except ImportError:
    from tuner.tuner_config import DEFAULT_TEXT_METRICS_V212_DIR  # type: ignore
    from metrics.v2_12_compat import line_coverage_arrays as local_line_coverage_arrays  # type: ignore
    from metrics.v2_12_compat import line_metric_bundle as local_line_metric_bundle  # type: ignore


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
_CONFIGURED_EXTERNAL_V212_DIR: Path | None = None
_CACHED_LOCAL_FUNCTIONS: V212MetricFunctions | None = None
_CACHED_EXTERNAL_FUNCTIONS: dict[Path, V212MetricFunctions] = {}

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
    """Remember the optional external v2.12 directory used for audit imports.

    Normal tuner execution uses the local compatibility implementation.  The
    external path is kept only so tests and diagnostic comparisons can ask for
    the historical implementation explicitly.
    """
    global _CONFIGURED_EXTERNAL_V212_DIR
    with _CONFIG_LOCK:
        if text_metrics_v212_dir is None:
            _CONFIGURED_EXTERNAL_V212_DIR = None
        else:
            _CONFIGURED_EXTERNAL_V212_DIR = Path(text_metrics_v212_dir).resolve()


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


def _functions_from_modules(
    *,
    line_metric_bundle: ModuleType,
    line_coverage_subtract: ModuleType,
    source_root: Path,
) -> V212MetricFunctions:
    """Build the stable callable bundle from metric modules."""
    return V212MetricFunctions(
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
        text_metrics_v212_dir=Path(source_root).resolve(),
    )


def get_v212_metric_functions(text_metrics_v212_dir: Path | None = None) -> V212MetricFunctions:
    """Return cached local v2.12-compatible metric functions."""
    global _CACHED_LOCAL_FUNCTIONS
    if text_metrics_v212_dir is not None:
        configure_text_metrics_v212_dir(text_metrics_v212_dir)

    with _CONFIG_LOCK:
        if _CACHED_LOCAL_FUNCTIONS is None:
            source_root = Path(local_line_metric_bundle.__file__).resolve().parent
            _CACHED_LOCAL_FUNCTIONS = _functions_from_modules(
                line_metric_bundle=local_line_metric_bundle,
                line_coverage_subtract=local_line_coverage_arrays,
                source_root=source_root,
            )
        return _CACHED_LOCAL_FUNCTIONS


def get_external_v212_metric_functions(text_metrics_v212_dir: Path | None = None) -> V212MetricFunctions:
    """Return metric functions loaded from the historical external v2.12 tree.

    This is for equivalence tests and audits only.  Runtime tuner scoring should
    call :func:`get_v212_metric_functions`, which returns the local implementation.
    """
    if text_metrics_v212_dir is not None:
        resolved_dir = Path(text_metrics_v212_dir).resolve()
    elif _CONFIGURED_EXTERNAL_V212_DIR is not None:
        resolved_dir = Path(_CONFIGURED_EXTERNAL_V212_DIR).resolve()
    else:
        resolved_dir = Path(DEFAULT_TEXT_METRICS_V212_DIR).resolve()

    if not resolved_dir.exists() or not resolved_dir.is_dir():
        raise NotADirectoryError(f"text_metrics_v2_12_parallel directory not found: {resolved_dir}")

    with _CONFIG_LOCK:
        cached = _CACHED_EXTERNAL_FUNCTIONS.get(resolved_dir)
        if cached is not None:
            return cached

        line_metric_bundle, line_coverage_subtract = _import_v212_modules_isolated(resolved_dir)
        functions = _functions_from_modules(
            line_metric_bundle=line_metric_bundle,
            line_coverage_subtract=line_coverage_subtract,
            source_root=resolved_dir,
        )
        _CACHED_EXTERNAL_FUNCTIONS[resolved_dir] = functions
        return functions


def build_v212_line_metric_bundle(**kwargs) -> dict:
    """Build a v2.12 line metric bundle using the read-only source function."""
    return get_v212_metric_functions().build_line_metric_bundle(**kwargs)


def build_v212_compact_line_scoring_payload(**kwargs) -> dict:
    """Build the tuner-local compact scoring payload for one line assignment.

    The historical external v2.12 tree does not contain this helper.  It is a
    tuner-owned optimization that uses the same local v2.12-compatible helper
    functions as the full bundle builder, but returns only fields needed by the
    hot-loop scorer.
    """
    return local_line_metric_bundle.build_compact_line_scoring_payload(**kwargs)


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
    "get_external_v212_metric_functions",
    "build_v212_compact_line_scoring_payload",
    "build_v212_line_metric_bundle",
    "build_v212_line_coverage_arrays_from_bundles",
    "build_v212_line_coverage_arrays_from_cached_refref_y",
    "build_v212_refref_y_coverage_array_from_bundle",
    "compute_v212_line_coverage_percentage_metrics_from_arrays",
    "compute_v212_line_coverage_ratio_metrics_from_arrays",
]
