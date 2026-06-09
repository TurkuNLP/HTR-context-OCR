"""Lazy loader for optional exact-result compiled backend modules.

The production pipeline must keep the pure-Python implementation as the
correctness reference. This loader never changes behavior by itself; it only
checks whether a prebuilt compiled module exists in the external runtime cache
and, if it does, imports that module by absolute file path.

Keeping this file-path loader separate from the build step gives the pipeline a
very clean contract:

1. the main process may build or refresh compiled backends before work starts,
2. worker processes only load already-built modules,
3. if anything is missing or broken, the Python reference path stays active.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

from pipeline.runtime_artifact_paths import exact_result_cython_backend_root

__all__ = [
    "load_optional_exact_result_cython_backend_module",
]

_ACTIVE_MANIFEST_PATH = exact_result_cython_backend_root() / "active_backend_manifest.json"
_LOADED_MODULES_BY_BASENAME: dict[str, ModuleType] = {}


# Read the active backend manifest if one exists and looks well-formed.
def _load_active_manifest() -> dict | None:
    """Return the active backend manifest, or ``None`` when unavailable."""
    if not _ACTIVE_MANIFEST_PATH.exists():
        return None

    try:
        manifest = json.loads(_ACTIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None

    if not isinstance(manifest, dict):
        return None
    if not isinstance(manifest.get("modules"), dict):
        return None
    return manifest


# Load one compiled backend module by absolute file path from the manifest.
def load_optional_exact_result_cython_backend_module(module_basename: str) -> ModuleType | None:
    """Load one optional compiled backend module from the runtime cache.

    Parameters
    ----------
    module_basename:
        Stable short name such as ``"greedy_diagonal_segment_merging_backend"``.

    Returns
    -------
    ModuleType | None
        The loaded compiled module, or ``None`` when no valid compiled backend is
        available.
    """
    normalized_module_basename = str(module_basename)
    cached_module = _LOADED_MODULES_BY_BASENAME.get(normalized_module_basename)
    if cached_module is not None:
        return cached_module

    active_manifest = _load_active_manifest()
    if active_manifest is None:
        return None

    module_paths_by_basename = active_manifest["modules"]
    compiled_module_path_string = module_paths_by_basename.get(normalized_module_basename)
    if not isinstance(compiled_module_path_string, str) or not compiled_module_path_string:
        return None

    compiled_module_path = Path(compiled_module_path_string)
    if not compiled_module_path.exists():
        return None

    # Load the extension module under the exact init name it was compiled with.
    import_name = f"_{normalized_module_basename}"
    module_spec = importlib.util.spec_from_file_location(import_name, compiled_module_path)
    if module_spec is None or module_spec.loader is None:
        return None

    loaded_module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(loaded_module)
    _LOADED_MODULES_BY_BASENAME[normalized_module_basename] = loaded_module
    return loaded_module
