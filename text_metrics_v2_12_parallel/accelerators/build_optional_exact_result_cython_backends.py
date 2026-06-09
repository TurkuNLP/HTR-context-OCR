"""Build optional exact-result Cython backends into the external runtime cache.

This module is intentionally conservative:

- it never changes the public pipeline contract,
- it never writes compiled artifacts into the source tree,
- it never blocks execution when the toolchain is unavailable,
- it never overrides the pure-Python path unless a compiled backend was built
  successfully and published through the runtime manifest.

The main report pipeline calls this once near startup. Workers then discover the
compiled modules through the manifest and load them lazily only if the build was
successful.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import io
import json
import platform
import sys
import sysconfig
import traceback
from pathlib import Path
from typing import Any

from pipeline.runtime_artifact_paths import exact_result_cython_backend_root

__all__ = [
    "prepare_optional_exact_result_cython_backends",
]

_BACKEND_SOURCE_DIR = Path(__file__).resolve().parent / "cython_sources"
_ACTIVE_MANIFEST_PATH = exact_result_cython_backend_root() / "active_backend_manifest.json"
_BUILD_LOCK_PATH = exact_result_cython_backend_root() / "build.lock"
_BACKEND_SPECS = {
    "greedy_diagonal_segment_merging_backend": _BACKEND_SOURCE_DIR / "greedy_diagonal_segment_merging_backend.pyx",
    "true_iou_filter_backend": _BACKEND_SOURCE_DIR / "true_iou_filter_backend.pyx",
}


# Read the previously-published active manifest if one exists.
def _read_active_manifest() -> dict | None:
    """Return the active manifest dictionary, or ``None`` when missing."""
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


# Check whether the module paths recorded in one manifest still exist.
def _manifest_module_paths_exist(manifest: dict) -> bool:
    """Return whether all compiled module files in ``manifest`` still exist."""
    for compiled_module_path_string in manifest.get("modules", {}).values():
        if not Path(str(compiled_module_path_string)).exists():
            return False
    return True


# Import the optional Cython build dependencies only when a rebuild is needed.
def _import_optional_build_dependencies() -> tuple[Any, Any, Any, Any] | None:
    """Return the optional build dependencies, or ``None`` when unavailable."""
    try:
        import numpy
        from Cython.Build import cythonize
        from setuptools import Distribution, Extension

        return numpy, cythonize, Distribution, Extension
    except Exception:
        return None


# Compute one stable build key from the exact backend sources and runtime ABI.
def _compute_build_key(*, cython_version: str, numpy_version: str) -> str:
    """Return a source-and-environment hash for the optional compiled backends."""
    hasher = hashlib.sha256()
    hasher.update(sys.version.encode("utf-8"))
    hasher.update(sys.executable.encode("utf-8"))
    hasher.update(sys.implementation.cache_tag.encode("utf-8"))
    hasher.update(sysconfig.get_config_var("EXT_SUFFIX").encode("utf-8"))
    hasher.update(platform.platform().encode("utf-8"))
    hasher.update(cython_version.encode("utf-8"))
    hasher.update(numpy_version.encode("utf-8"))
    hasher.update(Path(__file__).read_bytes())

    for source_name in sorted(_BACKEND_SPECS):
        source_path = _BACKEND_SPECS[source_name]
        hasher.update(source_name.encode("utf-8"))
        hasher.update(source_path.read_bytes())

    return hasher.hexdigest()[:24]


# Locate the compiled extension file that setuptools produced for one backend.
def _find_compiled_extension_path(build_dir: Path, module_basename: str) -> Path:
    """Return the compiled extension path for one backend module name."""
    matches = sorted(build_dir.glob(f"_{module_basename}*.so"))
    if not matches:
        matches = sorted(build_dir.glob(f"_{module_basename}*.pyd"))
    if not matches:
        raise FileNotFoundError(
            f"Compiled module for {module_basename!r} was not found in {build_dir}."
        )
    return matches[0]


# Write JSON with stable formatting so runtime artifacts remain inspectable.
def _write_json(path: Path, payload: dict) -> None:
    """Write one JSON file using stable human-readable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


# Build the optional backend extensions into the requested build directory.
def _build_backend_extensions(
    *,
    build_dir: Path,
    generated_c_dir: Path,
    build_temp_dir: Path,
    numpy_module: Any,
    cythonize_function: Any,
    distribution_class: Any,
    extension_class: Any,
) -> None:
    """Compile the optional exact-result backend extensions."""
    extension_modules = [
        extension_class(
            name=f"_{module_basename}",
            sources=[str(source_path)],
            include_dirs=[numpy_module.get_include()],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        )
        for module_basename, source_path in sorted(_BACKEND_SPECS.items())
    ]

    cythonized_extensions = cythonize_function(
        extension_modules,
        build_dir=str(generated_c_dir),
        compiler_directives={
            "language_level": "3",
            "binding": False,
            "boundscheck": False,
            "wraparound": False,
            "initializedcheck": False,
            "nonecheck": False,
        },
        annotate=False,
        nthreads=0,
        quiet=True,
    )

    distribution = distribution_class(
        {
            "name": "text_metrics_v2_12_parallel_exact_result_cython_backends",
            "ext_modules": cythonized_extensions,
        }
    )
    build_ext_command = distribution.get_command_obj("build_ext")
    build_ext_command.build_lib = str(build_dir)
    build_ext_command.build_temp = str(build_temp_dir)
    build_ext_command.force = False
    build_ext_command.inplace = False

    # Suppress the verbose setuptools chatter so pipeline logs stay readable.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        distribution.run_command("build_ext")


# Prepare optional compiled backends and publish the active manifest when successful.
def prepare_optional_exact_result_cython_backends() -> dict[str, object]:
    """Best-effort build or reuse of the optional exact-result Cython backends.

    Returns a small status dictionary so the caller can log what happened without
    changing any production result payloads.
    """
    backend_root = exact_result_cython_backend_root()
    backend_root.mkdir(parents=True, exist_ok=True)

    existing_manifest = _read_active_manifest()
    if existing_manifest is not None and _manifest_module_paths_exist(existing_manifest):
        dependency_bundle = _import_optional_build_dependencies()
        if dependency_bundle is None:
            return {
                "enabled": True,
                "message": "reusing previously built exact-result Cython backends",
                "manifest_path": str(_ACTIVE_MANIFEST_PATH),
                "build_key": existing_manifest.get("build_key"),
            }
        numpy_module, _, _, _ = dependency_bundle
        try:
            import Cython

            current_build_key = _compute_build_key(
                cython_version=str(Cython.__version__),
                numpy_version=str(numpy_module.__version__),
            )
            if str(existing_manifest.get("build_key")) == current_build_key:
                return {
                    "enabled": True,
                    "message": "reusing cached exact-result Cython backends",
                    "manifest_path": str(_ACTIVE_MANIFEST_PATH),
                    "build_key": current_build_key,
                }
        except Exception:
            return {
                "enabled": True,
                "message": "reusing previously built exact-result Cython backends",
                "manifest_path": str(_ACTIVE_MANIFEST_PATH),
                "build_key": existing_manifest.get("build_key"),
            }

    dependency_bundle = _import_optional_build_dependencies()
    if dependency_bundle is None:
        return {
            "enabled": False,
            "message": "Cython toolchain unavailable; keeping Python reference backends",
            "manifest_path": None,
            "build_key": None,
        }

    numpy_module, cythonize_function, distribution_class, extension_class = dependency_bundle
    import Cython

    build_key = _compute_build_key(
        cython_version=str(Cython.__version__),
        numpy_version=str(numpy_module.__version__),
    )
    build_dir = backend_root / f"build_{build_key}"
    generated_c_dir = build_dir / "generated_c_sources"
    build_temp_dir = build_dir / "build_temp"
    build_metadata_path = build_dir / "build_metadata.json"

    with _BUILD_LOCK_PATH.open("a+", encoding="utf-8") as build_lock_file:
        fcntl.flock(build_lock_file.fileno(), fcntl.LOCK_EX)
        try:
            existing_manifest = _read_active_manifest()
            if existing_manifest is not None and _manifest_module_paths_exist(existing_manifest):
                if str(existing_manifest.get("build_key")) == build_key:
                    return {
                        "enabled": True,
                        "message": "reusing cached exact-result Cython backends",
                        "manifest_path": str(_ACTIVE_MANIFEST_PATH),
                        "build_key": build_key,
                    }

            try:
                build_dir.mkdir(parents=True, exist_ok=True)
                generated_c_dir.mkdir(parents=True, exist_ok=True)
                build_temp_dir.mkdir(parents=True, exist_ok=True)

                needs_rebuild = not all(
                    _find_compiled_extension_path(build_dir, module_basename).exists()
                    for module_basename in sorted(_BACKEND_SPECS)
                ) if build_metadata_path.exists() else True

                if needs_rebuild:
                    _build_backend_extensions(
                        build_dir=build_dir,
                        generated_c_dir=generated_c_dir,
                        build_temp_dir=build_temp_dir,
                        numpy_module=numpy_module,
                        cythonize_function=cythonize_function,
                        distribution_class=distribution_class,
                        extension_class=extension_class,
                    )

                compiled_module_paths = {
                    module_basename: str(_find_compiled_extension_path(build_dir, module_basename))
                    for module_basename in sorted(_BACKEND_SPECS)
                }
                build_metadata = {
                    "build_key": build_key,
                    "python_executable": sys.executable,
                    "python_version": sys.version,
                    "cython_version": str(Cython.__version__),
                    "numpy_version": str(numpy_module.__version__),
                    "modules": compiled_module_paths,
                }
                _write_json(build_metadata_path, build_metadata)
                _write_json(_ACTIVE_MANIFEST_PATH, build_metadata)
                return {
                    "enabled": True,
                    "message": "built exact-result Cython backends",
                    "manifest_path": str(_ACTIVE_MANIFEST_PATH),
                    "build_key": build_key,
                }
            except Exception as exc:
                return {
                    "enabled": False,
                    "message": (
                        "failed to build exact-result Cython backends; "
                        f"keeping Python reference backends ({exc!r})"
                    ),
                    "manifest_path": None,
                    "build_key": None,
                    "traceback": traceback.format_exc(),
                }
        finally:
            fcntl.flock(build_lock_file.fileno(), fcntl.LOCK_UN)
