#!/usr/bin/env python3
from __future__ import annotations

"""Build the default Cython extensions for ``tuner_parallel_v2`` in place.

Run from the tuner directory after loading an environment that provides Cython:

```bash
cd /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2
module load pytorch
python cython_accel/build.py build_ext --inplace
```

The Slurm wrapper builds these extensions by default after loading the
``pytorch`` module.  The Python fallbacks still exist for debugging, but normal
scheduled tuner runs require the compiled helpers before the sweep starts.
"""

from pathlib import Path

from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except Exception as exc:  # pragma: no cover - exercised only in missing-Cython environments.
    raise RuntimeError(
        "Cython is required to build optional tuner extensions. "
        "Load an environment with Cython before running this build script."
    ) from exc


THIS_DIR = Path(__file__).resolve().parent


def build_extensions() -> list[Extension]:
    """Return the optional extension definitions used by setuptools."""
    return [
        Extension(
            name="cython_accel.along_lines_core",
            sources=[str(THIS_DIR / "along_lines_core.pyx")],
        ),
        Extension(
            name="cython_accel.filter_core",
            sources=[str(THIS_DIR / "filter_core.pyx")],
        ),
    ]


setup(
    name="tuner_parallel_v2_cython_accel",
    ext_modules=cythonize(
        build_extensions(),
        compiler_directives={"language_level": "3"},
    ),
)
