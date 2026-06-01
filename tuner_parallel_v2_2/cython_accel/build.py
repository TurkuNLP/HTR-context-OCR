#!/usr/bin/env python3
from __future__ import annotations

"""Build the default Cython extensions for ``tuner_parallel_v2_2`` in place.

Run from the tuner directory after loading an environment that provides Cython:

```bash
cd /scratch/project_2017385/dorian/Churro_copy/tuner_parallel_v2_2
module load pytorch
python cython_accel/build.py build_ext --inplace
```

The production Slurm launcher builds these extensions before submitting workers.
The Python fallbacks still exist for debugging and portability checks, while
scheduled production runs should keep the compiled helpers available.
"""

from pathlib import Path

import numpy as np
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
        Extension(
            name="cython_accel.roi_preprocessing_core",
            sources=[str(THIS_DIR / "roi_preprocessing_core.pyx")],
            include_dirs=[np.get_include()],
        ),
    ]


setup(
    name="tuner_parallel_v2_2_cython_accel",
    ext_modules=cythonize(
        build_extensions(),
        compiler_directives={"language_level": "3"},
    ),
)
