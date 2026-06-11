#!/usr/bin/env python3
from __future__ import annotations

"""Build tuner_simple Cython helpers in place."""

from pathlib import Path

import numpy as np
from setuptools import Extension, setup

try:
    from Cython.Build import cythonize
except Exception as exc:  # pragma: no cover - only used when Cython is missing.
    raise RuntimeError(
        "Cython is required to build tuner_simple accelerators. "
        "Load an environment with Cython before running this script."
    ) from exc

THIS_DIR = Path(__file__).resolve().parent

setup(
    name="tuner_simple_cython_accel",
    ext_modules=cythonize(
        [
            Extension(
                name="cython_accel.ownership_core",
                sources=[str(THIS_DIR / "ownership_core.pyx")],
                include_dirs=[np.get_include()],
            ),
            Extension(
                name="cython_accel.filter_core",
                sources=[str(THIS_DIR / "filter_core.pyx")],
            ),
        ],
        compiler_directives={"language_level": "3"},
    ),
)
