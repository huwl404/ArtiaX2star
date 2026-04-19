# vim: set expandtab shiftwidth=4 softtabstop=4:
"""Copies of AIS2star fiber2star / mem2star, modified for ArtiaX plugin use.

These modules must be executable by the filament conda env's Python
(cupy / cucim / numpy / scipy / skimage / starfile / mrcfile) and must NOT
import chimerax.
"""

import os


def package_dir() -> str:
    """Absolute path of this package; used by the ArtiaX runner to locate run_step.py."""
    return os.path.dirname(os.path.abspath(__file__))
