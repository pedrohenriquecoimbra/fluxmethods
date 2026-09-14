"""The shared layers the methods rest on, copied from ``oneflux_preproc/core``.

Not methods: units and their registry, constants, how a scalar was measured, the
moist-air thermodynamics, the run record, and two small numeric helpers. They
mirror that package's ``core/`` path for path -- with one deliberate exception,
in :mod:`.units`, which says so at the top of the file.

``units``, ``constants``, ``measure_type`` and ``micrometeorology`` are reached
lazily, because the first of them imports pint and the other three rest on it.
Only two of this collection's methods need that layer, and the other twenty-two
should not pay for it on ``import fluxmethods``.
"""

import importlib

from . import provenance, signal, utils

_LAZY = ("constants", "measure_type", "micrometeorology", "units")

__all__ = ["provenance", "signal", "utils", *_LAZY]


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
