"""fluxmethods: eddy-covariance methods, published as functions and nothing else.

There is no program here. No pipeline, no configuration, no registry of steps, no
I/O — only the estimators, each one a plain function, importable on its own. What
to do with the numbers is the caller's business.

The layout is the map
---------------------
One package per processing step, named for the step, holding modules named for
the work they implement; plus ``core``, the shared layers the estimators rest on::

    fluxmethods/axis_rotation/wilczak_et_al_2001.py
    fluxmethods/despiking/mauder_et_al_2013.py
    fluxmethods/core/micrometeorology.py

That is deliberately the same tree as ``oneflux_preproc`` — ``corrections/<step>/``
and ``core/`` — path for path and filename for filename, because these files are
copies of those. The layout *is* the equivalence map: ``diff -r`` the two trees
and every difference should be one you can name. They are named in the README.

The modules are the contract
----------------------------
The names re-exported here are a convenience, and one is deliberately absent:
``block_average`` is two different methods. In :mod:`fluxmethods.detrending` it
removes a period's mean and leaves the samples where they are; in
:mod:`fluxmethods.resampling` it is the mean of the samples falling in each target
interval. They belong to different steps, so reach for them through their step.
"""

import importlib

from . import (axis_rotation, core, despiking, detrending, resampling, spectral,
               time_lag)
from .axis_rotation.wilczak_et_al_2001 import (double_rotation, planarfit,
                                               triple_rotation)
from .core.signal import nandetrend, nanlinfit, xcov
from .despiking.mauder_et_al_2013 import mauder2013
from .despiking.vickers_et_al_1997 import spike_detection_vickers97
from .detrending.commonly_used import linear_detrend
from .resampling.commonly_used import fft_resample, linear, nearest
from .spectral.analytic import (analytic_tube, block_average_highpass,
                                sonic_response)
from .spectral.eddypro import (bpcf_anemometric_fluxes, bpcf_moncrieff_97,
                               bpcf_momentum)
from .spectral.lpfc import lpfc_lut
from .spectral.measured import (cutoff_lut, fully_analytical,
                                generic_experimental, ibrom_et_al_2007)
from .spectral.sensor_table import sensor_geometry
from .time_lag.fixed import fix_time_lag
from .time_lag.maximisation import time_lag, time_lag_w_default
from .time_lag.prescribed import prescribed_time_lag

__version__ = "1.0.0"

__all__ = [
    # the steps and the shared layers, which are the contract
    "axis_rotation", "core", "despiking", "detrending", "resampling",
    "spectral", "time_lag", "units",   # units is reached lazily, see below
    # axis rotation
    "double_rotation", "triple_rotation", "planarfit",
    # despiking
    "mauder2013", "spike_detection_vickers97",
    # detrending (see the note above on block_average)
    "linear_detrend",
    # resampling
    "nearest", "linear", "fft_resample",
    # time lag
    "time_lag", "time_lag_w_default", "fix_time_lag", "prescribed_time_lag",
    # spectral — no measured spectrum needed
    "block_average_highpass", "sonic_response", "analytic_tube",
    "bpcf_moncrieff_97", "bpcf_anemometric_fluxes", "bpcf_momentum", "lpfc_lut",
    # spectral — fitted to measured spectra
    "ibrom_et_al_2007", "generic_experimental", "fully_analytical", "cutoff_lut",
    # the instrument table the analytic corrections read
    "sensor_geometry",
    # shared numerics
    "nandetrend", "nanlinfit", "xcov",
]


#: Reached on first use rather than at import. :mod:`fluxmethods.units` rests on
#: pint, and only the two methods in it do; the other twenty-two should not pay
#: for a unit registry to despike a series.
def __getattr__(name):
    if name == "units":
        module = importlib.import_module(f"{__name__}.units")
        globals()["units"] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
