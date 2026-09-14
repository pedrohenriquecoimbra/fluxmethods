"""fluxmethods: eddy-covariance methods, published as functions and nothing else.

There is no program here. No pipeline, no configuration, no registry, no I/O --
only the estimators, each one a plain function, importable on its own. What to do
with the numbers is the caller's business.

Where a method is one work's, the module is named for that work, so a citation
and an import are the same string::

    from fluxmethods.wilczak_et_al_2001 import double_rotation
    from fluxmethods.mauder_et_al_2013 import mauder2013

Where a step has several methods and no single reference, the module is named for
the step (:mod:`fluxmethods.resampling`, :mod:`fluxmethods.detrending`,
:mod:`fluxmethods.time_lag`, :mod:`fluxmethods.spectral`).

**The modules are the contract.** The names below are re-exported for
convenience, and one thing is deliberately *not* among them: ``block_average``
means two different methods here. In :mod:`fluxmethods.detrending` it removes a
period's mean and leaves the samples where they are; in
:mod:`fluxmethods.resampling` it is the mean of the samples falling in each
target interval. They belong to different steps, and neither has a better claim
to the bare name, so reach for them through their module.
"""

from . import detrending, resampling, signal, spectral, time_lag
from .mauder_et_al_2013 import mauder2013
from .resampling import fft_resample, linear, nearest
from .detrending import linear_detrend
from .signal import nandetrend, nanlinfit, xcov
from .spectral.lpfc import lpfc_lut
from .spectral.sensor_table import sensor_geometry
from .time_lag.fixed import fix_time_lag
from .time_lag.maximisation import time_lag, time_lag_w_default
from .time_lag.prescribed import prescribed_time_lag
from .vickers_et_al_1997 import spike_detection_vickers97
from .wilczak_et_al_2001 import double_rotation, planarfit, triple_rotation

__version__ = "0.2.0"

__all__ = [
    # the modules, which are the contract
    "detrending", "resampling", "signal", "spectral", "time_lag",
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
    # spectral
    "lpfc_lut", "sensor_geometry",
    # shared numerics
    "nandetrend", "nanlinfit", "xcov",
]
