"""fluxmethods: eddy-covariance methods, published as functions and nothing else.

There is no program here. No pipeline, no configuration, no registry, no I/O --
only the estimators, each one a plain function, importable on its own. What to do
with the numbers is the caller's business.

The layout is the map
---------------------
One package per processing step, named for the step, holding modules named for
the work they implement::

    fluxmethods/axis_rotation/wilczak_et_al_2001.py
    fluxmethods/despiking/mauder_et_al_2013.py

That is deliberately the same tree as ``oneflux_preproc/corrections/``, path for
path and filename for filename, because these files are copies of those. The
layout *is* the equivalence map: ``diff -r`` the two trees and every difference
should be one you can name. Today there are two, both intentional -- a lazily
imported plotting stack in ``despiking/vickers_et_al_1997`` and one rebound import
in ``detrending/commonly_used``.

``signal`` sits at the top because it is shared across steps rather than owned by
one; it is a copy of ``oneflux_preproc/core/signal.py``.

The modules are the contract
----------------------------
The names re-exported here are a convenience, and one is deliberately absent:
``block_average`` is two different methods. In :mod:`fluxmethods.detrending` it
removes a period's mean and leaves the samples where they are; in
:mod:`fluxmethods.resampling` it is the mean of the samples falling in each target
interval. They belong to different steps, so reach for them through their step.
"""

from . import (axis_rotation, despiking, detrending, resampling, signal,
               spectral, time_lag, units, utils)
from .axis_rotation.wilczak_et_al_2001 import (double_rotation, planarfit,
                                               triple_rotation)
from .despiking.mauder_et_al_2013 import mauder2013
from .despiking.vickers_et_al_1997 import spike_detection_vickers97
from .detrending.commonly_used import linear_detrend
from .resampling.commonly_used import fft_resample, linear, nearest
from .signal import nandetrend, nanlinfit, xcov
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

__version__ = "0.5.0"

__all__ = [
    # the steps, which are the contract
    "axis_rotation", "despiking", "detrending", "resampling", "signal",
    "spectral", "time_lag", "units", "utils",
    # and the methods, by step
    "double_rotation", "triple_rotation", "planarfit",
    "mauder2013", "spike_detection_vickers97",
    "linear_detrend",                       # see the note above on block_average
    "nearest", "linear", "fft_resample",
    "time_lag", "time_lag_w_default", "fix_time_lag", "prescribed_time_lag",
    "block_average_highpass", "sonic_response", "analytic_tube",
    "bpcf_moncrieff_97", "bpcf_anemometric_fluxes", "bpcf_momentum",
    "lpfc_lut", "sensor_geometry",
    "ibrom_et_al_2007", "generic_experimental", "fully_analytical", "cutoff_lut",
    "nandetrend", "nanlinfit", "xcov",
]
