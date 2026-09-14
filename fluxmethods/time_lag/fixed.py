import numpy as np
import xarray as xr

from .commons import acq_freq as _acq_freq, seconds_to_shift, shift_to_seconds


def fix_time_lag(move, fix=None, tlag=0, acq_freq=None,
                 lag_units='seconds', **kwargs):
    """Apply a constant, prescribed time lag (``tlag`` a physical lag in seconds).

    Shifts ``move`` by the equivalent number of samples and returns an
    ``xr.Dataset`` — the contract expected by :meth:`Correction.apply` (which does
    ``ds.assign(**result)``). ``tlag`` maps to GEddySoft's ``CONST`` lag.

    Both the argument and the reported ``{name}_time_lag`` / ``{name}_time_lag_opt``
    are in **seconds**, positive when the scalar lags the wind (see
    :mod:`~.time_lag.commons`), so a configuration is portable across acquisition rates.
    """
    freq = _acq_freq(move, acq_freq)
    shift = seconds_to_shift(tlag, freq, lag_units)
    if shift is None:                       # frequency unknown: treat as samples
        shift = -int(round(float(tlag or 0)))
    moved = move.shift(time=shift).to_dataset()

    lag_s = shift_to_seconds(shift, freq)
    steps2time = xr.ones_like(move.mean('time'))
    moved = moved.assign(**{f'{move.name}_time_lag': steps2time * lag_s,
                            f'{move.name}_time_lag_opt': steps2time * lag_s})
    return moved
