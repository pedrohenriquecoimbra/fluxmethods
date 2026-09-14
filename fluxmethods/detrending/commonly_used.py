import numpy as np
import xarray as xr

from ..signal import nandetrend


def block_average(x, **kwargs):
    """Detrending-registry routine: remove the block mean of ``x``.

    Returns an ``xr.Dataset`` keyed by the variable name, matching the contract
    of :meth:`Correction.apply` (``ds.assign(**result)``), consistent with
    :func:`linear_detrend` and the despiking routines.
    """
    return (x - x.mean()).to_dataset()


def linear_detrend(x, **kwargs):
    """Detrending-registry routine: linearly detrend ``x`` (GEddySoft ``ld``).

    Removes a NaN-aware linear trend (see :func:`fluxmethods.signal.nandetrend`,
    ported from GEddySoft) and returns an ``xr.Dataset`` keyed by the variable
    name, matching the contract of :meth:`Correction.apply` (``ds.assign(**result)``).
    """
    stacked = x.stack({'_ld': x.dims})
    detrended = nandetrend(np.asarray(stacked.data, dtype=float))
    result = stacked.copy(data=detrended).unstack('_ld')
    return result.to_dataset()
