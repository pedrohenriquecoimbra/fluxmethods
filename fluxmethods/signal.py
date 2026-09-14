"""Shared signal-processing primitives (pure numpy), ported from GEddySoft.

These are low-level array functions used across several subsystems (QAQC steady-
state tests and flux uncertainties, the VOC tracer covariance, and linear
detrending), so they live in ``core`` to avoid cross-package imports between
``qaqc``, ``corrections`` and ``io``.

Ported from GEddySoft v4.1 (Bernard Heinesch, University of Liège, Gembloux
Agro-Bio Tech): ``xcov.py``, ``nanlinfit.py`` and ``nandetrend.py``. The release
is named because it matters: ``xcov`` counts its samples differently from v4.0,
and this package reproduces v4.1.
"""

import numpy as np


def xcov(x, y, lag):
    """Cross-covariance of ``x`` and ``y`` over the integer lag window ``lag``.

    Parameters
    ----------
    x, y : array_like
        1-D input series.
    lag : sequence of two ints
        ``[start, end]`` lag interval in samples (inclusive).

    Returns
    -------
    float or numpy.ndarray
        Covariance at each lag; a float if the window is a single lag.

    Notes
    -----
    Matches GEddySoft **v4.1** (the release this package reproduces, and the one
    the ``@geddysoft`` twins call), including
    its use of the truncated overlap at each lag and the wraparound ``np.roll``
    leaves in the negative branch.

    A sample pair counts when *both* of its members are finite, which is the one
    thing v4.1 changed here: v4.0 took the sums and the count over ``x`` alone,
    so a gap in ``y`` and not in ``x`` was normalised by the wrong count -- and
    on a real period, where a NaN is a dropped sample rather than a missing
    pair, that moved the statistics built on this function by orders of
    magnitude rather than by a rounding.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lags = np.linspace(lag[0], lag[1], num=lag[1] - lag[0] + 1, dtype='int64')
    crosscov = np.full(len(lags), np.nan)

    for i, li in enumerate(lags):
        yshifted = np.roll(y, li)
        if li > 0:
            xs = x[li:]
            ys = yshifted[li:]
        else:
            end = len(x) - 1 - li
            xs = x[0:end]
            ys = yshifted[0:end]
        mask = np.isfinite(xs) & np.isfinite(ys)
        n = int(mask.sum())
        if n > 1:
            xs, ys = xs[mask], ys[mask]
            crosscov[i] = (
                np.sum(xs * ys) - np.sum(xs) * np.sum(ys) / n
            ) / (n - 1)

    if len(crosscov) == 1:
        return float(crosscov[0])
    return crosscov


def nanlinfit(x):
    """Return ``[slope, offset]`` of a linear fit to ``x`` ignoring NaNs."""
    x = np.asarray(x, dtype=float)
    x = np.delete(x, np.where(np.isnan(x)))
    t = np.arange(0, len(x))
    coeff = np.polyfit(t, x, 1)
    return [coeff[0], coeff[1]]


def nandetrend(x):
    """Remove the linear trend from ``x``; NaNs remain NaN.

    The trend is estimated from the non-NaN samples (via :func:`nanlinfit`) and
    subtracted at every index, so the returned series keeps its length.
    """
    x = np.asarray(x, dtype=float)
    if np.all(np.isnan(x)):
        return x.copy()
    coeff = nanlinfit(x)
    trend = np.polyval(coeff, np.arange(0, len(x)))
    return x - trend
