"""A subset: the seven helpers the estimators in :mod:`.measured` call.

**Not a copy of the whole module.** The reference implementation's
``corrections/spectral/commons.py`` has twenty functions; thirteen of them bin
and ensemble-average spectra, build them by FFT or wavelet, or draw them, and
four of those reach for a pint registry while one reaches for matplotlib. None of
that is a method, so none of it is here.

What is here is verbatim, and it is the integration and fitting the correction
factors are built on: Simpson weights and the integral over them, a transfer
function resolved from a name or a callable, the covariance sanity check, and
the three ways a correction factor is formed.
"""

import logging

import numpy as np
import xarray as xr

from ..utils import resolve_variable

logger = logging.getLogger(__name__)

def correction_factor_from_transfer_function(ds, transfer_function, unattenuated='cospectrum_w_t_sonic_norm'):
    """Flux-recovery factor: the reference cospectrum's area, damped and undamped.

    ``NaN`` where the ratio cannot be formed, rather than a number. xarray's
    ``sum`` skips missing values, so a transfer function that is entirely missing
    -- which is exactly what a declined fit produces -- sums to zero rather than
    to nothing, which would make the ratio ``+inf``. An infinite correction factor
    multiplied into a flux is worse than a missing one:
    :func:`~....main._apply_spectral_factor` fills a missing factor with 1 and
    leaves the flux uncorrected, and the ``scf_`` variable then says the factor
    could not be formed, whereas an infinite one destroys the flux and announces
    nothing. Same rule, and the same test, as
    :func:`correction_factor_bracketed` and EddyPro's own integral.
    """
    unattenuated = resolve_variable(unattenuated, ds)
    transfer_function = resolve_variable(transfer_function, ds)

    denominator = (unattenuated * transfer_function).sum('frequency')
    usable = np.isfinite(denominator) & (denominator != 0)
    return unattenuated.sum('frequency') / denominator.where(usable)


def _resolve_tf_model(model):
    """Return a transfer-function callable ``f(freq, fc, F)`` from name or pass-through."""
    if callable(model):
        return model
    if model in TRANSFER_FUNCTION_MODELS:
        return TRANSFER_FUNCTION_MODELS[model]
    raise KeyError(
        f"Unknown transfer-function model {model!r}. "
        f"Choose from {sorted(TRANSFER_FUNCTION_MODELS.keys())} or pass a callable.")


def simpson_weights(values):
    """Apply Simpson 1/3 weighting along a 1D array.

    Computes ``(f[i-1] + f[i+1] + 4*f[i]) / 6`` with periodic edges (the edge
    values are meant to be dropped by the caller).

    Ported from ``Simpson`` in FreqCor (``src/FREQCOR_functions.py``) by
    A. Faurès and B. Heinesch, ULiège — https://github.com/BernardHeinesch/FreqCor.
    """
    v = np.asarray(values, dtype=float)
    return (np.roll(v, 1) + np.roll(v, -1) + 4 * v) / 6


def simpson_integral(values, freq):
    """Integrate a 1D (co)spectrum over frequency with Simpson weighting.

    Simpson-weight the values, drop the wrapped endpoints, and sum against a
    central-difference frequency increment ``df[i] = (f[i+1] - f[i-1]) / 2``.
    NaNs are ignored.

    Reproduces the correction-factor integral of ``FREQCOR_LUT_CF`` in FreqCor
    (``src/FREQCOR_LUT_CF.py``) by A. Faurès and B. Heinesch, ULiège, after
    M. Aubinet — https://github.com/BernardHeinesch/FreqCor (Apache-2.0).

    Parameters
    ----------
    values, freq : array-like
        Co-spectral values and their natural frequencies (Hz), same length.

    Returns
    -------
    float
        The integral, or NaN if fewer than three samples are supplied.
    """
    v = np.asarray(values, dtype=float)
    f = np.asarray(freq, dtype=float)
    if v.size < 3:
        return np.nan
    weighted = simpson_weights(v)[1:-1]
    df = (f[2:] - f[:-2]) / 2
    return np.nansum(weighted * df)


def transfer_function_cov_check(tf, freq, window=(0.021, 0.34), varclim=1.0):
    """Quality-check a transfer function via its coefficient of variation.

    Over the frequency ``window`` the normalisation factor of a valid transfer
    function should be roughly constant; the curve is discarded when the
    coefficient of variation (std / mean) over that band exceeds ``varclim``.
    A perfectly flat transfer function (cov == 0) is the ideal case and stays
    valid.

    Ported from ``check_var_tf`` in FreqCor (``src/FREQCOR_functions.py``) by
    A. Faurès and B. Heinesch, ULiège, after M. Aubinet —
    https://github.com/BernardHeinesch/FreqCor (Apache-2.0).

    Parameters
    ----------
    tf, freq : array-like
        Transfer-function values and their frequencies (Hz).
    window : tuple of float, optional
        ``(fmin, fmax)`` band (Hz) over which the check is performed.
    varclim : float, optional
        Maximum allowed coefficient of variation.

    Returns
    -------
    valid : bool
        Whether the transfer function passes the check.
    cov : float
        The coefficient of variation over the window.
    """
    tf = np.asarray(tf, dtype=float)
    f = np.asarray(freq, dtype=float)
    band = (f >= window[0]) & (f <= window[1])
    if not np.any(band):
        return False, np.nan
    mean = np.nanmean(tf[band])
    cov = np.nanstd(tf[band]) / mean if mean else np.nan
    # A constant transfer function (cov == 0) is the ideal case and stays valid;
    # only an unstable normalisation (cov > varclim) or a degenerate, non-positive
    # band mean (cov < 0 or non-finite) is rejected.
    valid = bool(np.isfinite(cov) and 0 <= cov <= varclim)
    return valid, float(cov)


def fit_cutoff_frequency(tf, freq, model='lorentzian', fit_window=None,
                         p0=(1.0, 1.0), bounds=((0.0, 0.5), (np.inf, 1.5))):
    """Fit a transfer-function model to derive the cut-off frequency.

    Fits ``model(freq, fc, F)`` to the empirical transfer function and returns
    the cut-off frequency ``fc``, the normalisation ``F`` and their 1-sigma
    uncertainties (from the fit covariance). Negative and non-finite transfer
    values are discarded before fitting.

    Ported from the fitting core of ``FREQCOR_cof`` in FreqCor
    (``src/FREQCOR_cof.py``) by A. Faurès and B. Heinesch, ULiège, after
    M. Aubinet — https://github.com/BernardHeinesch/FreqCor (Apache-2.0).
    The defaults (``p0``, ``bounds``) are FreqCor's.

    Parameters
    ----------
    tf, freq : array-like
        Empirical transfer function (attenuated/unattenuated ratio) and its
        frequencies (Hz).
    model : str or callable, optional
        Transfer-function model name (``'lorentzian'``, ``'gaussian'``,
        ``'lorentzian_peltola'``) or a callable ``f(freq, fc, F)``.
    fit_window : tuple of float, optional
        ``(fmin, fmax)`` frequency band (Hz) to restrict the fit. ``None`` uses
        all frequencies.
    p0 : tuple, optional
        Initial guess ``(fc, F)``.
    bounds : tuple, optional
        ``((fc_lo, F_lo), (fc_hi, F_hi))`` parameter bounds.

    Returns
    -------
    dict
        ``{'fc', 'F', 'fc_unc', 'F_unc'}``; values are NaN when the fit fails.
    """
    model_fn = _resolve_tf_model(model)
    x = np.asarray(freq, dtype=float)
    y = np.asarray(tf, dtype=float)

    if fit_window is not None:
        sel = (x >= fit_window[0]) & (x <= fit_window[1])
        x, y = x[sel], y[sel]

    y = np.where(y < 0, np.nan, y)
    valid = np.isfinite(x) & np.isfinite(y)

    failed = {'fc': np.nan, 'F': np.nan, 'fc_unc': np.nan, 'F_unc': np.nan}
    if np.count_nonzero(valid) < 3:
        return failed

    def _model(f, fc, F):
        return model_fn(f, fc, F)

    try:
        popt, pcov = curve_fit(_model, x[valid], y[valid], p0=p0, bounds=bounds)
    except (RuntimeError, ValueError) as exc:
        logger.warning("Cut-off frequency fit failed: %s", exc)
        return failed

    errs = np.sqrt(np.diag(pcov))
    return {'fc': float(popt[0]), 'F': float(popt[1]),
            'fc_unc': float(errs[0]), 'F_unc': float(errs[1])}


def correction_factor_bracketed(ideal, freq, fc, fc_unc=0.0, model='lorentzian'):
    """Correction factor from a cut-off frequency, with confidence brackets.

    Builds the model transfer function at ``fc`` (and at ``fc ± fc_unc``),
    degrades the ideal cospectrum with it, and forms the area ratio
    ``∫ ideal / ∫ (ideal · TF)`` using :func:`simpson_integral`. Because the
    transfer function attenuates, the factor is ``>= 1``.

    Ported from the per-half-hour correction-factor computation of
    ``FREQCOR_LUT_CF`` in FreqCor (``src/FREQCOR_LUT_CF.py``) by A. Faurès and
    B. Heinesch, ULiège, after M. Aubinet —
    https://github.com/BernardHeinesch/FreqCor (Apache-2.0).

    Parameters
    ----------
    ideal, freq : array-like
        Ideal (reference) cospectrum and its frequencies (Hz).
    fc : float
        Cut-off frequency (Hz).
    fc_unc : float, optional
        1-sigma uncertainty on ``fc`` used to build the brackets.
    model : str or callable, optional
        Transfer-function model (see :func:`fit_cutoff_frequency`).

    Returns
    -------
    dict
        ``{'M', 'L', 'H'}``: central factor, low bracket (``fc + fc_unc``) and
        high bracket (``fc - fc_unc``). Brackets are NaN when ``fc`` is invalid.
    """
    model_fn = _resolve_tf_model(model)
    ideal = np.asarray(ideal, dtype=float)
    f = np.asarray(freq, dtype=float)
    numerator = simpson_integral(ideal, f)

    def _cf(fc_value):
        if not np.isfinite(fc_value) or fc_value <= 0:
            return np.nan
        tf = model_fn(f, fc_value, 1)
        denominator = simpson_integral(ideal * tf, f)
        if not np.isfinite(denominator) or denominator == 0:
            return np.nan
        return numerator / denominator

    return {'M': _cf(fc), 'L': _cf(fc + fc_unc), 'H': _cf(fc - fc_unc)}
