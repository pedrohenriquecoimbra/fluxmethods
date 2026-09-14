"""Spectral corrections fitted to the spectra this site actually measured.

Each method here needs the measured power spectra or cospectra of the period --
``requires_spectra=True`` on every registration -- because the loss it describes
is estimated by fitting a transfer function to them rather than derived from the
instrument's geometry. That is the whole difference from :mod:`.analytic`, and it
is why these carry the fitting machinery (:func:`fit_transfer_function`, the
models in :mod:`.fitting_models`, the ensemble averaging in :mod:`.commons`)
while those carry none of it.

There is no facade here: these are the functions, and what to do with the
numbers is the caller's business.
"""

import logging

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from . import commons
from .commons import correction_factor_from_transfer_function
from .fitting_models import lorentzian, transfer_function_generator
from ..utils import resolve_variable

logger = logging.getLogger(__name__)

#: The variable a fitted analytic transfer function is stored under.
ANALYTIC_TRANSFER_FUNCTION = 'transfer_function_analytical'


def ibrom_et_al_2007(ds, var='co2', attenuated=None, unattenuated=None,
                     prescribed_fc=None, **kwargs):
    """Ibrom et al. (2007) empirical low-pass factor: a Lorentzian fitted to
    the measured gas / sonic-temperature power-spectral ratio.

    The default inputs are the ``power_spectrum_<v>_norm_binned_ensemble``
    variables because those are what :func:`prep_for_correction` builds.
    ``attenuated`` and ``unattenuated`` are explicit parameters, not swallowed
    keywords, so that a configuration naming a variable nothing produces can be
    corrected rather than only refused (``var`` picks the
    gas when they are not given). The remaining keywords stay swallowed on
    purpose: the apply facade forwards every config-block key to every method
    in a chain, and the model is pinned to the Lorentzian Ibrom et al. fit.

    ``prescribed_fc`` substitutes the fitted parameter: a cut-off frequency
    (scalar, or a per-date :class:`xarray.DataArray`) taken from elsewhere --
    e.g. another engine's own spectral assessment -- evaluated through exactly
    the factor integral the fitted path ends in. That is what lets the cut-off
    fitting and the factor evaluation be compared as separate steps.
    """
    if prescribed_fc is not None:
        fc = (prescribed_fc if isinstance(prescribed_fc, xr.DataArray)
              else xr.DataArray(float(prescribed_fc)))
        tf = lorentzian(ds['frequency'], fc)
        ds['scf_param_fc'] = fc
        ds['scf_param_fc'].attrs.update({
            'description': 'prescribed Lorentzian cut-off frequency',
            'units': 'Hz',
            'method': 'prescribed',
        })
        ds['transfer_function_fit'] = tf
        ds['transfer_function_used'] = tf
        ds['scf_ibrom_et_al_2007'] = correction_factor_from_transfer_function(
            ds, tf)
        return ds

    ds = generic_experimental(
        ds,
        attenuated=attenuated or f'power_spectrum_{var}_norm_binned_ensemble',
        unattenuated=(unattenuated
                      or 'power_spectrum_t_sonic_norm_binned_ensemble'),
        model_function=lorentzian,
    )

    ds = ds.rename({
        'scf_experimental': 'scf_ibrom_et_al_2007',
        'scf_param_0': 'scf_param_fc',
    })
    ds['scf_param_fc'].attrs.update({
        'description': 'Lorentzian fit cut-off frequency parameter per date',
        'units': 'Hz',
        'method': 'Nonlinear least squares fit using scipy.curve_fit'
    })
    return ds


def generic_experimental(
        ds,
        # The defaults name what prep_for_correction actually builds.
        attenuated='power_spectrum_co2_norm_binned_ensemble',
        unattenuated='power_spectrum_t_sonic_norm_binned_ensemble',
        model_function=lorentzian,
        **kwargs):
    unattenuated = resolve_variable(unattenuated, ds)
    attenuated = resolve_variable(attenuated, ds)

    # Calculate transfer function
    transfer_function = (attenuated / unattenuated)

    ds = fit_transfer_function(ds, transfer_function,
                                model_function=model_function, **kwargs)
    tf = ds['transfer_function_used']

    ds[f'scf_experimental'] = correction_factor_from_transfer_function(
        ds, tf)
    return ds


def fully_analytical(
        ds,
        **kwargs):
    tf = (
        transfer_function_generator(ds, freq='frequency', model='horst_1997', α=2) *
        transfer_function_generator(
            ds, model='sonic_path_averaging', freq='normalized_frequency')
    )
    ds = ds.assign({ANALYTIC_TRANSFER_FUNCTION: tf})

    ds[f'scf_analytical'] = correction_factor_from_transfer_function(
        ds, tf)
    return ds


def cutoff_lut(
        ds,
        attenuated='cospectrum_w_co2_norm_binned_ensemble',
        reference='cospectrum_w_t_sonic_norm_binned_ensemble',
        freq='frequency_bin',
        model='lorentzian',
        cov_window=(0.021, 0.34),
        fit_window=(0.021, 10.0),
        varclim=1.0,
        **kwargs) -> xr.Dataset:
    """Cut-off-frequency correction factors, after FreqCor.

    For each atmospheric stability class this routine forms the empirical
    transfer function (attenuated / reference binned-ensemble cospectra), fits a
    parametric model to extract the cut-off frequency and its uncertainty, then
    turns that cut-off into a flux correction factor (with confidence brackets)
    by degrading the reference cospectrum and integrating.

    This is the processing chain of FreqCor (``FREQCOR_cof`` →
    ``FREQCOR_LUT_cof`` → ``FREQCOR_LUT_CF``) expressed on this package's
    binned ensembles. FreqCor — https://github.com/BernardHeinesch/FreqCor
    (Apache-2.0) — was written by Ariane Faurès and Bernard Heinesch
    (University of Liège, Gembloux Agro-Bio Tech), after an original MATLAB
    version by Marc Aubinet; please cite it when using this routine.

    The inputs must already be prepared (see :func:`prep_for_correction`), which
    builds the ``*_norm_binned_ensemble`` variables and the ``frequency_bin``
    coordinate. Results are stored as ``cutoff_frequency_<model>`` and
    ``scf_cutoff_<model>`` (plus ``_low``/``_high`` brackets), indexed by
    ``class_stability`` when that dimension is present.

    Parameters
    ----------
    ds : xarray.Dataset
        Prepared dataset.
    attenuated, reference : str
        Names of the attenuated (gas) and reference (sonic) binned-ensemble
        cospectra.
    freq : str
        Name of the frequency coordinate used for the fit and integration.
    model : str or callable
        Transfer-function model (``'lorentzian'``, ``'gaussian'``,
        ``'lorentzian_peltola'``, or a callable).
    cov_window, fit_window : tuple of float
        Frequency bands (Hz) for the transfer-function quality check and for the
        cut-off fit, respectively.
    varclim : float
        Maximum coefficient of variation accepted by the quality check. A
        stability class whose transfer function fails it is reported with a
        ``NaN`` cut-off frequency and a unit correction factor, and logged.
    """
    label = model if isinstance(model, str) else getattr(model, '__name__', 'model')

    atn = resolve_variable(attenuated, ds)
    ref = resolve_variable(reference, ds)
    f = np.asarray(ds[freq].values, dtype=float)

    classes = (list(atn['class_stability'].values)
               if 'class_stability' in atn.dims else [None])

    fc, fc_unc, cf_m, cf_lo, cf_hi, cov = ([] for _ in range(6))
    for cls in classes:
        atn_v = atn.sel(class_stability=cls).values if cls is not None else atn.values
        ref_v = ref.sel(class_stability=cls).values if cls is not None else ref.values
        with np.errstate(divide='ignore', invalid='ignore'):
            tf = np.asarray(atn_v, dtype=float) / np.asarray(ref_v, dtype=float)

        valid, cov_value = commons.transfer_function_cov_check(
            tf, f, cov_window, varclim)
        if not valid:
            # The verdict has to be read: FreqCor discards such a curve, and
            # discarding it here means reporting no cut-off and a unit factor
            # rather than a factor built from a transfer function this very check
            # has just declared unusable.
            logger.warning(
                "cutoff_lut: transfer function for stability class %r rejected "
                "(coefficient of variation %.3g over %s Hz exceeds varclim=%.3g); "
                "no correction is applied for this class.",
                cls, cov_value, cov_window, varclim)
            fc.append(np.nan)
            fc_unc.append(np.nan)
            cf_m.append(1.0)
            cf_lo.append(1.0)
            cf_hi.append(1.0)
            cov.append(cov_value)
            continue

        fit = commons.fit_cutoff_frequency(tf, f, model=model, fit_window=fit_window)
        factor = commons.correction_factor_bracketed(
            ref_v, f, fit['fc'], fit['fc_unc'], model=model)

        fc.append(fit['fc'])
        fc_unc.append(fit['fc_unc'])
        cf_m.append(factor['M'])
        cf_lo.append(factor['L'])
        cf_hi.append(factor['H'])
        cov.append(cov_value)

    if classes == [None]:
        def wrap(values):
            return xr.DataArray(values[0])
    else:
        coords = {'class_stability': classes}
        def wrap(values):
            return xr.DataArray(values, dims='class_stability', coords=coords)

    ds[f'cutoff_frequency_{label}'] = wrap(fc)
    ds[f'cutoff_frequency_{label}'].attrs.update({
        'description': f'Cut-off frequency from {label} transfer-function fit',
        'units': 'Hz', 'model': label})
    ds[f'cutoff_frequency_{label}_unc'] = wrap(fc_unc)
    ds[f'scf_cutoff_{label}'] = wrap(cf_m)
    ds[f'scf_cutoff_{label}'].attrs.update({
        'description': f'Spectral correction factor from {label} cut-off frequency',
        'model': label})
    ds[f'scf_cutoff_{label}_low'] = wrap(cf_lo)
    ds[f'scf_cutoff_{label}_high'] = wrap(cf_hi)
    ds[f'transfer_function_cov_{label}'] = wrap(cov)
    return ds


def fit_transfer_function(
        ds,
        transfer_function,
        input_core_dims='frequency_bin',
        group_dims='group',
        model_function=lorentzian,
        model_prior=[0.1, 1],
        model_fit_window=()
        ):
    """
    Fit a model (e.g., Lorentzian) to transfer functions grouped by 'group_dim'.

    Parameters:
        ds: xarray.Dataset with dimension `group_dim` and `input_core_dims`
        transfer_function: DataArray with dims (group_dim, input_core_dims)
        input_core_dims: name of the frequency dimension
        group_dim: dimension over which to apply fitting (e.g., 'group')
        model_function: callable (e.g., lorentzian)
        model_prior: initial guess for fitting
        model_fit_window: tuple (start, end) for fitting window in frequency
    """
    freq = ds[input_core_dims]

    # If slicing the model fit window
    if model_fit_window:
        start, end = model_fit_window
        freq = freq.sel(**{input_core_dims: slice(start, end)})
        transfer_function = transfer_function.sel(
            **{input_core_dims: slice(start, end)})

    # The attenuated and unattenuated ensembles carry the same unit, so their
    # ratio is dimensionless; strip a pint wrapper once here rather than let
    # every vectorised curve_fit call shed a UnitStrippedWarning.
    if hasattr(transfer_function, 'pint') and transfer_function.pint.units is not None:
        transfer_function = transfer_function.pint.dequantify()

    # Empty logarithmic bins arrive as NaN by construction: ``bin_average``
    # reports a band nothing was measured in as missing rather than flat, and
    # at the bottom of a typical grid the log bands are narrower than the FFT's
    # linear spacing, so a few are *always* empty. Refusing to fit through any
    # NaN therefore refused every ensemble the prep stage can build. Dropping
    # the empty bins before fitting is the standard practice (the FreqCor port
    # in ``commons.fit_cutoff_frequency`` masks them the same way): the fit is
    # over the measured bands only, per group, and a curve left with fewer
    # finite samples than parameters plus one still declines to NaN parameters.
    # The reconstruction below evaluates the model at those NaN parameters, so
    # a declined group's curve is missing on the ``frequency`` axis and
    # ``correction_factor_from_transfer_function`` turns it into a missing
    # factor rather than an infinite one.
    if bool(transfer_function.isnull().any()):
        logger.warning('NaNs found in transfer function (empty frequency '
                       'bins); fitting through the finite samples only.')

    def fit_model(y, x, prior=model_prior, fc=model_function):
        # Fitting function that returns one parameter (e.g. gamma)
        y = np.asarray(y, dtype=float)
        x = np.asarray(x, dtype=float)
        valid = np.isfinite(x) & np.isfinite(y)
        if np.count_nonzero(valid) <= len(prior):
            return tuple(np.nan for _ in prior)
        popt, _ = curve_fit(fc, x[valid], y[valid], p0=prior)
        return popt[0], popt[1]  # or popt if returning all

    opt_params = xr.apply_ufunc(
        fit_model,
        transfer_function,          # shape: (date, freq)
        freq,       # shape: (freq,)
        input_core_dims=[[input_core_dims], [input_core_dims]],
        output_core_dims=[[] for _ in model_prior],
        output_dtypes=[float for _ in model_prior],
        vectorize=True,
        dask='parallelized' if transfer_function.chunks else False,
    )

    for i, op in enumerate(opt_params):
        ds[f'scf_param_{i}']= op
    ds['scf_param_0'].attrs.update({
        'model': model_function.__name__,
        'prior': model_prior,
    })

    # Reconstruct the fitted function per group
    ds['transfer_function_fit'] = xr.apply_ufunc(
        model_function,
        ds.frequency,
        opt_params[0],
        input_core_dims=[['frequency'], []],
        output_core_dims=[['frequency']],
        vectorize=True,
        dask='parallelized' if transfer_function.chunks else False,
        output_dtypes=[float]
    )
    ds[f'transfer_function_used'] = ds[f'transfer_function_fit']
    return ds
