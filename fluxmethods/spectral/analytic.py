"""Spectral corrections computed from the instrument, not from the site's spectra.

Every method here answers the same question -- what fraction of the flux did this
instrument and this averaging lose -- from a *model* cospectrum and the geometry
of the sensor: path lengths, tube dimensions, intake separation, the averaging
period. None of them reads a measured spectrum, which is what
``requires_spectra=False`` says on each registration, and it is why they run on a
period that has none.

The facade that registers them, decides which may be selected together and
applies them to a dataset is :mod:`.main`; the methods that fit a transfer
function to this site's own measured spectra are :mod:`.measured`.
"""

import logging

import numpy as np
import xarray as xr

from .sensor_table import sensor_geometry

logger = logging.getLogger(__name__)

#: Below this wind speed a period's advective time scales stop being meaningful
#: and the analytic transfer functions are not applied [m s-1].
MIN_WIND_SPEED = 0.3


def _known_sensor(model):
    """Published parameters for an instrument model, or ``None``.

    Setups name instruments with a trailing index (``hs_50_1``), so the longest
    matching prefix is taken.
    """
    name = str(model or '').strip().lower()
    table = sensor_geometry()
    matches = [k for k in table if name.startswith(k)]
    return table[max(matches, key=len)] if matches else None


def _find_tube(ds):
    """Closed-path analyser geometry, from the setup ``Instruments``.

    Returns ``{'L': length_m, 'r_mm': bore_radius_mm, 'lpm': flow_l_per_min,
    'separation': m}`` for the first instrument block that declares a
    ``tube_length`` (a closed-path analyser), or ``None`` for an open-path setup
    (no tube, no attenuation). ``separation`` is the 3-D distance between the
    analyser intake and the sonic path, from the block's ``*_separation`` fields
    (declared in centimetres, as EddyPro's metadata does); it drives the
    sensor-separation transfer function.

    Lengths follow EddyPro's metadata units: ``tube_length`` and the separations in
    **centimetres**, ``tube_diameter`` in **millimetres**. The tube length is
    converted here for the same reason the separations are.
    """
    instruments = ds.attrs.get('Instruments', {}) if hasattr(ds, 'attrs') else {}
    for block in (instruments or {}).values():
        if not hasattr(block, 'get') or not block.get('tube_length'):
            continue
        try:
            L = float(block['tube_length']) / 100.0        # cm -> m
            diam = float(block.get('tube_diameter', 0.0))  # mm, as the model wants
            lpm = float(block.get('tube_flowrate', 0.0))
        except (TypeError, ValueError):
            continue
        if not (L > 0 and diam > 0 and lpm > 0):
            continue
        offsets = {}
        for key in ('northward_separation', 'eastward_separation', 'vertical_separation'):
            try:
                offsets[key] = float(block.get(key, 0.0) or 0.0)
            except (TypeError, ValueError):
                offsets[key] = 0.0
        # Horizontal and vertical offsets attenuate independently (Moore 1986), so
        # they are kept apart rather than collapsed into one 3-D distance. The two
        # horizontal components combine into a single crosswind distance without
        # any wind-direction rotation, as EddyPro's hsep does.
        horizontal = float(np.hypot(offsets['northward_separation'],
                                    offsets['eastward_separation'])) / 100.0   # cm -> m
        vertical = abs(offsets['vertical_separation']) / 100.0

        # The analyser's own optical path and response, on the same footing as the
        # anemometer's: a declared value that passes a physical test, else the
        # published specification for the model.
        known = _known_sensor(block.get('model') or '') or {}

        def _declared(key):
            raw = block.get(key)
            if raw is None or raw == '':
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                logger.warning("analyser %r declares %s = %r, which is not a "
                               "number; ignored.", block.get('model'), key, raw)
                return None

        declared_path = _declared('vpath_length')
        declared_tau = _declared('tau')
        opt_path = (declared_path if (declared_path and 0.01 < declared_path < 0.5)
                    else known.get('vpath_length'))
        opt_tau = declared_tau if (declared_tau and 0 < declared_tau < 1.0) else known.get('tau')
        if declared_path is not None and not (0.01 < declared_path < 0.5):
            logger.warning(
                "analyser %r declares vpath_length = %s m, outside the plausible "
                "0.01-0.5 m; using %s.", block.get('model'), declared_path,
                opt_path if opt_path is not None else "nothing")
        if declared_tau is not None and not (0 < declared_tau < 1.0):
            logger.warning(
                "analyser %r declares tau = %s s, outside the plausible (0, 1) s; "
                "using %s.", block.get('model'), declared_tau,
                opt_tau if opt_tau is not None else "nothing")

        return {'L': L, 'r_mm': diam / 2.0, 'lpm': lpm,
                'separation': horizontal, 'vertical_separation': vertical,
                'path_length': opt_path, 'tau': opt_tau}
    return None


def _frequency_grid(acq, nfreq, freq_range=None, highpass_period_s=None):
    """Integration grid [Hz].

    Defaults to 1e-3 Hz up to the Nyquist frequency of the acquisition. A wider
    range can be requested -- the reference implementation integrates 1/5000 Hz to
    100 Hz -- which matters because the factor is a ratio of integrals and the tube
    keeps attenuating above Nyquist, so where the integral is truncated changes it.

    The 1e-3 Hz floor suits the low-pass losses, which live at the top of the band.
    A high-pass term does not: block averaging over ``T`` rolls off around ``1/T``,
    which for a half-hour window is 5.6e-4 Hz -- *below* that floor, where the
    transfer function has already recovered to 0.989. Integrating it on the default
    grid therefore sees almost none of the flux it is meant to restore. Passing
    ``highpass_period_s`` lowers the floor to ``0.2/T``, the point at which the
    filter has removed about 88\\% of the flux, which is the same bound
    :func:`block_average_highpass` picks for the same filter -- one physical filter
    should be integrated over one band, whichever method folds it in. An explicit
    ``freq_range`` always wins; the caller has then said where to integrate.
    """
    import numpy as np
    if freq_range:
        lo, hi = freq_range
    else:
        lo, hi = 1e-3, 0.5 * acq
        try:
            T = float(highpass_period_s)
        except (TypeError, ValueError):
            T = None
        if T is not None and np.isfinite(T) and T > 0:
            lo = min(lo, 0.2 / T)
    return np.logspace(np.log10(lo), np.log10(hi), int(nfreq))


def _highpass_transfer(F, minutes):
    """Block-averaging high-pass transfer function on the grid ``F``.

    ``H(f) = 1 - [sin(pi f T)/(pi f T)]^2`` for an averaging period of ``minutes``.
    Returned so it can be folded into the *same* integral as the low-pass terms:
    a correction factor is a ratio of integrals, so integrating the combined
    band-pass once is not the same number as multiplying two separately computed
    ratios, and the reference implementation combines them.
    """
    import numpy as np
    try:
        T = float(minutes) * 60.0
    except (TypeError, ValueError):
        T = 1800.0
    if not np.isfinite(T) or T <= 0:
        T = 1800.0
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.pi * np.asarray(F, dtype=float) * T
        H = 1.0 - (np.sin(x) / x) ** 2
    return np.clip(np.where(np.isfinite(H), H, 1.0), 0.0, 1.0)


def _averaging_minutes(ds):
    options = ds.attrs.get('Options', {}) or {}
    return (options.get('process_duration')
            or (ds.attrs.get('Files', {}) or {}).get('fileduration') or 30)


def _averaging_seconds(ds):
    """The averaging period in seconds, falling back the way the filter does.

    The high-pass corner is set by this period, so the integration grid and the
    transfer function have to agree on it down to the fallback.
    """
    import numpy as np
    try:
        T = float(_averaging_minutes(ds)) * 60.0
    except (TypeError, ValueError):
        return 1800.0
    return T if (np.isfinite(T) and T > 0) else 1800.0


def _cospectrum_shape(model, F, zmd, u_i, zL_i):
    """A similarity cospectrum evaluated on the frequency grid ``F`` for one period.

    Shapes the model by the period's mean wind and measurement height through the
    natural frequency ``n = f (z-d) / u``, and hands the stability to the model,
    which may or may not consume it: ``moncrieff_1997`` does, while the default
    ``kaimal_1972`` is the neutral form and ignores it (see
    :func:`~.fitting_models.cospec_kaimal_1972`). Non-finite or negative lobes are
    zeroed so the integrals below are well posed.
    """
    import numpy as np
    n = F * zmd / max(float(u_i), MIN_WIND_SPEED)
    try:
        form = np.asarray(model(n, zL_i), dtype=float)
    except (TypeError, ValueError):
        form = np.array([model(nk, zL_i) for nk in n], dtype=float)
    return np.where(np.isfinite(form) & (form > 0), form, 0.0)


def _sonic_geometry(ds, acq_freq, path_length=None):
    """``(path_length [m], tau [s])`` for the anemometer.

    The acoustic path is the *vertical* one, which sets the line averaging of both
    ``w`` and the sonic temperature (EddyPro normalises by ``vpath_length`` too).

    Each value is taken from the setup only when it is self-consistent, and
    otherwise from the instrument's published specification (the bundled files
    behind :func:`sensor_table.sensor_geometry`). The two consistency tests are
    physical, not tuned:

    * a path outside a few centimetres to half a metre is not an anemometer head;
    * a response slower than the acquisition rate could not sustain that rate, so
      such a number is metadata that was never filled in.

    A component with neither a usable declaration nor a known model is returned as
    ``None``, leaving the caller to decline rather than invent one.
    """
    declared_path = declared_tau = model = None
    for name, block in (ds.attrs.get('Instruments', {}) or {}).items():
        if not hasattr(block, 'get'):
            continue
        model = block.get('model') or name

        def _number(key):
            raw = block.get(key)
            if raw is None or raw == '':
                return None
            try:
                return float(raw)
            except (TypeError, ValueError):
                logger.warning("anemometer %r declares %s = %r, which is not a "
                               "number; ignored.", model, key, raw)
                return None

        declared_path, declared_tau = _number('vpath_length'), _number('tau')
        break

    known = _known_sensor(model) or {}
    usable_path = declared_path if (declared_path and 0.01 < declared_path < 0.5) else None
    usable_tau = (declared_tau if (declared_tau and acq_freq and 1.0 / declared_tau >= acq_freq)
                  else None)
    if declared_path is not None and usable_path is None and not path_length:
        logger.warning(
            "anemometer %r declares vpath_length = %s m, outside the plausible "
            "0.01-0.5 m; using %s.", model, declared_path,
            known.get('vpath_length') if known.get('vpath_length') is not None
            else "nothing -- methods that need it will skip")
    if declared_tau is not None and usable_tau is None:
        logger.warning(
            "anemometer %r declares tau = %s s, which cannot sustain the %s Hz "
            "acquisition; using %s.", model, declared_tau, acq_freq,
            known.get('tau') if known.get('tau') is not None
            else "nothing -- the acquisition frequency stands in")
    return (path_length or usable_path or known.get('vpath_length'),
            usable_tau or known.get('tau'))


def _period_geometry(ds, wind_speed, height, stability):
    """``(U, zmd, zL)`` for the spectral methods, or ``None`` if unavailable."""
    import numpy as np

    def _mag(name):
        if name not in ds:
            return None
        da = ds[name]
        try:
            return np.ravel(np.asarray(da.pint.magnitude, dtype=float))
        except (AttributeError, TypeError):
            return np.ravel(np.asarray(da.values, dtype=float))

    U, z = _mag(wind_speed), _mag(height)
    if U is None or z is None:
        return None
    zL = _mag(stability)
    return U, (float(z[0]) if z.size else np.nan), (np.zeros_like(U) if zL is None else zL)


def block_average_highpass(ds, cospectrum='kaimal_1972', wind_speed='wind_speed',
                           height='z-d', stability='z_L', averaging_minutes=None,
                           nfreq=400, **kwargs):
    """Recover the low-frequency flux removed by block averaging.

    Averaging over a finite period ``T`` and subtracting the block mean is a
    *high-pass* filter: the very eddies that are too slow to be resolved within the
    window are discarded along with the mean. Its transfer function is

        H(f) = 1 - [sin(pi f T) / (pi f T)]^2

    which is 0 at ``f = 0`` (the mean is removed) and approaches 1 well inside the
    window. The recovery factor is the usual area ratio of the model cospectrum
    with and without that filter, so it is ``>= 1``.

    Unlike the closed-path losses it applies to *every* flux -- the sonic heat
    flux included, since the loss is a property of the averaging window and the
    eddy scale, not of the instrument. It is therefore registered with
    ``applies_to='all'``.

    It answers the same question as EddyPro's ``lf_meth``, but it is **not** that
    correction and must not be selected alongside the transcribed EddyPro methods.
    EddyPro folds its high-pass transfer function into the *same* band-pass
    integral as the high-frequency losses, so its low-frequency term already lives
    inside ``bandpass_moncrieff_1997_gas`` / ``_heat`` / ``_momentum`` (their
    ``high_pass`` argument). On the FR-Gri sample that
    term is worth a median 0.215 % on the heat factor and 0.239 % on the gas one,
    while this routine -- its own grid, its own cospectrum, a separate ratio of
    integrals rather than one combined one -- returns a median 0.332 %. Selecting
    both put the heat flux 0.216 % and the CO2 flux 0.490 % above EddyPro's
    corrected columns, against 0.021 % and 0.285 % with the three methods alone.

    The filter is set by how the mean was removed, so this routine is tied to
    **block averaging** and checks the configured detrending method rather than
    assuming it. Linear detrending is a stronger high-pass filter with a different
    transfer function (Rannik and Vesala, 1999); rather than apply the
    block-average curve to it and quietly under-correct, an unsupported detrending
    method is skipped with a warning -- the same exact-or-refuse rule used when
    emitting a foreign config.

    ``stability`` is read and handed to the cospectrum, but the default
    ``kaimal_1972`` is the neutral form and ignores it
    (:func:`~.fitting_models.cospec_kaimal_1972`), so the default recovery factor
    varies with wind and height but not with ``z/L``.

    References
    ----------
    Moncrieff, J., Clement, R., Finnigan, J., and Meyers, T. (2004). Averaging,
    detrending, and filtering of eddy covariance time series. In: Handbook of
    Micrometeorology, Springer, 7-31.
    Rannik, U. and Vesala, T. (1999). Autoregressive filtering versus linear
    detrending in estimation of fluxes by the eddy covariance method.
    Boundary-Layer Meteorology, 91, 259-280.
    """
    import numpy as np
    from .fitting_models import COSPECTRAL_MODELS

    # The high-pass filter is a property of how the mean was removed, so only apply
    # it to the detrending it actually describes.
    detrending = ((ds.attrs.get('Corrections', {}) or {}).get('detrending', {}) or {})
    method = (detrending.get('method') if hasattr(detrending, 'get') else None) or 'ba'
    if str(method) not in ('ba', 'block_average'):
        logger.warning(
            "block_average_highpass describes block averaging, but detrending is %r; "
            "its high-pass transfer function differs (Rannik & Vesala 1999) and is "
            "not implemented, so no low-frequency correction is applied.", method)
        return ds

    geom = _period_geometry(ds, wind_speed, height, stability)
    if geom is None:
        logger.warning("block_average_highpass: missing %r or %r; skipping.",
                       wind_speed, height)
        return ds
    U, zmd, zL = geom

    if averaging_minutes is None:
        options = ds.attrs.get('Options', {}) or {}
        averaging_minutes = (options.get('process_duration')
                             or (ds.attrs.get('Files', {}) or {}).get('fileduration') or 30)
    try:
        T = float(averaging_minutes) * 60.0
    except (TypeError, ValueError):
        T = 1800.0
    if not np.isfinite(T) or T <= 0:
        T = 1800.0

    acq = float(ds.attrs.get('acquisition_frequency', 20.0)) or 20.0
    F = np.logspace(np.log10(0.2 / T), np.log10(0.5 * acq), int(nfreq))
    lnF = np.log(F)
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.pi * F * T
        H = 1.0 - (np.sin(x) / x) ** 2
    H = np.clip(np.where(np.isfinite(H), H, 1.0), 0.0, 1.0)

    model = COSPECTRAL_MODELS[cospectrum] if isinstance(cospectrum, str) else cospectrum
    trapz = getattr(np, 'trapezoid', None) or np.trapz

    cf = np.ones_like(U, dtype=float)
    for i in range(U.size):
        zL_i = float(np.atleast_1d(zL)[i]) if np.atleast_1d(zL).size > i else 0.0
        form = _cospectrum_shape(model, F, zmd, U[i], zL_i)
        den = trapz(form * H, lnF)
        cf[i] = trapz(form, lnF) / den if den > 0 else 1.0

    dims = ds[wind_speed].dims
    ds['scf_block_average'] = xr.DataArray(cf.reshape(ds[wind_speed].shape), dims=dims)
    ds['scf_block_average'].attrs.update({
        'description': 'low-frequency correction for block averaging (high-pass loss)',
        'method': 'highpass_block_average', 'averaging_period_s': T,
        'cospectrum': cospectrum})
    return ds


def sonic_response(ds, cospectrum='kaimal_1972', wind_speed='wind_speed',
                   height='z-d', stability='z_L', path_length=None, nfreq=400,
                   highpass=False, freq_range=None, inband_block_average=False,
                   **kwargs):
    """High-frequency loss of the *sonic* heat flux (path averaging + response).

    The sonic temperature flux has none of the closed-path losses -- no tube, and no
    intake separation, since ``w`` and ``T_s`` come from the same instrument -- but
    it is not loss-free: the anemometer averages along a finite acoustic path and
    has a finite response time, both of which attenuate the cospectrum at high
    frequency. This is the sonic counterpart of :func:`analytic_tube`, and it is
    what EddyPro's ``H_scf`` mostly consists of at a low tower, where the
    low-frequency (block-averaging) loss is negligible.

    Registered with ``applies_to='heat'``, so the stage applies it to the sonic
    temperature covariance and the heat flux only, never to the gases.

    As in :func:`analytic_tube`, the model cospectrum is placed in frequency by the
    period's mean wind and measurement height. ``stability`` is read and passed to
    the cospectrum, but the default ``kaimal_1972`` is the neutral form and ignores
    it, so the default factor does not vary with ``z/L``; use ``moncrieff_1997`` if
    that dependence is wanted.
    """
    import numpy as np
    from .fitting_models import (COSPECTRAL_MODELS,
                                 transfer_function_sonic_cospectrum as _sonic_tf,
        transfer_function_block_averaging_inband as _ba_tf)

    geom = _period_geometry(ds, wind_speed, height, stability)
    if geom is None:
        logger.warning("sonic_response: missing %r or %r; skipping.", wind_speed, height)
        return ds
    U, zmd, zL = geom

    acq = float(ds.attrs.get('acquisition_frequency', 20.0)) or 20.0
    _grid = _frequency_grid(acq, nfreq, freq_range,
                            highpass_period_s=_averaging_seconds(ds) if highpass else None)
    hp = _highpass_transfer(_grid, _averaging_minutes(ds)) if highpass else 1.0
    if inband_block_average:
        hp = hp * _ba_tf(_grid, acq_freq=acq) ** 2      # w and Ts legs alike
    path_length, tau = _sonic_geometry(ds, acq, path_length)
    if path_length is None:
        logger.warning("sonic_response: no usable anemometer path length "
                       "(declared or known for this model); skipping.")
        return ds
    F = _grid
    lnF = np.log(F)
    model = COSPECTRAL_MODELS[cospectrum] if isinstance(cospectrum, str) else cospectrum
    trapz = getattr(np, 'trapezoid', None) or np.trapz

    cf = np.ones_like(U, dtype=float)
    for i in range(U.size):
        u_i = max(float(U[i]), MIN_WIND_SPEED)
        zL_i = float(np.atleast_1d(zL)[i]) if np.atleast_1d(zL).size > i else 0.0
        form = _cospectrum_shape(model, F, zmd, u_i, zL_i)
        with np.errstate(divide='ignore', invalid='ignore'):
            H = _sonic_tf(F, ws=u_i, path_length=path_length, tau=tau, acq_freq=acq) * hp
        H = np.clip(np.where(np.isfinite(H), H, 1.0), 0.0, 1.0)
        den = trapz(form * H, lnF)
        cf[i] = trapz(form, lnF) / den if den > 0 else 1.0

    dims = ds[wind_speed].dims
    ds['scf_sonic_response'] = xr.DataArray(cf.reshape(ds[wind_speed].shape), dims=dims)
    ds['scf_sonic_response'].attrs.update({
        'description': 'sonic path-averaging / dynamic-response correction factor',
        'method': 'lowpass_analytic_sonic', 'path_length_m': path_length,
        'cospectrum': cospectrum})
    return ds


def analytic_tube(ds, cospectrum='kaimal_1972', wind_speed='wind_speed',
                  height='z-d', stability='z_L', tube=None, nfreq=400,
                  tube_model='foken_et_al_2012', sonic_path_length=None,
                  highpass=False, freq_range=None, inband_block_average=False,
                  **kwargs):
    """Analytic closed-path spectral correction factor (per period).

    The physically-transparent, spectra-free correction: for each averaging period
    it integrates a *model* scalar cospectrum against the analyser's high-frequency
    transfer function and returns the flux-recovery factor

        CF = ∫ Co(f) d(ln f)  /  ∫ Co(f) · H(f) d(ln f)   ( >= 1 )

    where ``Co`` is a similarity cospectrum (Kaimal 1972 / Moncrieff 1997) shaped by
    the period's mean wind and measurement height through the natural frequency
    ``n = f (z−d) / u``, and ``H`` damps **both legs** of the cospectrum: the gas
    leg through tube attenuation (``tube_model``: Foken et al. 2012 by default, or
    the older single-term Moore/Lenschow form), IRGA path averaging (LI-COR 2009)
    and the intake-to-sonic separation (Moore 1986, often the largest term on a
    short tower); and the wind leg through the anemometer's own path averaging and
    time response. Unlike the empirical cut-off methods this needs no measured
    (co)spectra — only the per-period mean wind speed and measurement height, plus
    the stability if the chosen cospectrum uses it — so it is bounded and robust.

    The **default** cospectrum, ``kaimal_1972``, is the neutral form and carries no
    ``z/L`` dependence, so the default factor responds to wind and height but is the
    same in stable and unstable air. That is a property of the model, not an
    oversight in the plumbing: the peak-frequency normalisation that would introduce
    a stability dependence belongs to Horst's coefficients, not Kaimal's, and
    applying it here would under-correct by an order of magnitude
    (:func:`~.fitting_models.cospec_kaimal_1972` sets this out). Pass
    ``cospectrum='moncrieff_1997'`` for a shape that does vary with ``z/L``.

    The same dry-tube factor is returned for CO2 and H2O; it therefore captures the
    tube + sensor attenuation but not the extra RH-dependent H2O tube sorption,
    which a more complete registered method would add. Attaches
    ``scf_analytic_tube`` (one value per ``date``).

    An **open-path** analyser has no tube, and this routine returns unity for it --
    which is not the whole story, since path averaging, sensor separation and the
    anemometer's response still attenuate an open-path cospectrum. Those need their
    own registered method; the unit factor here is a statement that *this* routine
    does not describe that instrument, and it says so in the log.
    """
    import numpy as np
    from .fitting_models import (
        COSPECTRAL_MODELS,
        transfer_function_tube_attenuation_foken_et_al_2012 as _tube_foken,
        transfer_function_tube_attenuation_moore_lenschow as _tube_moore,
        transfer_function_irga_response as _irga_tf,
        transfer_function_sensor_separation as _sep_tf,
        transfer_function_sonic_response as _sonic_w_tf,
        transfer_function_block_averaging_inband as _ba_tf)

    _tube_tf = _tube_moore if str(tube_model).startswith(('moore', 'lenschow'))         else _tube_foken

    def _mag(name):
        if name not in ds:
            return None
        da = ds[name]
        try:
            return np.ravel(np.asarray(da.pint.magnitude, dtype=float))
        except (AttributeError, TypeError):
            return np.ravel(np.asarray(da.values, dtype=float))

    U = _mag(wind_speed)
    z = _mag(height)
    if U is None or z is None:
        logger.warning("analytic_tube: missing %r or %r; skipping.", wind_speed, height)
        return ds
    zmd = float(z[0]) if z.size else np.nan
    zL = _mag(stability)
    if zL is None:
        zL = np.zeros_like(U)

    geom = tube if tube is not None else _find_tube(ds)
    dims = ds[wind_speed].dims
    if not geom:
        # No tube declared, so this routine has nothing to say about the analyser.
        # That is *not* the same as "no correction is due": an open-path instrument
        # still loses flux to path averaging, sensor separation and the anemometer's
        # response. Returning unity silently would understate those, so say so.
        logger.warning(
            "analytic_tube found no closed-path analyser (no tube geometry declared), "
            "so its factor is 1. An open-path setup still loses high-frequency flux to "
            "path averaging, sensor separation and sonic response; use a method that "
            "describes that instrument instead of relying on this one.")
        ds['scf_analytic_tube'] = xr.DataArray(np.ones_like(U).reshape(ds[wind_speed].shape), dims=dims)
        return ds

    acq = float(ds.attrs.get('acquisition_frequency', 20.0)) or 20.0
    F = _frequency_grid(acq, nfreq, freq_range,
                        highpass_period_s=_averaging_seconds(ds) if highpass else None)
    lnF = np.log(F)
    H = _tube_tf(F, r=geom['r_mm'], L=geom['L'], lpm=geom['lpm'])
    if highpass:
        H = H * _highpass_transfer(F, _averaging_minutes(ds))
    if inband_block_average:
        # One factor per leg: the anemometer's variable and the analyser's are both
        # block-averaged by the data system.
        H = H * _ba_tf(F, acq_freq=acq) ** 2
    model = COSPECTRAL_MODELS[cospectrum] if isinstance(cospectrum, str) else cospectrum
    trapz = getattr(np, 'trapezoid', None) or np.trapz
    path_len, sonic_tau = _sonic_geometry(ds, acq, sonic_path_length)
    if path_len is None:
        logger.warning(
            "analytic_tube: no usable anemometer path length (declared or known "
            "for this model), so the sonic leg of the cospectrum is left out and "
            "the factor understates the loss; declare vpath_length or add the "
            "model to the bundled instrument files.")
    irga_path = geom.get('path_length') or None      # falsy (0.0) is not a path
    if irga_path is None:
        irga_path = 0.127
        logger.warning(
            "analytic_tube: analyser model unknown and no plausible optical path "
            "declared; assuming the LI-7200's 0.127 m. Declare vpath_length on "
            "the tube's instrument block to remove the assumption.")
    if geom.get('tau') is None:
        logger.warning(
            "analytic_tube: no analyser response time (declared or known); the "
            "dynamic-response term is omitted from the factor.")

    cf = np.ones_like(U, dtype=float)
    for i in range(U.size):
        u_i = max(float(U[i]), MIN_WIND_SPEED)
        zL_i = float(np.atleast_1d(zL)[i]) if np.atleast_1d(zL).size > i else 0.0
        n = F * zmd / u_i
        try:
            form = np.asarray(model(n, zL_i), dtype=float)
        except (TypeError, ValueError):
            form = np.array([model(nk, zL_i) for nk in n], dtype=float)
        form = np.where(np.isfinite(form) & (form > 0), form, 0.0)
        # The separation loss depends on the period's wind speed (it is set by the
        # transit time across the gap), so it joins the static tube/IRGA response
        # inside the loop rather than outside it.
        # The cospectrum has two legs: the analyser damps the gas (tube, path,
        # separation) and the anemometer damps the wind. Both depend on this
        # period's wind speed, so they join the static response inside the loop.
        H_i = (H
               * _irga_tf(F, ws=u_i, path_length=irga_path,
                          tau=geom.get('tau'))
               * _sep_tf(F, separation=geom.get('separation', 0.0), ws=u_i,
                         vertical=geom.get('vertical_separation', 0.0)))
        if path_len is not None:
            H_i = H_i * _sonic_w_tf(F, ws=u_i, path_length=path_len,
                                    tau=sonic_tau, acq_freq=acq)
        H_i = np.clip(np.where(np.isfinite(H_i), H_i, 1.0), 0.0, 1.0)
        den = trapz(form * H_i, lnF)
        cf[i] = trapz(form, lnF) / den if den > 0 else 1.0

    ds['scf_analytic_tube'] = xr.DataArray(cf.reshape(ds[wind_speed].shape), dims=dims)
    ds['scf_analytic_tube'].attrs.update({
        'description': 'analytic closed-path spectral correction factor '
                       '(model cospectrum x tube+IRGA transfer function)',
        'method': 'lowpass_analytic_closedpath', 'cospectrum': cospectrum,
        'tube_length_m': geom['L'], 'tube_radius_mm': geom['r_mm'],
        'tube_flow_lpm': geom['lpm'], 'tube_model': str(tube_model),
        'irga_path_length_m': irga_path,
        'sonic_path_length_m': (path_len if path_len is not None
                                else 'unknown -- sonic leg omitted')})
    return ds
