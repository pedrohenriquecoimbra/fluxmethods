"""EddyPro's analytic band-pass spectral correction, transcribed from its source.

These routines exist to reproduce EddyPro *exactly*, not to improve on it. They are
a separate pair of registered methods rather than options on the built-in ones,
because the built-in routines make their own defensible choices (a scalar
line-averaging term for the sonic temperature, a single high-pass factor) and
bending those to match would trade a physical argument for a compatibility one.
Keeping both means a run can be checked against EddyPro without either
implementation being compromised -- and adding the second cost one module.

Everything here follows the EddyPro engine (LI-COR Biosciences; licensed for
non-commercial research use), as published at ``LI-COR-Environmental/eddypro-engine``:

========================================  =========================================
``bpcf_moncrieff_97.f90``                 gas fluxes: grid, natural frequency, order
``bpcf_anemometric_fluxes.f90``           sensible heat and momentum
``bpcf_cospectral_models.f90``            ``CospectraMoncrieff97``
``bpcf_analytic_transfer_functions.f90``  the transfer functions
``bpcf_aux_subs.f90``                     band-pass assembly and the integrals
========================================  =========================================

Two properties of that assembly are easy to get wrong and are the reason this is a
transcription rather than a re-derivation:

* **Every term enters once per cospectrum leg.** ``BandPassTransferFunction``
  multiplies ``LP(var1) * LP(var2)``, with terms that do not apply to a variable
  left at the 1 they were initialised to. So ``w'Ts'`` carries the sonic response
  and path averaging *squared* (both legs are sonic), and the block-averaging
  high pass is squared for every flux.
* **The integration bounds are a mask on the grid, not the grid.** The grid runs
  from ``1/7200`` Hz to 10 Hz in 500 logarithmic steps, while the sum runs only
  over points with ``nf > 1/5000`` -- dropping the lowest decade, which is where
  the high pass has already removed the flux. Using ``1/5000..100`` Hz *as* the
  grid instead makes the factors visibly worse.

References
----------
Moncrieff, J. B., Massheder, J. M., de Bruin, H., Elbers, J., Friborg, T.,
Heusinkveld, B., Kabat, P., Scott, S., Soegaard, H., and Verhoef, A. (1997). A
system to measure surface fluxes of momentum, sensible heat, water vapour and
carbon dioxide. Journal of Hydrology, 188-189, 589-611.
Moncrieff, J., Clement, R., Finnigan, J., and Meyers, T. (2004). Averaging,
detrending, and filtering of eddy covariance time series. In: Handbook of
Micrometeorology, Springer, 7-31.
Lenschow, D. H. and Raupach, M. R. (1991). The attenuation of fluctuations in
scalar concentrations through sampling tubes. J. Geophys. Res., 96, 15259-15268.
"""

import logging

import numpy as np
import xarray as xr

from ..units import convert_unit

logger = logging.getLogger(__name__)

# ``nseconds`` and ``nfreq`` are hard-coded in EddyPro too -- they define an
# artificial frequency range, not a property of the record.
NSECONDS = 7200
NFREQ = 500
FMAX = 10.0

# Bounds of the summation in ``SpectralCorrectionFactors``.
NF_MIN = 1.0 / 5000.0
NF_MAX = 100.0

# Molecular diffusivity used by the *laminar* branch of the tube attenuation
# [m2 s-1]. Carried over from the tube-attenuation models already in this package
# (``fitting_models``), where it is the default ``D``. EddyPro keeps one per gas
# (``Dc(var)``); a single value stands in here because this routine returns one
# factor for all closed-path gases, and because the laminar branch does not run at
# any normal sampling flow -- the sample's tube sits at Re ~3100, well turbulent.
LAMINAR_DIFFUSIVITY = 1.381e-5

# Reynolds number above which the tube flow is treated as turbulent, as EddyPro does.
TURBULENT_RE = 2300.0

_PI = np.pi


def _grid():
    """``nf``, exactly as EddyPro builds it.

    ``nf(1)`` is assigned directly and the loop then runs from ``i = 2``, so the
    formula is never evaluated at ``i = 1``; the first two points are not one
    logarithmic step apart. Reproduced rather than tidied, because the whole point
    of this module is to be the same array.
    """
    nf = np.empty(NFREQ, dtype=float)
    nf[0] = 1.0 / NSECONDS
    i = np.arange(2, NFREQ + 1, dtype=float)
    nf[1:] = np.exp(np.log(nf[0]) + (np.log(FMAX) - np.log(nf[0])) / NFREQ * i)
    return nf


def _cospectra_moncrieff97(nf, kf, zL, momentum=False):
    """``CospectraMoncrieff97``: returns ``Co(f)``, i.e. the model ``f Co(f)`` / ``nf``.

    ``momentum`` selects the Reynolds-stress branch, which EddyPro builds in the same
    subroutine from its own coefficients and its own unstable break point (0.24
    rather than 0.54). The scalar branch is shared by heat, water vapour and every
    gas -- EddyPro assigns ``Cospectrum%of(w_ts)`` and the rest from
    ``Cospectrum%of(w_co2)`` -- so only these two shapes exist.
    """
    if momentum:
        if zL > 0.0:
            Au = 0.124 * (1.0 + 7.9 * zL) ** 0.75
            Bu = 2.34 * Au ** (-1.1)
            return kf / (nf * (Au + Bu * kf ** 2.1))
        return np.where(
            kf < 0.24,
            20.78 * kf / (nf * (1.0 + 31.0 * kf) ** 1.575),
            12.66 * kf / (nf * (1.0 + 9.6 * kf) ** 2.4),
        )
    if zL > 0.0:
        Ac = 0.284 * (1.0 + 6.4 * zL) ** 0.75
        Bc = 2.34 * Ac ** (-1.1)
        return kf / (nf * (Ac + Bc * kf ** 2.1))
    return np.where(
        kf < 0.54,
        12.92 * kf / (nf * (1.0 + 26.7 * kf) ** 1.375),
        4.378 * kf / (nf * (1.0 + 3.8 * kf) ** 2.4),
    )


def _air_viscosity(t_air):
    """Kinematic viscosity of air [m2 s-1] as a cubic in temperature [K]."""
    return (-1.1555e-14 * t_air ** 3 + 9.5728e-11 * t_air ** 2
            + 3.7604e-8 * t_air - 3.4484e-6)


def _lp_sonic(nf, ws, vpath_length, tau):
    """``dsonic * wsonic`` -- set for every one of ``u``, ``v``, ``w``, ``ts``.

    ``ws`` in m s-1, ``vpath_length`` in m, ``tau`` in s: the frequencies below are
    formed as ``f * length / speed``, so mixed units would pass silently. The caller
    takes them from :func:`~.main._sonic_geometry` (metres and seconds by
    construction) and from a dataset already in preferred units.
    """
    fp = nf * abs(vpath_length / ws)
    dsonic = 1.0 / np.sqrt(1.0 + (2.0 * _PI * nf * tau) ** 2)
    e = np.exp(-2.0 * _PI * fp)
    wsonic = ((2.0 / (_PI * fp))
              * (1.0 + e / 2.0 - 3.0 * (1.0 - e) / (4.0 * _PI * fp)))
    return dsonic * wsonic


def _lp_irga(nf, ws, t_air, geom):
    """``dirga * wirga * t * sver * shor`` for a closed-path analyser.

    ``t_air`` in kelvin; the viscosity polynomial is a fit in absolute temperature.
    ``geom`` comes from :func:`~.main._find_tube`, which converts the metadata's
    declared units once, at that boundary: lengths and separations to metres, bore
    radius left in millimetres, flow left in litres per minute. The two remaining
    conversions are done here, next to the formula that needs them.

    Note the outer square root on ``wirga`` and its absence on ``wsonic``: that
    asymmetry is EddyPro's, and it is kept.
    """
    fp = nf * abs(geom['path_length'] / ws)
    fs_ver = nf * abs(geom['vertical_separation'] / ws)
    fs_lat = nf * abs(geom['separation'] / ws)

    radius = geom['r_mm'] * 1e-3                       # mm -> m
    diameter = 2.0 * radius
    flow = geom['lpm'] * 1e-3 / 60.0                   # l min-1 -> m3 s-1
    tube_velocity = flow / (_PI * radius ** 2)
    Re = tube_velocity * diameter / _air_viscosity(t_air)

    if Re < TURBULENT_RE:                              # laminar
        tube_time = geom['L'] / tube_velocity
        diffusivity = geom.get('diffusivity') or LAMINAR_DIFFUSIVITY
        t = np.exp(-tube_time * (_PI * radius * nf) ** 2 / (6.0 * diffusivity))
    else:                                              # Lenschow & Raupach (1991)
        t = np.exp((-80.0 * radius * nf ** 2 * Re ** (-0.125) * geom['L'])
                   / tube_velocity ** 2)

    dirga = 1.0 / np.sqrt(1.0 + (2.0 * _PI * nf * geom['tau']) ** 2)
    x = 2.0 * _PI * fp
    with np.errstate(divide='ignore', invalid='ignore'):
        wirga = np.sqrt((3.0 + np.exp(-x) - 4.0 / x * (1.0 - np.exp(-x))) / x)
    # The expression is 0/0 at the bottom of the grid, where its analytic limit is
    # exactly 1 (expanding the exponentials gives (x - x^2/6)/x -> 1). Substituting
    # that limit is the value of the function there, not a guard against it.
    wirga = np.where(np.isfinite(wirga), wirga, 1.0)
    sver = np.exp(-9.9 * fs_ver ** 1.5)
    shor = np.exp(-9.9 * fs_lat ** 1.5)
    return dirga * wirga * t * sver * shor


def _hp(nf, averaging_seconds):
    """``AnalyticHighPassTransferFunction`` for block-averaged records.

    ``1 - sinc^2(pi f T)`` lies in [0, 1] analytically; the clip only removes the
    rounding that puts it a few ulp outside near ``f T -> 0``, and the substitution
    is the function's own limit there. Neither changes the transfer function.
    """
    phase = _PI * nf * averaging_seconds
    with np.errstate(divide='ignore', invalid='ignore'):
        H = 1.0 - np.sin(phase) ** 2 / phase ** 2
    return np.clip(np.where(np.isfinite(H), H, 1.0), 0.0, 1.0)


def _bounded(bptf):
    """A transfer function is an attenuation: ``0 <= H <= 1``, and 1 where undefined.

    Every term entering ``bptf`` is bounded analytically, so this absorbs floating
    point at the ends of the grid, where several of them are 0/0 with a limit of 1
    (see :func:`_lp_irga`). It is not a cap on a factor that would otherwise be
    large: a run in which it bound materially would be describing an instrument
    these formulas do not describe.
    """
    return np.clip(np.where(np.isfinite(bptf), bptf, 1.0), 0.0, 1.0)


def _correction_factor(co, bptf, nf):
    """``SpectralCorrectionFactors``: left-endpoint rectangles, masked.

    ``NaN`` when the factor cannot be formed, rather than 1. The two are equivalent
    once applied -- :func:`~....main._apply_spectral_factor` fills a missing factor
    with 1, i.e. leaves the flux uncorrected -- but ``NaN`` says so in the attached
    ``scf_*`` variable instead of presenting an uncorrected period as a corrected one.
    """
    df = np.diff(nf)
    keep = (nf[:-1] > NF_MIN) & (nf[1:] < NF_MAX)
    denominator = np.sum(bptf[:-1][keep] * co[:-1][keep] * df[keep])
    if not np.isfinite(denominator) or denominator <= 0:
        return np.nan
    return np.sum(co[:-1][keep] * df[keep]) / denominator


def _kelvin(ds, name, size):
    """Per-period air temperature in kelvin, or ``None`` if it cannot be had.

    The unit is converted rather than assumed. ``convert_to_prefered_units`` runs
    before this stage and normalises lengths, speeds and fluxes, but it deliberately
    leaves temperature alone, so a dataset can reach here carrying degrees Celsius --
    and the viscosity polynomial in :func:`_air_viscosity` is a fit in *absolute*
    temperature, which a Celsius magnitude would silently corrupt rather than fail on.

    Returns ``None`` rather than standing in a nominal temperature: inventing one
    would put a number the run never measured into a published correction factor.
    """
    if name not in ds:
        return None
    da = ds[name]
    if hasattr(getattr(da, 'data', None), 'units'):        # a pint quantity
        # Imported here rather than at the top: pint is a dependency of the data,
        # not of this module, and this is the only branch that can meet one.
        from pint import DimensionalityError

        try:
            da = convert_unit(da, 'K')
        except (DimensionalityError, AttributeError, TypeError, ValueError):
            logger.warning("%r could not be converted to kelvin; its units are %r.",
                           name, getattr(da.data, 'units', 'unknown'))
            return None
    try:
        values = np.ravel(np.asarray(da.pint.magnitude, dtype=float))
    except (AttributeError, TypeError):
        values = np.ravel(np.asarray(da.values, dtype=float))
    return values if values.size == size else None


def _inputs(ds, wind_speed, height, stability, air_temperature):
    """``(U, z-d, zL, T_air)`` as plain arrays, or ``None`` if unavailable.

    ``U`` is in m s-1, ``z-d`` in m and ``zL`` dimensionless because the pipeline
    normalises those to preferred units before this stage; ``T_air`` is converted
    here, and may be ``None`` -- only the closed-path method needs it.
    """
    from .analytic import _period_geometry

    geom = _period_geometry(ds, wind_speed, height, stability)
    if geom is None:
        return None
    U, zmd, zL = geom
    return U, zmd, np.atleast_1d(zL), _kelvin(ds, air_temperature, U.size)


def _usable_wind(value):
    """The period's wind speed, or ``None`` if the factor cannot be formed from it.

    Deliberately *not* floored at :data:`~.main.MIN_WIND_SPEED`, unlike the built-in
    methods. That floor bounds a factor that would otherwise run away as the wind
    drops, which is a reasonable thing for a method that stands on its own to do --
    but EddyPro applies no floor, and on the twelve sample periods below 0.3 m/s it
    costs up to 1.8 %. A replication method that quietly disagreed there would make
    the agreement elsewhere worth less, so the only rejection here is of a wind speed
    no formula could use.
    """
    u = float(value)
    return u if np.isfinite(u) and u > 0.0 else None


def _detrending_is_block_average(ds):
    """EddyPro's high pass here is the block-averaging one; refuse to stand in for others."""
    detrending = ((ds.attrs.get('Corrections', {}) or {}).get('detrending', {}) or {})
    method = (detrending.get('method') if hasattr(detrending, 'get') else None) or 'ba'
    if str(method) in ('ba', 'block_average'):
        return True
    logger.warning(
        "the EddyPro replication methods implement its block-averaging high pass, "
        "but detrending is %r; EddyPro uses a different transfer function for that "
        "(linear detrending, running mean), which is not transcribed here, so the "
        "low-frequency term is omitted.", method)
    return False


def _averaging_seconds(ds):
    options = ds.attrs.get('Options', {}) or {}
    minutes = (options.get('process_duration')
               or (ds.attrs.get('Files', {}) or {}).get('fileduration') or 30)
    try:
        seconds = float(minutes) * 60.0
    except (TypeError, ValueError):
        seconds = 1800.0
    return seconds if np.isfinite(seconds) and seconds > 0 else 1800.0


def _attach(ds, name, values, wind_speed, attrs):
    dims = ds[wind_speed].dims
    ds[name] = xr.DataArray(np.asarray(values).reshape(ds[wind_speed].shape), dims=dims)
    ds[name].attrs.update(attrs)
    return ds


def _anemometric_factor(ds, wind_speed, height, stability, air_temperature,
                        high_pass, momentum, caller):
    """The factor shared by ``w'Ts'`` and ``w'u'``, or ``None`` if it cannot be formed.

    ``BPCF_AnemometricFluxes`` computes both from one transfer function: every leg of
    either cospectrum is an anemometer variable, so the sonic dynamic response and
    path averaging each enter twice, as does the high pass::

        BP(w_ts) = dsonic(w) dsonic(ts) wsonic(w) wsonic(ts) HP(w) HP(ts)
        BP(w_u)  = dsonic(w) dsonic(u)  wsonic(w) wsonic(u)  HP(w) HP(u)

    The two differ only in the cospectral model they are integrated against, which is
    why they share an implementation here rather than being written twice.
    """
    from .analytic import _sonic_geometry

    got = _inputs(ds, wind_speed, height, stability, air_temperature)
    if got is None:
        logger.warning("%s: missing %r or %r; skipping.", caller, wind_speed, height)
        return None
    U, zmd, zL, _ = got

    acq = float(ds.attrs.get('acquisition_frequency', 20.0)) or 20.0
    path_length, tau = _sonic_geometry(ds, acq, None)
    if path_length is None or not tau:
        logger.warning("%s: no usable anemometer path length or time constant; "
                       "skipping.", caller)
        return None

    nf = _grid()
    hp = (_hp(nf, _averaging_seconds(ds))
          if (high_pass and _detrending_is_block_average(ds)) else 1.0)

    cf = np.full(U.shape, np.nan, dtype=float)
    for i in range(U.size):
        u_i = _usable_wind(U[i])
        if u_i is None:
            continue
        zL_i = float(zL[i]) if zL.size > i else 0.0
        kf = nf * abs(zmd / u_i)
        co = _cospectra_moncrieff97(nf, kf, zL_i, momentum=momentum)
        bptf = _bounded(_lp_sonic(nf, u_i, path_length, tau) ** 2 * hp ** 2)
        cf[i] = _correction_factor(co, bptf, nf)
    return cf, path_length, tau


def bpcf_anemometric_fluxes(ds, wind_speed='wind_speed', height='z-d',
                            stability='z_L', air_temperature='air_temperature',
                            high_pass=True, **kwargs):
    """``BPCF_AnemometricFluxes``, scalar branch: the sensible-heat factor.

    Attaches ``scf_eddypro_anemometric``.
    """
    got = _anemometric_factor(ds, wind_speed, height, stability, air_temperature,
                              high_pass, momentum=False,
                              caller='bpcf_anemometric_fluxes')
    if got is None:
        return ds
    cf, path_length, tau = got
    return _attach(ds, 'scf_eddypro_anemometric', cf, wind_speed, {
        'description': "EddyPro analytic band-pass correction factor for the sonic "
                       "heat flux (Moncrieff et al. 1997/2004)",
        'units': 'dimensionless',
        'method': 'bandpass_moncrieff_1997_heat', 'sonic_path_length_m': path_length,
        'sonic_tau_s': tau, 'high_pass': bool(high_pass)})


def bpcf_momentum(ds, wind_speed='wind_speed', height='z-d', stability='z_L',
                  air_temperature='air_temperature', high_pass=True, **kwargs):
    """``BPCF_AnemometricFluxes``, Reynolds-stress branch: the momentum factor.

    Same transfer function as the heat flux -- both legs of ``w'u'`` are anemometer
    variables -- integrated against the momentum cospectrum, whose coefficients and
    unstable break point differ from the scalar one.

    This is the correction that makes EddyPro's *published* friction velocity larger
    than the one implied by the covariances it publishes. Measured on the FR-Gri
    sample by re-running the binary with the corrections toggled, its published
    ``u*`` is exactly 1.000000 times the covariance-derived value with the correction
    off and 1.0071 with it on. Since ``u* = (<u'w'>^2 + <v'w'>^2)^(1/4)``, a factor
    ``F`` on the momentum covariance moves ``u*`` by ``F^(1/2)``.

    Attaches ``scf_eddypro_momentum``.
    """
    got = _anemometric_factor(ds, wind_speed, height, stability, air_temperature,
                              high_pass, momentum=True, caller='bpcf_momentum')
    if got is None:
        return ds
    cf, path_length, tau = got
    return _attach(ds, 'scf_eddypro_momentum', cf, wind_speed, {
        'description': "EddyPro analytic band-pass correction factor for the "
                       "momentum flux (Moncrieff et al. 1997/2004)",
        'units': 'dimensionless',
        'method': 'bandpass_moncrieff_1997_momentum', 'sonic_path_length_m': path_length,
        'sonic_tau_s': tau, 'high_pass': bool(high_pass)})


def bpcf_moncrieff_97(ds, wind_speed='wind_speed', height='z-d', stability='z_L',
                      air_temperature='air_temperature', tube=None,
                      high_pass=True, **kwargs):
    """``BPCF_Moncrieff97``: the closed-path gas correction factor.

    One leg is the anemometer and the other the analyser::

        BP(w_gas) = dsonic(w) wsonic(w)
                  * dirga(g) wirga(g) t(g) sver(g) shor(g)
                  * HP(w) HP(g)

    Attaches ``scf_eddypro_moncrieff_1997``.
    """
    from .analytic import _find_tube, _sonic_geometry

    got = _inputs(ds, wind_speed, height, stability, air_temperature)
    if got is None:
        logger.warning("bpcf_moncrieff_97: missing %r or %r; skipping.",
                       wind_speed, height)
        return ds
    U, zmd, zL, Ta = got
    if Ta is None:
        logger.warning(
            "bpcf_moncrieff_97 needs %r in kelvin for the tube Reynolds number and "
            "cannot get it; skipping rather than assuming a temperature.",
            air_temperature)
        return ds

    geom = tube if tube is not None else _find_tube(ds)
    if not geom:
        logger.warning(
            "bpcf_moncrieff_97 describes a closed-path analyser and found no tube "
            "geometry; no factor is attached. An open-path setup still loses flux to "
            "path averaging, separation and sonic response -- use a method that "
            "describes that instrument.")
        return ds
    geom = dict(geom)
    if not geom.get('path_length') or not geom.get('tau'):
        logger.warning("bpcf_moncrieff_97: analyser path length or time constant "
                       "unknown; skipping.")
        return ds

    acq = float(ds.attrs.get('acquisition_frequency', 20.0)) or 20.0
    sonic_path, sonic_tau = _sonic_geometry(ds, acq, None)
    if sonic_path is None or not sonic_tau:
        logger.warning("bpcf_moncrieff_97: no usable anemometer geometry; skipping.")
        return ds

    nf = _grid()
    hp = (_hp(nf, _averaging_seconds(ds))
          if (high_pass and _detrending_is_block_average(ds)) else 1.0)

    cf = np.full(U.shape, np.nan, dtype=float)
    for i in range(U.size):
        u_i = _usable_wind(U[i])
        if u_i is None:
            continue
        zL_i = float(zL[i]) if zL.size > i else 0.0
        kf = nf * abs(zmd / u_i)
        co = _cospectra_moncrieff97(nf, kf, zL_i)
        bptf = _bounded(_lp_sonic(nf, u_i, sonic_path, sonic_tau)
                        * _lp_irga(nf, u_i, float(Ta[i]), geom)
                        * hp ** 2)
        cf[i] = _correction_factor(co, bptf, nf)

    return _attach(ds, 'scf_eddypro_moncrieff_1997', cf, wind_speed, {
        'description': "EddyPro analytic band-pass correction factor for closed-path "
                       "gas fluxes (Moncrieff et al. 1997/2004)",
        'units': 'dimensionless',
        'method': 'bandpass_moncrieff_1997_gas', 'tube_length_m': geom['L'],
        'tube_radius_mm': geom['r_mm'], 'tube_flow_lpm': geom['lpm'],
        'high_pass': bool(high_pass)})
