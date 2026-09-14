import numpy as np
import xarray as xr
from ..utils import resolve_variable
from regorator import create_registry, register


def correction_factor_horst_1997(u, z, tc=0.00796, zL=None, α=1):
    """
    Calculate the frequency response correction factor for eddy covariance
    flux measurements using first-order-response scalar sensors, based on
    Horst (1997), Equation 11.

    This correction accounts for high-frequency attenuation due to the sensor's
    time response and the spectral distribution of turbulence. The factor is
    frequency-independent: it is the ratio of attenuated to true flux obtained by
    integrating the cospectral model, so no per-frequency argument is required.

    Parameters
    ----------
    u : float
        Mean horizontal wind speed at the measurement height (m/s).
    z : float
        Measurement height above the surface (m).
    tc : float, optional (default=0.00796)
        Time constant of the scalar sensor (seconds). Smaller values represent
        faster sensor response.
    zL : float or None, optional (default=None)
        Atmospheric stability parameter (z/L). If None, neutral stability is assumed.
    α : float, optional (default=1)
        Coefficient used in the cospectral model. Represents spectral shape.
        Default of 1 assumes standard shape.

    Returns
    -------
    ⟨w′c′⟩⟨w′c′⟩m​​ : float or ndarray
        Correction factor SCF. To correct for sensor attenuation:
            Corrected Flux = Measured Flux / SCF

    References
    ----------
    Horst, T. W. (1997). A simple formula for attenuation of eddy fluxes measured 
    with first-order-response scalar sensors. Boundary-Layer Meteorology, 82, 219–233.

    Notes
    -----
    This function implements Equation 11 from Horst (1997), assuming neutral 
    atmospheric stability. If stability correction is desired, further modification 
    is required to adjust the spectral model accordingly.
    """
    α = np.where(zL > 0, 1, 7/8) if zL is not None else α
    nm = np.where(zL > 0, 2.0-1.915/(1+0.5*zL),
                  0.085) if zL is not None else 0.085
    return 1 / (1 + (2 * np.pi * nm * tc * u/z) ** α)


# Create registries
TRANSFER_FUNCTION_MODELS = create_registry()


def transfer_function_generator(ds, freq='frequency_bin', model='lorentzian', **kwargs):
    fq = resolve_variable(freq, ds)

    # Get the model function from the registry
    fcTF = TRANSFER_FUNCTION_MODELS[model] if isinstance(model, str) else model
    fcLabel = model if isinstance(
        model, str) else fcTF.__name__

    # Apply the model to the frequency
    tf = fcTF(fq, **kwargs)

    tf = xr.DataArray(tf,
                      coords={k: ds.coords[k] for k in list(
                          fq.dims)},
                      dims=list(fq.dims),
                      name=f'transfer_function_{fcLabel}')
    tf.attrs['source'] = fcLabel
    return tf

@register(name="lorentzian", registry=TRANSFER_FUNCTION_MODELS)
def lorentzian(f, fc, F=1):
    return F / (1 + (f / fc)**2)


@register(name="gaussian", registry=TRANSFER_FUNCTION_MODELS)
def gaussian(f, fc, F=1):
    """Gaussian transfer-function model used for cut-off frequency fitting.

    ``F * exp(-ln2 * (f / fc)**2)``. Like :func:`lorentzian`, it equals ``F/2``
    at ``f = fc`` (half-power cut-off), but decays faster at high frequency.
    Used jointly with the Lorentzian fit to bracket the empirical transfer
    function.

    Ported from ``fun_Gauss`` in FreqCor (``src/FREQCOR_functions.py``) by
    Ariane Faurès and Bernard Heinesch (University of Liège, Gembloux
    Agro-Bio Tech), after the original MATLAB version by Marc Aubinet.
    https://github.com/BernardHeinesch/FreqCor (Apache-2.0).

    Parameters
    ----------
    f : array-like
        Natural frequency (Hz).
    fc : float
        Cut-off frequency (Hz).
    F : float, optional
        Normalisation factor (default 1).
    """
    return F * np.exp(-np.log(2) * (f / fc)**2)


@register(name="lorentzian_peltola", registry=TRANSFER_FUNCTION_MODELS)
def lorentzian_peltola(f, fc, F=1):
    """Square-root Lorentzian transfer function (Peltola et al., 2021).

    ``F * sqrt(1 / (1 + (f / fc)**2))``. Peltola et al. (2021) showed that for
    the cospectral approach the transfer function is better described by the
    square root of the Lorentzian.

    Ported from ``fun_Lorentz_peltola`` in FreqCor
    (``src/FREQCOR_functions.py``) by Ariane Faurès and Bernard Heinesch
    (University of Liège, Gembloux Agro-Bio Tech), after the original MATLAB
    version by Marc Aubinet.
    https://github.com/BernardHeinesch/FreqCor (Apache-2.0).

    References
    ----------
    Peltola, O. et al. (2021). Suitability of fixed-frequency corrections... .

    Parameters
    ----------
    f : array-like
        Natural frequency (Hz).
    fc : float
        Cut-off frequency (Hz).
    F : float, optional
        Normalisation factor (default 1).
    """
    return F * np.sqrt(1 / (1 + (f / fc)**2))


@register(name="sonic_response", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_sonic_response(f, ws, path_length=0.125, tau=None,
                                     acq_freq=20.0):
    """The anemometer's own high-frequency loss on the **vertical wind** leg.

    Product of the sonic dynamic (time-response) transfer function and the square
    root of the path-averaging transfer function -- the square root because path
    averaging is a variance-like quantity and only one of the two legs of a
    cospectrum is the wind.

    This is a property of ``w``, so it attenuates **every** eddy-covariance flux,
    not just the sensible heat flux: the cospectrum ``w'c'`` between the wind and a
    gas is damped by the anemometer on the wind leg and by the analyser (tube, path
    averaging, response) plus their separation on the gas leg. EddyPro composes its
    band-pass transfer function the same way, one leg per variable.

    Ported from ``TFsonic`` in FreqCor (``src/FREQCOR_functions.py``) by
    Ariane Faurès and Bernard Heinesch (University of Liège, Gembloux
    Agro-Bio Tech), after the original MATLAB version by Marc Aubinet.
    https://github.com/BernardHeinesch/FreqCor (Apache-2.0).

    Unlike :func:`transfer_function_sonic_path_averaging`, this form depends on
    the mean wind speed (through the normalised frequency ``f * path_length/ws``)
    and bundles the dynamic response, so it cannot be evaluated from frequency
    alone — pass ``ws`` explicitly.

    Parameters
    ----------
    f : array-like
        Natural frequency (Hz).
    ws : float
        Mean horizontal wind speed (m/s).
    path_length : float, optional
        Vertical acoustic path length (m); this is the dimension the line averaging
        of ``w`` is set by, and the one the reference implementation normalises to.
    tau : float, optional
        Sensor response time constant (s). The response frequency is ``1/tau``, a
        property of the instrument rather than of the logging rate; when it is not
        known the acquisition frequency stands in.
    acq_freq : float, optional
        Acquisition frequency (Hz), used only as that stand-in.
    """
    response = (1.0 / tau) if (tau and tau > 0) else acq_freq
    fp = f * np.abs(path_length / ws)
    dynamic = 1 / np.sqrt(1 + (2 * np.pi * f / response)**2)
    exp_term = np.exp(-2 * np.pi * fp)
    path_averaging = (2 / (np.pi * fp)) * (
        1 + exp_term / 2 - 3 * (1 - exp_term) / (4 * np.pi * fp))
    return dynamic * np.sqrt(path_averaging)


@register(name="horst_1997", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_horst_1997(f, tc=0.00796, zL=None, α=None):
    # Transfer function of the electronic response of the sonic anemometer or IRGA
    # f, in Hz
    # tc, in s
    if not α:
        α = zL/zL*np.where(zL > 0, 1, 7/8) if zL is not None else 1
    return 1 / (1 + (2 * np.pi * f * tc) ** α)


@register(name="sonic_cospectrum_response", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_sonic_cospectrum(f, ws, path_length=np.sqrt(0.11**2 + 0.125**2),
                                       tau=None, acq_freq=20.0):
    """The anemometer's loss on the **w'Ts' cospectrum** -- both of its legs.

    The sonic temperature is line-averaged along the acoustic path exactly as the
    vertical wind is, so the heat-flux cospectrum carries *two* path-averaging
    terms, not one: the vector form for ``w`` and the scalar form for ``Ts``. Each
    is a variance transfer function and enters a cospectrum as its square root, so
    they combine as ``sqrt(pW * pTs)`` -- which is what
    :func:`transfer_function_sonic_path_averaging` returns.

    The dynamic (time-response) term is applied **once**: ``w`` and ``Ts`` are not
    two instruments but two quantities derived from the same acoustic sampling
    chain, so its response enters the measurement once.

    ``tau`` is the instrument time constant and the response frequency is ``1/tau``
    -- a property of the sensor, *not* the logging rate. It is used only when it is
    consistent with the sampling: a sensor whose response is slower than the
    acquisition rate could not support that rate, so such a value is metadata that
    was never filled in rather than a measurement (EddyPro's own default, 0.1 s,
    appears verbatim against every instrument in examples that never set it). Rather
    than invent a response time, the dynamic term is then omitted and only the path
    averaging is applied, which understates the correction slightly but keeps the
    result traceable to declared geometry.
    """
    fp = np.asarray(f, dtype=float) * abs(path_length / max(abs(ws), 1e-6))
    response = (1.0 / tau) if (tau and tau > 0) else None
    if response is not None and acq_freq and response < acq_freq:
        response = None                      # declared response cannot sample this fast
    dynamic = (1.0 if response is None else
               1.0 / np.sqrt(1.0 + (2.0 * np.pi * np.asarray(f, dtype=float) / response) ** 2))
    with np.errstate(divide='ignore', invalid='ignore'):
        path = transfer_function_sonic_path_averaging(fp)
    path = np.clip(np.where(np.isfinite(path), path, 1.0), 0.0, 1.0)
    return dynamic * path


@register(name="sonic_path_averaging", registry=TRANSFER_FUNCTION_MODELS)
def path_averaging_vector(fn):
    """Line-averaging transfer function for a *vector* component (the wind).

    ``fn`` is the frequency normalised by the transit time across the sensing path,
    ``f * path / u``. A variance transfer function: a cospectrum leg takes its
    square root.
    """
    x = 2 * np.pi * fn
    ex = np.exp(-x)
    return (2 / (np.pi * fn)) * (1 + ex / 2 - 3 * (1 - ex) / (2 * x))


def path_averaging_scalar(fn):
    """Line-averaging transfer function for a *scalar* (temperature, a gas).

    Same normalised frequency as :func:`path_averaging_vector`, different geometry:
    a scalar is averaged along the path rather than projected onto it.
    """
    x = 2 * np.pi * fn
    ex = np.exp(-x)
    return (1 / x) * (3 + ex - 4 * (1 - ex) / x)


def transfer_function_sonic_path_averaging(f):
    """Path averaging of the ``w'Ts'`` cospectrum: one leg vector, one scalar."""
    return np.sqrt(path_averaging_vector(f)) * np.sqrt(path_averaging_scalar(f))


@register(name="block_averaging_inband", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_block_averaging_inband(f, acq_freq=20.0, interface_rate=20.0):
    """Loss from the block averaging an interface box does when it downsamples.

    Distinct from the low-frequency correction, which is about the *flux averaging
    period*. This one is high-frequency: a data system whose internal rate exceeds
    the output rate averages over ``N`` internal samples per output sample, and that
    average is a filter in its own right::

        sqrt(| sinc(f * Tba) |),   Tba = round(interface_rate / acq_freq) / interface_rate

    It is a property of each measured variable, so a cospectrum carries one factor
    per leg. Written as the square root because it is a variance transfer function
    entering a cospectrum. The interface rate defaults to the 20 Hz of the LI-7550,
    which is what the reference implementation assumes; at a 20 Hz acquisition the
    two coincide and ``Tba`` is a single sample.
    """
    freq = np.asarray(f, dtype=float)
    n = max(int(round(float(interface_rate) / max(float(acq_freq), 1e-9))), 1)
    tba = n / float(interface_rate)
    x = np.pi * freq * tba
    sinc = np.where(np.abs(x) < 1e-12, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))
    return np.sqrt(np.abs(sinc))


@register(name="irga_response", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_irga_response(f, ws, path_length=0.127, tau=0.1):
    """The analyser's loss on the **gas leg** of a cospectrum.

    A gas concentration is a scalar averaged along the analyser's optical path, and
    the instrument has a finite response time, so its leg carries

        sqrt(path_averaging_scalar(f * path / u)) * 1 / sqrt(1 + (2 pi f tau)^2)

    Distinct from :func:`transfer_function_licor2009`, which is a fixed
    sinc-squared path term with a single hard-coded transit time and no response
    term at all. Both the path length
    and the time constant are instrument properties, so they are passed in from the
    declared or published geometry rather than assumed (see the bundled files
    behind ``sensor_table.sensor_geometry``); for
    a LI-7200 they are 0.127 m and 0.1 s. It is the same construction used for the
    anemometer's legs, and the one the reference implementation uses.
    """
    freq = np.asarray(f, dtype=float)
    fp = freq * abs(path_length / max(abs(ws), 1e-6))
    dynamic = 1.0 / np.sqrt(1.0 + (2.0 * np.pi * freq * float(tau)) ** 2) if tau else 1.0
    with np.errstate(divide='ignore', invalid='ignore'):
        path = np.sqrt(path_averaging_scalar(fp))
    path = np.clip(np.where(np.isfinite(path), path, 1.0), 0.0, 1.0)
    return dynamic * path


@register(name="horst_licor2009", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_licor2009(f, tc=0.1086):
    # Transfer function of the IRGA path averaging of the gas concentration
    # f, in Hz
    # tc, in s
    return (np.sin(np.pi * f * tc)/(np.pi * f * tc)) ** 2


@register(name="sensor_separation_moore_1986", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_sensor_separation(f, separation=0.0, ws=1.0, vertical=0.0):
    """Flux loss from the physical separation of the anemometer and the analyser.

    Two sensors a distance apart do not sample the same eddies at high frequency.
    Moore (1986) describes the resulting cospectral attenuation as

        T(n) = exp(-9.9 n^1.5),   n = f * s / u

    applied **per axis**: the horizontal (crosswind) offset and the vertical offset
    attenuate independently, so their transfer functions multiply. This is
    frequently the largest high-frequency loss for a closed-path system on a short
    tower, where the intake must sit clear of the sonic path: at FR-Gri the LI-7200
    inlet is ~0.17 m from the sonic, which at 1.5 m/s attenuates the cospectrum
    appreciably from a few tenths of a hertz upward.

    The horizontal offset is used whole, without resolving it into along- and
    cross-wind parts. One could argue the along-wind component is merely an
    advective delay already removed by the time-lag correction, but that is not how
    the effect is formulated: Moore's expression is in terms of the separation
    distance, and EddyPro likewise takes the full horizontal offset
    (``hsep = sqrt(nsep^2 + esep^2)``) with no wind-direction rotation.

    Parameters
    ----------
    f : array-like
        Natural frequency (Hz).
    separation : float
        Horizontal (crosswind) separation (m). Zero gives a unit transfer function.
    ws : float
        Mean horizontal wind speed (m/s).
    vertical : float, optional
        Vertical separation (m), attenuating independently of the horizontal one.

    References
    ----------
    Moore, C. J. (1986). Frequency response corrections for eddy correlation
    systems. Boundary-Layer Meteorology, 37, 17-35.
    """
    freq = np.asarray(f, dtype=float)
    speed = max(abs(ws), 1e-6)
    out = np.ones_like(freq)
    for offset in (separation, vertical):
        if offset:
            n = freq * abs(offset) / speed
            out = out * np.exp(-9.9 * n ** 1.5)
    return out


@register(name="tube_attenuation_moore_lenschow", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_tube_attenuation_moore_lenschow(f, r=6.35, L=41, lpm=17,
                                                      v=None, D=1.381e-5):
    """Tube attenuation in the older Moore (1986) / Lenschow form.

    The turbulent branch carries a single Reynolds-number term,

        exp(-80 r f^2 Re^(-1/8) L / v^2)

    against the two-term expression of
    :func:`transfer_function_tube_attenuation_foken_et_al_2012`, which adds the
    Massman and Ibrom (2008) refinement and attenuates roughly twice as strongly at
    a typical closed-path Reynolds number. Provided as a selectable alternative --
    it is the form EddyPro applies (``bpcf_analytic_transfer_functions.f90``), so
    choosing it removes one known difference when reproducing an EddyPro run.
    Which is the better description of a given tube is a question about the tube,
    not about either implementation.

    Parameters as in :func:`transfer_function_tube_attenuation_foken_et_al_2012`:
    ``r`` tube radius (mm), ``L`` length (m), ``lpm`` flow (l/min).
    """
    r = r * 0.001                                   # mm -> m
    if v is None:
        v = (lpm * 0.001 / 60) / (np.pi * r ** 2)   # m/s
    Q = lpm * 0.001 / 60                            # m3/s
    Re = 2 * Q / (np.pi * r * 1.48e-5)
    laminar = np.exp(-(np.pi ** 2 * r ** 4 * f ** 2 * L) / (6 * D * Q) * np.pi)
    turbulent = np.exp(-80 * r * f ** 2 * Re ** (-0.125) * L / v ** 2)
    return np.where(Re >= 2300, turbulent, laminar)


@register(name="tube_attenuation_moore_1986", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_tube_attenuation_moore_1986(f, r=6.35, L=41, lpm=17, v=None, Re=3006):
    # Transfer function of the gas concentration signal attenuation in the tube
    # Re, Reynolds number of the flow
    # r, tube radius in mm
    # L, the tube length in m
    # v, the flow rate in tube in m/s-1
    r = r * 0.001
    def lpm2ms(lmin, radius): return (
        lmin * 0.001 / 60) / (np.pi * (radius)**2)
    if v is None:
        v = lpm2ms(lpm, r)
    return np.exp((-160 * Re ** (-1/8) * r * f ** 2 * L)/(v ** 2))


@register(name="tube_attenuation_foken_et_al_2012", registry=TRANSFER_FUNCTION_MODELS)
def transfer_function_tube_attenuation_foken_et_al_2012(f, r=6.35, L=41, lpm=17, v=None, Re=3006, D=1.381*10**-5):
    # Foken, T., Leuning, R., Oncley, S. R., Mauder, M., and Aubinet, M.: Corrections and Data Quality Control,
    # in: Eddy Covariance: A Practical Guide to Measurement and Data Analysis,
    # edited by: Aubinet, M., Vesala, T., and Papale, D., Springer Netherlands, Dordrecht, 85–131,
    # 2012 (references.bib: foken2012).
    #
    # Transfer function of the gas concentration signal attenuation in the tube
    # Re, Reynolds number of the flow
    # r, tube radius in mm
    # L, the tube length in m
    # v, the flow rate in tube in m/s-1
    r = r * 0.001  # m
    if v is None:
        v = (lpm * 0.001 / 60) / (np.pi * r**2)  # m s-1
    Q = (lpm * 0.001 / 60)  # m3 s-1
    Re = 2 * Q / (np.pi * r * 1.48*10**-5)  # -
    laminar = np.exp(-(np.pi**3 * r**4 * f**2 * L)/(6 * D * Q))
    turbulent = np.exp(-160 * Re ** (-1/8) * (r * f**2 * L)/(v**2))
    turbulent_massmanibrom2008 = np.exp(
        -(160 * Re ** (-1/8) + 2666 * Re**(-29/40)) * (r * f**2 * L)/(v**2))
    return np.where(Re >= 2300, turbulent_massmanibrom2008, laminar)


# Registry for cospectral models
COSPECTRAL_MODELS = create_registry()


def cospectrum_generator(ds, zL='z/L', z='z-d', u='u', freq='frequency_bin', model='horst_1997'):
    """
    Generate cospectral correction values for a given model.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing frequency information.
    zL : str or array-like
        Obukhov stability parameter.
    z : float or array-like
        Measurement height.
    u : float or array-like
        Mean wind speed.
    freq : str, optional
        Name of frequency coordinate in `ds`.
    model : str, optional
        Name of the cospectral model to use.

    Returns
    -------
    xarray.DataArray
        Cospectral correction values.
    """
    zL = resolve_variable(zL, ds)
    u = np.clip(resolve_variable(u, ds), 0, 1e6)
    z = resolve_variable(z, ds)
    fq = resolve_variable(freq, ds)
    flabel = freq if isinstance(freq, str) else fq.name
    nm = np.where(zL > 0, 2.0-1.915/(1+0.5*zL), 0.085)
    ns = (fq * z/u).T

    # Get the model function from the registry
    fcCosp = COSPECTRAL_MODELS[model] if isinstance(model, str) else model
    fcLabel = model if isinstance(model, str) else fcCosp.__name__

    # Apply the model to each frequency and stability value
    cospectra = np.array([[fcCosp(n, zL_) for n in nl_]
                         for zL_, nl_ in zip(zL, ns)])

    cospectra = xr.DataArray(cospectra,
                             coords={k: ds.coords[k] for k in list(
                                 zL.dims) + [flabel]},
                             dims=list(zL.dims) + [flabel],
                             name=f'cospectra_{fcLabel}')
    cospectra.attrs['source'] = fcLabel
    cospectra.attrs[
        'description'] = f'Theoretical cospectral correction function ({fcLabel})'
    return cospectra


@register(name="massman", registry=COSPECTRAL_MODELS)
def massman(fn, zL, a0, fpeak, mu):
    # note that slope parameter "m" is fixed to m=0.75
    return a0*(fn/fpeak)/(1+(fn/fpeak)**(2*mu))**(1.167/mu)


@register(name="horst_1997", registry=COSPECTRAL_MODELS)
def cospec_horst_1997(n, zL):
    """
    Horst (1997) cospectral model.
    
    References
    ----------
    Horst, T. W. (1997). A simple formula for attenuation of eddy fluxes measured 
    with first-order-response scalar sensors. Boundary-Layer Meteorology, 82, 219–233.

    Notes
    -----
    This function implements Equations 8 and 10 from Horst (1997).

    The peak frequency ``nm`` is evaluated only on the branch that defines it:
    ``np.where`` would compute the stable-side formula even for zL <= 0, where its
    denominator vanishes at zL = -2 and a scalar zL raises rather than giving inf.
    """
    nm = 2.0 - 1.915 / (1 + 0.5 * zL) if zL > 0 else 0.085
    k = n / nm
    if zL > 0:
        return (0.637 * k) / (1 + 0.91 * k ** 2.1)
    else:
        return np.where(
            k <= 1,
            (1.05 * k) / (1 + 1.33 * k) ** (7 / 4),
            (0.387 * k) / (1 + 0.38 * k) ** (7 / 3)
        )


@register(name="kaimal_1972", registry=COSPECTRAL_MODELS)
def cospec_kaimal_1972(n, zL):
    """Kaimal et al. (1972) cospectral model, as ``f Co(f) / cov``.

    ``n`` is the natural frequency ``f (z-d) / u``. This is the **neutral** form,
    written directly in ``n``, and it is deliberately *not* rescaled by a
    peak-frequency ``nm(z/L)``. Kaimal's coefficients already put the peak of
    ``f Co(f)`` at ``n = 0.100``, which is the observed neutral peak; Horst's
    ``nm = 0.085`` (for ``z/L <= 0``) is the peak frequency of *his* coefficients,
    which are written in ``k = n / nm`` and peak at ``k = 1``. Dividing by it here
    counts the peak twice and moves this model's down by an order of magnitude in
    ``n``. That is not a rescaling one can absorb: on the FR-Gri geometry it shrinks
    the closed-path correction by close to an order of magnitude (the factor itself
    is measured on that sample by ``tests/test_eddypro_fluxes.py::
    test_analytic_spectral_correction_reproduces_eddypro_closed_path_fluxes``), and
    puts the model in the same order-of-magnitude disagreement with
    :func:`cospec_moncrieff_1997` on the unstable side, where the two otherwise
    agree to well under a per cent.

    ``zL`` is therefore accepted only to satisfy the uniform ``(n, zL)`` signature of
    :data:`COSPECTRAL_MODELS`, and does not enter. Every correction factor built on
    this model varies with mean wind and measurement height but is the *same* in
    stable and unstable air. Where the stable side matters -- its cospectrum peaks at
    higher ``n``, so more of the flux sits in the attenuated band -- select
    ``moncrieff_1997``, which carries a genuine ``z/L`` dependence.

    References
    ----------
    Kaimal, J. C., Wyngaard, J. C., Izumi, Y., and Cote, O. R. (1972). Spectral
    characteristics of surface-layer turbulence. Quarterly Journal of the Royal
    Meteorological Society, 98, 563-589.
    Horst, T. W. (1997). A simple formula for attenuation of eddy fluxes measured
    with first-order-response scalar sensors. Boundary-Layer Meteorology, 82,
    219-233.
    """
    return np.where(
        n <= 1,
        (11 * n) / (1 + 13.3 * n) ** (7 / 4),
        (4 * n) / (1 + 3.8 * n) ** (7 / 3)
    )


@register(name="moncrieff_1997", registry=COSPECTRAL_MODELS)
def cospec_moncrieff_1997(n, zL):
    """Moncrieff et al. (1997) cospectral model, as ``f Co(f) / cov``.

    ``n`` is the natural frequency ``f (z-d) / u``. The model is written directly
    in ``n``; unlike the Kaimal/Horst forms it carries no peak-frequency
    normalisation ``nm``, and introducing one changes the shape rather than
    rescaling it -- on the stable side the frequency dependence lives in the
    ``n**2.1`` term of the denominator, which is what produces the high-frequency
    roll-off.

    References
    ----------
    Moncrieff, J. B., Massheder, J. M., de Bruin, H., Elbers, J., Friborg, T.,
    Heusinkveld, B., Kabat, P., Scott, S., Soegaard, H., and Verhoef, A. (1997).
    A system to measure surface fluxes of momentum, sensible heat, water vapour
    and carbon dioxide. Journal of Hydrology, 188-189, 589-611.
    """
    n = np.asarray(n, dtype=float)
    zL = np.asarray(zL, dtype=float)

    # Clipped only so the stable coefficients stay real where they are discarded.
    Ac = 0.284 * (1.0 + 6.4 * np.clip(zL, 0.0, None)) ** 0.75
    Bc = 2.34 * Ac ** (-1.1)
    stable = n / (Ac + Bc * n ** 2.1)
    unstable = np.where(
        n < 0.54,
        12.92 * n / (1.0 + 26.7 * n) ** 1.375,
        4.378 * n / (1.0 + 3.8 * n) ** 2.4,
    )
    return np.where(zL > 0, stable, unstable)
