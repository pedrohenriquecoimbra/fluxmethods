"""The Webb-Pearman-Leuning density correction.

A gas analyser that reports a density, or a fraction of moist air, does not
report a conserved quantity. Warm the air and it expands; add water vapour and it
displaces dry air. Either makes the CO2 number fall with no CO2 having gone
anywhere, and the eddy covariance of that number therefore carries a flux of heat
and a flux of water alongside the flux of CO2. Webb, Pearman and Leuning (1980)
identified the constraint that separates them --- there is no net flux of *dry
air* through the surface --- and the correction that follows from it.

In density form (Webb et al. 1980, Eq. 24 and 25), with ``mu = Md/Mv`` and
``sigma = rho_v / rho_d``:

    E   = (1 + mu sigma) (w'rho_v' + (rho_v / T) w'T')
    F_c = w'rho_c' + mu (rho_c / rho_d) w'rho_v' + (1 + mu sigma) (rho_c / T) w'T'

Every term of that is a mean or a covariance this module has to be *given*, which
is the whole difficulty and the reason this is a step with declared inputs rather
than a few lines inside flux assembly. Two of them are not what a naive reading
suggests:

``w'rho_v'`` is the covariance with the water-vapour *mass density*, not whatever
the analyser reported. When H2O comes as a mole fraction or a mixing ratio, the
raw vapour flux the engine computes has already shed part of the thermal
expansion, and feeding it to Webb's equation as written double-counts that part.
So the vapour input is canonicalised first, through the same ``measure_type``
registry that chose the flux conversion, and the density covariance is
reconstructed from it exactly ---
``tests/test_wpl.py::test_the_vapour_input_is_canonical_however_the_water_was_reported``
is where that reconstruction is checked, on the parcel that file is built around.

``w'T'`` is the covariance with the *air* temperature. The sonic reports the
virtual temperature, whose covariance runs high by the latent contribution --- by
a few per cent on an ordinary midday parcel, and more the wetter the surface,
since the excess scales with the vapour flux and so with the inverse Bowen ratio.
The temperature flux is what :func:`~...process.L0_fluxes.schotanus_correction`
already recovers, so this reads its result rather than repeating it, which is
also what makes the ordering between the two explicit;
``tests/test_wpl.py::test_the_heat_input_is_the_air_temperature_flux_not_the_sonic_one``
is what holds the input to the recovered flux.

References
----------
Webb, E. K., Pearman, G. I., and Leuning, R. (1980). Correction of flux
measurements for density effects due to heat and water vapour transfer.
Quarterly Journal of the Royal Meteorological Society, 106, 85-100.
(``webb1980``)
"""

# built-in modules
import logging

# project modules
from ..core import constants
from ..core.measure_type import FLUX_CONVERSIONS, measure_type_of
from ..core.units import convert_to_prefered_units

logger = logging.getLogger(__name__)

#: The gases this correction knows how to correct, and where each keeps its parts:
#: the raw molar flux, the mean mass density carried past detrending
#: (see :func:`~...core.micrometeorology.add_gas_mass_densities`), the corrected
#: flux it writes, and the gas's molar mass.
GASES = {
    'co2': dict(raw='FC_L0', density='rho_co2', out='FC', molar_mass=constants.Mco2),
}


def vapour_mole_fraction_flux(data):
    """The water-vapour molar flux per mole of *moist* air, ``n_a <w'chi_v'>``.

    The one canonical vapour quantity both Webb equations need. Everything else
    about the water vapour --- the corrected evaporation, the dilution term of the
    CO2 flux, and the density covariance ``w'rho_v'`` itself --- follows from it
    and the temperature flux, so reducing the vapour input to this once is what
    keeps the rest free of any question about how H2O was reported.

    Getting there from ``FH2O_L0`` depends on that reporting, and the difference is
    not decorative: each form has shed a different part of the density
    fluctuation already.

    * ``mole_fraction``: ``FH2O_L0`` is already ``n_a <w'chi_v'>``.
    * ``mixing_ratio``: per mole of dry air, so it is larger by ``1/(1 - chi_v)``.
    * ``molar_density``: the covariance of ``chi_v n_a`` carries the thermal
      expansion of the air as well, and ``n_v <w'T'>/T`` is that part.

    All three are exact at constant pressure, which is the assumption Webb's
    derivation already rests on.
    """
    raw = data['FH2O_L0']
    chi = data['h2o_mole_fraction']
    kind = measure_type_of(data, 'h2o')
    if kind == 'mole_fraction':
        return raw
    if kind == 'mixing_ratio':
        return raw * (1 - chi)
    return raw + (data['rho_h2ov'] / constants.Mv) * (
        temperature_flux(data) / data['air_temperature'])


def temperature_flux(data):
    """``<w'T'>``, the kinematic flux of *air* temperature [K m s-1].

    Read back out of the sensible heat flux rather than recomputed: ``H`` is
    already the sonic (buoyancy) flux with the humidity contribution removed
    (Schotanus et al. 1983), and ``H = rho_m cp <w'T'>`` by definition, so this is
    the same quantity divided back down. Doing it this way means there is exactly
    one place where the sonic temperature becomes an air temperature, and it means
    this correction cannot silently disagree with the heat flux the run reports.

    ``cov_w_ts`` must not be used here: the sonic covariance runs high by the
    latent contribution, by an amount that grows as the Bowen ratio falls, and
    since the thermal term is comparable to the CO2 flux itself the excess would
    land more or less whole in the answer.
    ``tests/test_wpl.py::test_the_heat_input_is_the_air_temperature_flux_not_the_sonic_one``
    is what pins this to the recovered air-temperature flux.
    """
    return data['H'] / (data['rho_m'] * data['air_heat_capacity'])


def webb_et_al_1980(data, **kwargs):
    """Apply the density correction to every gas that is still owed one.

    Which gases those are is not asked here: it is read off each gas's declared
    ``measure_type``, through the registry that also chose its flux conversion. A
    molar density is owed the full correction, a wet mole fraction only the water
    dilution, and a dry mixing ratio nothing at all --- reporting a dry ratio *is*
    the correction. So a ``conversion`` step upstream retires this one for the gas
    it converted without either step knowing about the other.

    Writes ``FC`` (and ``E`` / ``LE``) beside the raw ``FC_L0`` / ``E_L0`` / ``LE_L0``,
    which are kept: the ``_L0`` suffix is this package's name for the assembled
    flux before the period-level corrections, and both belong in the output.

    Refuses, with a warning naming what is missing, rather than approximating: an
    absent input here does not make the correction smaller, it makes it wrong, and
    a correction that silently declines to happen is indistinguishable from one
    that ran.
    """
    for name in ('FH2O_L0', 'h2o_mole_fraction', 'rho_h2ov', 'rho_d', 'rho_m',
                 'air_temperature', 'air_heat_capacity', 'lambda_v', 'H'):
        if name not in data:
            logger.warning(
                "webb_et_al_1980 needs %r and the dataset has none; no density "
                "correction is applied. The fluxes reported are the uncorrected "
                "ones.", name)
            return data

    mu = constants.mu
    rho_v, rho_d, T = data['rho_h2ov'], data['rho_d'], data['air_temperature']
    sigma = rho_v / rho_d
    wT = temperature_flux(data)
    f_chi = vapour_mole_fraction_flux(data)

    # w'rho_v', the covariance Webb's E is written in terms of. It differs from
    # the vapour mole-fraction flux by exactly the air's thermal expansion, which
    # is the identity rho_v = chi_v n_a Mv differentiated at constant pressure.
    w_rho_v = constants.Mv * f_chi - rho_v * wT / T

    # Webb Eq. 25. The bracket is w'rho_v' + rho_v w'T'/T, which is Mv f_chi
    # again -- written out rather than cancelled, so the equation on the page is
    # the equation in the code.
    e_wpl = (1 + mu * sigma) * (w_rho_v + (rho_v / T) * wT)
    data['E'] = convert_to_prefered_units(e_wpl)
    data['LE'] = convert_to_prefered_units(data['lambda_v'] * e_wpl)

    for gas, part in GASES.items():
        if gas not in data or part['raw'] not in data:
            continue
        owed = FLUX_CONVERSIONS.meta[measure_type_of(data, gas)].get('wpl')
        if not owed:
            continue
        if part['density'] not in data:
            logger.warning(
                "%s is reported as a type that owes the %s density correction, "
                "but %r is not in the dataset -- micrometeorology derives it "
                "before detrending, and without it the correction would be scaled "
                "by a mean that detrending has already removed. Skipping %s.",
                gas, owed, part['density'], gas)
            continue

        rho_c = data[part['density']]
        raw = data[part['raw']] * part['molar_mass']          # mass flux [kg/m2/s]
        # The dilution term is owed by both types: a mole fraction of moist air
        # still falls when vapour displaces dry air. The thermal term is owed only
        # by a density -- a mole fraction is already per mole of air, so the
        # expansion has cancelled out of it. Subtracting mu*sigma is exactly that
        # cancellation, not a tuning: it removes the part of Webb's Eq. 24 heat
        # term that the ratio has already accounted for.
        heat = (1 + mu * sigma) if owed == 'full' else (mu * sigma)
        corrected = (raw
                     + mu * (rho_c / rho_d) * w_rho_v
                     + heat * (rho_c / T) * wT)
        data[part['out']] = convert_to_prefered_units(
            corrected * part['molar_mass'] ** -1)                # back to molar
        logger.info("webb_et_al_1980: %s corrected (%s), %s -> %s",
                    gas, owed, part['raw'], part['out'])
    return data
