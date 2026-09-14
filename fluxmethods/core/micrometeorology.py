import logging

import numpy as np
import xarray as xr
from .units import convert_unit, ureg, convert_to_prefered_units
from .measure_type import molar_density_of
from . import constants

logger = logging.getLogger(__name__)

#: Gases whose mean molar density is carried past detrending, and the molar mass
#: each is weighed with. A density correction needs the *mean* density of the gas
#: it corrects, and the concentration series does not survive to the point where
#: that correction runs -- detrending removes exactly the mean it would read. So
#: the density is taken here, at sample level, before any correction touches the
#: series, and carried through the average as its own variable. ``rho_h2ov`` has
#: always been derived this way; ``rho_co2`` had not, which is why the density
#: correction was reading a zero.
DENSITY_GASES = {'co2': constants.Mco2}

# Constants. The gas constants are ``constants.R``, ``constants.Rd`` and
# ``constants.Rv`` -- one copy each, so a value cannot move in one place only.
M_d = ureg('0.02897 kg/mol')       # kg/mol (dry air)
M_h2o = ureg('0.01802 kg/mol')     # kg/mol (water vapor)
e_base = np.e       # base of natural log


def add_micrometeorological_variables_to_data(data):
    chi = data.h2o.pint.to("ppt")
    if data.h2o.attrs.get('measure_type', None) == 'mixing_ratio':
        chi = chi / (1 + chi)
    data = data.assign(h2o_mole_fraction=chi,)


    # if (Stats % Mean(te) > 220d0 . and . Stats % Mean(te) < 340d0) Stats % T = Stats % Mean(te)
    # if (biomet % val(bTa) > 220d0 . and . biomet % val(bTa) < 340d0) Stats % T = biomet % val(bTa)
    # Stats % Pr = Metadata % bar_press
    # if (Stats % Mean(pe) > 40000 . and . Stats % Mean(pe) < 110000) Stats % Pr = Stats % Mean(pe)
    # if (biomet % val(bPa) > 40000 . and . biomet % val(bPa) < 110000) Stats % Pr = biomet % val(bPa)

    # Ambient air molar volume[m+3 mol-1] and air mass density[kg m-3]
    # if (Stats % Pr > 0d0 . and . Stats % T /= error) then
    data = data.assign(
        air_molar_volume=(constants.R * data.ts / data.air_pressure),)
    
    data = data.assign(
        Ma=molecular_weight_wet_air(chi),)
    data = data.assign(
        rho_h2ov=rho_h2o(chi, data.air_pressure, data.air_molar_volume),)
    data = data.assign(
        e=water_vapor_partial_pressure(data.rho_h2ov, data.ts),)

    data = data.assign(
        air_temperature_measured=data.air_temperature.pint.to('kelvin'),)
    data = data.assign(
        air_temperature_from_sonic=air_temperature_derived_from_sonic_temperature(data.ts, data.e, data.air_pressure),)
    # The sonic temperature as its own variable. ``ts`` is one of the series
    # detrending acts on, so its period mean does not reach the flux assembly;
    # the Obukhov length is built there on the virtual temperature the sonic
    # reads, the one its buoyancy flux <w'Ts'> is paired with.
    data = data.assign(sonic_temperature=data.ts)

    # The air temperature is the measured column where the run has one and the
    # humidity-corrected sonic elsewhere; the raw sonic only where even that is
    # NaN (no H2O), which is EddyPro's own last fallback. The importer always
    # carries ``air_temperature``, as an all-NaN placeholder when nothing was
    # measured, so this is a per-sample fill rather than a choice. The fill keeps
    # the placeholder's attrs, among them the ``dump`` mark that drops a variable
    # from the output, so they are cleared.
    air_temperature = (data.air_temperature.pint.to('kelvin')
                       .fillna(data.air_temperature_from_sonic)
                       .fillna(data.ts))
    air_temperature.attrs = {}
    data = data.assign(air_temperature=air_temperature)

    # The three quantities above were seeded with the *sonic* temperature, which is
    # the virtual temperature and here runs ~0.8 K warm. They describe ambient air,
    # so re-evaluate them at the air temperature now that it is known: the molar
    # volume and the water-vapour density are ~0.3 % out when taken at the sonic
    # temperature, and they carry into the specific humidity, the heat capacity and
    # any mole-fraction flux. The vapour pressure then reduces to Dalton's
    # e = chi * P, in which the temperature cancels.
    #
    # ``Options.gas_conversion_temperature = sonic`` keeps the molar volumes and
    # the water-vapour density at the sonic temperature while everything else
    # stays at the air temperature: EddyPro's state on a run without a measured
    # air temperature, where it forms the molar volume from the sonic mean before
    # deriving the air temperature and never revisits it.
    options = data.attrs.get('Options') or {}
    sonic_conversion = str(options.get('gas_conversion_temperature') or '').strip().lower() == 'sonic'
    T_conv = data.ts if sonic_conversion else data.air_temperature
    data = data.assign(
        air_molar_volume=(constants.R * T_conv / data.air_pressure),)
    data = data.assign(
        rho_h2ov=rho_h2o(chi, data.air_pressure, data.air_molar_volume),)
    data = data.assign(
        e=water_vapor_partial_pressure(data.rho_h2ov, data.air_temperature),)

    data = data.assign(
        es=saturation_vapor_pressure(data.air_temperature),)

    data = data.assign(
        RH=relative_humidity(data.e, data.es),)
    data = data.assign(
        VPD=vapor_pressure_deficit(data.e, data.es),)
    data = data.assign(
        Td=dew_point_temperature(data.e),)  # e in kPa
    data = data.assign(
        Pd=dry_air_partial_pressure(data.air_pressure, data.e),)
    data = data.assign(
        dry_air_molar_volume=dry_air_molar_volume(data.Pd, T_conv),)
    data = data.assign(
        sealevel_moist_air_molar_volume=sealevel_moist_air_molar_volume(
            data.air_temperature, data.e),)
    # Ambient-air thermodynamics are evaluated at the air temperature. Only a true
    # air temperature may be mapped to ``air_temperature``: the EddyPro adapter maps
    # its ``air_t`` column and never ``cell_t`` (an analyser probe runs several K
    # warm and would bias every density and heat capacity, hence the flux), and the
    # VOC meteo reader writes the meteorological air temperature.
    Ta_air = data.air_temperature
    data = data.assign(
        rho_d=dry_air_mass_density(data.Pd, Ta_air),)
    data = data.assign(
        rho_m=moist_air_density(data.rho_d, data.rho_h2ov),)
    data = data.assign(
        dry_air_heat_capacity=cp_d(Ta_air),)
    data = data.assign(
        cph2o=cp_h2o(data.RH, Ta_air),)
    data = data.assign(
        specific_humidity=specific_humidity(data.rho_h2ov, data.rho_m),)
    data = data.assign(
        Ta_refined=refine_Ta_sonic(data.specific_humidity, Ta_air),)
    data = data.assign(
        air_heat_capacity=moist_air_cp(data.specific_humidity, data.dry_air_heat_capacity, data.cph2o),)
    data = data.assign(
        lambda_v=latent_heat_vaporization(Ta_air),)
    data = data.assign(
        sigma=density_ratio_sigma(data.rho_h2ov, data.rho_d),)

    data = add_gas_mass_densities(data)

    data = convert_to_prefered_units(data)
    return data


def add_gas_mass_densities(data):
    """Carry each gas's mean mass density past the corrections, as ``rho_<gas>``.

    A density correction is written in terms of the mean density of the gas it
    corrects, and by the time it runs there is no series left to take that mean
    from: detrending has removed precisely the mean it needs. Deriving it here, at
    sample level and before any correction, is the same thing
    :func:`add_micrometeorological_variables_to_data` does for ``rho_h2ov``.

    The concentration is turned into a density through the ``measure_type``
    registry (:func:`~.measure_type.molar_density_of`), so a molar density passes
    through, a wet mole fraction is weighed by the moist-air molar density and a
    dry mixing ratio by the dry-air one -- and a ``conversion`` step that rewrote
    the type is followed without this function knowing one ran.

    Means only. A covariance must not be taken against these: they are derived
    before the time-lag compensation, so they are not aligned with ``w``. Means
    are, to a sample at the window edges.
    """
    for gas, molar_mass in DENSITY_GASES.items():
        name = f'rho_{gas}'
        if gas not in data or name in data:
            continue
        # A gas column that carries no sample at all is one the importer created
        # to satisfy IMPORTER_REQUIRED_COLUMNS, not one the run measured: a VOC
        # run has no IRGA, so its co2 and h2o are all-NaN placeholders. Their
        # density is NaN either way, but going on would infer a measure_type for
        # a column that has nothing to infer from and report that assumption --
        # a warning about a variable this run does not have, which is noise in
        # front of the ones it does.
        if not np.isfinite(np.asarray(data[gas].values, dtype=float)).any():
            continue
        try:
            data = data.assign(**{name: molar_density_of(data, gas) * molar_mass})
        except (KeyError, AttributeError) as err:
            # A gas whose density cannot be built (an undeclared type whose
            # inference needs a molar volume that is not there) is left out
            # rather than guessed at; the correction that wants it refuses.
            logger.warning("could not derive %s: %s", name, err)
    return data


def air_temperature_derived_from_sonic_temperature(Ts, e, Pa):
    return Ts * (1+0.32*e/Pa)**-1   # - 273.15


def molecular_weight_wet_air(chi_h2o):
    return constants.Mv * chi_h2o + constants.Md * (1 - chi_h2o)


def rho_h2o(chi_h2o, P, Va):
    # Using ideal gas: ρ = (P * χ) / (R_specific * T)
    return chi_h2o / Va * constants.Mv


def water_vapor_partial_pressure(rho_h2o, Ta):
    # Using ideal gas law: e = ρ * R_specific * T, Ta in Kelvin. R_specific is
    # EddyPro's rounded Rv (461.5 J/kg/K), not constants.R / Mv (461.4).
    return rho_h2o * constants.Rv * Ta


def saturation_vapor_pressure(Ta):
    # # Campbell & Norman (1998): es = 0.6108 * exp(17.27 * (T - 273.15)/(T - 35.85))
    es = Ta.copy()
    Tc = Ta.data.to('delta_degC').magnitude
    es.data = np.exp(77.345 + 0.0057 * Tc - 7235 / Tc) / (Tc**8.2)
    es.data = es * ureg('Pa')
    return es


def relative_humidity(e, es):
    return 100 * e / es


def vapor_pressure_deficit(e, es):
    return es - e


def dew_point_temperature(e):
    # # Campbell & Norman: Tdew = (243.5 * ln(e/0.6108)) / (17.27 - ln(e/0.6108)) + 273.15
    Td = e.copy()
    e_kPa = convert_unit(e, 'kPa') / ureg('1 kPa')
    Td.data = 240.97 * np.log(e_kPa / 0.611) / (17.502 - np.log(e_kPa / 0.611))
    Td.data = Td * ureg('kelvin')
    return Td


def dry_air_partial_pressure(Pa, e):
    return Pa - e


def dry_air_molar_volume(Pd, Ta):
    # constants.R is the universal gas constant in J/(mol·K)
    # Pd is the dry air partial pressure
    # Ta is the ambient temperature in Kelvins
    # Ambient%Vd = (Stats%Pr * Ambient%Va) / Ambient%p_d
    # Va = R * Ta / Pa
    Vd = constants.R * Ta / Pd
    return Vd


def sealevel_moist_air_molar_volume(Ta, e):
    Vsea = constants.R * Ta / (ureg("99767.5 pascal") - e)
    return Vsea

def dry_air_mass_density(Pd, Ta):
    """Calculate dry air mass density (rho_d) using the ideal gas law.

    Args:
        Pd: Dry air partial pressure (Pa)
        Ta: Ambient temperature (K)

    Returns:
        Dry air mass density (kg/m³)
    """
    return Pd / (constants.Rd * Ta)


def moist_air_density(rho_d, rho_h2o):
    return rho_d + rho_h2o


def cp_d(Tk):
    """
    Calculate the specific heat capacity of dry air at constant pressure (cp_d).

    Uses a temperature-dependent empirical formula to compute the specific heat
    capacity of dry air as a function of temperature.
    """
    return xr.ones_like(Tk) * ((Tk.pint.to('degC').data.magnitude + 23.12) ** 2 / ureg("3364 K kg/J")) + constants.Cpd
    cpd = Tk.copy()
    cpd.data = (constants.Cpd + (Tk.pint.to('degC') / ureg('degC') + 23.12) ** 2 / ureg("3364 K kg/J"))
    return cpd
    cpd = Tk.copy()
    # Ta is in celsius
    Ta = Tk.data.to('degC') / ureg('degC')
    cpd.data = (constants.Cpd.to("J/kg/K") +
                ((Ta + 23.12)**2 / 3364) * ureg("J/kg/K"))
    return cpd


def cp_h2o(RH, Tk):
    Ta = Tk.pint.to('degC').data.magnitude
    cpv = xr.zeros_like(Tk)
    return xr.ones_like(Tk) * (1859 + 0.13 * RH + (0.193 + 5.6 * 1e-3 * RH)
                               * Ta + (1e-3 + 5 * 1e-5 * RH) * Ta**2) * ureg("J/kg/K")
    # cpv.data = 
    return cpv


def specific_humidity(rho_h2o, rho_a):
    return rho_h2o / rho_a


def refine_Ta_sonic(q, Ta):
    return Ta / (1 + 0.51 * q)


def moist_air_cp(q, cp_d, cp_h2o):
    return (1 - q) * cp_d + q * cp_h2o


def latent_heat_vaporization(Tk):
    # lambda(Ta) = 1e3 (3147.5 - 2.37 Ta[K]) J/kg, the fit EddyPro applies
    # (flux_params.f90; Fleagle & Businger 1980).
    Ta = Tk.pint.to('K').data.magnitude
    return xr.ones_like(Tk) * (1e3 * (3147.5 - 2.37 * Ta)) * ureg('J/kg')


def density_ratio_sigma(rho_h2o, rho_d):
    return rho_h2o / rho_d


# Zero-plane displacement as a fraction of canopy height, for a dense uniform
# canopy (Monteith & Unsworth 2013, Sect. 9.2). It is the standard closure when a
# site describes its vegetation but not its displacement.
CANOPY_DISPLACEMENT_RATIO = 2.0 / 3.0


def _length_in_metres(value):
    """A declared length as a float in metres, or None if it says nothing.

    Accepts what a config can hold: a number, a bare numeric string, or a string
    carrying its unit (``'0.15m'``). A value that cannot be read as a length is
    None rather than an assumed zero, so the caller can tell "not declared" from
    "declared as zero".
    """
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(ureg(text).to('m').magnitude)
    except Exception:
        logger.warning("could not read %r as a length; ignoring it", value)
        return None


def _is_true(value):
    """A configured flag as a boolean. ConfigObj hands every value over as text."""
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ('true', '1', 'yes', 'on')


def displacement_height(declared=None, canopy_height=None,
                        height_above_displacement=False):
    """The zero-plane displacement height d, in metres.

    What the site declares wins. Failing that, d is derived from the canopy the
    site does describe, as :data:`CANOPY_DISPLACEMENT_RATIO` times its height.
    When neither is known the result is 0 m -- the measurement height then stands
    unmodified, which is the only honest answer for a site that describes neither
    and is reported by the caller rather than assumed quietly.

    A declared 0 is read as "not declared": it is what the EddyPro metadata
    writes for a site that never filled the field in, and a real tower has a
    displacement below its measurement height, not at the ground.

    ``height_above_displacement`` is the separate case of a site whose stated
    measurement height is *already* referenced to the displacement plane, so
    ``zm`` is (z - d) and d must be 0 for it not to be subtracted twice.
    GEddySoft's ``SENSOR_HEIGHT`` is documented in its own ini as "in meters
    above displacement height", so a config derived from one says this. It is a
    declaration, not a fallback, and so does not warn -- which is the whole
    point of having it: the 0 it produces is a decision on the record rather
    than the shrug the last branch returns.

    Args:
        declared: The site's own displacement height, in metres or as a string
            carrying its unit.
        canopy_height: The site's canopy height, same forms.
        height_above_displacement: The measurement height is already stated
            relative to the displacement plane.

    Returns:
        The displacement height in metres, as a float.
    """
    if _is_true(height_above_displacement):
        return 0.0

    d = _length_in_metres(declared)
    if d is not None and d > 0:
        return d

    h = _length_in_metres(canopy_height)
    if h is not None and h > 0:
        return CANOPY_DISPLACEMENT_RATIO * h

    logger.warning(
        "site declares neither a displacement height nor a canopy height; "
        "using d = 0 m, so (z-d) is the measurement height itself")
    return 0.0


def wind_direction(u, v, offset=0):
    """
    Calculate wind direction from u and v wind components.
    
    Parameters:
        u (float or np.ndarray): zonal wind component (positive eastward)
        v (float or np.ndarray): meridional wind component (positive northward)

    Returns:
        float or np.ndarray: wind direction in degrees, where 0° = North, 90° = East, etc.
    """
    # Work on bare magnitudes: pint treats angle units as dimensionless, so
    # mixing a degree Quantity with plain floats silently degrades the result.
    angle = np.degrees(np.arctan2(v, u))
    if hasattr(angle, 'pint'):  # xarray DataArray (possibly pint-backed)
        angle = angle.pint.dequantify()
    else:  # pint Quantity or plain numeric
        angle = getattr(angle, 'magnitude', angle)

    direction = (180.0 - angle + offset) % 360.0
    return direction * ureg('degree')
