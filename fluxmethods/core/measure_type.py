"""How a scalar was measured, and what follows from it.

A gas analyser does not report "the CO2". It reports an ambient molar density, a
mole fraction of moist air, or a mixing ratio per mole of dry air, and which one
it is decides three separate things downstream:

* what turns ``cov(w, x)`` into a molar flux -- the amount of air the
  concentration is expressed *per*;
* what turns the mean concentration into a molar density, which is the same
  question and therefore the same factor;
* whether a density (WPL) correction is still owed, and how much of it.

All three are answered here rather than in three places. Keyed by one
``measure_type`` vocabulary, that is what lets a conversion step rewrite a variable's
``measure_type`` and have every consequence follow by itself: convert a cell
molar density to a dry mixing ratio and the flux factor changes, the density
factor changes and the WPL correction stops being owed, with no step needing to
know that a conversion happened.

It lives in ``core`` because it is a property of the *measurement*, not of any
one stage: micrometeorology reads it to build the mean densities, flux assembly
reads it to build the fluxes, and the ``conversion`` correction writes it.
"""

import logging

from regorator import create_registry, register

from .units import ureg

logger = logging.getLogger(__name__)

__all__ = ["FLUX_CONVERSIONS", "measure_type_of", "flux_conversion",
           "molar_density_of", "wpl_is_owed"]


# --------------------------------------------------------------------------- #
# Concentration -> per-volume factor, keyed by how the scalar was measured
# --------------------------------------------------------------------------- #
#
# The registered factor answers "per how much air?", which serves both the flux
# and the mean density:
#
#   molar_density  [mol m-3] : already per volume; cov is already a flux, and the
#                              mean is already a density. Density fluctuations
#                              remain, so the full WPL correction is owed.
#   mixing_ratio   (dry)     : per mole of *dry* air, conserved along a sampling
#                              tube -- multiply by the dry-air molar density. No
#                              WPL: that is the point of reporting a dry ratio.
#   mole_fraction  (wet)     : per mole of *moist* air -- multiply by the moist-air
#                              molar density; only the water-dilution term remains.
#
# Registering these rather than hard-coding one of them lets a single run mix
# species (a config may declare CO2 as a density and H2O as a dry ratio) and makes
# the WPL coupling a declared consequence instead of a global flag.
FLUX_CONVERSIONS = create_registry(
    "Concentration -> per-volume factors, keyed by the scalar's ``measure_type``.",
    frozen=True)


@register(name='molar_density', registry=FLUX_CONVERSIONS,
          description='cov(w, rho_c) is already a molar flux',
          wpl='full', family='density', source='built-in')
def _from_molar_density(data):
    return 1


@register(name='mixing_ratio', registry=FLUX_CONVERSIONS,
          description='per mole of dry air: multiply by the dry-air molar density',
          wpl=None, family='ratio', source='built-in')
def _from_dry_mixing_ratio(data):
    return data['dry_air_molar_volume'] ** -1


@register(name='mole_fraction', registry=FLUX_CONVERSIONS,
          description='per mole of moist air: multiply by the moist-air molar density',
          wpl='dilution', family='ratio', source='built-in')
def _from_wet_mole_fraction(data):
    return data['air_molar_volume'] ** -1


def measure_type_of(data, name):
    """How the scalar ``name`` was measured, as a :data:`FLUX_CONVERSIONS` key.

    The declared ``measure_type`` wins -- it comes straight from the engine's own
    metadata (EddyPro's ``col_*_measuring_type``, carried into the variable's
    ``Attr`` by the compatibility layer), or from a ``conversion`` step that
    rewrote it. When it is absent or unknown the *dimensionality* still settles
    the family, because a molar density and a mixing ratio are not dimensionally
    alike: ``mol m-3`` is a density, and a dimensionless concentration is a ratio.
    Only dry-vs-wet cannot be inferred, so a bare dimensionless scalar is taken as
    a dry ``mixing_ratio`` -- the near universal convention for a reported gas
    concentration (EddyPro's ``CO2_DRY``) -- and the assumption is logged rather
    than made silently.
    """
    var = data[name]
    declared = (var.attrs.get('measure_type')
                or (var.attrs.get('Attr') or {}).get('measure_type'))
    if declared in FLUX_CONVERSIONS:
        return declared

    units = getattr(getattr(var, 'pint', None), 'units', None)
    if units is not None and ureg(str(units)).dimensionality == ureg('mol/m^3').dimensionality:
        inferred = 'molar_density'
    else:
        inferred = 'mixing_ratio'
    logger.warning(
        "%s declares no usable measure_type (%r); assuming %r from its units (%s). "
        "Declare measure_type in the config to remove this assumption.",
        name, declared, inferred, units)
    return inferred


def flux_conversion(data, name):
    """The factor turning ``cov(w, name)`` into a molar flux [mol m-2 s-1].

    Resolved through :data:`FLUX_CONVERSIONS` from the scalar's measurement type
    (see :func:`measure_type_of`). The factor carries units, so pairing it with the
    wrong covariance fails dimensional analysis instead of silently producing a
    plausible but wrong flux.
    """
    return FLUX_CONVERSIONS[measure_type_of(data, name)](data)


def molar_density_of(data, name):
    """The mean molar density [mol m-3] of the scalar ``name``.

    The same registry entry, applied to the mean rather than to the covariance,
    because it is the same question: a concentration times the amount of air it is
    expressed per is a density either way. A ``molar_density`` therefore passes
    through untouched, and a ratio is multiplied by the dry or moist molar density
    it is a ratio *of*.

    Whether the answer means anything depends on where it is called: it must see
    the concentration before detrending has removed its mean. That is why
    micrometeorology derives it, and why the density correction declares it as an
    input instead of recomputing it from a series it cannot vouch for.
    """
    return data[name] * flux_conversion(data, name)


def wpl_is_owed(data, *names):
    """Whether any of ``names`` still needs a density (WPL) correction.

    A dry mixing ratio needs none; a molar density needs the full correction and a
    wet mole fraction the dilution term. Queried from the same registry that chose
    the conversion, so the two can never disagree -- and so a ``conversion`` step
    that rewrites ``measure_type`` retires the correction by itself.
    """
    for name in names:
        if name in data and FLUX_CONVERSIONS.meta[measure_type_of(data, name)].get('wpl'):
            return True
    return False
