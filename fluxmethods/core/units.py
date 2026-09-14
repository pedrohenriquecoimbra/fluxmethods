"""Units, on the registry the application already uses.

**Where this collection deliberately behaves differently from the code it
copies.** ``oneflux_preproc/core/units.py`` builds its own registry and calls
``pint.set_application_registry`` on it. That is right for a *program*: it owns
the process. It is wrong for a *library* -- importing fluxmethods would otherwise
repoint the global registry of whatever imported it, and pint refuses to operate
across two registries, so every quantity the program had already made would be
orphaned the moment it imported us.

So the registry here is whichever one the application already uses. **It is
adopted, not replaced -- but it is modified, in two ways, and both are global
effects on an object this package does not own:**

1. The unit definitions the methods rely on (``ppm``, ``ppt``, ``µmol``,
   ``celsius``, ``ppbv``) are added if the registry lacks them. A name it already
   knows is left alone: the application's spelling wins over ours.
2. ``force_ndarray_like`` is set. This is **not** cosmetic and not optional: on a
   default registry a scalar quantity's magnitude is a ``float``, and with the
   flag it is a 0-d ndarray. The files in this package are verbatim copies of
   code written against a registry that sets it, so without it identical source
   would compute different types -- the one behavioural difference that a diff of
   the two trees could never show. ``pint_xarray.setup_registry`` happens to set
   it too; it is set explicitly here so the semantics do not rest on that.

Everything below this block is that module, unchanged.
"""

import logging

import pint
from pint.errors import UndefinedUnitError
import xarray as xr

logger = logging.getLogger(__name__)

ureg = pint.get_application_registry()

# See (2) above: the copied code's semantics depend on this, so it is declared
# rather than inherited from whatever else has touched the registry.
ureg.force_ndarray_like = True

#: ``name, definition, meaning`` for the units these methods rely on.
#:
#: ``meaning`` is what the name has to resolve to for the copied code to compute
#: what it computed in the reference implementation. It is checked rather than
#: assumed, because **resolvability is not definedness**: pint resolves a great
#: many strings through its prefix parser, so a name can look "already known"
#: while meaning something else entirely. ``ppt`` is the case that matters --
#: pint reads it as *pico-pint*, a volume, where these methods mean parts per
#: thousand. That is thirteen orders of magnitude and a dimension, on the unit
#: the VOC path is written in.
_DEFINITIONS = (
    ('ppm', 'ppm = 1e-6 = parts_per_million', '1e-6 dimensionless'),
    ('ppt', 'ppt = 1e-3 = parts_per_thousand', '1e-3 dimensionless'),
    ('micromol', 'µmol = 1e-6 mole = micromol', '1e-6 mole'),
    ('ppbv', 'ppbv = 1 = parts_per_million_by_volume', '1 dimensionless'),
)

#: ``celsius`` is declared by the reference implementation as an offset unit
#: (``kelvin; offset: 273.15``), which is what pint's own ``degree_Celsius``
#: already means. It is not checked by resolution the way the others are --
#: offset units cannot be built from a bare name -- and it is not redefined,
#: because redefining an offset unit on a shared registry is a good way to break
#: a host's temperatures for no gain.
_OFFSET = 'celsius = kelvin; offset: 273.15 = celsius'


def _means(name, expected):
    """Whether ``name`` already resolves to ``expected`` on this registry."""
    try:
        have, want = ureg(name).to_base_units(), ureg(expected).to_base_units()
    except Exception:
        return False
    return (have.dimensionality == want.dimensionality
            and float(have.magnitude) == float(want.magnitude))


for _name, _definition, _meaning in _DEFINITIONS:
    if _means(_name, _meaning):
        continue                                  # the registry already agrees
    try:
        _had = str(ureg(_name).to_base_units())
    except Exception:
        _had = None
    try:
        ureg.define(_definition)
    except Exception:                             # pragma: no cover - registry said no
        logger.error("could not define %r as %s on the application registry; "
                     "methods that use it will not compute what they should",
                     _name, _meaning)
        continue
    if _had is not None:
        # Loud on purpose: we have just changed what a name means on a registry
        # this package does not own, and the host may be using it.
        logger.warning(
            "redefined %r on the application registry: it resolved to %s, and "
            "the methods here need %s. pint reads 'ppt' as pico-pint, which is "
            "why this is checked by meaning rather than by whether the name "
            "resolves.", _name, _had, _meaning)

try:
    ureg(_OFFSET.split('=')[0].strip())
except Exception:
    try:
        ureg.define(_OFFSET)
    except Exception:                             # pragma: no cover
        pass

try:                                          # the .pint accessor, where available
    import pint_xarray
    pint_xarray.setup_registry(ureg)
except Exception:                             # pragma: no cover - optional
    pint_xarray = None

# Define aliases (case-insensitive mapping)
UNIT_ALIASES = {
    'ppt': '1e-3',
    'ppm': '1e-6',
    'ppb': '1e-9',
    'kpa': 'kilopascal',
    'μmol': 'micromole',
    'umol': 'micromole',
    'ug': 'microgram',
    'lit/m': 'L/min',
    # 'celsius': 'delta_degC',
    'kelvin': 'Kelvin',
}


def resolve_unit(unit_str: str):
    """
    Attempts to resolve a unit string using Pint.
    Falls back to alias lookup if the unit is not found.
    """
    unit_str = unit_str.replace('_', '/')

    if not unit_str:
        return ureg('dimensionless')  # Treat empty as dimensionless

    try:
        return ureg(unit_str)
    except UndefinedUnitError:
        # Try alias map (case-insensitive)
        unit_str_lower = unit_str.lower()
        alias = UNIT_ALIASES.get(unit_str_lower)
        if alias:
            return ureg(alias)
        else:
            raise UndefinedUnitError(
                f"Unit '{unit_str}' not found and no alias matched.")


def convert_unit(da, to_units):
    """
    Convert an xarray.DataArray containing Pint Quantities in .values to new units.

    Parameters:
        da: xarray.DataArray with pint.Quantity values
        to_units: string or Pint Unit to convert to, e.g. 'degC', 'meter'

    Returns:
        xarray.DataArray with converted magnitude and updated units attribute.
    """
    q = da.data  # should be a Pint Quantity
    converted_q = q.to(to_units)

    # Rebuild DataArray with converted magnitude and original coords/dims
    da_converted = da.copy()
    da_converted.data = converted_q  # .magnitude
    da_converted.attrs = da.attrs.copy()
    da_converted.attrs['unit_in'] = str(converted_q.units)
    return da_converted

def convert_to_prefered_units(ds, units=[
    "m", "s", "μmol/m²/s", "m/s", "W/m²", "g/m²/s", "kg/m^3", "m^3/mol", "J/kg/K", "dimensionless"]):
    array_input = False
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise ValueError("Input must be an xarray Dataset or DataArray")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name='data')
        array_input = True
    for var in ds.data_vars:
        for target_unit in units:
            try:
                # Try to convert
                ds[var].data = ds[var].data.to(target_unit)
                break  # Stop at the first successful conversion
            except Exception as e:
                continue  # Try next unit
    if array_input:
        return ds['data']
    return ds
