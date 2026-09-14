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

import pint
from pint.errors import UndefinedUnitError
import xarray as xr

ureg = pint.get_application_registry()

# See (2) above: the copied code's semantics depend on this, so it is declared
# rather than inherited from whatever else has touched the registry.
ureg.force_ndarray_like = True

#: ``name, definition`` for the units these methods rely on. A name the registry
#: already knows is left alone: the application's own spelling wins over ours.
_DEFINITIONS = (
    ('ppm', 'ppm = 1e-6 = parts_per_million'),
    ('ppt', 'ppt = 1e-3 = parts_per_thousand'),
    ('micromol', 'µmol = 1e-6 mole = micromol'),
    ('celsius', 'celsius = kelvin; offset: 273.15 = celsius'),
    ('ppbv', 'ppbv = 1 = parts_per_million_by_volume'),
)

for _name, _definition in _DEFINITIONS:
    try:
        getattr(ureg, _name)
    except Exception:
        try:
            ureg.define(_definition)
        except Exception:                     # pragma: no cover - registry said no
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
