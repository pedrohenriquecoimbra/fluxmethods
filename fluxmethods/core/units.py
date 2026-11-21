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

**A host that keeps a registry of its own names it** (:func:`adopt`). The
application registry is whatever the process has installed *at the moment of a
call*, and a program that installs its own only for the length of a run would
otherwise have these methods compute on pint's default registry everywhere else
-- where ``ppt`` is a volume. A named registry is the one the methods compute on
from then on, whichever is the application's; naming none leaves the behaviour
above.

Everything below the registry is that module, unchanged.
"""

import logging

import pint
from pint.errors import UndefinedUnitError
import xarray as xr

logger = logging.getLogger(__name__)

class _Registry:
    """The registry the methods compute on: the application's, or a named one.

    Everything is forwarded to the registry in force when it is asked, so a
    module that imported :data:`ureg` once follows a later :func:`adopt` -- and,
    with nothing adopted, a later ``pint.set_application_registry``.
    """

    __slots__ = ("_named",)

    def __init__(self):
        object.__setattr__(self, "_named", None)

    def get(self):
        """The registry itself, which is what a quantity has to be made on."""
        named = object.__getattribute__(self, "_named")
        return named if named is not None else pint.get_application_registry().get()

    def __getattr__(self, name):
        return getattr(self.get(), name)

    def __setattr__(self, name, value):
        setattr(self.get(), name, value)

    def __dir__(self):
        return dir(self.get())

    def __call__(self, *args, **kwargs):
        return self.get()(*args, **kwargs)

    def __contains__(self, item):
        return item in self.get()

    def __iter__(self):
        return iter(self.get())


ureg = _Registry()


def adopt(registry):
    """Name the registry these methods compute on; return the one named before.

    For a host that keeps a registry of its own and does not make it the
    application's for good. ``None`` goes back to following the application
    registry. The vocabulary below was checked on the registry in force when
    this module was imported, so a host names the registry it imported under.
    """
    previous = object.__getattribute__(ureg, "_named")
    if isinstance(registry, (_Registry, pint.ApplicationRegistry)):
        registry = registry.get()
    object.__setattr__(ureg, "_named", registry)
    return previous

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

def quantify(obj, units=None, **unit_kwargs):
    """``obj.pint.quantify(...)`` on :data:`ureg`, whatever the application uses.

    The accessor's own default is the application registry, which is not the one
    in force once a host has named another (:func:`adopt`).
    """
    args = () if units is None else (units,)
    return obj.pint.quantify(*args, unit_registry=ureg.get(), **unit_kwargs)


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

def _is_angle(q):
    """Whether ``q`` is an angle (degree, radian, ...).

    pint counts an angle as dimensionless, so it converts to ``dimensionless``
    without complaint -- a degree becoming its value in radians under a label that
    no longer says angle. Its root units still name the radian, which is what tells
    it from a ratio.
    """
    try:
        return any(name in ('radian', 'steradian')
                   for name, _ in ureg.Quantity(1, q.units).to_root_units().unit_items())
    except Exception:  # noqa: BLE001 - not a quantity: not an angle either
        return False


def convert_to_prefered_units(ds, units=[
    "m", "s", "μmol/m²/s", "m/s", "W/m²", "g/m²/s", "kg/m^3", "m^3/mol", "J/kg/K", "dimensionless"]):
    """Each variable in the first of ``units`` it converts to; the rest as they are.

    An angle is left as it is: it would otherwise reach ``dimensionless`` and be
    published in radians with nothing saying so (``wind_dir`` in particular).
    """
    array_input = False
    if not isinstance(ds, (xr.Dataset, xr.DataArray)):
        raise ValueError("Input must be an xarray Dataset or DataArray")
    if isinstance(ds, xr.DataArray):
        ds = ds.to_dataset(name='data')
        array_input = True
    for var in ds.data_vars:
        if _is_angle(ds[var].data):
            continue
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
