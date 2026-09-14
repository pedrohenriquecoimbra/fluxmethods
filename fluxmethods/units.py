"""The one unit conversion the estimators here need.

A copy of ``convert_unit`` from ``oneflux_preproc/core/units.py``. It is the
whole of that module this collection takes: the rest of it builds a pint
registry, defines site-specific units and normalises a dataset, which is
policy about how data is expressed rather than a method.

It needs no import of its own -- it calls ``.to()`` on the pint Quantity the
caller already has, so pint is a dependency of the *data*, not of this file.
"""

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
