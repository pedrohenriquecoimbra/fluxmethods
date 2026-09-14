"""The one name-resolution helper the estimators here share.

A copy of ``resolve_variable`` from ``oneflux_preproc/core/utils.py``: a method
that takes ``co2`` or the array itself should not care which it was given. The
rest of that module is config parsing and path handling, which is a program's
business rather than a method's.
"""

def resolve_variable(var, ds):
    """
    Resolves a variable input which may be a string (lookup in ds) or a DataArray.
    """
    if isinstance(var, str):
        if var not in ds:
            raise ValueError(f"Variable '{var}' not found in dataset.")
        return ds[var]
    return var
