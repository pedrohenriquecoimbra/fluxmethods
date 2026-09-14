"""Wind-speed / stability look-up low-pass filtering correction (GEddySoft LPFC=2).

GEddySoft offers two low-pass-filtering corrections:

* ``LPFC = 1`` — cut-off frequency fitted against a Massman reference cospectrum.
  This package already implements that science as the FreqCor-derived
  :func:`~oneflux_preproc.corrections.spectral.main.cutoff_lut` routine.
* ``LPFC = 2`` — a **precomputed** correction factor read from a look-up table
  binned by atmospheric stability (unstable/stable) and mean wind speed. This
  module ports that table lookup (GEddySoft ``correction_factor_lpf`` with a
  ``LUT_CF`` file).

The look-up table (``input_lpfc_LUT_CF_*.csv``) holds, per stability class, a
set of wind-speed bins with a correction factor ``CF_L`` and its uncertainty.
The correction is applied by multiplying the flux by the factor for the bin the
half-hour's mean wind speed falls into.

GEddySoft is developed by Bernard Heinesch and colleagues (University of Liège,
Gembloux Agro-Bio Tech).
"""

# built-in modules
import logging

# 3rd party modules
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)

_BLOCK_COLUMNS = ['class', 'ws_mean', 'ws_max', 'CF', 'unc', 'num']


def _is_number(value):
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_lpfc_lut_table(path):
    """Parse a GEddySoft ``LUT_CF`` CSV into ``{'unstable': df, 'stable': df}``.

    Each block is a table of wind-speed bins with columns ``ws_mean``,
    ``ws_max``, ``CF`` (correction factor) and ``unc`` (its uncertainty).
    """
    raw = pd.read_csv(path, header=None, dtype=str).fillna('')
    blocks = {}
    current, rows = None, []

    def _flush():
        if current and rows:
            df = pd.DataFrame(rows, columns=_BLOCK_COLUMNS[:len(rows[0])])
            blocks[current] = df.astype(float)

    for _, row in raw.iterrows():
        first = str(row[0]).strip()
        if first in ('unstable', 'stable'):
            _flush()
            current, rows = first, []
        elif current is not None and _is_number(first):
            rows.append([v for v in row.tolist()][:len(_BLOCK_COLUMNS)])
    _flush()
    return blocks


#: Stability below which GEddySoft reads the *unstable* block. Not zero: its
#: ``correction_factor_lpf`` tests ``zoL < 0.01``, so a slightly stable half-hour
#: is still corrected as unstable. Transcribed rather than rounded, because the two
#: blocks differ by about 5 % at a given wind speed and the classes either side of
#: the threshold are the populated ones.
UNSTABLE_BELOW = 0.01


def lookup_lpfc_cf(ws, is_unstable, table):
    """Return ``(cf, unc)`` for wind speed ``ws`` and stability sign.

    The first bin whose ``ws_max`` is at or above ``ws`` is used; wind speeds
    beyond the last bin fall back to the last bin.
    """
    key = 'unstable' if is_unstable else 'stable'
    block = table.get(key)
    if block is None or block.empty or not np.isfinite(ws):
        return np.nan, np.nan
    candidates = block[block['ws_max'] >= ws]
    row = candidates.iloc[0] if len(candidates) else block.iloc[-1]
    return float(row['CF']), float(row['unc'])


def _magnitude(da):
    """Return plain float array for a (possibly pint-quantified) DataArray."""
    try:
        return np.asarray(da.pint.magnitude)
    except (AttributeError, TypeError):
        return np.asarray(da.values)


def _mean_wind_speed(ds, wind_speed):
    """The wind speed GEddySoft looks the factor up on.

    Its ``GEddySoft_main`` passes ``sqrt(mean_u^2 + mean_v^2 + mean_w^2)``, the
    magnitude of the mean wind vector. After a double rotation that equals the
    rotated along-wind mean, so ``u`` gives the same number here; it is computed
    from the three components anyway, so the lookup does not silently depend on
    which rotation the run selected.
    """
    if wind_speed is not None:
        return np.atleast_1d(_magnitude(ds[wind_speed]))
    parts = [np.atleast_1d(_magnitude(ds[c])) for c in ('u', 'v', 'w') if c in ds]
    if not parts:
        raise KeyError("lpfc_lut: no wind components (u, v, w) to build a mean "
                       "wind speed from; pass wind_speed=<variable>")
    return np.sqrt(np.sum([np.square(p) for p in parts], axis=0))


def lpfc_lut(ds, table=None, lpfc_filepath=None,
             wind_speed=None, stability='z_L', unstable_below=UNSTABLE_BELOW,
             factor=None, **kwargs):
    """Attach the look-up low-pass correction factor ``scf_lpfc_lut`` per period.

    Parameters
    ----------
    ds : xarray.Dataset
        Averaged dataset carrying a per-``date`` mean wind speed and stability.
    table : dict, optional
        Preloaded table (see :func:`load_lpfc_lut_table`). If omitted,
        ``lpfc_filepath`` is loaded.
    lpfc_filepath : str, optional
        Path to a GEddySoft ``LUT_CF`` CSV.
    wind_speed : str, optional
        Variable to look the factor up on. Omitted, the magnitude of the mean wind
        vector is built from ``u``, ``v`` and ``w``, which is what GEddySoft passes.
    stability : str
        Variable name for the stability parameter (``z_L``).
    unstable_below : float
        Stability below which the unstable block is read (GEddySoft: 0.01).
    factor : callable, optional
        ``factor(ws, stability, table)`` returning the correction factor for one window,
        in place of the table lookup. The bridged twin passes GEddySoft's own
        routine here; everything else -- which wind speed, which stability, which
        windows, and the variables written -- stays this function's. The
        uncertainty column is still read from the table, the engine's routine
        reporting none.
    """
    if table is None:
        path = lpfc_filepath or ds.attrs.get('Files', {}).get('lpfc_filepath', '')
        if not path:
            logger.warning("lpfc_lut: no LUT table or path provided; skipping.")
            return ds
        table = load_lpfc_lut_table(path)

    ws = _mean_wind_speed(ds, wind_speed)
    stab = np.atleast_1d(_magnitude(ds[stability]))
    if not np.any(np.isfinite(stab)):
        # Every window would silently take the stable branch, which is a 5 % error
        # on this table rather than a missing value. Say so instead.
        logger.warning(
            "lpfc_lut: %r is entirely non-finite, so every window reads the "
            "stable block. Check that the measurement height reached the run "
            "(zm), since z_L is z-d over the Obukhov length.", stability)

    # The two need not come from the same variable, so a stability
    # declared once for the run (or per site) has to broadcast against the
    # per-window wind speed rather than index out of range.
    try:
        stab = np.broadcast_to(stab, ws.shape)
    except ValueError as exc:
        raise ValueError(
            f"lpfc_lut: {stability!r} has shape {stab.shape}, which does not "
            f"broadcast against the wind speed's {ws.shape}") from exc

    cf = np.full(ws.shape, np.nan)
    unc = np.full(ws.shape, np.nan)
    flat_ws, flat_stab = ws.ravel(), np.asarray(stab).ravel()
    flat_cf, flat_unc = cf.ravel(), unc.ravel()
    for i in range(flat_ws.size):
        flat_cf[i], flat_unc[i] = lookup_lpfc_cf(
            flat_ws[i], flat_stab[i] < unstable_below, table)
        if factor is not None:
            flat_cf[i] = factor(flat_ws[i], flat_stab[i], table)
    cf, unc = flat_cf.reshape(ws.shape), flat_unc.reshape(ws.shape)

    shape = ds[wind_speed].shape if wind_speed is not None else ds['u'].shape
    dims = ds[wind_speed].dims if wind_speed is not None else ds['u'].dims
    ds['scf_lpfc_lut'] = xr.DataArray(cf.reshape(shape), dims=dims)
    ds['scf_lpfc_lut'].attrs.update({
        'description': 'Low-pass filtering correction factor (GEddySoft LUT_CF)',
        'method': 'lowpass_lut_wind_stability'})
    ds['scf_lpfc_lut_unc'] = xr.DataArray(unc.reshape(shape), dims=dims)
    return ds
