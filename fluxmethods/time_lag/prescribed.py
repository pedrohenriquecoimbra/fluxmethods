"""Prescribed / table-driven time-lag detection, ported from GEddySoft.

GEddySoft (Bernard Heinesch, University of Liège, Gembloux Agro-Bio Tech)
supports three table-driven refinements of the time lag beyond covariance
maximisation:

* **PRESCRIBED** — a per-half-hour lag read from a file; when the running
  half-hour is missing, the median of the 10 nearest non-NaN lags is used.
  The tabulated value is the *total* lag (physical + clock drift) and is used
  as-is — see :func:`prescribed_time_lag`.
* **clock drift** — a time-dependent offset (``TDC - computer``) compensating
  acquisition-clock drift. It belongs to the *nominal* lag centre used by the
  CONST/MAX methods; it is already contained in the PRESCRIBED table, so it is
  not added on top of a tabulated lag (GEddySoft v4.1 double-correction fix).
* **RH dependency** — for VOC tracers, the expected lag is looked up from a
  relative-humidity table keyed by ``m/z``.

The lag values in the prescribed file are expressed in seconds; multiplying by
the (final) acquisition frequency yields the shift in samples (e.g. ``-13 s``
at ``10 Hz`` -> ``-130`` samples, matching ``LAG_WINDOW_CENTER``).

These helpers are pure and unit-tested here; wiring the file paths and the
per-period date / RH through the pipeline is done alongside the meteo and
tracer readers.
"""

# built-in modules
import logging
import re

# 3rd party modules
import numpy as np
import pandas as pd
import xarray as xr

logger = logging.getLogger(__name__)


def load_prescribed_lag_table(path, value_col=None):
    """Load a GEddySoft prescribed-lag CSV as a ``pd.Series`` of seconds.

    The file has a datetime first column (``DD/MM/YYYY HH:MM``) and one lag
    column (per ``m/z``). A half-hour with an empty cell is dropped, as
    GEddySoft's own reader drops it: the fallback for a half-hour that is not in
    the table is the median of the ten nearest *rows*, so a table that still
    carried the empty ones would count them among the ten and take the median of
    a set with no value in it.
    """
    df = pd.read_csv(path)
    time_col = df.columns[0]
    if value_col is None:
        value_col = df.columns[1]
    index = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    series = pd.Series(pd.to_numeric(df[value_col], errors='coerce').values,
                       index=index, name=value_col)
    return series[series.index.notna()].dropna().sort_index()


def load_clock_drift_table(path):
    """Load a GEddySoft clock-drift (TDC) CSV as a ``pd.Series`` of seconds.

    The offset is the ``TDC - computer`` column (second column); GEddySoft adds
    it (with a sign convention already baked into the file) to the lag.
    """
    df = pd.read_csv(path)
    time_col = df.columns[0]
    value_col = df.columns[1]
    index = pd.to_datetime(df[time_col], dayfirst=True, errors='coerce')
    series = pd.Series(pd.to_numeric(df[value_col], errors='coerce').values,
                       index=index, name='clock_drift')
    return series[series.index.notna()].sort_index()


def load_rh_dependency_table(path):
    """Load a GEddySoft RH-dependency CSV as a ``pd.DataFrame`` indexed by RH.

    The first row is a free-text note; the real header (``RH (%)``, ``mz ...``)
    follows. Columns are named by ``m/z`` and values are lags in samples.
    """
    df = pd.read_csv(path, skiprows=1)
    df = df.set_index(df.columns[0])
    df.index = pd.to_numeric(df.index, errors='coerce')
    df = df[df.index.notna()]
    df.columns = [_column_mz(c) for c in df.columns]
    return df.apply(pd.to_numeric, errors='coerce')


def _column_mz(col):
    """Extract the numeric ``m/z`` from a column label like ``'mz 33.034'``."""
    token = str(col).replace('mz', '').strip()
    try:
        return float(token)
    except ValueError:
        return col


def series_mz(series):
    """The m/z a prescribed-lag series is for, from its value-column name.

    A GEddySoft prescribed table is written per species -- the bundled one is
    ``input_lag_prescribed_69.csv``, whose value column is ``mz 69.069901`` --
    so the file itself says which tracer its lags belong to. Returns ``None``
    when the column carries no parseable mass, which is the single-species case
    where the caller has nothing to match against.
    """
    if series is None:
        return None
    mz = _column_mz(getattr(series, 'name', None))
    return float(mz) if isinstance(mz, (int, float)) else None


def _tracer_mz(name):
    """The m/z of a ``tracer_<mz>`` variable, or ``None`` for anything else."""
    match = re.fullmatch(r'tracer_(\d+(?:\.\d+)?)', str(name or ''))
    return float(match.group(1)) if match else None


def resolve_prescribed_lag(date, table, n_nearest=10):
    """Return the prescribed lag (seconds) for ``date``.

    Uses the exact half-hour if present and non-NaN; otherwise the median of the
    ``n_nearest`` closest non-NaN entries in time (GEddySoft's fallback).
    """
    date = pd.Timestamp(date)
    valid = table.dropna()
    if valid.empty:
        return np.nan
    if date in valid.index:
        return float(valid.loc[date])
    deltas = np.abs(valid.index - date)
    nearest = valid.iloc[np.argsort(deltas.values)[:n_nearest]]
    return float(np.median(nearest.values))


def resolve_rh_lag(rh, mz, table, tol=0.001):
    """Return the RH-dependent lag (samples) for humidity ``rh`` and ``mz``.

    The nearest RH row is used; the ``m/z`` column is matched within ``tol``.
    Returns NaN when no column matches.
    """
    rh = float(np.clip(rh, table.index.min(), table.index.max()))
    columns = np.array([c for c in table.columns if isinstance(c, (int, float))],
                       dtype=float)
    if columns.size == 0:
        return np.nan
    j = int(np.argmin(np.abs(columns - float(mz))))
    if abs(columns[j] - float(mz)) > tol:
        return np.nan
    row_idx = int(np.argmin(np.abs(np.asarray(table.index, dtype=float) - rh)))
    return float(table.iloc[row_idx][columns[j]])


def _acq_freq(move, acq_freq=None):
    if acq_freq:
        return float(acq_freq)
    return float(1.0 / np.abs(move.time.diff('time').mean().to_numpy()))


def prescribed_time_lag(move, fix=None, lag_table=None, date=None,
                        tlag=0, acq_freq=None, lag_samples=None, **kwargs):
    """Shift ``move`` by a prescribed (table-driven) lag; return an ``xr.Dataset``.

    ``lag_table`` is a :func:`load_prescribed_lag_table` series (seconds).
    ``date`` defaults to the scalar ``date`` coordinate of ``move``. When no
    table is supplied, the constant ``tlag`` is used as a fallback.

    Two sign conventions meet here and are deliberately kept apart. A GEddySoft
    prescribed-lag file is written in *its* shift convention (a scalar lagging the
    wind is tabulated negative, so ``-13`` s at 10 Hz is ``-130`` samples, matching
    ``LAG_WINDOW_CENTER``) and is applied as tabulated. The ``tlag`` fallback comes
    from this package's own configuration and is therefore a *physical* lag in
    seconds, positive when the scalar lags (see :mod:`~.time_lag.commons`). The reported
    ``{name}_time_lag`` is physical, so it is comparable across methods.

    The tabulated value is used as-is: a GEddySoft prescribed-lag file already
    holds the *total* lag (physical + clock drift), so the clock-drift offset
    must NOT be added again. Doing so double-corrects the lag — the error
    GEddySoft itself fixed in v4.1 ("input files containing physical +
    clock-drift lag no longer receive additional clock-drift correction").

    ``lag_samples(table, date, freq)`` is the table rule itself -- the exact
    half-hour, or the fallback when it is absent -- answering in samples. The
    bridged twin passes the engine's own; the sign conventions, the drift put back
    and the three columns reported stay this function's.
    """
    freq = _acq_freq(move, acq_freq)

    if date is None and 'date' in move.coords:
        date = pd.Timestamp(np.asarray(move['date'].values).ravel()[0])

    from .commons import seconds_to_shift, shift_to_seconds

    def _fallback_shift():
        shift = seconds_to_shift(tlag, freq)
        return -int(round(float(tlag or 0))) if shift is None else shift

    # A prescribed table is written for one species (its value column names the
    # m/z), so a run that selects several tracers must not give them all the lag
    # of the one the file is for. Mismatches fall back, and say so: the table
    # simply does not cover this species.
    table_mz = series_mz(lag_table)
    want_mz = _tracer_mz(getattr(move, 'name', None))
    covers = (lag_table is not None
              and (table_mz is None or want_mz is None
                   or abs(table_mz - want_mz) < 0.01))

    # The acquisition-clock offset ``lag_window: clock_drift`` already took off the
    # samples, read the way :func:`~.maximisation.return_time_lag` reads it.
    attr = move.attrs.get('Attr')
    drift = 0.0
    if isinstance(attr, dict):
        try:
            drift = float(attr.get('time_lag_clock_drift', 0.0) or 0.0)
        except (TypeError, ValueError):
            drift = 0.0

    if lag_table is not None and date is not None and covers and lag_samples:
        # A substituted rule answers in samples, so the drift goes back in samples
        # too; the default path below converts from seconds as it always has.
        tabulated = float(lag_samples(lag_table, date, freq))
        if not np.isfinite(tabulated):
            logger.warning("No prescribed lag resolved for %s; using default.", date)
            samples = _fallback_shift()
        else:
            samples = int(round(tabulated + drift * freq))
    elif lag_table is not None and date is not None and covers:
        lag_seconds = resolve_prescribed_lag(date, lag_table)
        if np.isnan(lag_seconds):
            logger.warning("No prescribed lag resolved for %s; using default.", date)
            samples = _fallback_shift()
        else:
            # The table holds the *total* lag -- physical + clock drift -- in
            # GEddySoft's shift convention, and GEddySoft applies it to a series
            # that still carries the drift. This one does not: ``clock_drift`` is a
            # `lag_window` step that corrects the clock on the samples, so applying
            # the tabulated total here would take the drift off a second time. On
            # the bundled BE-Vie sample that is a shift of the same order as the
            # whole search window -- tens of samples at 10 Hz, and growing across
            # the sample -- and it is the very double-correction GEddySoft fixed
            # in v4.1, arrived at from the other direction. The per-window size is
            # what ``clock_drift`` records as ``{name}_time_lag_clock_drift``,
            # checked against GEddySoft's own record by
            # ``test_geddysoft_intermediates.py::test_the_clock_drift_matches_every_window``.
            #
            # So the drift is added back into the shift, leaving the physical part,
            # and reported beside it: ``{name}_time_lag`` is physical like every
            # other method's, and physical + ``{name}_time_lag_clock_drift`` is
            # GEddySoft's ``lagtime``, which is what the comparison asserts.
            samples = int(round((lag_seconds + drift) * freq))
    else:
        # Falling back is a real answer -- the configured centre -- but it is not
        # a *prescribed* one, so the reason is named in the warning below rather
        # than reported as though the table had been used.
        if lag_table is None:
            why = "no prescribed table was supplied"
        elif not covers:
            why = (f"the table is for m/z {table_mz:g} and this is "
                   f"{want_mz:g}" if want_mz is not None else
                   f"the table is for m/z {table_mz:g}")
        else:
            why = "the window carries no date to look up"
        logger.warning(
            "prescribed: %s, so %s keeps the configured lag centre (%s s) rather "
            "than a tabulated one.", why, getattr(move, 'name', 'the series'), tlag)
        samples = _fallback_shift()

    moved = move.shift(time=samples).to_dataset()
    steps2time = xr.ones_like(move.mean('time'))
    lag_s = shift_to_seconds(samples, freq)
    # The same three columns, meaning the same things, as the maximisation methods
    # report -- otherwise a run that switched method could not be compared against
    # one that did not, and the drift would be readable from one and not the other.
    reported = {
        f'{move.name}_time_lag': (
            lag_s, 'physical lag applied, positive when the scalar lags the wind'),
        f'{move.name}_time_lag_opt': (
            lag_s, 'the prescribed lag; this method does not search'),
        f'{move.name}_time_lag_clock_drift': (
            drift, 'acquisition-clock offset removed from the series before the '
                   'lag was applied; add to the physical lag for the lag in the '
                   'recorded time base (GEddySoft "lagtime")'),
    }
    moved = moved.assign(**{k: steps2time * v for k, (v, _) in reported.items()})
    for name, (_, description) in reported.items():
        moved[name].attrs.update(units='s', long_name=description)
    return moved
