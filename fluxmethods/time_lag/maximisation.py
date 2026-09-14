"""Covariance-maximisation time-lag detection.

The lag between the vertical wind ``fix`` (w) and a scalar ``move`` is the
integer sample shift at the covariance extremum of ``fix`` with the shifted
``move`` (GEddySoft ``MAX`` / ``MAX_WITH_DEFAULT``) — a maximum or minimum
according to the sign of the mean covariance over the search window.

The peak is searched within the *inner* window ``[tlag_min, tlag_max]`` (=
``LAG_WINDOW_CENTER ± LAG_INNER_WINDOW_SIZE``). GEddySoft's wider outer window
pads the optional Hamming covariance smoothing (``LAG_COVPEAK_FILTER_LENGTH``,
via ``covpeak_filter_length``); it does not widen the peak search itself.

Unit convention (shared with the ``fixed`` and ``prescribed`` methods): every lag
crossing this module's boundary — the bounds ``tlag`` / ``tlag_min`` / ``tlag_max``
and the reported ``{name}_time_lag`` / ``{name}_time_lag_opt`` — is a **physical
lag in seconds**, positive when the scalar arrives after the wind. See
:mod:`~.time_lag.commons`: the conversion to the integer sample shifts used internally (and
the accompanying sign flip) happens once, on entry, using the acquisition
frequency. A configuration is therefore portable across acquisition rates.
"""

import numpy as np
import xarray as xr
from scipy.signal import filtfilt

from .commons import acq_freq as _acq_freq_of, seconds_to_shift, shift_to_seconds


def default_lag(length=711, diameter=5.3, pump=15, dt=20):
    """Tube-transit default lag in samples (geometry / flow)."""
    return int(np.round((length * (np.pi * (diameter / 2) ** 2) * 1e-6 / pump) * 60 * dt))


def _ensure_scalar(value):
    """Coerce scipy/xarray/array outputs to a plain float."""
    if np.isscalar(value):
        return float(value)
    arr = np.asarray(value).ravel()
    return float(arr[0]) if arr.size else 0.0


def _acq_freq(move):
    """Acquisition frequency [Hz] from the ``time`` coordinate spacing."""
    dt = np.abs(float(move['time'].diff('time').mean().to_numpy()))
    return 1.0 / dt if np.isfinite(dt) and dt > 0 else np.nan


def lag_series(da, shift_samples):
    """Shift ``da`` along ``time`` by an integer number of samples.

    A positive shift ``T`` maps to ``da.shift(time=T)`` (≈ ``np.roll(da, T)``),
    matching the ``fixed`` / ``prescribed`` / GEddySoft lag-sign convention.
    """
    return da.shift(time=int(round(_ensure_scalar(shift_samples))))


def _cov_at_lag(f, m, T, n):
    """Signed covariance of ``f[i]`` with ``shift(m, T)[i] = m[i-T]`` (NaN-aware).

    The overlap only: at lag ``T`` the two series share ``n - |T|`` samples, and a
    pair is used when *both* of its members are finite.
    """
    if abs(T) >= n:
        return np.nan
    if T >= 0:
        a, b = f[T:], m[:n - T]
    else:
        a, b = f[:n + T], m[-T:]
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 2:
        return np.nan
    return float(np.cov(a[mask], b[mask])[0, 1])


def _cov_at_lag_geddysoft(f, m, T, n):
    """GEddySoft's ``xcov`` at one lag, transcribed with its wraparound.

    Upstream shifts with ``np.roll`` and then slices the wrapped samples off. That
    works for a **positive** lag, where ``yshifted[T:]`` is ``y[0:n-T]``. The
    negative branch slices ``[0 : len(x) - 1 - T]``, which for ``T = -L`` is
    ``[0 : n-1+L]`` -- longer than the array, so it clips to the whole of it and
    keeps the ``L`` samples ``np.roll`` brought round from the far end. Its
    docstring says it "avoids circular effects from numpy's roll operation"; that
    holds only above zero.

    This matters here rather than in principle: a VOC search is centred on
    ``lag_phys_wd_center + drift``, about -108 samples on the bundled site, so
    *every* lag it evaluates takes the wrapping branch. The wrapped fraction grows
    with ``|T|``, so the bias is not a constant offset -- it tilts the covariance
    across the search window, which moves the peak as well as biasing its height.

    One smaller departure comes with it, and is transcribed rather than tidied:
    ``T = 0`` takes the ``else`` branch, whose ``[0 : n-1]`` drops the last
    sample. A pair counts only when both of its members are finite, which is
    v4.1's rule; v4.0 counted on ``x`` alone.

    The **joint trim comes first**, and it is not cosmetic. Upstream cuts both
    series to the scalar's finite span before it builds the curve at all
    (the ``trim w_prime and c_prime for NaNs`` block that opens ``GEddySoft_main``'s
    tracer covariance step), so the trimmed length is what sets ``n``, the
    normalisation, and above all how many samples ``np.roll`` brings round. Left
    untrimmed the wrapped fraction is computed against 18000 rather than against
    the ~17930 the scalar actually covers, which shifts every point of the curve
    by roughly ``|T| / n`` -- a few tenths of a percent on this site, enough to
    move the peak. Transcribing the wraparound without the trim reproduces
    neither the curve nor the lag.

    Registered as the ``geddysoft`` covariance kernel so the cost of it can be
    measured against :func:`_cov_at_lag`, which is the correct computation. It is
    not the default: this is a defect, not a convention.
    """
    ok = np.isfinite(m)
    if not ok.any():
        return np.nan
    first = int(np.argmax(ok))
    last = int(len(ok) - np.argmax(ok[::-1]) - 1)
    f, m = f[first:last + 1], m[first:last + 1]
    n = len(f)
    if abs(T) >= n or n < 2:
        return np.nan
    shifted = np.roll(m, T)
    part = slice(T, None) if T > 0 else slice(0, len(f) - 1 - T)
    a, b = f[part], shifted[part]
    mask = np.isfinite(a) & np.isfinite(b)
    count = int(mask.sum())
    if count < 2:
        return np.nan
    a, b = a[mask], b[mask]
    return float((np.sum(a * b) - np.sum(a) * np.sum(b) / count) / (count - 1))


#: Selectable ways of evaluating the covariance at a lag. ``pairwise`` is the
#: correct one and the default; ``geddysoft`` reproduces upstream's, wraparound
#: included, so a comparison can say what that costs instead of arguing about it.
XCOV_KERNELS = {
    'pairwise': _cov_at_lag,
    'geddysoft': _cov_at_lag_geddysoft,
}


def xcov_kernel(name=None):
    """The covariance kernel called ``name``; the correct one when unnamed."""
    if name is None or not str(name).strip():
        return _cov_at_lag
    try:
        return XCOV_KERNELS[str(name).strip().lower()]
    except KeyError:
        raise KeyError(
            f"unknown covariance kernel {name!r}; registered: "
            f"{', '.join(sorted(XCOV_KERNELS))}") from None


def _smooth_hamming(cov, filter_length):
    """Zero-phase Hamming smoothing of a covariance array (GEddySoft).

    Mirrors GEddySoft ``find_covariance_peak``: a normalised Hamming FIR applied
    forward-backward with ``filtfilt`` (``padtype='odd'``, ``padlen=3*(L-1)``).
    Falls back to the unsmoothed array when it is too short for the filter or
    contains non-finite values.
    """
    L = int(filter_length)
    if L <= 1 or not np.all(np.isfinite(cov)) or len(cov) <= 3 * (L - 1):
        return cov
    b = np.hamming(L) / np.hamming(L).sum()
    return filtfilt(b, [1.0], cov, padtype='odd', padlen=3 * (L - 1))


def _scan_peak(fix, move, lo, hi, filter_length=1, ext=None, kernel=None):
    """Integer lag ``T`` in [lo, hi] (samples) at the covariance extremum.

    The covariance ``cov(fix[i], shift(move, T)[i])`` is evaluated over [lo, hi]
    — or the wider ``ext`` extent when a Hamming ``filter_length > 1`` is given,
    so the smoother sees real data at the inner edges — then optionally smoothed,
    and the extremum taken within [lo, hi]. Following GEddySoft, the extremum is a
    maximum when the mean covariance over the window is positive and a minimum
    otherwise. Matches :func:`lag_series`, so the returned lag is applied with the
    same sign. NaN-aware.

    Both ends are inclusive, ``np.arange(lo, hi + 1)``: a window ``centre ± I``
    holds ``2I + 1`` lags. That is the loop EddyPro codes -- ``do i = lagmin,
    lagmax`` in ``CovMax`` (``src/src_rp/timelag_handle.f90``, the bounds being
    ``nint`` of the declared seconds times the rate) -- and the window GEddySoft's
    ini documents; GEddySoft's code slices ``2I`` lags, one short at the top
    (see :mod:`.geddysoft`).
    """
    lo, hi = int(round(lo)), int(round(hi))
    if hi < lo:
        lo, hi = hi, lo
    f = np.asarray(fix.values, dtype=float).ravel()
    m = np.asarray(move.values, dtype=float).ravel()
    n = len(f)

    ext_lo, ext_hi = (lo, hi) if ext is None else (int(round(ext[0])), int(round(ext[1])))
    ext_lo, ext_hi = min(ext_lo, lo), max(ext_hi, hi)
    at_lag = kernel or _cov_at_lag
    lags = np.arange(ext_lo, ext_hi + 1)
    cov = np.array([at_lag(f, m, int(T), n) for T in lags])
    cov = _smooth_hamming(cov, filter_length)

    inner = (lags >= lo) & (lags <= hi)
    inner_lags, inner_cov = lags[inner], cov[inner]
    if not np.any(np.isfinite(inner_cov)):
        return int(round((lo + hi) / 2))
    if np.nanmean(inner_cov) >= 0:
        return int(inner_lags[int(np.nanargmax(inner_cov))])
    return int(inner_lags[int(np.nanargmin(inner_cov))])


def _ext_window(tlag, lo, hi, outer, filter_length):
    """Covariance-evaluation extent [lo, hi] widened to pad Hamming smoothing.

    Uses the GEddySoft outer window ``center ± outer`` when supplied, else a
    margin of the filter's padding length; returns ``None`` when no smoothing is
    requested (evaluate over the inner window only).
    """
    if not (filter_length and int(filter_length) > 1):
        return None
    if outer:
        return float(tlag) - abs(float(outer)), float(tlag) + abs(float(outer))
    pad = 3 * (int(filter_length) - 1) + 1
    return lo - pad, hi + pad


def _inner_window(tlag, tlag_min, tlag_max, half_default=100):
    """Peak-search window ``[lo, hi]`` in samples, always in ascending order.

    GEddySoft searches the covariance peak within the *inner* window
    ``center ± INNER`` — supplied here as ``[tlag_min, tlag_max]`` (the adapter
    sets them to ``LAG_WINDOW_CENTER ± LAG_INNER_WINDOW_SIZE``). The wider outer
    window only exists in GEddySoft to pad the Hamming smoothing and for plots;
    it does not widen the peak search.

    The bounds are returned sorted because a perfectly valid window can arrive
    descending: a lag window imported from EddyPro is negated into this module's
    shift convention, which maps ``min_timelag < max_timelag`` (seconds) onto
    ``tlag_min > tlag_max`` (samples). Callers compare the located peak against
    these bounds, so an unordered pair would make every interior peak test as
    "on the boundary" and silently force the ``maxcov&default`` fallback.
    """
    if tlag_min is not None and tlag_max is not None:
        lo, hi = float(tlag_min), float(tlag_max)
        return (lo, hi) if lo <= hi else (hi, lo)
    return float(tlag) - half_default, float(tlag) + half_default


def return_time_lag(move, opt_samples, opt_use_samples=None):
    """Apply the integer-sample shift and record the lag it applied, in parts.

    Three values, all in seconds and positive when the scalar lags the wind:

    * ``{name}_time_lag`` -- the **physical** lag, the transit time down the path.
      This is the quantity a configuration declares (``LAG_WINDOW_CENTER``,
      EddyPro's ``nom_timelag``), so a reported lag can be read against the
      metadata that predicted it.
    * ``{name}_time_lag_clock_drift`` -- the acquisition-clock error taken off the
      series before the search, from the run's own drift table. Zero when no drift
      was corrected, which is a fact rather than an absence.
    * ``{name}_time_lag_opt`` -- the raw scanned peak, before any fallback, for
      diagnosing a window that fell back.

    The parts are kept apart rather than summed because they are different
    quantities: one is a property of the tubing, the other of a clock. GEddySoft
    reports the sum as ``lagtime`` and the second part as ``lagtime_clock_drift``,
    so its total is this pair added together -- which the GEddySoft comparison
    asserts, on the total *and* on each part.
    """
    opt_samples = _ensure_scalar(opt_samples)
    opt_use_samples = (_ensure_scalar(opt_use_samples)
                       if opt_use_samples is not None else opt_samples)
    moved = lag_series(move, opt_use_samples).to_dataset()

    freq = _acq_freq(move)
    # A series the clock-drift step realigned was searched with the clock error
    # already taken out, so the peak it found is the *physical* lag -- the tube
    # transit time, which is the quantity the configuration declares
    # (``LAG_WINDOW_CENTER``, EddyPro's ``nom_timelag``) and the one an instrument's
    # metadata can be checked against. That is what is reported.
    #
    # The clock error is reported beside it rather than folded into it, so a run
    # states the parts its lag is made of instead of one number that means different
    # things depending on whether a drift table was configured. GEddySoft folds them:
    # its ``lagtime`` is the peak of a search centred on ``lag_phys_wd_center +
    # lag_clock_drift_samples``, so it is physical plus clock error, with
    # ``lagtime_clock_drift`` stored alongside to be subtracted back out. Its total
    # is therefore this pair's sum, which is what the comparison asserts.
    attr = move.attrs.get('Attr')
    drift = 0.0
    if isinstance(attr, dict):
        try:
            drift = float(attr.get('time_lag_clock_drift', 0.0) or 0.0)
        except (TypeError, ValueError):
            drift = 0.0

    steps2time = xr.ones_like(move.mean('time'))
    reported = {
        f'{move.name}_time_lag': (
            shift_to_seconds(opt_use_samples, freq),
            'physical lag applied, positive when the scalar lags the wind'),
        f'{move.name}_time_lag_opt': (
            shift_to_seconds(opt_samples, freq),
            'raw covariance-peak lag, before any fallback'),
        f'{move.name}_time_lag_clock_drift': (
            drift, 'acquisition-clock offset removed from the series before the '
                   'search; add to the physical lag for the lag in the recorded '
                   'time base (GEddySoft "lagtime")'),
    }
    moved = moved.assign(**{k: steps2time * v for k, (v, _) in reported.items()})
    # Said on the variable, not only in the docs: these are seconds, and the one
    # place that could have made them samples now returns NaN instead.
    for name, (_, description) in reported.items():
        moved[name].attrs.update(units='s', long_name=description)
    return moved


def _window_in_samples(move, tlag, tlag_min, tlag_max, acq=None, lag_units='seconds'):
    """Translate the configured lag window into integer sample shifts.

    ``lag_units`` declares what the caller supplied ('seconds', the configuration
    convention, or 'samples' for a caller already working in shifts), so the
    translation never has to be inferred.
    """
    freq = _acq_freq_of(move, acq)
    centre = seconds_to_shift(tlag, freq, lag_units)
    return (0 if centre is None else centre,
            seconds_to_shift(tlag_min, freq, lag_units),
            seconds_to_shift(tlag_max, freq, lag_units))


def time_lag(move, fix, tlag=0, tlag_min=None, tlag_max=None, outer=None,
             covpeak_filter_length=1, acq_freq=None, lag_units='seconds',
             xcov=None, **kwargs):
    """MAX: covariance peak over the inner window ``[tlag_min, tlag_max]`` (seconds).

    The window is a physical lag range — e.g. an analyser declaring a nominal
    0.40 s lag with 0.15–0.65 s bounds — converted to sample shifts on entry.

    ``covpeak_filter_length`` (samples, GEddySoft ``LAG_COVPEAK_FILTER_LENGTH``)
    optionally Hamming-smooths the covariance before the peak search; ``outer``
    (samples) widens the covariance-evaluation extent for that smoothing.
    """
    tlag, tlag_min, tlag_max = _window_in_samples(
        move, tlag, tlag_min, tlag_max, acq_freq, lag_units)
    lo, hi = _inner_window(tlag, tlag_min, tlag_max)
    ext = _ext_window(tlag, lo, hi, outer, covpeak_filter_length)
    peak = _scan_peak(fix, move, lo, hi, covpeak_filter_length, ext,
                      kernel=xcov_kernel(xcov))
    return return_time_lag(move, peak)


def time_lag_w_default(move, fix, tlag=0, tlag_min=None, tlag_max=None, outer=None,
                       covpeak_filter_length=1, acq_freq=None, lag_units='seconds',
                       xcov=None, **kwargs):
    """MAX_WITH_DEFAULT: covariance peak over the inner window, with a fallback.

    The window ``[tlag_min, tlag_max]`` and the nominal ``tlag`` are physical lags
    in **seconds** (see :mod:`~.time_lag.commons`), converted to sample shifts on entry.

    The peak is searched within ``[tlag_min, tlag_max]`` (= ``LAG_WINDOW_CENTER ±
    LAG_INNER_WINDOW_SIZE``). If it lands on either boundary of that window the
    maximum is deemed unreliable — the true peak is most likely outside the
    window — so the nominal lag ``tlag`` (the window centre) is used instead
    (GEddySoft ``MAX_WITH_DEFAULT``).

    ``{name}_time_lag`` records the lag actually applied; ``{name}_time_lag_opt``
    keeps the raw scanned peak for diagnostics.

    Note on the boundary condition: GEddySoft searches the peak over lags in
    ``[center-INNER, center+INNER-1]`` (``find_covariance_peak.py``) but tests for
    the edge with ``lag == center-INNER+1 or lag == center+INNER``
    (``compute_time_lag.py``). Neither
    reachable edge is therefore flagged, and ``center+INNER`` cannot occur at all, so
    the fallback to the nominal lag fires only for a peak one sample inside the lower
    edge. This implementation uses the both-edges condition the fallback describes,
    which makes it fire more often than GEddySoft's on the same window.
    EddyPro's ``maxcov&default`` codes the same condition, ``RowLags == min_rl
    .or. RowLags == max_rl`` on the inclusive window its ``CovMax`` scans
    (``timelag_handle.f90``), so this is the rule one engine codes and the other
    documents; patch 0002 of the confirmed bundle brings GEddySoft's test to its
    reachable edges without widening its search.

    ``covpeak_filter_length`` / ``outer`` control the optional Hamming smoothing
    of the covariance (see :func:`time_lag`).
    """
    tlag, tlag_min, tlag_max = _window_in_samples(
        move, tlag, tlag_min, tlag_max, acq_freq, lag_units)
    lo, hi = _inner_window(tlag, tlag_min, tlag_max)
    ext = _ext_window(tlag, lo, hi, outer, covpeak_filter_length)
    peak = _scan_peak(fix, move, lo, hi, covpeak_filter_length, ext,
                      kernel=xcov_kernel(xcov))
    on_boundary = (peak <= lo) or (peak >= hi)
    opt_use = int(round(float(tlag))) if on_boundary else peak
    return return_time_lag(move, peak, opt_use)
