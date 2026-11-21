"""Unit conventions shared by the time-lag methods.

A time lag is a **physical property of the sampling path** -- the transit time down
a tube, the advection between an intake and the sonic -- so it is expressed in
**seconds**, positive when the scalar arrives *after* the wind. That is what
instrument metadata records (EddyPro's ``nom_timelag``), what this package's
configuration carries, and what the methods report, so a configuration is portable
across acquisition rates: resample 20 Hz to 10 Hz and the lag in seconds is
unchanged while the equivalent number of samples halves.

The routines themselves work in integer sample shifts, because that is what
``DataArray.shift`` takes. Converting between the two is this module's job, and it
happens once at the boundary of each method, so no caller has to know the internal
convention:

* a *positive physical lag* becomes a *negative shift*, since realigning a delayed
  scalar means moving it back in time (``move.shift(time=-n)``);
* the conversion needs the acquisition frequency, which is read from the series'
  own ``time`` coordinate unless supplied -- from its regular steps, so a gap in
  the records does not lower it, by the one helper :func:`acq_freq` every
  method uses.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)


def acq_freq(move, acq_freq=None):
    """Acquisition frequency [Hz] from an explicit value or the ``time`` coordinate.

    The spacing of the coordinate's *regular* steps -- those within half the
    median step -- not the mean of all of them: records missing from the time
    axis (an outage of a few minutes) widen the mean spacing and lower the rate,
    so a configured lag in seconds would become too few samples and a found
    peak would be reported as too many seconds. NaN when no rate can be read.

    The methods shift by sample index, which assumes a regular grid; when the
    coordinate has gaps the shift reaches across them, and that is said in a
    warning rather than passed over.
    """
    if acq_freq:
        return float(acq_freq)
    try:
        time = np.asarray(move['time'].values).ravel()
    except (KeyError, ValueError, TypeError, AttributeError):
        return np.nan
    if time.size < 2:
        return np.nan
    try:
        if time.dtype.kind in 'mM':
            dt = np.abs(np.diff(time) / np.timedelta64(1, 's')).astype(float)
        else:
            dt = np.abs(np.diff(time.astype(float)))
    except (ValueError, TypeError):
        return np.nan
    dt = dt[np.isfinite(dt)]
    if not dt.size:
        return np.nan
    typical = float(np.median(dt))
    if not (np.isfinite(typical) and typical > 0):
        return np.nan
    # The mean over the regular steps only: on a gap-free grid this is the plain
    # mean spacing, to the last bit, and a gap no longer drags it.
    regular = np.abs(dt - typical) <= 0.5 * typical
    step = float(np.mean(dt[regular]))
    gaps = int(np.sum(~regular))
    if gaps:
        logger.warning(
            "%s: the time coordinate is not a regular grid (%d irregular steps); "
            "the rate is taken from the median spacing (%g Hz), and the lag shift, "
            "which is by sample index, reaches across the gaps.",
            getattr(move, 'name', None) or 'time_lag', gaps, 1.0 / step)
    return 1.0 / step


def seconds_to_shift(seconds, freq, lag_units='seconds'):
    """Configured lag -> integer sample shift, or ``None`` if not resolvable.

    ``lag_units`` states what the caller is handing over, so the translation is an
    explicit contract rather than an assumption:

    * ``'seconds'`` (the configuration convention) -- a *physical* lag, positive
      when the scalar arrives after the wind. The sign flips on conversion, because
      realigning a delayed scalar means moving it back: ``+0.4`` s becomes
      ``-0.4 * freq`` samples.
    * ``'samples'`` -- already an integer sample shift; passed through untouched,
      which is what a caller working in the routine's own units wants.
    """
    if seconds is None:
        return None
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(value):
        return None
    if str(lag_units).lower().startswith('sample'):
        return int(round(value))
    if not (np.isfinite(freq) and freq > 0):
        return None
    # Python's round is half-to-even where EddyPro's nint is half-away-from-zero;
    # the two part only on a lag that is exactly half a sample.
    return -int(round(value * freq))


def shift_to_seconds(shift, freq):
    """Integer sample shift -> physical lag [s], the inverse of :func:`seconds_to_shift`.

    Without a usable frequency there is no conversion, and the result is NaN.
    Falling back to the raw *shift* instead, on the grounds that a diagnostic beats
    a NaN, would silently change the unit of a reported variable rather than its
    availability: the caller stores this as a lag in seconds, and 22 read as 22 s
    when it means 22 samples is not a diagnostic. A NaN, with the reason logged, is
    the honest answer.
    """
    value = float(shift)
    if not (np.isfinite(freq) and freq > 0):
        logger.warning(
            "no usable acquisition frequency, so a shift of %g samples cannot be "
            "reported as a lag in seconds; reporting NaN. The frequency is read "
            "from the series' own 'time' coordinate unless one is passed in.",
            value)
        return np.nan
    return -value / freq
