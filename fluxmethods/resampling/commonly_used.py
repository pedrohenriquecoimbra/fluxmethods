"""Ways of putting a series onto a different time axis.

Each routine answers one question: given ``values`` sampled at ``times``, what are
they on ``target``? Nothing here is specific to a sonic, to a rate ratio, or to a
direction — the same routines serve a 50 Hz sonic reduced to 10 Hz, a slow
meteorological series lifted onto a fast grid, and a block average onto half-hours.

The signature is deliberately array-level rather than dataset-level: the assembly of
asynchronous streams happens during import, before a dataset exists, and the same
choice has to be available there and in the correction sequence. ``main`` wraps these
for the pipeline.

All times are integer nanoseconds, which is what a pandas index gives without a
lossy conversion.
"""

# built-in modules
import logging

# 3rd party modules
import numpy as np

logger = logging.getLogger(__name__)


def nearest(values, times, target, period_ns=None, **kwargs):
    """Take each target point's nearest sample, unchanged.

    Nothing is averaged and nothing is invented: every returned value is a value
    that was actually measured. That is the point of it. Interpolating linearly
    between two samples is a two-tap filter ``[1-a, a]`` whose gain is
    ``|(1-a) + a*exp(-2*pi*i*f/fs)|``, so a series carried onto a grid offset by a
    fraction ``a`` of a sample is low-pass filtered: it loses ``|1-2a|`` of its
    amplitude at the Nyquist frequency of the target rate -- nearly all of it for
    an offset near half a sample, whatever the rate -- and a few per cent at the
    top of the band a flux is built from, the exact figure depending on that rate.
    The offset measured for the bundled tracer is stated in
    :mod:`oneflux_preproc.io.importer.tracer_pipeline`.
    For a series that is *already at the target rate* and only needs
    aligning onto a common axis, that filtering is pure loss: it attenuates the
    high-frequency part of a covariance while leaving the correlation intact,
    which reads as a flux that is too small for no visible reason.

    So this is the method for aligning, and the rate reducers
    (:func:`fft_resample`, :func:`block_average`) are for reducing. It is what
    GEddySoft uses to put both the reduced sonic and the tracer onto its window
    grid (``interp1d(..., kind='nearest')``).

    The cost is timing jitter of up to half a sample rather than amplitude loss,
    which is the right trade when the sample itself is the quantity of interest.
    Ties go to the earlier sample, as ``scipy.interpolate.interp1d`` does.
    """
    times = np.asarray(times)
    values = np.asarray(values, dtype=float)
    target = np.asarray(target)
    if times.size == 0:
        return np.full(target.shape, np.nan)
    if times.size == 1:
        return np.where(target == times[0], values[0], np.nan)

    idx = np.searchsorted(times, target)
    lo = np.clip(idx - 1, 0, times.size - 1)
    hi = np.clip(idx, 0, times.size - 1)
    # <= keeps a tie on the earlier sample, matching interp1d's 'nearest'.
    take = np.where(target - times[lo] <= times[hi] - target, lo, hi)
    out = values[take]
    # Outside the record there is no nearest sample, only an extrapolation --
    # but "outside" is half a period beyond the end, not one nanosecond beyond
    # it. A grid point whose nearest sample is within half a period *has* a
    # nearest sample; that is what makes this method nearest-neighbour rather
    # than interval containment.
    #
    # GEddySoft files each sample into ``rint((t - t0) * rate)``, so a sample
    # within half a period of a slot fills it. Stated the strict way instead, the
    # last grid point is dropped whenever the record ends a fraction of a period
    # before it. One slot is not the cost: the scalar's finite span is then
    # shorter than the
    # reference's, and a lagged covariance that trims to that span before
    # rolling wraps a different set of samples, which on a near-noise covariance
    # is worth tens of percent.
    tol = (float(period_ns) / 2.0) if period_ns else 0.0
    outside = (target < times[0] - tol) | (target > times[-1] + tol)
    return np.where(outside, np.nan, out)


def linear(values, times, target, **kwargs):
    """Straight-line interpolation onto ``target``.

    Cheap and exact for a series already close to the target rate. Going down in
    rate it is not the right tool: nothing removes the variance above the new
    Nyquist frequency, so it folds back into the retained band, and the
    interpolation attenuates what is left. Prefer :func:`fft_resample` for a real
    reduction, or :func:`block_average` where an average is what is wanted.
    """
    return np.interp(target, times, values)


def fft_resample(values, times, target, window_samples=None, period_ns=None,
                 **kwargs):
    """Band-limited resampling, then placement on ``target`` by nearest neighbour.

    ``scipy.signal.resample`` is an FFT method, so the variance above the new
    Nyquist frequency is removed rather than folded back.

    Two details of the axis matter as much as the resample itself, and both follow
    GEddySoft, whose ``GEddySoft_main`` this reproduces. The output length comes
    from the rate ratio, not from the length of the target axis. And the resampled
    samples sit on an *exactly regular* axis at the new period starting from the
    first sample: an FFT resample returns evenly spaced samples of a periodic
    extension, so spreading them over ``[t0, t_last]`` instead leaves the axis one
    period short overall, which over 18000 samples is a fifth of a sample by the end
    — enough for the nearest-neighbour placement to take the wrong one.

    ``window_samples`` truncates the input to a whole processing window first, as
    GEddySoft does: a record one sample longer moves every output sample, not just
    the tail.

    Placement is nearest, not linear, because blending neighbours would undo part of
    the band-limiting just applied.
    """
    from scipy.signal import resample

    values = np.asarray(values, dtype=float)
    times = np.asarray(times)
    target = np.asarray(target)
    if window_samples and len(values) >= window_samples:
        values, times = values[:window_samples], times[:window_samples]
    if len(values) < 2 or len(target) < 1:
        return np.interp(target, times, values)

    if period_ns is None:
        period_ns = (float(np.median(np.diff(target))) if len(target) > 1
                     else float(np.median(np.diff(times))))
    dt_in = float(np.median(np.diff(times)))
    if not (period_ns > 0 and dt_in > 0):
        return np.interp(target, times, values)

    # How many samples the reduction produces. A record that covers its window
    # gives the target's own length: 90000 sonic samples and 18000 grid points are
    # one 1800 s window either way.
    #
    # Deriving it from the measured spacing instead is what GEddySoft's
    # reconstruction exists to avoid: recorded timestamps carry transmission
    # jitter, so a median spacing yields fewer samples than the window holds, and
    # the shortfall accumulates across the window rather than sitting as a constant
    # offset. The series then drifts against the reference by the tail while its
    # variance is untouched -- invisible in any per-stream statistic, and it
    # displaces every covariance built from it.
    #
    # A record that does *not* cover its window is the one case where the target's
    # length is the wrong answer: it resamples the record onto more time than it
    # was measured over, scaling every frequency and again leaving the variance
    # untouched so nothing per-stream can see it.
    #
    # Shortening is gated on the sample count *and* the clock agreeing, because
    # neither is trustworthy alone. ``window_samples`` comes from an inferred
    # emission rate (``_window_samples`` in ``io/importer/stream_assembly.py``),
    # and on a minority of the bundled sonic files that inference mis-snaps: their
    # jittered median spacing reads a few hertz above the true rate, so the count
    # says "short" for windows that are complete. The elapsed span is what settles it: a
    # complete window spans its full duration whatever its sample count, and a
    # genuinely short record spans visibly less. Requiring both means a bad rate
    # inference can only ever leave the old behaviour in place, never invent a
    # shortfall. The span also survives dropped samples, which a count does not.
    span_ns = float(times[-1] - times[0])
    window_ns = float(len(target) - 1) * period_ns
    if (window_samples and len(target) > 1
            and len(values) < window_samples and span_ns < window_ns):
        n_out = int(round(span_ns / period_ns))
    elif window_samples and len(target) > 1:
        n_out = len(target)
    else:
        n_out = int(round(len(values) * dt_in / period_ns))
    if n_out < 2:
        return np.interp(target, times, values)
    reduced = resample(values, n_out)
    reduced_ns = times[0] + np.arange(n_out) * period_ns

    idx = np.clip(np.searchsorted(reduced_ns, target), 1, n_out - 1)
    take_left = (target - reduced_ns[idx - 1]) <= (reduced_ns[idx] - target)
    out = reduced[np.where(take_left, idx - 1, idx)]
    # Outside the reduced record is outside the window, not a value.
    return np.where((target >= reduced_ns[0]) & (target <= reduced_ns[-1]),
                    out, np.nan)


def block_average(values, times, target, **kwargs):
    """Mean of the samples falling in each target interval.

    The estimator to use when the target sample *is* an average over its interval
    rather than the signal read at an instant — a half-hourly mean from a fast
    series, say. Each target point takes the mean of the inputs nearest to it in
    the sense of the interval midpoints, so no sample is counted twice or dropped.
    NaN-aware; an interval with no finite sample yields NaN.
    """
    values = np.asarray(values, dtype=float)
    times = np.asarray(times, dtype='int64')
    target = np.asarray(target, dtype='int64')
    if len(target) == 0:
        return np.array([], dtype=float)
    if len(target) == 1:
        finite = values[np.isfinite(values)]
        return np.array([finite.mean() if finite.size else np.nan])

    edges = np.empty(len(target) + 1, dtype='float64')
    edges[1:-1] = (target[:-1] + target[1:]) / 2.0
    half = (target[1] - target[0]) / 2.0
    edges[0], edges[-1] = target[0] - half, target[-1] + half

    which = np.searchsorted(edges, times, side='right') - 1
    keep = (which >= 0) & (which < len(target)) & np.isfinite(values)
    total = np.bincount(which[keep], weights=values[keep], minlength=len(target))
    count = np.bincount(which[keep], minlength=len(target))
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out
