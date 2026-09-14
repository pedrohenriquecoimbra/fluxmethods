"""Despiking functions based on Mauder et al. 2013
"""

# built-in modules
import re
import warnings
import logging
from functools import reduce

# 3rd party modules
import numpy as np
import xarray as xr
from itertools import islice
from numpy import roll



logger = logging.getLogger('ep.corrections.despiking.mauder_et_al_2013')


def window(seq, n=3):
    """Returns a sliding window (of width n) over data from the iterable
        s = [s(i), s(i+1), ..., s(i+n-1)]
    """
    if n:
        it = iter(roll(seq, int(n/2)))
        result = tuple(islice(it, n))
        if len(result) == n:
            yield result
        for elem in it:
            result = result[1:] + (elem,)
            yield result
    else:
        yield seq


def label_true_groups_numpy(lst):
    arr = np.array(lst).astype('bool')
    # Create an array of the same length with zeros
    result = np.zeros_like(arr, dtype=int)

    # Find where the `True` values start (1) and stop (0)
    # The np.diff gives the change between consecutive elements
    # 1's indicate the start of a group of Trues
    group_start = np.diff(np.concatenate(([False], arr, []))) == 1

    # We create a running group counter using cumsum
    group_counter = np.cumsum(group_start)

    # Assign group size to each element in the group
    for group in np.unique(group_counter):
        result[group_counter == group] = np.sum(group_counter == group)

    # Ignore if False
    result = result * arr
    return result


def _mauder_series(x, q=7, n=None, max_consec_spikes=None):
    """MAD despiking of one contiguous series.

    Split out from :func:`mauder2013` so the median and the deviation are taken
    over one averaging period rather than over every period in the file at once.
    """
    x = x.stack({'tmp_mauder_2013': x.dims})

    if n and len(np.array(x)) < n:
        raise ValueError(
            "Input array must be at least as long as the window size.")
    for i, x_ in enumerate(window(
        np.array(x),
        n=n
    )):
        x_ = np.array(x_)
        x_med = float(np.nanmedian(x_))
        mad = float(np.nanmedian(np.abs(x_ - x_med)))
        bounds = (float(x_med - (q * mad) / 0.6745),
                  float(x_med + (q * mad) / 0.6745))

        flag = [(y < min(bounds)) or (y > max(bounds)) for y in x_]
        flag = _drop_unclosed_run(np.asarray(flag, dtype=bool))
        if max_consec_spikes:
            # Real run lengths -- see :func:`_run_lengths` for why
            # ``label_true_groups_numpy`` cannot be read as one. A 0 (or None)
            # means no cap, which is what the EddyPro importer emits for a MAD
            # project.
            lengths = _run_lengths(flag)
            flag = (np.asarray(flag, dtype=bool)
                    & (lengths <= int(max_consec_spikes)))
        flag = label_true_groups_numpy(flag)
        x = xr.where(flag, np.nan, x)

    # No interpolation here: this method flags and leaves NaN, which is what its
    # registered description says and what the sibling routine's ``spike_mode``
    # exists to opt into.
    return x.unstack()


def _run_lengths(flag):
    """Each ``True`` sample labelled with the length of the run it belongs to.

    Not :func:`label_true_groups_numpy`, despite the name: that one accumulates
    ``cumsum`` over run *starts*, so its segments run from one start to the next
    and each label is the run plus the gap that follows it -- 7 for a two-sample
    run followed by five clear samples. The difference is invisible to a caller
    reading only truthiness, and decisive for one reading the value as a length,
    which the run-length cap does.

    ``label_true_groups_numpy`` is left as it is because its output is pinned by
    ``tests/test_despiking.py``.
    """
    flag = np.asarray(flag, dtype=bool)
    out = np.zeros(flag.shape, dtype=int)
    start = None
    for i, v in enumerate(flag):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out[start:i] = i - start
            start = None
    if start is not None:
        out[start:] = len(flag) - start
    return out


def _drop_unclosed_run(flag):
    """Clear a run of outliers that is still outside the bounds at the last sample.

    A spike is an excursion that comes back. Deciding that a run *was* one takes
    the return: until the series is inside the bounds again, what has been seen
    is equally the start of a genuine change in the signal -- a step, a drift, a
    gap in the mixing -- and replacing it would erase real data. A run that never
    closes before the record ends offers no such return, so it is left alone.

    Measured against EddyPro on the FR-Gri sample: every run this routine flags
    strictly inside a record, EddyPro flags too. The whole remaining disagreement
    is runs that touch an edge, and almost all of its samples are in runs still
    outside the bounds at the *last* sample, which EddyPro leaves untouched. The
    comparison is ``tests/test_eddypro_variants.py::test_despike_mauder_measured``.

    Only the closing end. A run at the *start* does return, and EddyPro flags all
    but the shortest of them in the sample, so it is not the general "touches a
    boundary" rule it might look like -- it is specifically the run the scan
    never gets to close.

    The sibling routine already works this way: ``linear_interpolate_spikes``
    (:mod:`vickers_et_al_1997`) will not interpolate without a valid sample on
    both sides of the run, for the same reason.
    """
    if flag.size and flag[-1]:
        last_ok = np.flatnonzero(~flag)
        start = last_ok[-1] + 1 if last_ok.size else 0
        flag = flag.copy()
        flag[start:] = False
    return flag


def mauder2013(x, q=7, n=None, max_consec_spikes=None, **kwargs):
    """MAD despiking, with the statistic taken **within each averaging period**.

    :meth:`Correction.apply` hands this routine the whole ``(date, time)`` array.
    Stacking every dimension into one series would form a single median and median
    absolute deviation over every half-hour in the file together, so a day whose
    levels drift -- which is most days -- would have its MAD set by the spread
    *between* periods rather than the spread *within* one, and the threshold that
    implies is far too wide to flag anything.

    Despiking is a within-period operation -- the flux for a period is computed
    from that period's samples -- so the statistic is formed per period, which is
    also what Mauder et al. (2013) describe. Grouping here rather than in
    ``apply`` keeps it a property of this method: ``stddev_movingwindow`` beside
    it carries its own window and must not be regrouped.

    ``max_consec_spikes`` caps how many consecutive outliers still count as a
    spike; a longer run is a genuine excursion and is left alone. ``0`` or
    ``None`` means no cap.

    This is the criterion Vickers and Mahrt (1997) define, and the one EddyPro
    exposes as ``sr_num_spk``; the sibling ``stddev_movingwindow`` applies it,
    with the project's own value. Mauder et al. (2013) state no such limit, and
    EddyPro's MAD path does not apply one: on the FR-Gri sample it replaces
    hundreds of interior runs far longer than the ``sr_num_spk`` its own project
    declares, the longest running to thousands of samples. So the EddyPro
    importer emits ``0`` here for a MAD
    project (see ``compatibility/EddyPro/setup.py``), and the cap is off unless a
    configuration asks for it.

    Its cost is measured by
    ``tests/test_eddypro_variants.py::test_despike_mauder_measured`` and recorded
    in the manuscript's remaining-differences table.
    """
    if max_consec_spikes:
        # Loud, because it is usually not asked for: EddyPro spends
        # ``sr_num_spk`` on its other despiking routine, so a configuration
        # carrying it here suppresses most of what EddyPro flags on this path.
        # Honoured -- the configuration says so explicitly -- but never silently.
        logger.warning(
            "mad_perperiod: max_consec_spikes=%s caps how long a run may be and "
            "still count as a spike. Mauder et al. (2013) state no such limit "
            "and EddyPro applies none on this path; the value is Vickers and "
            "Mahrt's, which EddyPro reads for its other despiking routine. Set "
            "it to 0 unless you mean to depart from that.", max_consec_spikes)

    if 'date' in x.dims and x.sizes.get('date', 1) > 1:
        return (x.groupby('date')
                 .map(lambda period: _mauder_series(
                     period, q=q, n=n, max_consec_spikes=max_consec_spikes))
                 .to_dataset())
    return _mauder_series(
        x, q=q, n=n, max_consec_spikes=max_consec_spikes).to_dataset()
