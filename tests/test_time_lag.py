"""The lag estimators, on a pair whose true delay is known.

These need a ``time`` coordinate in **float seconds** — that is the reference
implementation's convention, and `commons.acq_freq` derives the rate from it by
differencing. A datetime64 coordinate differences to *nanoseconds* and yields
1e-8 Hz with no complaint, so the fixture below is also the documentation of the
input these functions expect.
"""

import unittest

import numpy as np
import xarray as xr

from fluxmethods.time_lag import commons, fixed, maximisation

HZ, N, SHIFT = 10, 2000, 7          # 7 samples at 10 Hz = 0.7 s


def _pair(shift=SHIFT):
    """``w`` and a scalar that is ``w`` delayed by ``shift`` samples."""
    rng = np.random.default_rng(1)
    base = rng.normal(0, 1, N + 50)
    coords = {"time": np.arange(N) / HZ}
    mk = lambda a, nm: xr.DataArray(a, dims="time", coords=coords, name=nm)
    return (mk(base[50:50 + N], "w"),
            mk(base[50 - shift:50 - shift + N], "co2"))


class TestAcquisitionRate(unittest.TestCase):
    def test_the_rate_comes_off_the_time_coordinate(self):
        _w, c = _pair()
        self.assertEqual(commons.acq_freq(c, None), float(HZ))

    def test_an_explicit_rate_wins_over_the_coordinate(self):
        _w, c = _pair()
        self.assertEqual(commons.acq_freq(c, 20), 20.0)

    def test_seconds_and_samples_round_trip(self):
        shift = commons.seconds_to_shift(0.7, HZ)
        self.assertEqual(shift, -SHIFT)
        self.assertAlmostEqual(commons.shift_to_seconds(shift, HZ), 0.7, places=12)


class TestMaximisation(unittest.TestCase):
    def test_the_search_recovers_the_delay_that_was_put_in(self):
        w, c = _pair()
        out = maximisation.time_lag(c, w, tlag_min=-1.0, tlag_max=1.0)
        self.assertAlmostEqual(float(out["co2_time_lag"]), 0.7, places=10)
        self.assertAlmostEqual(float(out["co2_time_lag_opt"]), 0.7, places=10)

    def test_the_reported_lag_is_in_seconds_and_says_so(self):
        """The units are on the variable, not only in the prose: a lag reported
        in samples and read as seconds is wrong by the acquisition rate."""
        w, c = _pair()
        out = maximisation.time_lag(c, w, tlag_min=-1.0, tlag_max=1.0)
        self.assertEqual(out["co2_time_lag"].attrs.get("units"), "s")

    def test_a_window_that_excludes_the_peak_does_not_report_it(self):
        w, c = _pair()
        out = maximisation.time_lag(c, w, tlag_min=-0.2, tlag_max=0.2)
        self.assertLessEqual(abs(float(out["co2_time_lag"])), 0.2 + 1e-12)


class TestFixed(unittest.TestCase):
    def test_a_constant_lag_is_applied_and_reported_as_given(self):
        w, c = _pair()
        out = fixed.fix_time_lag(c, w, tlag=0.7)
        self.assertAlmostEqual(float(out["co2_time_lag"]), 0.7, places=12)

    def test_applying_the_true_delay_lines_the_pair_back_up(self):
        """The point of the step: after the shift the two series are the same
        signal again, so their correlation is 1 where they overlap."""
        w, c = _pair()
        moved = fixed.fix_time_lag(c, w, tlag=0.7)["co2"]
        a, b = np.asarray(moved), np.asarray(w)
        ok = np.isfinite(a) & np.isfinite(b)
        self.assertGreater(np.corrcoef(a[ok], b[ok])[0, 1], 0.999999)


if __name__ == "__main__":
    unittest.main()
