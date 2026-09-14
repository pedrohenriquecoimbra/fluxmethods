"""The shared numerics, and the two detrending routines built on them."""

import unittest

import numpy as np
import xarray as xr

import fluxmethods
from fluxmethods import detrending, resampling, signal


class TestSignal(unittest.TestCase):
    def test_a_straight_line_is_fitted_exactly(self):
        slope, intercept = signal.nanlinfit(np.array([1.0, 3.0, 5.0, 7.0]))
        self.assertAlmostEqual(slope, 2.0, places=12)
        self.assertAlmostEqual(intercept, 1.0, places=12)

    def test_detrending_a_straight_line_leaves_nothing(self):
        out = signal.nandetrend(np.array([1.0, 3.0, 5.0, 7.0]))
        np.testing.assert_allclose(out, 0.0, atol=1e-12)

    def test_a_gap_closes_up_the_time_axis_KNOWN_DEFECT(self):
        """Pinned as it stands, and it is wrong.

        ``nanlinfit`` deletes the NaNs and fits against ``arange`` of what is
        left, so the fit runs on a *compacted* axis; ``nandetrend`` then
        subtracts that trend at the original indices. An exact straight line
        with one interior gap therefore does not detrend to zero, and the
        residual grows along the series.

        Not corrected here: this file is a verbatim copy of the reference
        implementation's ``core/signal.py`` and the equivalence claim is that
        the two are the same code. The fix belongs there -- fit against the
        surviving *positions* rather than their count -- and this test is what
        should start failing when it lands.
        """
        line_with_a_hole = np.array([1.0, 3.0, np.nan, 7.0, 9.0])
        slope, _ = signal.nanlinfit(line_with_a_hole)
        self.assertAlmostEqual(slope, 2.8, places=12)      # the truth is 2.0
        np.testing.assert_allclose(
            signal.nandetrend(line_with_a_hole),
            [0.2, -0.6, np.nan, -2.2, -3.0], atol=1e-12)   # the truth is 0.0

    def test_a_series_covaries_most_with_itself_at_zero_lag(self):
        rng = np.random.default_rng(0)
        x = rng.normal(0, 1, 400)
        curve = signal.xcov(x, x, (-3, 3))
        self.assertEqual(int(np.argmax(curve)), 3)   # the centre of -3..3


class TestDetrending(unittest.TestCase):
    @staticmethod
    def _series(values):
        return xr.DataArray(np.asarray(values, dtype=float), dims="time", name="w")

    def test_block_average_removes_the_mean_and_keeps_the_samples(self):
        out = detrending.block_average(self._series([1.0, 2.0, 3.0]))["w"]
        np.testing.assert_allclose(np.asarray(out), [-1.0, 0.0, 1.0])

    def test_linear_detrend_removes_a_trend_that_block_average_would_keep(self):
        ramp = self._series(np.arange(10.0))
        linear = np.asarray(detrending.linear_detrend(ramp)["w"])
        block = np.asarray(detrending.block_average(ramp)["w"])
        np.testing.assert_allclose(linear, 0.0, atol=1e-12)
        self.assertGreater(np.ptp(block), 8.0)


class TestTheTwoBlockAverages(unittest.TestCase):
    def test_they_are_different_methods_and_neither_takes_the_bare_name(self):
        """A detrending block average removes a period's mean; a resampling one
        is the mean of the samples in each target interval. Exporting either at
        the top level under the bare name would hand a caller the wrong step."""
        self.assertIsNot(detrending.block_average, resampling.block_average)
        self.assertNotIn("block_average", fluxmethods.__all__)
        self.assertFalse(hasattr(fluxmethods, "block_average"))


if __name__ == "__main__":
    unittest.main()
