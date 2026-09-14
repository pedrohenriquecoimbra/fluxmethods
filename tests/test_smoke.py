"""That the methods import and compute, on data small enough to check by hand.

Not a test of the physics -- that lives with the reference implementation, which
runs these same functions against its own copy and requires them equal to twelve
digits. This is the test that the package is importable, that nothing here
reaches for a plotting stack on the way in, and that each method still answers in
the shape its callers unpack.
"""

import math
import sys
import unittest

import numpy as np
import xarray as xr

import fluxmethods


class TestImportsAreClean(unittest.TestCase):
    def test_computing_a_method_does_not_pull_in_a_plotting_stack(self):
        """matplotlib is a diagnostic dependency, not a method one: the one plot
        in here imports it where it is drawn, so a caller computing a spike mask
        does not have to have it installed."""
        self.assertNotIn("matplotlib.pyplot", sys.modules)

    def test_every_exported_name_is_callable(self):
        for name in fluxmethods.__all__:
            self.assertTrue(callable(getattr(fluxmethods, name)), name)


class TestRotation(unittest.TestCase):
    """The rotations answer with an attribute object (``.u``, ``.v``, ``.w`` and
    the angles), which is the convention their callers unpack."""

    def test_a_double_rotation_puts_the_whole_wind_on_u(self):
        rng = np.random.default_rng(0)
        u = 2.0 + rng.normal(0, 0.1, 2000)
        v = 1.0 + rng.normal(0, 0.1, 2000)
        w = 0.3 + rng.normal(0, 0.1, 2000)

        out = fluxmethods.double_rotation(u, v, w)
        # What the two angles are chosen for: no mean crosswind, no mean
        # vertical wind, and the full speed carried on u.
        self.assertAlmostEqual(float(np.mean(out.v)), 0.0, places=10)
        self.assertAlmostEqual(float(np.mean(out.w)), 0.0, places=10)
        self.assertAlmostEqual(
            float(np.mean(out.u)),
            math.sqrt(np.mean(u) ** 2 + np.mean(v) ** 2 + np.mean(w) ** 2),
            places=10)

    def test_the_angles_are_the_ones_the_means_imply(self):
        u = np.full(100, 3.0)
        v = np.full(100, 4.0)
        w = np.zeros(100)
        out = fluxmethods.double_rotation(u, v, w)
        self.assertAlmostEqual(float(out.theta), math.atan2(4.0, 3.0), places=12)
        self.assertAlmostEqual(float(out.phi), 0.0, places=12)


class TestDespiking(unittest.TestCase):
    """``mauder2013`` is xarray-native and forms its statistic *within* each
    averaging period: it takes a named DataArray and answers a Dataset."""

    @staticmethod
    def _series(values):
        return xr.DataArray(np.asarray(values, dtype=float),
                            dims="time", name="w")

    def _despiked(self, values, **kwargs):
        return self._series(values).pipe(fluxmethods.mauder2013, **kwargs)["w"]

    def test_it_flags_the_one_sample_that_does_not_belong(self):
        x = np.zeros(600)
        x[300] = 500.0
        out = self._despiked(x, q=7)
        self.assertTrue(math.isnan(float(out[300])))
        self.assertEqual(int(np.isnan(out).sum()), 1)

    def test_a_series_with_no_spread_has_no_spikes(self):
        """MAD is zero, so the bounds collapse onto the median and the strict
        comparisons flag nothing."""
        out = self._despiked(np.ones(600), q=7)
        self.assertEqual(int(np.isnan(out).sum()), 0)

    def test_the_statistic_is_taken_within_each_period_not_across_them(self):
        """Two half-hours at different levels. Pooled, the spread between the
        periods would set the threshold and neither spike would be flagged."""
        block = np.zeros(300)
        x = np.stack([block.copy(), block + 100.0])
        x[0, 150] = 20.0
        x[1, 150] = 120.0
        da = xr.DataArray(x, dims=("date", "time"), name="w")

        out = fluxmethods.mauder2013(da, q=7)["w"]
        self.assertEqual(int(np.isnan(out).sum()), 2)
        self.assertTrue(math.isnan(float(out[0, 150])))
        self.assertTrue(math.isnan(float(out[1, 150])))


class TestResampling(unittest.TestCase):
    def test_block_average_bins_on_the_midpoints_between_target_points(self):
        """The edges sit halfway between the target timestamps, so each sample
        lands in exactly one bin and none is counted twice or dropped."""
        times = np.arange(0, 10, dtype="int64") * 1_000_000_000
        target = np.arange(0, 10, 2, dtype="int64") * 1_000_000_000
        out = np.asarray(fluxmethods.block_average(
            np.arange(10.0), times, target))
        # edges at -1, 1, 3, 5, 7, 9 s: {0}, {1,2}, {3,4}, {5,6}, {7,8}
        np.testing.assert_allclose(out, [0.0, 1.5, 3.5, 5.5, 7.5])

    def test_an_interval_with_no_finite_sample_is_nan(self):
        times = np.array([0, 1], dtype="int64") * 1_000_000_000
        target = np.array([0, 2, 4], dtype="int64") * 1_000_000_000
        out = np.asarray(fluxmethods.block_average([1.0, np.nan], times, target))
        np.testing.assert_allclose(out[0], 1.0)
        self.assertTrue(np.isnan(out[1]) and np.isnan(out[2]))


if __name__ == "__main__":
    unittest.main()
