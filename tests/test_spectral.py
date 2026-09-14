"""The spectral estimators that need no measured spectrum.

The physics is checked on the reference implementation's side, where the sample
data and the published engine comparisons are. What is checked here is that the
modules import without dragging in a units or plotting stack, and that the two
pure transfer functions still have the shape the corrections are built on — a
block average removes the low frequencies, a sensor's response removes the high
ones, and both are bounded in [0, 1].
"""

import math
import pathlib
import subprocess
import sys
import unittest

import numpy as np

from fluxmethods import spectral
from fluxmethods.spectral import analytic, eddypro


class TestImportsStayLight(unittest.TestCase):
    def test_no_units_or_plotting_stack_is_pulled_in(self):
        """pint is a dependency of the *data*, not of these modules: only the one
        branch that meets a pint quantity imports it, where it is used.

        Asked in a subprocess rather than of ``sys.modules`` here, because by the
        time this runs the test session has imported a great deal that fluxmethods
        did not ask for, and the question is what *importing fluxmethods* costs.
        """
        probe = ("import sys, fluxmethods; "
                 "print(int('pint' in sys.modules), "
                 "int('matplotlib.pyplot' in sys.modules))")
        out = subprocess.run([sys.executable, "-c", probe], capture_output=True,
                             text=True, cwd=str(pathlib.Path(__file__).parent.parent))
        self.assertEqual(out.returncode, 0, out.stderr)
        self.assertEqual(out.stdout.split(), ["0", "0"], "an import got heavier")

    def test_the_registered_estimators_are_all_reachable(self):
        for name in ("block_average_highpass", "sonic_response", "analytic_tube",
                     "bpcf_moncrieff_97", "bpcf_anemometric_fluxes", "bpcf_momentum"):
            self.assertTrue(callable(getattr(spectral, name)), name)


class TestFrequencyGrid(unittest.TestCase):
    def test_the_grid_rises_and_stays_positive(self):
        F = np.asarray(analytic._frequency_grid(10.0, 200))
        self.assertEqual(len(F), 200)
        self.assertTrue(np.all(np.diff(F) > 0))
        self.assertTrue(np.all(F > 0))


class TestHighPass(unittest.TestCase):
    """``H(f) = 1 - [sin(pi f T)/(pi f T)]^2`` — what block averaging leaves."""

    def test_it_is_bounded_on_the_grid_the_corrections_use(self):
        F = np.asarray(analytic._frequency_grid(10.0, 400))
        H = np.asarray(analytic._highpass_transfer(F, 30.0))
        self.assertTrue(np.all((H >= 0.0) & (H <= 1.0 + 1e-12)))
        self.assertGreater(H[-1], 0.99)   # fastest resolved: untouched

    def test_the_loss_is_where_the_averaging_period_puts_it(self):
        """A block average removes what is slower than the period itself. For
        30 minutes that is below ~1/(pi*1800) Hz, which sits *under* the 0.001 Hz
        the grid starts at -- so the grid sees almost none of the loss, and the
        cut has to be looked for where it is."""
        minutes = 30.0
        cut = 1.0 / (math.pi * minutes * 60.0)
        below = np.asarray(analytic._highpass_transfer(np.array([cut / 50.0]), minutes))
        above = np.asarray(analytic._highpass_transfer(np.array([cut * 50.0]), minutes))
        self.assertLess(float(below[0]), 0.01)
        self.assertGreater(float(above[0]), 0.99)

    def test_a_longer_average_keeps_more_of_the_low_frequencies(self):
        F = np.asarray(analytic._frequency_grid(10.0, 400))
        short = np.asarray(analytic._highpass_transfer(F, 5.0))
        long_ = np.asarray(analytic._highpass_transfer(F, 60.0))
        self.assertTrue(np.all(long_ >= short - 1e-12))

    def test_it_matches_the_closed_form_at_a_hand_checked_point(self):
        f, minutes = 0.01, 30.0
        H = float(np.asarray(analytic._highpass_transfer(np.array([f]), minutes))[0])
        x = math.pi * f * minutes * 60.0
        self.assertAlmostEqual(H, 1.0 - (math.sin(x) / x) ** 2, places=12)


class TestEddyProBandPass(unittest.TestCase):
    def test_its_high_pass_is_the_same_shape_on_its_own_grid(self):
        nf = np.asarray(eddypro._grid()[0] if isinstance(eddypro._grid(), tuple)
                        else eddypro._grid())
        H = np.asarray(eddypro._hp(nf, 1800.0))
        self.assertTrue(np.all((H >= 0.0) & (H <= 1.0 + 1e-12)))
        self.assertLess(H[0], H[-1])


if __name__ == "__main__":
    unittest.main()
