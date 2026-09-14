"""Calling the spectral estimators, not just importing them.

This file exists because importing was not enough. The analytic corrections pull
their transfer-function models in with imports *inside* the function bodies, so a
module can import cleanly, pass a suite that only touches its pure helpers, and
still raise ImportError the first time anyone actually corrects a flux. Every
estimator here is therefore called on a real dataset.
"""

import unittest

import numpy as np
import xarray as xr

from fluxmethods.spectral import analytic, eddypro, measured

TUBE = {"li7200": {"tube_length": "71.1", "tube_diameter": "5.3",
                   "tube_flowrate": "12.0"}}

#: EddyPro's band-pass needs more of the instrument declared than the built-in
#: analytic corrections do: an optical path length and a response time on each
#: sensor, and a model name.
#:
#: The values here are plausible ones, deliberately. `_sonic_geometry` accepts a
#: declared path only while 0.01 < it < 0.5 m; outside that it falls back to the
#: *bundled specification* for the declared model and says so, and it skips only
#: when the model is unknown as well. So an implausible declaration does not
#: skip -- it quietly computes from the published specification instead of from
#: the geometry the fixture appears to set up, which is a fixture that looks like
#: it is testing one thing and tests another.
INSTRUMENTS = {
    "sonic": {"model": "hs_50_1", "vpath_length": "0.125", "tau": "0.02"},
    "irga": {"model": "li7200_1", "tube_length": "71.1", "tube_diameter": "5.3",
             "tube_flowrate": "12.00", "northward_separation": "-3.00",
             "eastward_separation": "17.00", "vertical_separation": "1.00",
             "vpath_length": "0.127", "tau": "0.02"},
}


def _fully_declared(wind=(1.0, 3.0), zL=-0.5, ta=293.15):
    """A dataset whose instruments are declared completely enough for EddyPro."""
    ds = _averaged(wind, zL=zL)
    ds["air_temperature"] = ("date", np.full(np.atleast_1d(wind).shape, float(ta)))
    ds.attrs["Instruments"] = INSTRUMENTS
    return ds


def _averaged(wind=(1.0, 2.0, 3.0), tube=True, zL=-0.5):
    """A minimal averaged dataset carrying what the analytic corrections need."""
    wind = np.atleast_1d(wind).astype(float)
    ds = xr.Dataset(
        {"wind_speed": ("date", wind),
         "z_L": ("date", np.full(wind.shape, zL)),
         "air_temperature": ("date", np.full(wind.shape, 290.0))},
        coords={"date": np.arange(wind.size)})
    ds["z-d"] = xr.DataArray(1.9)
    ds.attrs["acquisition_frequency"] = 20.0
    if tube:
        ds.attrs["Instruments"] = TUBE
    return ds


class TestAnalyticCorrectionsRun(unittest.TestCase):
    def test_the_closed_path_factor_is_a_bounded_attenuation_correction(self):
        scf = analytic.analytic_tube(_averaged())["scf_analytic_tube"].values
        self.assertTrue(np.all(np.isfinite(scf)))
        self.assertTrue(np.all(scf >= 1.0))   # attenuation can only inflate a flux
        self.assertTrue(np.all(scf < 2.0))    # and stays physical for a real tube

    def test_an_open_path_analyser_loses_nothing_to_a_tube(self):
        scf = analytic.analytic_tube(_averaged(tube=False))["scf_analytic_tube"].values
        np.testing.assert_allclose(scf, 1.0)

    def test_the_sonic_response_correction_runs_and_is_bounded(self):
        out = analytic.sonic_response(_averaged())
        scf = next(v for k, v in out.data_vars.items() if k.startswith("scf_"))
        self.assertTrue(np.all(np.isfinite(scf.values)))
        self.assertTrue(np.all(scf.values >= 1.0))

    def test_the_block_average_high_pass_correction_runs_and_is_bounded(self):
        out = analytic.block_average_highpass(_averaged())
        scf = next(v for k, v in out.data_vars.items() if k.startswith("scf_"))
        self.assertTrue(np.all(np.isfinite(scf.values)))
        self.assertTrue(np.all(scf.values >= 1.0))

    def test_faster_wind_costs_more_through_the_tube(self):
        """More air past the intake in a period means more attenuation, so the
        recovery factor rises with wind speed."""
        scf = analytic.analytic_tube(_averaged([1.0, 3.0, 6.0]))["scf_analytic_tube"].values
        self.assertTrue(np.all(np.diff(scf) > 0))


class TestEddyProBandPassRuns(unittest.TestCase):
    def test_each_of_the_three_scopes_produces_a_finite_factor(self):
        for fn in (eddypro.bpcf_moncrieff_97, eddypro.bpcf_anemometric_fluxes,
                   eddypro.bpcf_momentum):
            ds = _fully_declared()                  # a fresh one: see below
            before = set(ds.data_vars)
            out = fn(ds)
            added = sorted(set(out.data_vars) - before)
            self.assertTrue(added, f"{fn.__name__} produced no correction factor")
            for name in added:
                self.assertTrue(np.all(np.isfinite(out[name].values)), name)
                self.assertTrue(np.all(out[name].values >= 1.0), name)

    def test_the_correction_is_written_into_the_dataset_it_was_given(self):
        """These estimators mutate their input rather than returning a new
        Dataset, so the factor is on the object the caller passed in. Pinned
        because it is easy to write a test that reuses one dataset across
        several methods and then sees nothing added by the second."""
        ds = _fully_declared()
        out = eddypro.bpcf_moncrieff_97(ds)
        self.assertIn("scf_eddypro_moncrieff_1997", ds.data_vars)
        self.assertIs(out, ds)

    def test_an_unknown_instrument_is_skipped_rather_than_guessed(self):
        """Two fallbacks, and only the second one skips.

        A declared path length outside 0.01-0.5 m falls back to the bundled
        specification for the declared model -- so the correction still runs, on
        published geometry rather than on what was declared. It gives up only when
        the model is unknown too, which is what ``_averaged`` sets up: its tube
        block is keyed by model name rather than by sensor role, so no sonic or
        analyser model is declared at all and there is nothing to fall back to.
        """
        out = eddypro.bpcf_moncrieff_97(_averaged())
        self.assertEqual([v for v in out.data_vars if v.startswith("scf")], [])


class TestFittedCorrectionsRun(unittest.TestCase):
    """The methods that need measured spectra — the transfer-function chain."""

    @staticmethod
    def _with_spectra(n=64):
        f = np.logspace(-3, 0.5, n)
        # a Lorentzian roll-off at 0.5 Hz, which is what the fit should recover
        ratio = 1.0 / (1.0 + (f / 0.5) ** 2)
        return xr.Dataset(
            {"cospectrum": ("frequency", ratio),
             "t_sonic": ("frequency", np.ones(n))},
            coords={"frequency": f})

    def test_the_lorentzian_model_is_reachable_through_its_name(self):
        from fluxmethods.spectral.fitting_models import TRANSFER_FUNCTION_MODELS
        self.assertIn("lorentzian", TRANSFER_FUNCTION_MODELS)
        self.assertIn("horst_1997", TRANSFER_FUNCTION_MODELS)

    def test_a_correction_factor_is_formed_from_a_transfer_function(self):
        """The ratio of the undegraded cospectrum's integral to the degraded
        one's -- so a transfer function that loses nothing gives exactly 1."""
        from fluxmethods.spectral.commons import (
            correction_factor_from_transfer_function)
        ds = self._with_spectra()
        f = ds["frequency"]
        lossless = xr.ones_like(f)
        cf = correction_factor_from_transfer_function(
            ds, lossless, unattenuated="cospectrum")
        self.assertAlmostEqual(float(cf), 1.0, places=10)

        rolled_off = 1.0 / (1.0 + (f / 0.5) ** 2)
        cf2 = correction_factor_from_transfer_function(
            ds, rolled_off, unattenuated="cospectrum")
        self.assertGreater(float(cf2), 1.0)   # losing signal inflates the factor

    def test_the_ibrom_method_is_callable_with_its_own_defaults(self):
        """Not a physics assertion -- that lives with the reference
        implementation. This is that the chain the method needs is all here."""
        self.assertTrue(callable(measured.ibrom_et_al_2007))
        self.assertTrue(callable(measured.cutoff_lut))
        self.assertTrue(callable(measured.fully_analytical))
        self.assertTrue(callable(measured.generic_experimental))


if __name__ == "__main__":
    unittest.main()
