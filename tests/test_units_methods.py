"""The two estimators that re-express quantities, called rather than imported.

These are the only methods here that need a unit registry, and the only ones that
reach into `fluxmethods.core` for constants and moist-air thermodynamics. The
fixture is one internally consistent air parcel — every quantity derived from the
same state — because a set of plausible-looking numbers cannot catch a correction
that treats a molar density and a mixing ratio alike.

The physics is pinned on the reference implementation's side, against hand-
computed values. What is pinned here is that the chain is complete: that these
run, on quantified data, and give back a finite flux in the right units.
"""

import unittest

import numpy as np
import xarray as xr

from fluxmethods.core import constants
from fluxmethods.units.webb_et_al_1980 import (temperature_flux,
                                               vapour_mole_fraction_flux,
                                               webb_et_al_1980)

R = float(constants.R.magnitude)
Mv, Md, Mco2 = 0.01802, 0.02897, 0.04401
T, P, CHI, CP = 293.0, 99000.0, 0.012, 1010.0
LV = 1e3 * (3147.5 - 2.37 * T)

N_A = P / (R * T)
N_D = N_A * (1 - CHI)
RHO_V, RHO_D = CHI * N_A * Mv, N_D * Md
RHO_M = RHO_D + RHO_V
C_CO2 = 420e-6 * N_D
WT = 0.15
H = RHO_M * CP * WT
F_CHI = N_A * 1.1e-4
FC_L0 = -20e-6


def _q(val, unit):
    return xr.DataArray(val).pint.quantify(unit)


def _parcel():
    ds = xr.Dataset({
        'co2': _q(C_CO2 * 1e3, 'mmol/m^3'),
        'h2o': _q(CHI * 1e6, 'ppm'),
        'rho_co2': _q(C_CO2 * Mco2, 'kg/m^3'),
        'rho_h2ov': _q(RHO_V, 'kg/m^3'),
        'rho_d': _q(RHO_D, 'kg/m^3'),
        'rho_m': _q(RHO_M, 'kg/m^3'),
        'h2o_mole_fraction': _q(CHI, 'dimensionless'),
        'air_temperature': _q(T, 'K'),
        'air_heat_capacity': _q(CP, 'J/kg/K'),
        'lambda_v': _q(LV, 'J/kg'),
        'air_molar_volume': _q(1 / N_A, 'm^3/mol'),
        'dry_air_molar_volume': _q(1 / N_D, 'm^3/mol'),
        'H': _q(H, 'W/m^2'),
        'FH2O_L0': _q(F_CHI, 'mol/m^2/s'),
        'E_L0': _q(Mv * F_CHI, 'kg/m^2/s'),
        'LE_L0': _q(Mv * F_CHI * LV, 'W/m^2'),
        'FC_L0': _q(FC_L0, 'mol/m^2/s'),
    })
    ds['co2'].attrs['measure_type'] = 'molar_density'
    ds['h2o'].attrs['measure_type'] = 'mole_fraction'
    return ds


class TestTheTwoInputsWplReconstructs(unittest.TestCase):
    def test_the_heat_input_is_the_air_temperature_flux(self):
        """``<w'T'>`` comes back out of the sensible heat flux, so it is the
        covariance with the humidity contribution already removed."""
        wt = temperature_flux(_parcel()).pint.to('K*m/s').pint.magnitude
        self.assertAlmostEqual(float(np.ravel(wt)[0]), WT, places=9)

    def test_the_vapour_input_is_a_mole_fraction_flux(self):
        f = vapour_mole_fraction_flux(_parcel()).pint.to('mol/m^2/s').pint.magnitude
        self.assertAlmostEqual(float(np.ravel(f)[0]), F_CHI, places=12)


class TestWebbRuns(unittest.TestCase):
    def test_the_correction_produces_a_finite_corrected_flux(self):
        out = webb_et_al_1980(_parcel())
        self.assertIsInstance(out, xr.Dataset)
        corrected = [v for v in out.data_vars if v.startswith("FC")]
        self.assertTrue(corrected, "no CO2 flux came back")
        for name in corrected:
            self.assertTrue(np.all(np.isfinite(
                np.ravel(out[name].pint.magnitude).astype(float))), name)

    def test_the_density_terms_move_the_flux_away_from_the_raw_one(self):
        """WPL exists because the raw density-path flux is wrong; a correction
        that returned it unchanged would not be doing anything."""
        out = webb_et_al_1980(_parcel())
        name = next(v for v in out.data_vars
                    if v.startswith("FC") and not v.endswith("_L0"))
        got = float(np.ravel(out[name].pint.to('mol/m^2/s').pint.magnitude)[0])
        self.assertNotAlmostEqual(got, FC_L0, places=9)
        self.assertTrue(np.isfinite(got))


class TestTheUnitsLayerIsNotSeized(unittest.TestCase):
    def test_the_registry_is_the_application_s_own(self):
        """A library must not install a global unit registry: the caller's own
        quantities belong to theirs, and pint refuses to mix two."""
        import pint

        from fluxmethods.core import units
        self.assertIs(units.ureg, pint.get_application_registry())


if __name__ == "__main__":
    unittest.main()
