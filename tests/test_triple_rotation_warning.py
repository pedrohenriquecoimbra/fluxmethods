"""The third rotation is applied whatever its angle, and says so where EddyPro's
would not be the same: past EddyPro's 10-deg limit, and where <v'^2> < <w'^2>
makes EddyPro's half atan differ from the half atan2 by 90 deg."""

import logging
import unittest

import numpy as np

from fluxmethods.axis_rotation import wilczak_et_al_2001 as wz


class _Warnings(logging.Handler):
    def __init__(self):
        super().__init__(logging.WARNING)
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _turned_crosswind(sd_v, sd_w, psi0_deg, n=4000):
    """Zero-mean v, w with no cross-stress, turned by ``psi0_deg`` about u."""
    rng = np.random.default_rng(20220514)
    v0 = sd_v * rng.standard_normal(n)
    w0 = sd_w * rng.standard_normal(n)
    v0, w0 = v0 - v0.mean(), w0 - w0.mean()
    a = np.radians(psi0_deg)
    v = v0 * np.cos(a) - w0 * np.sin(a)
    w = v0 * np.sin(a) + w0 * np.cos(a)
    return 2.0 + 0.5 * rng.standard_normal(n), v, w


class TestTripleRotationWarning(unittest.TestCase):

    def _rotate(self, *wind):
        handler = _Warnings()
        wz.logger.addHandler(handler)
        try:
            out = wz.triple_rotation(*wind)
        finally:
            wz.logger.removeHandler(handler)
        return out, " ".join(handler.messages)

    def _cross_stress(self, out):
        v3, w3 = np.asarray(out.v), np.asarray(out.w)
        return float(np.mean((v3 - v3.mean()) * (w3 - w3.mean())))

    def test_nothing_is_said_where_eddypro_would_agree(self):
        out, said = self._rotate(*_turned_crosswind(0.5, 0.2, 4.0))
        self.assertLess(abs(np.degrees(out.psi)), wz.EDDYPRO_ROLL_LIMIT_DEG)
        self.assertEqual(said, "")

    def test_past_eddypros_10_degrees_it_rotates_and_says_so(self):
        out, said = self._rotate(*_turned_crosswind(0.5, 0.2, 20.0))
        self.assertGreaterEqual(abs(np.degrees(out.psi)), wz.EDDYPRO_ROLL_LIMIT_DEG)
        self.assertAlmostEqual(self._cross_stress(out), 0.0, places=9)
        self.assertIn("would leave this period double-rotated", said)

    def test_where_eddypros_atan_loses_the_quadrant_it_says_so(self):
        # atan2 turns by about 90 deg; EddyPro's half atan by the small
        # remainder, which it would apply.
        out, said = self._rotate(*_turned_crosswind(0.2, 0.5, 4.0))
        self.assertGreater(abs(np.degrees(out.psi)), 45)
        self.assertAlmostEqual(self._cross_stress(out), 0.0, places=9)
        self.assertIn("atan (not atan2)", said)
        self.assertIn("which it would apply", said)


if __name__ == "__main__":
    unittest.main()
