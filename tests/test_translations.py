"""The R and Julia files give the numbers the Python gives.

A method here may have a translation beside it: `wilczak_et_al_2001.R` and
`wilczak_et_al_2001.jl` are `double_rotation` of `wilczak_et_al_2001.py`,
statement for statement, and `detrending/commonly_used.R` and `.jl` are
`block_average` of `commonly_used.py`. This runs each on one wind and compares.
The block average has no arctangent in it, so its three are identical with
nothing set aside; the rotation's claims are the three below.

There is no bridge in this repository, so the wind and the answers cross in
raw float64 files, which is exact. Each language is run by its own program
(`Rscript`, `julia`) and skipped where that is not on PATH.

Three claims, each with its test:

- the two translations are **identical**, bit for bit;
- each is **identical to the Python once the Python's arctangent is the C
  library's** (`math.atan2`), which is the one operation the three do not
  share: R's `atan2` is the C library's, the Julia file calls the same, and
  numpy's `arctan2` is its own routine on a CPU with AVX-512;
- each is the Python as it stands **to twelve digits**, with a floor of
  1e-12 m/s: a rotation zeroes the mean of v and of w, and next to zero a
  relative tolerance says nothing.
"""

import math
import pathlib
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from fluxmethods.axis_rotation import wilczak_et_al_2001 as python

HERE = pathlib.Path(python.__file__).resolve().parent
SIZE = 36000

R_PROGRAM = '''
arguments <- commandArgs(trailingOnly = TRUE)
source(arguments[[1L]])
a <- readBin(arguments[[2L]], "double", n = 3L * {size}L, size = 8L, endian = "little")
d <- double_rotation(a[1:{size}], a[{size} + 1:{size}], a[2L * {size} + 1:{size}])
writeBin(c(d$u, d$v, d$w, d$theta, d$phi), arguments[[3L]], size = 8L, endian = "little")
'''

JULIA_PROGRAM = '''
include(ARGS[1])
a = Vector{{Float64}}(undef, 3 * {size})
read!(ARGS[2], a)
d = double_rotation(a[1:{size}], a[{size}+1:2*{size}], a[2*{size}+1:end])
write(ARGS[3], vcat(d.u, d.v, d.w, d.theta, d.phi))
'''


def wind(seed=20261113):
    rng = np.random.default_rng(seed)
    out = [3 + rng.normal(size=SIZE), -1.5 + rng.normal(size=SIZE),
           0.1 + 0.3 * rng.normal(size=SIZE)]
    for series in out:
        series[rng.integers(0, SIZE, 40)] = np.nan
    return out


def flat(result):
    return np.concatenate([result.u, result.v, result.w,
                           [result.theta, result.phi]])


def in_python(u, v, w, c_atan2=False):
    """The Python's answer; with ``c_atan2``, its arctangent from the C library."""
    if not c_atan2:
        return flat(python.double_rotation(u, v, w))
    numpy_ = python.np

    class CLibrary:
        def __getattr__(self, name):
            return getattr(numpy_, name)

        @staticmethod
        def arctan2(y, x):
            return numpy_.float64(math.atan2(float(y), float(x)))

    python.np = CLibrary()
    try:
        return flat(python.double_rotation(u, v, w))
    finally:
        python.np = numpy_


def translated(language, u, v, w):
    """The translation's answer, or ``None`` where its program is not here."""
    program, source, text, flags = {
        "r": ("Rscript", "wilczak_et_al_2001.R", R_PROGRAM, ["--vanilla"]),
        "julia": ("julia", "wilczak_et_al_2001.jl", JULIA_PROGRAM,
                  ["--startup-file=no"]),
    }[language]
    if shutil.which(program) is None:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        script = scratch / ("run" + pathlib.Path(source).suffix)
        script.write_text(text.format(size=SIZE), encoding="ascii")
        np.concatenate([u, v, w]).astype("<f8").tofile(scratch / "in.bin")
        subprocess.run([program, *flags, str(script), str(HERE / source),
                        str(scratch / "in.bin"), str(scratch / "out.bin")],
                       check=True, capture_output=True, timeout=600)
        return np.fromfile(scratch / "out.bin", dtype="<f8")


BLOCK_R = '''
arguments <- commandArgs(trailingOnly = TRUE)
source(arguments[[1L]])
a <- readBin(arguments[[2L]], "double", n = {size}L, size = 8L, endian = "little")
writeBin(block_average(a), arguments[[3L]], size = 8L, endian = "little")
'''

BLOCK_JULIA = '''
include(ARGS[1])
a = Vector{{Float64}}(undef, {size})
read!(ARGS[2], a)
write(ARGS[3], block_average(a))
'''


def block_averaged(language, series):
    """The translation's block average, or ``None`` without its program."""
    program, source, text, flags = {
        "r": ("Rscript", "commonly_used.R", BLOCK_R, ["--vanilla"]),
        "julia": ("julia", "commonly_used.jl", BLOCK_JULIA,
                  ["--startup-file=no"]),
    }[language]
    if shutil.which(program) is None:
        return None
    with tempfile.TemporaryDirectory() as scratch:
        scratch = pathlib.Path(scratch)
        script = scratch / ("run" + pathlib.Path(source).suffix)
        script.write_text(text.format(size=series.size), encoding="ascii")
        series.astype("<f8").tofile(scratch / "in.bin")
        subprocess.run([program, *flags, str(script),
                        str(HERE.parent / "detrending" / source),
                        str(scratch / "in.bin"), str(scratch / "out.bin")],
                       check=True, capture_output=True, timeout=600)
        return np.fromfile(scratch / "out.bin", dtype="<f8")


def same(a, b):
    """Bit for bit, one NaN being another: a NaN's sign is not a value."""
    return bool(np.all((a == b) | (np.isnan(a) & np.isnan(b))))


class TestTranslationsOfTheDoubleRotation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wind = wind()
        cls.python = in_python(*cls.wind)
        cls.answers = {language: translated(language, *cls.wind)
                       for language in ("r", "julia")}

    def _answer(self, language):
        answer = self.answers[language]
        if answer is None:
            self.skipTest({"r": "no Rscript on PATH (https://cran.r-project.org)",
                           "julia": "no julia on PATH "
                                    "(https://julialang.org/downloads)"}[language])
        return answer

    def test_each_is_the_python_to_twelve_digits(self):
        for language in ("r", "julia"):
            with self.subTest(language=language):
                answer = self._answer(language)
                np.testing.assert_allclose(answer, self.python, rtol=1e-12,
                                           atol=1e-12, equal_nan=True)
                self.assertTrue(np.array_equal(np.isnan(answer),
                                               np.isnan(self.python)))

    def test_each_is_the_python_bit_for_bit_with_the_c_librarys_arctangent(self):
        expected = in_python(*self.wind, c_atan2=True)
        for language in ("r", "julia"):
            with self.subTest(language=language):
                self.assertTrue(same(self._answer(language), expected))

    def test_the_two_are_identical(self):
        self.assertTrue(same(self._answer("r"), self._answer("julia")))

    def test_the_summation_written_out_is_numpys(self):
        """`pairwise_sum` in the translations is `numpy.sum`'s order, which is
        what makes the means -- and so the angles -- the same. Pinned on the
        Python side as the model the two files transcribe."""
        def pairwise(a):
            n = len(a)
            if n < 8:
                res = 0.0
                for x in a:
                    res += float(x)
                return res
            if n <= 128:
                r = [float(x) for x in a[:8]]
                i = 8
                while i < n - (n % 8):
                    for j in range(8):
                        r[j] += float(a[i + j])
                    i += 8
                res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))
                while i < n:
                    res += float(a[i])
                    i += 1
                return res
            n2 = n // 2
            n2 -= n2 % 8
            return pairwise(a[:n2]) + pairwise(a[n2:])

        rng = np.random.default_rng(1)
        for n in list(range(1, 300)) + [1000, 4097, 36000, 36001]:
            a = rng.normal(size=n) * 10 ** rng.uniform(-3, 3)
            self.assertEqual(float(np.sum(a)), pairwise(a), n)


class TestTranslationsOfTheBlockAverage(unittest.TestCase):
    def test_each_is_the_python_bit_for_bit(self):
        import xarray as xr

        from fluxmethods.detrending.commonly_used import block_average

        series = 300 + wind(seed=5)[0]
        expected = block_average(
            xr.DataArray(series, dims="time", name="ts"))["ts"].values
        ran = 0
        for language in ("r", "julia"):
            answer = block_averaged(language, series)
            if answer is None:
                continue
            ran += 1
            with self.subTest(language=language):
                self.assertTrue(same(answer, expected))
                self.assertTrue(np.array_equal(np.isnan(answer),
                                               np.isnan(series)))
        if not ran:
            self.skipTest("neither Rscript nor julia is on PATH")


if __name__ == "__main__":
    unittest.main()
