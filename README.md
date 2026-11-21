# fluxmethods

Eddy-covariance methods, published as functions and nothing else.

There is no program here. No pipeline, no configuration file, no registry, no
I/O, no units layer — only the estimators, each one importable on its own:

```python
from fluxmethods.axis_rotation.wilczak_et_al_2001 import double_rotation
from fluxmethods.despiking.mauder_et_al_2013 import mauder2013
```

Each module is named for the work it implements, so a citation and an import are
the same string.

The functions are Python. Two of them are also here in R and in Julia, as a file
of the same name beside the Python one, held to the same numbers:
[one method, three languages](#one-method-three-languages).

## What this is, and what it is not

These modules are the estimators of
[`onefluxpro`](https://github.com/pedrohenriquecoimbra/ONEFluxPro), lifted out
of it, and **that package keeps no copy**: it registers these functions and runs
these files. A method is one file, and the file is here. They are not a
reimplementation and not a rewrite.

That is the whole difference from the same author's
[`miniflux`](https://github.com/pedrohenriquecoimbra/miniflux), which is a
*minimal reimplementation* of the same physics on the standard library.

| | question it answers |
|---|---|
| `onefluxpro` | give me the whole processing chain, with provenance and the engine comparisons |
| `miniflux` | what is the smallest thing that is still a flux program? |
| `fluxmethods` | give me the method, not the program |

Use this one when you have your own data layer and want a published, cited,
tested implementation of a single step to call from inside it.

## What is here

| step | module | methods |
|---|---|---|
| `axis_rotation` | `wilczak_et_al_2001` | `double_rotation`, `triple_rotation`, `planarfit` |
| `despiking` | `mauder_et_al_2013` | `mauder2013` — MAD, per averaging period |
| `despiking` | `vickers_et_al_1997` | `spike_detection_vickers97`, `linear_interpolate_spikes` |
| `detrending` | `commonly_used` | `block_average`, `linear_detrend` |
| `resampling` | `commonly_used` | `nearest`, `linear`, `fft_resample`, `block_average` |
| `time_lag` | `maximisation` | `time_lag`, `time_lag_w_default` |
| `time_lag` | `fixed` / `prescribed` | `fix_time_lag`, `prescribed_time_lag` |
| `time_lag` | `commons` | `acq_freq`, `seconds_to_shift`, `shift_to_seconds` |
| `spectral` | `analytic` | `block_average_highpass`, `sonic_response`, `analytic_tube` |
| `spectral` | `eddypro` | `bpcf_moncrieff_97`, `bpcf_anemometric_fluxes`, `bpcf_momentum` |
| `spectral` | `measured` | `ibrom_et_al_2007`, `generic_experimental`, `fully_analytical`, `cutoff_lut` |
| `spectral` | `lpfc` / `sensor_table` | `lpfc_lut`, `sensor_geometry` |
| `units` | `webb_et_al_1980` | the WPL density correction |
| `units` | `ibrom_et_al_2007` | molar density → dry mixing ratio |
| `core` | — | units, constants, measure types, moist-air thermodynamics, the run record, `signal`, `utils` |

Full references are in [`NOTICE`](NOTICE).

**The layout is the program's.** One package per processing step, under the
names `onefluxpro` gives its correction kinds: `corrections/<step>/` there holds
the facade that registers a step's methods and hands them a period, and
`<step>/` here holds the methods. `core` is the layers they rest on, kept here
too so the collection stands without a program around it
([inside a program](#inside-a-program)).

## One method, three languages

`double_rotation` and the detrending `block_average` are each here three times:

| file | language | call |
|---|---|---|
| `axis_rotation/wilczak_et_al_2001.py` | Python | `double_rotation(u, v, w)` → an object with `u`, `v`, `w`, `theta`, `phi` |
| `axis_rotation/wilczak_et_al_2001.R` | R | `double_rotation(u, v, w)` → a list of the same names |
| `axis_rotation/wilczak_et_al_2001.jl` | Julia | `double_rotation(u, v, w)` → a named tuple of the same names |
| `detrending/commonly_used.py` | Python | `block_average(x)`, a named DataArray → a Dataset |
| `detrending/commonly_used.R` | R | `block_average(x)` → the series |
| `detrending/commonly_used.jl` | Julia | `block_average(x)` → the series |

`block_average` is its mean and nothing else, so with the summation below its
three are identical bit for bit, with nothing set aside. Each file carries the
summation itself rather than sharing it: a file is then all of the code a call
of it runs, which is what its digest in a run's record claims.

**The Python file is the method. The other two are translations of it**, and
that is all they are: statement for statement, the same operations in the same
order, with the same handling of a missing value, so that the three give the
same numbers and not merely close ones. A translation carries no idea of its
own; where it and the Python disagree, the translation is wrong.

They sit beside the file they translate, under its name, so the layout stays
the equivalence map: a path names a method, and the extension names the
language. Neither needs a package: base R, and Julia with its standard library.
`source()` or `include()` the file and call the function.

```r
source("fluxmethods/axis_rotation/wilczak_et_al_2001.R")
out <- double_rotation(u, v, w)
```

```julia
include("fluxmethods/axis_rotation/wilczak_et_al_2001.jl")
out = double_rotation(u, v, w)
```

### How they are kept equal

An element-wise product or sum is the same in every language: IEEE 754 fixes
it. Two things in a rotation are not fixed by the standard, and each
translation deals with both the same way, by doing what the Python does.

**The mean.** The angles are arctangents of means, and languages add a series
up differently. `numpy.nanmean` sums pairwise in blocks of 128 with eight
running sums; R's `mean` sums in extended precision and corrects with a second
pass; Julia's `sum` is pairwise with other blocks. The three means differ in
the last bit. So both translations carry `pairwise_sum`, numpy's summation
written out in numpy's order, and it gives numpy's sum exactly
(`tests/test_translations.py` checks the model on 300 lengths).

**The arctangent, the cosine and the sine.** R's are the C library's. Julia's
are its own, written in Julia, and differ from the C library's in the last bit
for about one argument in ten (measured against glibc: `atan` 18 %, `cos` and
`sin` 3 %), so the Julia file calls the C library's where there is one (Linux,
macOS) and says in `LIBM` which it used. numpy's `cos` and `sin` are the C
library's as well.

**What is left, and it is on the Python side.** `numpy.arctan2` is not the C
library's `atan2` on every machine: on a CPU with AVX-512 numpy computes it
with a vector routine of its own, and the two differ in the last bit for a few
arguments in a hundred (3–7 % measured, numpy 2.4.6, AMD Zen 5). Where they
differ, the Python's angle is one unit in the last place from the
translations', and the rotated wind follows it: about one part in 1e16 of the
wind speed.

So the claim is in three parts, and `tests/test_translations.py` holds each,
skipping a language whose program is not installed:

- R and Julia are **identical**, bit for bit;
- each is **identical to the Python with `math.atan2` in place of
  `numpy.arctan2`** — which shows the arctangent is the whole difference;
- each is the Python as it stands **to twelve digits**, with a floor of
  1e-12 m/s, because a rotation zeroes the mean of `v` and of `w` and next to
  zero a relative tolerance says nothing.

Making the third line read "identical" as well is a change to the Python —
`math.atan2` for the two angles — and to the copy of it in the reference
implementation, since the two are one code. It is not made here.

*Identical* means the same value. A NaN where both `u` and `v` are missing can
come back with the other sign bit in R; that is not a value.

### In a pipeline

`descriptor.json` declares each translation as a row of its own, beside the
Python row it twins, with `"language": "r"` or `"julia"` in its `run` and the
file as its callable (`axis_rotation/wilczak_et_al_2001.R:double_rotation`).
The library block declares a provider per language, so a configuration of
`onefluxpro` selects one by name:

```
method = double_rotation@fluxmethods          # the Python
method = double_rotation@fluxmethods_r        # the R translation
method = double_rotation@fluxmethods_julia    # the Julia translation
```

and likewise `block_average@fluxmethods_r` and `@fluxmethods_julia` for the
detrending. A run is stamped with the file of the language that ran and its
digest. `onefluxpro` runs its sample chain once with each rotation and
compares every output (`scripts/compare_languages.py` there), and once with a
different language at each step — despiking in Python, rotation in R,
detrending in Julia (`scripts/mixed_language_pipeline.py`).

### Adding a translation

One file beside the Python, same name, the language's extension; the function
under the Python's name, taking arrays and returning its results by name; a row
in `descriptor.json`; a case in `tests/test_translations.py`. Translate the
statements, not the idea: a faster or more idiomatic version is a different
method, and belongs somewhere it can say so.

## Coverage

Of the 34 correction methods `onefluxpro` registers, nine are
GEddySoft's own and bridge to GEddySoft rather than living here. **All but one of
the other 25 are here.**

The exception is `constants`, and it is not an estimator: it calls
`library_globals.reset()`, returns `{}`, and exists to undo a `constants@<engine>`
swap performed by that package's library bridge. There are no engines here to
swap from, so there is nothing for it to restore. The constants themselves *are*
here, in `core/constants.py`.

## Inside a program

[`onefluxpro`](https://github.com/pedrohenriquecoimbra/ONEFluxPro) registers
these functions under the names a configuration writes and runs them from this
tree: a submodule pinned at one commit in a checkout, the same tree carried
inside the package in a wheel. Three things here exist for a program that does
that, and none of them changes what a function computes when it is called on its
own.

1. **`core/units.py` computes on the registry the application uses, or on one a
   host names.** A library must not install a global registry: importing it would
   repoint the registry of whatever imported it, and pint refuses to operate
   across two, so every quantity the program had already made would be orphaned
   by the import. So the registry is adopted. A program that keeps a registry of
   its own and installs it only for the length of a run names it once
   (`fluxmethods.core.units.adopt(registry)`), and the methods compute on that
   one whichever is the application's at the time of a call.

   **Adopted is not untouched.** Two global changes are made to a registry this
   package does not own, and both are necessary:

   * the unit definitions these methods rely on (`ppm`, `ppt`, `µmol`, `ppbv`)
     are checked **by meaning** and defined where the registry disagrees, with a
     warning naming what changed. Not by whether the name resolves: pint resolves
     a great many strings through its prefix parser, and `ppt` is read as
     *pico-pint*, a volume, where these methods mean parts per thousand. A guard
     that asked "is this name already known?" got `True` and skipped it, and the
     VOC path then computed in pico-pints, thirteen orders of magnitude and a
     dimension out, silently. Every meaning is pinned by a test;
   * `force_ndarray_like` is **set**. On a default registry a scalar quantity's
     magnitude is a `float`; with the flag it is a 0-d ndarray, which is what
     this code is written against. It is set explicitly here and pinned by a
     test rather than left to `pint_xarray.setup_registry`, which happens to set
     it too.
2. **`core/provenance.py` asks a host what a method cites.** A record written by
   a method names the work it implements, and a collection registers nothing
   under a name, so it has no catalogue to read that from. A program that runs
   the methods under its own names answers by binding
   `fluxmethods.core.provenance.citations`; unbound, a record carries no
   citation.
3. **`core/constants.py` is a namespace a program may set.** The density and
   conversion methods read their constants from it when they are called, so a
   run that computes with another engine's gas constant sets it here as well as
   in its own.

Some of the layout follows from standing alone:

* `spectral/commons` holds the seven helpers the estimators call. What bins,
  ensemble-averages, transforms or draws spectra is a program's, and stays in
  one.
* `core/utils` is `resolve_variable` alone; config parsing and path handling
  are a program's business, not a method's.
* `core/micrometeorology` carries the barometric rule for an unmeasured
  pressure (`assumed_air_pressure`), which a program keeps with its importer.
* `despiking/vickers_et_al_1997` imports matplotlib where the one diagnostic
  plot is drawn, and `spectral/eddypro` imports pint inside the single branch
  that can meet a quantity, so neither costs a caller who does not need it.

## Two conventions worth knowing before you call one

They are how the program's facades call these functions, and are kept rather
than tidied because a tidier signature would change every caller.

**The rotations answer with an attribute object, not a tuple.**

```python
out = double_rotation(u, v, w)
out.u, out.v, out.w, out.theta, out.phi
```

**`mauder2013` is xarray-native and per-period.** It takes a named `DataArray`
and answers a `Dataset`. Where the array carries a `date` dimension, the median
and MAD are formed *within* each period rather than over the whole file — which
is what Mauder et al. (2013) describe, and the difference between a threshold
set by the spread within a half-hour and one set by the drift between them:

```python
mauder2013(xr.DataArray(x, dims=("date", "time"), name="w"), q=7)["w"]
```

The resampling functions are plain `numpy`: `f(values, times, target)` with
`int64` nanosecond timestamps, and bin edges at the midpoints between target
points.

**The lag estimators want a `time` coordinate in float seconds.** That is
`onefluxpro`'s convention, and `time_lag.commons.acq_freq` derives
the rate by differencing it. Hand it a `datetime64` coordinate and the difference
is in *nanoseconds*, so you get 1e-8 Hz and a silently wrong answer — pass
`acq_freq=` explicitly if your data is timestamped.

**`block_average` is two different methods.** In `detrending` it removes a
period's mean and leaves the samples in place; in `resampling` it is the mean of
the samples falling in each target interval. Neither is exported at the top
level — reach for them through their module.

## A known defect, inherited and kept on purpose

`signal.nanlinfit` fits on a *compacted* axis — it deletes the NaNs and fits
against `arange` of what is left — while `nandetrend` evaluates that trend at the
original indices. So an exact straight line with one interior gap does not
detrend to zero, and the residual grows along the series. Since despiking puts
gaps in before detrending runs, this is not a rare input.

**It is not this project's bug.**
GEddySoft v4.1 compacts and evaluates exactly the same way, in the same two files;
`core/signal.py` here transcribes that faithfully, and the transcription is a bridged
twin held equal to GEddySoft's own code to twelve digits. Correcting it in any of
the three would not fix a defect — it would manufacture a divergence and turn
that comparison red.

So the fix belongs in GEddySoft's patch bundle, among the patches that change
what the release computes. `tests/test_signal_and_detrending.py` pins the
behaviour, says whose it is, and is what should start failing when an upstream
fix reaches this port.

## Install

**Not on PyPI, and not meant to be** — `pip install fluxmethods` does not work.
A program that runs these methods pins this repository as a submodule and ships
the tree it pinned. To call them yourself, install from the repository:

```
pip install git+https://github.com/pedrohenriquecoimbra/fluxmethods.git@v0.1.5
```

Python >= 3.9. `numpy`, `pandas`, `scipy`, `xarray`, `regorator`.

The R and Julia files install with the package, as data beside the modules; to
call one you need R (any 4.x; nothing but base R is used) or Julia (>= 1.9).

Append `#egg=fluxmethods[units]` to that URL, or `pip install .[units]` from a
checkout, to add `pint` for the two methods in `fluxmethods.units`.
That package is reached lazily, so `import fluxmethods` costs no unit registry
and the other twenty-two methods do not need one. No plotting stack either: the
one diagnostic plot imports `matplotlib` where it is drawn.

`pip install -e .` from a checkout for the working copy.

**The version is `fluxmethods.__version__`**, and that attribute is the single
source: `pyproject.toml` reads it for the distribution version, so the import and
`importlib.metadata.version("fluxmethods")` are the same string. It is an
attribute rather than metadata because that is what `core/provenance.py` reads
(`from .. import __version__`) to stamp a run.

## Tests

```
python -m pytest tests/
```

`unittest` only, and a smoke test rather than a physics one — the physics is
checked by `onefluxpro`'s suite, which runs these files against EddyPro's and
GEddySoft's output where the data and the published comparisons are.
`test_translations.py` runs the R and Julia
files against the Python and skips each language whose program (`Rscript`,
`julia`) is not on PATH.

## Licence

EUPL-1.2. See [`LICENCE`](LICENCE) and [`NOTICE`](NOTICE).
