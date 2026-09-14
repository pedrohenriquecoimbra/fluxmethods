# fluxmethods

Eddy-covariance methods, published as functions and nothing else.

There is no program here. No pipeline, no configuration file, no registry, no
I/O, no units layer — only the estimators, each one importable on its own:

```python
from fluxmethods.wilczak_et_al_2001 import double_rotation
from fluxmethods.mauder_et_al_2013 import mauder2013
```

Each module is named for the work it implements, so a citation and an import are
the same string.

## What this is, and what it is not

These modules are the estimators of
[`oneflux_preproc`](https://github.com/pedrohenriquecoimbra/ONEFlux-preproc),
lifted out of it **unchanged**. They are not a reimplementation and not a
rewrite: the file here and the file there are the same code, and
`oneflux_preproc` checks that on every release by running the two against each
other and requiring them equal to twelve digits. If they ever disagree, one of
them has drifted and the test says which.

That is the whole difference from the same author's
[`miniflux`](https://github.com/pedrohenriquecoimbra/miniflux), which is a
*minimal reimplementation* of the same physics on the standard library.

| | question it answers |
|---|---|
| `oneflux_preproc` | give me the whole processing chain, with provenance and the engine comparisons |
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

**The layout is the equivalence map.** One package per processing step, plus
`core`, the same tree as `oneflux_preproc`'s `corrections/<step>/` and `core/` —
path for path and filename for filename. `diff -r` the two and every difference
should be one you can name. They are all named below.

## Coverage

Of the 34 registered correction methods in the reference implementation, nine are
GEddySoft's own and bridge to GEddySoft rather than living here. **All but one of
the other 25 are here.**

The exception is `constants`, and it is not an estimator: it calls
`library_globals.reset()`, returns `{}`, and exists to undo a `constants@<engine>`
swap performed by that package's library bridge. There are no engines here to
swap from, so there is nothing for it to restore. The constants themselves *are*
here, in `core/constants.py`.

## The differences from the code this copies

Every file is a verbatim copy except these, and each one is deliberate:

1. **`core/units.py` takes the application's unit registry** instead of building
   one and calling `pint.set_application_registry`. That is right for a program,
   which owns its process, and wrong for a library: importing fluxmethods would
   otherwise repoint the global registry of whatever imported it, and pint refuses
   to operate across two registries. The definitions these methods rely on are
   added to whatever registry is in use, if it lacks them. **This is the only
   behavioural difference; the rest are imports.**
2. `despiking/vickers_et_al_1997` imports matplotlib where the one diagnostic
   plot is drawn, not at the top.
3. `spectral/eddypro` imports pint and `convert_unit` inside the single branch
   that can meet a pint quantity, so a band-pass factor costs no unit registry.
4. `spectral/commons` is a **subset**: the seven helpers the estimators here call.
   The other thirteen bin, ensemble-average, transform or draw spectra; four of
   those need a unit registry and one needs matplotlib, and none is a method.
5. `core/utils` is a **subset** too: `resolve_variable` alone, of seven. The rest
   is config parsing and path handling — a program's business, not a method's.
6. `spectral/fitting_models` drops an import of `ureg` the module never used.
7. `detrending/commonly_used`, `spectral/eddypro` and the `units/` pair have their
   imports rebound to the siblings here.

To check the claim yourself, against a checkout of the reference implementation:

```
diff -r --strip-trailing-cr <ref>/src/oneflux_preproc/corrections fluxmethods
```

`--strip-trailing-cr` matters on Windows: git normalises line endings per repo, so
without it every line of every file reads as changed and the real differences are
invisible in the noise.

## Two conventions worth knowing before you call one

They come from the reference implementation and are kept rather than tidied,
because tidying them is what would make the two copies stop being the same code.

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

**The lag estimators want a `time` coordinate in float seconds.** That is the
reference implementation's convention, and `time_lag.commons.acq_freq` derives
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

**It is not this project's bug, and not the reference implementation's either.**
GEddySoft v4.1 compacts and evaluates exactly the same way, in the same two files;
`oneflux_preproc` transcribes that faithfully, and the transcription is a bridged
twin held equal to GEddySoft's own code to twelve digits. Correcting it in any of
the three would not fix a defect — it would manufacture a divergence and turn
that comparison red.

So the fix belongs in GEddySoft's patch bundle, among the patches that change
what the release computes. `tests/test_signal_and_detrending.py` pins the
behaviour, says whose it is, and is what should start failing when an upstream
fix reaches this port.

## Install

```
pip install .
```

Python >= 3.9. `numpy`, `pandas`, `scipy`, `xarray`, `regorator`.

`pip install .[units]` adds `pint` for the two methods in `fluxmethods.units`.
That package is reached lazily, so `import fluxmethods` costs no unit registry
and the other twenty-two methods do not need one. No plotting stack either: the
one diagnostic plot imports `matplotlib` where it is drawn.

## Tests

```
python -m pytest tests/
```

`unittest` only, and a smoke test rather than a physics one — the physics is
checked against the reference implementation, on its side, where the data and
the published comparisons are.

## Licence

EUPL-1.2. See [`LICENCE`](LICENCE) and [`NOTICE`](NOTICE).
