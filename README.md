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
| `time_lag` | `maximisation` | `time_lag`, `time_lag_w_default`, `lag_series`, `xcov_kernel` |
| `time_lag` | `fixed` / `prescribed` | `fix_time_lag`, `prescribed_time_lag` and its table loaders |
| `time_lag` | `commons` | `acq_freq`, `seconds_to_shift`, `shift_to_seconds` |
| `spectral` | `lpfc` | `lpfc_lut`, `lookup_lpfc_cf`, `load_lpfc_lut_table` |
| `spectral` | `sensor_table` | `sensor_geometry`, `load_sensor_geometry`, `stamp_manifest` |
| — | `signal` | `xcov`, `nanlinfit`, `nandetrend` — shared across steps |

**The layout is the equivalence map.** One package per processing step, the same
tree as `oneflux_preproc/corrections/`, path for path and filename for filename —
so `diff -r` the two and every difference should be one you can name. Today there
are exactly two, both intentional: a lazily imported plotting stack in
`despiking/vickers_et_al_1997`, and one rebound import in
`detrending/commonly_used`. (`signal` is the one file from outside that tree; it
copies `oneflux_preproc/core/signal.py`, which is shared across steps.)

## What is NOT here yet

Thirteen registered methods that are not GEddySoft's have no copy here, in two
groups:

* **Not separable.** Seven spectral methods (`highpass_block_average`, the three
  `lowpass_analytic_*`, `lowpass_insitu_fit`, `lowpass_insitu_ibrom_2007`,
  `lowpass_insitu_cutoff`) and `constants` are registered *inside* their facade
  module, so there is no estimator file to copy. Lifting them means refactoring
  the reference implementation first.
* **Entangled with the units layer.** `webb_et_al_1980` (WPL),
  `molardensity_to_drymixingratio` (Ibrom) and the three
  `bandpass_moncrieff_1997_*` import `core.constants`, `core.measure_type`,
  `core.units` and `core.micrometeorology`. Copying them means bringing pint and
  the package's unit conventions along, which is the layer this collection exists
  to do without.


Full references are in [`NOTICE`](NOTICE).

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

## What is deliberately not here

The methods of `oneflux_preproc` that are *facade* rather than estimator — the
ones whose work is reading a sensor table, grouping a period, resolving units,
writing variable attributes — stay there. So do the ones entangled with its
units and measure-type layer, which cannot be lifted without bringing the layer
along: the WPL density correction, the Ibrom RH-dependent conversion, and the
spectral correction chain. Splitting those is a separate job, not a copy.

## Install

```
pip install .
```

Python >= 3.9. `numpy`, `pandas`, `scipy`, `xarray`. No plotting stack: the one
diagnostic plot imports `matplotlib` where it is drawn.

## Tests

```
python -m pytest tests/
```

`unittest` only, and a smoke test rather than a physics one — the physics is
checked against the reference implementation, on its side, where the data and
the published comparisons are.

## Licence

EUPL-1.2. See [`LICENCE`](LICENCE) and [`NOTICE`](NOTICE).
