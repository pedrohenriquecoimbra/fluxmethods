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

| module | methods |
|---|---|
| `wilczak_et_al_2001` | `double_rotation`, `triple_rotation`, `planarfit` |
| `mauder_et_al_2013` | `mauder2013` — MAD despiking, per averaging period |
| `vickers_et_al_1997` | `spike_detection_vickers97`, `linear_interpolate_spikes` |
| `resampling` | `nearest`, `linear`, `fft_resample`, `block_average` |

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
