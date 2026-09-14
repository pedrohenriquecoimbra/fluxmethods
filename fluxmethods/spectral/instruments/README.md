# Bundled instrument files

One JSON file per instrument model. The spectral correction reads this
directory on first use and builds the sensor table from it, so adding an
instrument to a source checkout means adding a file here — no code changes.
A site whose instrument is not listed does not need a file at all: plausible
geometry declared in the config's `[Instruments]` block always wins, and this
table is only the fallback behind placeholders.

These values only fill in *behind* the site metadata: a configuration that
declares plausible geometry always wins, and the table steps in when the
declared value is a placeholder (EddyPro writes 1.0 m and 0.1 s against every
recognised instrument whose geometry was never entered) or missing.

## Schema

| field | unit | meaning |
|---|---|---|
| `model` | — | lower-case model name; config instrument names are matched against it by longest prefix, so `hs_50_1` resolves to `hs_50` |
| `aliases` | — | additional names that resolve to this geometry |
| `vpath_length` | m | vertical (acoustic or optical) path; this is what sets the line averaging |
| `hpath_length` | m | horizontal path, kept for documentation; not consumed by the correction |
| `tau` | s | response time constant |
| `kind`, `manufacturer`, `source`, `note` | — | documentation; `source` names the published table the values transcribe |

Values must be physically plausible or the file is refused with a warning:
`vpath_length` in (0.01, 0.5) m, `tau` in (0, 1) s. These are transcription
sanity bounds, looser than the gate applied to *declared* site metadata at run
time (a declared sonic `tau` must also resolve the acquisition rate, which is
unknowable here): a bundled file is a stamped transcription of a published
table, trusted the way the table itself was when it lived in the code.

## The manifest

`_manifest.json` stamps each built-in file with a SHA-256 over the fields that
decide behaviour (`model`, `aliases`, `vpath_length`, `hpath_length`, `tau`).
At load, a stamped file whose values drifted is reported with a warning, a
file the stamp does not know is reported as user-provided, and a stamped file
that has gone missing is reported as absent; whatever is present is **used** —
the stamp alerts on provenance, it never blocks.

Never edit the manifest by hand. After changing a built-in value on purpose,
regenerate it:

```bash
python -m oneflux_preproc.corrections.spectral.sensor_table
```

Editing prose fields (`note`, `source`, …) raises no alert; the hash is over
the parsed values, so formatting and line endings do not count either.

This table transcribes EddyPro's (`src/src_rp/retrieve_sensor_params.f90`) and
is deliberately not merged with `core.constants.instruments`, which carries
PyFluxPro's values under its own names. The two disagree — a LI-7200 path is
0.127 m here and 0.125 m there — because they are two published sources, not
one source copied twice, and each belongs with the methods that cite it.
