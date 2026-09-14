"""Run provenance: a self-describing record of *how* an output was produced.

A processing result is only reproducible and comparable if you can recover the
exact conditions that made it. This module builds a JSON-serialisable
**manifest** capturing that context — code and library versions, the config (with
a content hash), the datetime range, the input files, any random seeds, and the
warnings raised during the run — and writes it as a sidecar file next to each
output. A compact subset is also embedded in the dataset attributes so it rides
inside the NetCDF itself.

The manifest is generic: it introspects the environment and the ``setup`` config
object, so it needs no per-method or per-step maintenance. See
:func:`build_manifest` for the captured fields and :class:`capture_warnings` for
collecting the run's warnings.
"""

import hashlib
import json
import logging
import os
import platform
import sys
from datetime import datetime, timezone

# Distribution names whose versions materially affect numerical results.
_TRACKED_LIBS = (
    "pandas", "xarray", "numpy", "scipy", "Pint", "pint-xarray",
    "regorator", "netCDF4", "matplotlib",
)

# Cap directory listings so a campaign directory of thousands of raw files does
# not bloat the manifest; the count is still reported in full.
_MAX_LISTED_INPUTS = 200


def _lib_versions():
    """Return ``{dist_name: version}`` for the tracked libraries, best-effort."""
    from importlib.metadata import PackageNotFoundError, version
    out = {}
    for name in _TRACKED_LIBS:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            continue
        except Exception:  # pragma: no cover - defensive
            continue
    return out


def _sha256(path):
    """Return the hex SHA-256 of a file's bytes, or ``None`` if unreadable."""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _find_seeds(obj, prefix=""):
    """Recursively collect ``{dotted.key: value}`` for any key containing 'seed'."""
    seeds = {}
    try:
        items = obj.items()
    except AttributeError:
        return seeds
    for key, value in items:
        dotted = f"{prefix}{key}"
        if hasattr(value, "items"):
            seeds.update(_find_seeds(value, prefix=f"{dotted}."))
        elif "seed" in str(key).lower():
            seeds[dotted] = value
    return seeds


def _input_identity(setup):
    """Best-effort identity of the run's raw inputs, from the ``[Files]`` block.

    Records the configured paths verbatim plus a listing (name/size/mtime) of the
    resolved input directory. Content-hashing high-frequency inputs would be
    expensive, so it is intentionally omitted here; the config hash plus the file
    listing is enough to detect that the inputs changed.
    """
    files = setup.get("Files", {}) or {}
    file_path = files.get("file_path", "") or ""
    info = {
        "file_path": file_path,
        "glob_files": files.get("glob_files", "") or "",
        "file_type": files.get("file_type", "") or "",
        "resolved_dir": os.path.abspath(file_path) if file_path else None,
        "dir_exists": False,
        "file_count": None,
        "files": [],
    }
    resolved = info["resolved_dir"]
    if resolved and os.path.isdir(resolved):
        info["dir_exists"] = True
        try:
            entries = [e for e in os.scandir(resolved) if e.is_file()]
        except OSError:
            entries = []
        info["file_count"] = len(entries)
        entries.sort(key=lambda e: e.name)
        for e in entries[:_MAX_LISTED_INPUTS]:
            try:
                st = e.stat()
                info["files"].append(
                    {"name": e.name, "size": st.st_size, "mtime": st.st_mtime})
            except OSError:
                continue
    return info


def build_manifest(setup, config_path=None, warnings=None, overrides=None):
    """Build the provenance manifest for a run.

    Args:
        setup: The loaded config (``ConfigObj`` or plain dict); ``.get`` is used
            so either works. This is the setup *as run* — with any per-run
            overrides already applied.
        config_path: Path to the config file, hashed if given.
        warnings: Iterable of warning strings collected during the run.
        overrides: The per-run overrides applied, as ``[{key, from, to}]`` (see
            :func:`~.config.apply_overrides`).

    Returns:
        A JSON-serialisable ``dict`` — see the module docstring for the fields.

    The ``config.sha256`` hashes the file *on disk*, which per-run overrides do
    not touch — so the hash alone would describe a run that did not happen.
    Recording ``overrides`` beside it is what keeps the pair honest: the file, plus
    exactly what was changed for this run.
    """
    from .. import __version__
    from . import plugins as _plugins
    from . import reproducibility as _repro
    from . import strictness as _strictness

    options = setup.get("Options", {}) or {}
    global_block = setup.get("Global", {}) or {}

    manifest = {
        "schema": "oneflux-preproc/provenance/1",
        "run_timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "oneflux_preproc_version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "libraries": _lib_versions(),
        "config": {
            "path": os.path.abspath(config_path) if config_path else None,
            "sha256": _sha256(config_path) if config_path else None,
            # What this run changed in memory. The hash above describes the file;
            # these describe the difference between the file and the run.
            "overrides": [dict(o) for o in (overrides or [])],
        },
        "site_name": (setup.get("site_name", "")
                      or global_block.get("site_name", "") or ""),
        "datetimerange": options.get("datetimerange", "") or "",
        "inputs": _input_identity(setup),
        "seeds": _find_seeds(options),
        "plugins": _plugins.plugin_status(setup),
        "reproducibility": _repro.lock_status(setup),
        "bibliography": _bibliography(setup),
        "warnings": list(warnings or []),
        # Whether refusals were waived, and which ones actually were. The flag is
        # recorded even when nothing tripped, because "this run was allowed to
        # skip steps" is a property of the run and not only of what went wrong;
        # the list is lifted out of the captured warnings so a reader does not
        # have to grep a log to find out what is missing from the result.
        "permissive": _strictness.is_permissive(setup),
        "waivers": _waivers(warnings),
    }
    return manifest


def _waivers(warnings):
    """The permissive waivers among the run's captured warnings, in order."""
    from . import strictness

    return [w for w in (warnings or []) if strictness.WAIVER_PREFIX in str(w)]


def _bibliography(setup):
    """Every published source the configured methods rest on, by step.

    A processed flux is a claim built out of a dozen papers, and the citation
    list is otherwise something a user has to reassemble by hand from the method
    names. Built from the configuration rather than from what ran, so a
    manifest written for a failed run still says what the run was asking for.

    Shape: ``{"<kind>:<method>": [references]}``, plus ``"_all"`` -- the
    deduplicated list, which is the part a data note would quote.
    """
    from ..corrections.main import is_capture, kind_of, methods_of

    out, seen = {}, []
    blocks = list((setup.get("Corrections", {}) or {}).items())
    qaqc = setup.get("qaqc_method") or (setup.get("Options", {}) or {}).get(
        "qaqc_method")
    if qaqc:
        blocks.append(("qaqc", {"method": qaqc}))

    for name, block in blocks:
        if name != "qaqc" and is_capture(name):
            continue
        kind = "qaqc" if name == "qaqc" else kind_of(name)
        for method in methods_of(block):
            if not method:
                continue
            cite = references_for(kind, method)
            if cite:
                out[f"{kind}:{method}"] = cite
                seen.extend(c for c in cite if c not in seen)
            else:
                reason = unreferenced_reason(kind, method)
                if reason:
                    out[f"{kind}:{method}"] = []
    if seen:
        out["_all"] = sorted(seen)
    return out


def compact_for_attrs(manifest):
    """Return the small provenance subset to embed in the dataset attributes."""
    return {
        "schema": manifest.get("schema"),
        "run_timestamp": manifest.get("run_timestamp"),
        "oneflux_preproc_version": manifest.get("oneflux_preproc_version"),
        "python": manifest.get("python"),
        "libraries": manifest.get("libraries", {}),
        "config_sha256": (manifest.get("config") or {}).get("sha256"),
        "datetimerange": manifest.get("datetimerange"),
        # The citation list travels with the data. A NetCDF handed to someone
        # else should not need this repository to say which papers its numbers
        # rest on.
        "bibliography": (manifest.get("bibliography") or {}).get("_all", []),
        "n_warnings": len(manifest.get("warnings", [])),
    }


# --------------------------------------------------------------------------- #
# what ran, and on what
# --------------------------------------------------------------------------- #
# ``corrections_applied`` is a structured record, not a sentence the steps
# concatenate:
#
#     "despiking w/ vickers_et_al_1997; axis rotation w/ double_rotation; ..."
#
# records that something ran but not *on what*, and can only be read back by eye
# or by substring search -- so a test asking whether the time lag was applied
# cannot tell "applied to co2" from "applied to nothing". The record below says
# which variables a step touched and which parameters it was given, in order, and
# renders back to that sentence for a human (:func:`render_applied`).

#: The attribute holding the record, on the dataset and on each variable.
APPLIED_ATTR = "corrections_applied"

#: The registration metadata key holding extra fields for that record.
APPLIED_META = "record"


#: ``(kind, method) -> references``, filled on first use.
#:
#: The registries are read once rather than per record: ``record_applied`` runs
#: per variable per period, and rebuilding the catalogue there would put a
#: registry walk inside the correction loop. Plugins register before a run
#: starts, so the mapping is stable by the time anything is recorded;
#: :func:`reset_reference_cache` exists for tests that register mid-process.
_REFERENCE_CACHE = {}


def reset_reference_cache():
    """Forget the cached references (after registering a method at run time)."""
    _REFERENCE_CACHE.clear()


def references_for(kind, method):
    """The published sources for a registered ``method``, or ``[]``.

    Resolves through the catalogue, so a retired name still finds the citation of
    the routine it now points at -- a config written before a rename must not
    produce an output that cites nothing.

    An empty list is not necessarily a gap: an elementary operation or a file
    format has no paper to cite, and those entries say so in
    ``no_scientific_reference``. :func:`unreferenced_reason` reads that.
    """
    key = (str(kind), str(method))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = _lookup(*key)
    return list(_REFERENCE_CACHE[key][0])


def unreferenced_reason(kind, method):
    """Why a method carries no citation, or ``None`` if it carries one."""
    key = (str(kind), str(method))
    if key not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[key] = _lookup(*key)
    return _REFERENCE_CACHE[key][1]


def _lookup(kind, method):
    """``(references, no_scientific_reference)`` for one registered method."""
    try:
        from ..catalogue import catalogue, resolve_method
    except Exception:  # noqa: BLE001 - provenance must never break a run
        return ([], None)
    try:
        cat = catalogue()
        group = cat.get(f"correction:{kind}") or cat.get(kind) or {}
        name = resolve_method(group, method)
        if name is None:
            # A step may name a kind the catalogue groups differently (qaqc,
            # reader, process). Fall back to a scan rather than reporting no
            # citation for a method that has one.
            for candidate in cat.values():
                found = resolve_method(candidate, method)
                if found is not None:
                    group, name = candidate, found
                    break
        info = (group or {}).get(name, {}) if name else {}
        return (list(info.get("references") or []),
                info.get("no_scientific_reference"))
    except Exception:  # noqa: BLE001 - as above
        return ([], None)


def applied_record(kind, method, on=(), **parameters):
    """One step of a correction chain, as a record.

    ``kind`` is the correction kind, ``method`` the registered routine that ran,
    ``on`` the variables it acted on (empty for a step that acts on the dataset
    as a whole), and ``parameters`` whatever the step needs a reader to know --
    the measure a conversion came from, the measure a density correction was
    owed for. Values that are ``None`` are dropped: an absent parameter and one
    explicitly set to nothing are not the same claim.
    """
    entry = {"kind": str(kind), "method": str(method)}
    # What the step cites. A record that names a routine but not the work it
    # implements leaves a reader to guess which Ibrom 2007, or which of two
    # boundary rules -- and the registry knew the answer all along.
    cite = references_for(kind, method)
    if cite:
        entry["cite"] = cite
    else:
        reason = unreferenced_reason(kind, method)
        if reason:
            entry["cite"] = []
            entry["cite_note"] = reason
    on = [str(v) for v in (on or ())]
    if on:
        entry["on"] = on
    params = {k: v for k, v in parameters.items() if v is not None}
    if params:
        entry["with"] = params
    return entry


def as_applied(value):
    """Whatever an attribute holds, as a list of records.

    Accepts the live list, the JSON a NetCDF round-trip turns it into, and the
    empty/missing case. Anything else is reported as one opaque record rather
    than dropped -- a provenance field that silently discards what it cannot
    parse is worse than one that says it does not understand.
    """
    if not value:
        return []
    if isinstance(value, dict):
        return [value]
    if isinstance(value, (list, tuple)):
        return [v for v in value if v]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (TypeError, ValueError):
            return [{"kind": "unparsed", "method": value}]
        return as_applied(parsed)
    return [{"kind": "unparsed", "method": str(value)}]


def step_record(info):
    """The extra fields a registration asks to have on its applied record.

    A registration may carry ``record=`` in its metadata: a mapping merged into
    the step's record wherever that step is applied, for anything the registry
    knows and the routine's arguments do not -- which implementation ran, the
    digest of the upstream file it ran. A callable is called for it, which is how
    a value that can only be resolved at run time gets in.
    """
    extra = (info or {}).get(APPLIED_META)
    try:
        if callable(extra):
            extra = extra()
        return dict(extra or {})
    except Exception:  # noqa: BLE001 - provenance must never break a run
        logger.debug("step record for %r could not be built", info, exc_info=True)
        return {}


def record_applied(attrs, kind, method, on=(), **parameters):
    """Append one step to ``attrs`` and return the whole record.

    ``attrs`` is a variable's or a dataset's attribute mapping; the step is
    appended in execution order, so the record reads as the chain that ran.
    """
    applied = as_applied(attrs.get(APPLIED_ATTR))
    applied.append(applied_record(kind, method, on=on, **parameters))
    attrs[APPLIED_ATTR] = applied
    return applied


def render_applied(value, sep="; "):
    """The record as a line a person can read.

    ``despiking w/ vickers_et_al_1997 on u, v, w`` -- the same phrasing the
    concatenated string used, so the readable form did not change when the
    stored form did.
    """
    parts = []
    for entry in as_applied(value):
        text = f"{entry.get('kind', '?')} w/ {entry.get('method', '?')}"
        if entry.get("on"):
            text += " on " + ", ".join(entry["on"])
        if entry.get("with"):
            text += " (" + ", ".join(
                f"{k}={v}" for k, v in entry["with"].items()) + ")"
        parts.append(text)
    return sep.join(parts)


def sidecar_path(output_path):
    """Return the manifest path sitting next to an output file.

    ``.../SITE_20240101.nc`` -> ``.../SITE_20240101.manifest.json``.
    """
    base, _ext = os.path.splitext(output_path)
    return base + ".manifest.json"


def write_manifest(output_path, manifest):
    """Write ``manifest`` as the JSON sidecar for ``output_path``; return its path."""
    path = sidecar_path(output_path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


class capture_warnings:
    """Context manager collecting ``WARNING``-and-above log records into a list.

    Usage::

        with capture_warnings() as warns:
            ... run the pipeline ...
        manifest = build_manifest(setup, warnings=warns)

    The handler is attached to the root logger so warnings from anywhere in the
    pipeline are captured, then removed on exit.
    """

    def __init__(self, level=logging.WARNING):
        self._level = level
        self._records = []
        self._handler = None

    def __enter__(self):
        records = self._records

        class _Collector(logging.Handler):
            def emit(self, record):
                try:
                    records.append(f"{record.levelname} {record.name}: "
                                   f"{record.getMessage()}")
                except Exception:  # pragma: no cover - never break the run
                    pass

        self._handler = _Collector(level=self._level)
        logging.getLogger().addHandler(self._handler)
        return records

    def __exit__(self, *exc):
        if self._handler is not None:
            logging.getLogger().removeHandler(self._handler)
        return False
