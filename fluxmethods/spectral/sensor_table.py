"""The published sensor table, read from one JSON file per instrument.

``instruments/`` beside this module holds one file per model, each carrying the
manufacturer's geometry as a reference implementation publishes it and naming
that source. Adding an instrument means adding a file; nothing here needs to
change. The README in that directory documents the schema.

A manifest (``_manifest.json``) stamps the built-in files with a hash over the
fields that decide behaviour, so the log reports whether the geometry in use
still carries the published values. A file that drifts from its stamp, or one
the stamp does not know, is reported and used -- provenance is alerted, never
enforced, because the user's file is the user's call.
"""

# Standard library
import hashlib
import json
import logging
import os

logger = logging.getLogger(__name__)

# One JSON file per instrument model, beside this module.
_BUNDLED_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "instruments")

_MANIFEST = '_manifest.json'

# The stamp covers the fields that decide behaviour -- the names a file claims
# and the geometry itself -- so editing a note or a source line raises no
# alert, while changing a value does.
_STAMPED_FIELDS = ('model', 'aliases', 'vpath_length', 'hpath_length', 'tau')

# Transcription-error bounds: a path outside 0.01--0.5 m or a response time
# outside (0, 1) s is a typo, not an instrument, and a file declaring one is
# refused loudly rather than trusted quietly. These are looser than the gate
# ``_sonic_geometry`` applies to *declared* site metadata (1/tau at or above
# the acquisition rate -- unknowable here, where no run exists yet), and
# deliberately so: a bundled file is a stamped transcription of a published
# table, trusted the way the table itself was when it lived in this module.
_PATH_BOUNDS = (0.01, 0.5)
_TAU_BOUNDS = (0.0, 1.0)


def _validated(record, filename):
    """``(model, aliases, geometry)`` from one parsed file, or ``None``."""
    if not isinstance(record, dict):
        logger.warning("instrument file %s does not hold a JSON object; skipped",
                       filename)
        return None
    model = str(record.get('model') or '').strip().lower()
    if not model:
        logger.warning("instrument file %s declares no model name; skipped",
                       filename)
        return None
    try:
        vpath = float(record['vpath_length'])
        tau = float(record['tau'])
    except (KeyError, TypeError, ValueError):
        logger.warning("instrument file %s lacks a numeric vpath_length or tau; "
                       "skipped", filename)
        return None
    if not _PATH_BOUNDS[0] < vpath < _PATH_BOUNDS[1]:
        logger.warning("instrument file %s declares an implausible vpath_length "
                       "(%s m); skipped", filename, vpath)
        return None
    if not _TAU_BOUNDS[0] < tau < _TAU_BOUNDS[1]:
        logger.warning("instrument file %s declares an implausible tau (%s s); "
                       "skipped", filename, tau)
        return None
    geometry = {'vpath_length': vpath}
    hpath = record.get('hpath_length')
    if hpath is not None:
        try:
            hpath = float(hpath)
        except (TypeError, ValueError):
            hpath = None
        if hpath and hpath > 0:
            geometry['hpath_length'] = hpath
    geometry['tau'] = tau
    aliases = record.get('aliases') or []
    if not isinstance(aliases, list):
        logger.warning("instrument file %s declares aliases that are not a "
                       "list; ignored", filename)
        aliases = []
    aliases = [a for a in (str(x).strip().lower() for x in aliases)
               if a and a != model]
    return model, aliases, geometry


def _value_hash(record):
    """SHA-256 over the stamped fields of a parsed file.

    Hashing the parsed content, not the bytes, keeps the stamp indifferent to
    formatting and line endings, which git rewrites between platforms.
    """
    subset = {k: record[k] for k in _STAMPED_FIELDS
              if isinstance(record, dict) and record.get(k) is not None}
    payload = json.dumps(subset, sort_keys=True, separators=(',', ':'))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def _read_manifest(directory):
    path = os.path.join(directory, _MANIFEST)
    try:
        with open(path, encoding='utf-8') as fh:
            manifest = json.load(fh)
    except OSError:
        return {}          # no stamp: no file claims built-in provenance
    except ValueError as exc:
        logger.warning("instrument manifest %s cannot be parsed (%s); treating "
                       "every file as unstamped", path, exc)
        return {}
    return manifest if isinstance(manifest, dict) else {}


def load_sensor_geometry(directory=None):
    """The ``model -> {vpath_length, hpath_length, tau}`` mapping.

    Aliases enter as extra keys onto the same geometry. Files are read in name
    order, and when two files claim the same name the first keeps it. A
    directory that cannot be read yields an empty table; the correction
    methods then skip or refuse exactly as they do for an unknown instrument,
    so missing data degrades loudly instead of being invented.
    """
    directory = _BUNDLED_DIR if directory is None else directory
    manifest = _read_manifest(directory)
    table = {}
    try:
        entries = sorted(os.listdir(directory))
    except OSError as exc:
        logger.error("instrument directory %s cannot be read (%s); the sensor "
                     "table is empty", directory, exc)
        return table
    present = [e for e in entries
               if e.endswith('.json') and not e.startswith('_')]
    for name in sorted(set(manifest) - set(present)):
        logger.warning("built-in instrument file %s is stamped but missing; "
                       "its models are absent from the table", name)
    for entry in present:
        try:
            with open(os.path.join(directory, entry), encoding='utf-8') as fh:
                record = json.load(fh)
        except (OSError, ValueError, RecursionError) as exc:
            logger.warning("instrument file %s cannot be parsed (%s); skipped",
                           entry, exc)
            continue
        validated = _validated(record, entry)
        if validated is None:
            continue
        model, aliases, geometry = validated
        expected = manifest.get(entry)
        if expected is None:
            logger.info("instrument file %s is not part of the built-in table; "
                        "using its values", entry)
        elif _value_hash(record) != expected:
            logger.warning("instrument file %s no longer carries the built-in "
                           "values; using the file's", entry)
        for name in (model, *aliases):
            if name in table:
                logger.warning("instrument name %r is claimed by more than one "
                               "file; the first keeps it", name)
                continue
            table[name] = geometry
    return table


_TABLE = None


def sensor_geometry():
    """The bundled table, read once on first use.

    Deferred past import so the provenance alerts land in whatever logging the
    application configured, not in the middle of ``import oneflux_preproc``.
    """
    global _TABLE
    if _TABLE is None:
        _TABLE = load_sensor_geometry()
    return _TABLE


def stamp_manifest(directory=None):
    """Rewrite the manifest from the files present, and return it.

    Maintainer step, mirroring how the setup build scripts stamp the config
    lock: never edit the manifest by hand, regenerate it::

        python -m oneflux_preproc.corrections.spectral.sensor_table
    """
    directory = _BUNDLED_DIR if directory is None else directory
    manifest = {}
    for entry in sorted(os.listdir(directory)):
        if not entry.endswith('.json') or entry.startswith('_'):
            continue
        try:
            with open(os.path.join(directory, entry), encoding='utf-8') as fh:
                record = json.load(fh)
        except (OSError, ValueError, RecursionError) as exc:
            logger.warning("instrument file %s cannot be parsed (%s); not "
                           "stamped", entry, exc)
            continue
        if _validated(record, entry) is None:
            continue       # a file the loader refuses cannot be a built-in
        manifest[entry] = _value_hash(record)
    path = os.path.join(directory, _MANIFEST)
    with open(path, 'w', encoding='utf-8', newline='\n') as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
        fh.write('\n')
    return manifest


if __name__ == '__main__':
    for _name, _digest in stamp_manifest().items():
        print(f'{_digest}  {_name}')
