"""The descriptor's claims about *this* package, checked against it.

`descriptor.json` names a callable for every row that says how to call a method.
Those names are the one part of the protocol this repository owns: rename a
function here and the row rots, and nothing in this tree noticed until the
consumer's contracts failed a repository away.

**This does not prove a binding works.** Resolving a name is import-level, and
the marshalling — what the roles hand over, what a `passthrough` gives back —
can only be exercised by a bridge, which lives in `oneflux_preproc` and not
here. A row can resolve cleanly and still be wrong there; that has happened, in
both directions. What these tests pin is narrower and entirely local: the
descriptor does not promise a function this package no longer has.
"""

import importlib
import json
import pathlib
import re
import unittest

DESCRIPTOR = pathlib.Path(__file__).resolve().parent.parent / "descriptor.json"

#: A role spec is `role` or `role:reference` -- the vocabulary itself belongs to
#: the bridge and is checked there, so this is the *form* only, which is what
#: catches a typo like "dataset :" that would read as an unknown role.
ROLE_SPEC = re.compile(r"^[a-z_]+:?[A-Za-z0-9_$.]*$")

#: The consumer's contract on a `run.description`. Pinned here because shipping
#: a violation costs a tag and a re-pin to find out.
DESCRIPTION_LIMIT = 160


def _document():
    return json.loads(DESCRIPTOR.read_text(encoding="utf-8"))


def _rows():
    """Every row, as ``(step, row)`` -- the key a row is uniquely named by."""
    return [(step, row)
            for step, rows in _document()["methods"].items()
            for row in rows]


class TestDescriptorIsWellFormed(unittest.TestCase):
    def test_it_is_valid_json_and_has_rows(self):
        self.assertGreater(len(_rows()), 0)

    def test_a_row_is_named_by_its_step_and_name_together(self):
        """`ibrom_et_al_2007` is a row in two different steps.

        Keyed by name alone the two are one key, which is how a note true of the
        spectral row was once written onto the conversion one as well. Anything
        mapping over these rows keys on the pair.
        """
        names = [row["name"] for _, row in _rows()]
        pairs = [(step, row["name"]) for step, row in _rows()]
        self.assertEqual(len(pairs), len(set(pairs)), "a (step, name) repeats")
        self.assertLess(len(set(names)), len(set(pairs)),
                        "expected at least one name shared across two steps")


class TestEveryDeclaredCallableStillExists(unittest.TestCase):
    """The rot this file exists to catch."""

    def test_each_run_callable_resolves_to_something_callable(self):
        for step, row in _rows():
            run = row.get("run")
            if not run:
                continue
            with self.subTest(row=f"{step}/{row['name']}"):
                module, sep, function = run["callable"].partition(":")
                self.assertTrue(sep, "a callable is 'module:function'")
                try:
                    found = getattr(importlib.import_module(module), function)
                except ImportError as exc:
                    # `fluxmethods.units` rests on pint, an optional extra. A
                    # tree without it cannot answer the question, and failing
                    # here would report a missing extra as a rotten descriptor.
                    if "pint" not in str(exc):
                        raise
                    self.skipTest(f"{module} needs the units extra: {exc}")
                self.assertTrue(callable(found), f"{run['callable']} is not callable")


class TestRunSpecsAreShaped(unittest.TestCase):
    def test_each_run_declares_a_shape_args_and_outputs(self):
        for step, row in _rows():
            run = row.get("run")
            if not run:
                continue
            with self.subTest(row=f"{step}/{row['name']}"):
                self.assertTrue(run.get("shape"))
                self.assertTrue(run.get("args"))
                self.assertTrue(run.get("outputs"))

    def test_each_argument_is_a_well_formed_role_spec(self):
        for step, row in _rows():
            run = row.get("run")
            if not run:
                continue
            for spec in run["args"]:
                with self.subTest(row=f"{step}/{row['name']}", spec=spec):
                    self.assertRegex(spec, ROLE_SPEC)

    def test_a_description_stays_within_the_consumers_limit(self):
        for step, row in _rows():
            run = row.get("run")
            if not run:
                continue
            with self.subTest(row=f"{step}/{row['name']}"):
                self.assertLessEqual(len(run.get("description", "")),
                                     DESCRIPTION_LIMIT)


if __name__ == "__main__":
    unittest.main()
