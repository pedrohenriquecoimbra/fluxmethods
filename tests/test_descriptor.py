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

#: The package directory, which a translation's callable is written relative to.
PACKAGE = DESCRIPTOR.parent / "fluxmethods"

#: How a function is defined in each language a row may say, as a pattern over
#: the file: enough to catch a rename, which is what this file is for.
DEFINED = {"r": r"^{name}\s*<-\s*function\b",
           "julia": r"^function\s+{name}\s*\(|^{name}\s*\(.*\)\s*="}


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
            if run.get("language"):
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


class TestEveryTranslationStillExists(unittest.TestCase):
    """The same rot, for a row whose function is not Python.

    Such a row says its ``language``, and its callable is
    ``<file under the package>:<function>``. Nothing here runs R or Julia --
    ``test_translations.py`` does, where they are installed -- so what is
    checked is what can be read: the file is there, beside the Python module of
    the same name, the function is defined in it, and the block declares the
    language.
    """

    def _translated(self):
        return [(step, row) for step, row in _rows()
                if (row.get("run") or {}).get("language")]

    def test_there_are_translations_and_each_language_is_declared(self):
        rows = self._translated()
        self.assertGreater(len(rows), 0)
        library = _document()["library"]
        for step, row in rows:
            language = row["run"]["language"]
            with self.subTest(row=f"{step}/{row['name']}"):
                self.assertIn(language, DEFINED)
                self.assertTrue(library.get(language, {}).get("engine"),
                                f"the library block declares no {language} "
                                f"provider")

    def test_each_translation_is_a_function_of_a_file_beside_the_python(self):
        for step, row in self._translated():
            run = row["run"]
            with self.subTest(row=f"{step}/{row['name']}"):
                path, sep, function = run["callable"].rpartition(":")
                self.assertTrue(sep, "a callable is 'file:function'")
                source = PACKAGE / path
                self.assertTrue(source.is_file(), f"{source} is not there")
                self.assertTrue(source.with_suffix(".py").is_file(),
                                "a translation sits beside the Python it "
                                "translates, under the same name")
                pattern = DEFINED[run["language"]].format(
                    name=re.escape(function))
                self.assertRegex(source.read_text(encoding="utf-8"),
                                 re.compile(pattern, re.MULTILINE))

    def test_a_translation_twins_a_method_the_python_holds(self):
        python = {(step, row["ours"]) for step, row in _rows()
                  if row.get("run") and not row["run"].get("language")}
        for step, row in self._translated():
            with self.subTest(row=f"{step}/{row['name']}"):
                self.assertIn((step, row["ours"]), python)


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
