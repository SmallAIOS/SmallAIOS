#!/usr/bin/env python3
"""Unit tests for scripts/test_matrix.py (run: python3 -m unittest
discover -s scripts). Stdlib-only, no cargo invocation."""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_matrix  # noqa: E402


SAMPLE = '''
# comment
schema = 1

[[group]]
name = "default"
crates = ["crate-a", "crate-b"]
features = ["crate-a/feat1", "crate-b/feat2"]
min_tests = 100

[[group]]
name = "special"
crates = ["crate-c"]
cargo_args = ["--lib"]
test_targets = ["conformance"]
runner = "macos-latest"
ci = false
clippy = false

[[exclusion]]
crate = "crate-d"
reason = "bare-metal only"
'''


class ParserTests(unittest.TestCase):
    def parse(self, text=SAMPLE):
        return test_matrix.parse_toml_subset(text)

    def test_groups_and_exclusions(self):
        doc = self.parse()
        self.assertEqual(len(doc["group"]), 2)
        self.assertEqual(doc["group"][0]["name"], "default")
        self.assertEqual(doc["group"][0]["crates"], ["crate-a", "crate-b"])
        self.assertEqual(doc["group"][0]["min_tests"], 100)
        self.assertEqual(doc["group"][1]["runner"], "macos-latest")
        self.assertIs(doc["group"][1]["ci"], False)
        self.assertEqual(doc["exclusion"][0]["reason"], "bare-metal only")
        self.assertEqual(doc["schema"], 1)

    def test_empty_array(self):
        doc = self.parse('[[group]]\nname = "g"\ncrates = []\n')
        self.assertEqual(doc["group"][0]["crates"], [])

    def test_rejects_multiline_array(self):
        with self.assertRaises(ValueError):
            self.parse('[[group]]\ncrates = [\n"a",\n]\n')

    def test_rejects_unknown_value(self):
        with self.assertRaises(ValueError):
            self.parse("key = 1.5\n")


class CargoArgsTests(unittest.TestCase):
    def test_full_group(self):
        group = {
            "crates": ["a", "b"],
            "features": ["a/x", "b/y"],
            "cargo_args": ["--lib"],
            "test_targets": ["conf"],
        }
        self.assertEqual(
            test_matrix.cargo_test_args(group),
            ["-p", "a", "-p", "b", "--test", "conf", "--lib", "--features", "a/x,b/y"],
        )

    def test_minimal_group(self):
        self.assertEqual(test_matrix.cargo_test_args({"crates": ["a"]}), ["-p", "a"])


class ResultParsingTests(unittest.TestCase):
    def test_sums_passed_and_failed(self):
        out = (
            "test result: ok. 689 passed; 0 failed; 9 ignored; 0 measured; 0 filtered out; finished in 1.05s\n"
            "noise line\n"
            "test result: FAILED. 10 passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.1s\n"
        )
        executed, lines = test_matrix.parse_executed(out)
        self.assertEqual(executed, 701)
        self.assertEqual(lines, 2)

    def test_zero_and_ignored_only(self):
        out = "test result: ok. 0 passed; 0 failed; 9 ignored; 0 measured; 0 filtered out; finished in 0.00s\n"
        executed, lines = test_matrix.parse_executed(out)
        self.assertEqual(executed, 0)
        self.assertEqual(lines, 1)

    def test_no_result_lines(self):
        self.assertEqual(test_matrix.parse_executed("nothing here"), (0, 0))


class EmitTests(unittest.TestCase):
    GROUPS = {
        "default": {"crates": ["a"], "features": ["a/x"]},
        "special": {"crates": ["b"], "cargo_args": ["--lib"]},
        "mac": {"crates": ["c"], "runner": "macos-latest"},
        "offci": {"crates": ["d"], "ci": False},
    }

    def test_cov_args_excludes_macos_cargo_args_and_offci(self):
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            test_matrix.emit_cov_args(self.GROUPS)
        self.assertEqual(buf.getvalue().strip(), "-p a --features a/x")

    def test_gha_except(self):
        import io, json as j
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf):
            test_matrix.emit_gha(self.GROUPS, except_groups=["default"])
        include = j.loads(buf.getvalue())["include"]
        self.assertEqual([e["group"] for e in include], ["mac", "special"])

    def test_host_compatible_arch_gate(self):
        import platform
        machine = platform.machine().lower()
        other = "x86_64" if machine in ("arm64", "aarch64") else "aarch64"
        ok, reason = test_matrix.host_compatible({"host_arch": other})
        self.assertFalse(ok)
        self.assertIn(other, reason)
        ok, _ = test_matrix.host_compatible({})
        self.assertTrue(ok)


class VerifyTests(unittest.TestCase):
    def setUp(self):
        self._orig = test_matrix.workspace_members

    def tearDown(self):
        test_matrix.workspace_members = self._orig

    def _verify(self, members, groups, exclusions):
        test_matrix.workspace_members = lambda: sorted(members)
        return test_matrix.verify(groups, exclusions)

    def test_all_covered(self):
        rc = self._verify(
            ["a", "b"], {"g": {"crates": ["a", "b"]}}, {},
        )
        self.assertEqual(rc, 0)

    def test_unclassified_member_fails(self):
        rc = self._verify(["a", "b"], {"g": {"crates": ["a"]}}, {})
        self.assertEqual(rc, 1)

    def test_exclusion_with_reason_passes(self):
        rc = self._verify(
            ["a", "b"], {"g": {"crates": ["a"]}}, {"b": "bare-metal only"},
        )
        self.assertEqual(rc, 0)

    def test_empty_reason_fails(self):
        rc = self._verify(["a", "b"], {"g": {"crates": ["a"]}}, {"b": "  "})
        self.assertEqual(rc, 1)

    def test_covered_and_excluded_conflict_fails(self):
        rc = self._verify(
            ["a"], {"g": {"crates": ["a"]}}, {"a": "reason"},
        )
        self.assertEqual(rc, 1)

    def test_stale_exclusion_fails(self):
        rc = self._verify(["a"], {"g": {"crates": ["a"]}}, {"zzz": "gone"})
        self.assertEqual(rc, 1)

    def test_typo_group_crate_fails(self):
        rc = self._verify(["a"], {"g": {"crates": ["a", "typo-crate"]}}, {})
        self.assertEqual(rc, 1)


if __name__ == "__main__":
    unittest.main()
