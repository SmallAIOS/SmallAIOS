#!/usr/bin/env python3
"""Single source of truth for the SmallAIOS host test matrix.

Reads ci/test-matrix.toml and provides:

    --emit gha           JSON for the GitHub Actions strategy.matrix
    --emit clippy-args   cargo clippy package/feature args for the matrix union
    --run <group>        run a group's cargo tests, enforcing executed-test
                         counts (zero executed tests, or fewer than the
                         group's min_tests, fails even if cargo exits 0)
    --verify             every cargo workspace member must be covered by a
                         group or excluded with a reason
    --list               print groups and coverage summary

Spec: openspec/changes/ci-test-gates-v1/specs/ci-test-matrix/spec.md
Stdlib-only; works on Python >= 3.9 (uses tomllib when available, else a
strict TOML-subset parser matching the constraints documented in the
matrix file header).
"""

import argparse
import json
import os
import re
import subprocess
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_MATRIX = os.path.join(REPO_ROOT, "ci", "test-matrix.toml")

# `test result: ok. 689 passed; 0 failed; 9 ignored; 0 measured; ...`
RESULT_RE = re.compile(
    r"^test result: (?:ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored;"
)


# ---------------------------------------------------------------------------
# Matrix loading
# ---------------------------------------------------------------------------

def parse_toml_subset(text):
    """Strict parser for the constrained TOML subset used by test-matrix.toml.

    Supports: comments, `[[table]]` array-of-tables, and single-line
    `key = value` where value is a double-quoted string, integer, boolean,
    or a single-line array of double-quoted strings. Anything else is a
    hard error — the matrix file must stay parseable on Python 3.9.
    """
    doc = {}
    current = doc
    string_re = re.compile(r'^"((?:[^"\\]|\\.)*)"$')

    def parse_value(raw, lineno):
        raw = raw.strip()
        if raw in ("true", "false"):
            return raw == "true"
        if re.fullmatch(r"-?\d+", raw):
            return int(raw)
        m = string_re.match(raw)
        if m:
            return m.group(1).replace('\\"', '"').replace("\\\\", "\\")
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            items = []
            for part in re.findall(r'"(?:[^"\\]|\\.)*"', inner):
                items.append(parse_value(part, lineno))
            # sanity: comma-separated quoted strings only
            residue = re.sub(r'"(?:[^"\\]|\\.)*"', "", inner).replace(",", "").strip()
            if residue:
                raise ValueError(f"line {lineno}: unsupported array syntax: {raw!r}")
            return items
        raise ValueError(f"line {lineno}: unsupported TOML value: {raw!r}")

    for lineno, line in enumerate(text.splitlines(), 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[[") and stripped.endswith("]]"):
            name = stripped[2:-2].strip()
            current = {}
            doc.setdefault(name, []).append(current)
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip()
            current = {}
            doc[name] = current
            continue
        if "=" in stripped:
            key, _, raw = stripped.partition("=")
            # strip trailing comments outside strings: only safe when the
            # value is not a string containing '#'; keep it simple — split
            # on ' #' only if the raw value parses cleanly afterwards.
            try:
                current[key.strip()] = parse_value(raw, lineno)
            except ValueError:
                trimmed = raw.split(" #", 1)[0]
                current[key.strip()] = parse_value(trimmed, lineno)
            continue
        raise ValueError(f"line {lineno}: unparseable line: {stripped!r}")
    return doc


def load_matrix(path):
    with open(path, "rb") as fh:
        data = fh.read()
    try:
        import tomllib  # Python >= 3.11
        doc = tomllib.loads(data.decode("utf-8"))
    except ImportError:
        doc = parse_toml_subset(data.decode("utf-8"))
    groups = {g["name"]: g for g in doc.get("group", [])}
    if len(groups) != len(doc.get("group", [])):
        raise SystemExit("error: duplicate group names in matrix")
    exclusions = {e["crate"]: e.get("reason", "") for e in doc.get("exclusion", [])}
    return groups, exclusions


# ---------------------------------------------------------------------------
# Cargo invocation building
# ---------------------------------------------------------------------------

def cargo_test_args(group):
    args = []
    for crate in group["crates"]:
        args += ["-p", crate]
    for target in group.get("test_targets", []):
        args += ["--test", target]
    args += group.get("cargo_args", [])
    features = group.get("features", [])
    if features:
        args += ["--features", ",".join(features)]
    return args


# ---------------------------------------------------------------------------
# Result parsing / enforcement
# ---------------------------------------------------------------------------

def parse_executed(output):
    """Sum executed (passed + failed) and result-line count from cargo output."""
    executed = 0
    lines = 0
    for line in output.splitlines():
        m = RESULT_RE.match(line.strip())
        if m:
            executed += int(m.group(1)) + int(m.group(2))
            lines += 1
    return executed, lines


def run_group(groups, name, dry_run=False):
    if name not in groups:
        raise SystemExit(
            f"error: unknown group {name!r}; available: {', '.join(sorted(groups))}"
        )
    group = groups[name]
    cmd = ["cargo", "test"] + cargo_test_args(group)
    print(f"[test-matrix] group {name!r}: {' '.join(cmd)}", flush=True)
    if dry_run:
        return 0

    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    captured = []
    for line in proc.stdout:
        sys.stdout.write(line)
        captured.append(line)
    proc.wait()
    executed, result_lines = parse_executed("".join(captured))

    if proc.returncode != 0:
        print(f"[test-matrix] group {name!r}: FAIL (cargo exit {proc.returncode})")
        return proc.returncode
    if executed == 0:
        print(
            f"[test-matrix] group {name!r}: FAIL — zero tests executed across "
            f"{result_lines} test binaries. A feature/filter change has made "
            f"this gate vacuous (spec: ci-test-matrix / No Vacuous Test Gates)."
        )
        return 3
    floor = group.get("min_tests", 1)
    if executed < floor:
        print(
            f"[test-matrix] group {name!r}: FAIL — {executed} tests executed, "
            f"below the declared min_tests floor of {floor}. If tests were "
            f"legitimately removed, lower min_tests in ci/test-matrix.toml."
        )
        return 4
    print(f"[test-matrix] group {name!r}: OK — {executed} tests executed (floor {floor})")
    return 0


# ---------------------------------------------------------------------------
# Verification against cargo metadata
# ---------------------------------------------------------------------------

def workspace_members():
    out = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1"],
        capture_output=True, text=True, cwd=REPO_ROOT, check=True,
    )
    meta = json.loads(out.stdout)
    return sorted(p["name"] for p in meta["packages"])


def verify(groups, exclusions):
    covered = set()
    for group in groups.values():
        covered.update(group["crates"])
    members = workspace_members()
    problems = []
    for member in members:
        if member in covered and member in exclusions:
            problems.append(f"{member}: both covered by a group and excluded")
        elif member not in covered and member not in exclusions:
            problems.append(
                f"{member}: not covered by any matrix group and not excluded — "
                f"classify it in ci/test-matrix.toml"
            )
    for crate, reason in sorted(exclusions.items()):
        if crate not in members:
            problems.append(f"exclusion {crate!r}: not a workspace member (stale?)")
        elif not reason.strip():
            problems.append(f"exclusion {crate!r}: empty reason")
        else:
            print(f"[test-matrix] excluded: {crate} — {reason}")
    unknown = covered.difference(members)
    for crate in sorted(unknown):
        problems.append(f"group crate {crate!r}: not a workspace member (typo?)")
    if problems:
        print("[test-matrix] VERIFY FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(
        f"[test-matrix] verify OK: {len(members)} workspace members, "
        f"{len(covered)} covered by {len(groups)} groups, "
        f"{len(exclusions)} excluded with reasons"
    )
    return 0


# ---------------------------------------------------------------------------
# Emitters
# ---------------------------------------------------------------------------

def emit_gha(groups, except_groups=()):
    include = [
        {"group": name, "runner": g.get("runner", "ubuntu-latest")}
        for name, g in sorted(groups.items())
        if g.get("ci", True) and name not in except_groups
    ]
    print(json.dumps({"include": include}))


def emit_cov_args(groups):
    """Crate/feature args for the cargo-llvm-cov gate: the union of ci=true
    groups that run plain `cargo test` on linux (groups with cargo_args or
    test_targets restrict targets in ways llvm-cov does not mirror, and
    macos-runner groups don't execute on the coverage runner)."""
    crates = []
    features = []
    for _, group in sorted(groups.items()):
        if not group.get("ci", True):
            continue
        if group.get("runner", "ubuntu-latest").startswith("macos"):
            continue
        if group.get("cargo_args") or group.get("test_targets"):
            continue
        for crate in group["crates"]:
            if crate not in crates:
                crates.append(crate)
        for feat in group.get("features", []):
            if feat not in features:
                features.append(feat)
    args = []
    for crate in crates:
        args += ["-p", crate]
    if features:
        args += ["--features", ",".join(features)]
    print(" ".join(args))


def host_compatible(group):
    """(ok, reason) — can this group run on the current machine?"""
    import platform
    runner = group.get("runner", "ubuntu-latest")
    system = platform.system()
    if runner.startswith("macos") and system != "Darwin":
        return False, f"needs {runner}, host is {system}"
    if runner.startswith("ubuntu") and system == "Darwin":
        # linux groups generally run fine on macOS hosts except when the
        # group demands a specific CPU architecture
        pass
    want_arch = group.get("host_arch")
    if want_arch:
        machine = platform.machine().lower()
        aliases = {"x86_64": {"x86_64", "amd64"}, "aarch64": {"aarch64", "arm64"}}
        if machine not in aliases.get(want_arch, {want_arch}):
            return False, f"needs host_arch {want_arch}, host is {machine}"
    return True, ""


def run_all(groups):
    worst = 0
    for name, group in sorted(groups.items()):
        if not group.get("ci", True) and not group.get("test_targets"):
            continue
        ok, reason = host_compatible(group)
        if not ok:
            print(f"[test-matrix] group {name!r}: SKIPPED on this host ({reason})")
            continue
        rc = run_group(groups, name)
        worst = worst or rc
    return worst


def emit_clippy_args(groups):
    crates = []
    features = []
    for _, group in sorted(groups.items()):
        if not group.get("clippy", True):
            continue
        for crate in group["crates"]:
            if crate not in crates:
                crates.append(crate)
        for feat in group.get("features", []):
            if feat not in features:
                features.append(feat)
    args = []
    for crate in crates:
        args += ["-p", crate]
    if features:
        args += ["--features", ",".join(features)]
    print(" ".join(args))


def list_groups(groups, exclusions):
    for name, g in sorted(groups.items()):
        ci = "" if g.get("ci", True) else "  [not in ci.yml matrix]"
        print(
            f"{name}: {len(g['crates'])} crate(s), min_tests="
            f"{g.get('min_tests', 1)}, runner={g.get('runner', 'ubuntu-latest')}{ci}"
        )
    for crate, reason in sorted(exclusions.items()):
        print(f"excluded: {crate} — {reason}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", default=DEFAULT_MATRIX)
    parser.add_argument("--emit", choices=["gha", "clippy-args", "cov-args"])
    parser.add_argument("--except", dest="except_groups", action="append", default=[],
                        metavar="GROUP", help="omit GROUP from --emit gha output")
    parser.add_argument("--run", metavar="GROUP")
    parser.add_argument("--run-all", action="store_true",
                        help="run every host-compatible ci group (skips print loudly)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)

    groups, exclusions = load_matrix(args.matrix)
    if args.emit == "gha":
        emit_gha(groups, except_groups=args.except_groups)
        return 0
    if args.emit == "clippy-args":
        emit_clippy_args(groups)
        return 0
    if args.emit == "cov-args":
        emit_cov_args(groups)
        return 0
    if args.run_all:
        return run_all(groups)
    if args.run:
        return run_group(groups, args.run, dry_run=args.dry_run)
    if args.verify:
        return verify(groups, exclusions)
    if args.list:
        list_groups(groups, exclusions)
        return 0
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
