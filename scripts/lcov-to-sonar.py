#!/usr/bin/env python3
"""Convert LCOV coverage report to SonarCloud Generic Test Data XML format.

Usage: python3 lcov-to-sonar.py lcov.info > sonar-coverage.xml

Handles merged LCOV files where the same source file may appear multiple times
by deduplicating and taking the max hit count per line. Strips absolute paths
to produce paths relative to the repository root. Skips non-workspace files
(e.g. rustc standard library sources).

SonarCloud generic format spec:
https://docs.sonarsource.com/sonarqube-cloud/enriching/test-coverage/generic-test-data/
"""
import os
import sys
from collections import defaultdict
from pathlib import Path
from xml.sax.saxutils import escape

# Workspace crate directories — only include coverage for these
WORKSPACE_CRATES = {
    "kernel", "security", "onnx-rt", "ipc", "net", "posix",
    "container", "bus", "peripheral", "usb", "sdr", "arch",
    "bench",
}


def _make_relative(filepath: str) -> str | None:
    """Convert an SF: path to a repo-relative path.

    Returns None for paths outside the workspace (e.g. /rustc/.../library/...).
    """
    # Already relative — check it starts with a known crate dir
    if not os.path.isabs(filepath):
        first = Path(filepath).parts[0] if Path(filepath).parts else ""
        if first in WORKSPACE_CRATES:
            return filepath
        return None

    # Absolute — find known crate dir in path components
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part in WORKSPACE_CRATES:
            return str(Path(*parts[i:]))

    # Not a workspace file (e.g. /rustc/hash/library/core/src/...)
    return None


def convert(lcov_path: str) -> None:
    # Parse LCOV and merge duplicate files (max hits per line)
    files: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    current_file: str | None = None
    skipped = 0

    with open(lcov_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                rel = _make_relative(line[3:])
                if rel is None:
                    current_file = None
                    skipped += 1
                else:
                    current_file = rel
            elif line.startswith("DA:") and current_file:
                parts = line[3:].split(",")
                if len(parts) >= 2:
                    line_num = int(parts[0])
                    hits = int(parts[1])
                    files[current_file][line_num] = max(
                        files[current_file][line_num], hits
                    )
            elif line == "end_of_record":
                current_file = None

    if skipped:
        print(
            f"Skipped {skipped} non-workspace source file(s)",
            file=sys.stderr,
        )

    # Output XML
    print('<coverage version="1">')
    for filepath in sorted(files):
        print(f'  <file path="{escape(filepath)}">')
        for line_num in sorted(files[filepath]):
            hits = files[filepath][line_num]
            covered = "true" if hits > 0 else "false"
            print(
                f'    <lineToCover lineNumber="{line_num}"'
                f' covered="{covered}"/>'
            )
        print("  </file>")
    print("</coverage>")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <lcov-file>", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1])
