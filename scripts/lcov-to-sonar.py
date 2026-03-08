#!/usr/bin/env python3
"""Convert LCOV coverage report to SonarCloud Generic Test Data XML format.

Usage: python3 lcov-to-sonar.py lcov.info > sonar-coverage.xml

Handles merged LCOV files where the same source file may appear multiple times
by deduplicating and taking the max hit count per line. Strips absolute paths
to produce paths relative to the repository root.

SonarCloud generic format spec:
https://docs.sonarsource.com/sonarcloud/enriching/test-coverage/generic-test-data/
"""
import os
import sys
from collections import defaultdict
from pathlib import Path


def _make_relative(filepath: str, lcov_path: str) -> str:
    """Convert an absolute SF: path to a repo-relative path.

    cargo llvm-cov emits absolute paths like
    /home/runner/work/SmallAIOS/SmallAIOS/kernel/src/lib.rs
    SonarCloud expects paths relative to the project root.
    """
    if not os.path.isabs(filepath):
        return filepath

    # Heuristic: find the repo root by looking for known top-level crate dirs
    # in the path components.
    known_roots = {
        "kernel", "security", "onnx-rt", "ipc", "net", "posix",
        "container", "bus", "peripheral", "usb", "sdr", "arch",
    }
    parts = Path(filepath).parts
    for i, part in enumerate(parts):
        if part in known_roots:
            return str(Path(*parts[i:]))

    # Fallback: try stripping the lcov file's directory as a common prefix
    lcov_dir = os.path.dirname(os.path.abspath(lcov_path))
    try:
        return os.path.relpath(filepath, lcov_dir)
    except ValueError:
        return filepath


def convert(lcov_path: str) -> None:
    # Parse LCOV and merge duplicate files (max hits per line)
    files = defaultdict(lambda: defaultdict(int))
    current_file = None

    with open(lcov_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                current_file = _make_relative(line[3:], lcov_path)
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

    # Output XML
    print('<?xml version="1.0" encoding="UTF-8"?>')
    print('<coverage version="1">')
    for filepath in sorted(files):
        print(f'  <file path="{filepath}">')
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
