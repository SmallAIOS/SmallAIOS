#!/usr/bin/env python3
"""Convert LCOV coverage report to SonarCloud Generic Test Data XML format.

Usage: python3 lcov-to-sonar.py lcov.info > sonar-coverage.xml

Handles merged LCOV files where the same source file may appear multiple times
by deduplicating and taking the max hit count per line. Filters out line numbers
that exceed the actual file length (LLVM coverage can produce these from macro
expansion and generic instantiation).

SonarCloud generic format spec:
https://docs.sonarsource.com/sonarqube-cloud/enriching/test-coverage/generic-test-data/
"""
import sys
from collections import defaultdict


def _count_lines(filepath):
    """Return the number of lines in a file, or None if not readable."""
    try:
        with open(filepath) as f:
            return sum(1 for _ in f)
    except OSError:
        return None


def convert(lcov_path):
    # Parse LCOV and merge duplicate files (max hits per line)
    files = defaultdict(lambda: defaultdict(int))
    current_file = None

    with open(lcov_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                current_file = line[3:]
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

    # Output XML, filtering out-of-range lines
    skipped_lines = 0
    print('<coverage version="1">')
    for filepath in sorted(files):
        max_line = _count_lines(filepath)
        valid_lines = {}
        for line_num, hits in sorted(files[filepath].items()):
            if max_line is not None and line_num > max_line:
                skipped_lines += 1
                continue
            valid_lines[line_num] = hits

        if not valid_lines:
            continue

        print(f'  <file path="{filepath}">')
        for line_num in sorted(valid_lines):
            hits = valid_lines[line_num]
            covered = "true" if hits > 0 else "false"
            print(
                f'    <lineToCover lineNumber="{line_num}"'
                f' covered="{covered}"/>'
            )
        print("  </file>")
    print("</coverage>")

    if skipped_lines:
        print(
            f"Filtered {skipped_lines} out-of-range line(s) "
            f"(LLVM macro/generic expansion artifacts)",
            file=sys.stderr,
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <lcov-file>", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1])
