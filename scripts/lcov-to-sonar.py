#!/usr/bin/env python3
"""Convert LCOV coverage report to SonarCloud Generic Test Data XML format.

Usage: python3 lcov-to-sonar.py lcov.info > sonar-coverage.xml

SonarCloud generic format spec:
https://docs.sonarsource.com/sonarcloud/enriching/test-coverage/generic-test-data/
"""
import sys


def convert(lcov_path: str) -> None:
    print('<coverage version="1">')
    current_file = None
    with open(lcov_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("SF:"):
                current_file = line[3:]
                print(f'  <file path="{current_file}">')
            elif line.startswith("DA:") and current_file:
                parts = line[3:].split(",")
                if len(parts) >= 2:
                    line_num = parts[0]
                    hits = int(parts[1])
                    covered = "true" if hits > 0 else "false"
                    print(
                        f'    <lineToCover lineNumber="{line_num}"'
                        f' covered="{covered}"/>'
                    )
            elif line == "end_of_record" and current_file:
                print("  </file>")
                current_file = None
    print("</coverage>")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <lcov-file>", file=sys.stderr)
        sys.exit(1)
    convert(sys.argv[1])
