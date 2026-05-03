#!/usr/bin/env python3
"""Generate a Design Structure Matrix (DSM) from cargo workspace metadata.

Outputs:
  build/analysis/dsm-matrix.json  — adjacency matrix with metadata
  build/analysis/dsm-matrix.csv   — CSV with crate names as headers

Dependency kind encoding:
  1 = normal dependency
  2 = dev dependency
  3 = build dependency

SPDX-License-Identifier: Apache-2.0
"""
import json
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT_DIR / "build" / "analysis"


def main():
    result = subprocess.run(
        ["cargo", "metadata", "--no-deps", "--format-version", "1",
         "--manifest-path", str(ROOT_DIR / "Cargo.toml")],
        capture_output=True, text=True, check=True,
    )
    meta = json.loads(result.stdout)

    members = set()
    for pkg in meta["packages"]:
        members.add(pkg["name"])

    crates = sorted(members)
    name_to_idx = {name: i for i, name in enumerate(crates)}
    n = len(crates)

    # Build adjacency matrix: matrix[row][col] = kind if row depends on col
    matrix = [[0] * n for _ in range(n)]

    for pkg in meta["packages"]:
        row = name_to_idx[pkg["name"]]
        for dep in pkg["dependencies"]:
            if dep["name"] in members and dep["name"] != pkg["name"]:
                col = name_to_idx[dep["name"]]
                kind = dep.get("kind")
                if kind is None:
                    val = 1  # normal
                elif kind == "dev":
                    val = 2
                elif kind == "build":
                    val = 3
                else:
                    val = 1
                # Keep highest priority (normal > dev > build)
                if matrix[row][col] == 0 or val < matrix[row][col]:
                    matrix[row][col] = val

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON output
    dsm_json = {
        "crates": crates,
        "matrix": matrix,
        "legend": {"1": "normal", "2": "dev", "3": "build", "0": "none"},
    }
    json_path = OUT_DIR / "dsm-matrix.json"
    with open(json_path, "w") as f:
        json.dump(dsm_json, f, indent=2)

    # CSV output
    csv_path = OUT_DIR / "dsm-matrix.csv"
    with open(csv_path, "w") as f:
        f.write("," + ",".join(crates) + "\n")
        for i, name in enumerate(crates):
            f.write(name + "," + ",".join(str(v) for v in matrix[i]) + "\n")

    print(f"DSM matrix: {n} crates")
    print(f"  JSON: {json_path}")
    print(f"  CSV:  {csv_path}")

    # Print summary of dependencies
    total_deps = sum(1 for row in matrix for v in row if v > 0)
    normal = sum(1 for row in matrix for v in row if v == 1)
    dev = sum(1 for row in matrix for v in row if v == 2)
    build = sum(1 for row in matrix for v in row if v == 3)
    print(f"  Total edges: {total_deps} (normal={normal}, dev={dev}, build={build})")


if __name__ == "__main__":
    main()
