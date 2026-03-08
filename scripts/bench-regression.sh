#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Benchmark regression detection script.
# Compares current criterion results against saved baselines.
# Exits with non-zero status if any benchmark regresses by >10%.
#
# Usage: ./scripts/bench-regression.sh [threshold_pct]
#   threshold_pct: regression threshold percentage (default: 10)

set -euo pipefail

THRESHOLD_PCT="${1:-10}"
CRITERION_DIR="target/criterion"
REGRESSED=0
TOTAL=0
REPORT=""

if [ ! -d "$CRITERION_DIR" ]; then
    echo "No criterion results found at $CRITERION_DIR"
    echo "Run 'cargo bench' first to generate baseline data."
    exit 0
fi

# Walk criterion output directories for estimates.json
while IFS= read -r estimates_file; do
    bench_path="${estimates_file%/new/estimates.json}"
    bench_name="${bench_path#$CRITERION_DIR/}"

    baseline_file="${bench_path}/base/estimates.json"
    if [ ! -f "$baseline_file" ]; then
        continue
    fi

    # Extract point estimates (nanoseconds)
    current_ns=$(python3 -c "
import json, sys
with open('$estimates_file') as f:
    d = json.load(f)
print(d.get('mean', d.get('median', {})).get('point_estimate', 0))
" 2>/dev/null || echo 0)

    baseline_ns=$(python3 -c "
import json, sys
with open('$baseline_file') as f:
    d = json.load(f)
print(d.get('mean', d.get('median', {})).get('point_estimate', 0))
" 2>/dev/null || echo 0)

    if [ "$baseline_ns" = "0" ] || [ "$current_ns" = "0" ]; then
        continue
    fi

    TOTAL=$((TOTAL + 1))

    # Calculate percent change
    pct_change=$(python3 -c "
b = $baseline_ns
c = $current_ns
if b > 0:
    change = ((c - b) / b) * 100
    print(f'{change:.1f}')
else:
    print('0.0')
")

    # Check regression
    is_regression=$(python3 -c "print('yes' if $pct_change > $THRESHOLD_PCT else 'no')")

    if [ "$is_regression" = "yes" ]; then
        REGRESSED=$((REGRESSED + 1))
        REPORT+="REGRESSION: $bench_name: ${pct_change}% slower (threshold: ${THRESHOLD_PCT}%)\n"
    else
        REPORT+="OK: $bench_name: ${pct_change}% change\n"
    fi
done < <(find "$CRITERION_DIR" -path "*/new/estimates.json" -type f 2>/dev/null)

echo "=== Benchmark Regression Report ==="
echo "Threshold: ${THRESHOLD_PCT}%"
echo "Benchmarks compared: $TOTAL"
echo "Regressions: $REGRESSED"
echo ""
echo -e "$REPORT"

if [ "$REGRESSED" -gt 0 ]; then
    echo "FAIL: $REGRESSED benchmark(s) regressed by more than ${THRESHOLD_PCT}%"
    exit 1
fi

echo "PASS: No significant regressions detected."
