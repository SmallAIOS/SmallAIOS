#!/usr/bin/env bash
# Check for cyclic dependencies in the SmallAIOS workspace
# Uses cargo metadata to extract the dependency graph and verify it is a DAG
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

cargo metadata --no-deps --format-version 1 --manifest-path "$ROOT_DIR/Cargo.toml" \
  | python3 -c "
import json, sys

meta = json.load(sys.stdin)
members = set()
for pkg in meta['packages']:
    members.add(pkg['name'])

adj = {}
for pkg in meta['packages']:
    name = pkg['name']
    adj[name] = []
    for dep in pkg['dependencies']:
        if dep['name'] in members and dep['name'] != name:
            adj[name].append(dep['name'])

WHITE, GRAY, BLACK = 0, 1, 2
color = {n: WHITE for n in adj}

def dfs(node, path):
    color[node] = GRAY
    path.append(node)
    for neighbor in adj[node]:
        if color[neighbor] == GRAY:
            idx = path.index(neighbor)
            cycle = path[idx:] + [neighbor]
            arrow = ' -> '
            print('CYCLE DETECTED: ' + arrow.join(cycle), file=sys.stderr)
            return True
        if color[neighbor] == WHITE:
            if dfs(neighbor, path):
                return True
    path.pop()
    color[node] = BLACK
    return False

found_cycle = False
for node in adj:
    if color[node] == WHITE:
        if dfs(node, []):
            found_cycle = True

if found_cycle:
    print('FAIL: workspace dependency graph contains cycles')
    sys.exit(1)
else:
    count = len(adj)
    print('OK: %d workspace crates form a DAG (no cycles)' % count)
    for name in sorted(adj):
        deps = adj[name]
        if deps:
            print('  %s -> %s' % (name, ', '.join(sorted(deps))))
        else:
            print('  %s (leaf)' % name)
"
