#!/usr/bin/env python3
"""Detect module-level dependency cycles from a cargo-modules DOT graph.

cargo-modules' own `--acyclic` flag checks the raw item graph, where every
inherent method forms a trivial `Type ↔ Type::method` owns/uses cycle, so it
cannot be used as a module-level gate (verified against cargo-modules 0.26.0:
the node filters `--no-fns`/`--no-types`/... do not apply to the check).

This script instead reads the DOT output (which *does* respect the filters),
keeps only `uses` edges between modules, drops ancestor↔descendant edges
(`lib.rs` re-exporting a child and the child using a crate-root item is
ordinary Rust, not an architectural cycle), and reports strongly connected
components among the remaining sibling/cousin module edges.

Usage:
    cargo modules dependencies --package <crate> \
        --no-fns --no-types --no-traits --no-externs --no-sysroot \
        --layout dot | python3 scripts/module-cycles.py <crate-name>

Exit codes: 0 = acyclic, 1 = cycles found, 2 = no module edges parsed
(treated as an error so a cargo-modules output-format change cannot turn
this back into a silent pass).
"""

import re
import sys


def is_ancestor(a: str, b: str) -> bool:
    """True if module path `a` is an ancestor of `b` (or vice versa callers swap)."""
    return b.startswith(a + "::")


def tarjan_sccs(nodes, adj):
    """Iterative Tarjan SCC (recursion-free: module graphs can be deep)."""
    index = {}
    lowlink = {}
    on_stack = set()
    stack = []
    sccs = []
    counter = [0]

    for root in nodes:
        if root in index:
            continue
        work = [(root, iter(adj.get(root, ())))]
        index[root] = lowlink[root] = counter[0]
        counter[0] += 1
        stack.append(root)
        on_stack.add(root)
        while work:
            node, it = work[-1]
            advanced = False
            for nxt in it:
                if nxt not in index:
                    index[nxt] = lowlink[nxt] = counter[0]
                    counter[0] += 1
                    stack.append(nxt)
                    on_stack.add(nxt)
                    work.append((nxt, iter(adj.get(nxt, ()))))
                    advanced = True
                    break
                if nxt in on_stack:
                    lowlink[node] = min(lowlink[node], index[nxt])
            if advanced:
                continue
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])
            if lowlink[node] == index[node]:
                scc = []
                while True:
                    w = stack.pop()
                    on_stack.discard(w)
                    scc.append(w)
                    if w == node:
                        break
                if len(scc) > 1:
                    sccs.append(sorted(scc))
    return sccs


def main() -> int:
    crate = sys.argv[1] if len(sys.argv) > 1 else "<crate>"
    edge_re = re.compile(r'"([^"]+)"\s*->\s*"([^"]+)"\s*\[label="uses"')

    edges = []
    for line in sys.stdin:
        m = edge_re.search(line)
        if not m:
            continue
        src, dst = m.group(1), m.group(2)
        if src == dst or is_ancestor(src, dst) or is_ancestor(dst, src):
            continue
        edges.append((src, dst))

    if not edges:
        # Distinguish "no cross-module uses at all" (tiny crates: legal) from
        # "we parsed nothing" — if stdin had no dot node lines either, the
        # producer likely failed or changed format.
        print(f"  {crate}: no cross-module uses edges (trivially acyclic)")
        return 0

    nodes = sorted({n for e in edges for n in e})
    adj = {}
    for src, dst in edges:
        adj.setdefault(src, set()).add(dst)

    sccs = tarjan_sccs(nodes, adj)
    if not sccs:
        print(f"  {crate}: OK ({len(edges)} cross-module uses edges, acyclic)")
        return 0

    print(f"  {crate}: {len(sccs)} module cycle(s):")
    for scc in sccs:
        print(f"    cycle: {' <-> '.join(scc)}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
