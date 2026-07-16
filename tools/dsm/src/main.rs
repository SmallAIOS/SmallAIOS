use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};
use std::env;
use std::fs;
use std::process;

// ---------------------------------------------------------------------------
// 2.2: DSM matrix parser
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct DsmJson {
    crates: Vec<String>,
    matrix: Vec<Vec<u8>>,
    #[allow(dead_code)]
    legend: Option<HashMap<String, String>>,
}

#[derive(Debug)]
struct DsmMatrix {
    crates: Vec<String>,
    matrix: Vec<Vec<u8>>,
}

impl DsmMatrix {
    fn from_json(path: &str) -> Self {
        let data = fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Error reading {path}: {e}");
            process::exit(1);
        });
        let parsed: DsmJson = serde_json::from_str(&data).unwrap_or_else(|e| {
            eprintln!("Error parsing JSON: {e}");
            process::exit(1);
        });
        let n = parsed.crates.len();
        assert_eq!(parsed.matrix.len(), n, "Matrix rows != crate count");
        for (i, row) in parsed.matrix.iter().enumerate() {
            assert_eq!(row.len(), n, "Row {i} length != crate count");
        }
        DsmMatrix {
            crates: parsed.crates,
            matrix: parsed.matrix,
        }
    }

    fn len(&self) -> usize {
        self.crates.len()
    }

    /// Returns true if row depends on col with normal dependency (kind=1).
    fn has_normal_dep(&self, row: usize, col: usize) -> bool {
        self.matrix[row][col] == 1
    }

    /// Returns true if row depends on col with any non-zero dependency.
    fn has_any_dep(&self, row: usize, col: usize) -> bool {
        self.matrix[row][col] != 0
    }

    /// Build adjacency list for normal deps: if row depends on col, then
    /// a change in col affects row. So edge: col -> row (dependents direction).
    fn dependents_graph_normal(&self) -> Vec<Vec<usize>> {
        let n = self.len();
        let mut adj = vec![vec![]; n];
        for row in 0..n {
            for col in 0..n {
                if self.has_normal_dep(row, col) {
                    // col is depended upon by row, so change in col propagates to row
                    adj[col].push(row);
                }
            }
        }
        adj
    }

    /// Build adjacency list where edge means "row depends on col" (dependency direction).
    fn dependency_graph_normal(&self) -> Vec<Vec<usize>> {
        let n = self.len();
        let mut adj = vec![vec![]; n];
        for row in 0..n {
            for col in 0..n {
                if self.has_normal_dep(row, col) {
                    adj[row].push(col);
                }
            }
        }
        adj
    }

    /// Build adjacency list for all deps (dependency direction).
    fn dependency_graph_all(&self) -> Vec<Vec<usize>> {
        let n = self.len();
        let mut adj = vec![vec![]; n];
        for row in 0..n {
            for col in 0..n {
                if self.has_any_dep(row, col) {
                    adj[row].push(col);
                }
            }
        }
        adj
    }
}

// ---------------------------------------------------------------------------
// 2.3: Transitive closure / propagation cost
// ---------------------------------------------------------------------------

/// For each crate, compute the percentage of crates transitively affected by
/// a change to it. Uses BFS on normal-dep dependents graph.
fn propagation_cost(dsm: &DsmMatrix) -> Vec<f64> {
    let adj = dsm.dependents_graph_normal();
    let n = dsm.len();
    let mut costs = vec![0.0; n];
    for start in 0..n {
        let reachable = bfs_reachable(&adj, start);
        // reachable includes self; propagation cost = reachable count / total
        costs[start] = reachable.len() as f64 / n as f64;
    }
    costs
}

fn bfs_reachable(adj: &[Vec<usize>], start: usize) -> HashSet<usize> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    queue.push_back(start);
    while let Some(node) = queue.pop_front() {
        for &next in &adj[node] {
            if visited.insert(next) {
                queue.push_back(next);
            }
        }
    }
    visited
}

// ---------------------------------------------------------------------------
// 2.4: Fan-in / fan-out
// ---------------------------------------------------------------------------

fn fan_in(dsm: &DsmMatrix) -> Vec<usize> {
    let n = dsm.len();
    let mut result = vec![0usize; n];
    for col in 0..n {
        for row in 0..n {
            if dsm.has_normal_dep(row, col) {
                result[col] += 1;
            }
        }
    }
    result
}

fn fan_out(dsm: &DsmMatrix) -> Vec<usize> {
    let n = dsm.len();
    let mut result = vec![0usize; n];
    for row in 0..n {
        for col in 0..n {
            if dsm.has_normal_dep(row, col) {
                result[row] += 1;
            }
        }
    }
    result
}

// ---------------------------------------------------------------------------
// 2.5: Coupling cluster detection (Tarjan's SCC)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
struct Cluster {
    crates: Vec<String>,
    dev_only: bool,
}

struct TarjanState {
    index_counter: usize,
    stack: Vec<usize>,
    on_stack: Vec<bool>,
    index: Vec<Option<usize>>,
    lowlink: Vec<usize>,
    sccs: Vec<Vec<usize>>,
}

fn tarjan_scc(adj: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = adj.len();
    let mut state = TarjanState {
        index_counter: 0,
        stack: Vec::new(),
        on_stack: vec![false; n],
        index: vec![None; n],
        lowlink: vec![0; n],
        sccs: Vec::new(),
    };
    for v in 0..n {
        if state.index[v].is_none() {
            tarjan_strongconnect(adj, v, &mut state);
        }
    }
    state.sccs
}

fn tarjan_strongconnect(adj: &[Vec<usize>], v: usize, state: &mut TarjanState) {
    state.index[v] = Some(state.index_counter);
    state.lowlink[v] = state.index_counter;
    state.index_counter += 1;
    state.stack.push(v);
    state.on_stack[v] = true;

    for &w in &adj[v] {
        if state.index[w].is_none() {
            tarjan_strongconnect(adj, w, state);
            state.lowlink[v] = state.lowlink[v].min(state.lowlink[w]);
        } else if state.on_stack[w] {
            state.lowlink[v] = state.lowlink[v].min(state.index[w].unwrap());
        }
    }

    if state.lowlink[v] == state.index[v].unwrap() {
        let mut scc = Vec::new();
        loop {
            let w = state.stack.pop().unwrap();
            state.on_stack[w] = false;
            scc.push(w);
            if w == v {
                break;
            }
        }
        scc.sort();
        state.sccs.push(scc);
    }
}

fn find_clusters(dsm: &DsmMatrix) -> Vec<Cluster> {
    let all_adj = dsm.dependency_graph_all();
    let normal_adj = dsm.dependency_graph_normal();

    let all_sccs = tarjan_scc(&all_adj);
    let normal_sccs = tarjan_scc(&normal_adj);

    // Collect normal SCCs with >1 node
    let normal_cycle_sets: HashSet<Vec<usize>> = normal_sccs
        .iter()
        .filter(|scc| scc.len() > 1)
        .cloned()
        .collect();

    let mut clusters = Vec::new();

    for scc in &all_sccs {
        if scc.len() <= 1 {
            continue;
        }
        let names: Vec<String> = scc.iter().map(|&i| dsm.crates[i].clone()).collect();
        // It's dev-only if the same SCC doesn't appear in normal-only graph
        let dev_only = !normal_cycle_sets.contains(scc);
        clusters.push(Cluster {
            crates: names,
            dev_only,
        });
    }

    clusters
}

// ---------------------------------------------------------------------------
// 2.6: Layering violation detection
// ---------------------------------------------------------------------------

fn layer(crate_name: &str) -> u8 {
    match crate_name {
        "smallaios-kernel" | "smallaios-security" => 0,
        "smallaios-net" | "smallaios-ipc" | "smallaios-posix" | "smallaios-onnx-rt"
        | "smallaios-usb" => 1,
        "smallaios-container" | "smallaios-bench" => 3,
        _ => 2, // arch-*, peripheral, bus, sdr
    }
}

#[derive(Debug, Clone, Serialize)]
struct LayeringViolation {
    from: String,
    to: String,
    from_layer: u8,
    to_layer: u8,
}

fn find_layering_violations(dsm: &DsmMatrix) -> Vec<LayeringViolation> {
    let n = dsm.len();
    let mut violations = Vec::new();

    for row in 0..n {
        for col in 0..n {
            if !dsm.has_normal_dep(row, col) {
                continue;
            }
            let from_name = &dsm.crates[row];
            let to_name = &dsm.crates[col];
            let from_layer = layer(from_name);
            let to_layer = layer(to_name);

            // Violation: depending on a HIGHER layer number (lower layer depending upward)
            if from_layer < to_layer {
                violations.push(LayeringViolation {
                    from: from_name.clone(),
                    to: to_name.clone(),
                    from_layer,
                    to_layer,
                });
            }

            // Same-layer deps that aren't kernel<->security
            if from_layer == to_layer {
                let is_kernel_security = {
                    let pair = (from_name.as_str(), to_name.as_str());
                    pair == ("smallaios-kernel", "smallaios-security")
                        || pair == ("smallaios-security", "smallaios-kernel")
                };
                if !is_kernel_security {
                    violations.push(LayeringViolation {
                        from: from_name.clone(),
                        to: to_name.clone(),
                        from_layer,
                        to_layer,
                    });
                }
            }
        }
    }

    violations
}

// ---------------------------------------------------------------------------
// 2.7: JSON output
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct MetricsOutput {
    propagation_cost: BTreeMap<String, f64>,
    fan_in: BTreeMap<String, usize>,
    fan_out: BTreeMap<String, usize>,
    clusters: Vec<Cluster>,
    layering_violations: Vec<LayeringViolation>,
    summary: Summary,
}

#[derive(Debug, Serialize)]
struct Summary {
    total_crates: usize,
    total_edges: usize,
    max_propagation_cost: f64,
    acyclic_production: bool,
}

fn build_output(dsm: &DsmMatrix) -> MetricsOutput {
    let costs = propagation_cost(dsm);
    let fi = fan_in(dsm);
    let fo = fan_out(dsm);
    let clusters = find_clusters(dsm);
    let violations = find_layering_violations(dsm);

    let n = dsm.len();
    let mut prop_map = BTreeMap::new();
    let mut fi_map = BTreeMap::new();
    let mut fo_map = BTreeMap::new();

    for i in 0..n {
        let name = &dsm.crates[i];
        prop_map.insert(name.clone(), (costs[i] * 100.0).round() / 100.0);
        fi_map.insert(name.clone(), fi[i]);
        fo_map.insert(name.clone(), fo[i]);
    }

    let total_edges = dsm
        .matrix
        .iter()
        .flat_map(|row| row.iter())
        .filter(|&&v| v == 1)
        .count();

    let max_cost = costs.iter().cloned().fold(0.0f64, f64::max);
    let max_cost_rounded = (max_cost * 100.0).round() / 100.0;

    let acyclic = clusters.iter().all(|c| c.dev_only);

    MetricsOutput {
        propagation_cost: prop_map,
        fan_in: fi_map,
        fan_out: fo_map,
        clusters,
        layering_violations: violations,
        summary: Summary {
            total_crates: n,
            total_edges,
            max_propagation_cost: max_cost_rounded,
            acyclic_production: acyclic,
        },
    }
}

// ---------------------------------------------------------------------------
// 2.8: Human-readable stdout summary
// ---------------------------------------------------------------------------

fn print_summary(dsm: &DsmMatrix, output: &MetricsOutput) {
    let n = dsm.len();
    let costs = propagation_cost(dsm);
    let fi = fan_in(dsm);
    let fo = fan_out(dsm);

    println!("DSM Architecture Metrics");
    println!("{}", "=".repeat(72));
    println!(
        "{:<30} {:>5} {:>6} {:>7} {:>10}",
        "Crate", "Layer", "Fan-in", "Fan-out", "Prop.Cost"
    );
    println!("{}", "-".repeat(72));

    for i in 0..n {
        let name = &dsm.crates[i];
        let l = layer(name);
        println!(
            "{:<30} {:>5} {:>6} {:>7} {:>9.0}%",
            name,
            l,
            fi[i],
            fo[i],
            costs[i] * 100.0
        );
    }

    println!();
    println!("Summary");
    println!("{}", "-".repeat(72));
    println!("  Total crates:          {}", output.summary.total_crates);
    println!("  Total normal edges:    {}", output.summary.total_edges);
    println!(
        "  Max propagation cost:  {:.0}%",
        output.summary.max_propagation_cost * 100.0
    );
    println!(
        "  Acyclic (production):  {}",
        output.summary.acyclic_production
    );

    if !output.clusters.is_empty() {
        println!();
        println!("Coupling Clusters (SCCs with >1 node)");
        println!("{}", "-".repeat(72));
        for (i, cluster) in output.clusters.iter().enumerate() {
            let kind = if cluster.dev_only {
                "dev-only"
            } else {
                "PRODUCTION"
            };
            println!(
                "  Cluster {}: [{}] ({})",
                i + 1,
                cluster.crates.join(", "),
                kind
            );
        }
    }

    if !output.layering_violations.is_empty() {
        println!();
        println!("Layering Violations");
        println!("{}", "-".repeat(72));
        for v in &output.layering_violations {
            let kind = if v.from_layer == v.to_layer {
                "same-layer"
            } else {
                "upward"
            };
            println!(
                "  {} (L{}) -> {} (L{})  [{}]",
                v.from, v.from_layer, v.to, v.to_layer, kind
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: dsm-analysis <dsm.json> [output.json]");
        process::exit(1);
    }

    let input_path = &args[1];
    let output_path = if args.len() > 2 {
        args[2].clone()
    } else {
        "dsm-metrics.json".to_string()
    };

    // Positional arguments only: a flag-like value here means a caller bug
    // (e.g. `-- input.json --output metrics.json` silently writing a file
    // literally named `--output`).
    for arg in [input_path.as_str(), output_path.as_str()] {
        if arg.starts_with("--") {
            eprintln!("Error: unexpected flag-like argument {arg:?}");
            eprintln!("Usage: dsm-analysis <dsm.json> [output.json]");
            process::exit(2);
        }
    }

    let dsm = DsmMatrix::from_json(input_path);
    let output = build_output(&dsm);

    print_summary(&dsm, &output);

    let json = serde_json::to_string_pretty(&output).expect("Failed to serialize output");
    fs::write(&output_path, &json).unwrap_or_else(|e| {
        eprintln!("Error writing {output_path}: {e}");
        process::exit(1);
    });
    println!();
    println!("Metrics written to {output_path}");
}

// ---------------------------------------------------------------------------
// 2.9: Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper to build a DsmMatrix from crate names and an adjacency matrix.
    fn make_dsm(names: &[&str], matrix: Vec<Vec<u8>>) -> DsmMatrix {
        DsmMatrix {
            crates: names.iter().map(|s| s.to_string()).collect(),
            matrix,
        }
    }

    // --- 2.9a: Simple DAG with known fan-in/out ---
    //
    // Graph (normal deps):  A -> B -> C
    // matrix[row][col] means row depends on col.
    //   A depends on B (row=0, col=1)
    //   B depends on C (row=1, col=2)
    #[test]
    fn test_dag_fan_in_out() {
        let dsm = make_dsm(
            &["A", "B", "C"],
            vec![
                vec![0, 1, 0], // A depends on B
                vec![0, 0, 1], // B depends on C
                vec![0, 0, 0], // C depends on nothing
            ],
        );

        let fi = fan_in(&dsm);
        let fo = fan_out(&dsm);

        // Fan-in: who depends on each? B has A depending on it, C has B depending on it.
        assert_eq!(fi, vec![0, 1, 1]);
        // Fan-out: A depends on 1, B depends on 1, C depends on 0.
        assert_eq!(fo, vec![1, 1, 0]);
    }

    #[test]
    fn test_dag_propagation_cost() {
        let dsm = make_dsm(
            &["A", "B", "C"],
            vec![vec![0, 1, 0], vec![0, 0, 1], vec![0, 0, 0]],
        );

        let costs = propagation_cost(&dsm);
        // Dependents graph (change propagation): C->B->A
        // Change to C affects B and A: reachable = {C, B, A} = 3/3 = 1.0
        assert!((costs[2] - 1.0).abs() < 1e-9);
        // Change to B affects A: reachable = {B, A} = 2/3
        assert!((costs[1] - 2.0 / 3.0).abs() < 1e-9);
        // Change to A affects nothing else: reachable = {A} = 1/3
        assert!((costs[0] - 1.0 / 3.0).abs() < 1e-9);
    }

    // --- 2.9b: Graph with a cycle (SCC detection) ---
    //
    // A -> B -> C -> A  (cycle in normal deps)
    #[test]
    fn test_cycle_scc() {
        let dsm = make_dsm(
            &["A", "B", "C"],
            vec![
                vec![0, 1, 0], // A depends on B
                vec![0, 0, 1], // B depends on C
                vec![1, 0, 0], // C depends on A
            ],
        );

        let clusters = find_clusters(&dsm);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0].crates.len(), 3);
        assert!(!clusters[0].dev_only);
    }

    #[test]
    fn test_dev_only_cycle() {
        // A dev-depends on B (kind=2), B dev-depends on A (kind=2) — no normal cycle.
        let dsm = make_dsm(
            &["A", "B"],
            vec![
                vec![0, 2], // A dev-depends on B
                vec![2, 0], // B dev-depends on A
            ],
        );

        let clusters = find_clusters(&dsm);
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].dev_only);
    }

    // --- 2.9c: Graph with a layering violation ---
    #[test]
    fn test_layering_violation() {
        // kernel (L0) depends on container (L3) — upward violation.
        let dsm = make_dsm(
            &["smallaios-kernel", "smallaios-container"],
            vec![
                vec![0, 1], // kernel depends on container
                vec![0, 0],
            ],
        );

        let violations = find_layering_violations(&dsm);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].from, "smallaios-kernel");
        assert_eq!(violations[0].to, "smallaios-container");
        assert_eq!(violations[0].from_layer, 0);
        assert_eq!(violations[0].to_layer, 3);
    }

    #[test]
    fn test_same_layer_violation() {
        // Two L2 crates depending on each other — same-layer violation.
        let dsm = make_dsm(
            &["smallaios-arch-x86_64", "smallaios-peripheral"],
            vec![vec![0, 1], vec![0, 0]],
        );

        let violations = find_layering_violations(&dsm);
        assert_eq!(violations.len(), 1);
        assert_eq!(violations[0].from_layer, 2);
        assert_eq!(violations[0].to_layer, 2);
    }

    #[test]
    fn test_kernel_security_no_violation() {
        // kernel <-> security is allowed (both L0).
        let dsm = make_dsm(
            &["smallaios-kernel", "smallaios-security"],
            vec![vec![0, 1], vec![1, 0]],
        );

        let violations = find_layering_violations(&dsm);
        assert!(violations.is_empty());
    }

    #[test]
    fn test_no_clusters_in_dag() {
        let dsm = make_dsm(
            &["A", "B", "C"],
            vec![vec![0, 1, 0], vec![0, 0, 1], vec![0, 0, 0]],
        );

        let clusters = find_clusters(&dsm);
        assert!(clusters.is_empty());
    }

    #[test]
    fn test_json_output_structure() {
        let dsm = make_dsm(&["A", "B"], vec![vec![0, 1], vec![0, 0]]);

        let output = build_output(&dsm);
        assert_eq!(output.summary.total_crates, 2);
        assert_eq!(output.summary.total_edges, 1);
        assert!(output.summary.acyclic_production);

        // Verify it serializes without error
        let json = serde_json::to_string_pretty(&output).unwrap();
        assert!(json.contains("propagation_cost"));
        assert!(json.contains("fan_in"));
        assert!(json.contains("fan_out"));
    }
}
