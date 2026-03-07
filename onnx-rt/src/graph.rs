// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Execution graph builder for the SmallAIOS ONNX runtime.
//!
//! Converts an ONNX `GraphProto` into an `ExecutionGraph` with resolved
//! dependencies and topological ordering. The execution order is computed
//! using Kahn's algorithm to ensure operators execute only after all their
//! inputs are available.

use alloc::string::String;
use alloc::vec::Vec;

use crate::onnx_types::GraphProto;

// Re-export for downstream consumers that need the tensor DataType.
#[allow(unused_imports)]
use crate::tensor::DataType;

/// Errors that can occur during execution graph construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    /// The graph contains a cycle and cannot be topologically sorted.
    CyclicGraph,
    /// Two or more nodes share the same name.
    DuplicateNodeName(String),
    /// A required input tensor has no producing node and is not a graph input.
    MissingInput(String),
    /// A node has an invalid configuration (e.g., no outputs).
    InvalidNode(String),
    /// The operator type is not supported by this runtime.
    UnsupportedOperator(String),
    /// Tensor shapes are incompatible between connected nodes.
    ShapeMismatch,
}

impl core::fmt::Display for GraphError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GraphError::CyclicGraph => write!(f, "execution graph contains a cycle"),
            GraphError::DuplicateNodeName(name) => {
                write!(f, "duplicate node name: {}", name)
            }
            GraphError::MissingInput(name) => {
                write!(f, "missing input tensor: {}", name)
            }
            GraphError::InvalidNode(msg) => {
                write!(f, "invalid node: {}", msg)
            }
            GraphError::UnsupportedOperator(op) => {
                write!(f, "unsupported operator: {}", op)
            }
            GraphError::ShapeMismatch => write!(f, "tensor shape mismatch"),
        }
    }
}

/// Index into the execution graph's node array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeIndex(usize);

impl NodeIndex {
    /// Creates a new `NodeIndex` from a raw index.
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the raw index value.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// Index into the execution graph's edge array.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct EdgeIndex(usize);

impl EdgeIndex {
    /// Creates a new `EdgeIndex` from a raw index.
    pub fn new(index: usize) -> Self {
        Self(index)
    }

    /// Returns the raw index value.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// A single node in the execution graph, representing one ONNX operator.
#[derive(Debug, Clone)]
pub struct ExecutionNode {
    /// This node's index in the execution graph.
    pub node_index: NodeIndex,
    /// The ONNX operator type (e.g., "Conv", "Relu").
    pub op_type: String,
    /// Human-readable node name.
    pub name: String,
    /// Names of input tensors consumed by this node.
    pub inputs: Vec<String>,
    /// Names of output tensors produced by this node.
    pub outputs: Vec<String>,
    /// Indices of nodes that must execute before this one.
    pub dependencies: Vec<NodeIndex>,
}

/// A directed acyclic graph of execution nodes with a computed
/// topological ordering.
#[derive(Debug, Clone)]
pub struct ExecutionGraph {
    /// All nodes in the graph, indexed by `NodeIndex`.
    pub nodes: Vec<ExecutionNode>,
    /// The topologically sorted execution order.
    pub topological_order: Vec<NodeIndex>,
    /// Names of the graph's external input tensors.
    pub input_names: Vec<String>,
    /// Names of the graph's external output tensors.
    pub output_names: Vec<String>,
}

impl Default for ExecutionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ExecutionGraph {
    /// Creates a new, empty execution graph.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            topological_order: Vec::new(),
            input_names: Vec::new(),
            output_names: Vec::new(),
        }
    }

    /// Returns the number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Returns the topologically sorted execution order.
    pub fn execution_order(&self) -> &[NodeIndex] {
        &self.topological_order
    }
}

/// Creates execution nodes from ONNX NodeProto entries.
fn create_execution_nodes(graph: &GraphProto) -> Vec<ExecutionNode> {
    graph
        .node
        .iter()
        .enumerate()
        .map(|(i, node_proto)| ExecutionNode {
            node_index: NodeIndex::new(i),
            op_type: node_proto.op_type.clone(),
            name: node_proto.name.clone(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            dependencies: Vec::new(),
        })
        .collect()
}

/// Builds the output-tensor-name to producing-node-index mapping.
fn build_output_producer_map(nodes: &[ExecutionNode]) -> Vec<(String, NodeIndex)> {
    let mut output_producers: Vec<(String, NodeIndex)> = Vec::new();
    for node in nodes {
        for output_name in &node.outputs {
            output_producers.push((output_name.clone(), node.node_index));
        }
    }
    output_producers
}

/// Resolves data dependencies for each node by matching inputs to
/// producing nodes. Returns per-node dependency lists.
fn resolve_dependencies(
    nodes: &[ExecutionNode],
    input_names: &[String],
    output_producers: &[(String, NodeIndex)],
) -> Result<Vec<Vec<NodeIndex>>, GraphError> {
    let num_nodes = nodes.len();
    let mut deps_per_node: Vec<Vec<NodeIndex>> = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        deps_per_node.push(Vec::new());
    }

    for (node_idx, deps) in deps_per_node.iter_mut().enumerate() {
        let inputs = &nodes[node_idx].inputs;
        for input_name in inputs {
            resolve_single_input(input_name, input_names, output_producers, deps)?;
        }
    }

    Ok(deps_per_node)
}

/// Resolve a single input name to its producing node, adding it to `deps`.
fn resolve_single_input(
    input_name: &str,
    input_names: &[String],
    output_producers: &[(String, NodeIndex)],
    deps: &mut Vec<NodeIndex>,
) -> Result<(), GraphError> {
    if input_names.iter().any(|n| n == input_name) {
        return Ok(());
    }
    let producer = output_producers.iter().find(|(name, _)| name == input_name);
    match producer {
        Some((_, prod_idx)) => {
            if !deps.iter().any(|d| d.index() == prod_idx.index()) {
                deps.push(*prod_idx);
            }
        }
        None => {
            if !input_name.is_empty() {
                return Err(GraphError::MissingInput(String::from(input_name)));
            }
        }
    }
    Ok(())
}

/// Builds an `ExecutionGraph` from an ONNX `GraphProto`.
///
/// This function:
/// 1. Creates an `ExecutionNode` for each `NodeProto` in the graph.
/// 2. Resolves data dependencies by matching each node's inputs to the
///    outputs of other nodes.
/// 3. Performs a topological sort using Kahn's algorithm.
/// 4. Detects cycles (returns `GraphError::CyclicGraph` if found).
pub fn build_execution_graph(graph: &GraphProto) -> Result<ExecutionGraph, GraphError> {
    let mut exec_graph = ExecutionGraph::new();

    // Copy graph-level input/output names from ValueInfoProto entries.
    exec_graph.input_names = graph.input.iter().map(|vi| vi.name.clone()).collect();
    exec_graph.output_names = graph.output.iter().map(|vi| vi.name.clone()).collect();

    // Phase 1: Create execution nodes.
    exec_graph.nodes = create_execution_nodes(graph);

    // Phase 2: Resolve dependencies.
    let output_producers = build_output_producer_map(&exec_graph.nodes);
    let deps_per_node = resolve_dependencies(
        &exec_graph.nodes,
        &exec_graph.input_names,
        &output_producers,
    )?;

    for (i, deps) in deps_per_node.into_iter().enumerate() {
        exec_graph.nodes[i].dependencies = deps;
    }

    // Phase 3: Topological sort.
    exec_graph.topological_order = topological_sort(&exec_graph.nodes)?;

    Ok(exec_graph)
}

/// Computes the in-degree (number of dependencies) for each node and returns
/// the initial set of nodes with zero in-degree.
fn compute_in_degrees(nodes: &[ExecutionNode]) -> (Vec<usize>, Vec<NodeIndex>) {
    let num_nodes = nodes.len();
    let mut in_degree: Vec<usize> = alloc::vec![0; num_nodes];

    for node in nodes {
        in_degree[node.node_index.index()] = node.dependencies.len();
    }

    let zero_degree_nodes: Vec<NodeIndex> = in_degree
        .iter()
        .enumerate()
        .filter(|(_, &deg)| deg == 0)
        .map(|(i, _)| NodeIndex::new(i))
        .collect();

    (in_degree, zero_degree_nodes)
}

/// Processes a single node from the BFS queue during topological sort,
/// decrementing in-degrees of dependent nodes and enqueuing newly ready ones.
fn process_dependents(
    nodes: &[ExecutionNode],
    current: NodeIndex,
    in_degree: &mut [usize],
    queue: &mut Vec<NodeIndex>,
) {
    for node in nodes {
        let depends_on_current = node
            .dependencies
            .iter()
            .any(|dep| dep.index() == current.index());
        if depends_on_current {
            in_degree[node.node_index.index()] -= 1;
            if in_degree[node.node_index.index()] == 0 {
                queue.push(node.node_index);
            }
        }
    }
}

/// Performs a topological sort of the execution nodes using Kahn's algorithm.
///
/// Returns the sorted node indices, or `GraphError::CyclicGraph` if the
/// dependency graph contains a cycle.
pub fn topological_sort(nodes: &[ExecutionNode]) -> Result<Vec<NodeIndex>, GraphError> {
    let num_nodes = nodes.len();
    if num_nodes == 0 {
        return Ok(Vec::new());
    }

    let (mut in_degree, initial_queue) = compute_in_degrees(nodes);
    let mut queue = initial_queue;
    let mut sorted: Vec<NodeIndex> = Vec::with_capacity(num_nodes);
    let mut head = 0;

    while head < queue.len() {
        let current = queue[head];
        head += 1;
        sorted.push(current);
        process_dependents(nodes, current, &mut in_degree, &mut queue);
    }

    if sorted.len() != num_nodes {
        return Err(GraphError::CyclicGraph);
    }

    Ok(sorted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_types::{NodeProto, ValueInfoProto};
    use alloc::string::ToString;
    use alloc::vec;

    /// Helper: creates a `NodeProto` with the given op, name, inputs, outputs.
    fn make_node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: name.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..NodeProto::default()
        }
    }

    /// Helper: creates a `GraphProto` with the given nodes, inputs, outputs.
    fn make_graph(
        name: &str,
        nodes: Vec<NodeProto>,
        inputs: &[&str],
        outputs: &[&str],
    ) -> GraphProto {
        GraphProto {
            name: name.to_string(),
            node: nodes,
            input: inputs
                .iter()
                .map(|s| ValueInfoProto {
                    name: s.to_string(),
                    ..ValueInfoProto::default()
                })
                .collect(),
            output: outputs
                .iter()
                .map(|s| ValueInfoProto {
                    name: s.to_string(),
                    ..ValueInfoProto::default()
                })
                .collect(),
            ..GraphProto::default()
        }
    }

    // ---------------------------------------------------------------
    // Empty graph
    // ---------------------------------------------------------------

    #[test]
    fn test_empty_graph() {
        let graph_proto = make_graph("empty", vec![], &[], &[]);
        let exec_graph = build_execution_graph(&graph_proto).unwrap();
        assert_eq!(exec_graph.node_count(), 0);
        assert!(exec_graph.execution_order().is_empty());
    }

    // ---------------------------------------------------------------
    // Single node
    // ---------------------------------------------------------------

    #[test]
    fn test_single_node() {
        let nodes = vec![make_node("Relu", "relu0", &["x"], &["y"])];
        let graph_proto = make_graph("single", nodes, &["x"], &["y"]);
        let exec_graph = build_execution_graph(&graph_proto).unwrap();
        assert_eq!(exec_graph.node_count(), 1);
        assert_eq!(exec_graph.execution_order().len(), 1);
        assert_eq!(exec_graph.execution_order()[0].index(), 0);
    }

    // ---------------------------------------------------------------
    // Linear chain: A -> B -> C
    // ---------------------------------------------------------------

    #[test]
    fn test_linear_chain() {
        let nodes = vec![
            make_node("Conv", "A", &["input"], &["a_out"]),
            make_node("Relu", "B", &["a_out"], &["b_out"]),
            make_node("Pool", "C", &["b_out"], &["output"]),
        ];
        let graph_proto = make_graph("linear", nodes, &["input"], &["output"]);
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        assert_eq!(exec_graph.node_count(), 3);
        let order = exec_graph.execution_order();
        assert_eq!(order.len(), 3);

        // A must come before B, B must come before C.
        let pos_a = order.iter().position(|n| n.index() == 0).unwrap();
        let pos_b = order.iter().position(|n| n.index() == 1).unwrap();
        let pos_c = order.iter().position(|n| n.index() == 2).unwrap();
        assert!(pos_a < pos_b);
        assert!(pos_b < pos_c);
    }

    // ---------------------------------------------------------------
    // Diamond: A -> B, A -> C, B -> D, C -> D
    // ---------------------------------------------------------------

    #[test]
    fn test_diamond_graph() {
        let nodes = vec![
            make_node("Conv", "A", &["input"], &["a_out"]),
            make_node("Relu", "B", &["a_out"], &["b_out"]),
            make_node("Sigmoid", "C", &["a_out"], &["c_out"]),
            make_node("Add", "D", &["b_out", "c_out"], &["output"]),
        ];
        let graph_proto = make_graph("diamond", nodes, &["input"], &["output"]);
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        assert_eq!(exec_graph.node_count(), 4);
        let order = exec_graph.execution_order();
        assert_eq!(order.len(), 4);

        let pos_a = order.iter().position(|n| n.index() == 0).unwrap();
        let pos_b = order.iter().position(|n| n.index() == 1).unwrap();
        let pos_c = order.iter().position(|n| n.index() == 2).unwrap();
        let pos_d = order.iter().position(|n| n.index() == 3).unwrap();

        // A before B and C; B and C before D.
        assert!(pos_a < pos_b);
        assert!(pos_a < pos_c);
        assert!(pos_b < pos_d);
        assert!(pos_c < pos_d);
    }

    // ---------------------------------------------------------------
    // Cycle detection
    // ---------------------------------------------------------------

    #[test]
    fn test_cycle_detection() {
        // Create a cycle: A -> B -> C -> A
        // We do this by having C output "c_out" which A consumes,
        // but "c_out" is NOT a graph input, so it must come from C.
        let nodes = vec![
            make_node("Op", "A", &["c_out"], &["a_out"]),
            make_node("Op", "B", &["a_out"], &["b_out"]),
            make_node("Op", "C", &["b_out"], &["c_out"]),
        ];
        let graph_proto = make_graph("cyclic", nodes, &[], &[]);
        let result = build_execution_graph(&graph_proto);
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), GraphError::CyclicGraph);
    }

    // ---------------------------------------------------------------
    // Missing input
    // ---------------------------------------------------------------

    #[test]
    fn test_missing_input() {
        let nodes = vec![make_node("Relu", "A", &["nonexistent"], &["output"])];
        // "nonexistent" is not a graph input and no node produces it.
        let graph_proto = make_graph("missing", nodes, &[], &["output"]);
        let result = build_execution_graph(&graph_proto);
        assert!(result.is_err());
        match result.unwrap_err() {
            GraphError::MissingInput(name) => assert_eq!(name, "nonexistent"),
            other => panic!("expected MissingInput, got {:?}", other),
        }
    }

    // ---------------------------------------------------------------
    // Node count
    // ---------------------------------------------------------------

    #[test]
    fn test_node_count() {
        let graph = ExecutionGraph::new();
        assert_eq!(graph.node_count(), 0);
    }

    // ---------------------------------------------------------------
    // NodeIndex and EdgeIndex accessors
    // ---------------------------------------------------------------

    #[test]
    fn test_node_index_accessors() {
        let idx = NodeIndex::new(42);
        assert_eq!(idx.index(), 42);
    }

    #[test]
    fn test_edge_index_accessors() {
        let idx = EdgeIndex::new(7);
        assert_eq!(idx.index(), 7);
    }

    // ---------------------------------------------------------------
    // Topological sort standalone
    // ---------------------------------------------------------------

    #[test]
    fn test_topological_sort_empty() {
        let nodes: Vec<ExecutionNode> = Vec::new();
        let order = topological_sort(&nodes).unwrap();
        assert!(order.is_empty());
    }

    // ---------------------------------------------------------------
    // Graph with optional (empty) inputs
    // ---------------------------------------------------------------

    #[test]
    fn test_optional_empty_inputs() {
        // ONNX allows empty strings for optional inputs.
        let nodes = vec![make_node("BatchNorm", "bn", &["x", "", ""], &["y"])];
        let graph_proto = make_graph("optional", nodes, &["x"], &["y"]);
        let exec_graph = build_execution_graph(&graph_proto).unwrap();
        assert_eq!(exec_graph.node_count(), 1);
    }

    // ---------------------------------------------------------------
    // Graph-level input and output names
    // ---------------------------------------------------------------

    #[test]
    fn test_graph_input_output_names() {
        let nodes = vec![make_node("Identity", "id", &["in0"], &["out0"])];
        let graph_proto = make_graph("io_test", nodes, &["in0"], &["out0"]);
        let exec_graph = build_execution_graph(&graph_proto).unwrap();
        assert_eq!(exec_graph.input_names, vec!["in0".to_string()]);
        assert_eq!(exec_graph.output_names, vec!["out0".to_string()]);
    }

    // ---------------------------------------------------------------
    // GraphError Display
    // ---------------------------------------------------------------

    #[test]
    fn test_graph_error_display() {
        use alloc::format;
        assert_eq!(
            format!("{}", GraphError::CyclicGraph),
            "execution graph contains a cycle"
        );
        assert_eq!(
            format!("{}", GraphError::MissingInput("t1".to_string())),
            "missing input tensor: t1"
        );
        assert_eq!(
            format!("{}", GraphError::DuplicateNodeName("n".to_string())),
            "duplicate node name: n"
        );
        assert_eq!(
            format!("{}", GraphError::InvalidNode("bad".to_string())),
            "invalid node: bad"
        );
        assert_eq!(
            format!(
                "{}",
                GraphError::UnsupportedOperator("CustomOp".to_string())
            ),
            "unsupported operator: CustomOp"
        );
        assert_eq!(
            format!("{}", GraphError::ShapeMismatch),
            "tensor shape mismatch"
        );
    }
}
