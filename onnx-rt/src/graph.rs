// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Execution graph builder for the SmallAIOS ONNX runtime.
//!
//! Converts an ONNX `GraphProto` into an `ExecutionGraph` with resolved
//! dependencies and topological ordering. The execution order is computed
//! using Kahn's algorithm to ensure operators execute only after all their
//! inputs are available.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::onnx_types::{AttributeProto, GraphProto};

/// Maximum depth of nested inner `ExecutionGraph`s produced by the graph
/// builder. Matches the protobuf parser's `MAX_GRAPH_NESTING_DEPTH`.
pub const MAX_GRAPH_NESTING_DEPTH: usize = 16;

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
    /// Nested inner-graph compilation exceeded [`MAX_GRAPH_NESTING_DEPTH`].
    NestingTooDeep,
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
            GraphError::NestingTooDeep => {
                write!(f, "nested inner-graph compilation exceeded max depth")
            }
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
    /// The operator domain (empty string = default ONNX domain,
    /// `"com.microsoft"` = Microsoft contrib op set).
    pub domain: String,
    /// Human-readable node name.
    pub name: String,
    /// Names of input tensors consumed by this node.
    pub inputs: Vec<String>,
    /// Names of output tensors produced by this node.
    pub outputs: Vec<String>,
    /// Indices of nodes that must execute before this one.
    pub dependencies: Vec<NodeIndex>,
    /// Operator-specific attributes from the ONNX NodeProto.
    pub attributes: Vec<AttributeProto>,
    /// Compiled inner graphs for control-flow operators (`If`/`Loop`/
    /// `Scan`), keyed by the originating attribute's name (e.g. `"body"`
    /// for `Loop`/`Scan`, `"then_branch"` / `"else_branch"` for `If`).
    ///
    /// Empty for operators that do not carry graph-valued attributes.
    /// Populated by the graph builder at session-load time via
    /// [`build_execution_graph`], which recursively compiles each
    /// `AttributeProto.g` value into its own standalone `ExecutionGraph`.
    pub inner_graphs: BTreeMap<String, ExecutionGraph>,
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
///
/// For each attribute whose `g` field is populated, recursively compiles
/// the inner `GraphProto` into its own standalone [`ExecutionGraph`] via
/// [`build_execution_graph_inner`] and stores the result in the node's
/// `inner_graphs` map, keyed by the attribute name. Returns
/// [`GraphError::NestingTooDeep`] if the nesting depth exceeds
/// [`MAX_GRAPH_NESTING_DEPTH`].
fn create_execution_nodes(
    graph: &GraphProto,
    depth: usize,
) -> Result<Vec<ExecutionNode>, GraphError> {
    let mut out = Vec::with_capacity(graph.node.len());
    for (i, node_proto) in graph.node.iter().enumerate() {
        let mut inner_graphs: BTreeMap<String, ExecutionGraph> = BTreeMap::new();
        for attr in &node_proto.attribute {
            if let Some(inner_proto) = &attr.g {
                let inner_exec = build_execution_graph_inner(inner_proto, depth + 1)?;
                inner_graphs.insert(attr.name.clone(), inner_exec);
            }
        }
        out.push(ExecutionNode {
            node_index: NodeIndex::new(i),
            op_type: node_proto.op_type.clone(),
            domain: node_proto.domain.clone(),
            name: node_proto.name.clone(),
            inputs: node_proto.input.clone(),
            outputs: node_proto.output.clone(),
            dependencies: Vec::new(),
            attributes: node_proto.attribute.clone(),
            inner_graphs,
        });
    }
    Ok(out)
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
///
/// When `allow_outer_refs` is `true`, input names that match neither a
/// graph input nor a sibling node output are treated as outer-scope
/// references (captured values from an enclosing graph) rather than
/// errors. This is required for the inner graphs of `If`/`Loop`/`Scan`
/// nodes, whose body graphs legitimately reference tensors defined in
/// the parent graph. The sub-graph executor supplies these names at
/// runtime from its parent scope.
fn resolve_dependencies(
    nodes: &[ExecutionNode],
    input_names: &[String],
    output_producers: &[(String, NodeIndex)],
    allow_outer_refs: bool,
) -> Result<Vec<Vec<NodeIndex>>, GraphError> {
    let num_nodes = nodes.len();
    let mut deps_per_node: Vec<Vec<NodeIndex>> = Vec::with_capacity(num_nodes);
    for _ in 0..num_nodes {
        deps_per_node.push(Vec::new());
    }

    for (node_idx, deps) in deps_per_node.iter_mut().enumerate() {
        let inputs = &nodes[node_idx].inputs;
        for input_name in inputs {
            resolve_single_input(
                input_name,
                input_names,
                output_producers,
                deps,
                allow_outer_refs,
            )?;
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
    allow_outer_refs: bool,
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
            if !input_name.is_empty() && !allow_outer_refs {
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
///    Any attribute whose `g` field is populated is recursively compiled
///    into a standalone inner `ExecutionGraph` and stored on the parent
///    node under `inner_graphs[attr.name]`.
/// 2. Resolves data dependencies by matching each node's inputs to the
///    outputs of other nodes.
/// 3. Performs a topological sort using Kahn's algorithm.
/// 4. Detects cycles (returns `GraphError::CyclicGraph` if found).
///
/// Returns `GraphError::NestingTooDeep` if inner-graph nesting exceeds
/// [`MAX_GRAPH_NESTING_DEPTH`].
pub fn build_execution_graph(graph: &GraphProto) -> Result<ExecutionGraph, GraphError> {
    build_execution_graph_with_depth(graph, 0, false)
}

/// Compiles an inner sub-graph (body of an `If`/`Loop`/`Scan`).
///
/// Unlike [`build_execution_graph`], inner compilation treats input
/// names that are neither declared graph inputs nor sibling outputs as
/// implicit outer-scope references rather than errors. The sub-graph
/// executor seeds these names at runtime from the parent value map.
pub fn build_execution_graph_inner(
    graph: &GraphProto,
    depth: usize,
) -> Result<ExecutionGraph, GraphError> {
    build_execution_graph_with_depth(graph, depth, true)
}

fn build_execution_graph_with_depth(
    graph: &GraphProto,
    depth: usize,
    allow_outer_refs: bool,
) -> Result<ExecutionGraph, GraphError> {
    if depth > MAX_GRAPH_NESTING_DEPTH {
        return Err(GraphError::NestingTooDeep);
    }

    let mut exec_graph = ExecutionGraph::new();

    // Copy graph-level input/output names from ValueInfoProto entries.
    exec_graph.input_names = graph.input.iter().map(|vi| vi.name.clone()).collect();
    exec_graph.output_names = graph.output.iter().map(|vi| vi.name.clone()).collect();

    // Phase 1: Create execution nodes (recursively compiles inner graphs).
    exec_graph.nodes = create_execution_nodes(graph, depth)?;

    // Phase 2: Resolve dependencies.
    let output_producers = build_output_producer_map(&exec_graph.nodes);
    let deps_per_node = resolve_dependencies(
        &exec_graph.nodes,
        &exec_graph.input_names,
        &output_producers,
        allow_outer_refs,
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

    // ---------------------------------------------------------------
    // Inner graph compilation (graph-attr-parser-v1)
    // ---------------------------------------------------------------

    use crate::onnx_types::{AttributeProto, AttributeType};

    fn graph_attr(name: &str, inner: GraphProto) -> AttributeProto {
        AttributeProto {
            name: name.to_string(),
            attr_type: AttributeType::Graph,
            g: Some(alloc::boxed::Box::new(inner)),
            ..AttributeProto::default()
        }
    }

    #[test]
    fn test_inner_graph_loop_body_compiled() {
        // Top-level graph with a single Loop node whose `body` attribute
        // carries an inner graph containing MatMul + Add.
        let inner_nodes = vec![
            make_node("MatMul", "mm", &["a", "W"], &["mm_out"]),
            make_node("Add", "add", &["mm_out", "b"], &["y"]),
        ];
        let inner_graph_proto = GraphProto {
            name: "body".to_string(),
            node: inner_nodes,
            // Body inputs: loop carries iter_num, cond_in, and a; W and b
            // are captured from the outer scope.
            input: vec![ValueInfoProto {
                name: "a".to_string(),
                ..ValueInfoProto::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..ValueInfoProto::default()
            }],
            ..GraphProto::default()
        };

        let loop_node = NodeProto {
            op_type: "Loop".to_string(),
            name: "lp".to_string(),
            input: vec!["M".to_string(), "cond".to_string(), "a0".to_string()],
            output: vec!["y_final".to_string()],
            attribute: vec![graph_attr("body", inner_graph_proto)],
            ..NodeProto::default()
        };

        let outer = GraphProto {
            name: "outer".to_string(),
            node: vec![loop_node],
            input: vec![
                ValueInfoProto {
                    name: "M".to_string(),
                    ..ValueInfoProto::default()
                },
                ValueInfoProto {
                    name: "cond".to_string(),
                    ..ValueInfoProto::default()
                },
                ValueInfoProto {
                    name: "a0".to_string(),
                    ..ValueInfoProto::default()
                },
            ],
            output: vec![ValueInfoProto {
                name: "y_final".to_string(),
                ..ValueInfoProto::default()
            }],
            ..GraphProto::default()
        };

        let exec = build_execution_graph(&outer).unwrap();
        assert_eq!(exec.nodes.len(), 1);
        let loop_exec = &exec.nodes[0];
        assert_eq!(loop_exec.op_type, "Loop");
        assert_eq!(loop_exec.inner_graphs.len(), 1);
        let body = loop_exec
            .inner_graphs
            .get("body")
            .expect("body inner graph should be populated");
        assert_eq!(body.node_count(), 2);
        assert_eq!(body.execution_order().len(), 2);
        // Parent still keeps the raw attribute too.
        assert_eq!(loop_exec.attributes.len(), 1);
    }

    #[test]
    fn test_inner_graph_if_both_branches() {
        let then_graph_proto = GraphProto {
            node: vec![make_node("Relu", "r", &["x"], &["y"])],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
                ..ValueInfoProto::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..ValueInfoProto::default()
            }],
            ..GraphProto::default()
        };
        let else_graph_proto = GraphProto {
            node: vec![make_node("Neg", "n", &["x"], &["y"])],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
                ..ValueInfoProto::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..ValueInfoProto::default()
            }],
            ..GraphProto::default()
        };
        let if_node = NodeProto {
            op_type: "If".to_string(),
            input: vec!["cond".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![
                graph_attr("then_branch", then_graph_proto),
                graph_attr("else_branch", else_graph_proto),
            ],
            ..NodeProto::default()
        };
        let outer = make_graph("outer", vec![if_node], &["cond", "x"], &["y"]);
        let exec = build_execution_graph(&outer).unwrap();
        let if_exec = &exec.nodes[0];
        assert_eq!(if_exec.inner_graphs.len(), 2);
        assert!(if_exec.inner_graphs.contains_key("then_branch"));
        assert!(if_exec.inner_graphs.contains_key("else_branch"));
        assert_eq!(if_exec.inner_graphs["then_branch"].nodes[0].op_type, "Relu");
        assert_eq!(if_exec.inner_graphs["else_branch"].nodes[0].op_type, "Neg");
    }

    #[test]
    fn test_inner_graph_outer_referenced_name_is_allowed() {
        // Inner graph references a name (`W`) that is neither an inner
        // input nor produced by an inner sibling node. This legitimately
        // refers to an outer-scope initializer. The inner builder must
        // NOT return MissingInput for it.
        let inner_graph_proto = GraphProto {
            node: vec![make_node("MatMul", "mm", &["x", "W"], &["y"])],
            input: vec![ValueInfoProto {
                name: "x".to_string(),
                ..ValueInfoProto::default()
            }],
            output: vec![ValueInfoProto {
                name: "y".to_string(),
                ..ValueInfoProto::default()
            }],
            ..GraphProto::default()
        };
        let outer_node = NodeProto {
            op_type: "Loop".to_string(),
            input: vec!["M".to_string(), "c".to_string(), "x0".to_string()],
            output: vec!["y".to_string()],
            attribute: vec![graph_attr("body", inner_graph_proto)],
            ..NodeProto::default()
        };
        // Note: `W` is an outer input in the top-level graph.
        let outer = make_graph("outer", vec![outer_node], &["M", "c", "x0", "W"], &["y"]);
        let exec = build_execution_graph(&outer).expect("outer ref should not be rejected");
        assert_eq!(exec.nodes[0].inner_graphs.len(), 1);
    }

    #[test]
    fn test_inner_graph_nesting_depth_overflow() {
        // Construct a chain of nested Loop bodies 17 levels deep and
        // verify the builder rejects the 17th level.
        fn wrap_loop(inner: GraphProto) -> GraphProto {
            let lp = NodeProto {
                op_type: "Loop".to_string(),
                attribute: vec![graph_attr("body", inner)],
                ..NodeProto::default()
            };
            GraphProto {
                node: vec![lp],
                ..GraphProto::default()
            }
        }
        let mut g = GraphProto::default();
        for _ in 0..17 {
            g = wrap_loop(g);
        }
        let err = build_execution_graph(&g).expect_err("depth 17 should reject");
        assert_eq!(err, GraphError::NestingTooDeep);
    }

    #[test]
    fn test_graph_error_nesting_too_deep_display() {
        use alloc::format;
        assert_eq!(
            format!("{}", GraphError::NestingTooDeep),
            "nested inner-graph compilation exceeded max depth"
        );
    }

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
