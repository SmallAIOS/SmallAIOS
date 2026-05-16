// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Programmatic [`ExecutionGraph`] construction for safetensors models.
//!
//! The existing ONNX path builds an `ExecutionGraph` from a parsed
//! `GraphProto` (see [`crate::graph::build_execution_graph`]). Safetensors
//! models (Gemma, Llama, ...) have no ONNX protobuf — the graph must be
//! constructed directly in Rust from a [`GemmaConfig`] plus weights pulled
//! from a [`crate::model_loader::SafetensorsFile`].
//!
//! [`GraphBuilder`] is that bridge: a thin helper that accumulates
//! [`ExecutionNode`]s sequentially, auto-assigns tensor names, bundles
//! initializer [`TensorProto`]s, and produces a [`BuiltGraph`] that the
//! existing host executor and the GPU executor (`execute_graph_gpu`)
//! already know how to run.
//!
//! The builder is scoped to Section 6 of `safetensors-model-loader-v1`.
//! Section 7 (next) will build the Gemma-specific layer template on top
//! of this primitive.

use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec::Vec;

use super::config::GemmaConfig;
use super::LoaderError;
use crate::graph::{topological_sort, ExecutionGraph, ExecutionNode, GraphError, NodeIndex};
use crate::onnx_types::{AttributeProto, AttributeType, TensorProto};

#[cfg(feature = "safetensors")]
use super::safetensors::TensorView;

#[cfg(feature = "cuda")]
use crate::cuda::gpu_executor::DeviceTensor;
#[cfg(feature = "cuda")]
use crate::cuda::memory::DeviceBuffer;
#[cfg(feature = "cuda")]
use crate::cuda::{CudaError, CudaRuntime};

// ─────────────────────────────────────────────────────────────────────
// BuiltGraph — bundles (ExecutionGraph, initializers) as the builder's
// output. The existing host executor consumes a graph + initializer slice
// pair, and the GPU executor takes the same shape. Returning a small
// struct avoids touching `graph.rs` / `session.rs` / `executor.rs`.
// ─────────────────────────────────────────────────────────────────────

/// A fully-built execution graph plus its initializer (weight) tensors.
///
/// Consumed by [`crate::executor::execute_graph`] and
/// [`crate::cuda::gpu_executor::execute_graph_gpu`], which both accept an
/// `ExecutionGraph` alongside an `&[TensorProto]` of initializers.
#[derive(Debug, Clone)]
pub struct BuiltGraph {
    /// Topologically-sorted execution graph.
    pub graph: ExecutionGraph,
    /// Model weight tensors referenced by nodes in the graph.
    pub initializers: Vec<TensorProto>,
}

// ─────────────────────────────────────────────────────────────────────
// GraphBuilder
// ─────────────────────────────────────────────────────────────────────

/// Builder for an [`ExecutionGraph`] that does not come from an ONNX
/// protobuf file.
///
/// The caller declares graph inputs and outputs up-front, registers
/// weight initializers, and then chains operator helpers to append
/// nodes. Each helper returns the freshly-allocated output tensor name,
/// which can be fed into the next helper. [`GraphBuilder::build`]
/// validates all tensor references, runs a topological sort, and
/// returns the ready-to-execute [`BuiltGraph`].
///
/// Nodes are appended in dependency order (the builder has no notion
/// of out-of-order insertion), so the resulting execution order is
/// simply `0..N`, but `build()` still runs `topological_sort` to
/// exercise the same path the ONNX loader uses and to detect accidental
/// cycles.
pub struct GraphBuilder {
    /// Nodes accumulated so far, in insertion order.
    nodes: Vec<ExecutionNode>,
    /// User-declared graph inputs (not initializers).
    input_names: Vec<String>,
    /// User-declared graph outputs — marked via [`GraphBuilder::output`]
    /// after the producing nodes are appended.
    output_names: Vec<String>,
    /// Model weight initializers, keyed by tensor name.
    initializers: BTreeMap<String, TensorProto>,
    /// Running set of every tensor name produced so far (graph inputs,
    /// initializers, or node outputs). Used for O(log n) validation in
    /// the op helpers and [`GraphBuilder::output`].
    produced: BTreeSet<String>,
    /// Counter for auto-generated `tensor_N` names.
    next_tensor_id: usize,
}

impl GraphBuilder {
    /// Create a new builder with the given graph input names. The
    /// `output_names` argument is reserved for callers that want to
    /// pre-declare outputs; outputs may also be added later via
    /// [`GraphBuilder::output`].
    pub fn new(input_names: Vec<String>, output_names: Vec<String>) -> Self {
        let mut produced: BTreeSet<String> = BTreeSet::new();
        for n in &input_names {
            produced.insert(n.clone());
        }
        Self {
            nodes: Vec::new(),
            input_names,
            output_names,
            initializers: BTreeMap::new(),
            produced,
            next_tensor_id: 0,
        }
    }

    /// Convenience constructor for a Gemma-style decoder graph.
    ///
    /// Declares a single `input_ids` graph input and a single `logits`
    /// graph output placeholder — Section 7 (the Gemma layer template)
    /// will fill in the body and call [`GraphBuilder::output`] once the
    /// final logits node has been appended.
    pub fn new_for_gemma(_config: &GemmaConfig) -> Self {
        // Gemma's token id input; the output name is not declared yet
        // because the builder in Section 7 decides which tensor is the
        // final logits output.
        Self::new(alloc::vec!["input_ids".to_string()], Vec::new())
    }

    // ── 6.2 Automatic tensor name allocation ─────────────────────────

    /// Allocate and return the next auto-generated tensor name
    /// (`tensor_0`, `tensor_1`, ...).
    pub fn fresh_tensor_name(&mut self) -> String {
        let name = format!("tensor_{}", self.next_tensor_id);
        self.next_tensor_id += 1;
        name
    }

    // ── 6.3 Initializers ─────────────────────────────────────────────

    /// Register a weight initializer. The `name` must not already be
    /// bound (to a graph input, a prior initializer, or a node output).
    pub fn add_initializer(
        &mut self,
        name: String,
        mut tensor_proto: TensorProto,
    ) -> Result<(), LoaderError> {
        if self.produced.contains(&name) {
            return Err(LoaderError::InvalidConfig(format!(
                "initializer '{name}' duplicates an existing tensor name"
            )));
        }
        tensor_proto.name = name.clone();
        self.produced.insert(name.clone());
        self.initializers.insert(name, tensor_proto);
        Ok(())
    }

    /// Register a weight initializer from a borrowed safetensors
    /// [`TensorView`]. Copies the bytes out of the mmap region into an
    /// owned `TensorProto` so the builder does not depend on the
    /// lifetime of the underlying file.
    #[cfg(feature = "safetensors")]
    pub fn add_initializer_from_view(
        &mut self,
        name: String,
        view: &TensorView<'_>,
    ) -> Result<(), LoaderError> {
        let proto = TensorProto {
            dims: view.shape.to_vec(),
            data_type: view.dtype as i32,
            name: name.clone(),
            raw_data: view.data.to_vec(),
            ..TensorProto::default()
        };
        self.add_initializer(name, proto)
    }

    // ── 6.4 Operator helpers ────────────────────────────────────────

    /// Append a node to the builder and record its single output name.
    fn push_node(
        &mut self,
        op_type: &str,
        domain: &str,
        inputs: Vec<String>,
        attributes: Vec<AttributeProto>,
    ) -> String {
        let output = self.fresh_tensor_name();
        self.push_node_with_outputs(
            op_type,
            domain,
            inputs,
            attributes,
            alloc::vec![output.clone()],
        );
        output
    }

    /// Append a node with caller-specified output names.
    fn push_node_with_outputs(
        &mut self,
        op_type: &str,
        domain: &str,
        inputs: Vec<String>,
        attributes: Vec<AttributeProto>,
        outputs: Vec<String>,
    ) {
        let idx = self.nodes.len();
        let name = format!("{}_{}", op_type, idx);
        for o in &outputs {
            self.produced.insert(o.clone());
        }
        self.nodes.push(ExecutionNode {
            node_index: NodeIndex::new(idx),
            op_type: op_type.to_string(),
            domain: domain.to_string(),
            name,
            inputs,
            outputs,
            dependencies: Vec::new(),
            attributes,
            inner_graphs: BTreeMap::new(),
        });
    }

    /// MatMul node. Returns the freshly-allocated output tensor name.
    pub fn matmul(&mut self, a: &str, b: &str) -> String {
        self.push_node(
            "MatMul",
            "",
            alloc::vec![a.to_string(), b.to_string()],
            Vec::new(),
        )
    }

    /// Gemm node (`alpha * A [^T] * B [^T] + beta * C`).
    ///
    /// `c` is optional — if `None` the bias input is omitted (empty
    /// string input per ONNX convention).
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(
        &mut self,
        a: &str,
        b: &str,
        c: Option<&str>,
        alpha: f32,
        beta: f32,
        trans_a: bool,
        trans_b: bool,
    ) -> String {
        let mut inputs = alloc::vec![a.to_string(), b.to_string()];
        if let Some(c_name) = c {
            inputs.push(c_name.to_string());
        }
        let attrs = alloc::vec![
            float_attr("alpha", alpha),
            float_attr("beta", beta),
            int_attr("transA", if trans_a { 1 } else { 0 }),
            int_attr("transB", if trans_b { 1 } else { 0 }),
        ];
        self.push_node("Gemm", "", inputs, attrs)
    }

    /// Elementwise Add.
    pub fn add(&mut self, a: &str, b: &str) -> String {
        self.push_node(
            "Add",
            "",
            alloc::vec![a.to_string(), b.to_string()],
            Vec::new(),
        )
    }

    /// Elementwise Mul.
    pub fn mul(&mut self, a: &str, b: &str) -> String {
        self.push_node(
            "Mul",
            "",
            alloc::vec![a.to_string(), b.to_string()],
            Vec::new(),
        )
    }

    /// RMSNormalization (standard ONNX op).
    pub fn rms_norm(&mut self, x: &str, weight: &str, epsilon: f32) -> String {
        let attrs = alloc::vec![float_attr("epsilon", epsilon)];
        self.push_node(
            "RMSNormalization",
            "",
            alloc::vec![x.to_string(), weight.to_string()],
            attrs,
        )
    }

    /// `com.microsoft.RotaryEmbedding` fused RoPE op.
    pub fn rotary_embedding(
        &mut self,
        x: &str,
        cos: &str,
        sin: &str,
        rotary_dim: i64,
        interleaved: bool,
    ) -> String {
        let attrs = alloc::vec![
            int_attr("rotary_embedding_dim", rotary_dim),
            int_attr("interleaved", if interleaved { 1 } else { 0 }),
        ];
        self.push_node(
            "RotaryEmbedding",
            "com.microsoft",
            alloc::vec![x.to_string(), cos.to_string(), sin.to_string()],
            attrs,
        )
    }

    /// `com.microsoft.GroupQueryAttention`. Returns the list of output
    /// tensor names: `[attn_out, present_k, present_v]` (present K/V
    /// are always produced — callers that don't care about the cache
    /// can ignore them).
    #[allow(clippy::too_many_arguments)]
    pub fn group_query_attention(
        &mut self,
        q: &str,
        k: &str,
        v: &str,
        past_k: Option<&str>,
        past_v: Option<&str>,
        num_heads: i64,
        num_kv_heads: i64,
        sliding_window: Option<i64>,
    ) -> Vec<String> {
        let mut inputs = alloc::vec![q.to_string(), k.to_string(), v.to_string()];
        // past_key / past_value are optional — fill with empty-string
        // placeholders when absent so positional indexing stays stable.
        inputs.push(past_k.map(|s| s.to_string()).unwrap_or_default());
        inputs.push(past_v.map(|s| s.to_string()).unwrap_or_default());

        let mut attrs = alloc::vec![
            int_attr("num_heads", num_heads),
            int_attr("kv_num_heads", num_kv_heads),
        ];
        if let Some(w) = sliding_window {
            attrs.push(int_attr("local_window_size", w));
        }

        let attn_out = self.fresh_tensor_name();
        let present_k = self.fresh_tensor_name();
        let present_v = self.fresh_tensor_name();
        let outputs = alloc::vec![attn_out.clone(), present_k.clone(), present_v.clone()];
        self.push_node_with_outputs(
            "GroupQueryAttention",
            "com.microsoft",
            inputs,
            attrs,
            outputs.clone(),
        );
        outputs
    }

    /// SwiGLU activation: `Mul(Silu(gate), up)`. Emits a Silu node
    /// followed by a Mul node and returns the Mul output.
    pub fn swiglu(&mut self, gate: &str, up: &str) -> String {
        let silu_out = self.push_node("Silu", "", alloc::vec![gate.to_string()], Vec::new());
        self.mul(&silu_out, up)
    }

    /// Embedding table lookup — emitted as an ONNX `Gather` along axis 0.
    /// `ids` is the integer index tensor, `weight` is the embedding
    /// table initializer.
    pub fn embedding_lookup(&mut self, ids: &str, weight: &str) -> String {
        let attrs = alloc::vec![int_attr("axis", 0)];
        // ONNX Gather: data = weight, indices = ids.
        self.push_node(
            "Gather",
            "",
            alloc::vec![weight.to_string(), ids.to_string()],
            attrs,
        )
    }

    /// Mark `tensor_name` as a graph output. Fails if the tensor has
    /// not been produced by a prior node and is not a graph input.
    pub fn output(&mut self, tensor_name: &str) -> Result<(), LoaderError> {
        if !self.produced.contains(tensor_name) {
            return Err(LoaderError::InvalidConfig(format!(
                "graph output '{tensor_name}' was never produced by any node or input"
            )));
        }
        self.output_names.push(tensor_name.to_string());
        Ok(())
    }

    // ── 6.5 Build ───────────────────────────────────────────────────

    /// Validate tensor references, run a topological sort, and return
    /// the finished [`BuiltGraph`].
    pub fn build(self) -> Result<BuiltGraph, LoaderError> {
        let GraphBuilder {
            mut nodes,
            input_names,
            output_names,
            initializers,
            produced: _,
            next_tensor_id: _,
        } = self;

        let initializer_names: BTreeSet<String> = initializers.keys().cloned().collect();
        let input_set: BTreeSet<String> = input_names.iter().cloned().collect();

        let output_to_node = resolve_node_dependencies(&mut nodes, &input_set, &initializer_names)?;
        validate_declared_outputs(&output_names, &output_to_node, &input_set)?;

        let topological_order = run_topo_sort(&nodes)?;

        let graph = ExecutionGraph {
            nodes,
            topological_order,
            input_names,
            output_names,
        };

        let initializers = initializers.into_values().collect();
        Ok(BuiltGraph {
            graph,
            initializers,
        })
    }

    // ── 6.6 GPU-resident weight loading ──────────────────────────────

    /// Copy every registered initializer to the device and return a
    /// map of `weight_name -> DeviceTensor`.
    ///
    /// Intended as the Section 7 bridge: the Gemma graph builder will
    /// call this right after registering safetensors-backed weights
    /// and hand the resulting map — along with
    /// [`BuiltGraph::graph`] — to
    /// [`crate::cuda::gpu_executor::execute_graph_gpu`].
    #[cfg(feature = "cuda")]
    pub fn load_weights_to_gpu(
        &self,
        _runtime: &CudaRuntime,
    ) -> Result<BTreeMap<String, DeviceTensor>, CudaError> {
        let mut out: BTreeMap<String, DeviceTensor> = BTreeMap::new();
        for (name, proto) in &self.initializers {
            let dtype = crate::tensor::DataType::from_i32(proto.data_type).ok_or(
                CudaError::RuntimeError {
                    op: "graph_builder::load_weights_to_gpu::dtype",
                    code: proto.data_type,
                },
            )?;
            // Determine byte count from the raw_data (safetensors-loaded
            // initializers always populate raw_data; typed-data fallbacks
            // are not expected on this path).
            let bytes = proto.raw_data.len();
            let buffer = DeviceBuffer::alloc(bytes)?;
            if bytes > 0 {
                buffer.copy_from_host(&proto.raw_data)?;
            }
            let dev = DeviceTensor {
                buffer,
                shape: proto.dims.clone(),
                dtype,
                name: name.clone(),
            };
            out.insert(name.clone(), dev);
        }
        Ok(out)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Attribute helpers — shape-compatible with the protobuf parser's
// output so the existing executor can read them transparently.
// ─────────────────────────────────────────────────────────────────────

// ─────────────────────────────────────────────────────────────────────
// `build()` helpers — extracted to keep cognitive complexity bounded.
// ─────────────────────────────────────────────────────────────────────

/// Walk each node, resolve its input names against graph inputs,
/// initializers, and prior-node outputs, and populate `dependencies`.
/// Returns a `name -> producing node` map used for later checks.
fn resolve_node_dependencies(
    nodes: &mut [ExecutionNode],
    input_set: &BTreeSet<String>,
    initializer_names: &BTreeSet<String>,
) -> Result<BTreeMap<String, NodeIndex>, LoaderError> {
    let mut output_to_node: BTreeMap<String, NodeIndex> = BTreeMap::new();
    let node_count = nodes.len();
    #[allow(clippy::needless_range_loop)]
    for i in 0..node_count {
        let node_inputs = nodes[i].inputs.clone();
        let deps = compute_node_deps(
            &node_inputs,
            &nodes[i].name,
            input_set,
            initializer_names,
            &output_to_node,
        )?;
        nodes[i].dependencies = deps;
        register_node_outputs(&mut output_to_node, &nodes[i].outputs, nodes[i].node_index);
    }
    Ok(output_to_node)
}

/// Resolve a single node's input list into a deduplicated dependency
/// vector. Returns an error if any input name is unknown.
fn compute_node_deps(
    node_inputs: &[String],
    node_name: &str,
    input_set: &BTreeSet<String>,
    initializer_names: &BTreeSet<String>,
    output_to_node: &BTreeMap<String, NodeIndex>,
) -> Result<Vec<NodeIndex>, LoaderError> {
    let mut deps: Vec<NodeIndex> = Vec::new();
    for input in node_inputs {
        if is_externally_provided(input, input_set, initializer_names) {
            continue;
        }
        let prod = output_to_node.get(input).copied().ok_or_else(|| {
            LoaderError::InvalidConfig(format!(
                "node '{node_name}' references unknown input '{input}'"
            ))
        })?;
        if !deps.iter().any(|d| d.index() == prod.index()) {
            deps.push(prod);
        }
    }
    Ok(deps)
}

/// True if `input` is empty (ONNX optional-input convention), a graph
/// input, or an initializer — i.e. not produced by a prior node.
fn is_externally_provided(
    input: &str,
    input_set: &BTreeSet<String>,
    initializer_names: &BTreeSet<String>,
) -> bool {
    input.is_empty() || input_set.contains(input) || initializer_names.contains(input)
}

/// Insert this node's non-empty outputs into the producer map.
fn register_node_outputs(
    output_to_node: &mut BTreeMap<String, NodeIndex>,
    outputs: &[String],
    node_index: NodeIndex,
) {
    for out in outputs {
        if !out.is_empty() {
            output_to_node.insert(out.clone(), node_index);
        }
    }
}

/// Ensure every declared graph output is either produced by a node or
/// is a passthrough graph input.
fn validate_declared_outputs(
    output_names: &[String],
    output_to_node: &BTreeMap<String, NodeIndex>,
    input_set: &BTreeSet<String>,
) -> Result<(), LoaderError> {
    for out in output_names {
        if !output_to_node.contains_key(out) && !input_set.contains(out) {
            return Err(LoaderError::InvalidConfig(format!(
                "declared graph output '{out}' is not produced by any node"
            )));
        }
    }
    Ok(())
}

/// Run topological sort and translate any errors back into the loader's
/// `InvalidConfig` family.
fn run_topo_sort(nodes: &[ExecutionNode]) -> Result<Vec<NodeIndex>, LoaderError> {
    topological_sort(nodes).map_err(|e| match e {
        GraphError::CyclicGraph => LoaderError::InvalidConfig("graph contains a cycle".to_string()),
        other => LoaderError::InvalidConfig(format!("graph build failed: {other}")),
    })
}

fn float_attr(name: &str, value: f32) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        attr_type: AttributeType::Float,
        f: value,
        ..AttributeProto::default()
    }
}

fn int_attr(name: &str, value: i64) -> AttributeProto {
    AttributeProto {
        name: name.to_string(),
        attr_type: AttributeType::Int,
        i: value,
        ..AttributeProto::default()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DataType;

    fn dummy_weight(name: &str, dims: Vec<i64>, bytes: usize) -> TensorProto {
        TensorProto {
            dims,
            data_type: DataType::Float as i32,
            name: name.to_string(),
            raw_data: alloc::vec![0u8; bytes],
            ..TensorProto::default()
        }
    }

    #[test]
    fn test_build_empty_graph() {
        let mut b = GraphBuilder::new(alloc::vec!["x".to_string()], Vec::new());
        b.output("x").unwrap();
        let built = b.build().expect("empty graph should build");
        assert_eq!(built.graph.nodes.len(), 0);
        assert!(built.graph.topological_order.is_empty());
        assert_eq!(built.graph.input_names, alloc::vec!["x".to_string()]);
        assert_eq!(built.graph.output_names, alloc::vec!["x".to_string()]);
        assert!(built.initializers.is_empty());
    }

    #[test]
    fn test_build_2_layer_matmul_chain() {
        // inputs: A (user), B + C (initializers)
        let mut b = GraphBuilder::new(alloc::vec!["A".to_string()], Vec::new());
        b.add_initializer("B".to_string(), dummy_weight("B", alloc::vec![4, 4], 64))
            .unwrap();
        b.add_initializer("C".to_string(), dummy_weight("C", alloc::vec![4, 4], 64))
            .unwrap();

        let tmp = b.matmul("A", "B");
        let out = b.matmul(&tmp, "C");
        b.output(&out).unwrap();
        let built = b.build().unwrap();

        assert_eq!(built.graph.nodes.len(), 2);
        assert_eq!(built.graph.topological_order.len(), 2);
        // Order: first MatMul, then second MatMul.
        assert_eq!(built.graph.topological_order[0].index(), 0);
        assert_eq!(built.graph.topological_order[1].index(), 1);

        // First node consumes A, B.
        assert_eq!(built.graph.nodes[0].op_type, "MatMul");
        assert_eq!(
            built.graph.nodes[0].inputs,
            alloc::vec!["A".to_string(), "B".to_string()]
        );
        // Second node consumes first node's output + C.
        assert_eq!(built.graph.nodes[1].inputs[0], tmp);
        assert_eq!(built.graph.nodes[1].inputs[1], "C".to_string());
        // Second node depends on first.
        assert_eq!(built.graph.nodes[1].dependencies.len(), 1);
        assert_eq!(built.graph.nodes[1].dependencies[0].index(), 0);
        // Output matches the second MatMul's output.
        assert_eq!(built.graph.output_names, alloc::vec![out]);

        // Both initializers are present.
        assert_eq!(built.initializers.len(), 2);
    }

    #[test]
    fn test_build_with_initializers() {
        let mut b = GraphBuilder::new(alloc::vec!["X".to_string()], Vec::new());
        b.add_initializer("W".to_string(), dummy_weight("W", alloc::vec![8, 8], 256))
            .unwrap();
        b.add_initializer("Bias".to_string(), dummy_weight("Bias", alloc::vec![8], 32))
            .unwrap();

        let y = b.gemm("X", "W", Some("Bias"), 1.0, 1.0, false, false);
        b.output(&y).unwrap();
        let built = b.build().unwrap();

        assert_eq!(built.graph.nodes.len(), 1);
        let node = &built.graph.nodes[0];
        assert_eq!(node.op_type, "Gemm");
        assert_eq!(
            node.inputs,
            alloc::vec!["X".to_string(), "W".to_string(), "Bias".to_string()]
        );
        // Attributes include alpha, beta, transA, transB.
        let names: Vec<&str> = node.attributes.iter().map(|a| a.name.as_str()).collect();
        assert!(names.contains(&"alpha"));
        assert!(names.contains(&"beta"));
        assert!(names.contains(&"transA"));
        assert!(names.contains(&"transB"));

        // Initializer names match what we registered.
        let init_names: BTreeSet<String> =
            built.initializers.iter().map(|t| t.name.clone()).collect();
        assert!(init_names.contains("W"));
        assert!(init_names.contains("Bias"));
    }

    #[test]
    fn test_missing_input_error() {
        let mut b = GraphBuilder::new(alloc::vec!["A".to_string()], Vec::new());
        // Force-inject a node referencing an undeclared input by
        // pushing directly — the helpers would never let this slip
        // through because they only take string names, not validate
        // them at insertion. But `build()` must reject it.
        b.nodes.push(ExecutionNode {
            node_index: NodeIndex::new(0),
            op_type: "MatMul".to_string(),
            domain: String::new(),
            name: "bad_0".to_string(),
            inputs: alloc::vec!["A".to_string(), "ghost".to_string()],
            outputs: alloc::vec!["out".to_string()],
            dependencies: Vec::new(),
            attributes: Vec::new(),
            inner_graphs: BTreeMap::new(),
        });
        b.produced.insert("out".to_string());
        b.output_names.push("out".to_string());

        let err = b.build().unwrap_err();
        match err {
            LoaderError::InvalidConfig(msg) => {
                assert!(msg.contains("ghost"), "message = {msg}");
                assert!(msg.contains("unknown input"), "message = {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_duplicate_initializer_error() {
        let mut b = GraphBuilder::new(Vec::new(), Vec::new());
        b.add_initializer("W".to_string(), dummy_weight("W", alloc::vec![4], 16))
            .unwrap();
        let err = b
            .add_initializer("W".to_string(), dummy_weight("W", alloc::vec![4], 16))
            .unwrap_err();
        match err {
            LoaderError::InvalidConfig(msg) => {
                assert!(msg.contains("duplicates"), "message = {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_unknown_output_error() {
        let mut b = GraphBuilder::new(alloc::vec!["A".to_string()], Vec::new());
        let err = b.output("does_not_exist").unwrap_err();
        match err {
            LoaderError::InvalidConfig(msg) => {
                assert!(msg.contains("does_not_exist"), "message = {msg}");
            }
            other => panic!("expected InvalidConfig, got {other:?}"),
        }
    }

    #[test]
    fn test_fresh_tensor_name_sequence() {
        let mut b = GraphBuilder::new(Vec::new(), Vec::new());
        assert_eq!(b.fresh_tensor_name(), "tensor_0");
        assert_eq!(b.fresh_tensor_name(), "tensor_1");
        assert_eq!(b.fresh_tensor_name(), "tensor_2");
    }

    #[test]
    fn test_swiglu_emits_silu_then_mul() {
        let mut b = GraphBuilder::new(
            alloc::vec!["gate".to_string(), "up".to_string()],
            Vec::new(),
        );
        let y = b.swiglu("gate", "up");
        b.output(&y).unwrap();
        let built = b.build().unwrap();
        assert_eq!(built.graph.nodes.len(), 2);
        assert_eq!(built.graph.nodes[0].op_type, "Silu");
        assert_eq!(built.graph.nodes[1].op_type, "Mul");
        // Mul depends on Silu.
        assert_eq!(built.graph.nodes[1].dependencies.len(), 1);
        assert_eq!(built.graph.nodes[1].dependencies[0].index(), 0);
    }

    #[test]
    fn test_rotary_and_gqa_helpers() {
        let mut b = GraphBuilder::new(
            alloc::vec![
                "q".to_string(),
                "k".to_string(),
                "v".to_string(),
                "cos".to_string(),
                "sin".to_string(),
            ],
            Vec::new(),
        );
        let q_rope = b.rotary_embedding("q", "cos", "sin", 64, false);
        let outs = b.group_query_attention(&q_rope, "k", "v", None, None, 8, 4, Some(4096));
        assert_eq!(outs.len(), 3);
        b.output(&outs[0]).unwrap();
        let built = b.build().unwrap();

        assert_eq!(built.graph.nodes.len(), 2);
        assert_eq!(built.graph.nodes[0].op_type, "RotaryEmbedding");
        assert_eq!(built.graph.nodes[0].domain, "com.microsoft");
        assert_eq!(built.graph.nodes[1].op_type, "GroupQueryAttention");
        assert_eq!(built.graph.nodes[1].domain, "com.microsoft");
        // GQA has three outputs.
        assert_eq!(built.graph.nodes[1].outputs.len(), 3);
    }

    #[test]
    fn test_embedding_lookup_uses_gather() {
        let mut b = GraphBuilder::new(alloc::vec!["ids".to_string()], Vec::new());
        b.add_initializer(
            "embed".to_string(),
            dummy_weight("embed", alloc::vec![256, 64], 256 * 64 * 4),
        )
        .unwrap();
        let y = b.embedding_lookup("ids", "embed");
        b.output(&y).unwrap();
        let built = b.build().unwrap();
        assert_eq!(built.graph.nodes[0].op_type, "Gather");
        // Gather inputs: data = embed, indices = ids.
        assert_eq!(built.graph.nodes[0].inputs[0], "embed");
        assert_eq!(built.graph.nodes[0].inputs[1], "ids");
    }

    #[test]
    fn test_rms_norm_attribute() {
        let mut b = GraphBuilder::new(alloc::vec!["x".to_string()], Vec::new());
        b.add_initializer(
            "gamma".to_string(),
            dummy_weight("gamma", alloc::vec![64], 256),
        )
        .unwrap();
        let y = b.rms_norm("x", "gamma", 1e-6);
        b.output(&y).unwrap();
        let built = b.build().unwrap();
        let node = &built.graph.nodes[0];
        assert_eq!(node.op_type, "RMSNormalization");
        assert_eq!(node.attributes.len(), 1);
        assert_eq!(node.attributes[0].name, "epsilon");
        assert!((node.attributes[0].f - 1e-6).abs() < 1e-12);
    }
}
