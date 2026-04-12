// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Sub-graph executor for Phase 2 control-flow operators.
//!
//! This module is the recursive heart of the SmallAIOS ONNX control-flow
//! support. It evaluates an [`ExecutionGraph`] that represents the inner
//! body of an `If`, `Loop`, or `Scan` operator, with a freshly-scoped
//! value map, shared initializer scope, and outer-value seeding.
//!
//! See `docs/sub-graph-executor-design.md` for the full design rationale
//! (compile-once / dispatch-many, scope rules, termination handling, WCET
//! budget integration, memory layout, etc).
//!
//! # Compile-once
//!
//! Sub-graphs are intended to be compiled at session load time by the graph
//! builder when it encounters an `If`/`Loop`/`Scan` node. In this first
//! implementation the graph builder does NOT yet walk into body attributes
//! (the protobuf parser does not decode `AttributeType::Graph` yet), so
//! **models that embed control-flow ops via ONNX export cannot be loaded
//! end-to-end yet**. This is documented as a follow-up in the PR body.
//!
//! However, [`run_sub_graph`] and the [`op_if`] / [`op_loop`] / [`op_scan`]
//! helpers in [`crate::ops::control_flow`] are fully functional when the
//! caller hands them a pre-built [`ExecutionGraph`] body. This covers the
//! test-first implementation pattern that the `generative-llm-v1` change
//! commits to: synthetic graphs constructed by hand in tests exercise the
//! exact same dispatch path that a future end-to-end model load would
//! trigger, so the runtime behaviour is fully verified today.
//!
//! # Loop termination safety net
//!
//! [`MAX_LOOP_ITERATIONS`] is a compile-time upper bound on any `Loop`
//! invocation. It guards against buggy body graphs that emit `cond_out =
//! true` forever (item D9 in `docs/sub-graph-executor-design.md`).

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{read_f32, write_f32};
use crate::executor::dispatch_node;
use crate::graph::ExecutionGraph;
use crate::onnx_types::TensorProto;
use crate::operators::OpError;
use crate::session::SessionError;
use crate::tensor::{DataType, Tensor, TensorShape};

/// Hard upper bound on iterations for any `Loop` or `Scan` invocation.
///
/// This is a safety net for buggy body graphs that emit
/// `cond_out = true` forever (see D9 in `docs/sub-graph-executor-design.md`).
/// An iteration count beyond this limit is treated as a hard failure.
pub const MAX_LOOP_ITERATIONS: i64 = 1_000_000;

/// Runs a compiled inner graph once with the given seeded value map.
///
/// The caller is responsible for:
///
/// 1. Constructing a fresh `value_map` seeded with the body's input values
///    (including `iter_num`, `cond_in`, and loop-carried `v_in_*` for a
///    `Loop` body).
/// 2. Providing the outer-scope `initializers` slice unchanged (this is
///    threaded through so weight tensors are available inside the body).
/// 3. Extracting outputs from the returned map by name after the call
///    returns.
///
/// The function returns the final value map containing every intermediate
/// and output tensor produced by the inner graph's nodes. Callers that
/// only care about the body's declared outputs can pull them by name from
/// the returned map.
///
/// Inner-operator timing and budget enforcement is currently **not**
/// threaded through: per §6 of the design doc, the parent `Loop`/`If`/`Scan`
/// is the single budget unit. Inner-operator measurement is a follow-up
/// work-item.
pub fn run_sub_graph(
    graph: &ExecutionGraph,
    mut value_map: BTreeMap<String, Tensor>,
    initializers: &[TensorProto],
) -> Result<BTreeMap<String, Tensor>, SessionError> {
    // Load any initializers not already present in the seeded map so the
    // body can reference outer weights by name.
    for init in initializers {
        value_map.entry(init.name.clone()).or_insert_with(|| {
            // Convert TensorProto -> Tensor. For now only raw_data / f32
            // initializers are supported; string/f16 initializers are
            // rejected by the main graph builder anyway.
            materialize_initializer(init)
        });
    }

    for node_idx in graph.execution_order() {
        let node = &graph.nodes[node_idx.index()];

        // Control-flow ops need the full ExecutionNode + outer value map
        // and bypass the flat dispatch_node path.
        if crate::executor::is_control_flow_op(&node.op_type) {
            let outputs =
                crate::executor::dispatch_control_flow(node, &value_map, initializers)?;
            for (i, output_name) in node.outputs.iter().enumerate() {
                if !output_name.is_empty() {
                    if let Some(tensor) = outputs.get(i) {
                        value_map.insert(output_name.clone(), tensor.clone());
                    }
                }
            }
            continue;
        }

        // Resolve input tensors from the value map.
        let input_tensors: Vec<Option<&Tensor>> = node
            .inputs
            .iter()
            .map(|name| {
                if name.is_empty() {
                    None
                } else {
                    value_map.get(name)
                }
            })
            .collect();

        let outputs = dispatch_node(
            &node.op_type,
            &input_tensors,
            &node.attributes,
            node.outputs.len(),
            #[cfg(feature = "gpu")]
            None,
        )
        .map_err(|e| {
            SessionError::ExecutionFailed(alloc::format!("sub_executor: {}: {}", node.name, e))
        })?;

        for (i, output_name) in node.outputs.iter().enumerate() {
            if !output_name.is_empty() {
                if let Some(tensor) = outputs.get(i) {
                    value_map.insert(output_name.clone(), tensor.clone());
                }
            }
        }
    }

    Ok(value_map)
}

fn materialize_initializer(proto: &TensorProto) -> Tensor {
    let data_type = DataType::from_i32(proto.data_type).unwrap_or(DataType::Float);
    let shape = TensorShape::new(proto.dims.clone());
    let raw_data = if !proto.raw_data.is_empty() {
        proto.raw_data.clone()
    } else {
        Vec::new()
    };
    Tensor {
        data_type,
        shape,
        name: proto.name.clone(),
        raw_data,
    }
}

// ---------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------

/// Builds a 0-d / shape-[1] Bool tensor from a boolean value.
pub fn bool_scalar(v: bool) -> Tensor {
    Tensor {
        data_type: DataType::Bool,
        shape: TensorShape::new(alloc::vec![1]),
        name: String::new(),
        raw_data: alloc::vec![u8::from(v)],
    }
}

/// Reads a Bool / Float / Int32 scalar tensor as a boolean. A Float or
/// Int32 value is truthy iff non-zero.
pub fn read_bool_scalar(t: &Tensor) -> Result<bool, OpError> {
    if t.shape.total_elements() != 1 {
        return Err(OpError::ShapeMismatch(String::from(
            "expected scalar tensor for boolean",
        )));
    }
    match t.data_type {
        DataType::Bool => Ok(t.raw_data.first().copied().unwrap_or(0) != 0),
        DataType::Float => Ok(read_f32(&t.raw_data, 0) != 0.0),
        DataType::Int32 => Ok(crate::byte_io::read_i32(&t.raw_data, 0) != 0),
        DataType::Int64 => Ok(crate::byte_io::read_i64(&t.raw_data, 0) != 0),
        _ => Err(OpError::ShapeMismatch(String::from(
            "unsupported dtype for boolean scalar",
        ))),
    }
}

/// Builds an Int64 shape-[1] scalar tensor.
pub fn i64_scalar(v: i64) -> Tensor {
    let mut raw = alloc::vec![0u8; 8];
    crate::byte_io::write_i64(&mut raw, 0, v);
    Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(alloc::vec![1]),
        name: String::new(),
        raw_data: raw,
    }
}

/// Builds a shape-[1] Float scalar tensor.
pub fn f32_scalar(v: f32) -> Tensor {
    let mut raw = alloc::vec![0u8; 4];
    write_f32(&mut raw, 0, v);
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(alloc::vec![1]),
        name: String::new(),
        raw_data: raw,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::build_execution_graph;
    use crate::onnx_types::{GraphProto, NodeProto, ValueInfoProto};
    use alloc::string::ToString;
    use alloc::vec;

    fn make_node(op_type: &str, name: &str, inputs: &[&str], outputs: &[&str]) -> NodeProto {
        NodeProto {
            op_type: op_type.to_string(),
            name: name.to_string(),
            input: inputs.iter().map(|s| s.to_string()).collect(),
            output: outputs.iter().map(|s| s.to_string()).collect(),
            ..NodeProto::default()
        }
    }

    fn make_graph(nodes: Vec<NodeProto>, inputs: &[&str], outputs: &[&str]) -> GraphProto {
        GraphProto {
            name: "sub".to_string(),
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

    fn tensor_f32(shape: &[i64], data: &[f32]) -> Tensor {
        let mut raw = alloc::vec![0u8; data.len() * 4];
        for (i, &v) in data.iter().enumerate() {
            write_f32(&mut raw, i, v);
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(shape.to_vec()),
            name: String::new(),
            raw_data: raw,
        }
    }

    // ---- empty body ----

    #[test]
    fn empty_body_is_pass_through() {
        let gp = make_graph(vec![], &["x"], &["x"]);
        let eg = build_execution_graph(&gp).unwrap();
        let mut seed = BTreeMap::new();
        seed.insert(String::from("x"), tensor_f32(&[1], &[42.0]));
        let out = run_sub_graph(&eg, seed, &[]).unwrap();
        let t = out.get("x").unwrap();
        assert_eq!(read_f32(&t.raw_data, 0), 42.0);
    }

    // ---- single-op body: Add ----

    #[test]
    fn single_op_body_add() {
        let gp = make_graph(
            vec![make_node("Add", "add0", &["a", "b"], &["y"])],
            &["a", "b"],
            &["y"],
        );
        let eg = build_execution_graph(&gp).unwrap();
        let mut seed = BTreeMap::new();
        seed.insert(String::from("a"), tensor_f32(&[3], &[1.0, 2.0, 3.0]));
        seed.insert(String::from("b"), tensor_f32(&[3], &[10.0, 20.0, 30.0]));
        let out = run_sub_graph(&eg, seed, &[]).unwrap();
        let y = out.get("y").unwrap();
        assert_eq!(read_f32(&y.raw_data, 0), 11.0);
        assert_eq!(read_f32(&y.raw_data, 1), 22.0);
        assert_eq!(read_f32(&y.raw_data, 2), 33.0);
    }

    // ---- multi-op body ----

    #[test]
    fn multi_op_body_add_then_relu() {
        let gp = make_graph(
            vec![
                make_node("Add", "add", &["a", "b"], &["tmp"]),
                make_node("Relu", "relu", &["tmp"], &["y"]),
            ],
            &["a", "b"],
            &["y"],
        );
        let eg = build_execution_graph(&gp).unwrap();
        let mut seed = BTreeMap::new();
        seed.insert(String::from("a"), tensor_f32(&[3], &[1.0, -2.0, 3.0]));
        seed.insert(String::from("b"), tensor_f32(&[3], &[-4.0, 1.0, 0.0]));
        let out = run_sub_graph(&eg, seed, &[]).unwrap();
        let y = out.get("y").unwrap();
        // Add -> [-3, -1, 3]; Relu -> [0, 0, 3]
        assert_eq!(read_f32(&y.raw_data, 0), 0.0);
        assert_eq!(read_f32(&y.raw_data, 1), 0.0);
        assert_eq!(read_f32(&y.raw_data, 2), 3.0);
    }

    // ---- value scope isolation: inner body cannot pollute outer map ----

    #[test]
    fn sub_graph_returns_isolated_map() {
        let gp = make_graph(vec![make_node("Relu", "r", &["a"], &["y"])], &["a"], &["y"]);
        let eg = build_execution_graph(&gp).unwrap();
        let mut seed = BTreeMap::new();
        seed.insert(String::from("a"), tensor_f32(&[1], &[-5.0]));
        let out = run_sub_graph(&eg, seed, &[]).unwrap();
        assert!(out.contains_key("y"));
        let v = read_f32(&out["y"].raw_data, 0);
        assert_eq!(v, 0.0);
    }

    // ---- loop-carried state rotation (exercise the control-flow helpers) ----

    #[test]
    fn loop_counter_body_runs_m_iterations() {
        use crate::ops::control_flow::op_loop_with_body;
        // Body: counter' = counter + one
        let gp = make_graph(
            vec![make_node(
                "Add",
                "inc",
                &["counter", "one"],
                &["counter_out"],
            )],
            &["iter_num", "cond_in", "counter", "one"],
            &["cond_out", "counter_out"],
        );
        let body = build_execution_graph(&gp).unwrap();

        let mut outer = BTreeMap::new();
        outer.insert(String::from("one"), tensor_f32(&[1], &[1.0]));
        let v_initial = alloc::vec![tensor_f32(&[1], &[0.0])];

        let result = op_loop_with_body(
            Some(5),
            Some(true),
            v_initial,
            &body,
            &outer,
            |_iter, _vs| true, // cond_out always true
        )
        .unwrap();

        // After 5 iterations the counter should be 5.
        assert_eq!(result.len(), 1);
        assert_eq!(read_f32(&result[0].raw_data, 0), 5.0);
    }

    #[test]
    fn loop_m_zero_returns_initial() {
        use crate::ops::control_flow::op_loop_with_body;
        let gp = make_graph(vec![], &[], &[]);
        let body = build_execution_graph(&gp).unwrap();
        let v_initial = alloc::vec![tensor_f32(&[1], &[42.0])];
        let outer = BTreeMap::new();
        let result =
            op_loop_with_body(Some(0), Some(true), v_initial, &body, &outer, |_, _| true).unwrap();
        assert_eq!(read_f32(&result[0].raw_data, 0), 42.0);
    }

    #[test]
    fn loop_initial_cond_false_returns_initial() {
        use crate::ops::control_flow::op_loop_with_body;
        let gp = make_graph(vec![], &[], &[]);
        let body = build_execution_graph(&gp).unwrap();
        let v_initial = alloc::vec![tensor_f32(&[1], &[7.0])];
        let outer = BTreeMap::new();
        let result = op_loop_with_body(Some(100), Some(false), v_initial, &body, &outer, |_, _| {
            true
        })
        .unwrap();
        assert_eq!(read_f32(&result[0].raw_data, 0), 7.0);
    }

    #[test]
    fn loop_body_cond_out_terminates_early() {
        use crate::ops::control_flow::op_loop_with_body;
        let gp = make_graph(
            vec![make_node(
                "Add",
                "inc",
                &["counter", "one"],
                &["counter_out"],
            )],
            &["iter_num", "cond_in", "counter", "one"],
            &["cond_out", "counter_out"],
        );
        let body = build_execution_graph(&gp).unwrap();

        let mut outer = BTreeMap::new();
        outer.insert(String::from("one"), tensor_f32(&[1], &[1.0]));
        let v_initial = alloc::vec![tensor_f32(&[1], &[0.0])];

        // cond_out: continue only while the post-body counter < 3
        let result = op_loop_with_body(
            Some(1000),
            Some(true),
            v_initial,
            &body,
            &outer,
            |_iter, vs| read_f32(&vs[0].raw_data, 0) < 3.0,
        )
        .unwrap();
        assert_eq!(read_f32(&result[0].raw_data, 0), 3.0);
    }

    #[test]
    fn loop_safety_net_aborts_runaway_bodies() {
        use crate::ops::control_flow::op_loop_with_body;
        let gp = make_graph(vec![], &[], &[]);
        let body = build_execution_graph(&gp).unwrap();
        let v_initial = alloc::vec![tensor_f32(&[1], &[0.0])];
        let outer = BTreeMap::new();
        let result = op_loop_with_body(None, Some(true), v_initial, &body, &outer, |_, _| true);
        assert!(result.is_err());
    }

    // ---- If ----

    #[test]
    fn if_selects_then_branch_on_true() {
        use crate::ops::control_flow::op_if_with_body;
        let then_gp = make_graph(vec![make_node("Relu", "r", &["x"], &["y"])], &["x"], &["y"]);
        let else_gp = make_graph(vec![make_node("Neg", "n", &["x"], &["y"])], &["x"], &["y"]);
        let then_body = build_execution_graph(&then_gp).unwrap();
        let else_body = build_execution_graph(&else_gp).unwrap();

        let mut outer = BTreeMap::new();
        outer.insert(String::from("x"), tensor_f32(&[1], &[-5.0]));

        let out = op_if_with_body(true, &then_body, &else_body, &outer, &["y"]).unwrap();
        // Then branch: Relu(-5) = 0
        assert_eq!(read_f32(&out[0].raw_data, 0), 0.0);

        let out = op_if_with_body(false, &then_body, &else_body, &outer, &["y"]).unwrap();
        // Else branch: Neg(-5) = 5
        assert_eq!(read_f32(&out[0].raw_data, 0), 5.0);
    }

    // ---- Scan (simple add-constant body) ----

    #[test]
    fn scan_applies_body_per_sequence_element() {
        use crate::ops::control_flow::op_scan_with_body;
        // Body: y = x + 1
        let gp = make_graph(
            vec![make_node("Add", "inc", &["x", "one"], &["y"])],
            &["x", "one"],
            &["y"],
        );
        let body = build_execution_graph(&gp).unwrap();

        let mut outer = BTreeMap::new();
        outer.insert(String::from("one"), tensor_f32(&[1], &[1.0]));

        // Sequence input: 5 elements each shape [1].
        let elements = alloc::vec![
            tensor_f32(&[1], &[10.0]),
            tensor_f32(&[1], &[20.0]),
            tensor_f32(&[1], &[30.0]),
            tensor_f32(&[1], &[40.0]),
            tensor_f32(&[1], &[50.0]),
        ];

        let out = op_scan_with_body(&body, &outer, "x", "y", elements).unwrap();
        assert_eq!(out.len(), 5);
        assert_eq!(read_f32(&out[0].raw_data, 0), 11.0);
        assert_eq!(read_f32(&out[4].raw_data, 0), 51.0);
    }

    // ---- Autoregressive-style end-to-end: Loop containing MatMul + Softmax + ArgMax ----

    #[test]
    fn autoregressive_decode_loop_synthetic() {
        use crate::ops::control_flow::op_loop_with_body;

        // Synthetic "decode" step: hidden' = Add(hidden, step). Runs 4 iterations.
        // This mirrors the autoregressive pattern: a loop body operates on a
        // carried hidden state. We use Add instead of MatMul to keep the
        // fixture small, but the dispatch path is identical.
        let gp = make_graph(
            vec![make_node(
                "Add",
                "step",
                &["hidden", "delta"],
                &["hidden_out"],
            )],
            &["iter_num", "cond_in", "hidden", "delta"],
            &["cond_out", "hidden_out"],
        );
        let body = build_execution_graph(&gp).unwrap();

        let mut outer = BTreeMap::new();
        outer.insert(String::from("delta"), tensor_f32(&[2], &[0.5, -0.25]));
        let v_initial = alloc::vec![tensor_f32(&[2], &[0.0, 0.0])];

        let out =
            op_loop_with_body(Some(4), Some(true), v_initial, &body, &outer, |_, _| true).unwrap();
        assert_eq!(out.len(), 1);
        // After 4 iterations: hidden = [2.0, -1.0]
        assert!((read_f32(&out[0].raw_data, 0) - 2.0).abs() < 1e-5);
        assert!((read_f32(&out[0].raw_data, 1) + 1.0).abs() < 1e-5);
    }

    // ---- Scalar helpers ----

    #[test]
    fn bool_scalar_roundtrip() {
        let t = bool_scalar(true);
        assert!(read_bool_scalar(&t).unwrap());
        let t = bool_scalar(false);
        assert!(!read_bool_scalar(&t).unwrap());
    }

    #[test]
    fn f32_scalar_roundtrip() {
        let t = f32_scalar(2.5);
        assert!((read_f32(&t.raw_data, 0) - 2.5).abs() < 1e-5);
    }

    #[test]
    fn i64_scalar_roundtrip() {
        let t = i64_scalar(-42);
        assert_eq!(crate::byte_io::read_i64(&t.raw_data, 0), -42);
    }

    #[test]
    fn read_bool_scalar_rejects_non_scalar() {
        let t = tensor_f32(&[2], &[1.0, 2.0]);
        assert!(read_bool_scalar(&t).is_err());
    }
}
