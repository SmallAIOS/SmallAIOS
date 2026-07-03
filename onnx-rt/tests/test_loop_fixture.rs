// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end fixture test for a synthetic ONNX model containing a `Loop`
//! operator with an embedded body `GraphProto`.
//!
//! This test exercises the full stack added by graph-attr-parser-v1:
//!   1. The protobuf parser decodes `AttributeProto.g` (field 6) into a
//!      nested `GraphProto`.
//!   2. The graph builder recursively compiles the body into an inner
//!      `ExecutionGraph` stored on the parent `ExecutionNode.inner_graphs`.
//!   3. The dispatcher reads the compiled body from `inner_graphs` and
//!      drives the sub-graph executor for the `Loop` op.
//!
//! The model is a minimal Loop that increments a float scalar accumulator
//! by 1.0 on each iteration. With M = 5 and an initial value of 0.0 the
//! final output should be 5.0.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use smallaios_onnx_rt::graph::build_execution_graph;
use smallaios_onnx_rt::onnx_types::{
    AttributeProto, AttributeType, GraphProto, ModelProto, NodeProto, OperatorSetIdProto,
    TensorProto, ValueInfoProto,
};
use smallaios_onnx_rt::session::{InferenceInput, Session, SessionConfig};
use smallaios_onnx_rt::sub_executor;
use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn f32_tensor(shape: &[i64], data: &[f32]) -> Tensor {
    let mut raw = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        smallaios_onnx_rt::byte_io::write_f32(&mut raw, i, v);
    }
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: raw,
    }
}

fn i64_tensor(shape: &[i64], data: &[i64]) -> Tensor {
    let mut raw = vec![0u8; data.len() * 8];
    for (i, &v) in data.iter().enumerate() {
        smallaios_onnx_rt::byte_io::write_i64(&mut raw, i, v);
    }
    Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: raw,
    }
}

fn bool_tensor(val: bool) -> Tensor {
    sub_executor::bool_scalar(val)
}

fn value_info(name: &str, elem_type: i32, shape: &[i64]) -> ValueInfoProto {
    ValueInfoProto {
        name: String::from(name),
        elem_type,
        shape: shape.to_vec(),
    }
}

fn graph_attr(name: &str, inner: GraphProto) -> AttributeProto {
    AttributeProto {
        name: String::from(name),
        attr_type: AttributeType::Graph,
        g: Some(alloc::boxed::Box::new(inner)),
        ..AttributeProto::default()
    }
}

fn one_constant_initializer() -> TensorProto {
    TensorProto {
        dims: vec![1],
        data_type: 1, // FLOAT
        name: String::from("one"),
        raw_data: 1.0f32.to_le_bytes().to_vec(),
        ..TensorProto::default()
    }
}

// ---------------------------------------------------------------------------
// Build the body graph: accumulator' = accumulator + one
//
// Body inputs:  iter_num (int64), cond_in (bool), accumulator (float[1])
// Body outputs: cond_out (bool), accumulator_out (float[1])
//
// The body always returns cond_out = true (loop only terminates when M
// iterations are reached).
// ---------------------------------------------------------------------------

fn build_loop_body() -> GraphProto {
    // Node 1: Identity on cond_in → cond_out (passthrough true)
    let cond_pass = NodeProto {
        op_type: String::from("Identity"),
        name: String::from("cond_pass"),
        input: vec![String::from("cond_in")],
        output: vec![String::from("cond_out")],
        ..NodeProto::default()
    };

    // Node 2: Add(accumulator, one) → accumulator_out
    let add = NodeProto {
        op_type: String::from("Add"),
        name: String::from("inc"),
        input: vec![String::from("accumulator"), String::from("one")],
        output: vec![String::from("accumulator_out")],
        ..NodeProto::default()
    };

    GraphProto {
        name: String::from("loop_body"),
        node: vec![cond_pass, add],
        input: vec![
            value_info("iter_num", 7, &[1]), // INT64
            value_info("cond_in", 9, &[1]),  // BOOL
            value_info("accumulator", 1, &[1]),
        ],
        output: vec![
            value_info("cond_out", 9, &[1]),
            value_info("accumulator_out", 1, &[1]),
        ],
        // `one` is supplied as an outer-scope initializer; NOT an input
        // to the body graph.
        initializer: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Build the top-level model
//
// Graph inputs:
//   M          (int64[1])  — iteration count
//   cond       (bool[1])   — initial condition
//   init_accum (float[1])  — initial accumulator
//
// Graph outputs:
//   result     (float[1])  — final accumulator after M iterations
//
// Nodes:
//   Loop { M, cond, init_accum -> result; body=loop_body }
//
// Initializers:
//   one = [1.0]
// ---------------------------------------------------------------------------

fn build_loop_model() -> ModelProto {
    let body = build_loop_body();

    let loop_node = NodeProto {
        op_type: String::from("Loop"),
        name: String::from("main_loop"),
        input: vec![
            String::from("M"),
            String::from("cond"),
            String::from("init_accum"),
        ],
        output: vec![String::from("result")],
        attribute: vec![graph_attr("body", body)],
        ..NodeProto::default()
    };

    let graph = GraphProto {
        name: String::from("main"),
        node: vec![loop_node],
        input: vec![
            value_info("M", 7, &[1]),
            value_info("cond", 9, &[1]),
            value_info("init_accum", 1, &[1]),
        ],
        output: vec![value_info("result", 1, &[1])],
        initializer: vec![one_constant_initializer()],
    };

    ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 21,
        }],
        producer_name: String::from("graph-attr-parser-v1-test"),
        graph: Some(graph),
        ..ModelProto::default()
    }
}

// ---------------------------------------------------------------------------
// Test 1: Graph builder produces inner_graphs for the Loop node
// ---------------------------------------------------------------------------

#[test]
fn graph_builder_compiles_loop_body() {
    let model = build_loop_model();
    let graph = model.graph.as_ref().unwrap();
    let exec = build_execution_graph(graph).unwrap();

    assert_eq!(exec.nodes.len(), 1);
    let loop_node = &exec.nodes[0];
    assert_eq!(loop_node.op_type, "Loop");
    assert_eq!(loop_node.inner_graphs.len(), 1);

    let body = loop_node
        .inner_graphs
        .get("body")
        .expect("body must be present");
    assert_eq!(body.node_count(), 2);
    assert_eq!(body.execution_order().len(), 2);
}

// ---------------------------------------------------------------------------
// Test 2: End-to-end via Session::initialize + Session::run
// ---------------------------------------------------------------------------

#[test]
fn loop_model_runs_end_to_end_via_session() {
    let model = build_loop_model();

    let mut session = Session::new(SessionConfig::default()).unwrap();
    session.initialize(&model).expect("initialize must succeed");

    let inputs = vec![
        InferenceInput {
            name: String::from("M"),
            tensor: i64_tensor(&[1], &[5]),
        },
        InferenceInput {
            name: String::from("cond"),
            tensor: bool_tensor(true),
        },
        InferenceInput {
            name: String::from("init_accum"),
            tensor: f32_tensor(&[1], &[0.0]),
        },
    ];

    let outputs = session.run(&inputs).expect("inference must succeed");
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0].name, "result");

    let result_val = smallaios_onnx_rt::byte_io::read_f32(&outputs[0].tensor.raw_data, 0);
    // 0 + 1 * 5 iterations = 5.0
    assert!(
        (result_val - 5.0).abs() < f32::EPSILON * 16.0,
        "expected 5.0, got {}",
        result_val
    );
}

// ---------------------------------------------------------------------------
// Test 3: Loop with M=0 returns initial accumulator
// ---------------------------------------------------------------------------

#[test]
fn loop_model_m_zero_returns_initial() {
    let model = build_loop_model();
    let mut session = Session::new(SessionConfig::default()).unwrap();
    session.initialize(&model).unwrap();

    let inputs = vec![
        InferenceInput {
            name: String::from("M"),
            tensor: i64_tensor(&[1], &[0]),
        },
        InferenceInput {
            name: String::from("cond"),
            tensor: bool_tensor(true),
        },
        InferenceInput {
            name: String::from("init_accum"),
            tensor: f32_tensor(&[1], &[42.0]),
        },
    ];

    let outputs = session.run(&inputs).unwrap();
    let val = smallaios_onnx_rt::byte_io::read_f32(&outputs[0].tensor.raw_data, 0);
    assert!(
        (val - 42.0).abs() < f32::EPSILON,
        "expected 42.0, got {}",
        val
    );
}

// ---------------------------------------------------------------------------
// Test 4: Model without control-flow ops still parses (regression)
// ---------------------------------------------------------------------------

#[test]
fn model_without_control_flow_still_parses() {
    let graph = GraphProto {
        name: String::from("simple"),
        node: vec![NodeProto {
            op_type: String::from("Add"),
            name: String::from("add0"),
            input: vec![String::from("a"), String::from("b")],
            output: vec![String::from("c")],
            ..NodeProto::default()
        }],
        input: vec![value_info("a", 1, &[2]), value_info("b", 1, &[2])],
        output: vec![value_info("c", 1, &[2])],
        ..GraphProto::default()
    };

    let exec = build_execution_graph(&graph).unwrap();
    assert_eq!(exec.nodes.len(), 1);
    assert!(exec.nodes[0].inner_graphs.is_empty());
}

// ---------------------------------------------------------------------------
// Test 5: If node dispatch selects the correct branch
// ---------------------------------------------------------------------------

#[test]
fn if_model_selects_correct_branch() {
    let then_graph = GraphProto {
        name: String::from("then"),
        node: vec![NodeProto {
            op_type: String::from("Relu"),
            name: String::from("r"),
            input: vec![String::from("x")],
            output: vec![String::from("y")],
            ..NodeProto::default()
        }],
        input: vec![value_info("x", 1, &[1])],
        output: vec![value_info("y", 1, &[1])],
        ..GraphProto::default()
    };
    let else_graph = GraphProto {
        name: String::from("else"),
        node: vec![NodeProto {
            op_type: String::from("Neg"),
            name: String::from("n"),
            input: vec![String::from("x")],
            output: vec![String::from("y")],
            ..NodeProto::default()
        }],
        input: vec![value_info("x", 1, &[1])],
        output: vec![value_info("y", 1, &[1])],
        ..GraphProto::default()
    };

    let if_node = NodeProto {
        op_type: String::from("If"),
        name: String::from("if0"),
        input: vec![String::from("cond")],
        output: vec![String::from("out")],
        attribute: vec![
            graph_attr("then_branch", then_graph),
            graph_attr("else_branch", else_graph),
        ],
        ..NodeProto::default()
    };

    let model = ModelProto {
        ir_version: 9,
        opset_import: vec![OperatorSetIdProto {
            domain: String::new(),
            version: 21,
        }],
        graph: Some(GraphProto {
            name: String::from("if_test"),
            node: vec![if_node],
            input: vec![value_info("cond", 9, &[1]), value_info("x", 1, &[1])],
            output: vec![value_info("out", 1, &[1])],
            ..GraphProto::default()
        }),
        ..ModelProto::default()
    };

    let mut session = Session::new(SessionConfig::default()).unwrap();
    session.initialize(&model).unwrap();

    // cond = true → then_branch → Relu(-5) = 0
    let outputs = session
        .run(&[
            InferenceInput {
                name: String::from("cond"),
                tensor: bool_tensor(true),
            },
            InferenceInput {
                name: String::from("x"),
                tensor: f32_tensor(&[1], &[-5.0]),
            },
        ])
        .unwrap();
    let val = smallaios_onnx_rt::byte_io::read_f32(&outputs[0].tensor.raw_data, 0);
    assert!(
        (val - 0.0).abs() < f32::EPSILON,
        "then_branch: expected 0.0, got {}",
        val
    );

    // cond = false → else_branch → Neg(-5) = 5
    let outputs = session
        .run(&[
            InferenceInput {
                name: String::from("cond"),
                tensor: bool_tensor(false),
            },
            InferenceInput {
                name: String::from("x"),
                tensor: f32_tensor(&[1], &[-5.0]),
            },
        ])
        .unwrap();
    let val = smallaios_onnx_rt::byte_io::read_f32(&outputs[0].tensor.raw_data, 0);
    assert!(
        (val - 5.0).abs() < f32::EPSILON,
        "else_branch: expected 5.0, got {}",
        val
    );
}
