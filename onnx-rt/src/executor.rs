// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph executor — traverses the execution graph in topological order,
//! dispatching each node to the corresponding CPU operator with tensor
//! I/O routing via a named tensor value map.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::graph::ExecutionGraph;
use crate::onnx_types::{AttributeProto, AttributeType, TensorProto};
use crate::operators::{self, OpError, OpKind};
use crate::session::{InferenceOutput, SessionError};
use crate::tensor::{DataType, Tensor, TensorShape};

/// Executes an ONNX graph end-to-end on CPU.
///
/// Iterates the topologically-sorted execution order, resolving each node's
/// inputs from `value_map`, dispatching to the corresponding operator, and
/// storing outputs back into the map. After all nodes execute, the graph's
/// declared outputs are extracted and returned.
pub fn execute_graph(
    graph: &ExecutionGraph,
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    yield_fn: Option<fn()>,
) -> Result<Vec<InferenceOutput>, SessionError> {
    let mut value_map: BTreeMap<String, Tensor> = BTreeMap::new();

    // Load user-provided inputs
    for (name, tensor) in inputs {
        value_map.insert(name.clone(), tensor.clone());
    }

    // Load initializer tensors (model weights)
    for init in initializers {
        if let Some(tensor) = tensor_from_proto(init) {
            value_map.insert(init.name.clone(), tensor);
        }
    }

    // Execute nodes in topological order
    for node_idx in graph.execution_order() {
        let node = &graph.nodes[node_idx.index()];

        // Resolve input tensors
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

        // Dispatch operator
        let outputs = dispatch_node(&node.op_type, &input_tensors, &node.attributes)
            .map_err(|e| SessionError::ExecutionFailed(alloc::format!("{}: {}", node.name, e)))?;

        // Store outputs in value map
        for (i, output_name) in node.outputs.iter().enumerate() {
            if !output_name.is_empty() {
                if let Some(tensor) = outputs.get(i) {
                    value_map.insert(output_name.clone(), tensor.clone());
                }
            }
        }

        // Yield to scheduler if callback provided
        if let Some(f) = yield_fn {
            f();
        }
    }

    // Extract graph outputs
    let mut results = Vec::new();
    for output_name in &graph.output_names {
        let tensor = value_map.remove(output_name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!(
                "output tensor '{}' not produced",
                output_name
            ))
        })?;
        results.push(InferenceOutput {
            name: output_name.clone(),
            tensor,
        });
    }

    Ok(results)
}

/// Converts a TensorProto (model initializer) to a runtime Tensor.
fn tensor_from_proto(proto: &TensorProto) -> Option<Tensor> {
    let data_type = DataType::from_i32(proto.data_type)?;
    let shape = TensorShape::new(proto.dims.clone());

    let raw_data = if !proto.raw_data.is_empty() {
        proto.raw_data.clone()
    } else if !proto.float_data.is_empty() {
        let mut bytes = alloc::vec![0u8; proto.float_data.len() * 4];
        for (i, &val) in proto.float_data.iter().enumerate() {
            let b = val.to_le_bytes();
            bytes[i * 4] = b[0];
            bytes[i * 4 + 1] = b[1];
            bytes[i * 4 + 2] = b[2];
            bytes[i * 4 + 3] = b[3];
        }
        bytes
    } else if !proto.int64_data.is_empty() {
        let mut bytes = alloc::vec![0u8; proto.int64_data.len() * 8];
        for (i, &val) in proto.int64_data.iter().enumerate() {
            let b = val.to_le_bytes();
            for j in 0..8 {
                bytes[i * 8 + j] = b[j];
            }
        }
        bytes
    } else if !proto.int32_data.is_empty() {
        let mut bytes = alloc::vec![0u8; proto.int32_data.len() * 4];
        for (i, &val) in proto.int32_data.iter().enumerate() {
            let b = val.to_le_bytes();
            for j in 0..4 {
                bytes[i * 4 + j] = b[j];
            }
        }
        bytes
    } else {
        Vec::new()
    };

    Some(Tensor {
        data_type,
        shape,
        name: proto.name.clone(),
        raw_data,
    })
}

// ---------------------------------------------------------------------------
// Attribute helpers
// ---------------------------------------------------------------------------

/// Extracts an integer attribute by name, returning a default if not found.
fn get_attr_int(attrs: &[AttributeProto], name: &str, default: i64) -> i64 {
    attrs
        .iter()
        .find(|a| a.name == name && a.attr_type == AttributeType::Int)
        .map(|a| a.i)
        .unwrap_or(default)
}

/// Extracts a float attribute by name, returning a default if not found.
fn get_attr_float(attrs: &[AttributeProto], name: &str, default: f32) -> f32 {
    attrs
        .iter()
        .find(|a| a.name == name && a.attr_type == AttributeType::Float)
        .map(|a| a.f)
        .unwrap_or(default)
}

/// Extracts an integer list attribute by name.
fn get_attr_ints<'a>(attrs: &'a [AttributeProto], name: &str) -> Option<&'a [i64]> {
    attrs
        .iter()
        .find(|a| a.name == name && a.attr_type == AttributeType::Ints)
        .map(|a| a.ints.as_slice())
}

// ---------------------------------------------------------------------------
// Operator dispatch
// ---------------------------------------------------------------------------

/// Dispatches a single graph node to the appropriate CPU operator function.
///
/// Matches the node's `op_type` to an `OpKind` and calls the corresponding
/// `op_*` function with the resolved input tensors and attributes.
/// Returns a vector of output tensors (most operators produce exactly one).
fn dispatch_node(
    op_type: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<Tensor>, OpError> {
    let kind =
        OpKind::parse_str(op_type).ok_or_else(|| OpError::UnsupportedOp(String::from(op_type)))?;

    let result = match kind {
        // Arithmetic
        OpKind::Add => {
            let refs = require_inputs(inputs, 2, "Add")?;
            operators::op_add(&refs)
        }
        OpKind::Sub => {
            let refs = require_inputs(inputs, 2, "Sub")?;
            operators::op_sub(&refs)
        }
        OpKind::Mul => {
            let refs = require_inputs(inputs, 2, "Mul")?;
            operators::op_mul(&refs)
        }
        OpKind::Div => {
            let refs = require_inputs(inputs, 2, "Div")?;
            operators::op_div(&refs)
        }
        OpKind::MatMul => {
            let refs = require_inputs(inputs, 2, "MatMul")?;
            operators::op_matmul(refs[0], refs[1])
        }
        OpKind::Gemm => {
            let a = require_input(inputs, 0, "Gemm")?;
            let b = require_input(inputs, 1, "Gemm")?;
            let c = optional_input(inputs, 2);
            let alpha = get_attr_float(attrs, "alpha", 1.0);
            let beta = get_attr_float(attrs, "beta", 1.0);
            let trans_a = get_attr_int(attrs, "transA", 0) != 0;
            let trans_b = get_attr_int(attrs, "transB", 0) != 0;
            operators::op_gemm(a, b, c, alpha, beta, trans_a, trans_b)
        }

        // Convolution
        OpKind::Conv => {
            let input = require_input(inputs, 0, "Conv")?;
            let weight = require_input(inputs, 1, "Conv")?;
            let bias = optional_input(inputs, 2);
            operators::op_conv(input, weight, bias)
        }

        // Activations
        OpKind::Relu => {
            let t = require_input(inputs, 0, "Relu")?;
            operators::op_relu(t)
        }
        OpKind::Sigmoid => {
            let t = require_input(inputs, 0, "Sigmoid")?;
            operators::op_sigmoid(t)
        }
        OpKind::Tanh => {
            let t = require_input(inputs, 0, "Tanh")?;
            operators::op_tanh(t)
        }
        OpKind::Softmax => {
            let t = require_input(inputs, 0, "Softmax")?;
            let axis = get_attr_int(attrs, "axis", -1);
            operators::op_softmax(t, axis)
        }

        // Shape manipulation
        OpKind::Reshape => {
            let t = require_input(inputs, 0, "Reshape")?;
            // Shape tensor is the second input
            let shape_tensor = require_input(inputs, 1, "Reshape")?;
            let shape = read_i64_tensor(shape_tensor);
            operators::op_reshape(t, &shape)
        }
        OpKind::Transpose => {
            let t = require_input(inputs, 0, "Transpose")?;
            let perm = get_attr_ints(attrs, "perm");
            operators::op_transpose(t, perm)
        }
        OpKind::Flatten => {
            let t = require_input(inputs, 0, "Flatten")?;
            let axis = get_attr_int(attrs, "axis", 1);
            operators::op_flatten(t, axis)
        }
        OpKind::Squeeze => {
            let t = require_input(inputs, 0, "Squeeze")?;
            // In opset 13+, axes come from second input tensor
            let axes_tensor = optional_input(inputs, 1);
            let axes = axes_tensor.map(read_i64_tensor);
            operators::op_squeeze(t, axes.as_deref())
        }
        OpKind::Unsqueeze => {
            let t = require_input(inputs, 0, "Unsqueeze")?;
            // In opset 13+, axes come from second input tensor
            let axes_tensor = require_input(inputs, 1, "Unsqueeze")?;
            let axes = read_i64_tensor(axes_tensor);
            operators::op_unsqueeze(t, &axes)
        }
        OpKind::Concat => {
            let refs: Vec<&Tensor> = inputs.iter().filter_map(|i| *i).collect();
            if refs.is_empty() {
                return Err(OpError::ShapeMismatch(String::from(
                    "Concat requires at least 1 input",
                )));
            }
            let axis = get_attr_int(attrs, "axis", 0);
            operators::op_concat(&refs, axis)
        }
        OpKind::Gather => {
            let t = require_input(inputs, 0, "Gather")?;
            let indices = require_input(inputs, 1, "Gather")?;
            let axis = get_attr_int(attrs, "axis", 0);
            operators::op_gather(t, indices, axis)
        }
        OpKind::Slice => {
            let t = require_input(inputs, 0, "Slice")?;
            let starts_tensor = require_input(inputs, 1, "Slice")?;
            let ends_tensor = require_input(inputs, 2, "Slice")?;
            let axes_tensor = optional_input(inputs, 3);
            let steps_tensor = optional_input(inputs, 4);
            let starts = read_i64_tensor(starts_tensor);
            let ends = read_i64_tensor(ends_tensor);
            let axes = axes_tensor.map(read_i64_tensor);
            let steps = steps_tensor.map(read_i64_tensor);
            operators::op_slice(t, &starts, &ends, axes.as_deref(), steps.as_deref())
        }
        OpKind::Pad => {
            let t = require_input(inputs, 0, "Pad")?;
            let pads_tensor = require_input(inputs, 1, "Pad")?;
            let constant_tensor = optional_input(inputs, 2);
            let pads = read_i64_tensor(pads_tensor);
            let mode_bytes = attrs
                .iter()
                .find(|a| a.name == "mode")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"constant");
            let mode_str = core::str::from_utf8(mode_bytes).unwrap_or("constant");
            let constant_value = constant_tensor
                .and_then(|ct| {
                    if ct.raw_data.len() >= 4 {
                        Some(f32::from_le_bytes([
                            ct.raw_data[0],
                            ct.raw_data[1],
                            ct.raw_data[2],
                            ct.raw_data[3],
                        ]))
                    } else {
                        None
                    }
                })
                .unwrap_or(0.0);
            operators::op_pad(t, &pads, mode_str, constant_value)
        }
        OpKind::Clip => {
            let t = require_input(inputs, 0, "Clip")?;
            let min_tensor = optional_input(inputs, 1);
            let max_tensor = optional_input(inputs, 2);
            let min_val = min_tensor.and_then(|mt| {
                if mt.raw_data.len() >= 4 {
                    Some(f32::from_le_bytes([
                        mt.raw_data[0],
                        mt.raw_data[1],
                        mt.raw_data[2],
                        mt.raw_data[3],
                    ]))
                } else {
                    None
                }
            });
            let max_val = max_tensor.and_then(|mt| {
                if mt.raw_data.len() >= 4 {
                    Some(f32::from_le_bytes([
                        mt.raw_data[0],
                        mt.raw_data[1],
                        mt.raw_data[2],
                        mt.raw_data[3],
                    ]))
                } else {
                    None
                }
            });
            operators::op_clip(t, min_val, max_val)
        }
        OpKind::Cast => {
            let t = require_input(inputs, 0, "Cast")?;
            let to_type = get_attr_int(attrs, "to", 1);
            let target = DataType::from_i32(to_type as i32).ok_or_else(|| {
                OpError::InvalidAttribute(String::from("unsupported cast target type"))
            })?;
            operators::op_cast(t, target)
        }

        // Normalization
        OpKind::BatchNormalization => {
            let x = require_input(inputs, 0, "BatchNormalization")?;
            let scale = require_input(inputs, 1, "BatchNormalization")?;
            let bias = require_input(inputs, 2, "BatchNormalization")?;
            let mean = require_input(inputs, 3, "BatchNormalization")?;
            let var = require_input(inputs, 4, "BatchNormalization")?;
            let epsilon = get_attr_float(attrs, "epsilon", 1e-5);
            operators::op_batch_normalization(x, scale, bias, mean, var, epsilon)
        }
        OpKind::LayerNormalization => {
            let x = require_input(inputs, 0, "LayerNormalization")?;
            let scale = require_input(inputs, 1, "LayerNormalization")?;
            let bias = optional_input(inputs, 2);
            let axis = get_attr_int(attrs, "axis", -1);
            let epsilon = get_attr_float(attrs, "epsilon", 1e-5);
            operators::op_layer_normalization(x, scale, bias, axis, epsilon)
        }

        // Pooling
        OpKind::MaxPool => {
            let x = require_input(inputs, 0, "MaxPool")?;
            let kernel_shape = get_attr_ints(attrs, "kernel_shape").ok_or_else(|| {
                OpError::InvalidAttribute(String::from("MaxPool requires kernel_shape"))
            })?;
            let strides = get_attr_ints(attrs, "strides");
            let pads = get_attr_ints(attrs, "pads");
            operators::op_maxpool(x, kernel_shape, strides, pads)
        }
        OpKind::AveragePool => {
            let x = require_input(inputs, 0, "AveragePool")?;
            let kernel_shape = get_attr_ints(attrs, "kernel_shape").ok_or_else(|| {
                OpError::InvalidAttribute(String::from("AveragePool requires kernel_shape"))
            })?;
            let strides = get_attr_ints(attrs, "strides");
            let pads = get_attr_ints(attrs, "pads");
            operators::op_averagepool(x, kernel_shape, strides, pads)
        }
        OpKind::GlobalAveragePool => {
            let x = require_input(inputs, 0, "GlobalAveragePool")?;
            operators::op_global_average_pool(x)
        }

        // Reduction
        OpKind::ReduceMean => {
            let x = require_input(inputs, 0, "ReduceMean")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            operators::op_reduce_mean(x, axes, keepdims)
        }
        OpKind::ReduceSum => {
            let x = require_input(inputs, 0, "ReduceSum")?;
            // Opset 13+: axes from second input; older: from attribute
            let axes_from_input = optional_input(inputs, 1).map(read_i64_tensor);
            let axes_attr = get_attr_ints(attrs, "axes");
            let axes = axes_from_input.as_deref().or(axes_attr).unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            operators::op_reduce_sum(x, axes, keepdims)
        }
    };

    result.map(|t| alloc::vec![t])
}

// ---------------------------------------------------------------------------
// Input resolution helpers
// ---------------------------------------------------------------------------

/// Requires that input at `index` exists and returns a reference.
fn require_input<'a>(
    inputs: &[Option<&'a Tensor>],
    index: usize,
    op_name: &str,
) -> Result<&'a Tensor, OpError> {
    inputs.get(index).and_then(|opt| *opt).ok_or_else(|| {
        OpError::ShapeMismatch(alloc::format!(
            "{} missing required input at index {}",
            op_name,
            index
        ))
    })
}

/// Returns the input at `index` if it exists, or None.
fn optional_input<'a>(inputs: &[Option<&'a Tensor>], index: usize) -> Option<&'a Tensor> {
    inputs.get(index).and_then(|opt| *opt)
}

/// Requires exactly `count` non-None inputs and returns refs.
fn require_inputs<'a>(
    inputs: &[Option<&'a Tensor>],
    count: usize,
    op_name: &str,
) -> Result<Vec<&'a Tensor>, OpError> {
    let refs: Vec<&Tensor> = inputs.iter().take(count).filter_map(|i| *i).collect();
    if refs.len() != count {
        return Err(OpError::ShapeMismatch(alloc::format!(
            "{} requires {} inputs, got {}",
            op_name,
            count,
            refs.len()
        )));
    }
    Ok(refs)
}

/// Reads an i64 tensor's raw_data as a Vec<i64>.
fn read_i64_tensor(tensor: &Tensor) -> Vec<i64> {
    if tensor.data_type == DataType::Int64 && tensor.raw_data.len() >= 8 {
        let count = tensor.raw_data.len() / 8;
        (0..count)
            .map(|i| {
                let off = i * 8;
                i64::from_le_bytes([
                    tensor.raw_data[off],
                    tensor.raw_data[off + 1],
                    tensor.raw_data[off + 2],
                    tensor.raw_data[off + 3],
                    tensor.raw_data[off + 4],
                    tensor.raw_data[off + 5],
                    tensor.raw_data[off + 6],
                    tensor.raw_data[off + 7],
                ])
            })
            .collect()
    } else {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    extern crate std;
    use super::*;
    use crate::graph::build_execution_graph;
    use crate::onnx_types::{GraphProto, NodeProto, ValueInfoProto};
    use alloc::string::ToString;
    use alloc::vec;

    /// Helper: create a float tensor with data.
    fn make_f32_tensor(name: &str, shape: &[i64], data: &[f32]) -> Tensor {
        let mut raw_data = alloc::vec![0u8; data.len() * 4];
        for (i, &val) in data.iter().enumerate() {
            let bytes = val.to_le_bytes();
            raw_data[i * 4] = bytes[0];
            raw_data[i * 4 + 1] = bytes[1];
            raw_data[i * 4 + 2] = bytes[2];
            raw_data[i * 4 + 3] = bytes[3];
        }
        Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(shape.to_vec()),
            name: String::from(name),
            raw_data,
        }
    }

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
            name: "test".to_string(),
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

    #[test]
    fn test_execute_relu_graph() {
        // Single Relu node: input "x" → output "y"
        let graph_proto = make_graph(
            vec![make_node("Relu", "relu0", &["x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 4], &[-1.0, 0.0, 1.0, 2.0]);
        let inputs = vec![("x".to_string(), input)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "y");

        let out_data: Vec<f32> = (0..4)
            .map(|i| {
                f32::from_le_bytes([
                    results[0].tensor.raw_data[i * 4],
                    results[0].tensor.raw_data[i * 4 + 1],
                    results[0].tensor.raw_data[i * 4 + 2],
                    results[0].tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(out_data, vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_execute_add_chain() {
        // x → Add(x, x) → y (doubles the input)
        let graph_proto = make_graph(
            vec![make_node("Add", "add0", &["x", "x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 3], &[1.0, 2.0, 3.0]);
        let inputs = vec![("x".to_string(), input)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        let out_data: Vec<f32> = (0..3)
            .map(|i| {
                f32::from_le_bytes([
                    results[0].tensor.raw_data[i * 4],
                    results[0].tensor.raw_data[i * 4 + 1],
                    results[0].tensor.raw_data[i * 4 + 2],
                    results[0].tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(out_data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_execute_matmul_add_relu() {
        // Linear chain: MatMul(x, w) → Add(_, b) → Relu → output
        let graph_proto = make_graph(
            vec![
                make_node("MatMul", "mm", &["x", "w"], &["mm_out"]),
                make_node("Add", "add", &["mm_out", "b"], &["add_out"]),
                make_node("Relu", "relu", &["add_out"], &["y"]),
            ],
            &["x", "w", "b"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        // x: [1, 2], w: [2, 2], b: [1, 2]
        let x = make_f32_tensor("x", &[1, 2], &[1.0, 2.0]);
        let w = make_f32_tensor("w", &[2, 2], &[1.0, 0.0, 0.0, 1.0]); // identity
        let b = make_f32_tensor("b", &[1, 2], &[-0.5, -3.0]);

        let inputs = vec![
            ("x".to_string(), x),
            ("w".to_string(), w),
            ("b".to_string(), b),
        ];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        // MatMul: [1,2] @ [2,2] = [1,2] → [1.0, 2.0]
        // Add: [1.0, 2.0] + [-0.5, -3.0] = [0.5, -1.0]
        // Relu: [0.5, 0.0]
        let out_data: Vec<f32> = (0..2)
            .map(|i| {
                f32::from_le_bytes([
                    results[0].tensor.raw_data[i * 4],
                    results[0].tensor.raw_data[i * 4 + 1],
                    results[0].tensor.raw_data[i * 4 + 2],
                    results[0].tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(out_data, vec![0.5, 0.0]);
    }

    #[test]
    fn test_tensor_from_proto_float_data() {
        let proto = TensorProto {
            dims: vec![2, 2],
            data_type: 1, // FLOAT
            name: "w".to_string(),
            float_data: vec![1.0, 2.0, 3.0, 4.0],
            ..TensorProto::default()
        };
        let tensor = tensor_from_proto(&proto).unwrap();
        assert_eq!(tensor.data_type, DataType::Float);
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(tensor.raw_data.len(), 16); // 4 floats × 4 bytes
    }

    #[test]
    fn test_tensor_from_proto_raw_data() {
        let raw = vec![0u8; 16]; // 4 floats
        let proto = TensorProto {
            dims: vec![4],
            data_type: 1,
            name: "t".to_string(),
            raw_data: raw.clone(),
            ..TensorProto::default()
        };
        let tensor = tensor_from_proto(&proto).unwrap();
        assert_eq!(tensor.raw_data, raw);
    }

    #[test]
    fn test_execute_with_initializers() {
        // Use initializer as weight: x → MatMul(x, w_init) → y
        // In ONNX, initializers are also listed as graph inputs
        let graph_proto = make_graph(
            vec![make_node("MatMul", "mm", &["x", "w"], &["y"])],
            &["x", "w"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let x = make_f32_tensor("x", &[1, 2], &[3.0, 4.0]);
        let inputs = vec![("x".to_string(), x)];

        // Weight as initializer (identity matrix)
        let w_init = TensorProto {
            dims: vec![2, 2],
            data_type: 1,
            name: "w".to_string(),
            float_data: vec![1.0, 0.0, 0.0, 1.0],
            ..TensorProto::default()
        };

        let results = execute_graph(&exec_graph, &inputs, &[w_init], None).unwrap();
        let out_data: Vec<f32> = (0..2)
            .map(|i| {
                f32::from_le_bytes([
                    results[0].tensor.raw_data[i * 4],
                    results[0].tensor.raw_data[i * 4 + 1],
                    results[0].tensor.raw_data[i * 4 + 2],
                    results[0].tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(out_data, vec![3.0, 4.0]);
    }
}
