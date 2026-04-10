// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph executor — traverses the execution graph in topological order,
//! dispatching each node to the corresponding CPU operator with tensor
//! I/O routing via a named tensor value map.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{self, allocate_tensor_data, I64_SIZE};
use crate::graph::ExecutionGraph;
use crate::onnx_types::{AttributeProto, AttributeType, TensorProto};
use crate::operators::{self, OpError, OpKind};
use crate::session::{InferenceOutput, SessionError};
use crate::tensor::{DataType, Tensor, TensorShape};

/// Executes an ONNX graph end-to-end.
///
/// Iterates the topologically-sorted execution order, resolving each node's
/// inputs from `value_map`, dispatching to the corresponding operator, and
/// storing outputs back into the map. After all nodes execute, the graph's
/// declared outputs are extracted and returned.
///
/// When the `gpu` feature is enabled and a [`GpuBackend`] is provided,
/// operators that the backend supports will be dispatched to GPU (once
/// fully implemented). Currently falls through to CPU for all ops.
pub fn execute_graph(
    graph: &ExecutionGraph,
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    yield_fn: Option<fn()>,
    #[cfg(feature = "gpu")] gpu_backend: Option<&smallaios_compute::GpuBackend>,
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
        let outputs = dispatch_node(
            &node.op_type,
            &input_tensors,
            &node.attributes,
            #[cfg(feature = "gpu")]
            gpu_backend,
        )
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
        let mut bytes = allocate_tensor_data(proto.float_data.len(), DataType::Float);
        for (i, &val) in proto.float_data.iter().enumerate() {
            byte_io::write_f32(&mut bytes, i, val);
        }
        bytes
    } else if !proto.int64_data.is_empty() {
        let mut bytes = allocate_tensor_data(proto.int64_data.len(), DataType::Int64);
        for (i, &val) in proto.int64_data.iter().enumerate() {
            byte_io::write_i64(&mut bytes, i, val);
        }
        bytes
    } else if !proto.int32_data.is_empty() {
        let mut bytes = allocate_tensor_data(proto.int32_data.len(), DataType::Int32);
        for (i, &val) in proto.int32_data.iter().enumerate() {
            byte_io::write_i32(&mut bytes, i, val);
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

/// Reads an f32 from the first 4 bytes of an optional tensor's raw data.
///
/// Returns `None` if the tensor is absent or its buffer is shorter than 4
/// bytes. Used for extracting scalar `min`/`max`/`constant_value` inputs
/// from the shape-operator dispatchers.
fn read_first_f32(tensor: Option<&Tensor>) -> Option<f32> {
    tensor.and_then(|t| {
        if t.raw_data.len() >= 4 {
            Some(byte_io::read_f32(&t.raw_data, 0))
        } else {
            None
        }
    })
}

/// Dispatches a single graph node to the appropriate operator function.
///
/// If a GPU backend is available and supports the operator, a future
/// implementation would dispatch to the GPU path. Currently, GPU support
/// is checked but all execution falls through to the CPU path since the
/// GPU backends are architectural stubs.
///
/// Matches the node's `op_type` to an `OpKind`, then delegates to a
/// category-specific dispatcher (`dispatch_arithmetic`,
/// `dispatch_activation`, etc.) to reduce cognitive complexity. Returns
/// a vector of output tensors (most operators produce exactly one).
fn dispatch_node(
    op_type: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
    #[cfg(feature = "gpu")] gpu_backend: Option<&smallaios_compute::GpuBackend>,
) -> Result<Vec<Tensor>, OpError> {
    // Check if GPU backend can handle this operator. If so, a real
    // implementation would transfer tensors to device, launch the GPU
    // kernel, and transfer results back. For now we note the support
    // and fall through to CPU execution.
    #[cfg(feature = "gpu")]
    let _gpu_supported = gpu_backend
        .map(|gb| gb.supports_op(op_type))
        .unwrap_or(false);
    let kind =
        OpKind::parse_str(op_type).ok_or_else(|| OpError::UnsupportedOp(String::from(op_type)))?;

    let result = match kind {
        OpKind::Add | OpKind::Sub | OpKind::Mul | OpKind::Div | OpKind::MatMul | OpKind::Gemm => {
            dispatch_arithmetic(kind, inputs, attrs)
        }
        OpKind::Relu | OpKind::Sigmoid | OpKind::Tanh | OpKind::Softmax => {
            dispatch_activation(kind, inputs, attrs)
        }
        OpKind::Conv => dispatch_convolution(kind, inputs, attrs),
        OpKind::BatchNormalization | OpKind::LayerNormalization => {
            dispatch_normalization(kind, inputs, attrs)
        }
        OpKind::MaxPool | OpKind::AveragePool | OpKind::GlobalAveragePool => {
            dispatch_pooling(kind, inputs, attrs)
        }
        OpKind::ReduceMean | OpKind::ReduceSum => dispatch_reduction(kind, inputs, attrs),
        OpKind::Reshape
        | OpKind::Transpose
        | OpKind::Flatten
        | OpKind::Squeeze
        | OpKind::Unsqueeze
        | OpKind::Concat
        | OpKind::Gather
        | OpKind::Slice
        | OpKind::Pad
        | OpKind::Cast
        | OpKind::Clip => dispatch_shape(kind, inputs, attrs),
    };

    result.map(|t| alloc::vec![t])
}

/// Dispatches arithmetic operators: Add, Sub, Mul, Div, MatMul, Gemm.
fn dispatch_arithmetic(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
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
        _ => Err(OpError::UnsupportedOp(String::from("arithmetic"))),
    }
}

/// Dispatches activation operators: Relu, Sigmoid, Tanh, Softmax.
fn dispatch_activation(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
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
        _ => Err(OpError::UnsupportedOp(String::from("activation"))),
    }
}

/// Dispatches convolution operators: Conv.
fn dispatch_convolution(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    _attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Conv => {
            let input = require_input(inputs, 0, "Conv")?;
            let weight = require_input(inputs, 1, "Conv")?;
            let bias = optional_input(inputs, 2);
            operators::op_conv(input, weight, bias)
        }
        _ => Err(OpError::UnsupportedOp(String::from("convolution"))),
    }
}

/// Dispatches normalization operators: BatchNormalization, LayerNormalization.
fn dispatch_normalization(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
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
        _ => Err(OpError::UnsupportedOp(String::from("normalization"))),
    }
}

/// Dispatches pooling operators: MaxPool, AveragePool, GlobalAveragePool.
fn dispatch_pooling(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
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
        _ => Err(OpError::UnsupportedOp(String::from("pooling"))),
    }
}

/// Dispatches reduction operators: ReduceMean, ReduceSum.
fn dispatch_reduction(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
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
        _ => Err(OpError::UnsupportedOp(String::from("reduction"))),
    }
}

/// Dispatches shape-manipulation operators: Reshape, Transpose, Flatten,
/// Squeeze, Unsqueeze, Concat, Gather, Slice, Pad, Cast, Clip.
fn dispatch_shape(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Reshape => {
            let t = require_input(inputs, 0, "Reshape")?;
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
            let axes_tensor = optional_input(inputs, 1);
            let axes = axes_tensor.map(read_i64_tensor);
            operators::op_squeeze(t, axes.as_deref())
        }
        OpKind::Unsqueeze => {
            let t = require_input(inputs, 0, "Unsqueeze")?;
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
            let constant_value = read_first_f32(constant_tensor).unwrap_or(0.0);
            operators::op_pad(t, &pads, mode_str, constant_value)
        }
        OpKind::Clip => {
            let t = require_input(inputs, 0, "Clip")?;
            let min_val = read_first_f32(optional_input(inputs, 1));
            let max_val = read_first_f32(optional_input(inputs, 2));
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
        _ => Err(OpError::UnsupportedOp(String::from("shape"))),
    }
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
    if tensor.data_type == DataType::Int64 && tensor.raw_data.len() >= I64_SIZE {
        let count = tensor.raw_data.len() / I64_SIZE;
        (0..count)
            .map(|i| byte_io::read_i64(&tensor.raw_data, i))
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

    /// Helper: read f32 values from a tensor's raw_data.
    fn read_f32_output(tensor: &Tensor, count: usize) -> Vec<f32> {
        (0..count)
            .map(|i| {
                f32::from_le_bytes([
                    tensor.raw_data[i * 4],
                    tensor.raw_data[i * 4 + 1],
                    tensor.raw_data[i * 4 + 2],
                    tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect()
    }

    #[test]
    fn test_execute_softmax_graph() {
        let graph_proto = make_graph(
            vec![make_node("Softmax", "sm0", &["x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = vec![("x".to_string(), input)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "y");

        let out_data = read_f32_output(&results[0].tensor, 4);
        let sum: f32 = out_data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-5,
            "softmax output should sum to 1.0, got {sum}"
        );
        // Each output should be positive
        for &v in &out_data {
            assert!(v > 0.0);
        }
    }

    #[test]
    fn test_execute_sigmoid_graph() {
        let graph_proto = make_graph(
            vec![make_node("Sigmoid", "sig0", &["x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 3], &[-10.0, 0.0, 10.0]);
        let inputs = vec![("x".to_string(), input)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        let out_data = read_f32_output(&results[0].tensor, 3);
        // sigmoid(0) ≈ 0.5
        assert!(
            (out_data[1] - 0.5).abs() < 1e-5,
            "sigmoid(0) should be ~0.5, got {}",
            out_data[1]
        );
        // sigmoid(-10) ≈ 0, sigmoid(10) ≈ 1
        assert!(out_data[0] < 0.01);
        assert!(out_data[2] > 0.99);
    }

    #[test]
    fn test_execute_sub_graph() {
        // x - x = 0
        let graph_proto = make_graph(
            vec![make_node("Sub", "sub0", &["x", "x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 3], &[5.0, 10.0, 15.0]);
        let inputs = vec![("x".to_string(), input)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        let out_data = read_f32_output(&results[0].tensor, 3);
        assert_eq!(out_data, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_execute_mul_graph() {
        // x * 2 using a constant input
        let graph_proto = make_graph(
            vec![make_node("Mul", "mul0", &["x", "two"], &["y"])],
            &["x", "two"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let x = make_f32_tensor("x", &[1, 3], &[1.0, 2.0, 3.0]);
        let two = make_f32_tensor("two", &[1, 3], &[2.0, 2.0, 2.0]);
        let inputs = vec![("x".to_string(), x), ("two".to_string(), two)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        let out_data = read_f32_output(&results[0].tensor, 3);
        assert_eq!(out_data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_execute_branching_graph() {
        // Input feeds two ops: Add(x, x) → sum_out and Sub(x, one) → diff_out
        let graph_proto = make_graph(
            vec![
                make_node("Add", "add0", &["x", "x"], &["sum_out"]),
                make_node("Sub", "sub0", &["x", "one"], &["diff_out"]),
            ],
            &["x", "one"],
            &["sum_out", "diff_out"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let x = make_f32_tensor("x", &[1, 3], &[3.0, 5.0, 7.0]);
        let one = make_f32_tensor("one", &[1, 3], &[1.0, 1.0, 1.0]);
        let inputs = vec![("x".to_string(), x), ("one".to_string(), one)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        assert_eq!(results.len(), 2);

        // Find outputs by name (order may vary)
        let sum_out = results.iter().find(|r| r.name == "sum_out").unwrap();
        let diff_out = results.iter().find(|r| r.name == "diff_out").unwrap();

        let sum_data = read_f32_output(&sum_out.tensor, 3);
        let diff_data = read_f32_output(&diff_out.tensor, 3);

        assert_eq!(sum_data, vec![6.0, 10.0, 14.0]);
        assert_eq!(diff_data, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_execute_missing_input_error() {
        // Graph expects "x" but we provide nothing
        let graph_proto = make_graph(
            vec![make_node("Relu", "relu0", &["x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let result = execute_graph(&exec_graph, &[], &[], None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            SessionError::ExecutionFailed(msg) => {
                assert!(
                    msg.contains("missing required input"),
                    "expected 'missing required input' in error, got: {msg}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_execute_unsupported_op_error() {
        let graph_proto = make_graph(
            vec![make_node("FakeOp", "fake0", &["x"], &["y"])],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 2], &[1.0, 2.0]);
        let inputs = vec![("x".to_string(), input)];

        let result = execute_graph(&exec_graph, &inputs, &[], None);
        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            SessionError::ExecutionFailed(msg) => {
                assert!(
                    msg.contains("FakeOp"),
                    "expected 'FakeOp' in error message, got: {msg}"
                );
            }
            other => panic!("expected ExecutionFailed, got: {other:?}"),
        }
    }

    #[test]
    fn test_execute_yield_callback() {
        use core::sync::atomic::{AtomicUsize, Ordering};
        static YIELD_COUNT: AtomicUsize = AtomicUsize::new(0);
        fn test_yield() {
            YIELD_COUNT.fetch_add(1, Ordering::Relaxed);
        }

        // Reset counter
        YIELD_COUNT.store(0, Ordering::Relaxed);

        // Graph with 3 nodes: Relu → Relu → Relu chain
        let graph_proto = make_graph(
            vec![
                make_node("Relu", "r0", &["x"], &["a"]),
                make_node("Relu", "r1", &["a"], &["b"]),
                make_node("Relu", "r2", &["b"], &["y"]),
            ],
            &["x"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let input = make_f32_tensor("x", &[1, 2], &[1.0, 2.0]);
        let inputs = vec![("x".to_string(), input)];

        let _results = execute_graph(&exec_graph, &inputs, &[], Some(test_yield)).unwrap();
        assert_eq!(
            YIELD_COUNT.load(Ordering::Relaxed),
            3,
            "yield_fn should be called once per node"
        );
    }

    #[test]
    fn test_tensor_from_proto_int64() {
        let proto = TensorProto {
            dims: vec![3],
            data_type: 7, // INT64
            name: "i64t".to_string(),
            int64_data: vec![10, 20, 30],
            ..TensorProto::default()
        };
        let tensor = tensor_from_proto(&proto).unwrap();
        assert_eq!(tensor.data_type, DataType::Int64);
        assert_eq!(tensor.shape.dims, vec![3]);
        assert_eq!(tensor.raw_data.len(), 24); // 3 × 8 bytes

        // Verify round-trip: read back the i64 values
        let vals = read_i64_tensor(&tensor);
        assert_eq!(vals, vec![10, 20, 30]);
    }

    #[test]
    fn test_tensor_from_proto_int32() {
        let proto = TensorProto {
            dims: vec![2, 2],
            data_type: 6, // INT32
            name: "i32t".to_string(),
            int32_data: vec![100, 200, 300, 400],
            ..TensorProto::default()
        };
        let tensor = tensor_from_proto(&proto).unwrap();
        assert_eq!(tensor.data_type, DataType::Int32);
        assert_eq!(tensor.shape.dims, vec![2, 2]);
        assert_eq!(tensor.raw_data.len(), 16); // 4 × 4 bytes

        // Verify individual values by reading raw bytes
        let vals: Vec<i32> = (0..4)
            .map(|i| {
                i32::from_le_bytes([
                    tensor.raw_data[i * 4],
                    tensor.raw_data[i * 4 + 1],
                    tensor.raw_data[i * 4 + 2],
                    tensor.raw_data[i * 4 + 3],
                ])
            })
            .collect();
        assert_eq!(vals, vec![100, 200, 300, 400]);
    }

    #[test]
    fn test_dispatch_conv() {
        // 1-channel conv with 1×1 kernel (identity-like)
        // Input: [N=1, C=1, H=2, W=2], Kernel: [1, 1, 1, 1] with weight=1.0
        let graph_proto = make_graph(
            vec![make_node("Conv", "conv0", &["x", "w"], &["y"])],
            &["x", "w"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let x = make_f32_tensor("x", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let w = make_f32_tensor("w", &[1, 1, 1, 1], &[1.0]); // 1×1 kernel, weight=1
        let inputs = vec![("x".to_string(), x), ("w".to_string(), w)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        assert_eq!(results.len(), 1);
        let out_data = read_f32_output(&results[0].tensor, 4);
        // 1×1 conv with weight=1.0 is identity
        assert_eq!(out_data, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_execute_reshape_graph() {
        // Reshape a [2, 3] tensor to [3, 2] using a shape tensor
        let graph_proto = make_graph(
            vec![make_node("Reshape", "rs0", &["x", "shape"], &["y"])],
            &["x", "shape"],
            &["y"],
        );
        let exec_graph = build_execution_graph(&graph_proto).unwrap();

        let x = make_f32_tensor("x", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        // Shape tensor: Int64 type encoding [3, 2]
        let shape_tensor = Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(vec![2]),
            name: String::from("shape"),
            raw_data: {
                let mut d = vec![0u8; 16];
                d[0..8].copy_from_slice(&3i64.to_le_bytes());
                d[8..16].copy_from_slice(&2i64.to_le_bytes());
                d
            },
        };

        let inputs = vec![("x".to_string(), x), ("shape".to_string(), shape_tensor)];

        let results = execute_graph(&exec_graph, &inputs, &[], None).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tensor.shape.dims, vec![3, 2]);

        // Data should be preserved
        let out_data = read_f32_output(&results[0].tensor, 6);
        assert_eq!(out_data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // Direct dispatch helper tests — exercise the extracted per-category
    // dispatch functions without going through `execute_graph`. These exist
    // primarily to give SonarCloud line coverage on the helpers themselves.
    // -----------------------------------------------------------------------

    fn make_i64_tensor(name: &str, shape: &[i64], data: &[i64]) -> Tensor {
        let mut raw = vec![0u8; data.len() * 8];
        for (i, &v) in data.iter().enumerate() {
            raw[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(shape.to_vec()),
            name: String::from(name),
            raw_data: raw,
        }
    }

    #[test]
    fn test_read_first_f32_some_value() {
        let t = make_f32_tensor("c", &[1], &[3.25]);
        assert_eq!(read_first_f32(Some(&t)), Some(3.25));
    }

    #[test]
    fn test_read_first_f32_none() {
        assert_eq!(read_first_f32(None), None);
    }

    #[test]
    fn test_read_first_f32_short_buffer() {
        let t = Tensor {
            data_type: DataType::Float,
            shape: TensorShape::new(vec![0]),
            name: String::from("empty"),
            raw_data: vec![1, 2, 3], // < 4 bytes
        };
        assert_eq!(read_first_f32(Some(&t)), None);
    }

    #[test]
    fn test_dispatch_arithmetic_add() {
        let a = make_f32_tensor("a", &[3], &[1.0, 2.0, 3.0]);
        let b = make_f32_tensor("b", &[3], &[4.0, 5.0, 6.0]);
        let inputs = [Some(&a), Some(&b)];
        let out = dispatch_arithmetic(OpKind::Add, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&out, 3), vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_dispatch_arithmetic_sub_mul_div() {
        let a = make_f32_tensor("a", &[2], &[6.0, 8.0]);
        let b = make_f32_tensor("b", &[2], &[2.0, 4.0]);
        let inputs = [Some(&a), Some(&b)];
        let sub = dispatch_arithmetic(OpKind::Sub, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&sub, 2), vec![4.0, 4.0]);
        let mul = dispatch_arithmetic(OpKind::Mul, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&mul, 2), vec![12.0, 32.0]);
        let div = dispatch_arithmetic(OpKind::Div, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&div, 2), vec![3.0, 2.0]);
    }

    #[test]
    fn test_dispatch_arithmetic_matmul() {
        let a = make_f32_tensor("a", &[1, 2], &[1.0, 2.0]);
        let b = make_f32_tensor("b", &[2, 1], &[3.0, 4.0]);
        let inputs = [Some(&a), Some(&b)];
        let out = dispatch_arithmetic(OpKind::MatMul, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&out, 1), vec![11.0]);
    }

    #[test]
    fn test_dispatch_arithmetic_wrong_kind_errors() {
        let a = make_f32_tensor("a", &[1], &[1.0]);
        let inputs = [Some(&a)];
        let r = dispatch_arithmetic(OpKind::Relu, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_activation_relu() {
        let t = make_f32_tensor("x", &[4], &[-1.0, 0.0, 1.0, 2.0]);
        let inputs = [Some(&t)];
        let out = dispatch_activation(OpKind::Relu, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&out, 4), vec![0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_dispatch_activation_sigmoid_tanh_softmax() {
        let t = make_f32_tensor("x", &[3], &[0.0, 0.5, -0.5]);
        let inputs = [Some(&t)];
        let sig = dispatch_activation(OpKind::Sigmoid, &inputs, &[]).unwrap();
        assert!((read_f32_output(&sig, 3)[0] - 0.5).abs() < 1e-5);
        let tanh = dispatch_activation(OpKind::Tanh, &inputs, &[]).unwrap();
        assert!(read_f32_output(&tanh, 3)[0].abs() < 1e-5);
        let sm = dispatch_activation(OpKind::Softmax, &inputs, &[]).unwrap();
        let sums: f32 = read_f32_output(&sm, 3).iter().sum();
        assert!((sums - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_dispatch_activation_wrong_kind_errors() {
        let t = make_f32_tensor("x", &[1], &[1.0]);
        let inputs = [Some(&t)];
        let r = dispatch_activation(OpKind::Add, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_convolution_conv() {
        // 1x1 identity conv
        let x = make_f32_tensor("x", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let w = make_f32_tensor("w", &[1, 1, 1, 1], &[1.0]);
        let inputs = [Some(&x), Some(&w)];
        let out = dispatch_convolution(OpKind::Conv, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&out, 4), vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_dispatch_convolution_wrong_kind_errors() {
        let x = make_f32_tensor("x", &[1], &[1.0]);
        let inputs = [Some(&x)];
        let r = dispatch_convolution(OpKind::Relu, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_pooling_global_average() {
        // Global avg pool: [1,1,2,2] with values averaged → [1,1,1,1] = 2.5
        let x = make_f32_tensor("x", &[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [Some(&x)];
        let out = dispatch_pooling(OpKind::GlobalAveragePool, &inputs, &[]).unwrap();
        let vals = read_f32_output(&out, 1);
        assert!((vals[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_dispatch_pooling_wrong_kind_errors() {
        let x = make_f32_tensor("x", &[1], &[1.0]);
        let inputs = [Some(&x)];
        let r = dispatch_pooling(OpKind::Relu, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_reduction_reduce_mean() {
        let x = make_f32_tensor("x", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [Some(&x)];
        let attrs = vec![AttributeProto {
            name: "keepdims".to_string(),
            attr_type: AttributeType::Int,
            i: 0,
            ..AttributeProto::default()
        }];
        let out = dispatch_reduction(OpKind::ReduceMean, &inputs, &attrs).unwrap();
        let vals = read_f32_output(&out, 1);
        assert!((vals[0] - 2.5).abs() < 1e-5);
    }

    #[test]
    fn test_dispatch_reduction_reduce_sum() {
        let x = make_f32_tensor("x", &[4], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [Some(&x)];
        let attrs = vec![AttributeProto {
            name: "keepdims".to_string(),
            attr_type: AttributeType::Int,
            i: 0,
            ..AttributeProto::default()
        }];
        let out = dispatch_reduction(OpKind::ReduceSum, &inputs, &attrs).unwrap();
        let vals = read_f32_output(&out, 1);
        assert!((vals[0] - 10.0).abs() < 1e-5);
    }

    #[test]
    fn test_dispatch_reduction_wrong_kind_errors() {
        let x = make_f32_tensor("x", &[1], &[1.0]);
        let inputs = [Some(&x)];
        let r = dispatch_reduction(OpKind::Relu, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_normalization_layer_norm() {
        let x = make_f32_tensor("x", &[1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let scale = make_f32_tensor("s", &[4], &[1.0, 1.0, 1.0, 1.0]);
        let inputs = [Some(&x), Some(&scale)];
        let attrs = vec![AttributeProto {
            name: "axis".to_string(),
            attr_type: AttributeType::Int,
            i: -1,
            ..AttributeProto::default()
        }];
        let out = dispatch_normalization(OpKind::LayerNormalization, &inputs, &attrs).unwrap();
        // Output should have same shape; check length
        assert_eq!(out.shape.dims, vec![1, 4]);
    }

    #[test]
    fn test_dispatch_normalization_wrong_kind_errors() {
        let x = make_f32_tensor("x", &[1], &[1.0]);
        let inputs = [Some(&x)];
        let r = dispatch_normalization(OpKind::Relu, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_shape_reshape() {
        let x = make_f32_tensor("x", &[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let shape = make_i64_tensor("shape", &[2], &[3, 2]);
        let inputs = [Some(&x), Some(&shape)];
        let out = dispatch_shape(OpKind::Reshape, &inputs, &[]).unwrap();
        assert_eq!(out.shape.dims, vec![3, 2]);
    }

    #[test]
    fn test_dispatch_shape_flatten() {
        let x = make_f32_tensor("x", &[2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let inputs = [Some(&x)];
        let out = dispatch_shape(OpKind::Flatten, &inputs, &[]).unwrap();
        assert_eq!(out.shape.total_elements(), 4);
    }

    #[test]
    fn test_dispatch_shape_clip_with_min_max() {
        let x = make_f32_tensor("x", &[3], &[-5.0, 0.0, 5.0]);
        let min_t = make_f32_tensor("min", &[1], &[-2.0]);
        let max_t = make_f32_tensor("max", &[1], &[2.0]);
        let inputs = [Some(&x), Some(&min_t), Some(&max_t)];
        let out = dispatch_shape(OpKind::Clip, &inputs, &[]).unwrap();
        assert_eq!(read_f32_output(&out, 3), vec![-2.0, 0.0, 2.0]);
    }

    #[test]
    fn test_dispatch_shape_cast() {
        let x = make_f32_tensor("x", &[2], &[1.7, -2.3]);
        let inputs = [Some(&x)];
        let attrs = vec![AttributeProto {
            name: "to".to_string(),
            attr_type: AttributeType::Int,
            i: 6, // INT32
            ..AttributeProto::default()
        }];
        let out = dispatch_shape(OpKind::Cast, &inputs, &attrs).unwrap();
        assert_eq!(out.data_type, DataType::Int32);
    }

    #[test]
    fn test_dispatch_shape_wrong_kind_errors() {
        let x = make_f32_tensor("x", &[1], &[1.0]);
        let inputs = [Some(&x)];
        let r = dispatch_shape(OpKind::Relu, &inputs, &[]);
        assert!(r.is_err());
    }

    #[test]
    fn test_dispatch_shape_concat_empty_errors() {
        let inputs: [Option<&Tensor>; 0] = [];
        let r = dispatch_shape(OpKind::Concat, &inputs, &[]);
        assert!(r.is_err());
    }
}
