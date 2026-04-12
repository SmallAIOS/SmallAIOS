// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Graph executor — traverses the execution graph in topological order,
//! dispatching each node to the corresponding CPU operator with tensor
//! I/O routing via a named tensor value map.

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use crate::byte_io::{self, allocate_tensor_data, I64_SIZE};
use crate::graph::{ExecutionGraph, ExecutionNode};
use crate::onnx_types::{AttributeProto, AttributeType, TensorProto};
use crate::operators::{self, OpError, OpKind};
use crate::ops;
#[cfg(test)]
use crate::profile::NullTimeSource;
use crate::profile::{
    classify_op, BudgetResult, InferenceProfile, OperatorBudget, OperatorMeasurement, TimeSource,
};
use crate::session::{InferenceOutput, SessionError};
use crate::sub_executor;
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
    mut profile: Option<&mut InferenceProfile>,
    budget: &OperatorBudget,
    time_source: &dyn TimeSource,
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

        // Control-flow ops need the full ExecutionNode + outer value map,
        // so they bypass `dispatch_node` and go through the compiled inner
        // graph path.
        if is_control_flow_op(&node.op_type) {
            let outputs = dispatch_control_flow(node, &value_map, initializers)?;
            for (i, output_name) in node.outputs.iter().enumerate() {
                if !output_name.is_empty() {
                    if let Some(tensor) = outputs.get(i) {
                        value_map.insert(output_name.clone(), tensor.clone());
                    }
                }
            }
            if let Some(f) = yield_fn {
                f();
            }
            continue;
        }

        // Dispatch operator (optionally wrapped in timing measurement).
        let outputs = if let Some(prof) = profile.as_mut() {
            let start = time_source.now_us();
            let outputs = dispatch_node(
                &node.op_type,
                &input_tensors,
                &node.attributes,
                node.outputs.len(),
                #[cfg(feature = "gpu")]
                gpu_backend,
            )
            .map_err(|e| SessionError::ExecutionFailed(alloc::format!("{}: {}", node.name, e)))?;
            let elapsed = time_source.now_us().saturating_sub(start);

            let class = classify_op(&node.op_type);
            let budget_result = budget.check(class, elapsed);

            prof.operators.push(OperatorMeasurement {
                op_type: node.op_type.clone(),
                class,
                actual_us: elapsed,
                budget_result,
            });
            prof.total_us = prof.total_us.saturating_add(elapsed);

            match budget_result {
                BudgetResult::Ok => {}
                BudgetResult::Warning => prof.warnings_count += 1,
                BudgetResult::SoftLimit => prof.soft_limit_count += 1,
                BudgetResult::HardLimit => {
                    prof.hard_limit_aborted = true;
                    return Err(SessionError::ExecutionFailed(alloc::format!(
                        "operator '{}' exceeded hard time limit: {} us",
                        node.op_type,
                        elapsed
                    )));
                }
            }

            outputs
        } else {
            // Zero-overhead path: no profiling.
            dispatch_node(
                &node.op_type,
                &input_tensors,
                &node.attributes,
                node.outputs.len(),
                #[cfg(feature = "gpu")]
                gpu_backend,
            )
            .map_err(|e| SessionError::ExecutionFailed(alloc::format!("{}: {}", node.name, e)))?
        };

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

/// Extracts a tensor attribute by name. Returns `None` if absent or if the
/// attribute's declared type is not `Tensor`.
fn get_attr_tensor<'a>(attrs: &'a [AttributeProto], name: &str) -> Option<&'a TensorProto> {
    attrs
        .iter()
        .find(|a| a.name == name && a.attr_type == AttributeType::Tensor)
        .and_then(|a| a.t.as_deref())
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
pub(crate) fn dispatch_node(
    op_type: &str,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
    output_count: usize,
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

    // Multi-output ops bypass the single-tensor wrapper.
    match kind {
        OpKind::Split => return dispatch_split(inputs, attrs, output_count),
        OpKind::LSTM => return dispatch_lstm(inputs, attrs),
        OpKind::GRU => return dispatch_gru(inputs, attrs),
        OpKind::RNN => return dispatch_rnn(inputs, attrs),
        OpKind::TopK => return dispatch_top_k(inputs, attrs),
        OpKind::DynamicQuantizeLinear => return dispatch_dynamic_quantize(inputs),
        _ => {}
    }

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
        OpKind::ReduceMean
        | OpKind::ReduceSum
        | OpKind::ReduceMax
        | OpKind::ReduceMin
        | OpKind::ReduceProd
        | OpKind::ArgMax
        | OpKind::ArgMin => dispatch_reduction(kind, inputs, attrs),
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
        OpKind::Pow
        | OpKind::Sqrt
        | OpKind::Exp
        | OpKind::Log
        | OpKind::Erf
        | OpKind::Neg
        | OpKind::Abs
        | OpKind::Floor
        | OpKind::Ceil
        | OpKind::Round
        | OpKind::Mod
        | OpKind::Sin
        | OpKind::Cos
        | OpKind::Reciprocal
        | OpKind::Sign
        | OpKind::Sum
        | OpKind::Mean
        | OpKind::And
        | OpKind::Or
        | OpKind::LogSoftmax => dispatch_math(kind, inputs, attrs),
        OpKind::Equal
        | OpKind::NotEqual
        | OpKind::Less
        | OpKind::LessOrEqual
        | OpKind::Greater
        | OpKind::GreaterOrEqual
        | OpKind::Where
        | OpKind::Min
        | OpKind::Max
        | OpKind::Not => dispatch_compare(kind, inputs, attrs),
        OpKind::Shape
        | OpKind::Size
        | OpKind::Identity
        | OpKind::Constant
        | OpKind::ConstantOfShape
        | OpKind::Range
        | OpKind::Trilu
        | OpKind::CumSum
        | OpKind::GatherND
        | OpKind::ScatterND => dispatch_shape_data(kind, inputs, attrs),
        OpKind::Gelu
        | OpKind::LeakyRelu
        | OpKind::Elu
        | OpKind::Swish
        | OpKind::HardSigmoid
        | OpKind::HardSwish
        | OpKind::PRelu
        | OpKind::Mish => dispatch_composite_activation(kind, inputs, attrs),
        OpKind::Expand | OpKind::Tile | OpKind::OneHot | OpKind::Einsum => {
            dispatch_transformer(kind, inputs, attrs)
        }
        OpKind::QuantizeLinear
        | OpKind::DequantizeLinear
        | OpKind::QLinearMatMul
        | OpKind::QLinearConv => dispatch_quantized(kind, inputs, attrs),
        OpKind::Resize
        | OpKind::DepthToSpace
        | OpKind::SpaceToDepth
        | OpKind::RoiAlign
        | OpKind::GridSample
        | OpKind::GlobalMaxPool
        | OpKind::CenterCropPad => dispatch_vision(kind, inputs, attrs),
        OpKind::Compress
        | OpKind::NonZero
        | OpKind::Unique
        | OpKind::GatherElements
        | OpKind::ScatterElements => dispatch_data_select(kind, inputs, attrs),
        OpKind::GroupNormalization | OpKind::InstanceNormalization => {
            dispatch_normalization_v2(kind, inputs, attrs)
        }
        // Phase 2 generative / normalization / quantized ops.
        OpKind::RMSNormalization
        | OpKind::MatMulInteger
        | OpKind::RandomNormal
        | OpKind::RandomNormalLike
        | OpKind::RandomUniform
        | OpKind::RandomUniformLike
        | OpKind::Multinomial
        | OpKind::Bernoulli
        | OpKind::Dropout
        | OpKind::EyeLike
        | OpKind::ReduceL1
        | OpKind::ReduceL2
        | OpKind::ReduceLogSum
        | OpKind::ReduceLogSumExp
        | OpKind::ReduceSumSquare
        | OpKind::LpNormalization
        | OpKind::MeanVarianceNormalization
        | OpKind::Softplus => dispatch_generative(kind, inputs, attrs),
        // Phase 2 control-flow ops are intercepted by `execute_graph` /
        // `run_sub_graph` before reaching `dispatch_node` — they need the
        // full `ExecutionNode` (for `inner_graphs`) and the outer value
        // map, which this op-level dispatcher does not carry. If a caller
        // somehow reaches this arm directly (e.g., an older test harness),
        // surface a clear error rather than silently mis-dispatching.
        OpKind::If | OpKind::Loop | OpKind::Scan => Err(OpError::InvalidAttribute(String::from(
            "control-flow op must be dispatched via dispatch_control_flow \
             with the full ExecutionNode and outer value map",
        ))),
        // Multi-output kinds handled above and returned early.
        OpKind::Split
        | OpKind::LSTM
        | OpKind::GRU
        | OpKind::RNN
        | OpKind::TopK
        | OpKind::DynamicQuantizeLinear => unreachable!(),
    };

    result.map(|t| alloc::vec![t])
}

// ---------------------------------------------------------------------------
// Control-flow dispatch (If / Loop / Scan)
// ---------------------------------------------------------------------------

/// Returns `true` if the operator is a control-flow op that requires the
/// compiled-inner-graph dispatch path rather than the flat `dispatch_node`.
pub(crate) fn is_control_flow_op(op_type: &str) -> bool {
    matches!(op_type, "If" | "Loop" | "Scan")
}

/// Dispatches an `If`/`Loop`/`Scan` node using its compiled inner graphs
/// and the current value map.
///
/// Unlike [`dispatch_node`], this helper has access to the full
/// [`ExecutionNode`] (for `inner_graphs`) and the outer value map, which
/// the sub-graph executor needs to seed the body's evaluation scope.
///
/// On success returns the node's output tensors in `node.outputs` order.
/// On error returns a [`SessionError`] describing the failure, including
/// `"missing inner graph '<name>'"` when a required body attribute is
/// absent.
pub(crate) fn dispatch_control_flow(
    node: &ExecutionNode,
    value_map: &BTreeMap<String, Tensor>,
    initializers: &[TensorProto],
) -> Result<Vec<Tensor>, SessionError> {
    match node.op_type.as_str() {
        "If" => dispatch_if(node, value_map, initializers),
        "Loop" => dispatch_loop(node, value_map, initializers),
        "Scan" => dispatch_scan(node, value_map, initializers),
        other => Err(SessionError::ExecutionFailed(alloc::format!(
            "dispatch_control_flow called on non-control-flow op '{}'",
            other
        ))),
    }
}

fn require_inner_graph<'a>(
    node: &'a ExecutionNode,
    key: &'static str,
) -> Result<&'a ExecutionGraph, SessionError> {
    node.inner_graphs.get(key).ok_or_else(|| {
        SessionError::ExecutionFailed(alloc::format!(
            "{}: missing inner graph '{}'",
            node.name,
            key
        ))
    })
}

fn dispatch_if(
    node: &ExecutionNode,
    value_map: &BTreeMap<String, Tensor>,
    initializers: &[TensorProto],
) -> Result<Vec<Tensor>, SessionError> {
    // ONNX `If`: inputs[0] is the scalar bool condition.
    let cond_name = node
        .inputs
        .first()
        .ok_or_else(|| SessionError::ExecutionFailed(String::from("If: missing cond input")))?;
    let cond_tensor = value_map.get(cond_name).ok_or_else(|| {
        SessionError::ExecutionFailed(alloc::format!("If: cond tensor '{}' not found", cond_name))
    })?;
    let cond = sub_executor::read_bool_scalar(cond_tensor)
        .map_err(|e| SessionError::ExecutionFailed(alloc::format!("If: {}", e)))?;

    let branch = if cond {
        require_inner_graph(node, "then_branch")?
    } else {
        require_inner_graph(node, "else_branch")?
    };

    // Run the selected branch with the outer value_map as its seed scope.
    let result = sub_executor::run_sub_graph(branch, value_map.clone(), initializers)?;
    // Extract the branch's declared outputs in order and return them
    // as this node's outputs. The ONNX spec guarantees both branches'
    // output lists align one-to-one with the If node's outputs.
    let mut out = Vec::with_capacity(node.outputs.len());
    for (i, _out_name) in node.outputs.iter().enumerate() {
        let branch_out_name = branch.output_names.get(i).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!("If: branch output {} not declared", i))
        })?;
        let tensor = result.get(branch_out_name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!(
                "If: branch missing output '{}'",
                branch_out_name
            ))
        })?;
        out.push(tensor.clone());
    }
    Ok(out)
}

fn dispatch_loop(
    node: &ExecutionNode,
    value_map: &BTreeMap<String, Tensor>,
    initializers: &[TensorProto],
) -> Result<Vec<Tensor>, SessionError> {
    // ONNX `Loop` signature (opset-21):
    //   inputs:  M (optional), cond (optional), v_initial_0..N-1
    //   outputs: v_final_0..N-1, scan_output_0..K-1
    //
    // Body signature:
    //   inputs:  iter_num, cond_in, v_in_0..N-1
    //   outputs: cond_out, v_out_0..N-1, scan_out_0..K-1
    //
    // This implementation handles the loop-carried case (N >= 0) with no
    // scan outputs (K = 0), which is sufficient for the autoregressive
    // LLM decoder pattern this change targets.
    let body = require_inner_graph(node, "body")?;

    // Read M and cond inputs (both optional — empty string means absent).
    let m: Option<i64> = read_optional_i64(node, 0, value_map);
    let cond_initial: Option<bool> = read_optional_bool(node, 1, value_map);

    // Collect initial loop-carried values (inputs[2..]).
    let mut v_current: Vec<Tensor> = Vec::new();
    for name in node.inputs.iter().skip(2) {
        if name.is_empty() {
            continue;
        }
        let tensor = value_map.get(name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!(
                "Loop: carried input '{}' not found",
                name
            ))
        })?;
        v_current.push(tensor.clone());
    }

    // Trivial-skip cases match the ONNX semantics.
    if m == Some(0) || cond_initial == Some(false) {
        return Ok(v_current);
    }

    // The body's input names: [iter_num, cond_in, v_in_0, v_in_1, ...].
    let carried_input_names: Vec<String> = if body.input_names.len() > 2 {
        body.input_names[2..].to_vec()
    } else {
        Vec::new()
    };
    // Body outputs: [cond_out, v_out_0, v_out_1, ...].
    let carried_output_names: Vec<String> = if body.output_names.len() > 1 {
        body.output_names[1..].to_vec()
    } else {
        Vec::new()
    };
    let cond_out_name = body.output_names.first().cloned();

    let mut cond_current = cond_initial.unwrap_or(true);
    let mut iter: i64 = 0;

    loop {
        if iter >= sub_executor::MAX_LOOP_ITERATIONS {
            return Err(SessionError::ExecutionFailed(alloc::format!(
                "Loop exceeded safety limit ({})",
                sub_executor::MAX_LOOP_ITERATIONS
            )));
        }
        if let Some(max) = m {
            if iter >= max {
                break;
            }
        }
        if !cond_current {
            break;
        }

        // Seed the inner value map from the outer scope and bind the
        // body's iter_num / cond_in / v_in_* inputs.
        let mut inner = value_map.clone();
        if let Some(name) = body.input_names.first() {
            if !name.is_empty() {
                inner.insert(name.clone(), sub_executor::i64_scalar(iter));
            }
        }
        if let Some(name) = body.input_names.get(1) {
            if !name.is_empty() {
                inner.insert(name.clone(), sub_executor::bool_scalar(cond_current));
            }
        }
        for (name, val) in carried_input_names.iter().zip(v_current.iter()) {
            inner.insert(name.clone(), val.clone());
        }

        let result = sub_executor::run_sub_graph(body, inner, initializers)?;

        // Extract loop-carried outputs.
        let mut next_v = Vec::with_capacity(carried_output_names.len());
        for name in &carried_output_names {
            let t = result.get(name).ok_or_else(|| {
                SessionError::ExecutionFailed(alloc::format!("Loop body missing output '{}'", name))
            })?;
            next_v.push(t.clone());
        }
        v_current = next_v;

        // Read body-emitted cond_out for the next iteration's gate.
        if let Some(name) = &cond_out_name {
            if let Some(t) = result.get(name) {
                cond_current = sub_executor::read_bool_scalar(t).unwrap_or(true);
            }
        }

        iter += 1;
    }

    Ok(v_current)
}

fn dispatch_scan(
    node: &ExecutionNode,
    value_map: &BTreeMap<String, Tensor>,
    initializers: &[TensorProto],
) -> Result<Vec<Tensor>, SessionError> {
    // Simplified Scan: num_scan_inputs = 1, forward direction, default axis.
    // This matches the Phase 2 `op_scan_with_body` semantics.
    let body = require_inner_graph(node, "body")?;

    // The single scan input is the last input of the node. We iterate
    // one body execution per "element" (current implementation treats
    // scan_input as pre-chunked and reuses the outer scope so a
    // sequence-capable Scan can be layered on later).
    let scan_input_name = node
        .inputs
        .last()
        .ok_or_else(|| SessionError::ExecutionFailed(String::from("Scan: missing input")))?;
    let _scan_tensor = value_map.get(scan_input_name).ok_or_else(|| {
        SessionError::ExecutionFailed(alloc::format!(
            "Scan: input '{}' not found",
            scan_input_name
        ))
    })?;

    // Run the body exactly once over the full sequence tensor. Proper
    // sequence slicing along a scan axis is deferred to a follow-up
    // change — this path is sufficient for smoke-testing on-disk Scan
    // nodes compiled via `inner_graphs`.
    let result = sub_executor::run_sub_graph(body, value_map.clone(), initializers)?;

    let mut out = Vec::with_capacity(node.outputs.len());
    for (i, _out_name) in node.outputs.iter().enumerate() {
        let body_out_name = body.output_names.get(i).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!("Scan: body output {} not declared", i))
        })?;
        let tensor = result.get(body_out_name).ok_or_else(|| {
            SessionError::ExecutionFailed(alloc::format!(
                "Scan: body missing output '{}'",
                body_out_name
            ))
        })?;
        out.push(tensor.clone());
    }
    Ok(out)
}

fn read_optional_i64(
    node: &ExecutionNode,
    input_idx: usize,
    value_map: &BTreeMap<String, Tensor>,
) -> Option<i64> {
    let name = node.inputs.get(input_idx)?;
    if name.is_empty() {
        return None;
    }
    let t = value_map.get(name)?;
    if t.raw_data.len() >= I64_SIZE {
        Some(byte_io::read_i64(&t.raw_data, 0))
    } else {
        None
    }
}

fn read_optional_bool(
    node: &ExecutionNode,
    input_idx: usize,
    value_map: &BTreeMap<String, Tensor>,
) -> Option<bool> {
    let name = node.inputs.get(input_idx)?;
    if name.is_empty() {
        return None;
    }
    let t = value_map.get(name)?;
    sub_executor::read_bool_scalar(t).ok()
}

// ---------------------------------------------------------------------------
// Tier 2 dispatch helpers
// ---------------------------------------------------------------------------

/// Dispatches Tier 2/3 element-wise math primitives.
fn dispatch_math(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Pow => {
            let base = require_input(inputs, 0, "Pow")?;
            let exp = require_input(inputs, 1, "Pow")?;
            ops::math::op_pow(base, exp)
        }
        OpKind::Sqrt => ops::math::op_sqrt(require_input(inputs, 0, "Sqrt")?),
        OpKind::Exp => ops::math::op_exp(require_input(inputs, 0, "Exp")?),
        OpKind::Log => ops::math::op_log(require_input(inputs, 0, "Log")?),
        OpKind::Erf => ops::math::op_erf(require_input(inputs, 0, "Erf")?),
        OpKind::Neg => ops::math::op_neg(require_input(inputs, 0, "Neg")?),
        OpKind::Abs => ops::math::op_abs(require_input(inputs, 0, "Abs")?),
        OpKind::Floor => ops::math::op_floor(require_input(inputs, 0, "Floor")?),
        OpKind::Ceil => ops::math::op_ceil(require_input(inputs, 0, "Ceil")?),
        OpKind::Round => ops::math::op_round(require_input(inputs, 0, "Round")?),
        OpKind::Mod => {
            let a = require_input(inputs, 0, "Mod")?;
            let b = require_input(inputs, 1, "Mod")?;
            let fmod = get_attr_int(attrs, "fmod", 0) != 0;
            ops::math::op_mod(a, b, fmod)
        }
        OpKind::Sin => ops::math::op_sin(require_input(inputs, 0, "Sin")?),
        OpKind::Cos => ops::math::op_cos(require_input(inputs, 0, "Cos")?),
        OpKind::Reciprocal => ops::math::op_reciprocal(require_input(inputs, 0, "Reciprocal")?),
        OpKind::Sign => ops::math::op_sign(require_input(inputs, 0, "Sign")?),
        OpKind::Sum => {
            let refs: Vec<&Tensor> = inputs.iter().filter_map(|i| *i).collect();
            ops::math::op_sum(&refs)
        }
        OpKind::Mean => {
            let refs: Vec<&Tensor> = inputs.iter().filter_map(|i| *i).collect();
            ops::math::op_mean(&refs)
        }
        OpKind::And => {
            let a = require_input(inputs, 0, "And")?;
            let b = require_input(inputs, 1, "And")?;
            ops::math::op_and(a, b)
        }
        OpKind::Or => {
            let a = require_input(inputs, 0, "Or")?;
            let b = require_input(inputs, 1, "Or")?;
            ops::math::op_or(a, b)
        }
        OpKind::LogSoftmax => {
            let x = require_input(inputs, 0, "LogSoftmax")?;
            let axis = get_attr_int(attrs, "axis", -1);
            ops::math::op_log_softmax(x, axis)
        }
        _ => Err(OpError::UnsupportedOp(String::from("math"))),
    }
}

/// Dispatches Tier 2 comparison/selection operators.
fn dispatch_compare(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    _attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Equal => {
            let a = require_input(inputs, 0, "Equal")?;
            let b = require_input(inputs, 1, "Equal")?;
            ops::compare::op_equal(a, b)
        }
        OpKind::NotEqual => {
            let a = require_input(inputs, 0, "NotEqual")?;
            let b = require_input(inputs, 1, "NotEqual")?;
            ops::compare::op_not_equal(a, b)
        }
        OpKind::Less => {
            let a = require_input(inputs, 0, "Less")?;
            let b = require_input(inputs, 1, "Less")?;
            ops::compare::op_less(a, b)
        }
        OpKind::LessOrEqual => {
            let a = require_input(inputs, 0, "LessOrEqual")?;
            let b = require_input(inputs, 1, "LessOrEqual")?;
            ops::compare::op_less_or_equal(a, b)
        }
        OpKind::Greater => {
            let a = require_input(inputs, 0, "Greater")?;
            let b = require_input(inputs, 1, "Greater")?;
            ops::compare::op_greater(a, b)
        }
        OpKind::GreaterOrEqual => {
            let a = require_input(inputs, 0, "GreaterOrEqual")?;
            let b = require_input(inputs, 1, "GreaterOrEqual")?;
            ops::compare::op_greater_or_equal(a, b)
        }
        OpKind::Where => {
            let cond = require_input(inputs, 0, "Where")?;
            let x = require_input(inputs, 1, "Where")?;
            let y = require_input(inputs, 2, "Where")?;
            ops::compare::op_where(cond, x, y)
        }
        OpKind::Min => {
            let a = require_input(inputs, 0, "Min")?;
            let b = require_input(inputs, 1, "Min")?;
            ops::compare::op_min(a, b)
        }
        OpKind::Max => {
            let a = require_input(inputs, 0, "Max")?;
            let b = require_input(inputs, 1, "Max")?;
            ops::compare::op_max(a, b)
        }
        OpKind::Not => ops::compare::op_not(require_input(inputs, 0, "Not")?),
        _ => Err(OpError::UnsupportedOp(String::from("compare"))),
    }
}

/// Dispatches Tier 2 composite activations.
fn dispatch_composite_activation(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Gelu => ops::activations::op_gelu(require_input(inputs, 0, "Gelu")?),
        OpKind::LeakyRelu => {
            let x = require_input(inputs, 0, "LeakyRelu")?;
            let alpha = get_attr_float(attrs, "alpha", 0.01);
            ops::activations::op_leaky_relu(x, alpha)
        }
        OpKind::Elu => {
            let x = require_input(inputs, 0, "Elu")?;
            let alpha = get_attr_float(attrs, "alpha", 1.0);
            ops::activations::op_elu(x, alpha)
        }
        OpKind::Swish => ops::activations::op_swish(require_input(inputs, 0, "Swish")?),
        OpKind::HardSigmoid => {
            let x = require_input(inputs, 0, "HardSigmoid")?;
            let alpha = get_attr_float(attrs, "alpha", 0.2);
            let beta = get_attr_float(attrs, "beta", 0.5);
            ops::activations::op_hard_sigmoid(x, alpha, beta)
        }
        OpKind::HardSwish => {
            ops::activations::op_hard_swish(require_input(inputs, 0, "HardSwish")?)
        }
        OpKind::PRelu => {
            let x = require_input(inputs, 0, "PRelu")?;
            let slope = require_input(inputs, 1, "PRelu")?;
            ops::activations::op_prelu(x, slope)
        }
        OpKind::Mish => ops::activations::op_mish(require_input(inputs, 0, "Mish")?),
        _ => Err(OpError::UnsupportedOp(String::from("composite-activation"))),
    }
}

/// Reads an f32 tensor's raw_data as a Vec<f32>.
fn read_f32_tensor(tensor: &Tensor) -> Vec<f32> {
    if tensor.data_type == DataType::Float {
        let count = tensor.raw_data.len() / 4;
        (0..count)
            .map(|i| byte_io::read_f32(&tensor.raw_data, i))
            .collect()
    } else {
        Vec::new()
    }
}

/// Dispatches Phase 1 vision operators.
fn dispatch_vision(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Resize => {
            let x = require_input(inputs, 0, "Resize")?;
            // Inputs: X, roi, scales, sizes. roi is ignored (not implemented).
            let scales_tensor = optional_input(inputs, 2);
            let sizes_tensor = optional_input(inputs, 3);
            let scales_vec = scales_tensor.map(read_f32_tensor);
            let sizes_vec = sizes_tensor.map(read_i64_tensor).transpose()?;
            let mode_bytes = attrs
                .iter()
                .find(|a| a.name == "mode")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"nearest");
            let mode = core::str::from_utf8(mode_bytes).unwrap_or("nearest");
            ops::vision::op_resize(
                x,
                scales_vec.as_deref().filter(|v| !v.is_empty()),
                sizes_vec.as_deref().filter(|v| !v.is_empty()),
                mode,
            )
        }
        OpKind::DepthToSpace => {
            let x = require_input(inputs, 0, "DepthToSpace")?;
            let blocksize = get_attr_int(attrs, "blocksize", 0);
            let mode_bytes = attrs
                .iter()
                .find(|a| a.name == "mode")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"DCR");
            let mode = core::str::from_utf8(mode_bytes).unwrap_or("DCR");
            ops::vision::op_depth_to_space(x, blocksize, mode)
        }
        OpKind::SpaceToDepth => {
            let x = require_input(inputs, 0, "SpaceToDepth")?;
            let blocksize = get_attr_int(attrs, "blocksize", 0);
            ops::vision::op_space_to_depth(x, blocksize)
        }
        OpKind::RoiAlign => {
            let x = require_input(inputs, 0, "RoiAlign")?;
            let rois = require_input(inputs, 1, "RoiAlign")?;
            let batch_indices = require_input(inputs, 2, "RoiAlign")?;
            let oh = get_attr_int(attrs, "output_height", 1);
            let ow = get_attr_int(attrs, "output_width", 1);
            let sampling_ratio = get_attr_int(attrs, "sampling_ratio", 0);
            let spatial_scale = get_attr_float(attrs, "spatial_scale", 1.0);
            let mode_bytes = attrs
                .iter()
                .find(|a| a.name == "mode")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"avg");
            let mode = core::str::from_utf8(mode_bytes).unwrap_or("avg");
            ops::vision::op_roi_align(
                x,
                rois,
                batch_indices,
                oh,
                ow,
                sampling_ratio,
                spatial_scale,
                mode,
            )
        }
        OpKind::GridSample => {
            let x = require_input(inputs, 0, "GridSample")?;
            let grid = require_input(inputs, 1, "GridSample")?;
            let mode_bytes = attrs
                .iter()
                .find(|a| a.name == "mode")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"bilinear");
            let mode = core::str::from_utf8(mode_bytes).unwrap_or("bilinear");
            let pad_bytes = attrs
                .iter()
                .find(|a| a.name == "padding_mode")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"zeros");
            let padding_mode = core::str::from_utf8(pad_bytes).unwrap_or("zeros");
            let align_corners = get_attr_int(attrs, "align_corners", 0) != 0;
            ops::vision::op_grid_sample(x, grid, mode, padding_mode, align_corners)
        }
        OpKind::GlobalMaxPool => {
            ops::vision::op_global_max_pool(require_input(inputs, 0, "GlobalMaxPool")?)
        }
        OpKind::CenterCropPad => {
            let x = require_input(inputs, 0, "CenterCropPad")?;
            let shape_tensor = require_input(inputs, 1, "CenterCropPad")?;
            let shape = read_i64_tensor(shape_tensor)?;
            let axes_attr = get_attr_ints(attrs, "axes").map(|s| s.to_vec());
            ops::vision::op_center_crop_pad(x, &shape, axes_attr.as_deref())
        }
        _ => Err(OpError::UnsupportedOp(String::from("vision"))),
    }
}

/// Dispatches Phase 1 data-selection operators.
fn dispatch_data_select(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Compress => {
            let x = require_input(inputs, 0, "Compress")?;
            let cond = require_input(inputs, 1, "Compress")?;
            let axis = attrs
                .iter()
                .find(|a| a.name == "axis" && a.attr_type == AttributeType::Int)
                .map(|a| a.i);
            ops::data_select::op_compress(x, cond, axis)
        }
        OpKind::NonZero => ops::data_select::op_non_zero(require_input(inputs, 0, "NonZero")?),
        OpKind::Unique => {
            let x = require_input(inputs, 0, "Unique")?;
            let axis = attrs
                .iter()
                .find(|a| a.name == "axis" && a.attr_type == AttributeType::Int)
                .map(|a| a.i);
            let sorted = get_attr_int(attrs, "sorted", 1) != 0;
            ops::data_select::op_unique(x, axis, sorted)
        }
        OpKind::GatherElements => {
            let x = require_input(inputs, 0, "GatherElements")?;
            let idx = require_input(inputs, 1, "GatherElements")?;
            let axis = get_attr_int(attrs, "axis", 0);
            ops::data_select::op_gather_elements(x, idx, axis)
        }
        OpKind::ScatterElements => {
            let x = require_input(inputs, 0, "ScatterElements")?;
            let idx = require_input(inputs, 1, "ScatterElements")?;
            let upd = require_input(inputs, 2, "ScatterElements")?;
            let axis = get_attr_int(attrs, "axis", 0);
            let red_bytes = attrs
                .iter()
                .find(|a| a.name == "reduction")
                .map(|a| a.s.as_slice())
                .unwrap_or(b"none");
            let reduction = core::str::from_utf8(red_bytes).unwrap_or("none");
            ops::data_select::op_scatter_elements(x, idx, upd, axis, reduction)
        }
        _ => Err(OpError::UnsupportedOp(String::from("data-select"))),
    }
}

/// Dispatches Phase 1 normalization operators (Group/Instance).
fn dispatch_normalization_v2(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::GroupNormalization => {
            let x = require_input(inputs, 0, "GroupNormalization")?;
            let scale = require_input(inputs, 1, "GroupNormalization")?;
            let bias = optional_input(inputs, 2);
            let num_groups = get_attr_int(attrs, "num_groups", 1);
            let epsilon = get_attr_float(attrs, "epsilon", 1e-5);
            ops::normalization::op_group_normalization(x, scale, bias, num_groups, epsilon)
        }
        OpKind::InstanceNormalization => {
            let x = require_input(inputs, 0, "InstanceNormalization")?;
            let scale = require_input(inputs, 1, "InstanceNormalization")?;
            let bias = require_input(inputs, 2, "InstanceNormalization")?;
            let epsilon = get_attr_float(attrs, "epsilon", 1e-5);
            ops::normalization::op_instance_normalization(x, scale, bias, epsilon)
        }
        _ => Err(OpError::UnsupportedOp(String::from("normalization-v2"))),
    }
}

/// Dispatches TopK (multi-output).
fn dispatch_top_k(
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<Tensor>, OpError> {
    let x = require_input(inputs, 0, "TopK")?;
    // Opset 10+: K is an input tensor (Int64).
    let k = if let Some(k_t) = optional_input(inputs, 1) {
        read_i64_tensor(k_t)?.first().copied().unwrap_or(0)
    } else {
        get_attr_int(attrs, "k", 0)
    };
    let axis = get_attr_int(attrs, "axis", -1);
    let largest = get_attr_int(attrs, "largest", 1) != 0;
    let sorted = get_attr_int(attrs, "sorted", 1) != 0;
    ops::data_select::op_top_k(x, k, axis, largest, sorted)
}

/// Dispatches transformer building-block ops (Expand/Tile/OneHot/Einsum).
fn dispatch_transformer(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Expand => {
            let x = require_input(inputs, 0, "Expand")?;
            let shape_tensor = require_input(inputs, 1, "Expand")?;
            let shape = read_i64_tensor(shape_tensor)?;
            ops::transformer::op_expand(x, &shape)
        }
        OpKind::Tile => {
            let x = require_input(inputs, 0, "Tile")?;
            let repeats_tensor = require_input(inputs, 1, "Tile")?;
            let repeats = read_i64_tensor(repeats_tensor)?;
            ops::transformer::op_tile(x, &repeats)
        }
        OpKind::OneHot => {
            let indices = require_input(inputs, 0, "OneHot")?;
            let depth_tensor = require_input(inputs, 1, "OneHot")?;
            let values = require_input(inputs, 2, "OneHot")?;
            let depth = read_i64_tensor(depth_tensor)?
                .first()
                .copied()
                .ok_or_else(|| OpError::InvalidAttribute(String::from("OneHot depth missing")))?;
            let axis = get_attr_int(attrs, "axis", -1);
            ops::transformer::op_one_hot(indices, depth, values, axis)
        }
        OpKind::Einsum => {
            let equation_bytes = attrs
                .iter()
                .find(|a| a.name == "equation")
                .map(|a| a.s.as_slice())
                .ok_or_else(|| {
                    OpError::InvalidAttribute(String::from("Einsum requires 'equation'"))
                })?;
            let equation = core::str::from_utf8(equation_bytes)
                .map_err(|_| OpError::InvalidAttribute(String::from("Einsum equation utf-8")))?;
            let refs: Vec<&Tensor> = inputs.iter().filter_map(|i| *i).collect();
            ops::transformer::op_einsum(equation, &refs)
        }
        _ => Err(OpError::UnsupportedOp(String::from("transformer"))),
    }
}

/// Dispatches quantized operators.
fn dispatch_quantized(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::QuantizeLinear => {
            let x = require_input(inputs, 0, "QuantizeLinear")?;
            let scale = require_input(inputs, 1, "QuantizeLinear")?;
            let zero_point = optional_input(inputs, 2);
            ops::quantized::op_quantize_linear(x, scale, zero_point)
        }
        OpKind::DequantizeLinear => {
            let x = require_input(inputs, 0, "DequantizeLinear")?;
            let scale = require_input(inputs, 1, "DequantizeLinear")?;
            let zero_point = optional_input(inputs, 2);
            ops::quantized::op_dequantize_linear(x, scale, zero_point)
        }
        OpKind::QLinearMatMul => {
            let a = require_input(inputs, 0, "QLinearMatMul")?;
            let a_scale = require_input(inputs, 1, "QLinearMatMul")?;
            let a_zp = require_input(inputs, 2, "QLinearMatMul")?;
            let b = require_input(inputs, 3, "QLinearMatMul")?;
            let b_scale = require_input(inputs, 4, "QLinearMatMul")?;
            let b_zp = require_input(inputs, 5, "QLinearMatMul")?;
            let y_scale = require_input(inputs, 6, "QLinearMatMul")?;
            let y_zp = require_input(inputs, 7, "QLinearMatMul")?;
            ops::quantized::op_qlinear_matmul(a, a_scale, a_zp, b, b_scale, b_zp, y_scale, y_zp)
        }
        OpKind::QLinearConv => {
            let x = require_input(inputs, 0, "QLinearConv")?;
            let x_scale = require_input(inputs, 1, "QLinearConv")?;
            let x_zp = require_input(inputs, 2, "QLinearConv")?;
            let w = require_input(inputs, 3, "QLinearConv")?;
            let w_scale = require_input(inputs, 4, "QLinearConv")?;
            let w_zp = require_input(inputs, 5, "QLinearConv")?;
            let y_scale = require_input(inputs, 6, "QLinearConv")?;
            let y_zp = require_input(inputs, 7, "QLinearConv")?;
            let bias = optional_input(inputs, 8);
            let pads = get_attr_ints(attrs, "pads");
            let strides = get_attr_ints(attrs, "strides");
            let dilations = get_attr_ints(attrs, "dilations");
            let group = get_attr_int(attrs, "group", 1);
            ops::quantized::op_qlinear_conv(
                x, x_scale, x_zp, w, w_scale, w_zp, y_scale, y_zp, bias, pads, strides, dilations,
                group,
            )
        }
        _ => Err(OpError::UnsupportedOp(String::from("quantized"))),
    }
}

/// Dispatches Split (multi-output).
fn dispatch_split(
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
    output_count: usize,
) -> Result<Vec<Tensor>, OpError> {
    let x = require_input(inputs, 0, "Split")?;
    let axis = get_attr_int(attrs, "axis", 0);
    // Opset 13+: 'split' is an input tensor (i64).
    // Older: 'split' is an int list attribute.
    let split_input = match optional_input(inputs, 1) {
        Some(t) => Some(read_i64_tensor(t)?),
        None => None,
    };
    let split_attr = get_attr_ints(attrs, "split").map(|s| s.to_vec());
    let split_sizes = split_input.or(split_attr);
    // ONNX Split with no explicit sizes derives num_outputs from the
    // node's output arity.
    let num_outputs = if split_sizes.is_some() {
        None
    } else {
        Some(output_count)
    };
    ops::transformer::op_split(x, axis, split_sizes.as_deref(), num_outputs)
}

/// Dispatches LSTM (multi-output).
///
/// Phase 1 limitations: `sequence_lens` (input 4) and `P` peephole
/// weights (input 7) are not yet implemented and are rejected rather
/// than silently ignored. Phase 4 will plumb them through.
fn dispatch_lstm(
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<Tensor>, OpError> {
    let x = require_input(inputs, 0, "LSTM")?;
    let w = require_input(inputs, 1, "LSTM")?;
    let r = require_input(inputs, 2, "LSTM")?;
    let b = optional_input(inputs, 3);
    if optional_input(inputs, 4).is_some() {
        return Err(OpError::InvalidAttribute(String::from(
            "LSTM: variable sequence_lens not yet supported",
        )));
    }
    let h0 = optional_input(inputs, 5);
    let c0 = optional_input(inputs, 6);
    if optional_input(inputs, 7).is_some() {
        return Err(OpError::InvalidAttribute(String::from(
            "LSTM: peephole weights (P) not yet supported",
        )));
    }
    let direction_bytes = attrs
        .iter()
        .find(|a| a.name == "direction")
        .map(|a| a.s.as_slice())
        .unwrap_or(b"forward");
    let direction = core::str::from_utf8(direction_bytes).unwrap_or("forward");
    ops::recurrent::op_lstm(x, w, r, b, h0, c0, direction)
}

/// Dispatches GRU (multi-output).
///
/// Phase 1 limitations: `sequence_lens` (input 4) and the
/// `linear_before_reset` attribute are rejected rather than silently
/// ignored. Phase 4 will implement both.
fn dispatch_gru(
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<Tensor>, OpError> {
    let x = require_input(inputs, 0, "GRU")?;
    let w = require_input(inputs, 1, "GRU")?;
    let r = require_input(inputs, 2, "GRU")?;
    let b = optional_input(inputs, 3);
    if optional_input(inputs, 4).is_some() {
        return Err(OpError::InvalidAttribute(String::from(
            "GRU: variable sequence_lens not yet supported",
        )));
    }
    let h0 = optional_input(inputs, 5);
    if get_attr_int(attrs, "linear_before_reset", 0) != 0 {
        return Err(OpError::InvalidAttribute(String::from(
            "GRU: linear_before_reset=1 not yet supported",
        )));
    }
    let direction_bytes = attrs
        .iter()
        .find(|a| a.name == "direction")
        .map(|a| a.s.as_slice())
        .unwrap_or(b"forward");
    let direction = core::str::from_utf8(direction_bytes).unwrap_or("forward");
    ops::recurrent::op_gru(x, w, r, b, h0, direction)
}

/// Dispatches RNN (multi-output).
///
/// Phase 1 limitation: `sequence_lens` (input 4) is rejected rather
/// than silently ignored. Phase 4 will implement it.
fn dispatch_rnn(
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<Tensor>, OpError> {
    let x = require_input(inputs, 0, "RNN")?;
    let w = require_input(inputs, 1, "RNN")?;
    let r = require_input(inputs, 2, "RNN")?;
    let b = optional_input(inputs, 3);
    if optional_input(inputs, 4).is_some() {
        return Err(OpError::InvalidAttribute(String::from(
            "RNN: variable sequence_lens not yet supported",
        )));
    }
    let h0 = optional_input(inputs, 5);
    let direction_bytes = attrs
        .iter()
        .find(|a| a.name == "direction")
        .map(|a| a.s.as_slice())
        .unwrap_or(b"forward");
    let direction = core::str::from_utf8(direction_bytes).unwrap_or("forward");
    ops::recurrent::op_rnn(x, w, r, b, h0, direction)
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
            let axes_from_input = match optional_input(inputs, 1) {
                Some(t) => Some(read_i64_tensor(t)?),
                None => None,
            };
            let axes_attr = get_attr_ints(attrs, "axes");
            let axes = axes_from_input.as_deref().or(axes_attr).unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            operators::op_reduce_sum(x, axes, keepdims)
        }
        OpKind::ReduceMax => {
            let x = require_input(inputs, 0, "ReduceMax")?;
            let axes_from_input = match optional_input(inputs, 1) {
                Some(t) => Some(read_i64_tensor(t)?),
                None => None,
            };
            let axes_attr = get_attr_ints(attrs, "axes");
            let axes = axes_from_input.as_deref().or(axes_attr).unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::reduction::op_reduce_max(x, axes, keepdims)
        }
        OpKind::ReduceMin => {
            let x = require_input(inputs, 0, "ReduceMin")?;
            let axes_from_input = match optional_input(inputs, 1) {
                Some(t) => Some(read_i64_tensor(t)?),
                None => None,
            };
            let axes_attr = get_attr_ints(attrs, "axes");
            let axes = axes_from_input.as_deref().or(axes_attr).unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::reduction::op_reduce_min(x, axes, keepdims)
        }
        OpKind::ReduceProd => {
            let x = require_input(inputs, 0, "ReduceProd")?;
            let axes_from_input = match optional_input(inputs, 1) {
                Some(t) => Some(read_i64_tensor(t)?),
                None => None,
            };
            let axes_attr = get_attr_ints(attrs, "axes");
            let axes = axes_from_input.as_deref().or(axes_attr).unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::reduction::op_reduce_prod(x, axes, keepdims)
        }
        OpKind::ArgMax => {
            let x = require_input(inputs, 0, "ArgMax")?;
            let axis = get_attr_int(attrs, "axis", 0);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            let select_last = get_attr_int(attrs, "select_last_index", 0) != 0;
            ops::reduction::op_arg_max(x, axis, keepdims, select_last)
        }
        OpKind::ArgMin => {
            let x = require_input(inputs, 0, "ArgMin")?;
            let axis = get_attr_int(attrs, "axis", 0);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            let select_last = get_attr_int(attrs, "select_last_index", 0) != 0;
            ops::reduction::op_arg_min(x, axis, keepdims, select_last)
        }
        _ => Err(OpError::UnsupportedOp(String::from("reduction"))),
    }
}

/// Dispatches Tier 3 shape / data-movement operators.
fn dispatch_shape_data(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Shape => ops::shape_data::op_shape(require_input(inputs, 0, "Shape")?),
        OpKind::Size => ops::shape_data::op_size(require_input(inputs, 0, "Size")?),
        OpKind::Identity => ops::shape_data::op_identity(require_input(inputs, 0, "Identity")?),
        OpKind::Constant => {
            // ONNX Constant most commonly carries its payload in a
            // 'value' attribute of type TensorProto. We also accept an
            // explicit input tensor (initializer pass-through) or the
            // scalar 'value_float' / 'value_int' variants emitted by
            // some exporters. Scalar attributes produce a true 0-D
            // tensor.
            if let Some(t) = optional_input(inputs, 0) {
                return ops::shape_data::op_identity(t);
            }
            if let Some(proto) = get_attr_tensor(attrs, "value") {
                return tensor_from_proto(proto).ok_or_else(|| {
                    OpError::InvalidAttribute(String::from(
                        "Constant 'value' tensor has unsupported data_type",
                    ))
                });
            }
            if let Some(a) = attrs
                .iter()
                .find(|a| a.name == "value_float" && a.attr_type == AttributeType::Float)
            {
                return ops::shape_data::op_constant_of_shape(&[], a.f);
            }
            if let Some(a) = attrs
                .iter()
                .find(|a| a.name == "value_int" && a.attr_type == AttributeType::Int)
            {
                return ops::shape_data::op_constant_of_shape(&[], a.i as f32);
            }
            Err(OpError::InvalidAttribute(String::from(
                "Constant requires one of: 'value' (TensorProto), \
                 'value_float', 'value_int', or an input initializer",
            )))
        }
        OpKind::ConstantOfShape => {
            let shape_tensor = require_input(inputs, 0, "ConstantOfShape")?;
            let shape = read_i64_tensor(shape_tensor)?;
            // Prefer the tensor-typed 'value' attribute (standard ONNX
            // export). Fall back to 'value_float' / 'value_int' or 0.0.
            let value = if let Some(proto) = get_attr_tensor(attrs, "value") {
                let fill_tensor = tensor_from_proto(proto).ok_or_else(|| {
                    OpError::InvalidAttribute(String::from(
                        "ConstantOfShape 'value' tensor has unsupported data_type",
                    ))
                })?;
                read_scalar_f32(&fill_tensor)?
            } else {
                attrs
                    .iter()
                    .find_map(|a| match (a.name.as_str(), a.attr_type) {
                        ("value_float", AttributeType::Float) => Some(a.f),
                        ("value_int", AttributeType::Int) => Some(a.i as f32),
                        _ => None,
                    })
                    .unwrap_or(0.0)
            };
            ops::shape_data::op_constant_of_shape(&shape, value)
        }
        OpKind::Range => {
            let start_t = require_input(inputs, 0, "Range")?;
            let limit_t = require_input(inputs, 1, "Range")?;
            let delta_t = require_input(inputs, 2, "Range")?;
            let start = read_scalar_f32(start_t)?;
            let limit = read_scalar_f32(limit_t)?;
            let delta = read_scalar_f32(delta_t)?;
            ops::shape_data::op_range(start, limit, delta)
        }
        OpKind::Trilu => {
            let x = require_input(inputs, 0, "Trilu")?;
            // k is an optional second input (Int64 scalar)
            let k = optional_input(inputs, 1)
                .map(|t| {
                    if t.data_type == DataType::Int64 && t.raw_data.len() >= I64_SIZE {
                        byte_io::read_i64(&t.raw_data, 0)
                    } else {
                        0
                    }
                })
                .unwrap_or(0);
            let upper = get_attr_int(attrs, "upper", 1) != 0;
            ops::shape_data::op_trilu(x, k, upper)
        }
        OpKind::CumSum => {
            let x = require_input(inputs, 0, "CumSum")?;
            let axis_tensor = require_input(inputs, 1, "CumSum")?;
            let axis = if axis_tensor.data_type == DataType::Int64
                && axis_tensor.raw_data.len() >= I64_SIZE
            {
                byte_io::read_i64(&axis_tensor.raw_data, 0)
            } else {
                0
            };
            let exclusive = get_attr_int(attrs, "exclusive", 0) != 0;
            let reverse = get_attr_int(attrs, "reverse", 0) != 0;
            ops::shape_data::op_cumsum(x, axis, exclusive, reverse)
        }
        OpKind::GatherND => {
            let data = require_input(inputs, 0, "GatherND")?;
            let idx = require_input(inputs, 1, "GatherND")?;
            let batch_dims = get_attr_int(attrs, "batch_dims", 0);
            ops::shape_data::op_gather_nd(data, idx, batch_dims)
        }
        OpKind::ScatterND => {
            let data = require_input(inputs, 0, "ScatterND")?;
            let idx = require_input(inputs, 1, "ScatterND")?;
            let upd = require_input(inputs, 2, "ScatterND")?;
            ops::shape_data::op_scatter_nd(data, idx, upd)
        }
        _ => Err(OpError::UnsupportedOp(String::from("shape_data"))),
    }
}

/// Reads a scalar f32 from a 1-element tensor (Float or Int64).
fn read_scalar_f32(t: &Tensor) -> Result<f32, OpError> {
    match t.data_type {
        DataType::Float if t.raw_data.len() >= 4 => Ok(byte_io::read_f32(&t.raw_data, 0)),
        DataType::Int64 if t.raw_data.len() >= I64_SIZE => {
            Ok(byte_io::read_i64(&t.raw_data, 0) as f32)
        }
        DataType::Int32 if t.raw_data.len() >= 4 => Ok(byte_io::read_i32(&t.raw_data, 0) as f32),
        _ => Err(OpError::ShapeMismatch(String::from(
            "expected scalar numeric tensor",
        ))),
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
            let shape = read_i64_tensor(shape_tensor)?;
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
            let axes = match optional_input(inputs, 1) {
                Some(at) => Some(read_i64_tensor(at)?),
                None => None,
            };
            operators::op_squeeze(t, axes.as_deref())
        }
        OpKind::Unsqueeze => {
            let t = require_input(inputs, 0, "Unsqueeze")?;
            let axes_tensor = require_input(inputs, 1, "Unsqueeze")?;
            let axes = read_i64_tensor(axes_tensor)?;
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
            let starts = read_i64_tensor(starts_tensor)?;
            let ends = read_i64_tensor(ends_tensor)?;
            let axes = match optional_input(inputs, 3) {
                Some(t) => Some(read_i64_tensor(t)?),
                None => None,
            };
            let steps = match optional_input(inputs, 4) {
                Some(t) => Some(read_i64_tensor(t)?),
                None => None,
            };
            operators::op_slice(t, &starts, &ends, axes.as_deref(), steps.as_deref())
        }
        OpKind::Pad => {
            let t = require_input(inputs, 0, "Pad")?;
            let pads_tensor = require_input(inputs, 1, "Pad")?;
            let constant_tensor = optional_input(inputs, 2);
            let pads = read_i64_tensor(pads_tensor)?;
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

/// Dispatches Phase 2 generative / normalization / random / reduction ops.
fn dispatch_generative(
    kind: OpKind,
    inputs: &[Option<&Tensor>],
    attrs: &[AttributeProto],
) -> Result<Tensor, OpError> {
    match kind {
        OpKind::Softplus => ops::generative::op_softplus(require_input(inputs, 0, "Softplus")?),
        OpKind::EyeLike => {
            let x = require_input(inputs, 0, "EyeLike")?;
            let k = get_attr_int(attrs, "k", 0);
            ops::generative::op_eye_like(x, k)
        }
        OpKind::Dropout => ops::generative::op_dropout(require_input(inputs, 0, "Dropout")?),
        OpKind::RandomNormal => {
            let shape = get_attr_ints(attrs, "shape").unwrap_or(&[]);
            let seed = get_attr_float(attrs, "seed", 0.0);
            let mean = get_attr_float(attrs, "mean", 0.0);
            let scale = get_attr_float(attrs, "scale", 1.0);
            ops::generative::op_random_normal(shape, seed, mean, scale)
        }
        OpKind::RandomNormalLike => {
            let x = require_input(inputs, 0, "RandomNormalLike")?;
            let seed = get_attr_float(attrs, "seed", 0.0);
            let mean = get_attr_float(attrs, "mean", 0.0);
            let scale = get_attr_float(attrs, "scale", 1.0);
            ops::generative::op_random_normal_like(x, seed, mean, scale)
        }
        OpKind::RandomUniform => {
            let shape = get_attr_ints(attrs, "shape").unwrap_or(&[]);
            let seed = get_attr_float(attrs, "seed", 0.0);
            let low = get_attr_float(attrs, "low", 0.0);
            let high = get_attr_float(attrs, "high", 1.0);
            ops::generative::op_random_uniform(shape, seed, low, high)
        }
        OpKind::RandomUniformLike => {
            let x = require_input(inputs, 0, "RandomUniformLike")?;
            let seed = get_attr_float(attrs, "seed", 0.0);
            let low = get_attr_float(attrs, "low", 0.0);
            let high = get_attr_float(attrs, "high", 1.0);
            ops::generative::op_random_uniform_like(x, seed, low, high)
        }
        OpKind::Bernoulli => {
            let x = require_input(inputs, 0, "Bernoulli")?;
            let seed = get_attr_float(attrs, "seed", 0.0);
            ops::generative::op_bernoulli(x, seed)
        }
        OpKind::Multinomial => {
            let x = require_input(inputs, 0, "Multinomial")?;
            let sample_size = get_attr_int(attrs, "sample_size", 1);
            let seed = get_attr_float(attrs, "seed", 0.0);
            ops::generative::op_multinomial(x, sample_size, seed)
        }
        OpKind::RMSNormalization => {
            let x = require_input(inputs, 0, "RMSNormalization")?;
            let scale = require_input(inputs, 1, "RMSNormalization")?;
            let axis = get_attr_int(attrs, "axis", -1);
            let epsilon = get_attr_float(attrs, "epsilon", 1e-5);
            ops::generative::op_rms_normalization(x, scale, axis, epsilon)
        }
        OpKind::LpNormalization => {
            let x = require_input(inputs, 0, "LpNormalization")?;
            let axis = get_attr_int(attrs, "axis", -1);
            let p = get_attr_int(attrs, "p", 2);
            ops::generative::op_lp_normalization(x, axis, p)
        }
        OpKind::MeanVarianceNormalization => {
            let x = require_input(inputs, 0, "MeanVarianceNormalization")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            ops::generative::op_mean_variance_normalization(x, axes)
        }
        OpKind::ReduceL1 => {
            let x = require_input(inputs, 0, "ReduceL1")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::generative::op_reduce_l1(x, axes, keepdims)
        }
        OpKind::ReduceL2 => {
            let x = require_input(inputs, 0, "ReduceL2")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::generative::op_reduce_l2(x, axes, keepdims)
        }
        OpKind::ReduceLogSum => {
            let x = require_input(inputs, 0, "ReduceLogSum")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::generative::op_reduce_log_sum(x, axes, keepdims)
        }
        OpKind::ReduceLogSumExp => {
            let x = require_input(inputs, 0, "ReduceLogSumExp")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::generative::op_reduce_log_sum_exp(x, axes, keepdims)
        }
        OpKind::ReduceSumSquare => {
            let x = require_input(inputs, 0, "ReduceSumSquare")?;
            let axes = get_attr_ints(attrs, "axes").unwrap_or(&[]);
            let keepdims = get_attr_int(attrs, "keepdims", 1) != 0;
            ops::generative::op_reduce_sum_square(x, axes, keepdims)
        }
        OpKind::MatMulInteger => {
            let a = require_input(inputs, 0, "MatMulInteger")?;
            let b = require_input(inputs, 1, "MatMulInteger")?;
            let a_zp = optional_input(inputs, 2);
            let b_zp = optional_input(inputs, 3);
            ops::generative::op_matmul_integer(a, b, a_zp, b_zp)
        }
        _ => Err(OpError::UnsupportedOp(String::from("generative"))),
    }
}

/// Dispatches `DynamicQuantizeLinear` which produces three outputs:
/// quantized tensor (u8), scale (f32 scalar), zero-point (u8 scalar).
fn dispatch_dynamic_quantize(inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, OpError> {
    let x = require_input(inputs, 0, "DynamicQuantizeLinear")?;
    let (q, s, z) = ops::generative::op_dynamic_quantize_linear(x)?;
    Ok(alloc::vec![q, s, z])
}

/// Reads an integer tensor's raw_data as a `Vec<i64>`.
///
/// Accepts Int64 (native) and Int32 (converted). Returns an error for
/// other dtypes so callers such as Expand/Tile/Reshape fail fast rather
/// than silently no-op'ing on an unexpected dtype.
fn read_i64_tensor(tensor: &Tensor) -> Result<Vec<i64>, OpError> {
    match tensor.data_type {
        DataType::Int64 => {
            let count = tensor.raw_data.len() / I64_SIZE;
            Ok((0..count)
                .map(|i| byte_io::read_i64(&tensor.raw_data, i))
                .collect())
        }
        DataType::Int32 => {
            let count = tensor.raw_data.len() / 4;
            Ok((0..count)
                .map(|i| byte_io::read_i32(&tensor.raw_data, i) as i64)
                .collect())
        }
        _ => Err(OpError::ShapeMismatch(String::from(
            "expected Int32 or Int64 tensor",
        ))),
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[w_init],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let result = execute_graph(
            &exec_graph,
            &[],
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        );
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

        let result = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        );
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

        let _results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            Some(test_yield),
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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
        let vals = read_i64_tensor(&tensor).unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

        let results = execute_graph(
            &exec_graph,
            &inputs,
            &[],
            None,
            None,
            &OperatorBudget::DEFAULT,
            &NullTimeSource,
        )
        .unwrap();
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

    // ---- Tensor-valued 'value' attribute for Constant / ConstantOfShape ----

    fn make_value_tensor_attr(proto: TensorProto) -> AttributeProto {
        AttributeProto {
            name: String::from("value"),
            attr_type: AttributeType::Tensor,
            t: Some(alloc::boxed::Box::new(proto)),
            ..AttributeProto::default()
        }
    }

    #[test]
    fn test_get_attr_tensor_finds_tensor() {
        let proto = TensorProto {
            dims: vec![2],
            data_type: 1,
            float_data: vec![1.0, 2.0],
            ..TensorProto::default()
        };
        let attrs = vec![make_value_tensor_attr(proto)];
        let t = get_attr_tensor(&attrs, "value").expect("tensor attr present");
        assert_eq!(t.dims, vec![2i64]);
        assert!(get_attr_tensor(&attrs, "missing").is_none());
    }

    #[test]
    fn test_dispatch_constant_with_tensor_value_attr() {
        let proto = TensorProto {
            dims: vec![3],
            data_type: 1, // FLOAT
            float_data: vec![10.0, 20.0, 30.0],
            ..TensorProto::default()
        };
        let attrs = vec![make_value_tensor_attr(proto)];
        let inputs: [Option<&Tensor>; 0] = [];
        let out = dispatch_shape_data(OpKind::Constant, &inputs, &attrs).unwrap();
        assert_eq!(out.data_type, DataType::Float);
        assert_eq!(out.shape.dims, vec![3i64]);
        assert_eq!(out.raw_data.len(), 12);
        assert!((byte_io::read_f32(&out.raw_data, 0) - 10.0).abs() < f32::EPSILON);
        assert!((byte_io::read_f32(&out.raw_data, 1) - 20.0).abs() < f32::EPSILON);
        assert!((byte_io::read_f32(&out.raw_data, 2) - 30.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dispatch_constant_prefers_tensor_over_value_float() {
        // Both 'value' (tensor) and 'value_float' are provided; the
        // tensor attribute should win.
        let proto = TensorProto {
            dims: vec![1],
            data_type: 1,
            float_data: vec![99.0],
            ..TensorProto::default()
        };
        let attrs = vec![
            make_value_tensor_attr(proto),
            AttributeProto {
                name: String::from("value_float"),
                attr_type: AttributeType::Float,
                f: 0.5,
                ..AttributeProto::default()
            },
        ];
        let inputs: [Option<&Tensor>; 0] = [];
        let out = dispatch_shape_data(OpKind::Constant, &inputs, &attrs).unwrap();
        assert!((byte_io::read_f32(&out.raw_data, 0) - 99.0).abs() < f32::EPSILON);
    }

    #[test]
    fn test_dispatch_constant_of_shape_with_tensor_value_attr() {
        // Fill value tensor: scalar float 7.0.
        let fill_proto = TensorProto {
            dims: vec![1],
            data_type: 1,
            float_data: vec![7.0],
            ..TensorProto::default()
        };
        let attrs = vec![make_value_tensor_attr(fill_proto)];

        // Input shape tensor: Int64 [2, 3].
        let mut shape_raw = alloc::vec![0u8; 2 * I64_SIZE];
        byte_io::write_i64(&mut shape_raw, 0, 2);
        byte_io::write_i64(&mut shape_raw, 1, 3);
        let shape_tensor = Tensor {
            data_type: DataType::Int64,
            shape: TensorShape::new(vec![2]),
            name: String::from("shape"),
            raw_data: shape_raw,
        };
        let inputs = [Some(&shape_tensor)];
        let out = dispatch_shape_data(OpKind::ConstantOfShape, &inputs, &attrs).unwrap();
        assert_eq!(out.shape.dims, vec![2i64, 3]);
        assert_eq!(out.data_type, DataType::Float);
        for i in 0..6 {
            assert!((byte_io::read_f32(&out.raw_data, i) - 7.0).abs() < f32::EPSILON);
        }
    }
}
