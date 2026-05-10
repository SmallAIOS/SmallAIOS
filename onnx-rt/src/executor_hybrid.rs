// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Hybrid CPU/GPU graph executor.
//!
//! The standard `execute_graph` performs op-by-op dispatch with all
//! intermediate tensors copied back to host memory between ops. For
//! vision models that means every Conv pays a host↔device round-trip
//! and adjacent BatchNorm/Relu/Pool/Add ops only ever run on CPU.
//!
//! This module implements a hybrid executor that tracks each named
//! tensor's residency (host vs device) and keeps activations on the
//! GPU across consecutive GPU-supported ops. CPU-only ops still work
//! — the runtime copies their inputs back to host transparently and
//! the next GPU op picks up from the resulting host tensor.
//!
//! Wired in only when `SessionConfig::gpu_residency = GpuResidency::Hybrid`.

#![cfg(feature = "cuda")]

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

use crate::cuda::{
    self,
    activation::{gpu_clip, gpu_relu},
    batchnorm::gpu_batchnorm,
    elementwise::gpu_add,
    ffi,
    gpu_executor::{tensor_to_device, DeviceTensor},
    graph::CudaGraph,
    graph_cache::{CudaGraphCache, GraphEntry, GraphKey},
    memory::DeviceBuffer,
    pool::{gpu_averagepool, gpu_globalaveragepool, gpu_maxpool},
    streams::StreamPool,
    CudaRuntime,
};
use crate::executor;
use crate::graph::ExecutionGraph;
use crate::onnx_types::{AttributeProto, TensorProto};
use crate::operators::{BatchNormAttrs, ConvAttrs, OpError, PoolAttrs};
use crate::session::{CudaGraphMode, InferenceOutput, SessionError};
use crate::tensor::{DataType, Tensor, TensorShape};

/// Per-named-value residency: a tensor lives on either host or device
/// at any moment, never both. Branching (multiple consumers) shares the
/// same `Arc<DeviceTensor>` to avoid device→device copies.
enum ValueLocation {
    Host(Tensor),
    Device(Arc<DeviceTensor>),
}

impl ValueLocation {
    fn dtype(&self) -> DataType {
        match self {
            ValueLocation::Host(t) => t.data_type,
            ValueLocation::Device(d) => d.dtype,
        }
    }
}

/// Returns `true` when the named operator with the given input dtype
/// has a device implementation in this build.
fn gpu_op_supported(op_type: &str, dtype: DataType) -> bool {
    if !matches!(dtype, DataType::Float | DataType::BFloat16) {
        return false;
    }
    matches!(
        op_type,
        "Conv"
            | "Gemm"
            | "MatMul"
            | "BatchNormalization"
            | "Relu"
            | "Clip"
            | "MaxPool"
            | "AveragePool"
            | "GlobalAveragePool"
            | "Add"
    )
}

/// Read a named value from the value map, materializing on host. If the
/// canonical copy is on device, copies device→host *and replaces the
/// device entry with the host one* so subsequent host accesses are
/// cheap.
fn ensure_host(map: &mut BTreeMap<String, ValueLocation>, name: &str) -> Result<(), SessionError> {
    if let Some(ValueLocation::Device(dt)) = map.get(name) {
        #[cfg(feature = "gpu-profile")]
        let _start = std::time::Instant::now();
        #[cfg(feature = "gpu-profile")]
        let _bytes = dt.byte_size() as u64;
        let host = dt
            .to_host()
            .map_err(|e| SessionError::ExecutionFailed(format!("{}: device→host: {}", name, e)))?;
        #[cfg(feature = "gpu-profile")]
        crate::cuda::profile::record_memcpy(
            crate::cuda::profile::EventKind::DeviceToHost,
            _bytes,
            _start,
        );
        map.insert(String::from(name), ValueLocation::Host(host));
    }
    Ok(())
}

/// Read a named value from the value map, materializing on device. If
/// the canonical copy is on host, copies host→device and replaces the
/// host entry.
fn ensure_device(
    map: &mut BTreeMap<String, ValueLocation>,
    name: &str,
    rt: &CudaRuntime,
) -> Result<(), SessionError> {
    if let Some(ValueLocation::Host(t)) = map.get(name) {
        #[cfg(feature = "gpu-profile")]
        let _start = std::time::Instant::now();
        #[cfg(feature = "gpu-profile")]
        let _bytes = t.raw_data.len() as u64;
        let dt = tensor_to_device(t, rt)
            .map_err(|e| SessionError::ExecutionFailed(format!("{}: host→device: {}", name, e)))?;
        #[cfg(feature = "gpu-profile")]
        crate::cuda::profile::record_memcpy(
            crate::cuda::profile::EventKind::HostToDevice,
            _bytes,
            _start,
        );
        map.insert(String::from(name), ValueLocation::Device(Arc::new(dt)));
    }
    Ok(())
}

#[allow(dead_code)]
fn require_host<'a>(
    map: &'a BTreeMap<String, ValueLocation>,
    name: &str,
) -> Result<&'a Tensor, SessionError> {
    match map.get(name) {
        Some(ValueLocation::Host(t)) => Ok(t),
        Some(ValueLocation::Device(_)) => Err(SessionError::ExecutionFailed(format!(
            "internal: expected '{}' on host but found device",
            name
        ))),
        None => Err(SessionError::ExecutionFailed(format!(
            "value '{}' not produced",
            name
        ))),
    }
}

fn require_device(
    map: &BTreeMap<String, ValueLocation>,
    name: &str,
) -> Result<Arc<DeviceTensor>, SessionError> {
    match map.get(name) {
        Some(ValueLocation::Device(d)) => Ok(d.clone()),
        Some(ValueLocation::Host(_)) => Err(SessionError::ExecutionFailed(format!(
            "internal: expected '{}' on device but found host",
            name
        ))),
        None => Err(SessionError::ExecutionFailed(format!(
            "value '{}' not produced",
            name
        ))),
    }
}

fn op_err_to_session(node_name: &str, e: OpError) -> SessionError {
    SessionError::ExecutionFailed(format!("{}: {}", node_name, e))
}

fn cuda_err_to_session(node_name: &str, e: cuda::CudaError) -> SessionError {
    SessionError::ExecutionFailed(format!("{}: {}", node_name, e))
}

/// Parsed Gemm attribute bundle (`transA`, `transB`, `alpha`, `beta`).
struct GemmAttrs {
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
}

impl GemmAttrs {
    fn parse(attrs: &[AttributeProto]) -> Self {
        let mut g = GemmAttrs {
            trans_a: false,
            trans_b: false,
            alpha: 1.0,
            beta: 1.0,
        };
        for a in attrs {
            match a.name.as_str() {
                "transA" => g.trans_a = a.i != 0,
                "transB" => g.trans_b = a.i != 0,
                "alpha" => g.alpha = a.f,
                "beta" => g.beta = a.f,
                _ => {}
            }
        }
        g
    }
}

/// Extract the optional `(min, max)` clip bounds from `Clip` attributes.
/// Returns `None` if either bound is absent — the caller must fall back
/// to the CPU dispatch path because opset < 11 always uses attributes.
fn parse_clip_bounds(attrs: &[AttributeProto]) -> Option<(f32, f32)> {
    let mut min = None;
    let mut max = None;
    for a in attrs {
        if a.name == "min" {
            min = Some(a.f);
        }
        if a.name == "max" {
            max = Some(a.f);
        }
    }
    match (min, max) {
        (Some(mn), Some(mx)) => Some((mn, mx)),
        _ => None,
    }
}

/// GPU dispatch for `Conv`. Returns `Ok(None)` if the input arity is wrong.
fn gpu_dispatch_conv(
    rt: &CudaRuntime,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    if inputs.len() < 2 {
        return Ok(None);
    }
    let conv_attrs =
        ConvAttrs::from_attributes(attrs).map_err(|e| op_err_to_session(node_name, e))?;
    let bias = inputs.get(2).map(|d| d.as_ref());
    cuda::gpu_executor::gpu_conv2d_device(
        rt,
        &inputs[0],
        &inputs[1],
        bias,
        &conv_attrs.pads[..2],
        &conv_attrs.strides,
        &conv_attrs.dilations,
        conv_attrs.group,
    )
    .map(Some)
    .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `Gemm`. Returns `Ok(None)` if input arity is wrong.
fn gpu_dispatch_gemm(
    rt: &CudaRuntime,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    if inputs.len() < 2 {
        return Ok(None);
    }
    let g = GemmAttrs::parse(attrs);
    let bias = inputs.get(2).map(|d| d.as_ref());
    cuda::gpu_executor::gpu_gemm_device(
        rt, &inputs[0], &inputs[1], g.trans_a, g.trans_b, g.alpha, g.beta, bias,
    )
    .map(Some)
    .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `MatMul`. Returns `Ok(None)` if input arity is wrong.
fn gpu_dispatch_matmul(
    rt: &CudaRuntime,
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    if inputs.len() != 2 {
        return Ok(None);
    }
    cuda::gpu_executor::gpu_gemm_device(rt, &inputs[0], &inputs[1], false, false, 1.0, 0.0, None)
        .map(Some)
        .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `BatchNormalization`. Returns `Ok(None)` if input
/// arity is wrong.
fn gpu_dispatch_batchnorm(
    rt: &CudaRuntime,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    if inputs.len() != 5 {
        return Ok(None);
    }
    let bn = BatchNormAttrs::from_attributes(attrs).map_err(|e| op_err_to_session(node_name, e))?;
    gpu_batchnorm(
        rt, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], bn.epsilon,
    )
    .map(Some)
    .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `Relu`. Returns `Ok(None)` if input arity is wrong.
fn gpu_dispatch_relu(
    rt: &CudaRuntime,
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    if inputs.len() != 1 {
        return Ok(None);
    }
    gpu_relu(rt, &inputs[0])
        .map(Some)
        .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `Clip`. Falls back to CPU when the opset >= 11
/// input form is used (min/max not in attributes).
fn gpu_dispatch_clip(
    rt: &CudaRuntime,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    // Clip in opset >= 11 takes min/max as inputs (idx 1, 2);
    // here all three inputs would already be on device, but
    // min/max are scalars so easier to require the attribute
    // form. Fall back to CPU otherwise.
    let (mn, mx) = match parse_clip_bounds(attrs) {
        Some(b) => b,
        None => return Ok(None),
    };
    gpu_clip(rt, &inputs[0], mn, mx)
        .map(Some)
        .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `MaxPool`. Returns the result tensor or a parse error.
fn gpu_dispatch_maxpool(
    rt: &CudaRuntime,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    let p = PoolAttrs::from_attributes(attrs, true).map_err(|e| op_err_to_session(node_name, e))?;
    gpu_maxpool(rt, &inputs[0], &p)
        .map(Some)
        .map_err(|e| cuda_err_to_session(node_name, e))
}

/// GPU dispatch for `AveragePool`. Returns the result tensor or a parse error.
fn gpu_dispatch_averagepool(
    rt: &CudaRuntime,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    let p = PoolAttrs::from_attributes(attrs, true).map_err(|e| op_err_to_session(node_name, e))?;
    gpu_averagepool(rt, &inputs[0], &p)
        .map(Some)
        .map_err(|e| cuda_err_to_session(node_name, e))
}

/// Try to dispatch a single op on GPU. Returns `Some(out_dt)` on
/// success, `Ok(None)` if the op shape/dtype isn't GPU-supportable
/// (fallback expected), or `Err` for hard failures.
#[allow(clippy::too_many_arguments)]
fn try_gpu_dispatch(
    rt: &CudaRuntime,
    op_type: &str,
    attrs: &[AttributeProto],
    inputs: &[Arc<DeviceTensor>],
    node_name: &str,
) -> Result<Option<DeviceTensor>, SessionError> {
    match op_type {
        "Conv" => gpu_dispatch_conv(rt, attrs, inputs, node_name),
        "Gemm" => gpu_dispatch_gemm(rt, attrs, inputs, node_name),
        "MatMul" => gpu_dispatch_matmul(rt, inputs, node_name),
        "BatchNormalization" => gpu_dispatch_batchnorm(rt, attrs, inputs, node_name),
        "Relu" => gpu_dispatch_relu(rt, inputs, node_name),
        "Clip" => gpu_dispatch_clip(rt, attrs, inputs, node_name),
        "MaxPool" => gpu_dispatch_maxpool(rt, attrs, inputs, node_name),
        "AveragePool" => gpu_dispatch_averagepool(rt, attrs, inputs, node_name),
        "GlobalAveragePool" => gpu_globalaveragepool(rt, &inputs[0])
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e)),
        "Add" => gpu_add(rt, &inputs[0], &inputs[1])
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e)),
        _ => Ok(None),
    }
}

/// Hybrid CPU/GPU execution. Mirrors `executor::execute_graph`'s
/// signature but routes GPU-supported ops through the device path with
/// residency tracking.
///
/// `device_initializer_cache` (when `Some`) is a per-Session preloaded
/// map of initializer name → device-resident tensor. Names found in
/// the cache are inserted into the value map as already-on-device
/// values, skipping the per-inference TensorProto decode + host→device
/// memcpy for those initializers. Names NOT in the cache (or `None`
/// cache) fall back to per-inference decoding from the protobuf.
pub fn execute_graph_hybrid(
    graph: &ExecutionGraph,
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    runtime: &CudaRuntime,
    device_initializer_cache: Option<&BTreeMap<String, Arc<DeviceTensor>>>,
) -> Result<Vec<InferenceOutput>, SessionError> {
    let mut value_map = build_initial_value_map(inputs, initializers, device_initializer_cache);
    run_op_loop(graph, &mut value_map, runtime)?;
    extract_host_outputs(&graph.output_names, &mut value_map)
}

/// Seed a fresh value map with the per-inference inputs and the
/// (possibly cached) device-resident initializers. Initializers
/// missing from the cache fall back to a host-side decode of their
/// `TensorProto`.
fn build_initial_value_map(
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    device_initializer_cache: Option<&BTreeMap<String, Arc<DeviceTensor>>>,
) -> BTreeMap<String, ValueLocation> {
    let mut value_map: BTreeMap<String, ValueLocation> = BTreeMap::new();

    for (name, tensor) in inputs {
        value_map.insert(name.clone(), ValueLocation::Host(tensor.clone()));
    }

    for init in initializers {
        if let Some(cache) = device_initializer_cache {
            if let Some(d) = cache.get(&init.name) {
                value_map.insert(init.name.clone(), ValueLocation::Device(d.clone()));
                continue;
            }
        }
        if let Some(tensor) = executor::tensor_from_proto(init) {
            value_map.insert(init.name.clone(), ValueLocation::Host(tensor));
        }
    }

    value_map
}

/// Determine whether a node is eligible for GPU dispatch given the
/// dtype of its first non-empty input. Returns `false` when the input
/// has no recorded dtype yet.
fn is_node_gpu_eligible(
    node: &crate::graph::ExecutionNode,
    value_map: &BTreeMap<String, ValueLocation>,
) -> bool {
    let first_input_dtype = node
        .inputs
        .iter()
        .find(|n| !n.is_empty())
        .and_then(|n| value_map.get(n).map(|v| v.dtype()));
    match first_input_dtype {
        Some(dt) => gpu_op_supported(&node.op_type, dt),
        None => false,
    }
}

/// Pull every non-empty input of `node` to device. Returns `true` when
/// every input could be materialized on the device, `false` if any
/// input failed (caller falls through to CPU dispatch).
fn materialize_inputs_on_device(
    node: &crate::graph::ExecutionNode,
    value_map: &mut BTreeMap<String, ValueLocation>,
    runtime: &CudaRuntime,
) -> bool {
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        if ensure_device(value_map, name, runtime).is_err() {
            return false;
        }
    }
    true
}

/// Collect device-resident `DeviceTensor` references for every
/// non-empty input. Assumes [`materialize_inputs_on_device`] just
/// returned `true`; any missing entry is treated as a hard error.
fn collect_device_inputs(
    node: &crate::graph::ExecutionNode,
    value_map: &BTreeMap<String, ValueLocation>,
) -> Result<Vec<Arc<DeviceTensor>>, SessionError> {
    let mut device_inputs: Vec<Arc<DeviceTensor>> = Vec::new();
    for name in &node.inputs {
        if name.is_empty() {
            continue;
        }
        device_inputs.push(require_device(value_map, name)?);
    }
    Ok(device_inputs)
}

/// Insert the GPU result tensor into the value map under the node's
/// (single) output name. No-op when the output name is empty.
fn store_gpu_output(
    node: &crate::graph::ExecutionNode,
    out_dt: DeviceTensor,
    value_map: &mut BTreeMap<String, ValueLocation>,
) {
    if let Some(name) = node.outputs.first() {
        if !name.is_empty() {
            value_map.insert(name.clone(), ValueLocation::Device(Arc::new(out_dt)));
        }
    }
}

/// Attempt to run `node` on the GPU. Returns `Ok(true)` when the GPU
/// path produced and stored a single-output result, `Ok(false)` when
/// the caller must fall back to CPU dispatch (ineligible op,
/// materialization failure, multi-output op, or `try_gpu_dispatch`
/// returning `None`).
fn try_run_node_on_gpu(
    node: &crate::graph::ExecutionNode,
    value_map: &mut BTreeMap<String, ValueLocation>,
    runtime: &CudaRuntime,
) -> Result<bool, SessionError> {
    if !is_node_gpu_eligible(node, value_map) {
        return Ok(false);
    }
    if !materialize_inputs_on_device(node, value_map, runtime) {
        return Ok(false);
    }
    let device_inputs = collect_device_inputs(node, value_map)?;

    #[cfg(feature = "gpu-profile")]
    let _op_start = std::time::Instant::now();
    let dispatched = try_gpu_dispatch(
        runtime,
        &node.op_type,
        &node.attributes,
        &device_inputs,
        &node.name,
    )?;
    #[cfg(feature = "gpu-profile")]
    if dispatched.is_some() {
        crate::cuda::profile::record_op(&node.op_type, _op_start);
    }

    // GPU dispatchers currently produce a single output tensor. ONNX
    // permits multi-output ops (e.g. `MaxPool` with an indices output);
    // for those we fall through to the CPU executor which can populate
    // every output name. Single-output ops take the device-resident
    // fast path.
    match dispatched {
        Some(out_dt) if node.outputs.len() <= 1 => {
            store_gpu_output(node, out_dt, value_map);
            Ok(true)
        }
        // Multi-output op: discard the partial GPU result and fall
        // through to the CPU dispatch path so every named output gets
        // populated. Or `None`: fall through to CPU dispatch.
        Some(_) | None => Ok(false),
    }
}

/// Build the per-input `Option<&Tensor>` slice the CPU dispatcher
/// expects. `None` for empty input names or values that are
/// (unexpectedly) on device after [`ensure_host`].
fn build_host_inputs<'a>(
    node: &crate::graph::ExecutionNode,
    value_map: &'a BTreeMap<String, ValueLocation>,
) -> Vec<Option<&'a Tensor>> {
    node.inputs
        .iter()
        .map(|n| {
            if n.is_empty() {
                None
            } else {
                match value_map.get(n) {
                    Some(ValueLocation::Host(t)) => Some(t),
                    _ => None,
                }
            }
        })
        .collect()
}

/// CPU fallback: pull device-resident inputs back to host, dispatch
/// through the CPU executor, and store every non-empty output name as
/// a host tensor.
fn run_node_on_cpu(
    node: &crate::graph::ExecutionNode,
    value_map: &mut BTreeMap<String, ValueLocation>,
    runtime: &CudaRuntime,
) -> Result<(), SessionError> {
    // Bring all device-resident inputs back to host first.
    for name in &node.inputs {
        if !name.is_empty() {
            ensure_host(value_map, name)?;
        }
    }
    let inputs_host = build_host_inputs(node, value_map);

    let outputs = executor::dispatch_node_with_domain(
        &node.op_type,
        &node.domain,
        &inputs_host,
        &node.attributes,
        node.outputs.len(),
        #[cfg(feature = "gpu")]
        None,
        Some(runtime),
        #[cfg(all(feature = "metal", target_os = "macos"))]
        None,
    )
    .map_err(|e| SessionError::ExecutionFailed(format!("{}: {}", node.name, e)))?;

    for (i, output_name) in node.outputs.iter().enumerate() {
        if !output_name.is_empty() {
            if let Some(t) = outputs.get(i) {
                value_map.insert(output_name.clone(), ValueLocation::Host(t.clone()));
            }
        }
    }
    Ok(())
}

/// Run the per-op dispatch loop against a pre-seeded `value_map`.
/// Splits the residency-tracking core out of [`execute_graph_hybrid`]
/// so the CUDA Graph capture path can wrap exactly the same op
/// sequence in `cudaStreamBeginCapture` / `cudaStreamEndCapture`
/// without re-implementing dispatch.
fn run_op_loop(
    graph: &ExecutionGraph,
    value_map: &mut BTreeMap<String, ValueLocation>,
    runtime: &CudaRuntime,
) -> Result<(), SessionError> {
    for node_idx in graph.execution_order() {
        let node = &graph.nodes[node_idx.index()];
        if !try_run_node_on_gpu(node, value_map, runtime)? {
            run_node_on_cpu(node, value_map, runtime)?;
        }
    }

    let _ = ConvAttrs::default(); // suppress unused-import warning
    Ok(())
}

/// Pull each named graph output back to the host and assemble the
/// final `Vec<InferenceOutput>`. Removes entries from `value_map` so
/// the caller (capture path) can reuse the map for liveness anchoring
/// of intermediates.
fn extract_host_outputs(
    output_names: &[String],
    value_map: &mut BTreeMap<String, ValueLocation>,
) -> Result<Vec<InferenceOutput>, SessionError> {
    let mut results = Vec::new();
    for output_name in output_names {
        ensure_host(value_map, output_name)?;
        let tensor = match value_map.remove(output_name) {
            Some(ValueLocation::Host(t)) => t,
            _ => {
                return Err(SessionError::ExecutionFailed(format!(
                    "output tensor '{}' not produced",
                    output_name
                )));
            }
        };
        results.push(InferenceOutput {
            name: output_name.clone(),
            tensor,
        });
    }
    Ok(results)
}

// ───────────────────────────────────────────────────────────────────
// CUDA Graph capture / replay
// ───────────────────────────────────────────────────────────────────

/// Hybrid execution with optional CUDA Graph capture / replay.
///
/// Behaves identically to [`execute_graph_hybrid`] when `mode` is
/// `Off`, the runtime's cache is disabled, or the cache lookup
/// fails. When the cache is hit, replays the captured graph. When
/// missed and not yet disabled, attempts to capture a new graph for
/// the next inference and falls through to per-op for *this*
/// inference — capture during the warm-up has subtle correctness
/// requirements (see `attempt_capture` for details) so we keep it
/// out of the hot path.
///
/// Errors from capture / replay never propagate: on any failure we
/// log a single warning, mark the cache disabled, and fall back to
/// per-op execution. Only actual op failures (e.g. cuDNN returning
/// `CUDNN_STATUS_BAD_PARAM`) surface as `SessionError`.
#[allow(clippy::too_many_arguments)]
pub fn execute_graph_hybrid_with_capture(
    graph: &ExecutionGraph,
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    runtime: &CudaRuntime,
    device_initializer_cache: Option<&BTreeMap<String, Arc<DeviceTensor>>>,
    cache_slot: &mut Option<CudaGraphCache>,
    mode: CudaGraphMode,
    // Optional multi-stream pool. When present and the replay path
    // is taken, H2D + compute + D2H are issued on dedicated streams
    // gated by CUDA events instead of all going to the cache's
    // capture stream. No effect on the per-op fallback.
    stream_pool: Option<&StreamPool>,
) -> Result<Vec<InferenceOutput>, SessionError> {
    if !matches!(mode, CudaGraphMode::Capture) {
        return execute_graph_hybrid(
            graph,
            inputs,
            initializers,
            runtime,
            device_initializer_cache,
        );
    }

    // Lazily create the cache on first capture-mode call.
    if cache_slot.is_none() {
        match CudaGraphCache::new() {
            Ok(c) => *cache_slot = Some(c),
            Err(e) => {
                // Stream creation failed — log once, fall through.
                #[cfg(feature = "std")]
                eprintln!(
                    "cuda-graphs: stream creation failed ({}); disabling capture",
                    e
                );
                let _ = e;
                return execute_graph_hybrid(
                    graph,
                    inputs,
                    initializers,
                    runtime,
                    device_initializer_cache,
                );
            }
        }
    }

    let cache = cache_slot.as_mut().expect("just initialized");
    cache.note_inference();

    if cache.disabled {
        return execute_graph_hybrid(
            graph,
            inputs,
            initializers,
            runtime,
            device_initializer_cache,
        );
    }

    let key = GraphKey::from_inputs(inputs);

    // Cache hit → try replay.
    if cache.entries.contains_key(&key) {
        match try_replay(cache, &key, inputs, stream_pool) {
            Ok(outputs) => return Ok(outputs),
            Err(e) => {
                #[cfg(feature = "std")]
                eprintln!(
                    "cuda-graphs: replay failed ({:?}); discarding cached entry and falling back",
                    e
                );
                let _ = e;
                cache.entries.remove(&key);
                return execute_graph_hybrid(
                    graph,
                    inputs,
                    initializers,
                    runtime,
                    device_initializer_cache,
                );
            }
        }
    }

    // Cache miss → run per-op (always correct), then attempt capture
    // for the next inference. The first call pays the per-op cost;
    // subsequent calls of the same shape replay the captured graph.
    let outputs = execute_graph_hybrid(
        graph,
        inputs,
        initializers,
        runtime,
        device_initializer_cache,
    )?;

    match attempt_capture(
        graph,
        inputs,
        initializers,
        runtime,
        device_initializer_cache,
        cache,
        &key,
    ) {
        Ok(()) => {
            cache.note_rebuild();
        }
        Err(e) => {
            #[cfg(feature = "std")]
            eprintln!(
                "cuda-graphs: capture failed ({:?}); disabling for this Session",
                e
            );
            let _ = e;
            cache.disabled = true;
        }
    }

    Ok(outputs)
}

/// Replay path: copy host inputs into the cached input buffers, launch
/// the captured graph, copy outputs back to host.
///
/// `stream_pool` is `Some` when `SessionConfig::stream_config =
/// Overlap`. With `transfer_streams >= 1` the H2D, compute, and D2H
/// phases run on three different CUDA streams gated by events, so
/// the next inference's H2D can begin while this inference's D2H is
/// still in flight. With `None` (or `transfer_streams == 0`),
/// everything runs on the cache's single capture stream.
fn try_replay(
    cache: &mut CudaGraphCache,
    key: &GraphKey,
    inputs: &[(String, Tensor)],
    stream_pool: Option<&StreamPool>,
) -> Result<Vec<InferenceOutput>, SessionError> {
    let entry = cache
        .entries
        .get(key)
        .ok_or_else(|| SessionError::ExecutionFailed("cache miss in try_replay".into()))?;

    if inputs.len() != entry.input_tensors.len() {
        return Err(SessionError::ExecutionFailed(format!(
            "input count mismatch: cached {}, got {}",
            entry.input_tensors.len(),
            inputs.len()
        )));
    }

    // Choose H2D / compute / D2H streams. When the pool has transfer
    // streams, route the three phases to dedicated streams gated by
    // CUDA events; otherwise fall back to the single capture stream
    // (legacy cuda-graphs-v1 behavior).
    use crate::cuda::streams::Stream;
    let single_stream_handle: ffi::cudaStream_t = cache.capture_stream.raw();
    let (h2d_stream_handle, compute_stream_handle, d2h_stream_handle, use_overlap) =
        match stream_pool {
            Some(p) if !p.h2d.is_empty() && !p.d2h.is_empty() => {
                (p.h2d[0].raw(), p.compute.raw(), p.d2h[0].raw(), true)
            }
            _ => (
                single_stream_handle,
                single_stream_handle,
                single_stream_handle,
                false,
            ),
        };
    let _ = Stream::new; // silence unused-import on legacy build paths

    // H2D async into the persistent input tensors' buffers. The
    // captured graph hard-codes these device pointers; replay reuses
    // them by writing fresh bytes in place.
    for (i, (_, t)) in inputs.iter().enumerate() {
        let dt = &entry.input_tensors[i];
        if t.raw_data.len() > dt.buffer.size() {
            return Err(SessionError::ExecutionFailed(format!(
                "input #{} byte size {} exceeds cached buffer {}",
                i,
                t.raw_data.len(),
                dt.buffer.size()
            )));
        }
        let err = unsafe {
            ffi::cudaMemcpyAsync(
                dt.buffer.as_mut_ptr(),
                t.raw_data.as_ptr() as *const _,
                t.raw_data.len(),
                ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
                h2d_stream_handle,
            )
        };
        if err != ffi::CUDA_SUCCESS {
            return Err(SessionError::ExecutionFailed(format!(
                "cudaMemcpyAsync H2D input #{}: code {}",
                i, err
            )));
        }
    }

    // Cross-stream gating: compute waits for H2D done.
    let h2d_done = if use_overlap {
        let pool = stream_pool.expect("use_overlap implies pool");
        let evt = pool
            .acquire_event()
            .map_err(|e| SessionError::ExecutionFailed(format!("acquire H2D event: {}", e)))?;
        evt.record(&pool.h2d[0])
            .map_err(|e| SessionError::ExecutionFailed(format!("record H2D event: {}", e)))?;
        evt.wait(&pool.compute)
            .map_err(|e| SessionError::ExecutionFailed(format!("compute waits H2D: {}", e)))?;
        Some(evt)
    } else {
        None
    };

    // Single host-side launch replaces N cuDNN/cuBLAS calls.
    #[cfg(feature = "gpu-profile")]
    let _launch_start = std::time::Instant::now();
    let launch_err = unsafe { ffi::cudaGraphLaunch(entry.graph_exec.raw(), compute_stream_handle) };
    if launch_err != ffi::CUDA_SUCCESS {
        return Err(SessionError::ExecutionFailed(format!(
            "cudaGraphLaunch: code {}",
            launch_err
        )));
    }
    #[cfg(feature = "gpu-profile")]
    crate::cuda::profile::record_op("graph_launch", _launch_start);

    // Cross-stream gating: D2H waits for compute done.
    let compute_done = if use_overlap {
        let pool = stream_pool.expect("use_overlap implies pool");
        let evt = pool
            .acquire_event()
            .map_err(|e| SessionError::ExecutionFailed(format!("acquire compute event: {}", e)))?;
        evt.record(&pool.compute)
            .map_err(|e| SessionError::ExecutionFailed(format!("record compute event: {}", e)))?;
        evt.wait(&pool.d2h[0])
            .map_err(|e| SessionError::ExecutionFailed(format!("d2h waits compute: {}", e)))?;
        Some(evt)
    } else {
        None
    };

    // Issue async D2H of every output, then sync the D2H stream.
    // (The output D2H is wired through DeviceTensor::to_host, which
    // uses synchronous cudaMemcpy. For overlap mode we sync the D2H
    // stream first so to_host's sync memcpy is a no-op wait.)
    if use_overlap {
        let pool = stream_pool.expect("use_overlap implies pool");
        pool.d2h[0]
            .synchronize()
            .map_err(|e| SessionError::ExecutionFailed(format!("D2H sync: {}", e)))?;
        // Return events to the pool for reuse.
        if let Some(e) = h2d_done {
            pool.release_event(e);
        }
        if let Some(e) = compute_done {
            pool.release_event(e);
        }
    } else {
        // Legacy single-stream path: just sync the capture stream.
        cache
            .capture_stream
            .synchronize()
            .map_err(|e| SessionError::ExecutionFailed(format!("cudaStreamSynchronize: {}", e)))?;
    }
    let _ = (h2d_stream_handle, d2h_stream_handle); // silence unused-var on single-stream path

    let mut results = Vec::with_capacity(entry.output_tensors.len());
    for (name, dt) in entry.output_names.iter().zip(entry.output_tensors.iter()) {
        let tensor = dt
            .to_host()
            .map_err(|e| SessionError::ExecutionFailed(format!("output D2H: {}", e)))?;
        results.push(InferenceOutput {
            name: name.clone(),
            tensor: Tensor {
                data_type: tensor.data_type,
                shape: tensor.shape,
                name: name.clone(),
                raw_data: tensor.raw_data,
            },
        });
    }
    Ok(results)
}

/// Capture path: pre-allocate persistent input buffers, bind cuDNN /
/// cuBLAS handles to the capture stream, run the existing op-loop
/// inside `cudaStreamBeginCapture` / `cudaStreamEndCapture`,
/// instantiate the resulting graph, and store everything in the
/// cache.
///
/// Anchors every device tensor produced during capture in the
/// `GraphEntry` so intermediate buffers don't get freed (the graph
/// references their pointers).
fn attempt_capture(
    graph: &ExecutionGraph,
    inputs: &[(String, Tensor)],
    initializers: &[TensorProto],
    runtime: &CudaRuntime,
    device_initializer_cache: Option<&BTreeMap<String, Arc<DeviceTensor>>>,
    cache: &mut CudaGraphCache,
    key: &GraphKey,
) -> Result<(), SessionError> {
    // 1. Allocate persistent input DeviceTensors and pre-load with
    //    the current input bytes (so the first replay produces the
    //    same outputs the per-op warm-up just produced). The buffer
    //    pointers stored here are what the captured graph will
    //    reference; they must stay alive for the cache lifetime.
    let mut input_dts: Vec<Arc<DeviceTensor>> = Vec::with_capacity(inputs.len());
    for (_, t) in inputs {
        let buf = DeviceBuffer::alloc(t.raw_data.len())
            .map_err(|e| SessionError::ExecutionFailed(format!("input alloc: {}", e)))?;
        buf.copy_from_host(&t.raw_data)
            .map_err(|e| SessionError::ExecutionFailed(format!("input H2D: {}", e)))?;
        let dt = DeviceTensor {
            buffer: buf,
            shape: t.shape.dims.clone(),
            dtype: t.data_type,
            name: String::new(),
        };
        input_dts.push(Arc::new(dt));
    }

    // 2. Bind cuDNN / cuBLAS handles to the capture stream so their
    //    launches enqueue on it.
    bind_handles_to_stream(runtime, cache.capture_stream.raw())?;

    // RAII guard: always reset handles to the default stream when this
    // function returns, success or failure.
    struct ResetOnDrop<'a> {
        rt: &'a CudaRuntime,
    }
    impl Drop for ResetOnDrop<'_> {
        fn drop(&mut self) {
            let _ = bind_handles_to_stream(self.rt, core::ptr::null_mut());
        }
    }
    let _reset = ResetOnDrop { rt: runtime };

    // 3. Begin capture.
    let err = unsafe {
        ffi::cudaStreamBeginCapture(
            cache.capture_stream.raw(),
            ffi::cudaStreamCaptureMode::ThreadLocal,
        )
    };
    if err != ffi::CUDA_SUCCESS {
        return Err(SessionError::ExecutionFailed(format!(
            "cudaStreamBeginCapture: code {}",
            err
        )));
    }

    // 4. Seed the value map with the persistent input tensors and
    //    initializers, then run the op loop.
    let mut value_map: BTreeMap<String, ValueLocation> =
        build_initial_value_map(&[], initializers, device_initializer_cache);
    for ((name, _), dt) in inputs.iter().zip(input_dts.iter()) {
        value_map.insert(name.clone(), ValueLocation::Device(dt.clone()));
    }

    let op_loop_result = run_op_loop(graph, &mut value_map, runtime);

    // 5. End capture regardless of op-loop outcome — failing to call
    //    EndCapture leaves the stream in a stuck state.
    let mut graph_handle: ffi::cudaGraph_t = core::ptr::null_mut();
    let end_err =
        unsafe { ffi::cudaStreamEndCapture(cache.capture_stream.raw(), &mut graph_handle) };

    op_loop_result?;
    if end_err != ffi::CUDA_SUCCESS {
        return Err(SessionError::ExecutionFailed(format!(
            "cudaStreamEndCapture: code {}",
            end_err
        )));
    }
    let cuda_graph = unsafe { CudaGraph::from_raw(graph_handle) };

    // 6. Instantiate. CUDA 12+ takes (exec, graph, flags).
    #[cfg(feature = "gpu-profile")]
    let _capture_start = std::time::Instant::now();
    let exec = crate::cuda::graph::CudaGraphExec::instantiate(&cuda_graph)
        .map_err(|e| SessionError::ExecutionFailed(format!("graph instantiate: {}", e)))?;
    #[cfg(feature = "gpu-profile")]
    crate::cuda::profile::record_op("graph_capture", _capture_start);

    // 7. Collect outputs and intermediates from the value map. The
    //    graph references intermediate buffers by pointer; we anchor
    //    them in `intermediate_keep_alive` to prevent free.
    let mut output_tensors: Vec<Arc<DeviceTensor>> = Vec::new();
    for output_name in &graph.output_names {
        match value_map.remove(output_name) {
            Some(ValueLocation::Device(dt)) => output_tensors.push(dt),
            Some(ValueLocation::Host(t)) => {
                // The output landed on host (CPU fallback for the last
                // op). Capture is technically valid up to the point
                // where the host transition happened, but the output
                // pointer the graph would reference doesn't exist on
                // device. Convert host → device for replay.
                let dt = DeviceTensor::from_host(&t).map_err(|e| {
                    SessionError::ExecutionFailed(format!("host output to device: {}", e))
                })?;
                output_tensors.push(Arc::new(dt));
            }
            None => {
                return Err(SessionError::ExecutionFailed(format!(
                    "captured graph missing output '{}'",
                    output_name
                )));
            }
        }
    }
    let mut intermediate_keep_alive: Vec<Arc<DeviceTensor>> = Vec::new();
    for (_, v) in value_map.into_iter() {
        if let ValueLocation::Device(dt) = v {
            intermediate_keep_alive.push(dt);
        }
    }

    // 8. Build and store the entry. The input DeviceTensors stay live
    //    via `input_tensors`; intermediate DeviceTensors stay live via
    //    `intermediate_keep_alive`; output DeviceTensors stay live via
    //    `output_tensors`. The captured graph references all of their
    //    buffer pointers and they must all outlive the GraphExec.
    let entry = GraphEntry {
        graph: cuda_graph,
        graph_exec: exec,
        input_tensors: input_dts,
        input_shapes: key.input_shapes.clone(),
        input_dtypes: key.input_dtypes.clone(),
        output_tensors,
        output_names: graph.output_names.clone(),
        intermediate_keep_alive,
    };
    cache.entries.insert(key.clone(), entry);

    // 9. Synchronize before returning so any captured state is settled.
    cache
        .capture_stream
        .synchronize()
        .map_err(|e| SessionError::ExecutionFailed(format!("post-capture sync: {}", e)))?;

    Ok(())
}

/// Bind the runtime's cuDNN and cuBLAS handles to a particular CUDA
/// stream. Passing `null_mut()` reverts to the default stream.
fn bind_handles_to_stream(
    runtime: &CudaRuntime,
    stream: ffi::cudaStream_t,
) -> Result<(), SessionError> {
    let cudnn_err = unsafe { ffi::cudnnSetStream(runtime.cudnn.raw(), stream) };
    if cudnn_err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(SessionError::ExecutionFailed(format!(
            "cudnnSetStream: code {}",
            cudnn_err
        )));
    }
    let cublas_err = unsafe { ffi::cublasSetStream_v2(runtime.cublas.raw(), stream) };
    if cublas_err != ffi::CUBLAS_STATUS_SUCCESS {
        return Err(SessionError::ExecutionFailed(format!(
            "cublasSetStream_v2: code {}",
            cublas_err
        )));
    }
    Ok(())
}

/// Suppress unused-import warning when `feature = "gpu-profile"` is off.
#[allow(dead_code)]
fn _suppress_unused_imports() {
    let _ = TensorShape::new(Vec::new());
}
