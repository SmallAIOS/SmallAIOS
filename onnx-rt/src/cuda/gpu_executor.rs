// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU-resident graph executor.
//!
//! Parallel path to [`crate::executor::execute_graph`] that keeps all
//! intermediate tensors in device VRAM for the entire forward pass. This
//! eliminates the per-operator host<->device round-trips that the
//! mainline executor incurs when dispatching individual ops via
//! `try_cuda_dispatch`. Intended for large-model inference (e.g. Gemma
//! 4 31B) where per-op transfer overhead would otherwise dominate.
//!
//! ## Design
//!
//! * Inputs and outputs are [`DeviceTensor`]s. Graph initializers (host
//!   `TensorProto`s) are materialized to the device once at the start of
//!   [`execute_graph_gpu`].
//! * The value map is `BTreeMap<String, DeviceTensor>`. Ops read device
//!   inputs by name, produce device outputs, and insert them under the
//!   graph node's declared output names.
//! * Supported ops (Section 5 scope): `MatMul`, `Gemm`, `MatMulInteger`,
//!   `Conv`. Any other op type is a hard error — no CPU fallback.
//! * The per-op primitives (`gpu_gemm_device`, `gpu_gemm_int8_device`,
//!   `gpu_conv2d_device`) take `DeviceTensor` inputs directly and skip
//!   the host boundary. They duplicate the cuBLAS/cuDNN setup from
//!   [`crate::cuda::dispatch`] and [`crate::cuda::conv`] with the minimum
//!   necessary code — see TODOs for the consolidation plan.

extern crate alloc;

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::ffi;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::byte_io::{self, allocate_tensor_data};
use crate::graph::ExecutionGraph;
use crate::onnx_types::{AttributeProto, AttributeType, TensorProto};
use crate::operators::OpError;
use crate::session::SessionError;
use crate::tensor::{DataType, Tensor, TensorShape};

// ── DeviceTensor ────────────────────────────────────────────────────

/// A tensor resident entirely in GPU device memory.
///
/// Mirrors the host [`Tensor`] shape/dtype/name metadata but stores the
/// element data in a [`DeviceBuffer`] instead of a `Vec<u8>`. Intended
/// for use inside [`execute_graph_gpu`] so that the forward pass never
/// touches host memory for intermediate activations.
pub struct DeviceTensor {
    /// Device memory holding the raw element bytes.
    pub buffer: DeviceBuffer,
    /// Tensor shape (ONNX-style dimensions).
    pub shape: Vec<i64>,
    /// Element data type.
    pub dtype: DataType,
    /// Tensor name (matches the graph-output name when produced by an op).
    pub name: String,
}

impl core::fmt::Debug for DeviceTensor {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("DeviceTensor")
            .field("name", &self.name)
            .field("dtype", &self.dtype)
            .field("shape", &self.shape)
            .field("byte_size", &self.byte_size())
            .finish()
    }
}

impl DeviceTensor {
    /// Total number of elements (product of shape dims).
    pub fn total_elements(&self) -> usize {
        let mut n: usize = 1;
        for d in &self.shape {
            if *d < 0 {
                return 0;
            }
            n = n.saturating_mul(*d as usize);
        }
        n
    }

    /// Byte size = element count * element_size(dtype).
    pub fn byte_size(&self) -> usize {
        self.total_elements()
            .saturating_mul(self.dtype.element_size())
    }

    /// Allocate a fresh device tensor of `shape` and `dtype`. The buffer
    /// contents are uninitialized — the caller is responsible for filling
    /// them (via a kernel, a copy, or a descriptor-sized GEMM output).
    pub fn alloc(shape: Vec<i64>, dtype: DataType) -> Result<Self, CudaError> {
        let mut count: usize = 1;
        for d in &shape {
            if *d < 0 {
                count = 0;
                break;
            }
            count = count.saturating_mul(*d as usize);
        }
        let bytes = count.saturating_mul(dtype.element_size());
        let buffer = DeviceBuffer::alloc(bytes)?;
        Ok(Self {
            buffer,
            shape,
            dtype,
            name: String::new(),
        })
    }

    /// Copy a host [`Tensor`] to the device.
    pub fn from_host(t: &Tensor) -> Result<Self, CudaError> {
        let buffer = DeviceBuffer::alloc(t.raw_data.len())?;
        buffer.copy_from_host(&t.raw_data)?;
        Ok(Self {
            buffer,
            shape: t.shape.dims.clone(),
            dtype: t.data_type,
            name: t.name.clone(),
        })
    }

    /// Copy this device tensor back to the host.
    pub fn to_host(&self) -> Result<Tensor, CudaError> {
        let bytes = self.byte_size();
        let mut raw_data = vec![0u8; bytes];
        self.buffer.copy_to_host(&mut raw_data)?;
        Ok(Tensor {
            data_type: self.dtype,
            shape: TensorShape::new(self.shape.clone()),
            name: self.name.clone(),
            raw_data,
        })
    }
}

/// Free function that copies a host [`Tensor`] to a new [`DeviceTensor`].
///
/// Kept out of an `impl Tensor` block so `tensor.rs` stays cfg-agnostic
/// (no `cuda` feature gates in the host tensor module).
pub fn tensor_to_device(t: &Tensor, _runtime: &CudaRuntime) -> Result<DeviceTensor, CudaError> {
    DeviceTensor::from_host(t)
}

// ── Initializer materialization helper ──────────────────────────────

/// Mirror of `executor::tensor_from_proto`, duplicated here so this
/// module does not have to depend on a crate-private helper.
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

// ── initializers_to_gpu ─────────────────────────────────────────────

/// Materialize a slice of host `TensorProto` initializers as
/// [`DeviceTensor`]s in one pass.
///
/// Used by [`crate::session::Session::from_safetensors`] (Section 9) to
/// preload a Gemma graph's weights onto the GPU once at session
/// construction time, rather than re-uploading them on every
/// [`execute_graph_gpu`] call.
pub fn initializers_to_gpu(
    initializers: &[TensorProto],
    _runtime: &CudaRuntime,
) -> Result<BTreeMap<String, DeviceTensor>, CudaError> {
    let mut out: BTreeMap<String, DeviceTensor> = BTreeMap::new();
    for init in initializers {
        let host_t = match tensor_from_proto(init) {
            Some(t) => t,
            None => continue,
        };
        let dev_t = DeviceTensor::from_host(&host_t)?;
        out.insert(init.name.clone(), dev_t);
    }
    Ok(out)
}

// ── execute_graph_gpu ───────────────────────────────────────────────

/// Run an [`ExecutionGraph`] end-to-end on the GPU.
///
/// All intermediate tensors stay in VRAM until the caller pulls the
/// graph outputs back to the host via [`DeviceTensor::to_host`].
///
/// `input_device_tensors` supplies each graph input by name. `initializers`
/// are host-side `TensorProto`s that get copied to the device at the start
/// of the call (the Section 6 graph builder will later be able to load
/// weights directly from a safetensors mmap, skipping this host bridge).
///
/// Any operator without a GPU implementation produces
/// [`SessionError::ExecutionFailed`] with an
/// [`OpError::InternalError`] message — there is no silent CPU fallback.
pub fn execute_graph_gpu(
    graph: &ExecutionGraph,
    input_device_tensors: &[(String, DeviceTensor)],
    initializers: &[TensorProto],
    runtime: &CudaRuntime,
) -> Result<Vec<DeviceTensor>, SessionError> {
    execute_graph_gpu_with_weights_and_cache(
        graph,
        input_device_tensors,
        initializers,
        None,
        None,
        runtime,
    )
}

/// Extended entry point accepting an optional pre-loaded GPU weight map.
///
/// When `pre_loaded_weights` is `Some`, any tensor whose name matches a
/// key in the map is resolved from that map (via a D2D clone) instead of
/// being uploaded from the host `TensorProto`. This is the path used by
/// safetensors-backed [`crate::session::Session`]s.
pub fn execute_graph_gpu_with_weights(
    graph: &ExecutionGraph,
    input_device_tensors: &[(String, DeviceTensor)],
    initializers: &[TensorProto],
    pre_loaded_weights: Option<&BTreeMap<String, DeviceTensor>>,
    runtime: &CudaRuntime,
) -> Result<Vec<DeviceTensor>, SessionError> {
    execute_graph_gpu_with_weights_and_cache(
        graph,
        input_device_tensors,
        initializers,
        pre_loaded_weights,
        None,
        runtime,
    )
}

/// State threaded through the executor for KV-cache-aware dispatch.
///
/// `next_layer_idx` is incremented each time a `GroupQueryAttention`
/// node is dispatched, mapping graph order to per-layer cache slots
/// (every transformer layer emits exactly one `GroupQueryAttention`
/// node — see `model_loader::gemma::build_layer`).
pub struct KvCacheDispatchCtx<'a> {
    pub cache: &'a mut crate::cuda::GpuKvCache,
    pub next_layer_idx: usize,
}

/// Final entry point combining pre-loaded weights and a KV cache.
///
/// `kv_cache` is `Some` for autoregressive safetensors sessions; the
/// dispatcher routes each `GroupQueryAttention` node to its
/// corresponding layer slot via a per-call layer counter and calls
/// `gpu_gqa_with_cache`. After the forward pass the executor calls
/// `cache.advance_position()` once per token (handled by the caller —
/// the executor does not advance the position itself, only appends).
pub fn execute_graph_gpu_with_weights_and_cache(
    graph: &ExecutionGraph,
    input_device_tensors: &[(String, DeviceTensor)],
    initializers: &[TensorProto],
    pre_loaded_weights: Option<&BTreeMap<String, DeviceTensor>>,
    kv_cache: Option<&mut crate::cuda::GpuKvCache>,
    runtime: &CudaRuntime,
) -> Result<Vec<DeviceTensor>, SessionError> {
    let mut value_map: BTreeMap<String, DeviceTensor> = BTreeMap::new();

    seed_inputs_into_value_map(&mut value_map, input_device_tensors)?;
    seed_initializers_into_value_map(&mut value_map, initializers, pre_loaded_weights)?;

    // Wrap the cache (if any) in the dispatch context that carries the
    // running layer index across nodes.
    let mut cache_ctx = kv_cache.map(|c| KvCacheDispatchCtx {
        cache: c,
        next_layer_idx: 0,
    });

    // Walk the graph in topological order.
    for node_idx in graph.execution_order() {
        let node = &graph.nodes[node_idx.index()];
        run_one_node(runtime, node, &mut value_map, cache_ctx.as_mut())?;
    }

    // §6.14 / §7.4: commit the per-layer K/V appends from this forward
    // pass with a single advance_position call. Skipped if no GQA op
    // ran (e.g. graph contains no attention nodes).
    finalize_kv_cache(cache_ctx.as_mut())?;

    collect_graph_outputs(&mut value_map, &graph.output_names)
}

/// Copy each user-supplied input DeviceTensor into the value map via D2D.
fn seed_inputs_into_value_map(
    value_map: &mut BTreeMap<String, DeviceTensor>,
    input_device_tensors: &[(String, DeviceTensor)],
) -> Result<(), SessionError> {
    for (name, dev_t) in input_device_tensors {
        let cloned = clone_device_tensor(dev_t)
            .map_err(|e| SessionError::ExecutionFailed(format!("input '{}': {}", name, e)))?;
        value_map.insert(name.clone(), cloned);
    }
    Ok(())
}

/// Materialize graph initializers into the value map.
///
/// Prefers the pre-loaded GPU weight map (via D2D clone) when available;
/// otherwise bridges each `TensorProto` host -> device.
fn seed_initializers_into_value_map(
    value_map: &mut BTreeMap<String, DeviceTensor>,
    initializers: &[TensorProto],
    pre_loaded_weights: Option<&BTreeMap<String, DeviceTensor>>,
) -> Result<(), SessionError> {
    for init in initializers {
        if let Some(dev_t) = try_clone_preloaded(init, pre_loaded_weights)? {
            value_map.insert(init.name.clone(), dev_t);
            continue;
        }
        if let Some(host_t) = tensor_from_proto(init) {
            let dev_t = DeviceTensor::from_host(&host_t).map_err(|e| {
                SessionError::ExecutionFailed(format!("initializer '{}': {}", init.name, e))
            })?;
            value_map.insert(init.name.clone(), dev_t);
        }
    }
    Ok(())
}

/// Return a cloned device tensor from the pre-loaded weight map if the
/// initializer name has a match. Returns `Ok(None)` when there is no map
/// or when the name is missing — the caller falls back to host bridging.
fn try_clone_preloaded(
    init: &TensorProto,
    pre_loaded_weights: Option<&BTreeMap<String, DeviceTensor>>,
) -> Result<Option<DeviceTensor>, SessionError> {
    let Some(map) = pre_loaded_weights else {
        return Ok(None);
    };
    let Some(dev_src) = map.get(&init.name) else {
        return Ok(None);
    };
    let cloned = clone_device_tensor(dev_src).map_err(|e| {
        SessionError::ExecutionFailed(format!("pre-loaded weight '{}': {}", init.name, e))
    })?;
    Ok(Some(cloned))
}

/// Dispatch a single graph node and store its outputs in the value map.
fn run_one_node(
    runtime: &CudaRuntime,
    node: &crate::graph::ExecutionNode,
    value_map: &mut BTreeMap<String, DeviceTensor>,
    cache_ctx: Option<&mut KvCacheDispatchCtx<'_>>,
) -> Result<(), SessionError> {
    let input_refs: Vec<Option<&DeviceTensor>> = node
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

    let outputs = dispatch_gpu_node(
        runtime,
        &node.op_type,
        &input_refs,
        &node.attributes,
        cache_ctx,
    )
    .map_err(|e| SessionError::ExecutionFailed(format!("{}: {}", node.name, e)))?;

    // Store outputs under their declared names. All Section 5 ops
    // are single-output; pair by index and let any trailing declared
    // outputs be silently skipped (they stay absent from the map).
    for (mut dev_t, output_name) in outputs.into_iter().zip(node.outputs.iter()) {
        if output_name.is_empty() {
            continue;
        }
        dev_t.name = output_name.clone();
        value_map.insert(output_name.clone(), dev_t);
    }
    Ok(())
}

/// Commit per-layer K/V appends accumulated during the forward pass.
fn finalize_kv_cache(cache_ctx: Option<&mut KvCacheDispatchCtx<'_>>) -> Result<(), SessionError> {
    let Some(ctx) = cache_ctx else {
        return Ok(());
    };
    if ctx.next_layer_idx == 0 {
        return Ok(());
    }
    ctx.cache
        .advance_position()
        .map_err(|e| SessionError::ExecutionFailed(format!("kv_cache advance_position: {}", e)))
}

/// Extract the declared graph outputs from the value map in declared order.
fn collect_graph_outputs(
    value_map: &mut BTreeMap<String, DeviceTensor>,
    output_names: &[String],
) -> Result<Vec<DeviceTensor>, SessionError> {
    let mut results = Vec::new();
    for output_name in output_names {
        let t = value_map.remove(output_name).ok_or_else(|| {
            SessionError::ExecutionFailed(format!(
                "GPU executor: output tensor '{}' not produced",
                output_name
            ))
        })?;
        results.push(t);
    }
    Ok(results)
}

/// Clone a device tensor by allocating a new buffer and doing a D2D copy.
fn clone_device_tensor(src: &DeviceTensor) -> Result<DeviceTensor, CudaError> {
    let bytes = src.byte_size();
    let buffer = DeviceBuffer::alloc(bytes)?;
    if bytes > 0 {
        let err = unsafe {
            ffi::cudaMemcpy(
                buffer.as_mut_ptr(),
                src.buffer.as_ptr(),
                bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            )
        };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::CopyFailed {
                msg: "device-to-device",
                code: err,
            });
        }
    }
    Ok(DeviceTensor {
        buffer,
        shape: src.shape.clone(),
        dtype: src.dtype,
        name: src.name.clone(),
    })
}

// ── Dispatch ────────────────────────────────────────────────────────

/// Dispatch a single graph node to its GPU-only implementation.
///
/// `kv_cache_ctx` is `Some` when the caller wants `GroupQueryAttention`
/// nodes to read/write a [`crate::cuda::GpuKvCache`]. Each successful
/// GQA dispatch increments `next_layer_idx`, mapping graph order to
/// per-layer cache slots.
///
/// Returns `OpError::InternalError` for any unsupported op type.
fn dispatch_gpu_node(
    runtime: &CudaRuntime,
    op_type: &str,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
    kv_cache_ctx: Option<&mut KvCacheDispatchCtx<'_>>,
) -> Result<Vec<DeviceTensor>, OpError> {
    match op_type {
        "MatMul" => dispatch_matmul(runtime, inputs),
        "Gemm" => dispatch_gemm(runtime, inputs, attrs),
        "MatMulInteger" => dispatch_matmul_integer(runtime, inputs),
        "Conv" => dispatch_conv(runtime, inputs, attrs),
        "Add" => dispatch_elementwise_binary(runtime, inputs, "Add"),
        "Mul" => dispatch_elementwise_binary(runtime, inputs, "Mul"),
        "Silu" => dispatch_silu(runtime, inputs),
        "Gather" => dispatch_gather(runtime, inputs, attrs),
        "RMSNormalization" => dispatch_rms_norm(runtime, inputs, attrs),
        "RotaryEmbedding" => dispatch_rotary_embedding(runtime, inputs, attrs, kv_cache_ctx),
        "GroupQueryAttention" => dispatch_gqa(runtime, inputs, attrs, kv_cache_ctx),
        other => Err(OpError::InternalError(format!(
            "GPU executor: no GPU implementation for {}",
            other
        ))),
    }
}

fn dispatch_matmul(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
) -> Result<Vec<DeviceTensor>, OpError> {
    let a = take_input(inputs, 0, "MatMul", "A")?;
    let b = take_input(inputs, 1, "MatMul", "B")?;
    let out = gpu_gemm_device(runtime, a, b, false, false, 1.0, 0.0, None)
        .map_err(|e| OpError::InternalError(format!("CUDA MatMul: {}", e)))?;
    Ok(vec![out])
}

/// Parsed Gemm attributes (transA/transB/alpha/beta).
struct GemmAttrs {
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
}

fn parse_gemm_attrs(attrs: &[AttributeProto], has_bias: bool) -> GemmAttrs {
    let mut out = GemmAttrs {
        trans_a: false,
        trans_b: false,
        alpha: 1.0,
        beta: 1.0,
    };
    for attr in attrs {
        match attr.name.as_str() {
            "transA" => out.trans_a = attr.i != 0,
            "transB" => out.trans_b = attr.i != 0,
            "alpha" => out.alpha = attr.f,
            "beta" => out.beta = if has_bias { attr.f } else { 0.0 },
            _ => {}
        }
    }
    if !has_bias {
        out.beta = 0.0;
    }
    out
}

fn dispatch_gemm(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<DeviceTensor>, OpError> {
    let a = take_input(inputs, 0, "Gemm", "A")?;
    let b = take_input(inputs, 1, "Gemm", "B")?;
    let c_bias = inputs.get(2).and_then(|o| *o);
    let g = parse_gemm_attrs(attrs, c_bias.is_some());
    let out = gpu_gemm_device(runtime, a, b, g.trans_a, g.trans_b, g.alpha, g.beta, c_bias)
        .map_err(|e| OpError::InternalError(format!("CUDA Gemm: {}", e)))?;
    Ok(vec![out])
}

fn dispatch_matmul_integer(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
) -> Result<Vec<DeviceTensor>, OpError> {
    let a = take_input(inputs, 0, "MatMulInteger", "A")?;
    let b = take_input(inputs, 1, "MatMulInteger", "B")?;
    if a.dtype != DataType::Int8 || b.dtype != DataType::Int8 {
        return Err(OpError::InternalError(format!(
            "GPU executor: MatMulInteger requires Int8 inputs (got {:?}/{:?})",
            a.dtype, b.dtype
        )));
    }
    let out = gpu_gemm_int8_device(runtime, a, b)
        .map_err(|e| OpError::InternalError(format!("CUDA MatMulInteger: {}", e)))?;
    Ok(vec![out])
}

/// Parsed Conv attributes (pads / strides / dilations / group).
struct ConvAttrs {
    pads: Vec<i32>,
    strides: Vec<i32>,
    dilations: Vec<i32>,
    group: i32,
}

fn parse_conv_attrs(attrs: &[AttributeProto]) -> ConvAttrs {
    let mut out = ConvAttrs {
        pads: vec![0, 0, 0, 0],
        strides: vec![1, 1],
        dilations: vec![1, 1],
        group: 1,
    };
    for attr in attrs {
        match (attr.name.as_str(), attr.attr_type) {
            ("pads", AttributeType::Ints) => {
                out.pads = attr.ints.iter().map(|&v| v as i32).collect()
            }
            ("strides", AttributeType::Ints) => {
                out.strides = attr.ints.iter().map(|&v| v as i32).collect()
            }
            ("dilations", AttributeType::Ints) => {
                out.dilations = attr.ints.iter().map(|&v| v as i32).collect()
            }
            ("group", AttributeType::Int) => out.group = attr.i as i32,
            _ => {}
        }
    }
    out
}

fn dispatch_conv(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<DeviceTensor>, OpError> {
    let x = take_input(inputs, 0, "Conv", "X")?;
    let w = take_input(inputs, 1, "Conv", "W")?;
    let bias = inputs.get(2).and_then(|o| *o);
    let c = parse_conv_attrs(attrs);
    let out = gpu_conv2d_device(
        runtime,
        x,
        w,
        bias,
        &c.pads,
        &c.strides,
        &c.dilations,
        c.group,
    )
    .map_err(|e| OpError::InternalError(format!("CUDA Conv: {}", e)))?;
    Ok(vec![out])
}

fn dispatch_elementwise_binary(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    op: &str,
) -> Result<Vec<DeviceTensor>, OpError> {
    let a = take_input(inputs, 0, op, "A")?;
    let b = take_input(inputs, 1, op, "B")?;
    let out = match op {
        "Add" => crate::cuda::kernels::elementwise::add_gpu(runtime, a, b),
        "Mul" => crate::cuda::kernels::elementwise::mul_gpu(runtime, a, b),
        _ => {
            return Err(OpError::InternalError(format!(
                "GPU executor: dispatch_elementwise_binary called with unsupported op {}",
                op
            )))
        }
    }
    .map_err(|e| OpError::InternalError(format!("CUDA {}: {}", op, e)))?;
    Ok(vec![out])
}

fn dispatch_silu(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
) -> Result<Vec<DeviceTensor>, OpError> {
    let x = take_input(inputs, 0, "Silu", "X")?;
    let out = crate::cuda::kernels::elementwise::silu_gpu(runtime, x)
        .map_err(|e| OpError::InternalError(format!("CUDA Silu: {}", e)))?;
    Ok(vec![out])
}

fn dispatch_gather(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<DeviceTensor>, OpError> {
    let data = take_input(inputs, 0, "Gather", "data")?;
    let indices = take_input(inputs, 1, "Gather", "indices")?;
    let axis = find_int_attr(attrs, "axis").unwrap_or(0);
    let out = crate::cuda::kernels::gather::gather_gpu(runtime, data, indices, axis)
        .map_err(|e| OpError::InternalError(format!("CUDA Gather: {}", e)))?;
    Ok(vec![out])
}

fn dispatch_rms_norm(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<DeviceTensor>, OpError> {
    let x = take_input(inputs, 0, "RMSNormalization", "X")?;
    let weight = take_input(inputs, 1, "RMSNormalization", "weight")?;
    let eps = find_float_attr(attrs, "epsilon").unwrap_or(1e-6);
    let out = crate::cuda::kernels::rms_norm::rms_norm_gpu(runtime, x, weight, eps)
        .map_err(|e| OpError::InternalError(format!("CUDA RMSNormalization: {}", e)))?;
    Ok(vec![out])
}

/// Find the first int attribute with the given name.
fn find_int_attr(attrs: &[AttributeProto], name: &str) -> Option<i64> {
    attrs.iter().find(|a| a.name == name).map(|a| a.i)
}

/// Find the first float attribute with the given name.
fn find_float_attr(attrs: &[AttributeProto], name: &str) -> Option<f32> {
    attrs.iter().find(|a| a.name == name).map(|a| a.f)
}

/// Find the first bool-style int attribute (`!= 0`) with the given name.
fn find_bool_attr(attrs: &[AttributeProto], name: &str) -> Option<bool> {
    find_int_attr(attrs, name).map(|v| v != 0)
}

/// RotaryEmbedding inputs after disambiguating the two accepted forms:
///   - microsoft: [X, position_ids, cos_cache, sin_cache]
///   - gemma-style: [X, cos_cache, sin_cache]
struct RotaryInputs<'a> {
    cos_cache: &'a DeviceTensor,
    sin_cache: &'a DeviceTensor,
    explicit_positions: Option<&'a DeviceTensor>,
}

fn resolve_rotary_inputs<'a>(
    inputs: &'a [Option<&'a DeviceTensor>],
) -> Result<RotaryInputs<'a>, OpError> {
    if inputs.len() >= 4 {
        let pos = take_input(inputs, 1, "RotaryEmbedding", "position_ids")?;
        let cos = take_input(inputs, 2, "RotaryEmbedding", "cos_cache")?;
        let sin = take_input(inputs, 3, "RotaryEmbedding", "sin_cache")?;
        Ok(RotaryInputs {
            cos_cache: cos,
            sin_cache: sin,
            explicit_positions: Some(pos),
        })
    } else {
        let cos = take_input(inputs, 1, "RotaryEmbedding", "cos_cache")?;
        let sin = take_input(inputs, 2, "RotaryEmbedding", "sin_cache")?;
        Ok(RotaryInputs {
            cos_cache: cos,
            sin_cache: sin,
            explicit_positions: None,
        })
    }
}

/// Read the first i64 from a host-resident position_ids tensor.
fn read_first_i64_position(pos: &DeviceTensor) -> Result<i32, OpError> {
    let pos_host = pos
        .to_host()
        .map_err(|e| OpError::InternalError(format!("RotaryEmbedding pos D2H: {}", e)))?;
    if pos_host.raw_data.len() < 8 {
        return Ok(0);
    }
    let bytes: [u8; 8] = pos_host.raw_data[..8].try_into().unwrap_or([0; 8]);
    Ok(i64::from_le_bytes(bytes) as i32)
}

/// Pick the rotary starting position from explicit IDs or the KV cache.
fn pick_rotary_position(
    explicit_positions: Option<&DeviceTensor>,
    kv_cache_ctx: Option<&KvCacheDispatchCtx<'_>>,
) -> Result<i32, OpError> {
    if let Some(pos) = explicit_positions {
        return read_first_i64_position(pos);
    }
    Ok(kv_cache_ctx
        .map(|c| c.cache.current_position() as i32)
        .unwrap_or(0))
}

/// Cast a cos/sin table to F32 if it's BF16. Returns either a borrow of
/// the existing tensor or a newly-allocated F32 DeviceTensor that the
/// caller must keep alive.
fn cast_rotary_table_to_f32<'a>(
    runtime: &CudaRuntime,
    table: &'a DeviceTensor,
    owned_slot: &'a mut Option<DeviceTensor>,
    which: &str,
) -> Result<&'a DeviceTensor, OpError> {
    if table.dtype != DataType::BFloat16 {
        return Ok(table);
    }
    let casted =
        crate::cuda::kernels::attention::cast_bf16_to_f32_gpu(runtime, table).map_err(|e| {
            OpError::InternalError(format!("RotaryEmbedding {} cast bf16->f32: {}", which, e))
        })?;
    *owned_slot = Some(casted);
    Ok(owned_slot.as_ref().expect("just stored Some"))
}

/// Compute num_heads for rank-3 rotary inputs.
fn rotary_num_heads_for_rank3(
    x: &DeviceTensor,
    cos_ref: &DeviceTensor,
    num_heads_attr: Option<i64>,
) -> Option<i64> {
    if x.shape.len() != 3 {
        return None;
    }
    if let Some(n) = num_heads_attr {
        return Some(n);
    }
    let head_dim_from_cos = cos_ref.shape.last().copied().unwrap_or(0) * 2;
    let hidden = x.shape[2];
    if head_dim_from_cos > 0 && hidden % head_dim_from_cos == 0 {
        Some(hidden / head_dim_from_cos)
    } else {
        None
    }
}

fn dispatch_rotary_embedding(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
    kv_cache_ctx: Option<&mut KvCacheDispatchCtx<'_>>,
) -> Result<Vec<DeviceTensor>, OpError> {
    let x = take_input(inputs, 0, "RotaryEmbedding", "X")?;
    let RotaryInputs {
        cos_cache,
        sin_cache,
        explicit_positions,
    } = resolve_rotary_inputs(inputs)?;
    let interleaved = find_bool_attr(attrs, "interleaved").unwrap_or(false);
    // Borrow the ctx immutably here — pick_rotary_position only reads.
    let position = pick_rotary_position(explicit_positions, kv_cache_ctx.as_deref())?;

    // The rotary kernels read F32 cos/sin tables; the gemma builder
    // emits BF16 tables. Cast on the fly when needed.
    let mut cos_f32_owned: Option<DeviceTensor> = None;
    let mut sin_f32_owned: Option<DeviceTensor> = None;
    let cos_ref = cast_rotary_table_to_f32(runtime, cos_cache, &mut cos_f32_owned, "cos")?;
    let sin_ref = cast_rotary_table_to_f32(runtime, sin_cache, &mut sin_f32_owned, "sin")?;

    let num_heads_attr = find_int_attr(attrs, "num_heads");
    let num_heads_for_rank3 = rotary_num_heads_for_rank3(x, cos_ref, num_heads_attr);

    let out = crate::cuda::kernels::rotary::rotary_gpu(
        runtime,
        x,
        cos_ref,
        sin_ref,
        position,
        interleaved,
        num_heads_for_rank3,
    )
    .map_err(|e| OpError::InternalError(format!("CUDA RotaryEmbedding: {}", e)))?;
    Ok(vec![out])
}

/// Parsed GQA attributes (num_heads / kv_num_heads / local_window_size).
struct GqaAttrs {
    num_heads: i64,
    kv_num_heads: i64,
    /// Sliding-window size, already converted to the kernel's
    /// `Option<i32>` (None when the attribute is absent or non-positive).
    window: Option<i32>,
}

fn parse_gqa_attrs(attrs: &[AttributeProto]) -> GqaAttrs {
    let mut num_heads: i64 = 0;
    let mut kv_num_heads: i64 = 0;
    let mut local_window: i64 = -1;
    for a in attrs {
        match a.name.as_str() {
            "local_window_size" => local_window = a.i,
            "num_heads" => num_heads = a.i,
            "kv_num_heads" => kv_num_heads = a.i,
            _ => {}
        }
    }
    let window = if local_window > 0 {
        Some(local_window as i32)
    } else {
        None
    };
    GqaAttrs {
        num_heads,
        kv_num_heads,
        window,
    }
}

fn gqa_rank3(
    runtime: &CudaRuntime,
    q: &DeviceTensor,
    new_k: &DeviceTensor,
    new_v: &DeviceTensor,
    a: &GqaAttrs,
    kv_cache_ctx: Option<&mut KvCacheDispatchCtx<'_>>,
) -> Result<DeviceTensor, OpError> {
    if a.num_heads <= 0 || a.kv_num_heads <= 0 {
        return Err(OpError::InternalError(
            "GroupQueryAttention rank-3: need num_heads / kv_num_heads attrs".into(),
        ));
    }
    // Still advance layer counter so subsequent GQA nodes
    // index the next layer consistently.
    if let Some(ctx) = kv_cache_ctx {
        ctx.next_layer_idx += 1;
    }
    crate::cuda::kernels::attention::gpu_gqa_rank3(
        runtime,
        q,
        new_k,
        new_v,
        a.num_heads,
        a.kv_num_heads,
        a.window,
        None,
    )
    .map_err(|e| OpError::InternalError(format!("CUDA GroupQueryAttention (rank-3): {}", e)))
}

fn gqa_with_cache(
    runtime: &CudaRuntime,
    q: &DeviceTensor,
    new_k: &DeviceTensor,
    new_v: &DeviceTensor,
    window: Option<i32>,
    ctx: &mut KvCacheDispatchCtx<'_>,
) -> Result<DeviceTensor, OpError> {
    let layer_idx = ctx.next_layer_idx;
    ctx.next_layer_idx += 1;
    crate::cuda::kernels::attention::gpu_gqa_with_cache(
        runtime, q, new_k, new_v, ctx.cache, layer_idx, window, None,
    )
    .map_err(|e| OpError::InternalError(format!("CUDA GroupQueryAttention (cache): {}", e)))
}

fn gqa_cacheless(
    runtime: &CudaRuntime,
    q: &DeviceTensor,
    new_k: &DeviceTensor,
    new_v: &DeviceTensor,
    window: Option<i32>,
) -> Result<DeviceTensor, OpError> {
    crate::cuda::kernels::attention::gpu_gqa(runtime, q, new_k, new_v, window, None)
        .map_err(|e| OpError::InternalError(format!("CUDA GroupQueryAttention: {}", e)))
}

fn dispatch_gqa(
    runtime: &CudaRuntime,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
    kv_cache_ctx: Option<&mut KvCacheDispatchCtx<'_>>,
) -> Result<Vec<DeviceTensor>, OpError> {
    let q = take_input(inputs, 0, "GroupQueryAttention", "Q")?;
    let new_k = take_input(inputs, 1, "GroupQueryAttention", "K")?;
    let new_v = take_input(inputs, 2, "GroupQueryAttention", "V")?;
    let a = parse_gqa_attrs(attrs);

    // Rank-3 inputs come straight from the gemma Gemm projections —
    // dispatch through `gpu_gqa_rank3`. For cache semantics we currently
    // fall back to the no-cache path when input is rank-3.
    let is_rank3 = q.shape.len() == 3 && new_k.shape.len() == 3 && new_v.shape.len() == 3;
    let out = if is_rank3 {
        gqa_rank3(runtime, q, new_k, new_v, &a, kv_cache_ctx)?
    } else if let Some(ctx) = kv_cache_ctx {
        gqa_with_cache(runtime, q, new_k, new_v, a.window, ctx)?
    } else {
        gqa_cacheless(runtime, q, new_k, new_v, a.window)?
    };
    // The microsoft GQA op declares 3 outputs (attn_out, present_k,
    // present_v). v1 dispatcher only emits attn_out — present K/V live
    // inside the persistent GpuKvCache, not in the value map.
    Ok(vec![out])
}

fn take_input<'a>(
    inputs: &'a [Option<&'a DeviceTensor>],
    idx: usize,
    op: &str,
    role: &str,
) -> Result<&'a DeviceTensor, OpError> {
    inputs
        .get(idx)
        .and_then(|o| *o)
        .ok_or_else(|| OpError::InternalError(format!("GPU {}: missing input '{}'", op, role)))
}

// ── GPU-resident primitives ─────────────────────────────────────────
//
// TODO(consolidation): these helpers duplicate the cuBLAS / cuDNN
// descriptor setup from `cuda::dispatch` and `cuda::conv` so that the
// host-Tensor entry points (`gpu_gemm`, `gpu_gemm_int8`, `gpu_conv2d`)
// remain untouched and the existing 26 CUDA tests stay green. A later
// refactor should make the device-tensor versions the primitive and
// rewrite the host-Tensor versions as thin copy-in / copy-out wrappers.

/// Device-tensor GEMM. Mirrors [`crate::cuda::dispatch::gpu_gemm`] but
/// reads/writes `DeviceBuffer`s directly — no host transfers.
/// (m, k) for an A-matrix of dims `[..., M, K]` (or `[..., K, M]` when `trans_a`).
fn gemm_a_extents(a_dims: &[i64], trans_a: bool) -> (usize, usize) {
    let last = a_dims.len() - 1;
    if trans_a {
        (a_dims[last] as usize, a_dims[last - 1] as usize)
    } else {
        (a_dims[last - 1] as usize, a_dims[last] as usize)
    }
}

/// (k, n) for a B-matrix of dims `[..., K, N]` (or `[..., N, K]` when `trans_b`).
fn gemm_b_extents(b_dims: &[i64], trans_b: bool) -> (usize, usize) {
    let last = b_dims.len() - 1;
    if trans_b {
        (b_dims[last] as usize, b_dims[last - 1] as usize)
    } else {
        (b_dims[last - 1] as usize, b_dims[last] as usize)
    }
}

/// I/O dtype family selected from A and B's element types.
#[derive(Clone, Copy)]
struct GemmDtype {
    elem_size: usize,
    io_dtype: ffi::cudaDataType_t,
    out_data_type: DataType,
    is_bf16: bool,
}

fn pick_gemm_dtype(a: &DeviceTensor, b: &DeviceTensor) -> Result<GemmDtype, CudaError> {
    let is_bf16 = a.dtype == DataType::BFloat16 && b.dtype == DataType::BFloat16;
    let is_f32 = a.dtype == DataType::Float && b.dtype == DataType::Float;
    if !is_bf16 && !is_f32 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_device: unsupported dtype",
            code: -3,
        });
    }
    Ok(GemmDtype {
        elem_size: if is_bf16 { 2 } else { 4 },
        io_dtype: if is_bf16 {
            ffi::cudaDataType_t::CUDA_R_16BF
        } else {
            ffi::cudaDataType_t::CUDA_R_32F
        },
        out_data_type: if is_bf16 {
            DataType::BFloat16
        } else {
            DataType::Float
        },
        is_bf16,
    })
}

/// Initialize the C buffer with a bias tensor (matching dtype). Returns
/// `Ok(true)` if the bias was applied directly, `Ok(false)` if the bias
/// shape was unknown (caller zero-fills and lets beta multiply zeros).
fn copy_gemm_bias_into_c(
    c_buf: &DeviceBuffer,
    bias: &DeviceTensor,
    dt: &GemmDtype,
    m: usize,
    n: usize,
    c_bytes: usize,
) -> Result<bool, CudaError> {
    if bias.dtype != dt.out_data_type {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_device: bias dtype mismatch",
            code: -4,
        });
    }
    let bias_elems: usize = bias.shape.iter().map(|&d| d as usize).product();
    if bias_elems == m * n {
        // Full C matrix: D2D copy bias -> c_buf.
        let err = unsafe {
            ffi::cudaMemcpy(
                c_buf.as_mut_ptr(),
                bias.buffer.as_ptr(),
                c_bytes,
                ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            )
        };
        if err != ffi::CUDA_SUCCESS {
            return Err(CudaError::CopyFailed {
                msg: "gpu_gemm_device bias D2D",
                code: err,
            });
        }
        return Ok(true);
    }
    if bias_elems == n {
        // Row-broadcast: build on host then copy up. Slower but
        // matches the semantics of the host-Tensor path.
        let row_bytes = n * dt.elem_size;
        let mut host_bias_row = vec![0u8; row_bytes];
        bias.buffer.copy_to_host(&mut host_bias_row)?;
        let mut host_c = vec![0u8; c_bytes];
        for row in 0..m {
            host_c[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(&host_bias_row);
        }
        c_buf.copy_from_host(&host_c)?;
        return Ok(true);
    }
    Ok(false)
}

/// Initialize C (bias or zeros). Section 5 supports: beta == 0 (most
/// cases), full-C bias, or row-broadcast bias. Unknown bias shapes fall
/// back to zeros and rely on beta's multiplication.
fn init_gemm_c_buffer(
    c_buf: &DeviceBuffer,
    c_bias: Option<&DeviceTensor>,
    dt: &GemmDtype,
    beta: f32,
    m: usize,
    n: usize,
    c_bytes: usize,
) -> Result<(), CudaError> {
    if beta != 0.0 {
        if let Some(bias) = c_bias {
            if copy_gemm_bias_into_c(c_buf, bias, dt, m, n, c_bytes)? {
                return Ok(());
            }
        }
    }
    // beta == 0 (cuBLAS ignores C), or unknown bias shape: zero-fill.
    c_buf.copy_from_host(&vec![0u8; c_bytes])
}

fn gemm_out_shape(a_dims: &[i64], m: usize, n: usize) -> Vec<i64> {
    // Preserve leading batch dims from `a`. For a rank-3 input
    // `[B, M, K]` (or `[B, K, M]` when trans_a) we emit rank-3
    // `[B, M, N]`; rank-2 inputs stay rank-2.
    if a_dims.len() > 2 {
        let mut s: Vec<i64> = a_dims[..a_dims.len() - 2].to_vec();
        s.push(m as i64);
        s.push(n as i64);
        s
    } else {
        vec![m as i64, n as i64]
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_gemm_device(
    runtime: &CudaRuntime,
    a: &DeviceTensor,
    b: &DeviceTensor,
    trans_a: bool,
    trans_b: bool,
    alpha: f32,
    beta: f32,
    c_bias: Option<&DeviceTensor>,
) -> Result<DeviceTensor, CudaError> {
    let a_dims = &a.shape;
    let b_dims = &b.shape;
    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_device: need 2D inputs",
            code: -1,
        });
    }

    let (m, k_a) = gemm_a_extents(a_dims, trans_a);
    let (k_b, n) = gemm_b_extents(b_dims, trans_b);
    if k_a != k_b {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_device: k mismatch",
            code: -2,
        });
    }
    let k = k_a;

    let dt = pick_gemm_dtype(a, b)?;

    let c_bytes = m * n * dt.elem_size;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;
    init_gemm_c_buffer(&c_buf, c_bias, &dt, beta, m, n, c_bytes)?;

    // cuBLAS is column-major; swap A/B + transpose flags. Same trick as
    // the host-Tensor dispatch path.
    let transa = if trans_b {
        ffi::cublasOperation_t::CUBLAS_OP_T
    } else {
        ffi::cublasOperation_t::CUBLAS_OP_N
    };
    let transb = if trans_a {
        ffi::cublasOperation_t::CUBLAS_OP_T
    } else {
        ffi::cublasOperation_t::CUBLAS_OP_N
    };

    let lda = if trans_b { k as i32 } else { n as i32 };
    let ldb = if trans_a { m as i32 } else { k as i32 };
    let ldc = n as i32;

    let compute_type = if dt.is_bf16 {
        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_16BF
    } else {
        runtime.precision.to_cublas_compute_type()
    };

    runtime.cublas.gemm_ex(
        transa,
        transb,
        n as i32,
        m as i32,
        k as i32,
        &alpha as *const f32 as *const core::ffi::c_void,
        &b.buffer,
        dt.io_dtype,
        lda,
        &a.buffer,
        dt.io_dtype,
        ldb,
        &beta as *const f32 as *const core::ffi::c_void,
        &c_buf,
        dt.io_dtype,
        ldc,
        compute_type,
    )?;

    super::synchronize()?;

    Ok(DeviceTensor {
        buffer: c_buf,
        shape: gemm_out_shape(a_dims, m, n),
        dtype: dt.out_data_type,
        name: String::new(),
    })
}

/// Device-tensor INT8 GEMM. Mirrors
/// [`crate::cuda::dispatch::gpu_gemm_int8`].
pub fn gpu_gemm_int8_device(
    runtime: &CudaRuntime,
    a: &DeviceTensor,
    b: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    let a_dims = &a.shape;
    let b_dims = &b.shape;
    if a_dims.len() < 2 || b_dims.len() < 2 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_int8_device: need 2D inputs",
            code: -1,
        });
    }
    let m = a_dims[a_dims.len() - 2] as usize;
    let k = a_dims[a_dims.len() - 1] as usize;
    let n = b_dims[b_dims.len() - 1] as usize;
    if !m.is_multiple_of(4) || !n.is_multiple_of(4) || !k.is_multiple_of(4) {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_int8_device: dims must be 4-aligned",
            code: -2,
        });
    }

    let c_bytes = m * n * 4;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;
    c_buf.copy_from_host(&vec![0u8; c_bytes])?;

    let alpha: i32 = 1;
    let beta: i32 = 0;

    runtime.cublas.gemm_ex(
        ffi::cublasOperation_t::CUBLAS_OP_N,
        ffi::cublasOperation_t::CUBLAS_OP_N,
        n as i32,
        m as i32,
        k as i32,
        &alpha as *const i32 as *const core::ffi::c_void,
        &b.buffer,
        ffi::cudaDataType_t::CUDA_R_8I,
        n as i32,
        &a.buffer,
        ffi::cudaDataType_t::CUDA_R_8I,
        k as i32,
        &beta as *const i32 as *const core::ffi::c_void,
        &c_buf,
        ffi::cudaDataType_t::CUDA_R_32I,
        n as i32,
        ffi::cublasComputeType_t::CUBLAS_COMPUTE_32I,
    )?;

    super::synchronize()?;

    Ok(DeviceTensor {
        buffer: c_buf,
        shape: vec![m as i64, n as i64],
        dtype: DataType::Int32,
        name: String::new(),
    })
}

/// Device-tensor Conv2d. Mirrors [`crate::cuda::conv::gpu_conv2d`] but
/// takes device inputs and supports `group`. Bias is not yet supported
/// (would need a device-side bias-add kernel); the caller falls back
/// to CPU dispatch if a bias is present.
///
/// Algo selection: tries a small list of robust algos in priority
/// order, querying workspace size and allocating if needed. Stops at
/// the first algo that succeeds. cuDNN's preference is
/// `IMPLICIT_PRECOMP_GEMM` for typical shapes, but it returns
/// `BAD_PARAM` on some Conv shapes when workspace=0; widening the
/// search restores robustness.
/// cuDNN/output dtype selection for Conv2d.
#[derive(Clone, Copy)]
struct ConvDtype {
    dnn_dtype: ffi::cudnnDataType_t,
    elem_size: usize,
    out_dtype: DataType,
}

fn pick_conv_dtype(x: &DeviceTensor, w: &DeviceTensor) -> Result<ConvDtype, CudaError> {
    let is_bf16 = x.dtype == DataType::BFloat16 && w.dtype == DataType::BFloat16;
    let is_f32 = x.dtype == DataType::Float && w.dtype == DataType::Float;
    if !is_bf16 && !is_f32 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: unsupported dtype",
            code: -2,
        });
    }
    Ok(ConvDtype {
        dnn_dtype: if is_bf16 {
            ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16
        } else {
            ffi::cudnnDataType_t::CUDNN_DATA_FLOAT
        },
        elem_size: if is_bf16 { 2 } else { 4 },
        out_dtype: if is_bf16 {
            DataType::BFloat16
        } else {
            DataType::Float
        },
    })
}

/// Layout dims for a single 4-D Conv pass.
struct ConvDims {
    n: i32,
    c_in: i32,
    h_in: i32,
    w_in: i32,
    k: i32,
    c_w: i32, // c_in / group — filter in-channel count
    kh: i32,
    kw: i32,
}

fn read_conv_dims(x: &DeviceTensor, w: &DeviceTensor, group: i32) -> Result<ConvDims, CudaError> {
    let d = ConvDims {
        n: x.shape[0] as i32,
        c_in: x.shape[1] as i32,
        h_in: x.shape[2] as i32,
        w_in: x.shape[3] as i32,
        k: w.shape[0] as i32,
        c_w: w.shape[1] as i32,
        kh: w.shape[2] as i32,
        kw: w.shape[3] as i32,
    };
    if d.c_in != d.c_w * group || d.k % group != 0 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: input/weight channels do not satisfy c_in == c_w * group, c_out % group == 0",
            code: -6,
        });
    }
    Ok(d)
}

/// Conv-2d spatial parameters in (pad_h, pad_w, stride_h, stride_w, dil_h, dil_w) form.
struct ConvSpatial {
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
}

fn read_conv_spatial(pads: &[i32], strides: &[i32], dilations: &[i32]) -> ConvSpatial {
    let (pad_h, pad_w) = if pads.len() >= 2 {
        (pads[0], pads[1])
    } else {
        (0, 0)
    };
    ConvSpatial {
        pad_h,
        pad_w,
        stride_h: strides.first().copied().unwrap_or(1),
        stride_w: strides.get(1).copied().unwrap_or(1),
        dil_h: dilations.first().copied().unwrap_or(1),
        dil_w: dilations.get(1).copied().unwrap_or(1),
    }
}

fn conv_output_hw(d: &ConvDims, s: &ConvSpatial) -> Result<(i32, i32), CudaError> {
    let h_out = (d.h_in + 2 * s.pad_h - s.dil_h * (d.kh - 1) - 1) / s.stride_h + 1;
    let w_out = (d.w_in + 2 * s.pad_w - s.dil_w * (d.kw - 1) - 1) / s.stride_w + 1;
    if h_out <= 0 || w_out <= 0 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: non-positive output dim",
            code: -3,
        });
    }
    Ok((h_out, w_out))
}

/// Bundle of cuDNN descriptors + device buffers needed to invoke
/// `cudnnConvolutionForward` and to query workspace sizes.
struct ConvForwardCtx<'a> {
    runtime: &'a CudaRuntime,
    x: &'a DeviceTensor,
    w: &'a DeviceTensor,
    x_desc: &'a LocalTensorDesc,
    w_desc: &'a LocalFilterDesc,
    conv_desc: &'a LocalConvDesc,
    y_desc: &'a LocalTensorDesc,
    y_buf: &'a DeviceBuffer,
}

/// Try one cuDNN forward algo. Returns `Ok(true)` on success, `Ok(false)`
/// on a non-fatal cuDNN error (caller continues to the next algo). The
/// `*last_err` slot tracks the most recent cuDNN error code for reporting.
fn try_conv_algo(
    ctx: &ConvForwardCtx<'_>,
    algo: ffi::cudnnConvolutionFwdAlgo_t,
    last_err: &mut ffi::cudnnStatus_t,
) -> Result<bool, CudaError> {
    // Query workspace size for this algo. If query fails we still try
    // with workspace=0 — the forward call may succeed if it doesn't
    // actually need a workspace.
    let mut ws_size: usize = 0;
    let _ = unsafe {
        ffi::cudnnGetConvolutionForwardWorkspaceSize(
            ctx.runtime.cudnn.raw(),
            ctx.x_desc.desc,
            ctx.w_desc.desc,
            ctx.conv_desc.desc,
            ctx.y_desc.desc,
            algo,
            &mut ws_size as *mut usize,
        )
    };

    let workspace = if ws_size > 0 {
        Some(DeviceBuffer::alloc(ws_size)?)
    } else {
        None
    };
    let ws_ptr = workspace
        .as_ref()
        .map(|b| b.as_mut_ptr())
        .unwrap_or(core::ptr::null_mut());

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let err = unsafe {
        ffi::cudnnConvolutionForward(
            ctx.runtime.cudnn.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            ctx.x_desc.desc,
            ctx.x.buffer.as_ptr(),
            ctx.w_desc.desc,
            ctx.w.buffer.as_ptr(),
            ctx.conv_desc.desc,
            algo,
            ws_ptr,
            ws_size,
            &beta as *const f32 as *const core::ffi::c_void,
            ctx.y_desc.desc,
            ctx.y_buf.as_mut_ptr(),
        )
    };
    if err == ffi::CUDNN_STATUS_SUCCESS {
        return Ok(true);
    }
    *last_err = err;
    Ok(false)
}

/// Run cuDNN forward with a priority-ordered algo fallback list.
fn run_conv_forward(ctx: &ConvForwardCtx<'_>) -> Result<(), CudaError> {
    // Algo fallback list: most cuDNN 9.x shapes accept one of these
    // with an explicitly-allocated workspace.
    let candidates = [
        ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
        ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_GEMM,
        ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM,
        ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_DIRECT,
        ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD,
    ];
    let mut last_err = ffi::CUDNN_STATUS_SUCCESS;
    for algo in candidates {
        if try_conv_algo(ctx, algo, &mut last_err)? {
            return Ok(());
        }
    }
    Err(CudaError::DnnError {
        op: "cudnnConvolutionForward (device, all algos failed)",
        code: last_err,
    })
}

/// Per-channel bias add via `cudnnAddTensor`: `bias[1, K, 1, 1] + y[N, K, H, W]`.
fn apply_conv_bias(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    bias: &DeviceTensor,
    k: i32,
    dnn_dtype: ffi::cudnnDataType_t,
    y_desc: &LocalTensorDesc,
    y_buf: &DeviceBuffer,
) -> Result<(), CudaError> {
    if bias.dtype != x.dtype {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: bias dtype must match input dtype",
            code: -7,
        });
    }
    let bias_desc = create_tensor_4d(1, k, 1, 1, dnn_dtype)?;
    let alpha_b: f32 = 1.0;
    let beta_y: f32 = 1.0;
    let err = unsafe {
        ffi::cudnnAddTensor(
            runtime.cudnn.raw(),
            &alpha_b as *const f32 as *const core::ffi::c_void,
            bias_desc.desc,
            bias.buffer.as_ptr(),
            &beta_y as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnAddTensor (Conv bias)",
            code: err,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d_device(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    w: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    pads: &[i32],
    strides: &[i32],
    dilations: &[i32],
    group: i32,
) -> Result<DeviceTensor, CudaError> {
    if x.shape.len() != 4 || w.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: need 4D inputs",
            code: -1,
        });
    }
    if group < 1 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: group must be >= 1",
            code: -5,
        });
    }

    let dt = pick_conv_dtype(x, w)?;
    let d = read_conv_dims(x, w, group)?;
    let s = read_conv_spatial(pads, strides, dilations);
    let (h_out, w_out) = conv_output_hw(&d, &s)?;

    // Descriptors. Filter uses `c_w = c_in / group`, matching ONNX's
    // weight layout `[K, C/group, KH, KW]`.
    let x_desc = create_tensor_4d(d.n, d.c_in, d.h_in, d.w_in, dt.dnn_dtype)?;
    let w_desc = create_filter_4d(d.k, d.c_w, d.kh, d.kw, dt.dnn_dtype)?;
    let conv_desc = create_conv_2d(
        s.pad_h, s.pad_w, s.stride_h, s.stride_w, s.dil_h, s.dil_w, group,
    )?;
    let y_desc = create_tensor_4d(d.n, d.k, h_out, w_out, dt.dnn_dtype)?;

    let y_bytes = (d.n * d.k * h_out * w_out) as usize * dt.elem_size;
    let y_buf = DeviceBuffer::alloc(y_bytes)?;

    run_conv_forward(&ConvForwardCtx {
        runtime,
        x,
        w,
        x_desc: &x_desc,
        w_desc: &w_desc,
        conv_desc: &conv_desc,
        y_desc: &y_desc,
        y_buf: &y_buf,
    })?;

    if let Some(b) = bias {
        apply_conv_bias(runtime, x, b, d.k, dt.dnn_dtype, &y_desc, &y_buf)?;
    }
    super::synchronize()?;

    Ok(DeviceTensor {
        buffer: y_buf,
        shape: vec![d.n as i64, d.k as i64, h_out as i64, w_out as i64],
        dtype: dt.out_dtype,
        name: String::new(),
    })
}

// ── Local cuDNN descriptor wrappers (RAII) ──────────────────────────

struct LocalTensorDesc {
    desc: ffi::cudnnTensorDescriptor_t,
}
impl Drop for LocalTensorDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyTensorDescriptor(self.desc);
        }
    }
}
fn create_tensor_4d(
    n: i32,
    c: i32,
    h: i32,
    w: i32,
    dtype: ffi::cudnnDataType_t,
) -> Result<LocalTensorDesc, CudaError> {
    let mut desc: ffi::cudnnTensorDescriptor_t = core::ptr::null_mut();
    let err = unsafe { ffi::cudnnCreateTensorDescriptor(&mut desc) };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "createTensorDesc",
            code: err,
        });
    }
    let err = unsafe {
        ffi::cudnnSetTensor4dDescriptor(
            desc,
            ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            dtype,
            n,
            c,
            h,
            w,
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        unsafe {
            ffi::cudnnDestroyTensorDescriptor(desc);
        }
        return Err(CudaError::DnnError {
            op: "setTensor4dDesc",
            code: err,
        });
    }
    Ok(LocalTensorDesc { desc })
}

struct LocalFilterDesc {
    desc: ffi::cudnnFilterDescriptor_t,
}
impl Drop for LocalFilterDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyFilterDescriptor(self.desc);
        }
    }
}
fn create_filter_4d(
    k: i32,
    c: i32,
    h: i32,
    w: i32,
    dtype: ffi::cudnnDataType_t,
) -> Result<LocalFilterDesc, CudaError> {
    let mut desc: ffi::cudnnFilterDescriptor_t = core::ptr::null_mut();
    let err = unsafe { ffi::cudnnCreateFilterDescriptor(&mut desc) };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "createFilterDesc",
            code: err,
        });
    }
    let err = unsafe {
        ffi::cudnnSetFilter4dDescriptor(
            desc,
            dtype,
            ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            k,
            c,
            h,
            w,
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        unsafe {
            ffi::cudnnDestroyFilterDescriptor(desc);
        }
        return Err(CudaError::DnnError {
            op: "setFilter4dDesc",
            code: err,
        });
    }
    Ok(LocalFilterDesc { desc })
}

struct LocalConvDesc {
    desc: ffi::cudnnConvolutionDescriptor_t,
}
impl Drop for LocalConvDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyConvolutionDescriptor(self.desc);
        }
    }
}
#[allow(clippy::too_many_arguments)]
fn create_conv_2d(
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    group: i32,
) -> Result<LocalConvDesc, CudaError> {
    let mut desc: ffi::cudnnConvolutionDescriptor_t = core::ptr::null_mut();
    let err = unsafe { ffi::cudnnCreateConvolutionDescriptor(&mut desc) };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "createConvDesc",
            code: err,
        });
    }
    let err = unsafe {
        ffi::cudnnSetConvolution2dDescriptor(
            desc,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            ffi::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        unsafe {
            ffi::cudnnDestroyConvolutionDescriptor(desc);
        }
        return Err(CudaError::DnnError {
            op: "setConv2dDesc",
            code: err,
        });
    }
    if group > 1 {
        let err = unsafe { ffi::cudnnSetConvolutionGroupCount(desc, group) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe {
                ffi::cudnnDestroyConvolutionDescriptor(desc);
            }
            return Err(CudaError::DnnError {
                op: "setConvGroupCount",
                code: err,
            });
        }
    }
    Ok(LocalConvDesc { desc })
}

// Full end-to-end tests live in `onnx-rt/tests/test_cuda.rs` under
// `#[ignore]` so they only run on a machine with a real CUDA device.
