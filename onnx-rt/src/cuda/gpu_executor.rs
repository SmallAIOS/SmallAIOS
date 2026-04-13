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
    execute_graph_gpu_with_weights(graph, input_device_tensors, initializers, None, runtime)
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
    let mut value_map: BTreeMap<String, DeviceTensor> = BTreeMap::new();

    // Move user inputs into the value map. We take ownership of the input
    // DeviceTensors so there is no cloning of device buffers.
    for (name, dev_t) in input_device_tensors {
        // DeviceBuffer is not Clone — duplicate via a fresh alloc + D2D
        // copy. For the Section 5 test cases this is acceptable; callers
        // wanting true zero-copy ownership transfer should construct the
        // value map themselves.
        let cloned = clone_device_tensor(dev_t)
            .map_err(|e| SessionError::ExecutionFailed(format!("input '{}': {}", name, e)))?;
        value_map.insert(name.clone(), cloned);
    }

    // Materialize graph initializers. Prefer the pre-loaded GPU map (from
    // Session::from_safetensors) when available; otherwise bridge each
    // proto host -> device as the legacy path does.
    for init in initializers {
        if let Some(map) = pre_loaded_weights {
            if let Some(dev_src) = map.get(&init.name) {
                let cloned = clone_device_tensor(dev_src).map_err(|e| {
                    SessionError::ExecutionFailed(format!(
                        "pre-loaded weight '{}': {}",
                        init.name, e
                    ))
                })?;
                value_map.insert(init.name.clone(), cloned);
                continue;
            }
        }
        if let Some(host_t) = tensor_from_proto(init) {
            let dev_t = DeviceTensor::from_host(&host_t).map_err(|e| {
                SessionError::ExecutionFailed(format!("initializer '{}': {}", init.name, e))
            })?;
            value_map.insert(init.name.clone(), dev_t);
        }
    }

    // Walk the graph in topological order.
    for node_idx in graph.execution_order() {
        let node = &graph.nodes[node_idx.index()];

        // Resolve input device tensors by name.
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

        let outputs = dispatch_gpu_node(runtime, &node.op_type, &input_refs, &node.attributes)
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
    }

    // Extract graph outputs in declared order.
    let mut results = Vec::new();
    for output_name in &graph.output_names {
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
/// Returns `OpError::InternalError` for any unsupported op type — Section
/// 5 intentionally covers only the four operators with existing GPU
/// kernels (MatMul, Gemm, MatMulInteger, Conv).
fn dispatch_gpu_node(
    runtime: &CudaRuntime,
    op_type: &str,
    inputs: &[Option<&DeviceTensor>],
    attrs: &[AttributeProto],
) -> Result<Vec<DeviceTensor>, OpError> {
    match op_type {
        "MatMul" => {
            let a = take_input(inputs, 0, "MatMul", "A")?;
            let b = take_input(inputs, 1, "MatMul", "B")?;
            let out = gpu_gemm_device(runtime, a, b, false, false, 1.0, 0.0, None)
                .map_err(|e| OpError::InternalError(format!("CUDA MatMul: {}", e)))?;
            Ok(vec![out])
        }
        "Gemm" => {
            let a = take_input(inputs, 0, "Gemm", "A")?;
            let b = take_input(inputs, 1, "Gemm", "B")?;
            let c_bias = inputs.get(2).and_then(|o| *o);

            let mut trans_a = false;
            let mut trans_b = false;
            let mut alpha = 1.0f32;
            let mut beta = 1.0f32;
            for attr in attrs {
                match attr.name.as_str() {
                    "transA" => trans_a = attr.i != 0,
                    "transB" => trans_b = attr.i != 0,
                    "alpha" => alpha = attr.f,
                    "beta" => beta = if c_bias.is_some() { attr.f } else { 0.0 },
                    _ => {}
                }
            }
            if c_bias.is_none() {
                beta = 0.0;
            }
            let out = gpu_gemm_device(runtime, a, b, trans_a, trans_b, alpha, beta, c_bias)
                .map_err(|e| OpError::InternalError(format!("CUDA Gemm: {}", e)))?;
            Ok(vec![out])
        }
        "MatMulInteger" => {
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
        "Conv" => {
            let x = take_input(inputs, 0, "Conv", "X")?;
            let w = take_input(inputs, 1, "Conv", "W")?;
            let bias = inputs.get(2).and_then(|o| *o);

            let mut pads: Vec<i32> = vec![0, 0, 0, 0];
            let mut strides: Vec<i32> = vec![1, 1];
            let mut dilations: Vec<i32> = vec![1, 1];
            for attr in attrs {
                if attr.attr_type != AttributeType::Ints {
                    continue;
                }
                match attr.name.as_str() {
                    "pads" => pads = attr.ints.iter().map(|&v| v as i32).collect(),
                    "strides" => strides = attr.ints.iter().map(|&v| v as i32).collect(),
                    "dilations" => dilations = attr.ints.iter().map(|&v| v as i32).collect(),
                    _ => {}
                }
            }
            let out = gpu_conv2d_device(runtime, x, w, bias, &pads, &strides, &dilations)
                .map_err(|e| OpError::InternalError(format!("CUDA Conv: {}", e)))?;
            Ok(vec![out])
        }
        other => Err(OpError::InternalError(format!(
            "GPU executor: no GPU implementation for {}",
            other
        ))),
    }
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

    let (m, k_a) = if trans_a {
        (
            a_dims[a_dims.len() - 1] as usize,
            a_dims[a_dims.len() - 2] as usize,
        )
    } else {
        (
            a_dims[a_dims.len() - 2] as usize,
            a_dims[a_dims.len() - 1] as usize,
        )
    };
    let (k_b, n) = if trans_b {
        (
            b_dims[b_dims.len() - 1] as usize,
            b_dims[b_dims.len() - 2] as usize,
        )
    } else {
        (
            b_dims[b_dims.len() - 2] as usize,
            b_dims[b_dims.len() - 1] as usize,
        )
    };

    if k_a != k_b {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_device: k mismatch",
            code: -2,
        });
    }
    let k = k_a;

    let is_bf16 = a.dtype == DataType::BFloat16 && b.dtype == DataType::BFloat16;
    let is_f32 = a.dtype == DataType::Float && b.dtype == DataType::Float;
    if !is_bf16 && !is_f32 {
        return Err(CudaError::RuntimeError {
            op: "gpu_gemm_device: unsupported dtype",
            code: -3,
        });
    }

    let elem_size = if is_bf16 { 2 } else { 4 };
    let io_dtype = if is_bf16 {
        ffi::cudaDataType_t::CUDA_R_16BF
    } else {
        ffi::cudaDataType_t::CUDA_R_32F
    };
    let out_data_type = if is_bf16 {
        DataType::BFloat16
    } else {
        DataType::Float
    };

    let c_bytes = m * n * elem_size;
    let c_buf = DeviceBuffer::alloc(c_bytes)?;

    // Initialize C (bias or zeros). For the Section 5 scope we support
    // only: beta == 0 (most cases) OR bias shape == full C matrix with
    // matching dtype. Row-broadcast bias is rare for the supported ops
    // and not exercised by any Gemma-lite test.
    if beta != 0.0 {
        if let Some(bias) = c_bias {
            if bias.dtype != out_data_type {
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
            } else if bias_elems == n {
                // Row-broadcast: build on host then copy up. Slower but
                // matches the semantics of the host-Tensor path.
                let row_bytes = n * elem_size;
                let mut host_bias_row = vec![0u8; row_bytes];
                bias.buffer.copy_to_host(&mut host_bias_row)?;
                let mut host_c = vec![0u8; c_bytes];
                for row in 0..m {
                    host_c[row * row_bytes..(row + 1) * row_bytes].copy_from_slice(&host_bias_row);
                }
                c_buf.copy_from_host(&host_c)?;
            } else {
                // Unknown bias shape: zero-fill and let beta multiply zeros.
                c_buf.copy_from_host(&vec![0u8; c_bytes])?;
            }
        } else {
            c_buf.copy_from_host(&vec![0u8; c_bytes])?;
        }
    } else {
        // beta == 0: cuBLAS ignores C input, but we still need the buffer.
        c_buf.copy_from_host(&vec![0u8; c_bytes])?;
    }

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

    let compute_type = if is_bf16 {
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
        io_dtype,
        lda,
        &a.buffer,
        io_dtype,
        ldb,
        &beta as *const f32 as *const core::ffi::c_void,
        &c_buf,
        io_dtype,
        ldc,
        compute_type,
    )?;

    super::synchronize()?;

    Ok(DeviceTensor {
        buffer: c_buf,
        shape: vec![m as i64, n as i64],
        dtype: out_data_type,
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
/// takes device inputs. Bias is currently ignored (matches the rest of
/// the Section 5 scope — Conv w/ bias is not exercised). A future commit
/// should add a device-side bias add kernel.
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d_device(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    w: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    pads: &[i32],
    strides: &[i32],
    dilations: &[i32],
) -> Result<DeviceTensor, CudaError> {
    if x.shape.len() != 4 || w.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: need 4D inputs",
            code: -1,
        });
    }

    let is_bf16 = x.dtype == DataType::BFloat16 && w.dtype == DataType::BFloat16;
    let is_f32 = x.dtype == DataType::Float && w.dtype == DataType::Float;
    if !is_bf16 && !is_f32 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: unsupported dtype",
            code: -2,
        });
    }
    let dnn_dtype = if is_bf16 {
        ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16
    } else {
        ffi::cudnnDataType_t::CUDNN_DATA_FLOAT
    };
    let elem_size: usize = if is_bf16 { 2 } else { 4 };
    let out_dtype = if is_bf16 {
        DataType::BFloat16
    } else {
        DataType::Float
    };

    let n = x.shape[0] as i32;
    let c_in = x.shape[1] as i32;
    let h_in = x.shape[2] as i32;
    let w_in = x.shape[3] as i32;
    let k = w.shape[0] as i32;
    let kh = w.shape[2] as i32;
    let kw = w.shape[3] as i32;

    let (pad_h, pad_w) = if pads.len() >= 2 {
        (pads[0], pads[1])
    } else {
        (0, 0)
    };
    let stride_h = strides.first().copied().unwrap_or(1);
    let stride_w = strides.get(1).copied().unwrap_or(1);
    let dil_h = dilations.first().copied().unwrap_or(1);
    let dil_w = dilations.get(1).copied().unwrap_or(1);

    let h_out = (h_in + 2 * pad_h - dil_h * (kh - 1) - 1) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - dil_w * (kw - 1) - 1) / stride_w + 1;
    if h_out <= 0 || w_out <= 0 {
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: non-positive output dim",
            code: -3,
        });
    }

    // Descriptors — same shape as cuda::conv but inlined to avoid
    // exposing its private TensorDesc/FilterDesc/ConvDesc types.
    let x_desc = create_tensor_4d(n, c_in, h_in, w_in, dnn_dtype)?;
    let w_desc = create_filter_4d(k, c_in, kh, kw, dnn_dtype)?;
    let conv_desc = create_conv_2d(pad_h, pad_w, stride_h, stride_w, dil_h, dil_w)?;
    let y_desc = create_tensor_4d(n, k, h_out, w_out, dnn_dtype)?;

    let y_bytes = (n * k * h_out * w_out) as usize * elem_size;
    let y_buf = DeviceBuffer::alloc(y_bytes)?;
    y_buf.copy_from_host(&vec![0u8; y_bytes])?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let err = unsafe {
        ffi::cudnnConvolutionForward(
            runtime.cudnn.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x.buffer.as_ptr(),
            w_desc.desc,
            w.buffer.as_ptr(),
            conv_desc.desc,
            ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            core::ptr::null_mut(),
            0,
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnConvolutionForward (device)",
            code: err,
        });
    }
    super::synchronize()?;

    if bias.is_some() {
        // TODO: device-side bias add. For now return an error so the
        // caller knows Conv+bias is not wired up in the GPU executor.
        return Err(CudaError::RuntimeError {
            op: "gpu_conv2d_device: bias not yet supported",
            code: -4,
        });
    }

    Ok(DeviceTensor {
        buffer: y_buf,
        shape: vec![n as i64, k as i64, h_out as i64, w_out as i64],
        dtype: out_dtype,
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
fn create_conv_2d(
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
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
    Ok(LocalConvDesc { desc })
}

// Full end-to-end tests live in `onnx-rt/tests/test_cuda.rs` under
// `#[ignore]` so they only run on a machine with a real CUDA device.
