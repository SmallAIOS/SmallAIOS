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
    gpu_executor::{tensor_to_device, DeviceTensor},
    pool::{gpu_averagepool, gpu_globalaveragepool, gpu_maxpool},
    CudaRuntime,
};
use crate::executor;
use crate::graph::ExecutionGraph;
use crate::onnx_types::{AttributeProto, TensorProto};
use crate::operators::{BatchNormAttrs, ConvAttrs, OpError, PoolAttrs};
use crate::session::{InferenceOutput, SessionError};
use crate::tensor::{DataType, Tensor};

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
    let result = match op_type {
        "Conv" => {
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
            .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "Gemm" => {
            if inputs.len() < 2 {
                return Ok(None);
            }
            let mut trans_a = false;
            let mut trans_b = false;
            let mut alpha = 1.0f32;
            let mut beta = 1.0f32;
            for a in attrs {
                match a.name.as_str() {
                    "transA" => trans_a = a.i != 0,
                    "transB" => trans_b = a.i != 0,
                    "alpha" => alpha = a.f,
                    "beta" => beta = a.f,
                    _ => {}
                }
            }
            let bias = inputs.get(2).map(|d| d.as_ref());
            cuda::gpu_executor::gpu_gemm_device(
                rt, &inputs[0], &inputs[1], trans_a, trans_b, alpha, beta, bias,
            )
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "MatMul" => {
            if inputs.len() != 2 {
                return Ok(None);
            }
            cuda::gpu_executor::gpu_gemm_device(
                rt, &inputs[0], &inputs[1], false, false, 1.0, 0.0, None,
            )
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "BatchNormalization" => {
            if inputs.len() != 5 {
                return Ok(None);
            }
            let bn = BatchNormAttrs::from_attributes(attrs)
                .map_err(|e| op_err_to_session(node_name, e))?;
            gpu_batchnorm(
                rt, &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], bn.epsilon,
            )
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "Relu" => {
            if inputs.len() != 1 {
                return Ok(None);
            }
            gpu_relu(rt, &inputs[0])
                .map(Some)
                .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "Clip" => {
            // Clip in opset >= 11 takes min/max as inputs (idx 1, 2);
            // here all three inputs would already be on device, but
            // min/max are scalars so easier to require the attribute
            // form. Fall back to CPU otherwise.
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
                (Some(mn), Some(mx)) => gpu_clip(rt, &inputs[0], mn, mx)
                    .map(Some)
                    .map_err(|e| cuda_err_to_session(node_name, e))?,
                _ => return Ok(None),
            }
        }
        "MaxPool" => {
            let p = PoolAttrs::from_attributes(attrs, true)
                .map_err(|e| op_err_to_session(node_name, e))?;
            gpu_maxpool(rt, &inputs[0], &p)
                .map(Some)
                .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "AveragePool" => {
            let p = PoolAttrs::from_attributes(attrs, true)
                .map_err(|e| op_err_to_session(node_name, e))?;
            gpu_averagepool(rt, &inputs[0], &p)
                .map(Some)
                .map_err(|e| cuda_err_to_session(node_name, e))?
        }
        "GlobalAveragePool" => gpu_globalaveragepool(rt, &inputs[0])
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e))?,
        "Add" => gpu_add(rt, &inputs[0], &inputs[1])
            .map(Some)
            .map_err(|e| cuda_err_to_session(node_name, e))?,
        _ => None,
    };
    Ok(result)
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

    for node_idx in graph.execution_order() {
        let node = &graph.nodes[node_idx.index()];

        // Decide GPU vs CPU dispatch based on op type + dtype of the
        // first input tensor (proxy for op dtype).
        let first_input_dtype = node
            .inputs
            .iter()
            .find(|n| !n.is_empty())
            .and_then(|n| value_map.get(n).map(|v| v.dtype()));

        let gpu_eligible = match first_input_dtype {
            Some(dt) => gpu_op_supported(&node.op_type, dt),
            None => false,
        };

        let mut handled_on_gpu = false;

        if gpu_eligible {
            // Pull all inputs to device. If any input materialization
            // fails, fall through to CPU.
            let mut all_ok = true;
            for name in &node.inputs {
                if name.is_empty() {
                    continue;
                }
                if ensure_device(&mut value_map, name, runtime).is_err() {
                    all_ok = false;
                    break;
                }
            }
            if all_ok {
                let mut device_inputs: Vec<Arc<DeviceTensor>> = Vec::new();
                for name in &node.inputs {
                    if name.is_empty() {
                        continue;
                    }
                    device_inputs.push(require_device(&value_map, name)?);
                }
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
                match dispatched {
                    Some(out_dt) => {
                        if let Some(name) = node.outputs.first() {
                            if !name.is_empty() {
                                value_map
                                    .insert(name.clone(), ValueLocation::Device(Arc::new(out_dt)));
                            }
                        }
                        handled_on_gpu = true;
                    }
                    None => {
                        // Fall through to CPU dispatch.
                    }
                }
            }
        }

        if !handled_on_gpu {
            // Bring all device-resident inputs back to host first.
            for name in &node.inputs {
                if !name.is_empty() {
                    ensure_host(&mut value_map, name)?;
                }
            }
            let inputs_host: Vec<Option<&Tensor>> = node
                .inputs
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
                .collect();

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
        }
    }

    let mut results = Vec::new();
    for output_name in &graph.output_names {
        ensure_host(&mut value_map, output_name)?;
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

    let _ = ConvAttrs::default(); // suppress unused-import warning
    Ok(results)
}
