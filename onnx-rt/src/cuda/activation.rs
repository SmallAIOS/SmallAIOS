// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU activation functions (Relu, Clip, LeakyRelu) via cuDNN.

use super::descriptors::{dnn_dtype, ActivationDesc, TensorDesc};
use super::ffi;
use super::gpu_executor::DeviceTensor;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};

/// Convert an `i64` tensor dimension into the `i32` cuDNN expects, returning
/// a runtime error rather than truncating on overflow.
fn dim_to_i32(d: i64, op: &'static str) -> Result<i32, CudaError> {
    i32::try_from(d).map_err(|_| CudaError::RuntimeError { op, code: -10 })
}

/// Apply an arbitrary cuDNN activation mode element-wise.
fn run_activation(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    mode: ffi::cudnnActivationMode_t,
    coef: f64,
) -> Result<DeviceTensor, CudaError> {
    let dt = dnn_dtype(x.dtype)?;
    // cuDNN tensor descriptors are 4-D; pad with leading 1s for lower-
    // rank tensors. Activations are element-wise, so as long as the
    // total element count matches we get the same result. Use checked
    // arithmetic in i128 for the product so a malformed shape can't
    // silently overflow into a valid-looking i32.
    let mut total: i128 = 1;
    for d in &x.shape {
        if *d < 0 {
            return Err(CudaError::RuntimeError {
                op: "activation: negative shape dim",
                code: -11,
            });
        }
        total = total
            .checked_mul(*d as i128)
            .ok_or(CudaError::RuntimeError {
                op: "activation: shape product overflow",
                code: -12,
            })?;
    }
    let total_i32 = i32::try_from(total).map_err(|_| CudaError::RuntimeError {
        op: "activation: total elements exceed i32",
        code: -13,
    })?;
    let (n, c, h, w) = match x.shape.as_slice() {
        [a, b, c, d] => (
            dim_to_i32(*a, "activation: dim N overflow")?,
            dim_to_i32(*b, "activation: dim C overflow")?,
            dim_to_i32(*c, "activation: dim H overflow")?,
            dim_to_i32(*d, "activation: dim W overflow")?,
        ),
        [a, b, c] => (
            dim_to_i32(*a, "activation: dim N overflow")?,
            dim_to_i32(*b, "activation: dim C overflow")?,
            dim_to_i32(*c, "activation: dim H overflow")?,
            1,
        ),
        [a, b] => (
            dim_to_i32(*a, "activation: dim N overflow")?,
            dim_to_i32(*b, "activation: dim C overflow")?,
            1,
            1,
        ),
        [a] => (dim_to_i32(*a, "activation: dim N overflow")?, 1, 1, 1),
        [] => (1, 1, 1, 1),
        _ => (1, 1, 1, total_i32),
    };

    let in_desc = TensorDesc::new_4d(n, c, h, w, dt)?;
    let out_desc = TensorDesc::new_4d(n, c, h, w, dt)?;
    let act = ActivationDesc::new(mode, coef)?;

    let y_buf = DeviceBuffer::alloc(x.byte_size())?;
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let err = unsafe {
        ffi::cudnnActivationForward(
            runtime.cudnn.raw(),
            act.desc,
            &alpha as *const f32 as *const _,
            in_desc.desc,
            x.buffer.as_ptr(),
            &beta as *const f32 as *const _,
            out_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnActivationForward",
            code: err,
        });
    }
    Ok(DeviceTensor {
        buffer: y_buf,
        shape: x.shape.clone(),
        dtype: x.dtype,
        name: alloc::string::String::new(),
    })
}

/// `Relu` — `max(0, x)`.
pub fn gpu_relu(runtime: &CudaRuntime, x: &DeviceTensor) -> Result<DeviceTensor, CudaError> {
    run_activation(
        runtime,
        x,
        ffi::cudnnActivationMode_t::CUDNN_ACTIVATION_RELU,
        0.0,
    )
}

/// `Clip(min, max)` — implemented via cuDNN clipped-Relu when min == 0
/// and max > 0; falls back to a runtime error otherwise (caller handles
/// CPU fallback).
pub fn gpu_clip(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    min: f32,
    max: f32,
) -> Result<DeviceTensor, CudaError> {
    if min != 0.0 || max <= 0.0 {
        return Err(CudaError::RuntimeError {
            op: "gpu_clip: only min=0 with positive max is supported on GPU",
            code: -3,
        });
    }
    run_activation(
        runtime,
        x,
        ffi::cudnnActivationMode_t::CUDNN_ACTIVATION_CLIPPED_RELU,
        max as f64,
    )
}

/// `LeakyRelu(alpha)` — cuDNN does not have a true leaky-Relu mode, so
/// this returns an error and the hybrid executor will fall back to CPU.
pub fn gpu_leaky_relu(
    _runtime: &CudaRuntime,
    _x: &DeviceTensor,
    _alpha: f32,
) -> Result<DeviceTensor, CudaError> {
    Err(CudaError::RuntimeError {
        op: "gpu_leaky_relu: not supported by cuDNN; use CPU fallback",
        code: -4,
    })
}
