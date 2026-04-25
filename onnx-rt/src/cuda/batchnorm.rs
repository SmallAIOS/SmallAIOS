// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU BatchNormalization (inference) via cuDNN.

use super::descriptors::TensorDesc;
use super::ffi;
use super::gpu_executor::DeviceTensor;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::tensor::DataType;

/// Inference-mode `cudnnBatchNormalizationForwardInference` over a 4-D
/// `[N, C, H, W]` input. Scale, bias, mean, variance are
/// per-channel (`[1, C, 1, 1]`) `DeviceTensor`s.
pub fn gpu_batchnorm(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    scale: &DeviceTensor,
    bias: &DeviceTensor,
    mean: &DeviceTensor,
    var: &DeviceTensor,
    epsilon: f32,
) -> Result<DeviceTensor, CudaError> {
    if x.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_batchnorm: input must be 4D NCHW",
            code: -1,
        });
    }
    let dnn_dtype = match x.dtype {
        DataType::Float => ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
        DataType::BFloat16 => ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
        _ => {
            return Err(CudaError::RuntimeError {
                op: "gpu_batchnorm: unsupported dtype",
                code: -2,
            })
        }
    };
    // BN parameter buffers are always f32 in cuDNN regardless of x dtype.
    let n = x.shape[0] as i32;
    let c = x.shape[1] as i32;
    let h = x.shape[2] as i32;
    let w = x.shape[3] as i32;

    let x_desc = TensorDesc::new_4d(n, c, h, w, dnn_dtype)?;
    let y_desc = TensorDesc::new_4d(n, c, h, w, dnn_dtype)?;
    let bn_desc = TensorDesc::new_4d(1, c, 1, 1, ffi::cudnnDataType_t::CUDNN_DATA_FLOAT)?;

    let y_bytes = x.byte_size();
    let y_buf = DeviceBuffer::alloc(y_bytes)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let err = unsafe {
        ffi::cudnnBatchNormalizationForwardInference(
            runtime.cudnn.raw(),
            ffi::cudnnBatchNormMode_t::CUDNN_BATCHNORM_SPATIAL,
            &alpha as *const f32 as *const _,
            &beta as *const f32 as *const _,
            x_desc.desc,
            x.buffer.as_ptr(),
            y_desc.desc,
            y_buf.as_mut_ptr(),
            bn_desc.desc,
            scale.buffer.as_ptr(),
            bias.buffer.as_ptr(),
            mean.buffer.as_ptr(),
            var.buffer.as_ptr(),
            epsilon as f64,
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnBatchNormalizationForwardInference",
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
