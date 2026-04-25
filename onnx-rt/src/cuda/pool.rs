// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU 2-D pooling (MaxPool, AveragePool, GlobalAveragePool) via cuDNN.

use super::descriptors::{PoolDesc, TensorDesc};
use super::ffi;
use super::gpu_executor::DeviceTensor;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::operators::PoolAttrs;
use crate::tensor::DataType;

fn dnn_dtype(dt: DataType) -> Result<ffi::cudnnDataType_t, CudaError> {
    match dt {
        DataType::Float => Ok(ffi::cudnnDataType_t::CUDNN_DATA_FLOAT),
        DataType::BFloat16 => Ok(ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16),
        _ => Err(CudaError::RuntimeError {
            op: "pool: unsupported dtype",
            code: -1,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_pool(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    mode: ffi::cudnnPoolingMode_t,
    kh: i32,
    kw: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
) -> Result<DeviceTensor, CudaError> {
    if x.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "pool: input must be 4D NCHW",
            code: -2,
        });
    }
    let dt = dnn_dtype(x.dtype)?;
    let n = x.shape[0] as i32;
    let c = x.shape[1] as i32;
    let h_in = x.shape[2] as i32;
    let w_in = x.shape[3] as i32;
    let h_out = (h_in + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - kw) / stride_w + 1;
    if h_out <= 0 || w_out <= 0 {
        return Err(CudaError::RuntimeError {
            op: "pool: non-positive output dim",
            code: -3,
        });
    }

    let x_desc = TensorDesc::new_4d(n, c, h_in, w_in, dt)?;
    let y_desc = TensorDesc::new_4d(n, c, h_out, w_out, dt)?;
    let pool = PoolDesc::new_2d(mode, kh, kw, pad_h, pad_w, stride_h, stride_w)?;

    let elem_size = match x.dtype {
        DataType::Float => 4,
        DataType::BFloat16 => 2,
        _ => 4,
    };
    let y_bytes = (n * c * h_out * w_out) as usize * elem_size;
    let y_buf = DeviceBuffer::alloc(y_bytes)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let err = unsafe {
        ffi::cudnnPoolingForward(
            runtime.cudnn.raw(),
            pool.desc,
            &alpha as *const f32 as *const _,
            x_desc.desc,
            x.buffer.as_ptr(),
            &beta as *const f32 as *const _,
            y_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnPoolingForward",
            code: err,
        });
    }

    Ok(DeviceTensor {
        buffer: y_buf,
        shape: alloc::vec![n as i64, c as i64, h_out as i64, w_out as i64],
        dtype: x.dtype,
        name: alloc::string::String::new(),
    })
}

/// 2-D max-pooling with explicit attributes.
pub fn gpu_maxpool(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    attrs: &PoolAttrs,
) -> Result<DeviceTensor, CudaError> {
    let kh = attrs.kernel_shape[0];
    let kw = attrs.kernel_shape[1];
    let pad_h = attrs.pads[0];
    let pad_w = attrs.pads[1];
    let stride_h = attrs.strides[0];
    let stride_w = attrs.strides[1];
    run_pool(
        runtime,
        x,
        ffi::cudnnPoolingMode_t::CUDNN_POOLING_MAX,
        kh,
        kw,
        pad_h,
        pad_w,
        stride_h,
        stride_w,
    )
}

/// 2-D average-pooling with explicit attributes. Honours
/// `count_include_pad` to pick the cuDNN mode.
pub fn gpu_averagepool(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    attrs: &PoolAttrs,
) -> Result<DeviceTensor, CudaError> {
    let kh = attrs.kernel_shape[0];
    let kw = attrs.kernel_shape[1];
    let pad_h = attrs.pads[0];
    let pad_w = attrs.pads[1];
    let stride_h = attrs.strides[0];
    let stride_w = attrs.strides[1];
    let mode = if attrs.count_include_pad {
        ffi::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING
    } else {
        ffi::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING
    };
    run_pool(runtime, x, mode, kh, kw, pad_h, pad_w, stride_h, stride_w)
}

/// `GlobalAveragePool` reduces spatial dims to 1×1, derived from the
/// input shape.
pub fn gpu_globalaveragepool(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    if x.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_globalaveragepool: input must be 4D NCHW",
            code: -1,
        });
    }
    let kh = x.shape[2] as i32;
    let kw = x.shape[3] as i32;
    run_pool(
        runtime,
        x,
        ffi::cudnnPoolingMode_t::CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING,
        kh,
        kw,
        0,
        0,
        1,
        1,
    )
}
