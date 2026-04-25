// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU 2-D pooling (MaxPool, AveragePool, GlobalAveragePool) via cuDNN.

use super::descriptors::{dnn_dtype, PoolDesc, TensorDesc};
use super::ffi;
use super::gpu_executor::DeviceTensor;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::operators::PoolAttrs;
use crate::tensor::DataType;

/// Convert an `i64` shape dim into the `i32` cuDNN expects, returning an
/// error rather than truncating on overflow.
fn dim_to_i32(d: i64, op: &'static str) -> Result<i32, CudaError> {
    i32::try_from(d).map_err(|_| CudaError::RuntimeError { op, code: -10 })
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
    if kh <= 0 || kw <= 0 {
        return Err(CudaError::RuntimeError {
            op: "pool: kernel_shape must be positive",
            code: -4,
        });
    }
    if stride_h <= 0 || stride_w <= 0 {
        return Err(CudaError::RuntimeError {
            op: "pool: strides must be positive",
            code: -5,
        });
    }
    if pad_h < 0 || pad_w < 0 {
        return Err(CudaError::RuntimeError {
            op: "pool: pads must be non-negative",
            code: -6,
        });
    }
    let dt = dnn_dtype(x.dtype)?;
    let n = dim_to_i32(x.shape[0], "pool: dim N overflow")?;
    let c = dim_to_i32(x.shape[1], "pool: dim C overflow")?;
    let h_in = dim_to_i32(x.shape[2], "pool: dim H overflow")?;
    let w_in = dim_to_i32(x.shape[3], "pool: dim W overflow")?;
    // Output-dim arithmetic in i64 then narrowed to i32 — guards against
    // pathological pad/kernel combinations producing overflow during the
    // intermediate `h_in + 2*pad_h - kh`.
    let h_out_i64 = (h_in as i64 + 2 * pad_h as i64 - kh as i64) / stride_h as i64 + 1;
    let w_out_i64 = (w_in as i64 + 2 * pad_w as i64 - kw as i64) / stride_w as i64 + 1;
    if h_out_i64 <= 0 || w_out_i64 <= 0 {
        return Err(CudaError::RuntimeError {
            op: "pool: non-positive output dim",
            code: -3,
        });
    }
    let h_out = dim_to_i32(h_out_i64, "pool: output H overflow")?;
    let w_out = dim_to_i32(w_out_i64, "pool: output W overflow")?;

    let x_desc = TensorDesc::new_4d(n, c, h_in, w_in, dt)?;
    let y_desc = TensorDesc::new_4d(n, c, h_out, w_out, dt)?;
    let pool = PoolDesc::new_2d(mode, kh, kw, pad_h, pad_w, stride_h, stride_w)?;

    let elem_size = match x.dtype {
        DataType::Float => 4u64,
        DataType::BFloat16 => 2u64,
        _ => 4u64,
    };
    // Compute y_bytes as u64 with checked multiplication so an oversized
    // tensor produces a clean error instead of a wrap-around alloc.
    let y_elems: u64 = (n as u64)
        .checked_mul(c as u64)
        .and_then(|v| v.checked_mul(h_out as u64))
        .and_then(|v| v.checked_mul(w_out as u64))
        .ok_or(CudaError::RuntimeError {
            op: "pool: output element count overflow",
            code: -7,
        })?;
    let y_bytes_u64 = y_elems
        .checked_mul(elem_size)
        .ok_or(CudaError::RuntimeError {
            op: "pool: output byte size overflow",
            code: -8,
        })?;
    let y_bytes = usize::try_from(y_bytes_u64).map_err(|_| CudaError::RuntimeError {
        op: "pool: output byte size exceeds usize",
        code: -9,
    })?;
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

/// Validate ONNX `PoolAttrs` against cuDNN's capabilities and pull out
/// the symmetric pad / kernel / stride values. cuDNN's
/// `cudnnSetPooling2dDescriptor` only takes a single pad/stride/kernel
/// per axis, so asymmetric padding (`pad_begin != pad_end`) and
/// `ceil_mode = true` are not representable and must fall back to CPU.
fn unpack_pool_attrs(
    attrs: &PoolAttrs,
    op: &'static str,
) -> Result<(i32, i32, i32, i32, i32, i32), CudaError> {
    if attrs.ceil_mode {
        return Err(CudaError::RuntimeError { op, code: -20 });
    }
    // ONNX 2-D pads are [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end].
    // Default-constructed PoolAttrs uses [0; 4], so an unset value is
    // implicitly symmetric.
    if attrs.pads[0] != attrs.pads[2] || attrs.pads[1] != attrs.pads[3] {
        return Err(CudaError::RuntimeError { op, code: -21 });
    }
    Ok((
        attrs.kernel_shape[0],
        attrs.kernel_shape[1],
        attrs.pads[0],
        attrs.pads[1],
        attrs.strides[0],
        attrs.strides[1],
    ))
}

/// 2-D max-pooling with explicit attributes.
pub fn gpu_maxpool(
    runtime: &CudaRuntime,
    x: &DeviceTensor,
    attrs: &PoolAttrs,
) -> Result<DeviceTensor, CudaError> {
    let (kh, kw, pad_h, pad_w, stride_h, stride_w) =
        unpack_pool_attrs(attrs, "gpu_maxpool: ceil_mode/asymmetric pads unsupported")?;
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
    let (kh, kw, pad_h, pad_w, stride_h, stride_w) = unpack_pool_attrs(
        attrs,
        "gpu_averagepool: ceil_mode/asymmetric pads unsupported",
    )?;
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
    let kh = dim_to_i32(x.shape[2], "gpu_globalaveragepool: H overflow")?;
    let kw = dim_to_i32(x.shape[3], "gpu_globalaveragepool: W overflow")?;
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
