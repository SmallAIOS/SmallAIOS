// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU element-wise operators (currently: Add) via `cudnnOpTensor`.

use super::descriptors::{dnn_dtype, OpTensorDesc, TensorDesc};
use super::ffi;
use super::gpu_executor::DeviceTensor;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};

/// Element-wise `Add` for tensors of identical shape (no broadcasting
/// beyond what cuDNN's `OpTensor` natively handles). Caller must verify
/// shape compatibility before calling; mismatched ranks return an error
/// and the hybrid executor will fall back to CPU.
pub fn gpu_add(
    runtime: &CudaRuntime,
    a: &DeviceTensor,
    b: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    if a.shape != b.shape {
        return Err(CudaError::RuntimeError {
            op: "gpu_add: shapes must match (cuDNN broadcasting limited)",
            code: -2,
        });
    }
    if a.dtype != b.dtype {
        return Err(CudaError::RuntimeError {
            op: "gpu_add: dtypes must match",
            code: -3,
        });
    }
    if a.shape.len() != 4 {
        return Err(CudaError::RuntimeError {
            op: "gpu_add: input must be 4D",
            code: -4,
        });
    }

    let dt = dnn_dtype(a.dtype)?;
    let n = a.shape[0] as i32;
    let c = a.shape[1] as i32;
    let h = a.shape[2] as i32;
    let w = a.shape[3] as i32;

    let a_desc = TensorDesc::new_4d(n, c, h, w, dt)?;
    let b_desc = TensorDesc::new_4d(n, c, h, w, dt)?;
    let c_desc = TensorDesc::new_4d(n, c, h, w, dt)?;
    let op = OpTensorDesc::new(
        ffi::cudnnOpTensorOp_t::CUDNN_OP_TENSOR_ADD,
        ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
    )?;

    let y_buf = DeviceBuffer::alloc(a.byte_size())?;
    let alpha1: f32 = 1.0;
    let alpha2: f32 = 1.0;
    let beta: f32 = 0.0;
    let err = unsafe {
        ffi::cudnnOpTensor(
            runtime.cudnn.raw(),
            op.desc,
            &alpha1 as *const f32 as *const _,
            a_desc.desc,
            a.buffer.as_ptr(),
            &alpha2 as *const f32 as *const _,
            b_desc.desc,
            b.buffer.as_ptr(),
            &beta as *const f32 as *const _,
            c_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnOpTensor(ADD)",
            code: err,
        });
    }
    Ok(DeviceTensor {
        buffer: y_buf,
        shape: a.shape.clone(),
        dtype: a.dtype,
        name: alloc::string::String::new(),
    })
}
