// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! RAII wrappers for cuDNN tensor / activation / pool / op-tensor
//! descriptors. Shared across the device-op modules
//! (batchnorm, activation, pool, elementwise).

use super::ffi;
use super::CudaError;
use crate::tensor::DataType;

/// Map a tensor [`DataType`] to the corresponding cuDNN tensor data
/// type. Shared across the device-op modules so they don't each
/// hand-roll the same conversion.
pub(crate) fn dnn_dtype(dt: DataType) -> Result<ffi::cudnnDataType_t, CudaError> {
    match dt {
        DataType::Float => Ok(ffi::cudnnDataType_t::CUDNN_DATA_FLOAT),
        DataType::BFloat16 => Ok(ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16),
        _ => Err(CudaError::RuntimeError {
            op: "cuda op: unsupported dtype",
            code: -1,
        }),
    }
}

/// Convert a cuDNN status code into a `CudaError::DnnError` when it is
/// not `CUDNN_STATUS_SUCCESS`. Shared across the descriptor wrappers
/// so each one doesn't hand-roll the same `if err != SUCCESS` check.
#[inline]
fn check_dnn(status: ffi::cudnnStatus_t, op: &'static str) -> Result<(), CudaError> {
    if status == ffi::CUDNN_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(CudaError::DnnError { op, code: status })
    }
}

/// Build a descriptor by:
///   1. allocating a handle via `create_fn`,
///   2. configuring it via `set_call` (a closure that returns the
///      cuDNN status code from the `cudnnSet*Descriptor` call),
///   3. destroying the handle via `destroy_fn` if step 2 fails.
///
/// This collapses the create-and-set boilerplate that every descriptor
/// wrapper used to hand-roll.
#[inline]
fn build_desc<H>(
    create_op: &'static str,
    set_op: &'static str,
    create_fn: unsafe extern "C" fn(*mut H) -> ffi::cudnnStatus_t,
    set_call: impl FnOnce(H) -> ffi::cudnnStatus_t,
    destroy_fn: unsafe extern "C" fn(H) -> ffi::cudnnStatus_t,
) -> Result<H, CudaError>
where
    H: Copy,
{
    // SAFETY: we initialise `desc` to a null/zero handle before calling
    // `create_fn`, which is documented to write a valid handle on
    // success.
    let mut desc: H = unsafe { core::mem::zeroed() };
    // SAFETY: `create_fn` matches the descriptor type `H` and only
    // writes through the `&mut H` we pass.
    check_dnn(unsafe { create_fn(&mut desc) }, create_op)?;
    if let Err(e) = check_dnn(set_call(desc), set_op) {
        // SAFETY: `desc` was successfully created above; we must
        // destroy it to avoid leaking the handle on the failure path.
        unsafe {
            let _ = destroy_fn(desc);
        }
        return Err(e);
    }
    Ok(desc)
}

pub(crate) struct TensorDesc {
    pub desc: ffi::cudnnTensorDescriptor_t,
}

impl TensorDesc {
    pub fn new_4d(
        n: i32,
        c: i32,
        h: i32,
        w: i32,
        dtype: ffi::cudnnDataType_t,
    ) -> Result<Self, CudaError> {
        let desc = build_desc(
            "createTensorDesc",
            "setTensor4dDesc",
            ffi::cudnnCreateTensorDescriptor,
            |d| unsafe {
                ffi::cudnnSetTensor4dDescriptor(
                    d,
                    ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                    dtype,
                    n,
                    c,
                    h,
                    w,
                )
            },
            ffi::cudnnDestroyTensorDescriptor,
        )?;
        Ok(Self { desc })
    }
}

impl Drop for TensorDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyTensorDescriptor(self.desc);
        }
    }
}

pub(crate) struct PoolDesc {
    pub desc: ffi::cudnnPoolingDescriptor_t,
}

impl PoolDesc {
    #[allow(clippy::too_many_arguments)]
    pub fn new_2d(
        mode: ffi::cudnnPoolingMode_t,
        kh: i32,
        kw: i32,
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
    ) -> Result<Self, CudaError> {
        let desc = build_desc(
            "createPoolingDesc",
            "setPooling2dDesc",
            ffi::cudnnCreatePoolingDescriptor,
            |d| unsafe {
                ffi::cudnnSetPooling2dDescriptor(
                    d,
                    mode,
                    ffi::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    kh,
                    kw,
                    pad_h,
                    pad_w,
                    stride_h,
                    stride_w,
                )
            },
            ffi::cudnnDestroyPoolingDescriptor,
        )?;
        Ok(Self { desc })
    }
}

impl Drop for PoolDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyPoolingDescriptor(self.desc);
        }
    }
}

pub(crate) struct ActivationDesc {
    pub desc: ffi::cudnnActivationDescriptor_t,
}

impl ActivationDesc {
    pub fn new(mode: ffi::cudnnActivationMode_t, coef: f64) -> Result<Self, CudaError> {
        let desc = build_desc(
            "createActivationDesc",
            "setActivationDesc",
            ffi::cudnnCreateActivationDescriptor,
            |d| unsafe {
                ffi::cudnnSetActivationDescriptor(
                    d,
                    mode,
                    ffi::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                    coef,
                )
            },
            ffi::cudnnDestroyActivationDescriptor,
        )?;
        Ok(Self { desc })
    }
}

impl Drop for ActivationDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyActivationDescriptor(self.desc);
        }
    }
}

pub(crate) struct OpTensorDesc {
    pub desc: ffi::cudnnOpTensorDescriptor_t,
}

impl OpTensorDesc {
    pub fn new(
        op: ffi::cudnnOpTensorOp_t,
        comp_type: ffi::cudnnDataType_t,
    ) -> Result<Self, CudaError> {
        let desc = build_desc(
            "createOpTensorDesc",
            "setOpTensorDesc",
            ffi::cudnnCreateOpTensorDescriptor,
            |d| unsafe {
                ffi::cudnnSetOpTensorDescriptor(
                    d,
                    op,
                    comp_type,
                    ffi::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                )
            },
            ffi::cudnnDestroyOpTensorDescriptor,
        )?;
        Ok(Self { desc })
    }
}

impl Drop for OpTensorDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyOpTensorDescriptor(self.desc);
        }
    }
}
