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
        let mut desc: ffi::cudnnPoolingDescriptor_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreatePoolingDescriptor(&mut desc) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError {
                op: "createPoolingDesc",
                code: err,
            });
        }
        let err = unsafe {
            ffi::cudnnSetPooling2dDescriptor(
                desc,
                mode,
                ffi::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                kh,
                kw,
                pad_h,
                pad_w,
                stride_h,
                stride_w,
            )
        };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe {
                ffi::cudnnDestroyPoolingDescriptor(desc);
            }
            return Err(CudaError::DnnError {
                op: "setPooling2dDesc",
                code: err,
            });
        }
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
        let mut desc: ffi::cudnnActivationDescriptor_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreateActivationDescriptor(&mut desc) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError {
                op: "createActivationDesc",
                code: err,
            });
        }
        let err = unsafe {
            ffi::cudnnSetActivationDescriptor(
                desc,
                mode,
                ffi::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
                coef,
            )
        };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe {
                ffi::cudnnDestroyActivationDescriptor(desc);
            }
            return Err(CudaError::DnnError {
                op: "setActivationDesc",
                code: err,
            });
        }
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
        let mut desc: ffi::cudnnOpTensorDescriptor_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreateOpTensorDescriptor(&mut desc) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError {
                op: "createOpTensorDesc",
                code: err,
            });
        }
        let err = unsafe {
            ffi::cudnnSetOpTensorDescriptor(
                desc,
                op,
                comp_type,
                ffi::cudnnNanPropagation_t::CUDNN_NOT_PROPAGATE_NAN,
            )
        };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe {
                ffi::cudnnDestroyOpTensorDescriptor(desc);
            }
            return Err(CudaError::DnnError {
                op: "setOpTensorDesc",
                code: err,
            });
        }
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
