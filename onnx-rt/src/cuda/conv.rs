// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU Conv2d dispatch via cuDNN.
//!
//! Wraps `cudnnConvolutionForward` with RAII descriptor management.

extern crate alloc;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use super::ffi;
use super::memory::DeviceBuffer;
use super::{CudaError, CudaRuntime};
use crate::tensor::{DataType, Tensor, TensorShape};

/// RAII wrapper for cuDNN tensor descriptor.
struct TensorDesc {
    desc: ffi::cudnnTensorDescriptor_t,
}

impl TensorDesc {
    fn new_4d(n: i32, c: i32, h: i32, w: i32) -> Result<Self, CudaError> {
        let mut desc: ffi::cudnnTensorDescriptor_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreateTensorDescriptor(&mut desc) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError { op: "createTensorDesc", code: err });
        }
        let err = unsafe {
            ffi::cudnnSetTensor4dDescriptor(
                desc,
                ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
                n, c, h, w,
            )
        };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnDestroyTensorDescriptor(desc); }
            return Err(CudaError::DnnError { op: "setTensor4dDesc", code: err });
        }
        Ok(Self { desc })
    }
}

impl Drop for TensorDesc {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyTensorDescriptor(self.desc); }
    }
}

/// RAII wrapper for cuDNN filter descriptor.
struct FilterDesc {
    desc: ffi::cudnnFilterDescriptor_t,
}

impl FilterDesc {
    fn new_4d(k: i32, c: i32, h: i32, w: i32) -> Result<Self, CudaError> {
        let mut desc: ffi::cudnnFilterDescriptor_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreateFilterDescriptor(&mut desc) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError { op: "createFilterDesc", code: err });
        }
        let err = unsafe {
            ffi::cudnnSetFilter4dDescriptor(
                desc,
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
                ffi::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                k, c, h, w,
            )
        };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnDestroyFilterDescriptor(desc); }
            return Err(CudaError::DnnError { op: "setFilter4dDesc", code: err });
        }
        Ok(Self { desc })
    }
}

impl Drop for FilterDesc {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyFilterDescriptor(self.desc); }
    }
}

/// RAII wrapper for cuDNN convolution descriptor.
struct ConvDesc {
    desc: ffi::cudnnConvolutionDescriptor_t,
}

impl ConvDesc {
    fn new_2d(
        pad_h: i32, pad_w: i32,
        stride_h: i32, stride_w: i32,
        dilation_h: i32, dilation_w: i32,
    ) -> Result<Self, CudaError> {
        let mut desc: ffi::cudnnConvolutionDescriptor_t = core::ptr::null_mut();
        let err = unsafe { ffi::cudnnCreateConvolutionDescriptor(&mut desc) };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            return Err(CudaError::DnnError { op: "createConvDesc", code: err });
        }
        let err = unsafe {
            ffi::cudnnSetConvolution2dDescriptor(
                desc,
                pad_h, pad_w,
                stride_h, stride_w,
                dilation_h, dilation_w,
                ffi::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
            )
        };
        if err != ffi::CUDNN_STATUS_SUCCESS {
            unsafe { ffi::cudnnDestroyConvolutionDescriptor(desc); }
            return Err(CudaError::DnnError { op: "setConv2dDesc", code: err });
        }
        Ok(Self { desc })
    }
}

impl Drop for ConvDesc {
    fn drop(&mut self) {
        unsafe { ffi::cudnnDestroyConvolutionDescriptor(self.desc); }
    }
}

/// Execute a 2D convolution on GPU via cuDNN.
///
/// Input format: NCHW (batch, channels, height, width).
/// Weight format: [out_channels, in_channels, kH, kW].
///
/// Returns None if the input shapes don't match a supported 4D conv pattern.
pub fn gpu_conv2d(
    runtime: &CudaRuntime,
    input: &Tensor,      // [N, C, H, W]
    weight: &Tensor,     // [K, C, kH, kW]
    bias: Option<&Tensor>, // [K]
    pads: &[i32],        // [pad_top, pad_left, pad_bottom, pad_right] or [pad_h, pad_w]
    strides: &[i32],     // [stride_h, stride_w]
    dilations: &[i32],   // [dilation_h, dilation_w]
) -> Result<Option<Tensor>, CudaError> {
    let x_dims = &input.shape.dims;
    let w_dims = &weight.shape.dims;

    if x_dims.len() != 4 || w_dims.len() != 4 {
        return Ok(None); // Not a 4D conv, fall back to CPU
    }

    let n = x_dims[0] as i32;
    let c_in = x_dims[1] as i32;
    let h_in = x_dims[2] as i32;
    let w_in = x_dims[3] as i32;

    let k = w_dims[0] as i32;       // output channels
    let _c_w = w_dims[1] as i32;    // should equal c_in (or c_in/group)
    let kh = w_dims[2] as i32;
    let kw = w_dims[3] as i32;

    // Parse padding (support both 2-element and 4-element forms).
    let (pad_h, pad_w) = if pads.len() >= 4 {
        (pads[0], pads[1])
    } else if pads.len() >= 2 {
        (pads[0], pads[1])
    } else {
        (0, 0)
    };

    let stride_h = strides.first().copied().unwrap_or(1);
    let stride_w = strides.get(1).copied().unwrap_or(1);
    let dil_h = dilations.first().copied().unwrap_or(1);
    let dil_w = dilations.get(1).copied().unwrap_or(1);

    // Compute output dimensions.
    let h_out = (h_in + 2 * pad_h - dil_h * (kh - 1) - 1) / stride_h + 1;
    let w_out = (w_in + 2 * pad_w - dil_w * (kw - 1) - 1) / stride_w + 1;

    if h_out <= 0 || w_out <= 0 {
        return Ok(None);
    }

    // Create cuDNN descriptors.
    let x_desc = TensorDesc::new_4d(n, c_in, h_in, w_in)?;
    let w_desc = FilterDesc::new_4d(k, c_in, kh, kw)?;
    let conv_desc = ConvDesc::new_2d(pad_h, pad_w, stride_h, stride_w, dil_h, dil_w)?;
    let y_desc = TensorDesc::new_4d(n, k, h_out, w_out)?;

    // Allocate device buffers.
    let x_bytes = input.raw_data.len();
    let w_bytes = weight.raw_data.len();
    let y_bytes = (n * k * h_out * w_out) as usize * 4; // f32

    let x_buf = DeviceBuffer::alloc(x_bytes)?;
    let w_buf = DeviceBuffer::alloc(w_bytes)?;
    let y_buf = DeviceBuffer::alloc(y_bytes)?;

    x_buf.copy_from_host(&input.raw_data)?;
    w_buf.copy_from_host(&weight.raw_data)?;
    y_buf.copy_from_host(&vec![0u8; y_bytes])?;

    // Run convolution.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    let err = unsafe {
        ffi::cudnnConvolutionForward(
            runtime.cudnn.raw(),
            &alpha as *const f32 as *const core::ffi::c_void,
            x_desc.desc,
            x_buf.as_ptr(),
            w_desc.desc,
            w_buf.as_ptr(),
            conv_desc.desc,
            ffi::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM,
            core::ptr::null_mut(), // no workspace
            0,                     // workspace size = 0
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError { op: "cudnnConvolutionForward", code: err });
    }

    super::synchronize()?;

    // Transfer result back.
    let mut result_bytes = vec![0u8; y_bytes];
    y_buf.copy_to_host(&mut result_bytes)?;

    // Add bias if present.
    if let Some(bias_tensor) = bias {
        if bias_tensor.raw_data.len() == (k as usize) * 4 {
            let bias_f32: Vec<f32> = bias_tensor.raw_data.chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            // Add bias per output channel: y[n,c,h,w] += bias[c]
            let spatial = (h_out * w_out) as usize;
            for b_idx in 0..n as usize {
                for c_idx in 0..k as usize {
                    let base = (b_idx * k as usize + c_idx) * spatial;
                    for s in 0..spatial {
                        let offset = (base + s) * 4;
                        let val = f32::from_le_bytes([
                            result_bytes[offset], result_bytes[offset + 1],
                            result_bytes[offset + 2], result_bytes[offset + 3],
                        ]);
                        let biased = val + bias_f32[c_idx];
                        result_bytes[offset..offset + 4].copy_from_slice(&biased.to_le_bytes());
                    }
                }
            }
        }
    }

    Ok(Some(Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![n as i64, k as i64, h_out as i64, w_out as i64]),
        name: String::new(),
        raw_data: result_bytes,
    }))
}
