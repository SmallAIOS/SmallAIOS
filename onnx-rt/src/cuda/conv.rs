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
use crate::tensor::{f32_to_bf16, DataType, Tensor, TensorShape};

/// RAII wrapper for cuDNN tensor descriptor.
struct TensorDesc {
    desc: ffi::cudnnTensorDescriptor_t,
}

impl TensorDesc {
    fn new_4d_dtype(
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

/// RAII wrapper for cuDNN filter descriptor.
struct FilterDesc {
    desc: ffi::cudnnFilterDescriptor_t,
}

impl FilterDesc {
    fn new_4d_dtype(
        k: i32,
        c: i32,
        h: i32,
        w: i32,
        dtype: ffi::cudnnDataType_t,
    ) -> Result<Self, CudaError> {
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
        Ok(Self { desc })
    }
}

impl Drop for FilterDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyFilterDescriptor(self.desc);
        }
    }
}

/// RAII wrapper for cuDNN convolution descriptor.
struct ConvDesc {
    desc: ffi::cudnnConvolutionDescriptor_t,
}

impl ConvDesc {
    #[allow(clippy::too_many_arguments)]
    fn new_2d(
        pad_h: i32,
        pad_w: i32,
        stride_h: i32,
        stride_w: i32,
        dilation_h: i32,
        dilation_w: i32,
        group: i32,
    ) -> Result<Self, CudaError> {
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
                dilation_h,
                dilation_w,
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
        // cuDNN docs require setting group count after 2d-descriptor
        // setup and before algorithm selection. Skip for group=1 to
        // preserve byte-identical behavior on the default path.
        if group > 1 {
            let err = unsafe { ffi::cudnnSetConvolutionGroupCount(desc, group) };
            if err != ffi::CUDNN_STATUS_SUCCESS {
                unsafe {
                    ffi::cudnnDestroyConvolutionDescriptor(desc);
                }
                return Err(CudaError::DnnError {
                    op: "setConvGroupCount",
                    code: err,
                });
            }
        }
        Ok(Self { desc })
    }
}

impl Drop for ConvDesc {
    fn drop(&mut self) {
        unsafe {
            ffi::cudnnDestroyConvolutionDescriptor(self.desc);
        }
    }
}

/// Selected cuDNN dtype configuration for a 2-D convolution.
///
/// Encapsulates the BF16-vs-F32 fork so [`gpu_conv2d`] does not have to
/// re-branch on `is_bf16` for each downstream concern (descriptor dtype,
/// element size, output tensor dtype, bias path).
struct ConvDtype {
    is_bf16: bool,
    dnn_dtype: ffi::cudnnDataType_t,
    elem_size: usize,
    out_data_type: DataType,
}

impl ConvDtype {
    /// Choose the cuDNN dtype config for the given input + weight tensors.
    ///
    /// Returns `None` if the dtype combination is not supported (mixed
    /// dtypes, or anything other than `Float`/`BFloat16`).
    fn pick(input: &Tensor, weight: &Tensor) -> Option<Self> {
        let is_bf16 =
            input.data_type == DataType::BFloat16 && weight.data_type == DataType::BFloat16;
        if !is_bf16 && (input.data_type != DataType::Float || weight.data_type != DataType::Float) {
            return None;
        }
        let (dnn_dtype, elem_size, out_data_type) = if is_bf16 {
            (
                ffi::cudnnDataType_t::CUDNN_DATA_BFLOAT16,
                2usize,
                DataType::BFloat16,
            )
        } else {
            (
                ffi::cudnnDataType_t::CUDNN_DATA_FLOAT,
                4usize,
                DataType::Float,
            )
        };
        Some(Self {
            is_bf16,
            dnn_dtype,
            elem_size,
            out_data_type,
        })
    }
}

/// Validated shape parameters for a 2-D convolution.
struct ConvParams {
    n: i32,
    c_in: i32,
    h_in: i32,
    w_in: i32,
    k: i32,
    c_w: i32,
    kh: i32,
    kw: i32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    h_out: i32,
    w_out: i32,
}

/// Parse a `Conv` attribute pair into `(h, w)` with `default` if missing.
fn pair_or(values: &[i32], default: i32) -> (i32, i32) {
    let h = values.first().copied().unwrap_or(default);
    let w = values.get(1).copied().unwrap_or(default);
    (h, w)
}

impl ConvParams {
    /// Validate shapes and compute output dimensions for a 2-D conv.
    ///
    /// Returns `None` if the configuration is unsupported (non-4D inputs,
    /// invalid group, non-positive output). The cuDNN dispatcher falls
    /// back to CPU on `None`.
    fn build(
        input: &Tensor,
        weight: &Tensor,
        pads: &[i32],
        strides: &[i32],
        dilations: &[i32],
        group: i32,
    ) -> Option<Self> {
        let x_dims = &input.shape.dims;
        let w_dims = &weight.shape.dims;
        if x_dims.len() != 4 || w_dims.len() != 4 || group < 1 {
            return None;
        }

        let n = x_dims[0] as i32;
        let c_in = x_dims[1] as i32;
        let h_in = x_dims[2] as i32;
        let w_in = x_dims[3] as i32;

        let k = w_dims[0] as i32;
        let c_w = w_dims[1] as i32;
        let kh = w_dims[2] as i32;
        let kw = w_dims[3] as i32;

        if c_in != c_w * group || k % group != 0 {
            return None;
        }

        // Padding supports both 2-element [h, w] and 4-element [t, l, b, r] forms.
        let (pad_h, pad_w) = if pads.len() >= 2 {
            (pads[0], pads[1])
        } else {
            (0, 0)
        };
        let (stride_h, stride_w) = pair_or(strides, 1);
        let (dil_h, dil_w) = pair_or(dilations, 1);

        let h_out = (h_in + 2 * pad_h - dil_h * (kh - 1) - 1) / stride_h + 1;
        let w_out = (w_in + 2 * pad_w - dil_w * (kw - 1) - 1) / stride_w + 1;
        if h_out <= 0 || w_out <= 0 {
            return None;
        }
        Some(Self {
            n,
            c_in,
            h_in,
            w_in,
            k,
            c_w,
            kh,
            kw,
            pad_h,
            pad_w,
            stride_h,
            stride_w,
            dil_h,
            dil_w,
            h_out,
            w_out,
        })
    }
}

/// Decode the bias tensor (F32 or BF16) to an F32 vector of length `k`.
///
/// Returns an empty vector if the bias dtype or byte length does not
/// match what we expect — the caller treats that as "skip the bias add",
/// matching the original byte-for-byte behaviour.
fn decode_bias_to_f32(bias: &Tensor, k: usize) -> Vec<f32> {
    match bias.data_type {
        DataType::Float if bias.raw_data.len() == k * 4 => bias
            .raw_data
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect(),
        DataType::BFloat16 if bias.raw_data.len() == k * 2 => {
            crate::tensor::bf16_to_f32(&bias.raw_data)
        }
        _ => Vec::new(),
    }
}

/// In-place add `bias` to one F32 channel slab inside `result_bytes`.
fn add_bias_channel_f32(result_bytes: &mut [u8], base: usize, spatial: usize, add: f32) {
    for s in 0..spatial {
        let offset = (base + s) * 4;
        let val = f32::from_le_bytes([
            result_bytes[offset],
            result_bytes[offset + 1],
            result_bytes[offset + 2],
            result_bytes[offset + 3],
        ]);
        let biased = val + add;
        result_bytes[offset..offset + 4].copy_from_slice(&biased.to_le_bytes());
    }
}

/// In-place add `bias` to one BF16 channel slab inside `result_bytes`.
fn add_bias_channel_bf16(result_bytes: &mut [u8], base: usize, spatial: usize, add: f32) {
    for s in 0..spatial {
        let offset = (base + s) * 2;
        let lo = result_bytes[offset] as u32;
        let hi = result_bytes[offset + 1] as u32;
        let bits = ((hi << 8) | lo) << 16;
        let val = f32::from_bits(bits) + add;
        let enc = f32_to_bf16(&[val]);
        result_bytes[offset] = enc[0];
        result_bytes[offset + 1] = enc[1];
    }
}

/// Apply per-channel bias on the host-side conv output.
///
/// Skips silently if the bias element count or dtype does not match the
/// expected `[k]` shape — the original code did the same.
fn apply_bias(result_bytes: &mut [u8], bias: &Tensor, params: &ConvParams, is_bf16: bool) {
    let k = params.k as usize;
    if bias.shape.total_elements() != k {
        return;
    }
    let bias_f32 = decode_bias_to_f32(bias, k);
    if bias_f32.len() != k {
        return;
    }
    let spatial = (params.h_out * params.w_out) as usize;
    for b_idx in 0..params.n as usize {
        for (c_idx, &add) in bias_f32.iter().enumerate().take(k) {
            let base = (b_idx * k + c_idx) * spatial;
            if is_bf16 {
                add_bias_channel_bf16(result_bytes, base, spatial, add);
            } else {
                add_bias_channel_f32(result_bytes, base, spatial, add);
            }
        }
    }
}

/// Run `cudnnConvolutionForward` on the prepared descriptors / buffers.
#[allow(clippy::too_many_arguments)]
fn run_conv_forward(
    runtime: &CudaRuntime,
    x_desc: &TensorDesc,
    w_desc: &FilterDesc,
    conv_desc: &ConvDesc,
    y_desc: &TensorDesc,
    x_buf: &DeviceBuffer,
    w_buf: &DeviceBuffer,
    y_buf: &DeviceBuffer,
) -> Result<(), CudaError> {
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
            core::ptr::null_mut(),
            0,
            &beta as *const f32 as *const core::ffi::c_void,
            y_desc.desc,
            y_buf.as_mut_ptr(),
        )
    };
    if err != ffi::CUDNN_STATUS_SUCCESS {
        return Err(CudaError::DnnError {
            op: "cudnnConvolutionForward",
            code: err,
        });
    }
    Ok(())
}

/// Execute a 2D convolution on GPU via cuDNN.
///
/// Input format: NCHW (batch, channels, height, width).
/// Weight format: [out_channels, in_channels, kH, kW].
///
/// Returns None if the input shapes don't match a supported 4D conv pattern.
#[allow(clippy::too_many_arguments)]
pub fn gpu_conv2d(
    runtime: &CudaRuntime,
    input: &Tensor,        // [N, C, H, W]
    weight: &Tensor,       // [K, C/group, kH, kW]
    bias: Option<&Tensor>, // [K]
    pads: &[i32],          // [pad_top, pad_left, pad_bottom, pad_right] or [pad_h, pad_w]
    strides: &[i32],       // [stride_h, stride_w]
    dilations: &[i32],     // [dilation_h, dilation_w]
    group: i32,
) -> Result<Option<Tensor>, CudaError> {
    // Pick cuDNN I/O dtype. BF16 inputs stay in BF16 end-to-end (f32
    // accumulation in the convolution descriptor); anything else uses the
    // f32 path. Mixed input/weight dtypes fall back to CPU.
    let Some(dtype) = ConvDtype::pick(input, weight) else {
        return Ok(None);
    };

    // Validate shapes and compute output dimensions.
    let Some(params) = ConvParams::build(input, weight, pads, strides, dilations, group) else {
        return Ok(None);
    };

    // Create cuDNN descriptors. Input/filter/output use dnn_dtype (BF16 or
    // f32); the convolution descriptor itself uses f32 accumulation regardless
    // so BF16 inputs get the same accuracy guarantees as BF16 cuBLAS GEMM.
    let x_desc = TensorDesc::new_4d_dtype(
        params.n,
        params.c_in,
        params.h_in,
        params.w_in,
        dtype.dnn_dtype,
    )?;
    // Filter descriptor uses per-group input channels (c_w = c_in / group).
    let w_desc =
        FilterDesc::new_4d_dtype(params.k, params.c_w, params.kh, params.kw, dtype.dnn_dtype)?;
    let conv_desc = ConvDesc::new_2d(
        params.pad_h,
        params.pad_w,
        params.stride_h,
        params.stride_w,
        params.dil_h,
        params.dil_w,
        group,
    )?;
    let y_desc = TensorDesc::new_4d_dtype(
        params.n,
        params.k,
        params.h_out,
        params.w_out,
        dtype.dnn_dtype,
    )?;

    // Allocate device buffers.
    let x_bytes = input.raw_data.len();
    let w_bytes = weight.raw_data.len();
    let y_bytes = (params.n * params.k * params.h_out * params.w_out) as usize * dtype.elem_size;

    let x_buf = DeviceBuffer::alloc(x_bytes)?;
    let w_buf = DeviceBuffer::alloc(w_bytes)?;
    let y_buf = DeviceBuffer::alloc(y_bytes)?;

    x_buf.copy_from_host(&input.raw_data)?;
    w_buf.copy_from_host(&weight.raw_data)?;
    y_buf.copy_from_host(&vec![0u8; y_bytes])?;

    run_conv_forward(
        runtime, &x_desc, &w_desc, &conv_desc, &y_desc, &x_buf, &w_buf, &y_buf,
    )?;

    super::synchronize()?;

    // Transfer result back.
    let mut result_bytes = vec![0u8; y_bytes];
    y_buf.copy_to_host(&mut result_bytes)?;

    // Add bias if present. Host-side per-channel add: y[n,c,h,w] += bias[c].
    // For BF16 output we round-trip through f32 for each element — BF16 is
    // not directly addable, and the intermediate accumulation is done in f32
    // anyway. For f32 output this matches the original behaviour exactly.
    if let Some(bias_tensor) = bias {
        apply_bias(&mut result_bytes, bias_tensor, &params, dtype.is_bf16);
    }

    Ok(Some(Tensor {
        data_type: dtype.out_data_type,
        shape: TensorShape::new(vec![
            params.n as i64,
            params.k as i64,
            params.h_out as i64,
            params.w_out as i64,
        ]),
        name: String::new(),
        raw_data: result_bytes,
    }))
}
