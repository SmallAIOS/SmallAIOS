// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Element-wise GPU operators: `Add`, `Mul`, `Silu`.
//!
//! Implements the NVRTC kernel sources and safe Rust launch wrappers for
//! the first batch of transformer-gpu-kernels-v1 operators:
//!
//! | Operator | F32 kernel   | BF16 kernel   | Broadcasting |
//! |----------|--------------|---------------|--------------|
//! | Add      | `add_f32`    | `add_bf16`    | yes          |
//! | Mul      | `mul_f32`    | `mul_bf16`    | yes          |
//! | Silu     | `silu_f32`   | `silu_bf16`   | no           |
//!
//! All kernels use a grid-stride loop with 256 threads per block (design
//! D4). Broadcasting is implemented on the host: the caller computes the
//! NumPy-style output shape plus per-input strides (0 along broadcast
//! axes) and passes them as three small device-side `int32` buffers. The
//! kernel decomposes a linear output index into per-dimension indices and
//! gathers the inputs with those strides.
//!
//! BF16 variants include `<cuda_bf16.h>` and round-trip through `float`
//! via `__bfloat162float` / `__float2bfloat16` because BF16 lacks native
//! arithmetic operators.

extern crate alloc;

use alloc::format;
use alloc::string::ToString;
use alloc::vec::Vec;

use super::{launch_kernel, Kernel};
use crate::cuda::gpu_executor::DeviceTensor;
use crate::cuda::memory::DeviceBuffer;
use crate::cuda::{CudaError, CudaRuntime};
use crate::tensor::DataType;

// ── Kernel sources ─────────────────────────────────────────────────

/// `add_f32`: `C[i] = A[i] + B[i]` with right-aligned broadcasting.
pub const ADD_F32_SRC: &str = r#"
extern "C" __global__ void add_f32(
    const float* a, const float* b, float* c,
    int ndim,
    const int* shape_c,
    const int* stride_a,
    const int* stride_b,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (; i < total; i += step) {
        int a_off = 0;
        int b_off = 0;
        int remaining = i;
        for (int d = ndim - 1; d >= 0; d--) {
            int dim = shape_c[d];
            int idx = remaining % dim;
            remaining /= dim;
            a_off += idx * stride_a[d];
            b_off += idx * stride_b[d];
        }
        c[i] = a[a_off] + b[b_off];
    }
}
"#;

/// `add_bf16`: BF16 variant. Arithmetic happens in `float` after
/// `__bfloat162float`; the result is converted back via `__float2bfloat16`.
pub const ADD_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
    int ndim,
    const int* shape_c,
    const int* stride_a,
    const int* stride_b,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (; i < total; i += step) {
        int a_off = 0;
        int b_off = 0;
        int remaining = i;
        for (int d = ndim - 1; d >= 0; d--) {
            int dim = shape_c[d];
            int idx = remaining % dim;
            remaining /= dim;
            a_off += idx * stride_a[d];
            b_off += idx * stride_b[d];
        }
        float av = __bfloat162float(a[a_off]);
        float bv = __bfloat162float(b[b_off]);
        c[i] = __float2bfloat16(av + bv);
    }
}
"#;

/// `mul_f32`: `C[i] = A[i] * B[i]` with broadcasting.
pub const MUL_F32_SRC: &str = r#"
extern "C" __global__ void mul_f32(
    const float* a, const float* b, float* c,
    int ndim,
    const int* shape_c,
    const int* stride_a,
    const int* stride_b,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (; i < total; i += step) {
        int a_off = 0;
        int b_off = 0;
        int remaining = i;
        for (int d = ndim - 1; d >= 0; d--) {
            int dim = shape_c[d];
            int idx = remaining % dim;
            remaining /= dim;
            a_off += idx * stride_a[d];
            b_off += idx * stride_b[d];
        }
        c[i] = a[a_off] * b[b_off];
    }
}
"#;

/// `mul_bf16`: BF16 multiply with broadcasting.
pub const MUL_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void mul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, __nv_bfloat16* c,
    int ndim,
    const int* shape_c,
    const int* stride_a,
    const int* stride_b,
    int total
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (; i < total; i += step) {
        int a_off = 0;
        int b_off = 0;
        int remaining = i;
        for (int d = ndim - 1; d >= 0; d--) {
            int dim = shape_c[d];
            int idx = remaining % dim;
            remaining /= dim;
            a_off += idx * stride_a[d];
            b_off += idx * stride_b[d];
        }
        float av = __bfloat162float(a[a_off]);
        float bv = __bfloat162float(b[b_off]);
        c[i] = __float2bfloat16(av * bv);
    }
}
"#;

/// `silu_f32`: `y = x * sigmoid(x)` = `x / (1 + exp(-x))`.
pub const SILU_F32_SRC: &str = r#"
extern "C" __global__ void silu_f32(const float* x, float* y, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (; i < total; i += step) {
        float v = x[i];
        y[i] = v / (1.0f + expf(-v));
    }
}
"#;

/// `silu_bf16`: BF16 Silu. Computes in `float` internally.
pub const SILU_BF16_SRC: &str = r#"
#include <cuda_bf16.h>
extern "C" __global__ void silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* y, int total) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int step = blockDim.x * gridDim.x;
    for (; i < total; i += step) {
        float v = __bfloat162float(x[i]);
        float s = v / (1.0f + expf(-v));
        y[i] = __float2bfloat16(s);
    }
}
"#;

// ── Broadcasting helper ────────────────────────────────────────────

/// NumPy-style broadcast of two shapes, right-aligned.
///
/// Returns `(output_shape, stride_a, stride_b)`:
/// - `output_shape` — per-dim max of the two inputs after left-padding
///   with 1s.
/// - `stride_a` / `stride_b` — row-major strides *from the output's point
///   of view*. A stride of 0 on an axis means "broadcast this axis" —
///   produced when the input has dim 1 but the output has dim > 1, or
///   when the input had fewer dims (left-padded).
///
/// Shapes are i64 to match ONNX; strides are i32 because the kernel
/// consumes `const int*`.
/// Result of broadcasting two shapes: `(output_shape, stride_a, stride_b)`.
type BroadcastLayout = (Vec<i64>, Vec<i32>, Vec<i32>);

fn broadcast_shapes(a_shape: &[i64], b_shape: &[i64]) -> Result<BroadcastLayout, CudaError> {
    let ndim = core::cmp::max(a_shape.len(), b_shape.len());
    let mut out_shape: Vec<i64> = Vec::with_capacity(ndim);
    // Right-align: pad the shorter shape on the left with 1s.
    let a_pad = ndim - a_shape.len();
    let b_pad = ndim - b_shape.len();

    let get_a = |i: usize| -> i64 {
        if i < a_pad {
            1
        } else {
            a_shape[i - a_pad]
        }
    };
    let get_b = |i: usize| -> i64 {
        if i < b_pad {
            1
        } else {
            b_shape[i - b_pad]
        }
    };

    for i in 0..ndim {
        let da = get_a(i);
        let db = get_b(i);
        if da == db || da == 1 || db == 1 {
            out_shape.push(core::cmp::max(da, db));
        } else {
            return Err(CudaError::RuntimeError {
                op: "elementwise: incompatible broadcast shapes",
                code: -1,
            });
        }
    }

    // Compute row-major strides for the original (un-broadcast) input
    // shapes first, then project onto the output's axis layout.
    fn row_major_strides(shape: &[i64]) -> Vec<i64> {
        let n = shape.len();
        let mut s = alloc::vec![1i64; n];
        for i in (0..n.saturating_sub(1)).rev() {
            s[i] = s[i + 1] * shape[i + 1];
        }
        s
    }

    let a_base = row_major_strides(a_shape);
    let b_base = row_major_strides(b_shape);

    let mut stride_a: Vec<i32> = Vec::with_capacity(ndim);
    let mut stride_b: Vec<i32> = Vec::with_capacity(ndim);
    for i in 0..ndim {
        // For input A: if the padded/original dim was 1 (or padded-in),
        // its contribution for this output axis is zero (broadcast);
        // otherwise use its real row-major stride.
        if i < a_pad || a_shape[i - a_pad] == 1 {
            stride_a.push(0);
        } else {
            stride_a.push(a_base[i - a_pad] as i32);
        }
        if i < b_pad || b_shape[i - b_pad] == 1 {
            stride_b.push(0);
        } else {
            stride_b.push(b_base[i - b_pad] as i32);
        }
    }

    Ok((out_shape, stride_a, stride_b))
}

// ── Common launch helper (binary element-wise) ─────────────────────

/// Shared Add/Mul launcher. Dispatches to `f32_name` or `bf16_name`
/// based on the input dtype (which must match on both inputs).
fn elementwise_binary_gpu(
    runtime: &CudaRuntime,
    a: &DeviceTensor,
    b: &DeviceTensor,
    op_label: &'static str,
    f32_name: &'static str,
    bf16_name: &'static str,
) -> Result<DeviceTensor, CudaError> {
    if a.dtype != b.dtype {
        return Err(CudaError::RuntimeError {
            op: "elementwise: input dtype mismatch",
            code: -1,
        });
    }
    let kernel_name = match a.dtype {
        DataType::Float => f32_name,
        DataType::BFloat16 => bf16_name,
        _ => {
            return Err(CudaError::RuntimeError {
                op: "elementwise: unsupported dtype",
                code: -1,
            });
        }
    };

    let (out_shape, stride_a, stride_b) = broadcast_shapes(&a.shape, &b.shape)?;
    let ndim = out_shape.len();
    let total_i64: i64 = out_shape.iter().copied().product::<i64>().max(0);
    let total: usize = total_i64 as usize;

    // Allocate output tensor (uninitialized device memory).
    let out = DeviceTensor::alloc(out_shape.clone(), a.dtype)?;

    // Build device-side shape + stride buffers (int32).
    let shape_i32: Vec<i32> = out_shape.iter().map(|&d| d as i32).collect();
    let shape_bytes = i32_slice_as_bytes(&shape_i32);
    let stride_a_bytes = i32_slice_as_bytes(&stride_a);
    let stride_b_bytes = i32_slice_as_bytes(&stride_b);

    let shape_dev = DeviceBuffer::alloc(shape_bytes.len())?;
    shape_dev.copy_from_host(shape_bytes)?;
    let stride_a_dev = DeviceBuffer::alloc(stride_a_bytes.len())?;
    stride_a_dev.copy_from_host(stride_a_bytes)?;
    let stride_b_dev = DeviceBuffer::alloc(stride_b_bytes.len())?;
    stride_b_dev.copy_from_host(stride_b_bytes)?;

    // Argument pack for cuLaunchKernel. Each slot is the address of a
    // local variable holding the argument value — this is the contract
    // documented on `launch_kernel`.
    let mut a_ptr = a.buffer.as_mut_ptr();
    let mut b_ptr = b.buffer.as_mut_ptr();
    let mut c_ptr = out.buffer.as_mut_ptr();
    let mut ndim_arg: i32 = ndim as i32;
    let mut shape_ptr = shape_dev.as_mut_ptr();
    let mut stride_a_ptr = stride_a_dev.as_mut_ptr();
    let mut stride_b_ptr = stride_b_dev.as_mut_ptr();
    let mut total_arg: i32 = total as i32;

    let args: [*mut core::ffi::c_void; 8] = [
        &mut a_ptr as *mut _ as *mut core::ffi::c_void,
        &mut b_ptr as *mut _ as *mut core::ffi::c_void,
        &mut c_ptr as *mut _ as *mut core::ffi::c_void,
        &mut ndim_arg as *mut _ as *mut core::ffi::c_void,
        &mut shape_ptr as *mut _ as *mut core::ffi::c_void,
        &mut stride_a_ptr as *mut _ as *mut core::ffi::c_void,
        &mut stride_b_ptr as *mut _ as *mut core::ffi::c_void,
        &mut total_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let (grid, block) = launch_dims(total);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: format!("{op_label}: kernel {kernel_name} not registered"),
            cuda_result: -1,
        })??;

    // Keep the aux buffers alive until after the launch returns. The
    // launch is asynchronous, but the driver has copied the argument
    // slots; the device buffers just need to outlive the launch call,
    // which they do because the `?` above returned before we drop.
    drop(shape_dev);
    drop(stride_a_dev);
    drop(stride_b_dev);

    Ok(out)
}

/// Compute grid / block dimensions for a linear element-wise kernel.
///
/// 256 threads per block (design D4), grid capped at 65,535 blocks on
/// the x axis — the grid-stride loop inside the kernel handles any
/// remainder.
fn launch_dims(total: usize) -> ((u32, u32, u32), (u32, u32, u32)) {
    let block = (256u32, 1u32, 1u32);
    let raw_blocks = total.div_ceil(256).max(1);
    let grid_x = core::cmp::min(raw_blocks, 65_535) as u32;
    ((grid_x, 1, 1), block)
}

/// Reinterpret an `&[i32]` as a byte slice for device upload.
///
/// Little-endian hosts (x86_64, aarch64) match NVIDIA's device layout,
/// so this is a plain reinterpret.
fn i32_slice_as_bytes(s: &[i32]) -> &[u8] {
    // Safety: `i32` has no padding and no interior pointers; the output
    // slice lives only as long as `s`, and callers pass it straight to
    // `copy_from_host` which snapshots the bytes.
    unsafe { core::slice::from_raw_parts(s.as_ptr() as *const u8, core::mem::size_of_val(s)) }
}

// ── Public wrappers ────────────────────────────────────────────────

/// GPU implementation of ONNX `Add`.
///
/// Supports NumPy-style broadcasting. Both inputs must share the same
/// dtype (`Float` or `BFloat16`). The caller is responsible for
/// synchronizing the device (typically once per forward pass) before
/// reading the output on the host.
pub fn add_gpu(
    runtime: &CudaRuntime,
    a: &DeviceTensor,
    b: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    elementwise_binary_gpu(runtime, a, b, "add_gpu", "add_f32", "add_bf16")
}

/// GPU implementation of ONNX `Mul`.
///
/// Same constraints as [`add_gpu`].
pub fn mul_gpu(
    runtime: &CudaRuntime,
    a: &DeviceTensor,
    b: &DeviceTensor,
) -> Result<DeviceTensor, CudaError> {
    elementwise_binary_gpu(runtime, a, b, "mul_gpu", "mul_f32", "mul_bf16")
}

/// GPU implementation of the `Silu` / `Swish` activation
/// (`y = x * sigmoid(x)`).
///
/// Element-wise with no broadcasting. Supports `Float` and `BFloat16`.
pub fn silu_gpu(runtime: &CudaRuntime, x: &DeviceTensor) -> Result<DeviceTensor, CudaError> {
    let kernel_name = match x.dtype {
        DataType::Float => "silu_f32",
        DataType::BFloat16 => "silu_bf16",
        _ => {
            return Err(CudaError::RuntimeError {
                op: "silu_gpu: unsupported dtype",
                code: -1,
            });
        }
    };

    let total_i64: i64 = x.shape.iter().copied().product::<i64>().max(0);
    let total: usize = total_i64 as usize;

    let out = DeviceTensor::alloc(x.shape.clone(), x.dtype)?;

    let mut x_ptr = x.buffer.as_mut_ptr();
    let mut y_ptr = out.buffer.as_mut_ptr();
    let mut total_arg: i32 = total as i32;
    let args: [*mut core::ffi::c_void; 3] = [
        &mut x_ptr as *mut _ as *mut core::ffi::c_void,
        &mut y_ptr as *mut _ as *mut core::ffi::c_void,
        &mut total_arg as *mut _ as *mut core::ffi::c_void,
    ];

    let (grid, block) = launch_dims(total);

    runtime
        .with_kernel(kernel_name, |k: &Kernel| {
            launch_kernel(k, grid, block, &args, 0)
        })
        .ok_or_else(|| CudaError::KernelLoadFailed {
            name: ("silu_gpu: kernel ".to_string() + kernel_name + " not registered"),
            cuda_result: -1,
        })??;

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn broadcast_same_shape_has_dense_strides() {
        let (out, sa, sb) = broadcast_shapes(&[2, 3], &[2, 3]).unwrap();
        assert_eq!(out, alloc::vec![2, 3]);
        assert_eq!(sa, alloc::vec![3, 1]);
        assert_eq!(sb, alloc::vec![3, 1]);
    }

    #[test]
    fn broadcast_row_vector_into_matrix() {
        // [1, 4] + [3, 4] = [3, 4]; A broadcasts along axis 0.
        let (out, sa, sb) = broadcast_shapes(&[1, 4], &[3, 4]).unwrap();
        assert_eq!(out, alloc::vec![3, 4]);
        assert_eq!(sa, alloc::vec![0, 1]);
        assert_eq!(sb, alloc::vec![4, 1]);
    }

    #[test]
    fn broadcast_rank_padding() {
        // [4] + [3, 4] = [3, 4]; A's missing leading axis becomes stride 0.
        let (out, sa, sb) = broadcast_shapes(&[4], &[3, 4]).unwrap();
        assert_eq!(out, alloc::vec![3, 4]);
        assert_eq!(sa, alloc::vec![0, 1]);
        assert_eq!(sb, alloc::vec![4, 1]);
    }

    #[test]
    fn broadcast_rejects_incompatible() {
        assert!(broadcast_shapes(&[3, 4], &[5, 4]).is_err());
    }
}
