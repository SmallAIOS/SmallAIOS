// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metal GPU dispatch for ONNX operator execution.
//!
//! [`MetalDispatcher`] maps ONNX operator calls to Metal kernel launches
//! using the pre-built MSL shaders from [`smallaios_arch_apple::shaders`].
//! Input tensors are copied to GPU buffers (with caching), the appropriate
//! kernel is launched, and the result is copied back to a host [`Tensor`].
//!
//! If an operator is not supported on GPU, `try_execute` returns `Ok(None)`
//! and the caller falls back to the CPU implementation.

use alloc::format;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;

use smallaios_arch_apple::{MetalError, MetalProvider, MetalTensorCache};
use smallaios_compute::ComputeProvider;

use crate::onnx_types::AttributeProto;
use crate::operators::OpError;
use crate::tensor::{DataType, Tensor, TensorShape};

// ---------------------------------------------------------------------------
// MetalDispatcher
// ---------------------------------------------------------------------------

/// Dispatches ONNX operators to Metal GPU kernels.
///
/// Wraps a [`MetalProvider`] and a [`MetalTensorCache`] to manage device
/// memory and kernel compilation. The `try_execute` method is the main
/// entry point: it checks if the operator is GPU-supported, transfers
/// input tensors, launches the kernel, and returns the output tensor(s).
pub struct MetalDispatcher {
    provider: MetalProvider,
    tensor_cache: MetalTensorCache,
}

impl MetalDispatcher {
    /// Creates a new Metal dispatcher, initializing the Metal device.
    ///
    /// Returns an error if no Metal-capable device is available.
    pub fn new() -> Result<Self, MetalError> {
        let provider = MetalProvider::new()?;
        Ok(Self {
            provider,
            tensor_cache: MetalTensorCache::new(),
        })
    }

    /// Attempts to execute an ONNX operator on the Metal GPU.
    ///
    /// Returns `Ok(Some(outputs))` if the operator was executed on GPU,
    /// `Ok(None)` if the operator is not supported (caller should fall back
    /// to CPU), or `Err` if a GPU error occurred.
    pub fn try_execute(
        &mut self,
        op_type: &str,
        inputs: &[Option<&Tensor>],
        attrs: &[AttributeProto],
    ) -> Result<Option<Vec<Tensor>>, OpError> {
        if !self.provider.supports_op(op_type) {
            return Ok(None);
        }

        let result = match op_type {
            "Add" | "Sub" | "Mul" | "Div" => self.dispatch_elementwise_binary(op_type, inputs),
            "Relu" | "Sigmoid" | "Tanh" => self.dispatch_elementwise_unary(op_type, inputs),
            "MatMul" => self.dispatch_matmul(inputs),
            "Softmax" => self.dispatch_softmax(inputs, attrs),
            "Conv" => self.dispatch_conv2d(inputs, attrs),
            "Gemm" => self.dispatch_matmul(inputs), // Gemm uses same kernel path
            _ => return Ok(None),
        };

        result.map(Some)
    }

    /// Returns `true` if the given operator can be executed on Metal.
    pub fn supports_op(&self, op_type: &str) -> bool {
        self.provider.supports_op(op_type)
    }

    /// Clears the tensor cache, releasing all cached GPU buffers.
    pub fn clear_cache(&mut self) {
        self.tensor_cache.clear();
    }
}

// ---------------------------------------------------------------------------
// Helper: convert MetalError to OpError
// ---------------------------------------------------------------------------

fn metal_err(e: MetalError) -> OpError {
    OpError::InternalError(format!("Metal GPU error: {}", e.0))
}

// ---------------------------------------------------------------------------
// Element-wise binary ops (Add, Sub, Mul, Div)
// ---------------------------------------------------------------------------

impl MetalDispatcher {
    fn dispatch_elementwise_binary(
        &mut self,
        op_type: &str,
        inputs: &[Option<&Tensor>],
    ) -> Result<Vec<Tensor>, OpError> {
        let a = inputs
            .first()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing input A")))?;
        let b = inputs
            .get(1)
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing input B")))?;

        if a.data_type != DataType::Float || b.data_type != DataType::Float {
            return Err(OpError::InternalError(String::from(
                "Metal elementwise ops require Float tensors",
            )));
        }

        let n = a.shape.total_elements();
        if n != b.shape.total_elements() {
            return Err(OpError::ShapeMismatch(format!(
                "elementwise shape mismatch: {} vs {}",
                n,
                b.shape.total_elements()
            )));
        }

        let byte_size = n * core::mem::size_of::<f32>();

        // Upload inputs
        let buf_a = self
            .tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_a", &a.raw_data)
            .map_err(metal_err)?;
        // We need the raw pointer info before the second mutable borrow, so
        // we work with the cache in two steps.
        let _a_size = buf_a.size();

        let buf_b = self
            .tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_b", &b.raw_data)
            .map_err(metal_err)?;
        let _b_size = buf_b.size();

        // Allocate output buffer
        let _out_buf = self
            .tensor_cache
            .get_or_create(&mut self.provider, "__gpu_out", byte_size)
            .map_err(metal_err)?;

        // Select kernel source and function name
        let (kernel_name, kernel_source) = match op_type {
            "Add" => (
                "elementwise_add",
                smallaios_arch_apple::shaders::ELEMENTWISE_ADD,
            ),
            "Sub" => (
                "elementwise_sub",
                smallaios_arch_apple::shaders::ELEMENTWISE_SUB,
            ),
            "Mul" => (
                "elementwise_mul",
                smallaios_arch_apple::shaders::ELEMENTWISE_MUL,
            ),
            "Div" => (
                "elementwise_div",
                smallaios_arch_apple::shaders::ELEMENTWISE_DIV,
            ),
            _ => unreachable!(),
        };

        // Compile kernel
        let kernel = self
            .provider
            .load_kernel(kernel_name, kernel_source.as_bytes())
            .map_err(metal_err)?;

        // Get buffer references for launch
        let buf_a_ref = self.tensor_cache.get("__gpu_a").unwrap();
        let buf_b_ref = self.tensor_cache.get("__gpu_b").unwrap();
        let buf_out_ref = self.tensor_cache.get("__gpu_out").unwrap();

        // Launch: grid = N elements, threadgroup = min(256, N)
        let tg_size = (n as u32).min(256);
        self.provider
            .launch(
                &kernel,
                [n as u32, 1, 1],
                [tg_size, 1, 1],
                &[buf_a_ref, buf_b_ref, buf_out_ref],
            )
            .map_err(metal_err)?;

        self.provider.synchronize().map_err(metal_err)?;

        // Read back result
        let buf_out_ref = self.tensor_cache.get("__gpu_out").unwrap();
        let raw_data = MetalTensorCache::copy_from_device(&self.provider, buf_out_ref, byte_size)
            .map_err(metal_err)?;

        let mut output = Tensor::new(DataType::Float, a.shape.clone(), String::new());
        output.raw_data = raw_data;

        Ok(vec![output])
    }
}

// ---------------------------------------------------------------------------
// Element-wise unary ops (Relu, Sigmoid, Tanh)
// ---------------------------------------------------------------------------

impl MetalDispatcher {
    fn dispatch_elementwise_unary(
        &mut self,
        op_type: &str,
        inputs: &[Option<&Tensor>],
    ) -> Result<Vec<Tensor>, OpError> {
        let input = inputs
            .first()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing input")))?;

        if input.data_type != DataType::Float {
            return Err(OpError::InternalError(String::from(
                "Metal unary ops require Float tensors",
            )));
        }

        let n = input.shape.total_elements();
        let byte_size = n * core::mem::size_of::<f32>();

        // Upload input
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_in", &input.raw_data)
            .map_err(metal_err)?;

        // Allocate output
        self.tensor_cache
            .get_or_create(&mut self.provider, "__gpu_out", byte_size)
            .map_err(metal_err)?;

        let (kernel_name, kernel_source) = match op_type {
            "Relu" => (
                "elementwise_relu",
                smallaios_arch_apple::shaders::ELEMENTWISE_RELU,
            ),
            "Sigmoid" => (
                "elementwise_sigmoid",
                smallaios_arch_apple::shaders::ELEMENTWISE_SIGMOID,
            ),
            "Tanh" => (
                "elementwise_tanh",
                smallaios_arch_apple::shaders::ELEMENTWISE_TANH,
            ),
            _ => unreachable!(),
        };

        let kernel = self
            .provider
            .load_kernel(kernel_name, kernel_source.as_bytes())
            .map_err(metal_err)?;

        let buf_in_ref = self.tensor_cache.get("__gpu_in").unwrap();
        let buf_out_ref = self.tensor_cache.get("__gpu_out").unwrap();

        let tg_size = (n as u32).min(256);
        self.provider
            .launch(
                &kernel,
                [n as u32, 1, 1],
                [tg_size, 1, 1],
                &[buf_in_ref, buf_out_ref],
            )
            .map_err(metal_err)?;

        self.provider.synchronize().map_err(metal_err)?;

        let buf_out_ref = self.tensor_cache.get("__gpu_out").unwrap();
        let raw_data = MetalTensorCache::copy_from_device(&self.provider, buf_out_ref, byte_size)
            .map_err(metal_err)?;

        let mut output = Tensor::new(DataType::Float, input.shape.clone(), String::new());
        output.raw_data = raw_data;

        Ok(vec![output])
    }
}

// ---------------------------------------------------------------------------
// MatMul
// ---------------------------------------------------------------------------

impl MetalDispatcher {
    fn dispatch_matmul(&mut self, inputs: &[Option<&Tensor>]) -> Result<Vec<Tensor>, OpError> {
        let a = inputs
            .first()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing input A")))?;
        let b = inputs
            .get(1)
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing input B")))?;

        if a.data_type != DataType::Float || b.data_type != DataType::Float {
            return Err(OpError::InternalError(String::from(
                "Metal MatMul requires Float tensors",
            )));
        }

        // Extract dimensions: A is [M, K], B is [K, N]
        let a_dims = &a.shape.dims;
        let b_dims = &b.shape.dims;
        if a_dims.len() < 2 || b_dims.len() < 2 {
            return Err(OpError::ShapeMismatch(String::from(
                "MatMul requires at least 2D tensors",
            )));
        }

        let m = a_dims[a_dims.len() - 2] as u32;
        let k = a_dims[a_dims.len() - 1] as u32;
        let n = b_dims[b_dims.len() - 1] as u32;

        if k != b_dims[b_dims.len() - 2] as u32 {
            return Err(OpError::ShapeMismatch(format!(
                "MatMul inner dimensions mismatch: K={} vs {}",
                k,
                b_dims[b_dims.len() - 2]
            )));
        }

        let out_elements = (m as usize) * (n as usize);
        let out_byte_size = out_elements * core::mem::size_of::<f32>();

        // Upload inputs
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_a", &a.raw_data)
            .map_err(metal_err)?;
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_b", &b.raw_data)
            .map_err(metal_err)?;

        // Allocate output buffer
        self.tensor_cache
            .get_or_create(&mut self.provider, "__gpu_out", out_byte_size)
            .map_err(metal_err)?;

        // Create dims buffer: [M, K, N]
        let dims_data: Vec<u8> = [m, k, n].iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_dims", &dims_data)
            .map_err(metal_err)?;

        // Select kernel: use tiled version for larger matrices, naive for small ones.
        // The tiled kernel requires threadgroup size of 16x16 to fill shared memory.
        let use_tiled = m >= 16 && n >= 16;
        let (kernel_name, kernel_source) = if use_tiled {
            ("matmul_tiled", smallaios_arch_apple::shaders::MATMUL_TILED)
        } else {
            ("matmul", smallaios_arch_apple::shaders::MATMUL)
        };

        let kernel = self
            .provider
            .load_kernel(kernel_name, kernel_source.as_bytes())
            .map_err(metal_err)?;

        let buf_a = self.tensor_cache.get("__gpu_a").unwrap();
        let buf_b = self.tensor_cache.get("__gpu_b").unwrap();
        let buf_out = self.tensor_cache.get("__gpu_out").unwrap();
        let buf_dims = self.tensor_cache.get("__gpu_dims").unwrap();

        // Grid = (N, M) — each thread computes one output element.
        // For the tiled kernel, round up to multiples of 16.
        let (grid_x, grid_y, tg_x, tg_y) = if use_tiled {
            let gx = n.div_ceil(16) * 16;
            let gy = m.div_ceil(16) * 16;
            (gx, gy, 16u32, 16u32)
        } else {
            (n, m, n.min(16), m.min(16))
        };

        self.provider
            .launch(
                &kernel,
                [grid_x, grid_y, 1],
                [tg_x, tg_y, 1],
                &[buf_a, buf_b, buf_out, buf_dims],
            )
            .map_err(metal_err)?;

        self.provider.synchronize().map_err(metal_err)?;

        let buf_out = self.tensor_cache.get("__gpu_out").unwrap();
        let raw_data = MetalTensorCache::copy_from_device(&self.provider, buf_out, out_byte_size)
            .map_err(metal_err)?;

        let out_shape = TensorShape::new(vec![m as i64, n as i64]);
        let mut output = Tensor::new(DataType::Float, out_shape, String::new());
        output.raw_data = raw_data;

        Ok(vec![output])
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

impl MetalDispatcher {
    fn dispatch_softmax(
        &mut self,
        inputs: &[Option<&Tensor>],
        _attrs: &[AttributeProto],
    ) -> Result<Vec<Tensor>, OpError> {
        let input = inputs
            .first()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing input")))?;

        if input.data_type != DataType::Float {
            return Err(OpError::InternalError(String::from(
                "Metal Softmax requires Float tensors",
            )));
        }

        let total = input.shape.total_elements();
        let byte_size = total * core::mem::size_of::<f32>();

        // For softmax, N = last dimension size (row length)
        let n = if input.shape.dims.is_empty() {
            1u32
        } else {
            *input.shape.dims.last().unwrap() as u32
        };

        // Upload input
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_in", &input.raw_data)
            .map_err(metal_err)?;

        // Allocate output
        self.tensor_cache
            .get_or_create(&mut self.provider, "__gpu_out", byte_size)
            .map_err(metal_err)?;

        // Dims buffer: [N]
        let dims_data: Vec<u8> = n.to_ne_bytes().to_vec();
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_dims", &dims_data)
            .map_err(metal_err)?;

        let kernel = self
            .provider
            .load_kernel("softmax", smallaios_arch_apple::shaders::SOFTMAX.as_bytes())
            .map_err(metal_err)?;

        let buf_in = self.tensor_cache.get("__gpu_in").unwrap();
        let buf_out = self.tensor_cache.get("__gpu_out").unwrap();
        let buf_dims = self.tensor_cache.get("__gpu_dims").unwrap();

        let tg_size = (total as u32).min(256);
        self.provider
            .launch(
                &kernel,
                [total as u32, 1, 1],
                [tg_size, 1, 1],
                &[buf_in, buf_out, buf_dims],
            )
            .map_err(metal_err)?;

        self.provider.synchronize().map_err(metal_err)?;

        let buf_out = self.tensor_cache.get("__gpu_out").unwrap();
        let raw_data = MetalTensorCache::copy_from_device(&self.provider, buf_out, byte_size)
            .map_err(metal_err)?;

        let mut output = Tensor::new(DataType::Float, input.shape.clone(), String::new());
        output.raw_data = raw_data;

        Ok(vec![output])
    }
}

// ---------------------------------------------------------------------------
// Conv2D
// ---------------------------------------------------------------------------

/// Extracts an integer list attribute, returning a default pair.
fn get_ints_attr(attrs: &[AttributeProto], name: &str, default: &[i64]) -> Vec<i64> {
    attrs
        .iter()
        .find(|a| a.name == name)
        .map(|a| {
            if a.ints.is_empty() {
                default.to_vec()
            } else {
                a.ints.clone()
            }
        })
        .unwrap_or_else(|| default.to_vec())
}

impl MetalDispatcher {
    fn dispatch_conv2d(
        &mut self,
        inputs: &[Option<&Tensor>],
        attrs: &[AttributeProto],
    ) -> Result<Vec<Tensor>, OpError> {
        let input = inputs
            .first()
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing Conv input")))?;
        let weight = inputs
            .get(1)
            .and_then(|o| o.as_ref())
            .ok_or_else(|| OpError::ShapeMismatch(String::from("missing Conv weight")))?;

        if input.data_type != DataType::Float || weight.data_type != DataType::Float {
            return Err(OpError::InternalError(String::from(
                "Metal Conv2D requires Float tensors",
            )));
        }

        // Input: [batch, in_channels, in_h, in_w]
        let in_dims = &input.shape.dims;
        if in_dims.len() != 4 {
            return Err(OpError::ShapeMismatch(String::from(
                "Conv2D input must be 4D (NCHW)",
            )));
        }
        let _batch = in_dims[0] as u32;
        let in_channels = in_dims[1] as u32;
        let in_h = in_dims[2] as u32;
        let in_w = in_dims[3] as u32;

        // Weight: [out_channels, in_channels, kernel_h, kernel_w]
        let w_dims = &weight.shape.dims;
        if w_dims.len() != 4 {
            return Err(OpError::ShapeMismatch(String::from(
                "Conv2D weight must be 4D",
            )));
        }
        let out_channels = w_dims[0] as u32;
        let kernel_h = w_dims[2] as u32;
        let kernel_w = w_dims[3] as u32;

        let strides = get_ints_attr(attrs, "strides", &[1, 1]);
        let pads = get_ints_attr(attrs, "pads", &[0, 0, 0, 0]);
        let stride_h = strides[0] as u32;
        let stride_w = if strides.len() > 1 {
            strides[1] as u32
        } else {
            stride_h
        };
        let pad_h = pads[0] as u32;
        let pad_w = if pads.len() > 1 {
            pads[1] as u32
        } else {
            pad_h
        };

        let out_h = (in_h + 2 * pad_h - kernel_h) / stride_h + 1;
        let out_w = (in_w + 2 * pad_w - kernel_w) / stride_w + 1;
        let out_elements = (out_channels as usize) * (out_h as usize) * (out_w as usize);
        let out_byte_size = out_elements * core::mem::size_of::<f32>();

        // Upload input and weight
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_in", &input.raw_data)
            .map_err(metal_err)?;
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_weight", &weight.raw_data)
            .map_err(metal_err)?;

        // Allocate output
        self.tensor_cache
            .get_or_create(&mut self.provider, "__gpu_out", out_byte_size)
            .map_err(metal_err)?;

        // Dims buffer: [batch, in_channels, in_h, in_w, out_channels,
        //               kernel_h, kernel_w, stride_h, stride_w, pad_h, pad_w]
        let dims_values: [u32; 11] = [
            1, // batch (we dispatch one sample)
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
        ];
        let dims_data: Vec<u8> = dims_values.iter().flat_map(|v| v.to_ne_bytes()).collect();
        self.tensor_cache
            .copy_to_device(&mut self.provider, "__gpu_dims", &dims_data)
            .map_err(metal_err)?;

        let kernel = self
            .provider
            .load_kernel("conv2d", smallaios_arch_apple::shaders::CONV2D.as_bytes())
            .map_err(metal_err)?;

        let buf_in = self.tensor_cache.get("__gpu_in").unwrap();
        let buf_weight = self.tensor_cache.get("__gpu_weight").unwrap();
        let buf_out = self.tensor_cache.get("__gpu_out").unwrap();
        let buf_dims = self.tensor_cache.get("__gpu_dims").unwrap();

        let tg_size = 256u32.min(out_w);
        self.provider
            .launch(
                &kernel,
                [out_w, out_h, out_channels],
                [tg_size.min(out_w).max(1), 1, 1],
                &[buf_in, buf_weight, buf_out, buf_dims],
            )
            .map_err(metal_err)?;

        self.provider.synchronize().map_err(metal_err)?;

        let buf_out = self.tensor_cache.get("__gpu_out").unwrap();
        let raw_data = MetalTensorCache::copy_from_device(&self.provider, buf_out, out_byte_size)
            .map_err(metal_err)?;

        let out_shape = TensorShape::new(vec![1, out_channels as i64, out_h as i64, out_w as i64]);
        let mut output = Tensor::new(DataType::Float, out_shape, String::new());
        output.raw_data = raw_data;

        Ok(vec![output])
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(target_os = "macos")]
mod tests {
    use super::*;
    use alloc::string::String;
    use alloc::vec;
    use alloc::vec::Vec;

    /// Helper: create a Float tensor from f32 data.
    fn make_tensor(shape: Vec<i64>, data: &[f32]) -> Tensor {
        let raw_data: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut t = Tensor::new(DataType::Float, TensorShape::new(shape), String::new());
        t.raw_data = raw_data;
        t
    }

    /// Helper: extract f32 values from a tensor.
    fn tensor_f32(t: &Tensor) -> Vec<f32> {
        t.raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Helper: assert two f32 slices are approximately equal.
    fn assert_approx_eq(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (a, e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "element {} differs: actual={}, expected={}, tol={}",
                i,
                a,
                e,
                tol
            );
        }
    }

    // ---- Elementwise Add ----

    #[test]
    fn test_elementwise_add_gpu_matches_cpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_tensor(vec![4], &[5.0, 6.0, 7.0, 8.0]);
        let result = disp
            .try_execute("Add", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[6.0, 8.0, 10.0, 12.0], 1e-5);
    }

    // ---- Elementwise Sub ----

    #[test]
    fn test_elementwise_sub_gpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![3], &[10.0, 20.0, 30.0]);
        let b = make_tensor(vec![3], &[1.0, 2.0, 3.0]);
        let result = disp
            .try_execute("Sub", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[9.0, 18.0, 27.0], 1e-5);
    }

    // ---- Elementwise Mul ----

    #[test]
    fn test_elementwise_mul_gpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![3], &[2.0, 3.0, 4.0]);
        let b = make_tensor(vec![3], &[5.0, 6.0, 7.0]);
        let result = disp
            .try_execute("Mul", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[10.0, 18.0, 28.0], 1e-5);
    }

    // ---- Elementwise Div ----

    #[test]
    fn test_elementwise_div_gpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![3], &[10.0, 20.0, 30.0]);
        let b = make_tensor(vec![3], &[2.0, 5.0, 10.0]);
        let result = disp
            .try_execute("Div", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[5.0, 4.0, 3.0], 1e-5);
    }

    // ---- Relu ----

    #[test]
    fn test_relu_gpu_matches_cpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let input = make_tensor(vec![6], &[-3.0, -1.0, 0.0, 1.0, 2.0, 3.0]);
        let result = disp
            .try_execute("Relu", &[Some(&input)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[0.0, 0.0, 0.0, 1.0, 2.0, 3.0], 1e-5);
    }

    // ---- Sigmoid ----

    #[test]
    fn test_sigmoid_gpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let input = make_tensor(vec![3], &[0.0, 1.0, -1.0]);
        let result = disp
            .try_execute("Sigmoid", &[Some(&input)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        // sigmoid(0)=0.5, sigmoid(1)~0.7311, sigmoid(-1)~0.2689
        assert_approx_eq(&out, &[0.5, 0.7310586, 0.26894143], 1e-4);
    }

    // ---- Tanh ----

    #[test]
    fn test_tanh_gpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let input = make_tensor(vec![3], &[0.0, 1.0, -1.0]);
        let result = disp
            .try_execute("Tanh", &[Some(&input)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[0.0, 0.7615942, -0.7615942], 1e-4);
    }

    // ---- MatMul ----

    #[test]
    fn test_matmul_gpu_matches_cpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        // A = [[1, 2], [3, 4]] (2x2), B = [[5, 6], [7, 8]] (2x2)
        // C = [[19, 22], [43, 50]]
        let a = make_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_tensor(vec![2, 2], &[5.0, 6.0, 7.0, 8.0]);
        let result = disp
            .try_execute("MatMul", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[19.0, 22.0, 43.0, 50.0], 1e-4);
    }

    // ---- MatMul non-square ----

    #[test]
    fn test_matmul_non_square_gpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        // A = [[1, 2, 3]] (1x3), B = [[4], [5], [6]] (3x1)
        // C = [[32]]
        let a = make_tensor(vec![1, 3], &[1.0, 2.0, 3.0]);
        let b = make_tensor(vec![3, 1], &[4.0, 5.0, 6.0]);
        let result = disp
            .try_execute("MatMul", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        assert_approx_eq(&out, &[32.0], 1e-4);
    }

    // ---- Softmax ----

    #[test]
    fn test_softmax_gpu_matches_cpu() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let input = make_tensor(vec![1, 4], &[1.0, 2.0, 3.0, 4.0]);
        let result = disp
            .try_execute("Softmax", &[Some(&input)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&result[0]);
        // softmax([1,2,3,4])
        let sum_exp: f32 = [1.0f32, 2.0, 3.0, 4.0].iter().map(|x| x.exp()).sum();
        let expected: Vec<f32> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .map(|x| x.exp() / sum_exp)
            .collect();
        assert_approx_eq(&out, &expected, 1e-4);
    }

    // ---- Tensor cache reuse ----

    #[test]
    fn test_tensor_cache_reuse() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        // Run Add twice; the second run should reuse cached buffers.
        let a = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let b = make_tensor(vec![4], &[1.0, 1.0, 1.0, 1.0]);

        let _r1 = disp
            .try_execute("Add", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();

        // Cache should have entries now
        assert!(!disp.tensor_cache.is_empty());

        // Second execution reuses cache
        let c = make_tensor(vec![4], &[10.0, 20.0, 30.0, 40.0]);
        let r2 = disp
            .try_execute("Add", &[Some(&c), Some(&b)], &[])
            .unwrap()
            .unwrap();
        let out = tensor_f32(&r2[0]);
        assert_approx_eq(&out, &[11.0, 21.0, 31.0, 41.0], 1e-5);
    }

    // ---- CPU fallback for unsupported op ----

    #[test]
    fn test_cpu_fallback_for_unsupported_op() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let result = disp.try_execute("Reshape", &[Some(&a)], &[]).unwrap();
        assert!(result.is_none(), "unsupported op should return None");
    }

    // ---- Unsupported op returns None ----

    #[test]
    fn test_unsupported_op_returns_none() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![2], &[1.0, 2.0]);
        assert!(disp
            .try_execute("Gather", &[Some(&a)], &[])
            .unwrap()
            .is_none());
        assert!(disp
            .try_execute("Slice", &[Some(&a)], &[])
            .unwrap()
            .is_none());
        assert!(disp.try_execute("Pad", &[Some(&a)], &[]).unwrap().is_none());
    }

    // ---- supports_op ----

    #[test]
    fn test_dispatcher_supports_op() {
        let disp = MetalDispatcher::new().expect("Metal required");
        assert!(disp.supports_op("Add"));
        assert!(disp.supports_op("MatMul"));
        assert!(disp.supports_op("Relu"));
        assert!(disp.supports_op("Conv"));
        assert!(!disp.supports_op("Reshape"));
    }

    // ---- Clear cache ----

    #[test]
    fn test_clear_cache() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![2], &[1.0, 2.0]);
        let b = make_tensor(vec![2], &[3.0, 4.0]);
        let _ = disp.try_execute("Add", &[Some(&a), Some(&b)], &[]).unwrap();
        assert!(!disp.tensor_cache.is_empty());
        disp.clear_cache();
        assert!(disp.tensor_cache.is_empty());
    }

    // ---- Shape mismatch errors ----

    #[test]
    fn test_elementwise_shape_mismatch() {
        let mut disp = MetalDispatcher::new().expect("Metal required");
        let a = make_tensor(vec![3], &[1.0, 2.0, 3.0]);
        let b = make_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]);
        let result = disp.try_execute("Add", &[Some(&a), Some(&b)], &[]);
        assert!(result.is_err());
    }
}
