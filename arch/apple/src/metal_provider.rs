// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Metal compute provider implementation.
//!
//! [`MetalProvider`] wraps a Metal device, command queue, and kernel cache to
//! implement [`smallaios_compute::ComputeProvider`]. Buffers use shared storage
//! mode (unified memory on Apple Silicon), and kernels are compiled from MSL
//! source strings at runtime.

use alloc::collections::BTreeMap;
use alloc::format;
use alloc::string::String;

use metal::objc::rc::autoreleasepool;
use smallaios_compute::{BackendType, ComputeProvider, DeviceInfo};

// ---------------------------------------------------------------------------
// MetalBuffer
// ---------------------------------------------------------------------------

/// Handle to a Metal GPU buffer allocation.
pub struct MetalBuffer {
    inner: metal::Buffer,
    size: usize,
}

// ---------------------------------------------------------------------------
// MetalKernel
// ---------------------------------------------------------------------------

/// Handle to a compiled Metal compute pipeline.
pub struct MetalKernel {
    pipeline: metal::ComputePipelineState,
    name: String,
}

// ---------------------------------------------------------------------------
// MetalError
// ---------------------------------------------------------------------------

/// Error type for Metal operations.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MetalError(pub String);

impl core::fmt::Display for MetalError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Metal error: {}", self.0)
    }
}

// ---------------------------------------------------------------------------
// MetalProvider
// ---------------------------------------------------------------------------

/// Apple Metal compute provider.
///
/// Owns a Metal device, command queue, and cache of compiled compute
/// pipelines keyed by kernel name.
pub struct MetalProvider {
    device: metal::Device,
    command_queue: metal::CommandQueue,
    kernel_cache: BTreeMap<String, metal::ComputePipelineState>,
}

impl MetalProvider {
    /// Create a new Metal provider using the system default GPU device.
    ///
    /// Returns an error if no Metal-capable device is found.
    pub fn new() -> Result<Self, MetalError> {
        let device = metal::Device::system_default()
            .ok_or_else(|| MetalError(String::from("no Metal device found")))?;
        let command_queue = device.new_command_queue();
        Ok(Self {
            device,
            command_queue,
            kernel_cache: BTreeMap::new(),
        })
    }

    /// Returns a reference to the underlying Metal device.
    pub fn device(&self) -> &metal::Device {
        &self.device
    }

    /// Returns a reference to the command queue.
    pub fn command_queue(&self) -> &metal::CommandQueue {
        &self.command_queue
    }

    /// Number of cached kernel pipelines.
    pub fn cached_kernel_count(&self) -> usize {
        self.kernel_cache.len()
    }
}

// ---------------------------------------------------------------------------
// ComputeProvider implementation
// ---------------------------------------------------------------------------

impl ComputeProvider for MetalProvider {
    type Buffer = MetalBuffer;
    type Kernel = MetalKernel;
    type Error = MetalError;

    fn device_info(&self) -> DeviceInfo {
        DeviceInfo {
            name: String::from(self.device.name()),
            memory_bytes: self.device.recommended_max_working_set_size(),
            compute_units: 0, // Metal does not expose EU/core count directly
            backend_type: BackendType::Metal,
        }
    }

    fn init(&mut self) -> Result<(), Self::Error> {
        // Device and command queue are created in `new()`.
        Ok(())
    }

    fn alloc(&mut self, size: usize) -> Result<Self::Buffer, Self::Error> {
        let buffer = self
            .device
            .new_buffer(size as u64, metal::MTLResourceOptions::StorageModeShared);
        Ok(MetalBuffer {
            inner: buffer,
            size,
        })
    }

    fn free(&mut self, _buf: Self::Buffer) -> Result<(), Self::Error> {
        // Metal buffers are reference-counted; dropping releases the allocation.
        Ok(())
    }

    fn copy_host_to_device(
        &mut self,
        src: &[u8],
        dst: &mut Self::Buffer,
    ) -> Result<(), Self::Error> {
        if src.len() > dst.size {
            return Err(MetalError(format!(
                "source ({} bytes) exceeds buffer size ({} bytes)",
                src.len(),
                dst.size
            )));
        }
        let contents = dst.inner.contents() as *mut u8;
        unsafe {
            core::ptr::copy_nonoverlapping(src.as_ptr(), contents, src.len());
        }
        Ok(())
    }

    fn copy_device_to_host(&self, src: &Self::Buffer, dst: &mut [u8]) -> Result<(), Self::Error> {
        let copy_len = dst.len().min(src.size);
        let contents = src.inner.contents() as *const u8;
        unsafe {
            core::ptr::copy_nonoverlapping(contents, dst.as_mut_ptr(), copy_len);
        }
        Ok(())
    }

    fn load_kernel(&mut self, name: &str, source: &[u8]) -> Result<Self::Kernel, Self::Error> {
        // Check cache first.
        if let Some(cached) = self.kernel_cache.get(name) {
            return Ok(MetalKernel {
                pipeline: cached.clone(),
                name: String::from(name),
            });
        }

        let source_str = core::str::from_utf8(source)
            .map_err(|_| MetalError(String::from("invalid UTF-8 in MSL source")))?;

        let options = metal::CompileOptions::new();
        let library = self
            .device
            .new_library_with_source(source_str, &options)
            .map_err(|e| MetalError(format!("MSL compile failed: {}", e)))?;

        let function = library
            .get_function(name, None)
            .map_err(|e| MetalError(format!("function '{}' not found: {}", name, e)))?;

        let pipeline = self
            .device
            .new_compute_pipeline_state_with_function(&function)
            .map_err(|e| MetalError(format!("pipeline creation failed for '{}': {}", name, e)))?;

        self.kernel_cache
            .insert(String::from(name), pipeline.clone());

        Ok(MetalKernel {
            pipeline,
            name: String::from(name),
        })
    }

    fn launch(
        &mut self,
        kernel: &Self::Kernel,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[&Self::Buffer],
    ) -> Result<(), Self::Error> {
        autoreleasepool(|| {
            let command_buffer = self.command_queue.new_command_buffer();
            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&kernel.pipeline);

            for (i, buf) in args.iter().enumerate() {
                encoder.set_buffer(i as u64, Some(&buf.inner), 0);
            }

            let threads_per_grid =
                metal::MTLSize::new(grid[0] as u64, grid[1] as u64, grid[2] as u64);
            let threads_per_group =
                metal::MTLSize::new(block[0] as u64, block[1] as u64, block[2] as u64);

            encoder.dispatch_threads(threads_per_grid, threads_per_group);
            encoder.end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        });

        Ok(())
    }

    fn synchronize(&mut self) -> Result<(), Self::Error> {
        // Each launch already calls wait_until_completed. For truly async
        // dispatch we would track the last command buffer here.
        Ok(())
    }

    fn supports_op(&self, op: &str) -> bool {
        matches!(
            op,
            "Add"
                | "Sub"
                | "Mul"
                | "Div"
                | "Relu"
                | "Sigmoid"
                | "Tanh"
                | "MatMul"
                | "Gemm"
                | "Softmax"
                | "Conv"
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;
    use alloc::string::String;
    use smallaios_compute::ComputeProvider;

    // Helper: create provider or skip if no Metal device.
    fn test_provider() -> MetalProvider {
        MetalProvider::new().expect("Metal device required for tests")
    }

    // -- Device info --

    #[test]
    fn test_device_info() {
        let prov = test_provider();
        let info = prov.device_info();
        assert_eq!(info.backend_type, BackendType::Metal);
        assert!(!info.name.is_empty());
        assert!(info.memory_bytes > 0);
    }

    // -- Init --

    #[test]
    fn test_init() {
        let mut prov = test_provider();
        assert!(prov.init().is_ok());
    }

    // -- Alloc / Free --

    #[test]
    fn test_alloc_and_free() {
        let mut prov = test_provider();
        let buf = prov.alloc(4096).unwrap();
        assert_eq!(buf.size, 4096);
        assert!(prov.free(buf).is_ok());
    }

    // -- Host-to-device and device-to-host round-trip --

    #[test]
    fn test_copy_round_trip() {
        let mut prov = test_provider();
        let data: [u8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut buf = prov.alloc(16).unwrap();

        prov.copy_host_to_device(&data, &mut buf).unwrap();

        let mut out = [0u8; 16];
        prov.copy_device_to_host(&buf, &mut out).unwrap();

        assert_eq!(data, out);
    }

    // -- Copy overflow check --

    #[test]
    fn test_copy_h2d_overflow_rejected() {
        let mut prov = test_provider();
        let mut buf = prov.alloc(4).unwrap();
        let big_data = [0u8; 8];
        assert!(prov.copy_host_to_device(&big_data, &mut buf).is_err());
    }

    // -- Load kernel + launch --

    #[test]
    fn test_load_and_launch_add_kernel() {
        let mut prov = test_provider();

        let n: usize = 256;
        let byte_size = n * core::mem::size_of::<f32>();

        // Prepare input buffers.
        let mut buf_a = prov.alloc(byte_size).unwrap();
        let mut buf_b = prov.alloc(byte_size).unwrap();
        let buf_c = prov.alloc(byte_size).unwrap();

        // Fill a = [1.0; 256], b = [2.0; 256].
        let ones: alloc::vec::Vec<u8> = (0..n).flat_map(|_| 1.0_f32.to_ne_bytes()).collect();
        let twos: alloc::vec::Vec<u8> = (0..n).flat_map(|_| 2.0_f32.to_ne_bytes()).collect();

        prov.copy_host_to_device(&ones, &mut buf_a).unwrap();
        prov.copy_host_to_device(&twos, &mut buf_b).unwrap();

        // Compile and launch the add kernel.
        let kernel = prov
            .load_kernel(
                "elementwise_add",
                crate::shaders::ELEMENTWISE_ADD.as_bytes(),
            )
            .unwrap();

        prov.launch(
            &kernel,
            [n as u32, 1, 1],
            [64, 1, 1],
            &[&buf_a, &buf_b, &buf_c],
        )
        .unwrap();

        prov.synchronize().unwrap();

        // Read back and verify c = 3.0 everywhere.
        let mut out_bytes = alloc::vec![0u8; byte_size];
        prov.copy_device_to_host(&buf_c, &mut out_bytes).unwrap();

        for i in 0..n {
            let offset = i * 4;
            let val = f32::from_ne_bytes([
                out_bytes[offset],
                out_bytes[offset + 1],
                out_bytes[offset + 2],
                out_bytes[offset + 3],
            ]);
            assert!(
                (val - 3.0).abs() < 1e-6,
                "Element {} was {}, expected 3.0",
                i,
                val
            );
        }
    }

    // -- Kernel caching --

    #[test]
    fn test_kernel_caching() {
        let mut prov = test_provider();
        assert_eq!(prov.cached_kernel_count(), 0);

        let _k1 = prov
            .load_kernel(
                "elementwise_add",
                crate::shaders::ELEMENTWISE_ADD.as_bytes(),
            )
            .unwrap();
        assert_eq!(prov.cached_kernel_count(), 1);

        // Loading same name should use cache, not increase count.
        let _k2 = prov
            .load_kernel(
                "elementwise_add",
                crate::shaders::ELEMENTWISE_ADD.as_bytes(),
            )
            .unwrap();
        assert_eq!(prov.cached_kernel_count(), 1);
    }

    // -- supports_op --

    #[test]
    fn test_supports_op() {
        let prov = test_provider();
        assert!(prov.supports_op("Add"));
        assert!(prov.supports_op("MatMul"));
        assert!(prov.supports_op("Relu"));
        assert!(prov.supports_op("Softmax"));
        assert!(prov.supports_op("Conv"));
        assert!(!prov.supports_op("Reshape"));
        assert!(!prov.supports_op("UnknownOp"));
    }

    // -- Error display --

    #[test]
    fn test_error_display() {
        let e = MetalError(String::from("test error"));
        let msg = format!("{}", e);
        assert!(msg.contains("Metal error"));
        assert!(msg.contains("test error"));
    }

    // -- Load kernel with invalid source --

    #[test]
    fn test_load_kernel_invalid_source() {
        let mut prov = test_provider();
        let result = prov.load_kernel("bad", b"this is not valid MSL");
        assert!(result.is_err());
    }

    // -- Relu kernel --

    #[test]
    fn test_relu_kernel() {
        let mut prov = test_provider();

        let n: usize = 8;
        let byte_size = n * core::mem::size_of::<f32>();

        let mut buf_in = prov.alloc(byte_size).unwrap();
        let buf_out = prov.alloc(byte_size).unwrap();

        // Input: [-4, -3, -2, -1, 0, 1, 2, 3]
        let input_data: alloc::vec::Vec<u8> = [-4.0_f32, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
            .iter()
            .flat_map(|v| v.to_ne_bytes())
            .collect();

        prov.copy_host_to_device(&input_data, &mut buf_in).unwrap();

        let kernel = prov
            .load_kernel(
                "elementwise_relu",
                crate::shaders::ELEMENTWISE_RELU.as_bytes(),
            )
            .unwrap();

        prov.launch(&kernel, [n as u32, 1, 1], [8, 1, 1], &[&buf_in, &buf_out])
            .unwrap();
        prov.synchronize().unwrap();

        let mut out_bytes = alloc::vec![0u8; byte_size];
        prov.copy_device_to_host(&buf_out, &mut out_bytes).unwrap();

        let expected = [0.0_f32, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0];
        for i in 0..n {
            let offset = i * 4;
            let val = f32::from_ne_bytes([
                out_bytes[offset],
                out_bytes[offset + 1],
                out_bytes[offset + 2],
                out_bytes[offset + 3],
            ]);
            assert!(
                (val - expected[i]).abs() < 1e-6,
                "Relu element {} was {}, expected {}",
                i,
                val,
                expected[i]
            );
        }
    }
}
