// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX inference demo for Jetson Nano boot validation.
//!
//! Runs a simple 2-layer neural network (MatMul -> ReLU -> Softmax) on CPU
//! and reports GPU HAL status. This serves as a "hello world" smoke test
//! for the ONNX runtime on real hardware.

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use smallaios_onnx_rt::operators::{op_matmul, op_relu, op_softmax};
use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};

use crate::uart;

/// Create an f32 tensor from shape and data.
fn make_f32_tensor(shape: &[i64], data: &[f32]) -> Tensor {
    let mut raw = vec![0u8; data.len() * 4];
    for (i, &v) in data.iter().enumerate() {
        let bytes = v.to_le_bytes();
        raw[i * 4] = bytes[0];
        raw[i * 4 + 1] = bytes[1];
        raw[i * 4 + 2] = bytes[2];
        raw[i * 4 + 3] = bytes[3];
    }
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(Vec::from(shape)),
        name: String::new(),
        raw_data: raw,
    }
}

/// Read f32 values from a tensor's raw bytes.
fn read_f32_values(tensor: &Tensor) -> Vec<f32> {
    let count = tensor.raw_data.len() / 4;
    let mut values = Vec::with_capacity(count);
    for i in 0..count {
        let bytes = [
            tensor.raw_data[i * 4],
            tensor.raw_data[i * 4 + 1],
            tensor.raw_data[i * 4 + 2],
            tensor.raw_data[i * 4 + 3],
        ];
        values.push(f32::from_le_bytes(bytes));
    }
    values
}

/// Print a vector of f32 values via UART.
fn print_f32_vec(values: &[f32]) {
    for (i, &v) in values.iter().enumerate() {
        if i > 0 {
            uart::putc(b' ');
        }
        uart::put_f32(v);
    }
}

/// Run a simple CPU inference demo: MatMul -> ReLU -> Softmax.
///
/// Input:  [1, 4] x [4, 2] -> [1, 2] -> ReLU -> Softmax
pub fn run_cpu_inference_demo() {
    uart::puts("[onnx] CPU inference: MatMul -> ReLU -> Softmax\n");

    // Input: 1x4 vector
    let input_data: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
    let input = make_f32_tensor(&[1, 4], &input_data);
    uart::puts("[onnx] Input [1x4]: ");
    print_f32_vec(&input_data);
    uart::putc(b'\n');

    // Weights: 4x2 matrix
    #[rustfmt::skip]
    let weight_data: [f32; 8] = [
        0.1, -0.2,
        0.3,  0.4,
       -0.5,  0.6,
        0.7, -0.8,
    ];
    let weights = make_f32_tensor(&[4, 2], &weight_data);
    uart::puts("[onnx] Weights [4x2]: ");
    print_f32_vec(&weight_data);
    uart::putc(b'\n');

    // Step 1: MatMul
    match op_matmul(&input, &weights) {
        Ok(matmul_result) => {
            let matmul_vals = read_f32_values(&matmul_result);
            uart::puts("[onnx] MatMul result [1x2]: ");
            print_f32_vec(&matmul_vals);
            uart::putc(b'\n');

            // Step 2: ReLU
            match op_relu(&matmul_result) {
                Ok(relu_result) => {
                    let relu_vals = read_f32_values(&relu_result);
                    uart::puts("[onnx] ReLU result [1x2]: ");
                    print_f32_vec(&relu_vals);
                    uart::putc(b'\n');

                    // Step 3: Softmax
                    match op_softmax(&relu_result, 1) {
                        Ok(softmax_result) => {
                            let softmax_vals = read_f32_values(&softmax_result);
                            uart::puts("[onnx] Softmax result [1x2]: ");
                            print_f32_vec(&softmax_vals);
                            uart::putc(b'\n');
                            uart::puts("[onnx] CPU inference OK\n");
                        }
                        Err(_) => uart::puts("[onnx] ERROR: Softmax failed\n"),
                    }
                }
                Err(_) => uart::puts("[onnx] ERROR: ReLU failed\n"),
            }
        }
        Err(_) => uart::puts("[onnx] ERROR: MatMul failed\n"),
    }
}

/// Report Tegra GPU HAL status and attempt GPU platform initialization.
///
/// When the `tegra-x1` feature is enabled, this function:
/// 1. Creates a `TegraGpuPlatform` with hardware MMIO access
/// 2. Runs the GPU init sequence (power, clocks, GPCPLL, interrupts)
/// 3. Reports firmware availability status
/// 4. Prints GPU info to UART on success
/// 5. On failure, prints the error and continues boot (does not panic)
///
/// When `tegra-x1` is not enabled, this prints a static status message.
pub fn run_gpu_status_demo() {
    uart::puts("\n[gpu]  Tegra GM20B (Maxwell, CC 5.3)\n");
    uart::puts("[gpu]  128 CUDA cores @ BAR0 0x57000000\n");

    #[cfg(feature = "tegra-x1")]
    {
        run_tegra_gpu_init();
    }

    #[cfg(not(feature = "tegra-x1"))]
    {
        uart::puts("[gpu]  Status: HAL stub -- compute not yet implemented\n");
    }
}

/// Attempt Tegra GPU platform initialization via the nvidia HAL.
///
/// This calls into the `smallaios-arch-nvidia` tegra module to perform:
/// - PMC power partition enable
/// - CAR clock gate and reset release
/// - GPCPLL configuration (614 MHz default)
/// - GPU PMC interrupt enable (NV_PMC_INTR_EN_0)
///
/// Errors are reported via UART but do not halt the boot process.
/// Firmware loading is skipped when firmware stubs are empty (the normal
/// case until real firmware blobs are extracted from NVIDIA L4T).
#[cfg(feature = "tegra-x1")]
fn run_tegra_gpu_init() {
    use smallaios_arch_nvidia::tegra::power::HwMmio;
    use smallaios_arch_nvidia::tegra::TegraGpuPlatform;

    uart::puts("[gpu]  Initializing GPU platform (power, clocks, GPCPLL)...\n");

    let mmio = HwMmio;
    let mut platform = TegraGpuPlatform::new(mmio);

    match platform.init() {
        Ok(()) => {
            uart::puts("[gpu]  Platform ready: power ON, clocks ON, GPCPLL ");
            uart::put_dec(platform.gpcpll_freq() as u64);
            uart::puts(" MHz\n");
            uart::puts("[gpu]  Interrupts: NV_PMC_INTR_EN_0 configured\n");

            // TODO(hardware): On real hardware, also enable GICv2 SPI 189/190:
            //   crate::gicv2::enable_irq(crate::platform::GPU_IRQ_STALL);
            //   crate::gicv2::enable_irq(crate::platform::GPU_IRQ_NONSTALL);
            uart::puts("[gpu]  GICv2 SPI 189/190: deferred (requires GICv2 init)\n");

            // Check firmware availability
            if smallaios_arch_nvidia::tegra::falcon::firmware_available() {
                uart::puts("[gpu]  Firmware: available, loading...\n");
                // Full firmware load would happen here in production
            } else {
                uart::puts("[gpu]  Firmware: stubs only (real blobs not yet embedded)\n");
                uart::puts("[gpu]  Compute: not available until firmware is loaded\n");
            }
        }
        Err(_e) => {
            // Report error but continue boot -- GPU is not required for
            // basic system operation. On QEMU or without real hardware,
            // MMIO reads return 0 and init will likely timeout.
            uart::puts("[gpu]  WARN: GPU platform init failed (expected without real HW)\n");
            uart::puts("[gpu]  Continuing boot without GPU compute capability\n");
        }
    }
}
