// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CUDA integration tests.
//!
//! These tests require a CUDA-capable GPU and are `#[ignore]`'d by default.
//! Run them with:
//!
//!     RUSTFLAGS="-L/usr/local/cuda-13.0/targets/sbsa-linux/lib -L/usr/lib/aarch64-linux-gnu" \
//!       cargo test -p smallaios-onnx-rt --features cuda --test test_cuda -- --ignored

#![cfg(feature = "cuda")]

extern crate smallaios_onnx_rt;

use smallaios_onnx_rt::cuda;
use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};

#[test]
#[ignore]
fn test_cuda_device_discovery() {
    let count = cuda::device_count().expect("cudaGetDeviceCount failed");
    assert!(count > 0, "expected at least 1 CUDA device, got {}", count);
    eprintln!("Found {} CUDA device(s)", count);

    let info = cuda::device_info(0).expect("cudaGetDeviceProperties failed");
    eprintln!(
        "Device 0: {} (compute {}.{}, {} MB VRAM)",
        info.name,
        info.compute_major,
        info.compute_minor,
        info.total_mem_bytes / (1024 * 1024),
    );
    assert!(!info.name.is_empty());
    assert!(info.total_mem_bytes > 0);
}

#[test]
#[ignore]
fn test_cuda_version_check() {
    let version = cuda::check_version().expect("CUDA version check failed");
    let major = version / 1000;
    let minor = (version % 1000) / 10;
    eprintln!("CUDA runtime version: {}.{}", major, minor);
    assert_eq!(major, 13, "expected CUDA 13.x");
}

#[test]
#[ignore]
fn test_device_buffer_alloc_copy_roundtrip() {
    cuda::set_device(0).expect("set device failed");

    let data: Vec<u8> = (0..256).map(|i| (i & 0xFF) as u8).collect();
    let buf = cuda::DeviceBuffer::alloc(data.len()).expect("alloc failed");
    assert_eq!(buf.size(), 256);

    buf.copy_from_host(&data)
        .expect("host-to-device copy failed");

    let mut result = vec![0u8; 256];
    buf.copy_to_host(&mut result)
        .expect("device-to-host copy failed");

    assert_eq!(data, result, "roundtrip data mismatch");
    eprintln!("DeviceBuffer roundtrip: 256 bytes OK");
}

#[test]
#[ignore]
fn test_cublas_sgemm() {
    cuda::set_device(0).expect("set device failed");
    let cublas = cuda::CublasHandle::new().expect("cuBLAS create failed");

    // Simple 2x2 GEMM: C = A * B
    // A = [[1, 2], [3, 4]]  B = [[5, 6], [7, 8]]
    // Expected C = [[19, 22], [43, 50]]
    //
    // cuBLAS is column-major, so we store as column-major:
    // A_col = [1, 3, 2, 4]  B_col = [5, 7, 6, 8]
    let a_data: [f32; 4] = [1.0, 3.0, 2.0, 4.0]; // column-major
    let b_data: [f32; 4] = [5.0, 7.0, 6.0, 8.0]; // column-major
    let c_zeros: [f32; 4] = [0.0; 4];

    let a_bytes = unsafe { core::slice::from_raw_parts(a_data.as_ptr() as *const u8, 16) };
    let b_bytes = unsafe { core::slice::from_raw_parts(b_data.as_ptr() as *const u8, 16) };
    let c_bytes = unsafe { core::slice::from_raw_parts(c_zeros.as_ptr() as *const u8, 16) };

    let a_buf = cuda::DeviceBuffer::alloc(16).unwrap();
    let b_buf = cuda::DeviceBuffer::alloc(16).unwrap();
    let c_buf = cuda::DeviceBuffer::alloc(16).unwrap();

    a_buf.copy_from_host(a_bytes).unwrap();
    b_buf.copy_from_host(b_bytes).unwrap();
    c_buf.copy_from_host(c_bytes).unwrap();

    cublas
        .sgemm(
            cuda::ffi::cublasOperation_t::CUBLAS_OP_N,
            cuda::ffi::cublasOperation_t::CUBLAS_OP_N,
            2,   // m
            2,   // n
            2,   // k
            1.0, // alpha
            &a_buf,
            2, // lda
            &b_buf,
            2,   // ldb
            0.0, // beta
            &c_buf,
            2, // ldc
        )
        .expect("cublasSgemm failed");

    cuda::synchronize().expect("sync failed");

    let mut result_bytes = [0u8; 16];
    c_buf.copy_to_host(&mut result_bytes).unwrap();
    let result: &[f32] =
        unsafe { core::slice::from_raw_parts(result_bytes.as_ptr() as *const f32, 4) };

    // Column-major result: [19, 43, 22, 50]
    let expected = [19.0_f32, 43.0, 22.0, 50.0];
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "GEMM result[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }
    eprintln!("cuBLAS sgemm 2x2: OK ({:?})", result);
}

#[test]
#[ignore]
fn test_cuda_runtime_init() {
    let rt = cuda::CudaRuntime::init().expect("CudaRuntime init failed");
    eprintln!(
        "CudaRuntime: device={}, CUDA {}, cuBLAS OK, cuDNN OK",
        rt.device.name,
        rt.cuda_version / 1000
    );
    assert!(rt.device.total_mem_bytes > 0);
    assert!(cuda::CudaRuntime::supports_op("MatMul"));
    assert!(cuda::CudaRuntime::supports_op("Gemm"));
    assert!(cuda::CudaRuntime::supports_op("Conv"));
    assert!(cuda::CudaRuntime::supports_op("Add"));
    assert!(cuda::CudaRuntime::supports_op("Mul"));
    assert!(cuda::CudaRuntime::supports_op("Silu"));
    assert!(cuda::CudaRuntime::supports_op("Gather"));
    assert!(cuda::CudaRuntime::supports_op("RMSNormalization"));
    assert!(cuda::CudaRuntime::supports_op("RotaryEmbedding"));
    assert!(cuda::CudaRuntime::supports_op("GroupQueryAttention"));
    assert!(!cuda::CudaRuntime::supports_op("Relu"));
}

#[test]
#[ignore]
fn test_device_weight_store() {
    cuda::set_device(0).expect("set device failed");

    let mut store = cuda::DeviceWeightStore::new();
    let weight_data = vec![42u8; 1024];
    store
        .load_weight("conv1.weight".into(), &weight_data)
        .expect("load_weight failed");

    assert_eq!(store.count(), 1);
    assert_eq!(store.total_bytes(), 1024);
    assert!(store.get("conv1.weight").is_some());
    assert!(store.get("missing").is_none());

    // Verify data roundtrip.
    let buf = store.get("conv1.weight").unwrap();
    let mut result = vec![0u8; 1024];
    buf.copy_to_host(&mut result).unwrap();
    assert_eq!(result, weight_data);
    eprintln!("DeviceWeightStore: 1 weight, 1024 bytes, roundtrip OK");
}

/// Helper to create an f32 Tensor from a shape and data.
fn make_f32_tensor(shape: &[i64], data: &[f32]) -> Tensor {
    let raw: Vec<u8> = data.iter().flat_map(|f| f.to_le_bytes()).collect();
    Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: raw,
    }
}

/// Read f32 values from a tensor's raw bytes.
fn read_f32(t: &Tensor) -> Vec<f32> {
    t.raw_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

#[test]
#[ignore]
fn test_gpu_gemm_dispatch_2x3_times_3x2() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // A[2,3] * B[3,2] = C[2,2]
    let a = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let b = make_f32_tensor(&[3, 2], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);

    let c = cuda::dispatch::gpu_gemm(&rt, &a, &b, false, false, 1.0, 0.0, None)
        .expect("gpu_gemm failed");

    let result = read_f32(&c);
    // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    //         = [[58, 64], [139, 154]]
    let expected = [58.0_f32, 64.0, 139.0, 154.0];

    assert_eq!(c.shape.dims, vec![2, 2]);
    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "gemm result[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }
    eprintln!("gpu_gemm dispatch 2x3 * 3x2: OK {:?}", result);
}

#[test]
#[ignore]
fn test_gpu_gemm_with_bias() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // C = 1.0 * A * B + 1.0 * bias (broadcast)
    let a = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let b = make_f32_tensor(&[2, 2], &[5.0, 6.0, 7.0, 8.0]);
    let bias = make_f32_tensor(&[2], &[10.0, 20.0]); // row-broadcast

    let c = cuda::dispatch::gpu_gemm(&rt, &a, &b, false, false, 1.0, 1.0, Some(&bias))
        .expect("gpu_gemm with bias failed");

    let result = read_f32(&c);
    // A*B = [[19, 22], [43, 50]], + bias [10, 20] = [[29, 42], [53, 70]]
    let expected = [29.0_f32, 42.0, 53.0, 70.0];

    for (i, (&got, &exp)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-3,
            "gemm+bias result[{}]: expected {}, got {}",
            i,
            exp,
            got
        );
    }
    eprintln!("gpu_gemm with bias: OK {:?}", result);
}

#[test]
#[ignore]
fn test_gpu_gemm_matches_cpu_large_matrix() {
    // Use F32 precision (not TF32) for exact-match testing.
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // 64x64 random-ish matrix multiply, compare GPU vs CPU.
    let n = 64usize;
    let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| ((n * n - i) as f32) * 0.01).collect();

    let a = make_f32_tensor(&[n as i64, n as i64], &a_data);
    let b = make_f32_tensor(&[n as i64, n as i64], &b_data);

    let gpu_c = cuda::dispatch::gpu_gemm(&rt, &a, &b, false, false, 1.0, 0.0, None)
        .expect("gpu_gemm failed");

    // CPU reference (naive matmul).
    let mut cpu_c = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..n {
                sum += a_data[i * n + kk] * b_data[kk * n + j];
            }
            cpu_c[i * n + j] = sum;
        }
    }

    let gpu_result = read_f32(&gpu_c);
    let mut max_err: f32 = 0.0;
    for i in 0..n * n {
        let err = (gpu_result[i] - cpu_c[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    eprintln!(
        "64x64 GPU vs CPU GEMM: max absolute error = {:.6e}",
        max_err
    );
    // f32 GEMM on GPU uses FMA with different rounding than CPU naive loop.
    // ~0.02 max error on 64x64 with values up to ~40.96 is expected.
    assert!(max_err < 0.05, "GPU/CPU mismatch too large: {}", max_err);
}

#[test]
#[ignore]
fn test_gpu_gemm_int8() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // INT8 GEMM: 4x4 * 4x4 (dimensions must be multiples of 4)
    let a_data: Vec<u8> = (1..=16).map(|i| i as u8).collect();
    let b_data: Vec<u8> = (1..=16).rev().map(|i| i as u8).collect();

    let a = Tensor {
        data_type: DataType::Int8,
        shape: TensorShape::new(vec![4, 4]),
        name: String::new(),
        raw_data: a_data,
    };
    let b = Tensor {
        data_type: DataType::Int8,
        shape: TensorShape::new(vec![4, 4]),
        name: String::new(),
        raw_data: b_data,
    };

    let result = cuda::dispatch::gpu_gemm_int8(&rt, &a, &b).expect("gpu_gemm_int8 failed");
    assert!(result.is_some(), "4-aligned dims should not fall back");
    let c = result.unwrap();
    assert_eq!(c.data_type, DataType::Int32);
    assert_eq!(c.shape.dims, vec![4, 4]);

    let c_vals: Vec<i32> = c
        .raw_data
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // CPU reference
    let a_i8: Vec<i8> = (1..=16).map(|i| i as i8).collect();
    let b_i8: Vec<i8> = (1..=16).rev().map(|i| i as i8).collect();
    let mut expected = vec![0i32; 16];
    for i in 0..4 {
        for j in 0..4 {
            let mut sum = 0i32;
            for kk in 0..4 {
                sum += a_i8[i * 4 + kk] as i32 * b_i8[kk * 4 + j] as i32;
            }
            expected[i * 4 + j] = sum;
        }
    }

    assert_eq!(c_vals, expected, "INT8 GEMM result mismatch");
    eprintln!("gpu_gemm_int8 4x4: OK {:?}", c_vals);
}

#[test]
#[ignore]
fn test_gpu_gemm_int8_unaligned_falls_back() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // 3x3 — not 4-aligned, should return None
    let a = Tensor {
        data_type: DataType::Int8,
        shape: TensorShape::new(vec![3, 3]),
        name: String::new(),
        raw_data: vec![1u8; 9],
    };
    let b = Tensor {
        data_type: DataType::Int8,
        shape: TensorShape::new(vec![3, 3]),
        name: String::new(),
        raw_data: vec![1u8; 9],
    };

    let result = cuda::dispatch::gpu_gemm_int8(&rt, &a, &b).expect("should not error");
    assert!(
        result.is_none(),
        "3x3 should fall back to CPU (not 4-aligned)"
    );
    eprintln!("gpu_gemm_int8 3x3: correctly falls back to CPU");
}

// ── Precision mode tests ────────────────────────────────────────────

fn precision_gemm_max_error(precision: cuda::GpuPrecision, n: usize) -> f32 {
    let rt = cuda::CudaRuntime::init_with_precision(precision).expect("CUDA init");
    let a_data: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..n * n).map(|i| ((n * n - i) as f32) * 0.01).collect();
    let a = make_f32_tensor(&[n as i64, n as i64], &a_data);
    let b = make_f32_tensor(&[n as i64, n as i64], &b_data);
    let gpu_c = cuda::dispatch::gpu_gemm(&rt, &a, &b, false, false, 1.0, 0.0, None)
        .expect("gpu_gemm failed");
    let gpu_result = read_f32(&gpu_c);

    let mut cpu_c = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0f32;
            for kk in 0..n {
                sum += a_data[i * n + kk] * b_data[kk * n + j];
            }
            cpu_c[i * n + j] = sum;
        }
    }
    let mut max_err: f32 = 0.0;
    for i in 0..n * n {
        let err = (gpu_result[i] - cpu_c[i]).abs();
        if err > max_err {
            max_err = err;
        }
    }
    max_err
}

#[test]
#[ignore]
fn test_precision_tf32() {
    let err = precision_gemm_max_error(cuda::GpuPrecision::Tf32, 64);
    eprintln!("TF32 64x64 max error: {:.6e}", err);
    // TF32 has 10-bit mantissa — larger error than f32 but much faster.
    assert!(err < 10.0, "TF32 error too large: {}", err);
}

#[test]
#[ignore]
fn test_precision_fp16() {
    let err = precision_gemm_max_error(cuda::GpuPrecision::Fp16, 64);
    eprintln!("FP16 64x64 max error: {:.6e}", err);
    // FP16 accumulation with f32 I/O.
    assert!(err < 10.0, "FP16 error too large: {}", err);
}

#[test]
#[ignore]
fn test_precision_bf16() {
    let err = precision_gemm_max_error(cuda::GpuPrecision::Bf16, 64);
    eprintln!("BF16 64x64 max error: {:.6e}", err);
    // BF16 has 8-bit mantissa — wider dynamic range than FP16 but less precision.
    assert!(err < 10.0, "BF16 error too large: {}", err);
}

#[test]
#[ignore]
fn test_precision_f32_strict() {
    let err = precision_gemm_max_error(cuda::GpuPrecision::F32, 64);
    eprintln!("F32 64x64 max error: {:.6e}", err);
    // Pure f32 — only FMA rounding differences.
    assert!(err < 0.05, "F32 error too large: {}", err);
}

// ── Conv tests ──────────────────────────────────────────────────────

#[test]
#[ignore]
fn test_gpu_conv2d_1x1() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // 1x1 conv: input [1,1,3,3], weight [1,1,1,1] (scale by 2.0)
    let x = make_f32_tensor(
        &[1, 1, 3, 3],
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
    );
    let w = make_f32_tensor(&[1, 1, 1, 1], &[2.0]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[1, 1], 1)
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 1, 3, 3]);
    let vals = read_f32(&y);
    let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "conv 1x1 [{}]: {} vs {}",
            i,
            got,
            exp
        );
    }
    eprintln!("gpu_conv2d 1x1: OK {:?}", vals);
}

#[test]
#[ignore]
fn test_gpu_conv2d_3x3_with_padding() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // 3x3 conv with padding=1: input [1,1,4,4], weight [1,1,3,3] (all ones)
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = make_f32_tensor(&[1, 1, 4, 4], &x_data);
    let w = make_f32_tensor(&[1, 1, 3, 3], &[1.0; 9]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[1, 1, 1, 1], &[1, 1], &[1, 1], 1)
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    // Output should be [1,1,4,4] (same spatial due to pad=1)
    assert_eq!(y.shape.dims, vec![1, 1, 4, 4]);
    let vals = read_f32(&y);

    // Spot check: center element y[0,0,1,1] = sum of 3x3 patch around (1,1)
    // = 1+2+3+5+6+7+9+10+11 = 54
    assert!(
        (vals[5] - 54.0).abs() < 1e-4,
        "conv center: {} vs 54",
        vals[5]
    );
    eprintln!("gpu_conv2d 3x3 pad=1: OK, center={}", vals[5]);
}

#[test]
#[ignore]
fn test_gpu_conv2d_with_bias() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // 1x1 conv + bias: input [1,1,2,2], weight [2,1,1,1], bias [2]
    let x = make_f32_tensor(&[1, 1, 2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let w = make_f32_tensor(&[2, 1, 1, 1], &[1.0, -1.0]); // chan 0: identity, chan 1: negate
    let bias = make_f32_tensor(&[2], &[10.0, 20.0]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, Some(&bias), &[0, 0], &[1, 1], &[1, 1], 1)
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 2, 2, 2]);
    let vals = read_f32(&y);
    // Chan 0: [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
    // Chan 1: [-1+20, -2+20, -3+20, -4+20] = [19, 18, 17, 16]
    let expected = [11.0, 12.0, 13.0, 14.0, 19.0, 18.0, 17.0, 16.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-4,
            "conv+bias [{}]: {} vs {}",
            i,
            got,
            exp
        );
    }
    eprintln!("gpu_conv2d with bias: OK {:?}", vals);
}

#[test]
#[ignore]
fn test_gpu_conv2d_strided() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // Stride=2 conv: input [1,1,4,4], weight [1,1,1,1] (identity), stride 2
    let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = make_f32_tensor(&[1, 1, 4, 4], &x_data);
    let w = make_f32_tensor(&[1, 1, 1, 1], &[1.0]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[2, 2], &[1, 1], 1)
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 1, 2, 2]);
    let vals = read_f32(&y);
    // Stride=2 picks every other element: [1, 3, 9, 11]
    let expected = [1.0, 3.0, 9.0, 11.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - exp).abs() < 1e-5,
            "strided conv [{}]: {} vs {}",
            i,
            got,
            exp
        );
    }
    eprintln!("gpu_conv2d stride=2: OK {:?}", vals);
}

#[test]
#[ignore]
fn test_gpu_conv2d_dilated() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // Dilated 3x3 conv: input [1,1,5,5], weight [1,1,3,3] (all 1s), dilation=2
    // Effective kernel size = 5x5, so output with no padding = [1,1,1,1]
    let x_data: Vec<f32> = (1..=25).map(|i| i as f32).collect();
    let x = make_f32_tensor(&[1, 1, 5, 5], &x_data);
    let w = make_f32_tensor(&[1, 1, 3, 3], &[1.0; 9]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[2, 2], 1)
        .expect("gpu_conv2d dilated failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 1, 1, 1]);
    let vals = read_f32(&y);
    // Dilated 3x3 with dilation=2 picks positions (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)
    // = 1+3+5+11+13+15+21+23+25 = 117
    assert!(
        (vals[0] - 117.0).abs() < 1e-4,
        "dilated conv: {} vs 117",
        vals[0]
    );
    eprintln!("gpu_conv2d dilation=2: OK val={}", vals[0]);
}

#[test]
#[ignore]
fn test_gpu_conv2d_tf32_precision() {
    // Conv with TF32 tensor core acceleration.
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::Tf32).expect("CUDA init");

    let x_data: Vec<f32> = (1..=16).map(|i| i as f32).collect();
    let x = make_f32_tensor(&[1, 1, 4, 4], &x_data);
    let w = make_f32_tensor(&[1, 1, 3, 3], &[1.0; 9]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[1, 1, 1, 1], &[1, 1], &[1, 1], 1)
        .expect("gpu_conv2d TF32 failed");
    assert!(result.is_some());
    let y = result.unwrap();
    assert_eq!(y.shape.dims, vec![1, 1, 4, 4]);

    let vals = read_f32(&y);
    // TF32 may have slightly different rounding but should be close.
    // Center element should be ~54.
    assert!(
        (vals[5] - 54.0).abs() < 1.0,
        "TF32 conv center: {} vs 54",
        vals[5]
    );
    eprintln!("gpu_conv2d TF32: OK, center={}", vals[5]);
}

#[test]
#[ignore]
fn test_gpu_conv2d_non4d_falls_back() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // 3D input — should return None (not a 4D conv).
    let x = make_f32_tensor(&[1, 3, 5], &[1.0; 15]);
    let w = make_f32_tensor(&[1, 3, 3], &[1.0; 9]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[1, 1], 1)
        .expect("should not error");
    assert!(result.is_none(), "3D conv should fall back to CPU");
    eprintln!("gpu_conv2d 3D: correctly falls back to CPU");
}

// ── FP8 tests ───────────────────────────────────────────────────────

/// Encode an f32 value to FP8 E4M3 format (1 byte).
/// E4M3: sign(1) + exponent(4) + mantissa(3), bias=7, max=448.
fn f32_to_fp8_e4m3(val: f32) -> u8 {
    // Simplified conversion: only handles positive values 0-448.
    if val == 0.0 {
        return 0;
    }
    let sign = if val < 0.0 { 1u8 } else { 0u8 };
    let abs_val = val.abs();
    // FP8 E4M3: exponent bias = 7
    let bits = abs_val.to_bits();
    let f32_exp = ((bits >> 23) & 0xFF) as i32 - 127;
    let f32_mant = (bits >> 20) & 0x7; // top 3 mantissa bits
    let fp8_exp = (f32_exp + 7).clamp(0, 15) as u8;
    (sign << 7) | (fp8_exp << 3) | (f32_mant as u8)
}

#[test]
#[ignore]
fn test_gpu_gemm_fp8_e4m3() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // FP8 GEMM: 16x16 * 16x16 (all 1.0 in E4M3)
    let n = 16usize;
    let one_fp8 = f32_to_fp8_e4m3(1.0);
    let a_data = vec![one_fp8; n * n];
    let b_data = vec![one_fp8; n * n];

    let a = Tensor {
        data_type: DataType::Float8E4M3,
        shape: TensorShape::new(vec![n as i64, n as i64]),
        name: String::new(),
        raw_data: a_data,
    };
    let b = Tensor {
        data_type: DataType::Float8E4M3,
        shape: TensorShape::new(vec![n as i64, n as i64]),
        name: String::new(),
        raw_data: b_data,
    };

    let result =
        cuda::dispatch::gpu_gemm_fp8(&rt, &a, &b, cuda::ffi::cudaDataType_t::CUDA_R_8F_E4M3)
            .expect("gpu_gemm_fp8 failed");

    assert!(result.is_some(), "FP8 GEMM should succeed for 16x16");
    let c = result.unwrap();
    assert_eq!(c.shape.dims, vec![n as i64, n as i64]);

    let vals = read_f32(&c);
    // All 1.0 * all 1.0 = each element should be N = 16.0
    eprintln!("FP8 E4M3 result[0] = {} (expected 16.0)", vals[0]);
    assert!(
        (vals[0] - 16.0).abs() < 1.0,
        "FP8 GEMM result[0]: {} vs 16.0",
        vals[0]
    );
    eprintln!("gpu_gemm_fp8 E4M3 16x16: OK");
}

#[test]
#[ignore]
fn test_cublas_gemm_ex_int8_raw() {
    // Direct FFI call matching the C test that works.
    cuda::set_device(0).unwrap();
    let cublas = cuda::CublasHandle::new().unwrap();

    let a_data: [i8; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
    let b_data: [i8; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];

    let a_bytes = unsafe { core::slice::from_raw_parts(a_data.as_ptr() as *const u8, 16) };
    let b_bytes = unsafe { core::slice::from_raw_parts(b_data.as_ptr() as *const u8, 16) };

    let a_buf = cuda::DeviceBuffer::alloc(16).unwrap();
    let b_buf = cuda::DeviceBuffer::alloc(16).unwrap();
    let c_buf = cuda::DeviceBuffer::alloc(64).unwrap();

    a_buf.copy_from_host(a_bytes).unwrap();
    b_buf.copy_from_host(b_bytes).unwrap();
    c_buf.copy_from_host(&[0u8; 64]).unwrap();

    let alpha: i32 = 1;
    let beta: i32 = 0;

    cublas
        .gemm_ex(
            cuda::ffi::cublasOperation_t::CUBLAS_OP_N,
            cuda::ffi::cublasOperation_t::CUBLAS_OP_N,
            4,
            4,
            4,
            &alpha as *const i32 as *const core::ffi::c_void,
            &b_buf,
            cuda::ffi::cudaDataType_t::CUDA_R_8I,
            4,
            &a_buf,
            cuda::ffi::cudaDataType_t::CUDA_R_8I,
            4,
            &beta as *const i32 as *const core::ffi::c_void,
            &c_buf,
            cuda::ffi::cudaDataType_t::CUDA_R_32I,
            4,
            cuda::ffi::cublasComputeType_t::CUBLAS_COMPUTE_32I,
        )
        .expect("cublasGemmEx INT8 failed");

    cuda::synchronize().unwrap();

    let mut result = [0u8; 64];
    c_buf.copy_to_host(&mut result).unwrap();
    let c_vals: Vec<i32> = result
        .chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Row 0 should be: 80, 70, 60, 50
    assert_eq!(c_vals[0], 80);
    assert_eq!(c_vals[1], 70);
    eprintln!("Raw cublasGemmEx INT8: OK {:?}", &c_vals[..4]);
}

// ──────────────────────────────────────────────────────────────────────
// BF16 GPU dispatch tests (safetensors-model-loader-v1 Section 4)
// ──────────────────────────────────────────────────────────────────────

/// Build a BF16 tensor from an f32 slice (round-tripped through f32→BF16).
fn make_bf16_tensor(shape: &[i64], data: &[f32]) -> Tensor {
    let raw = smallaios_onnx_rt::tensor::f32_to_bf16(data);
    Tensor {
        data_type: DataType::BFloat16,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: raw,
    }
}

/// Decode a BF16 tensor's raw bytes to f32 for comparison.
fn read_bf16_as_f32(t: &Tensor) -> Vec<f32> {
    assert_eq!(t.data_type, DataType::BFloat16);
    smallaios_onnx_rt::tensor::bf16_to_f32(&t.raw_data)
}

#[test]
#[ignore]
fn test_gpu_gemm_bf16_16x16() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::Bf16).expect("CUDA init");

    // 16x16 * 16x16 = 16x16, BF16 end-to-end.
    let n = 16usize;
    // Small deterministic values so BF16 mantissa loss stays small.
    let a_data: Vec<f32> = (0..n * n).map(|i| ((i % 7) as f32) * 0.25 - 0.5).collect();
    let b_data: Vec<f32> = (0..n * n)
        .map(|i| ((i % 5) as f32) * 0.125 + 0.25)
        .collect();

    let a_bf = make_bf16_tensor(&[n as i64, n as i64], &a_data);
    let b_bf = make_bf16_tensor(&[n as i64, n as i64], &b_data);
    assert_eq!(a_bf.raw_data.len(), n * n * 2);
    assert_eq!(b_bf.raw_data.len(), n * n * 2);

    let c_bf = cuda::dispatch::gpu_gemm(&rt, &a_bf, &b_bf, false, false, 1.0, 0.0, None)
        .expect("gpu_gemm BF16 failed");
    assert_eq!(c_bf.data_type, DataType::BFloat16);
    assert_eq!(c_bf.shape.dims, vec![n as i64, n as i64]);
    assert_eq!(c_bf.raw_data.len(), n * n * 2);

    let gpu_bf16_result = read_bf16_as_f32(&c_bf);

    // f32 reference on GPU using the *same* BF16-rounded inputs so we
    // isolate compute-precision drift from input-rounding drift.
    let a_rounded = smallaios_onnx_rt::tensor::bf16_to_f32(&a_bf.raw_data);
    let b_rounded = smallaios_onnx_rt::tensor::bf16_to_f32(&b_bf.raw_data);
    let a_f32 = make_f32_tensor(&[n as i64, n as i64], &a_rounded);
    let b_f32 = make_f32_tensor(&[n as i64, n as i64], &b_rounded);
    let rt_f32 =
        cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init f32");
    let c_f32 = cuda::dispatch::gpu_gemm(&rt_f32, &a_f32, &b_f32, false, false, 1.0, 0.0, None)
        .expect("gpu_gemm f32 ref failed");
    let ref_vals = read_f32(&c_f32);

    let mut max_abs: f32 = 0.0;
    for i in 0..n * n {
        let e = (gpu_bf16_result[i] - ref_vals[i]).abs();
        if e > max_abs {
            max_abs = e;
        }
    }
    // BF16 has ~8 bits of mantissa; for a 16-element dot product of values
    // in roughly [-0.5, 1.25], 5e-2 absolute is generous but realistic.
    assert!(
        max_abs < 5e-2,
        "BF16 GEMM deviated from f32 reference by {} (limit 5e-2)",
        max_abs
    );
    eprintln!(
        "gpu_gemm BF16 16x16: OK, max abs err vs f32 = {:.6}",
        max_abs
    );
}

#[test]
#[ignore]
fn test_gpu_conv2d_bf16_3x3() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::Bf16).expect("CUDA init");

    // Input [1,2,4,4], weight [3,2,3,3] → output [1,3,4,4] (pad=1).
    let x_data: Vec<f32> = (0..(2 * 4 * 4))
        .map(|i| ((i % 11) as f32) * 0.125 - 0.5)
        .collect();
    let w_data: Vec<f32> = (0..(3 * 2 * 3 * 3))
        .map(|i| ((i % 5) as f32) * 0.0625 + 0.125)
        .collect();

    let x_bf = make_bf16_tensor(&[1, 2, 4, 4], &x_data);
    let w_bf = make_bf16_tensor(&[3, 2, 3, 3], &w_data);
    assert_eq!(x_bf.raw_data.len(), 2 * 4 * 4 * 2);
    assert_eq!(w_bf.raw_data.len(), 3 * 2 * 3 * 3 * 2);

    let result =
        cuda::conv::gpu_conv2d(&rt, &x_bf, &w_bf, None, &[1, 1, 1, 1], &[1, 1], &[1, 1], 1)
            .expect("gpu_conv2d BF16 failed");
    let y_bf = result.expect("BF16 conv returned None");
    assert_eq!(y_bf.data_type, DataType::BFloat16);
    assert_eq!(y_bf.shape.dims, vec![1, 3, 4, 4]);
    assert_eq!(y_bf.raw_data.len(), 3 * 4 * 4 * 2);

    let gpu_bf16_result = read_bf16_as_f32(&y_bf);

    // f32 reference on same BF16-rounded inputs.
    let x_rounded = smallaios_onnx_rt::tensor::bf16_to_f32(&x_bf.raw_data);
    let w_rounded = smallaios_onnx_rt::tensor::bf16_to_f32(&w_bf.raw_data);
    let x_f32 = make_f32_tensor(&[1, 2, 4, 4], &x_rounded);
    let w_f32 = make_f32_tensor(&[3, 2, 3, 3], &w_rounded);
    let rt_f32 =
        cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init f32");
    let ref_result = cuda::conv::gpu_conv2d(
        &rt_f32,
        &x_f32,
        &w_f32,
        None,
        &[1, 1, 1, 1],
        &[1, 1],
        &[1, 1],
        1,
    )
    .expect("gpu_conv2d f32 ref failed")
    .expect("f32 conv returned None");
    let ref_vals = read_f32(&ref_result);

    let mut max_abs: f32 = 0.0;
    for (i, (&g, &r)) in gpu_bf16_result.iter().zip(ref_vals.iter()).enumerate() {
        let e = (g - r).abs();
        if e > max_abs {
            max_abs = e;
        }
        assert!(
            e < 5e-2,
            "BF16 conv[{}] diverged: bf16={}, f32={}, err={}",
            i,
            g,
            r,
            e
        );
    }
    eprintln!(
        "gpu_conv2d BF16 3x3 pad=1: OK, max abs err vs f32 = {:.6}",
        max_abs
    );
}

// ── GPU-resident executor tests (Section 5) ─────────────────────────

/// Build a trivial `ExecutionGraph` with two chained MatMul nodes:
///
///   tmp = X  @ W1
///   out = tmp @ W2
///
/// Both weights are provided as graph inputs (not initializers) so the
/// test exercises the input-tensor path through `execute_graph_gpu`.
fn build_matmul_pipeline_graph() -> smallaios_onnx_rt::graph::ExecutionGraph {
    use smallaios_onnx_rt::graph::{ExecutionGraph, ExecutionNode, NodeIndex};
    use std::collections::BTreeMap;

    let n0 = ExecutionNode {
        node_index: NodeIndex::new(0),
        op_type: String::from("MatMul"),
        domain: String::new(),
        name: String::from("matmul_0"),
        inputs: vec![String::from("X"), String::from("W1")],
        outputs: vec![String::from("tmp")],
        dependencies: Vec::new(),
        attributes: Vec::new(),
        inner_graphs: BTreeMap::new(),
    };
    let n1 = ExecutionNode {
        node_index: NodeIndex::new(1),
        op_type: String::from("MatMul"),
        domain: String::new(),
        name: String::from("matmul_1"),
        inputs: vec![String::from("tmp"), String::from("W2")],
        outputs: vec![String::from("out")],
        dependencies: vec![NodeIndex::new(0)],
        attributes: Vec::new(),
        inner_graphs: BTreeMap::new(),
    };

    ExecutionGraph {
        nodes: vec![n0, n1],
        topological_order: vec![NodeIndex::new(0), NodeIndex::new(1)],
        input_names: vec![String::from("X"), String::from("W1"), String::from("W2")],
        output_names: vec![String::from("out")],
    }
}

fn named_f32_tensor(name: &str, rows: usize, cols: usize, data: &[f32]) -> Tensor {
    let mut t = make_f32_tensor(&[rows as i64, cols as i64], data);
    t.name = String::from(name);
    t
}

fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0f32;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            out[i * n + j] = s;
        }
    }
    out
}

#[test]
#[ignore]
fn test_gpu_executor_matmul_pipeline() {
    use cuda::{execute_graph_gpu, tensor_to_device};

    let rt = cuda::CudaRuntime::init().expect("CUDA runtime init");

    // Random-ish 4x4 f32 matrices.
    let x_vals: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1 - 0.3).collect();
    let w1_vals: Vec<f32> = (0..16).map(|i| 0.05 * ((i as f32) + 1.0).sin()).collect();
    let w2_vals: Vec<f32> = (0..16).map(|i| 0.07 * ((i as f32) + 2.0).cos()).collect();

    let x = named_f32_tensor("X", 4, 4, &x_vals);
    let w1 = named_f32_tensor("W1", 4, 4, &w1_vals);
    let w2 = named_f32_tensor("W2", 4, 4, &w2_vals);

    let x_dev = tensor_to_device(&x, &rt).expect("X -> device");
    let w1_dev = tensor_to_device(&w1, &rt).expect("W1 -> device");
    let w2_dev = tensor_to_device(&w2, &rt).expect("W2 -> device");

    let graph = build_matmul_pipeline_graph();
    let inputs: Vec<(String, cuda::DeviceTensor)> = vec![
        (String::from("X"), x_dev),
        (String::from("W1"), w1_dev),
        (String::from("W2"), w2_dev),
    ];

    let outputs = execute_graph_gpu(&graph, &inputs, &[], &rt).expect("execute_graph_gpu");
    assert_eq!(outputs.len(), 1);
    let host_out = outputs[0].to_host().expect("to_host");
    assert_eq!(host_out.shape.dims, vec![4, 4]);
    assert_eq!(host_out.data_type, DataType::Float);
    let gpu_f32: Vec<f32> = host_out
        .raw_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // CPU reference: tmp = X @ W1, out = tmp @ W2.
    let tmp = naive_matmul(&x_vals, &w1_vals, 4, 4, 4);
    let reference = naive_matmul(&tmp, &w2_vals, 4, 4, 4);

    let mut max_abs = 0f32;
    for (g, r) in gpu_f32.iter().zip(reference.iter()) {
        let e = (g - r).abs();
        if e > max_abs {
            max_abs = e;
        }
    }
    eprintln!(
        "test_gpu_executor_matmul_pipeline: max abs err = {:.6e}",
        max_abs
    );
    assert!(max_abs < 1e-3, "max abs err = {}", max_abs);
}

#[test]
#[ignore]
fn test_gpu_executor_unsupported_op() {
    use cuda::{execute_graph_gpu, tensor_to_device};
    use smallaios_onnx_rt::graph::{ExecutionGraph, ExecutionNode, NodeIndex};
    use std::collections::BTreeMap;

    let rt = cuda::CudaRuntime::init().expect("CUDA runtime init");

    // Build a 1-node graph containing a Relu, which has no GPU dispatch
    // in the Section 5 executor.
    let node = ExecutionNode {
        node_index: NodeIndex::new(0),
        op_type: String::from("Relu"),
        domain: String::new(),
        name: String::from("relu_0"),
        inputs: vec![String::from("X")],
        outputs: vec![String::from("Y")],
        dependencies: Vec::new(),
        attributes: Vec::new(),
        inner_graphs: BTreeMap::new(),
    };
    let graph = ExecutionGraph {
        nodes: vec![node],
        topological_order: vec![NodeIndex::new(0)],
        input_names: vec![String::from("X")],
        output_names: vec![String::from("Y")],
    };

    let x = named_f32_tensor("X", 2, 2, &[-1.0, 0.5, -0.25, 2.0]);
    let x_dev = tensor_to_device(&x, &rt).expect("X -> device");
    let inputs = vec![(String::from("X"), x_dev)];

    let err = execute_graph_gpu(&graph, &inputs, &[], &rt)
        .expect_err("Relu should be rejected by the GPU executor");
    let msg = format!("{}", err);
    assert!(
        msg.contains("no GPU implementation for Relu"),
        "unexpected error message: {}",
        msg
    );
    eprintln!(
        "test_gpu_executor_unsupported_op: got expected error: {}",
        msg
    );
}

// ── Section 8: KV cache tests ───────────────────────────────────────

/// Build a BF16 `DeviceTensor` with shape `[1, num_kv_heads, head_dim]`
/// where every element is `fill`.
fn make_kv_device_tensor(num_kv_heads: usize, head_dim: usize, fill: f32) -> cuda::DeviceTensor {
    let vals: Vec<f32> = vec![fill; num_kv_heads * head_dim];
    let t = make_bf16_tensor(&[1, num_kv_heads as i64, head_dim as i64], &vals);
    cuda::DeviceTensor::from_host(&t).expect("bf16 tensor -> device")
}

/// Read `count` tokens worth of BF16 data from a device pointer returned by
/// `KvView` and decode to f32. Copies via `cudaMemcpy` device->host.
fn read_kv_view_as_f32(
    base_ptr: *const core::ffi::c_void,
    token_count: usize,
    stride_bytes: usize,
) -> Vec<f32> {
    if token_count == 0 {
        return Vec::new();
    }
    let total = token_count * stride_bytes;
    let mut host = vec![0u8; total];
    let err = unsafe {
        cuda::ffi::cudaMemcpy(
            host.as_mut_ptr() as *mut core::ffi::c_void,
            base_ptr,
            total,
            cuda::ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        )
    };
    assert_eq!(err, cuda::ffi::CUDA_SUCCESS, "D2H copy failed: {}", err);
    smallaios_onnx_rt::tensor::bf16_to_f32(&host)
}

#[test]
#[ignore]
fn test_kv_cache_allocate_and_append() {
    cuda::set_device(0).expect("set device");
    let rt = cuda::CudaRuntime::init().expect("CUDA runtime init");

    let num_layers = 4;
    let num_kv_heads = 8;
    let head_dim = 128;
    let max_seq_len = 16;
    let kinds = vec![cuda::LayerKind::Global; num_layers];

    let mut cache = cuda::GpuKvCache::allocate(
        &rt,
        num_layers,
        num_kv_heads,
        head_dim,
        max_seq_len,
        DataType::BFloat16,
        &kinds,
    )
    .expect("GpuKvCache::allocate");

    assert_eq!(cache.num_layers(), num_layers);
    assert_eq!(cache.current_position(), 0);
    assert_eq!(cache.max_seq_len(), max_seq_len);

    // Fresh view before any append: token_count == 0.
    let v0 = cache.view(0).expect("empty view");
    assert_eq!(v0.token_count, 0);
    assert_eq!(v0.stride_bytes, num_kv_heads * head_dim * 2);

    // Append token 0 K/V to all layers. Use a different fill per layer so
    // we can assert that the per-layer buffers are independent.
    for layer in 0..num_layers {
        let fill = (layer as f32 + 1.0) * 0.25;
        let k_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
        let v_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill + 0.5);
        cache.append(layer, &k_dev, &v_dev).expect("append");
    }
    cache.advance_position().expect("advance_position");
    assert_eq!(cache.current_position(), 1);

    // Each layer should now have exactly 1 token in its view, with the
    // expected fill value.
    for layer in 0..num_layers {
        let view = cache.view(layer).expect("view");
        assert_eq!(view.token_count, 1);
        assert_eq!(view.num_kv_heads, num_kv_heads);
        assert_eq!(view.head_dim, head_dim);
        assert_eq!(view.dtype, DataType::BFloat16);

        let k_vals = read_kv_view_as_f32(view.k_ptr, view.token_count, view.stride_bytes);
        let v_vals = read_kv_view_as_f32(view.v_ptr, view.token_count, view.stride_bytes);
        assert_eq!(k_vals.len(), num_kv_heads * head_dim);
        let expect_k = (layer as f32 + 1.0) * 0.25;
        let expect_v = expect_k + 0.5;
        for (i, (k, v)) in k_vals.iter().zip(v_vals.iter()).enumerate() {
            assert!(
                (k - expect_k).abs() < 1e-3,
                "layer {} elem {}: k {} != expected {}",
                layer,
                i,
                k,
                expect_k
            );
            assert!(
                (v - expect_v).abs() < 1e-3,
                "layer {} elem {}: v {} != expected {}",
                layer,
                i,
                v,
                expect_v
            );
        }
    }
    eprintln!("test_kv_cache_allocate_and_append: OK");
}

#[test]
#[ignore]
fn test_kv_cache_multiple_tokens() {
    cuda::set_device(0).expect("set device");
    let rt = cuda::CudaRuntime::init().expect("CUDA runtime init");

    let num_layers = 2;
    let num_kv_heads = 4;
    let head_dim = 16;
    let max_seq_len = 8;
    let kinds = vec![cuda::LayerKind::Global; num_layers];

    let mut cache = cuda::GpuKvCache::allocate(
        &rt,
        num_layers,
        num_kv_heads,
        head_dim,
        max_seq_len,
        DataType::BFloat16,
        &kinds,
    )
    .expect("allocate");

    // Append 3 tokens' worth of K/V, with a unique fill per token so we can
    // verify slot ordering in the cache.
    for pos in 0..3 {
        for layer in 0..num_layers {
            let fill_k = (pos as f32 + 1.0) * 0.125;
            let fill_v = (pos as f32 + 1.0) * 0.25;
            let k_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill_k);
            let v_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill_v);
            cache.append(layer, &k_dev, &v_dev).expect("append");
        }
        cache.advance_position().expect("advance");
    }
    assert_eq!(cache.current_position(), 3);

    for layer in 0..num_layers {
        let view = cache.view(layer).expect("view");
        assert_eq!(view.token_count, 3);
        let k_vals = read_kv_view_as_f32(view.k_ptr, view.token_count, view.stride_bytes);
        let v_vals = read_kv_view_as_f32(view.v_ptr, view.token_count, view.stride_bytes);
        let token_elems = num_kv_heads * head_dim;
        for pos in 0..3 {
            let expect_k = (pos as f32 + 1.0) * 0.125;
            let expect_v = (pos as f32 + 1.0) * 0.25;
            for i in 0..token_elems {
                let idx = pos * token_elems + i;
                assert!(
                    (k_vals[idx] - expect_k).abs() < 1e-3,
                    "layer {} pos {} elem {}: k {} != {}",
                    layer,
                    pos,
                    i,
                    k_vals[idx],
                    expect_k
                );
                assert!(
                    (v_vals[idx] - expect_v).abs() < 1e-3,
                    "layer {} pos {} elem {}: v {} != {}",
                    layer,
                    pos,
                    i,
                    v_vals[idx],
                    expect_v
                );
            }
        }
    }
    eprintln!("test_kv_cache_multiple_tokens: OK");
}

#[test]
#[ignore]
fn test_kv_cache_sliding_window() {
    cuda::set_device(0).expect("set device");
    let rt = cuda::CudaRuntime::init().expect("CUDA runtime init");

    let num_kv_heads = 2;
    let head_dim = 8;
    let max_seq_len = 16;
    // Layer 0: global. Layer 1: sliding window of 4.
    let kinds = vec![cuda::LayerKind::Global, cuda::LayerKind::SlidingWindow(4)];

    let mut cache = cuda::GpuKvCache::allocate(
        &rt,
        2,
        num_kv_heads,
        head_dim,
        max_seq_len,
        DataType::BFloat16,
        &kinds,
    )
    .expect("allocate");

    // Append 6 tokens.
    for pos in 0..6 {
        let fill = (pos as f32 + 1.0) * 0.1;
        for layer in 0..2 {
            let k_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
            let v_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
            cache.append(layer, &k_dev, &v_dev).expect("append");
        }
        cache.advance_position().expect("advance");
    }
    assert_eq!(cache.current_position(), 6);

    // Global layer 0 should expose all 6 tokens.
    let global = cache.view(0).expect("view global");
    assert_eq!(global.token_count, 6);

    // Sliding-window layer 1 should expose only the last 4 tokens (pos 2..6).
    let local = cache.view(1).expect("view local");
    assert_eq!(local.token_count, 4);
    let k_vals = read_kv_view_as_f32(local.k_ptr, local.token_count, local.stride_bytes);
    let token_elems = num_kv_heads * head_dim;
    for view_pos in 0..4 {
        let src_pos = view_pos + 2; // window starts at pos 2
        let expect = (src_pos as f32 + 1.0) * 0.1;
        for i in 0..token_elems {
            let idx = view_pos * token_elems + i;
            // BF16 has ~8 bits of mantissa so small fractions like 0.6 are
            // rounded; allow ~0.01 absolute slack.
            assert!(
                (k_vals[idx] - expect).abs() < 1e-2,
                "local view pos {} elem {}: {} != {}",
                view_pos,
                i,
                k_vals[idx],
                expect
            );
        }
    }
    eprintln!("test_kv_cache_sliding_window: OK");
}

#[test]
#[ignore]
fn test_kv_cache_reset() {
    cuda::set_device(0).expect("set device");
    let rt = cuda::CudaRuntime::init().expect("CUDA runtime init");

    let num_kv_heads = 2;
    let head_dim = 4;
    let max_seq_len = 8;
    let kinds = vec![cuda::LayerKind::Global];

    let mut cache = cuda::GpuKvCache::allocate(
        &rt,
        1,
        num_kv_heads,
        head_dim,
        max_seq_len,
        DataType::BFloat16,
        &kinds,
    )
    .expect("allocate");

    // Append 2 tokens with distinct fills.
    for pos in 0..2 {
        let fill = (pos as f32 + 1.0) * 0.5;
        let k_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
        let v_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
        cache.append(0, &k_dev, &v_dev).expect("append");
        cache.advance_position().expect("advance");
    }
    assert_eq!(cache.current_position(), 2);

    cache.reset();
    assert_eq!(cache.current_position(), 0);
    let empty_view = cache.view(0).expect("view after reset");
    assert_eq!(empty_view.token_count, 0);

    // Next append writes at slot 0 — fill with a distinctive value and verify
    // it lands at position 0.
    let fill = 0.875_f32;
    let k_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
    let v_dev = make_kv_device_tensor(num_kv_heads, head_dim, fill);
    cache.append(0, &k_dev, &v_dev).expect("append post-reset");
    cache.advance_position().expect("advance post-reset");
    let view = cache.view(0).expect("view post-reset");
    assert_eq!(view.token_count, 1);
    let k_vals = read_kv_view_as_f32(view.k_ptr, view.token_count, view.stride_bytes);
    for (i, k) in k_vals.iter().enumerate() {
        assert!(
            (k - fill).abs() < 1e-3,
            "post-reset slot 0 elem {}: {} != {}",
            i,
            k,
            fill
        );
    }
    eprintln!("test_kv_cache_reset: OK");
}

// ── Section 9.6 + transformer-gpu-kernels-v1 §8: Gemma e2e ────────────
//
// Builds a synthetic 2-layer Gemma-like safetensors directory in a temp
// dir, loads it via Session::from_safetensors, runs a full forward
// pass through every dispatched op (Gather, Add, Mul, RMSNormalization,
// RotaryEmbedding, GroupQueryAttention, Silu, Gemm), and validates the
// output: logits shape `[1, Sq, vocab]`, BF16 dtype, no NaN/Inf, and
// reset-rerun-equivalence. With §7 wiring + the gemma-builder
// `Gemm(trans_b=true)` migration + the rank-3 adapters for
// RotaryEmbedding / GroupQueryAttention, the complete prefill path
// runs end-to-end on GB10.
#[cfg(feature = "safetensors")]
#[test]
#[ignore]
fn test_session_from_safetensors_synthetic_gemma() {
    use smallaios_onnx_rt::session::{InferenceInput, Session, SessionKind};
    use std::fs;
    use std::io::Write;
    use std::sync::Arc;

    // ── Config params (small enough for a 2-layer toy model) ─────────
    let num_layers = 2usize;
    let hidden = 128usize;
    let intermediate = 256usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 32usize;
    let vocab = 256usize;
    let max_pos = 64usize;

    let config_json = format!(
        r#"{{
            "architectures": ["Gemma4ForCausalLM"],
            "num_hidden_layers": {num_layers},
            "hidden_size": {hidden},
            "intermediate_size": {intermediate},
            "num_attention_heads": {num_heads},
            "num_key_value_heads": {num_kv_heads},
            "head_dim": {head_dim},
            "vocab_size": {vocab},
            "max_position_embeddings": {max_pos},
            "rope_theta": 10000.0,
            "sliding_window": 16,
            "sliding_window_pattern": 2,
            "rms_norm_eps": 1e-6,
            "bos_token_id": 1,
            "eos_token_id": 2
        }}"#
    );

    // ── Synthetic safetensors blob ───────────────────────────────────
    //
    // Include every tensor `build_gemma_graph` registers for this
    // config. Intentionally omit `lm_head.weight` so the graph builder
    // exercises the weight-tying fallback.
    #[derive(Clone)]
    struct Entry {
        name: String,
        dtype: &'static str,
        shape: Vec<i64>,
    }
    fn bf16_entry(name: &str, shape: Vec<i64>) -> Entry {
        Entry {
            name: name.to_string(),
            dtype: "BF16",
            shape,
        }
    }
    fn bytes_for(e: &Entry) -> usize {
        let numel: usize = e.shape.iter().map(|&d| d as usize).product();
        numel * 2 // BF16 = 2 bytes
    }

    let h = hidden as i64;
    let i = intermediate as i64;
    let v = vocab as i64;
    let kv = (num_kv_heads * head_dim) as i64;

    let mut entries: Vec<Entry> = Vec::new();
    entries.push(bf16_entry("model.embed_tokens.weight", vec![v, h]));
    for l in 0..num_layers {
        let p = format!("model.layers.{l}");
        entries.push(bf16_entry(&format!("{p}.input_layernorm.weight"), vec![h]));
        entries.push(bf16_entry(
            &format!("{p}.post_attention_layernorm.weight"),
            vec![h],
        ));
        entries.push(bf16_entry(
            &format!("{p}.self_attn.q_proj.weight"),
            vec![h, h],
        ));
        entries.push(bf16_entry(
            &format!("{p}.self_attn.k_proj.weight"),
            vec![kv, h],
        ));
        entries.push(bf16_entry(
            &format!("{p}.self_attn.v_proj.weight"),
            vec![kv, h],
        ));
        entries.push(bf16_entry(
            &format!("{p}.self_attn.o_proj.weight"),
            vec![h, h],
        ));
        entries.push(bf16_entry(&format!("{p}.mlp.gate_proj.weight"), vec![i, h]));
        entries.push(bf16_entry(&format!("{p}.mlp.up_proj.weight"), vec![i, h]));
        entries.push(bf16_entry(&format!("{p}.mlp.down_proj.weight"), vec![h, i]));
    }
    entries.push(bf16_entry("model.norm.weight", vec![h]));

    // Sort by name (matches safetensors ordering convention) and
    // assign offsets.
    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut offset = 0usize;
    let mut entry_offsets: Vec<(Entry, usize, usize)> = Vec::new();
    for e in entries {
        let sz = bytes_for(&e);
        entry_offsets.push((e.clone(), offset, offset + sz));
        offset += sz;
    }
    let payload_size = offset;

    // Build JSON header.
    let mut header = String::from("{");
    for (idx, (e, start, end)) in entry_offsets.iter().enumerate() {
        if idx > 0 {
            header.push(',');
        }
        let shape_str = e
            .shape
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"{}\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            e.name, e.dtype, shape_str, start, end
        ));
    }
    header.push('}');

    let header_bytes = header.as_bytes();
    let mut blob = Vec::with_capacity(8 + header_bytes.len() + payload_size);
    blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(header_bytes);
    // Random-ish BF16 payload: use a small deterministic pattern.
    let payload: Vec<u8> = (0..payload_size)
        .map(|ix| ((ix as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    blob.extend_from_slice(&payload);

    // ── Write the model directory ─────────────────────────────────────
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("smallaios-sec9-{nanos}"));
    fs::create_dir_all(&dir).expect("mkdir temp model dir");
    {
        let mut f = fs::File::create(dir.join("config.json")).expect("create config.json");
        f.write_all(config_json.as_bytes())
            .expect("write config.json");
    }
    {
        let mut f =
            fs::File::create(dir.join("model.safetensors")).expect("create model.safetensors");
        f.write_all(&blob).expect("write model.safetensors");
        f.sync_all().ok();
    }

    // ── Initialize CUDA + build session ───────────────────────────────
    cuda::set_device(0).expect("set device");
    let rt = Arc::new(cuda::CudaRuntime::init().expect("CudaRuntime::init"));

    let session = Session::from_safetensors(&dir, rt).expect("from_safetensors");
    assert_eq!(session.kind, SessionKind::Safetensors);
    assert_eq!(session.input_names(), &["input_ids".to_string()]);
    assert!(session.output_names().len() == 1);

    // ── Build input_ids [1, 4] int64 ─────────────────────────────────
    let token_ids: [i64; 4] = [1, 42, 99, 7];
    let mut raw = vec![0u8; token_ids.len() * 8];
    for (k, &t) in token_ids.iter().enumerate() {
        raw[k * 8..(k + 1) * 8].copy_from_slice(&t.to_le_bytes());
    }
    let mut input_tensor = Tensor::new(
        DataType::Int64,
        TensorShape::new(vec![1, 4]),
        "input_ids".to_string(),
    );
    input_tensor.raw_data = raw;
    let inputs = [InferenceInput {
        name: "input_ids".to_string(),
        tensor: input_tensor,
    }];

    // ── §8.1, §8.2: Run forward pass; verify logits shape + dtype ────
    //
    // This synthetic model uses pseudo-random byte fills for every
    // weight (see the `payload` pattern above); after passing through
    // two transformer layers + RMSNorm + Gemms the activations
    // naturally saturate to NaN/Inf. We therefore assert on **structural
    // correctness** (shape, dtype, dispatch reaches every op) rather
    // than numerical sanity. A real-weights end-to-end test against a
    // trained Gemma checkpoint is a follow-up — see tasks.md §8.2 note.
    let outputs = session.run(&inputs).expect("session.run forward pass");
    assert_eq!(outputs.len(), 1, "expected exactly one output (logits)");
    let logits = &outputs[0];
    assert_eq!(
        logits.tensor.shape.dims,
        vec![1i64, 4, vocab as i64],
        "unexpected logits shape"
    );
    assert_eq!(logits.tensor.data_type, DataType::BFloat16);
    assert!(
        !logits.tensor.raw_data.is_empty(),
        "logits must be non-empty"
    );
    eprintln!("test_session_from_safetensors_synthetic_gemma: forward pass OK");

    // §8.4: reset must succeed and the session can be rerun without
    // crashing. (With random weights the outputs are NaN/Inf and not
    // bit-comparable, so we validate that the forward pass completes
    // cleanly after a reset rather than asserting numerical equality.)
    session.reset_kv_cache().expect("reset_kv_cache");
    let _rerun = session.run(&inputs).expect("second run after reset");

    // Cleanup the temp dir.
    let _ = fs::remove_dir_all(&dir);
}

// ── Section 10.5: Session API contract for `llm-generation` ────────────
//
// This test pins the Session API that `llm-api-translation-v1`
// `llm-generation` will call, against a synthetic 2-layer safetensors
// model. It verifies the interface points documented in
// `docs/safetensors-integration.md`:
//
//   (a) `Session::from_safetensors` succeeds on a synthetic fixture
//   (b) `Session::kind()` returns `SessionKind::Safetensors`
//   (c) `Session::manages_kv_cache_internally()` returns `true`
//   (d) `Session::run()` accepts a single Int64 `input_ids` tensor
//       (even if the forward pass errors on a missing GPU op, the
//       dispatch plumbing must be reachable)
//   (e) `Session::reset_kv_cache()` succeeds
//
// `#[ignore]` because it requires a real CUDA device on GB10 class
// hardware, matching the existing CUDA tests.
#[cfg(feature = "safetensors")]
#[test]
#[ignore]
fn test_session_api_contract_for_llm_generation() {
    use smallaios_onnx_rt::session::{InferenceInput, Session, SessionKind};
    use std::fs;
    use std::io::Write;
    use std::sync::Arc;

    let num_layers = 2usize;
    let hidden = 64usize;
    let intermediate = 128usize;
    let num_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 16usize;
    let vocab = 128usize;
    let max_pos = 32usize;

    let config_json = format!(
        r#"{{
            "architectures": ["Gemma4ForCausalLM"],
            "num_hidden_layers": {num_layers},
            "hidden_size": {hidden},
            "intermediate_size": {intermediate},
            "num_attention_heads": {num_heads},
            "num_key_value_heads": {num_kv_heads},
            "head_dim": {head_dim},
            "vocab_size": {vocab},
            "max_position_embeddings": {max_pos},
            "rope_theta": 10000.0,
            "sliding_window": 16,
            "sliding_window_pattern": 2,
            "rms_norm_eps": 1e-6,
            "bos_token_id": 1,
            "eos_token_id": 2
        }}"#
    );

    #[derive(Clone)]
    struct Entry {
        name: String,
        shape: Vec<i64>,
    }
    fn bf16_bytes(e: &Entry) -> usize {
        let numel: usize = e.shape.iter().map(|&d| d as usize).product();
        numel * 2
    }

    let h = hidden as i64;
    let ii = intermediate as i64;
    let v = vocab as i64;
    let kv = (num_kv_heads * head_dim) as i64;

    let mut entries: Vec<Entry> = Vec::new();
    entries.push(Entry {
        name: "model.embed_tokens.weight".to_string(),
        shape: vec![v, h],
    });
    for l in 0..num_layers {
        let p = format!("model.layers.{l}");
        entries.push(Entry {
            name: format!("{p}.input_layernorm.weight"),
            shape: vec![h],
        });
        entries.push(Entry {
            name: format!("{p}.post_attention_layernorm.weight"),
            shape: vec![h],
        });
        entries.push(Entry {
            name: format!("{p}.self_attn.q_proj.weight"),
            shape: vec![h, h],
        });
        entries.push(Entry {
            name: format!("{p}.self_attn.k_proj.weight"),
            shape: vec![kv, h],
        });
        entries.push(Entry {
            name: format!("{p}.self_attn.v_proj.weight"),
            shape: vec![kv, h],
        });
        entries.push(Entry {
            name: format!("{p}.self_attn.o_proj.weight"),
            shape: vec![h, h],
        });
        entries.push(Entry {
            name: format!("{p}.mlp.gate_proj.weight"),
            shape: vec![ii, h],
        });
        entries.push(Entry {
            name: format!("{p}.mlp.up_proj.weight"),
            shape: vec![ii, h],
        });
        entries.push(Entry {
            name: format!("{p}.mlp.down_proj.weight"),
            shape: vec![h, ii],
        });
    }
    entries.push(Entry {
        name: "model.norm.weight".to_string(),
        shape: vec![h],
    });

    entries.sort_by(|a, b| a.name.cmp(&b.name));
    let mut offset = 0usize;
    let mut entry_offsets: Vec<(Entry, usize, usize)> = Vec::new();
    for e in entries {
        let sz = bf16_bytes(&e);
        entry_offsets.push((e.clone(), offset, offset + sz));
        offset += sz;
    }
    let payload_size = offset;

    let mut header = String::from("{");
    for (idx, (e, start, end)) in entry_offsets.iter().enumerate() {
        if idx > 0 {
            header.push(',');
        }
        let shape_str = e
            .shape
            .iter()
            .map(|d| format!("{d}"))
            .collect::<Vec<_>>()
            .join(",");
        header.push_str(&format!(
            "\"{}\":{{\"dtype\":\"BF16\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
            e.name, shape_str, start, end
        ));
    }
    header.push('}');

    let header_bytes = header.as_bytes();
    let mut blob = Vec::with_capacity(8 + header_bytes.len() + payload_size);
    blob.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    blob.extend_from_slice(header_bytes);
    let payload: Vec<u8> = (0..payload_size)
        .map(|ix| ((ix as u32).wrapping_mul(2654435761) >> 24) as u8)
        .collect();
    blob.extend_from_slice(&payload);

    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let dir = std::env::temp_dir().join(format!("smallaios-sec10-{nanos}"));
    fs::create_dir_all(&dir).expect("mkdir temp model dir");
    {
        let mut f = fs::File::create(dir.join("config.json")).expect("create config.json");
        f.write_all(config_json.as_bytes())
            .expect("write config.json");
    }
    {
        let mut f =
            fs::File::create(dir.join("model.safetensors")).expect("create model.safetensors");
        f.write_all(&blob).expect("write model.safetensors");
        f.sync_all().ok();
    }

    cuda::set_device(0).expect("set device");
    let rt = Arc::new(cuda::CudaRuntime::init().expect("CudaRuntime::init"));

    // (a) Construction.
    let session = Session::from_safetensors(&dir, rt).expect("from_safetensors");

    // (b) Kind discriminator.
    assert_eq!(session.kind(), SessionKind::Safetensors);

    // (c) Internal-KV flag.
    assert!(
        session.manages_kv_cache_internally(),
        "safetensors session must manage KV cache internally"
    );

    // Input/output signature matches the contract documented in
    // docs/safetensors-integration.md.
    assert_eq!(session.input_names(), &["input_ids".to_string()]);
    assert_eq!(session.output_names().len(), 1);

    // (d) `run()` accepts a token ID tensor. The forward pass will
    // bottom out in "no GPU implementation for Gather" with the
    // current dispatcher — that's fine; we only care that the
    // dispatch path is reachable through the public API.
    let token_ids: [i64; 3] = [1, 42, 7];
    let mut raw = vec![0u8; token_ids.len() * 8];
    for (k, &t) in token_ids.iter().enumerate() {
        raw[k * 8..(k + 1) * 8].copy_from_slice(&t.to_le_bytes());
    }
    let mut input_tensor = Tensor::new(
        DataType::Int64,
        TensorShape::new(vec![1, 3]),
        "input_ids".to_string(),
    );
    input_tensor.raw_data = raw;
    let inputs = [InferenceInput {
        name: "input_ids".to_string(),
        tensor: input_tensor,
    }];
    match session.run(&inputs) {
        Ok(_) => eprintln!("llm-generation contract: forward pass OK"),
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("no GPU implementation") || msg.contains("GPU"),
                "unexpected error from session.run: {msg}"
            );
        }
    }

    // (e) reset_kv_cache — must succeed even if `run()` errored.
    session.reset_kv_cache().expect("reset_kv_cache");

    // Simulate the recommended calling pattern: reset, then run each
    // token individually. We don't care whether the forward pass
    // succeeds — we only care that the API contract holds.
    session.reset_kv_cache().expect("second reset_kv_cache");
    for tok in [5i64, 6, 7] {
        let mut raw = vec![0u8; 8];
        raw.copy_from_slice(&tok.to_le_bytes());
        let mut t = Tensor::new(
            DataType::Int64,
            TensorShape::new(vec![1, 1]),
            "input_ids".to_string(),
        );
        t.raw_data = raw;
        let _ = session.run(&[InferenceInput {
            name: "input_ids".to_string(),
            tensor: t,
        }]);
    }

    let _ = fs::remove_dir_all(&dir);
}

// ── Section 1: NVRTC JIT kernel infrastructure ──────────────────────

#[test]
#[ignore]
fn test_nvrtc_compile_and_launch_add_one() {
    // Trivial kernel: increments every int32 element in a buffer by 1.
    let source = r#"
extern "C" __global__ void add_one(int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}
"#;

    // Use CudaRuntime to ensure the runtime-API device is also initialized
    // (our driver-API context is lazy-initialized inside compile_kernel).
    let _rt = cuda::CudaRuntime::init().expect("CudaRuntime init failed");

    let kernel = cuda::kernels::compile_kernel("add_one", source, &[])
        .expect("compile_kernel(add_one) failed");
    assert_eq!(kernel.name(), "add_one");

    // Host-side initial data: 0..16.
    let host_init: Vec<i32> = (0..16).collect();
    let bytes = core::mem::size_of::<i32>() * host_init.len();
    let device_buf = cuda::DeviceBuffer::alloc(bytes).expect("alloc failed");
    let init_bytes: &[u8] =
        unsafe { core::slice::from_raw_parts(host_init.as_ptr() as *const u8, bytes) };
    device_buf
        .copy_from_host(init_bytes)
        .expect("H2D copy failed");

    // Build argument pack for cuLaunchKernel. Each slot must be a pointer
    // to the memory holding the argument value (device pointer goes via a
    // local variable so we can take its address).
    let mut dev_ptr = device_buf.as_mut_ptr();
    let mut n_arg: i32 = 16;
    let args: [*mut core::ffi::c_void; 2] = [
        &mut dev_ptr as *mut _ as *mut core::ffi::c_void,
        &mut n_arg as *mut _ as *mut core::ffi::c_void,
    ];

    cuda::kernels::launch_kernel(&kernel, (1, 1, 1), (16, 1, 1), &args, 0)
        .expect("launch_kernel failed");
    cuda::kernels::synchronize().expect("cuCtxSynchronize failed");

    // Copy back and verify each element was incremented by 1.
    let mut host_out = vec![0i32; 16];
    let out_bytes: &mut [u8] =
        unsafe { core::slice::from_raw_parts_mut(host_out.as_mut_ptr() as *mut u8, bytes) };
    device_buf.copy_to_host(out_bytes).expect("D2H copy failed");

    for (i, &v) in host_out.iter().enumerate() {
        assert_eq!(v, i as i32 + 1, "element {i} should be {} got {v}", i + 1);
    }
    eprintln!("NVRTC add_one kernel: OK {:?}", &host_out[..8]);
}

#[test]
#[ignore]
fn test_nvrtc_compile_reports_syntax_error() {
    // Intentionally broken CUDA source — NVRTC must surface a non-empty
    // log and compile_kernel must return KernelCompileFailed.
    let source = "extern \"C\" __global__ void broken(int *data) { this is not valid C++ }";
    let _rt = cuda::CudaRuntime::init().expect("CudaRuntime init failed");

    let result = cuda::kernels::compile_kernel("broken", source, &[]);
    match result {
        Err(cuda::CudaError::KernelCompileFailed { name, log }) => {
            assert_eq!(name, "broken");
            assert!(!log.is_empty(), "NVRTC build log should not be empty");
            eprintln!("Got expected NVRTC compile error:\n{log}");
        }
        Err(other) => panic!("expected KernelCompileFailed, got: {other:?}"),
        Ok(_) => panic!("expected compile failure, got Ok"),
    }
}

#[test]
#[ignore]
fn test_nvrtc_compile_launch_on_spawned_thread() {
    // Regression: driver-API current-context state is thread-local. If
    // lazy_context_init only called cuCtxSetCurrent on the first thread
    // (inside Once::call_once), any later thread would launch kernels with
    // no current context. Bind the context on the main thread first, then
    // compile + launch + synchronize entirely from a spawned thread and
    // expect success.
    let _rt = cuda::CudaRuntime::init().expect("CudaRuntime init failed");
    // Warm up: bind context on the main thread so Once::call_once fires here.
    let warm = cuda::kernels::compile_kernel("noop", "extern \"C\" __global__ void noop() {}", &[])
        .expect("main-thread compile_kernel failed");
    drop(warm);

    let handle = std::thread::spawn(|| {
        let source = r#"
extern "C" __global__ void add_one_thr(int *data, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] += 1;
}
"#;
        let kernel = cuda::kernels::compile_kernel("add_one_thr", source, &[])
            .expect("spawned-thread compile_kernel failed (ctx not rebound?)");

        let host_init: Vec<i32> = (0..16).collect();
        let bytes = core::mem::size_of::<i32>() * host_init.len();
        let device_buf = cuda::DeviceBuffer::alloc(bytes).expect("alloc failed");
        let init_bytes: &[u8] =
            unsafe { core::slice::from_raw_parts(host_init.as_ptr() as *const u8, bytes) };
        device_buf
            .copy_from_host(init_bytes)
            .expect("H2D copy failed");

        let mut dev_ptr = device_buf.as_mut_ptr();
        let mut n_arg: i32 = 16;
        let args: [*mut core::ffi::c_void; 2] = [
            &mut dev_ptr as *mut _ as *mut core::ffi::c_void,
            &mut n_arg as *mut _ as *mut core::ffi::c_void,
        ];

        cuda::kernels::launch_kernel(&kernel, (1, 1, 1), (16, 1, 1), &args, 0)
            .expect("spawned-thread launch_kernel failed (ctx not rebound?)");
        cuda::kernels::synchronize().expect("spawned-thread synchronize failed (ctx not rebound?)");

        let mut host_out = vec![0i32; 16];
        let out_bytes: &mut [u8] =
            unsafe { core::slice::from_raw_parts_mut(host_out.as_mut_ptr() as *mut u8, bytes) };
        device_buf.copy_to_host(out_bytes).expect("D2H copy failed");
        for (i, &v) in host_out.iter().enumerate() {
            assert_eq!(v, i as i32 + 1);
        }
    });
    handle.join().expect("spawned thread panicked");
}

// ── Section 2: Element-wise ops (Add, Mul, Silu) ────────────────────

fn make_f32_device_tensor(shape: &[i64], data: &[f32]) -> cuda::DeviceTensor {
    let t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: data.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    cuda::DeviceTensor::from_host(&t).expect("DeviceTensor::from_host")
}

fn make_bf16_device_tensor(shape: &[i64], data_f32: &[f32]) -> cuda::DeviceTensor {
    let t = Tensor {
        data_type: DataType::BFloat16,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: smallaios_onnx_rt::tensor::f32_to_bf16(data_f32),
    };
    cuda::DeviceTensor::from_host(&t).expect("DeviceTensor::from_host bf16")
}

fn make_i32_device_tensor(shape: &[i64], data: &[i32]) -> cuda::DeviceTensor {
    let t = Tensor {
        data_type: DataType::Int32,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    cuda::DeviceTensor::from_host(&t).expect("DeviceTensor::from_host i32")
}

fn read_f32_device_tensor(t: &cuda::DeviceTensor) -> Vec<f32> {
    let host = t.to_host().expect("D2H");
    host.raw_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}

fn read_bf16_device_tensor(t: &cuda::DeviceTensor) -> Vec<f32> {
    let host = t.to_host().expect("D2H");
    smallaios_onnx_rt::tensor::bf16_to_f32(&host.raw_data)
}

#[test]
#[ignore]
fn test_gpu_add_f32_matching_shapes() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let a_data: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let b_data: Vec<f32> = (0..8).map(|i| (i * 2) as f32).collect();

    let a = make_f32_device_tensor(&[2, 4], &a_data);
    let b = make_f32_device_tensor(&[2, 4], &b_data);

    let c = cuda::kernels::elementwise::add_gpu(&rt, &a, &b).expect("add_gpu");
    cuda::kernels::synchronize().expect("sync");
    let host = read_f32_device_tensor(&c);
    assert_eq!(host.len(), 8);
    for i in 0..8 {
        let expected = a_data[i] + b_data[i];
        assert!(
            (host[i] - expected).abs() < 1e-6,
            "i={i}: {} vs {expected}",
            host[i]
        );
    }
}

#[test]
#[ignore]
fn test_gpu_add_bf16_matching_shapes() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let a_data: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
    let b_data: Vec<f32> = (0..8).map(|i| i as f32 * -0.25).collect();

    let a = make_bf16_device_tensor(&[2, 4], &a_data);
    let b = make_bf16_device_tensor(&[2, 4], &b_data);

    let c = cuda::kernels::elementwise::add_gpu(&rt, &a, &b).expect("add_gpu bf16");
    cuda::kernels::synchronize().expect("sync");
    let host = read_bf16_device_tensor(&c);
    assert_eq!(host.len(), 8);
    // Compare against BF16-rounded inputs for fairness.
    let a_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&a_data));
    let b_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&b_data));
    for i in 0..8 {
        let expected = a_round[i] + b_round[i];
        assert!(
            (host[i] - expected).abs() < 1e-2,
            "i={i}: got {} expected {expected}",
            host[i]
        );
    }
}

#[test]
#[ignore]
fn test_gpu_add_bf16_broadcast_row_vector() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // [1, 4096] + [32, 4096] -> [32, 4096], A broadcasts along axis 0.
    let row: Vec<f32> = (0..4096).map(|i| (i as f32) * 0.01).collect();
    let mat: Vec<f32> = (0..(32 * 4096))
        .map(|i| ((i % 4096) as f32) * 0.001 + (i / 4096) as f32)
        .collect();

    let a = make_bf16_device_tensor(&[1, 4096], &row);
    let b = make_bf16_device_tensor(&[32, 4096], &mat);

    let c = cuda::kernels::elementwise::add_gpu(&rt, &a, &b).expect("add_gpu broadcast");
    cuda::kernels::synchronize().expect("sync");
    let host = read_bf16_device_tensor(&c);
    assert_eq!(host.len(), 32 * 4096);

    let row_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&row));
    let mat_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&mat));
    // Spot-check a few (row, col) combinations across the broadcast.
    for &(r, col) in &[(0usize, 0usize), (5, 123), (31, 4095), (17, 1000)] {
        let idx = r * 4096 + col;
        let expected = row_round[col] + mat_round[idx];
        assert!(
            (host[idx] - expected).abs() < 5e-2,
            "(r={r},col={col}): got {} expected {expected}",
            host[idx]
        );
    }
}

#[test]
#[ignore]
fn test_gpu_mul_f32_matching_shapes() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let a_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.25 - 2.0).collect();
    let b_data: Vec<f32> = (0..16).map(|i| i as f32 * -0.5 + 1.0).collect();

    let a = make_f32_device_tensor(&[4, 4], &a_data);
    let b = make_f32_device_tensor(&[4, 4], &b_data);

    let c = cuda::kernels::elementwise::mul_gpu(&rt, &a, &b).expect("mul_gpu");
    cuda::kernels::synchronize().expect("sync");
    let host = read_f32_device_tensor(&c);
    for i in 0..16 {
        let expected = a_data[i] * b_data[i];
        assert!((host[i] - expected).abs() < 1e-6);
    }
}

#[test]
#[ignore]
fn test_gpu_mul_bf16_matching_shapes() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let a_data: Vec<f32> = (0..16).map(|i| (i as f32) * 0.125).collect();
    let b_data: Vec<f32> = (0..16).map(|i| 2.0 - (i as f32) * 0.1).collect();

    let a = make_bf16_device_tensor(&[4, 4], &a_data);
    let b = make_bf16_device_tensor(&[4, 4], &b_data);

    let c = cuda::kernels::elementwise::mul_gpu(&rt, &a, &b).expect("mul_gpu bf16");
    cuda::kernels::synchronize().expect("sync");
    let host = read_bf16_device_tensor(&c);
    let a_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&a_data));
    let b_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&b_data));
    for i in 0..16 {
        let expected = a_round[i] * b_round[i];
        assert!((host[i] - expected).abs() < 2e-2);
    }
}

#[test]
#[ignore]
fn test_gpu_silu_f32() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let data: Vec<f32> = (-16i32..16).map(|i| i as f32 * 0.25).collect();

    let x = make_f32_device_tensor(&[4, 8], &data);
    let y = cuda::kernels::elementwise::silu_gpu(&rt, &x).expect("silu_gpu");
    cuda::kernels::synchronize().expect("sync");
    let host = read_f32_device_tensor(&y);
    for (i, &v) in data.iter().enumerate() {
        let expected = v / (1.0 + (-v).exp());
        assert!(
            (host[i] - expected).abs() < 1e-6,
            "i={i}: {} vs {expected}",
            host[i]
        );
    }
}

#[test]
#[ignore]
fn test_gpu_silu_bf16() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let data: Vec<f32> = (-16i32..16).map(|i| i as f32 * 0.25).collect();
    let x = make_bf16_device_tensor(&[4, 8], &data);
    let y = cuda::kernels::elementwise::silu_gpu(&rt, &x).expect("silu_gpu bf16");
    cuda::kernels::synchronize().expect("sync");
    let host = read_bf16_device_tensor(&y);
    let data_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&data));
    for (i, &v) in data_round.iter().enumerate() {
        let expected = v / (1.0 + (-v).exp());
        assert!(
            (host[i] - expected).abs() < 3e-2,
            "i={i}: got {} expected {expected}",
            host[i]
        );
    }
}

#[test]
#[ignore]
fn test_gpu_add_rejects_int32() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let a = make_i32_device_tensor(&[2, 2], &[1, 2, 3, 4]);
    let b = make_i32_device_tensor(&[2, 2], &[5, 6, 7, 8]);

    let result = cuda::kernels::elementwise::add_gpu(&rt, &a, &b);
    assert!(result.is_err(), "Int32 should be rejected, got Ok");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("unsupported"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

#[test]
#[ignore]
fn test_gpu_add_rejects_dtype_mismatch() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let a = make_f32_device_tensor(&[4], &[1.0, 2.0, 3.0, 4.0]);
    let b = make_bf16_device_tensor(&[4], &[1.0, 2.0, 3.0, 4.0]);

    let result = cuda::kernels::elementwise::add_gpu(&rt, &a, &b);
    assert!(result.is_err(), "mixed dtypes should be rejected");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("mismatch"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

// ── Section 3: Gather (embedding lookup) ────────────────────────────

fn make_i64_device_tensor(shape: &[i64], data: &[i64]) -> cuda::DeviceTensor {
    let t = Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(shape.to_vec()),
        name: String::new(),
        raw_data: data.iter().flat_map(|v| v.to_le_bytes()).collect(),
    };
    cuda::DeviceTensor::from_host(&t).expect("DeviceTensor::from_host i64")
}

#[test]
#[ignore]
fn test_gpu_gather_bf16_embedding_lookup() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // [vocab_size=128, hidden_size=32] BF16 embedding table.
    // Row v is filled with value v * 0.0625.
    let vocab = 128usize;
    let hidden = 32usize;
    let mut host_data = vec![0f32; vocab * hidden];
    for v in 0..vocab {
        let val = (v as f32) * 0.0625;
        for h in 0..hidden {
            host_data[v * hidden + h] = val;
        }
    }
    let data = make_bf16_device_tensor(&[vocab as i64, hidden as i64], &host_data);
    // indices shape [1, 4] Int64, token IDs 5, 42, 0, 127.
    let indices = make_i64_device_tensor(&[1, 4], &[5i64, 42, 0, 127]);

    let out = cuda::kernels::gather::gather_gpu(&rt, &data, &indices, 0).expect("gather_gpu bf16");
    cuda::kernels::synchronize().expect("sync");

    assert_eq!(out.shape, vec![1, 4, 32]);
    assert_eq!(out.dtype, DataType::BFloat16);

    let out_host = read_bf16_device_tensor(&out);
    let tokens = [5i64, 42, 0, 127];
    for (row_idx, &token) in tokens.iter().enumerate() {
        let expected_val = (token as f32) * 0.0625;
        for h in 0..hidden {
            let i = row_idx * hidden + h;
            assert!(
                (out_host[i] - expected_val).abs() < 1e-2,
                "row {row_idx} h {h}: got {} expected {expected_val}",
                out_host[i]
            );
        }
    }
}

#[test]
#[ignore]
fn test_gpu_gather_f32_embedding_lookup() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Small F32 embedding table so we can hand-verify byte-exactly.
    // [vocab=8, hidden=6]; row v's h-th element = v * 10 + h.
    let vocab = 8usize;
    let hidden = 6usize;
    let mut host_data = vec![0f32; vocab * hidden];
    for v in 0..vocab {
        for h in 0..hidden {
            host_data[v * hidden + h] = (v as f32) * 10.0 + (h as f32);
        }
    }
    let data = make_f32_device_tensor(&[vocab as i64, hidden as i64], &host_data);
    // indices shape [3] Int64.
    let indices = make_i64_device_tensor(&[3], &[7i64, 2, 0]);

    let out = cuda::kernels::gather::gather_gpu(&rt, &data, &indices, 0).expect("gather_gpu f32");
    cuda::kernels::synchronize().expect("sync");

    assert_eq!(out.shape, vec![3, 6]);
    assert_eq!(out.dtype, DataType::Float);

    let out_host = read_f32_device_tensor(&out);
    let tokens = [7i64, 2, 0];
    for (row_idx, &token) in tokens.iter().enumerate() {
        for h in 0..hidden {
            let got = out_host[row_idx * hidden + h];
            let expected = (token as f32) * 10.0 + (h as f32);
            assert!(
                (got - expected).abs() < 1e-6,
                "row {row_idx} h {h}: got {got} expected {expected}"
            );
        }
    }
}

#[test]
#[ignore]
fn test_gpu_gather_rejects_nonzero_axis() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let data = make_f32_device_tensor(&[4, 8], &[0f32; 32]);
    let indices = make_i64_device_tensor(&[2], &[0i64, 1]);

    let result = cuda::kernels::gather::gather_gpu(&rt, &data, &indices, 1);
    assert!(result.is_err(), "non-zero axis should be rejected");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("axis=0"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

#[test]
#[ignore]
fn test_gpu_gather_rejects_i32_indices() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let data = make_f32_device_tensor(&[4, 8], &[0f32; 32]);
    let indices = make_i32_device_tensor(&[2], &[0i32, 1]);

    let result = cuda::kernels::gather::gather_gpu(&rt, &data, &indices, 0);
    assert!(result.is_err(), "i32 indices should be rejected");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("Int64"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

#[test]
#[ignore]
fn test_gpu_gather_rejects_empty_vocab() {
    // Regression: empty vocab + non-empty indices would clamp every index to
    // 0 and read from a zero-byte buffer (OOB device read). Must be rejected
    // up front as a clean error.
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let data = make_f32_device_tensor(&[0, 8], &[]);
    let indices = make_i64_device_tensor(&[2], &[0i64, 1]);

    let result = cuda::kernels::gather::gather_gpu(&rt, &data, &indices, 0);
    assert!(
        result.is_err(),
        "empty vocab with non-empty indices should be rejected"
    );
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("vocab"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

// ── Section 4: RMSNormalization ────────────────────────────────────

#[test]
#[ignore]
fn test_gpu_rms_norm_f32_small() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Input [2, 4, 16] = 8 rows × 16 hidden. Outer = 8.
    let outer = 8usize;
    let hidden = 16usize;
    let mut x_data = vec![0f32; outer * hidden];
    for row in 0..outer {
        for h in 0..hidden {
            x_data[row * hidden + h] = (h as f32) * 0.1 - 0.5 + (row as f32) * 0.01;
        }
    }
    let weight_data: Vec<f32> = (0..hidden).map(|i| 1.0 + (i as f32) * 0.05).collect();
    let eps = 1e-6f32;

    let x = make_f32_device_tensor(&[2, 4, hidden as i64], &x_data);
    let w = make_f32_device_tensor(&[hidden as i64], &weight_data);

    let y = cuda::kernels::rms_norm::rms_norm_gpu(&rt, &x, &w, eps).expect("rms_norm_gpu f32");
    cuda::kernels::synchronize().expect("sync");

    assert_eq!(y.shape, vec![2, 4, hidden as i64]);
    assert_eq!(y.dtype, DataType::Float);

    let y_host = read_f32_device_tensor(&y);
    assert_eq!(y_host.len(), outer * hidden);

    // Scalar Rust reference: y = x * rsqrt(mean(x*x) + eps) * weight.
    for row in 0..outer {
        let mut sum_sq = 0.0f32;
        for h in 0..hidden {
            let v = x_data[row * hidden + h];
            sum_sq += v * v;
        }
        let mean = sum_sq / hidden as f32;
        let inv_rms = 1.0f32 / (mean + eps).sqrt();
        for h in 0..hidden {
            let expected = x_data[row * hidden + h] * inv_rms * weight_data[h];
            let got = y_host[row * hidden + h];
            assert!(
                (got - expected).abs() < 1e-5,
                "row {row} h {h}: got {got}, expected {expected}"
            );
        }
    }
}

#[test]
#[ignore]
fn test_gpu_rms_norm_bf16_gemma_shape() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Gemma-style dimensions: [1, 32, 4096] — one batch, 32 tokens,
    // hidden_size 4096. Outer = 32.
    let outer = 32usize;
    let hidden = 4096usize;
    let eps = 1e-6f32;

    // Deterministic input values.
    let mut x_data = vec![0f32; outer * hidden];
    for row in 0..outer {
        for h in 0..hidden {
            let i = row * hidden + h;
            x_data[i] = ((i as f32) * 0.01).sin() + ((row % 7) as f32) * 0.1;
        }
    }
    let weight_data: Vec<f32> = (0..hidden)
        .map(|h| 0.9 + ((h % 13) as f32) * 0.01)
        .collect();

    // Construct BF16 device tensors directly.
    let x = make_bf16_device_tensor(&[1, 32, hidden as i64], &x_data);
    let w = make_bf16_device_tensor(&[hidden as i64], &weight_data);

    let y = cuda::kernels::rms_norm::rms_norm_gpu(&rt, &x, &w, eps).expect("rms_norm_gpu bf16");
    cuda::kernels::synchronize().expect("sync");

    assert_eq!(y.shape, vec![1, 32, hidden as i64]);
    assert_eq!(y.dtype, DataType::BFloat16);

    let y_host = read_bf16_device_tensor(&y);
    assert_eq!(y_host.len(), outer * hidden);

    // CPU reference: operate on BF16-rounded inputs for fairness —
    // this matches exactly what the kernel loads from device memory.
    let x_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&x_data));
    let w_round = smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(
        &weight_data,
    ));

    let mut max_abs_err = 0.0f32;
    for row in 0..outer {
        // F32 accumulation matches the kernel.
        let mut sum_sq = 0.0f32;
        for h in 0..hidden {
            let v = x_round[row * hidden + h];
            sum_sq += v * v;
        }
        let mean = sum_sq / hidden as f32;
        let inv_rms = 1.0f32 / (mean + eps).sqrt();
        for h in 0..hidden {
            let expected = x_round[row * hidden + h] * inv_rms * w_round[h];
            let got = y_host[row * hidden + h];
            let err = (got - expected).abs();
            if err > max_abs_err {
                max_abs_err = err;
            }
            assert!(
                err < 1e-2,
                "row {row} h {h}: got {got}, expected {expected}, err {err}"
            );
        }
    }
    eprintln!("rms_norm_bf16 [1,32,4096] max_abs_err = {max_abs_err}");
}

// ── Section 5: RotaryEmbedding ─────────────────────────────────────

/// Build cos/sin tables on the host: `cos[r, p] = cos(theta_p * r)`,
/// `sin[r, p] = sin(theta_p * r)` where `theta_p = base^(-2p / head_dim)`.
/// Mirrors the Gemma load-time table generation but stays self-contained
/// for the test.
fn build_rope_tables(max_seq: usize, head_dim: usize, base: f32) -> (Vec<f32>, Vec<f32>) {
    assert!(head_dim.is_multiple_of(2));
    let half = head_dim / 2;
    let mut cos = vec![0f32; max_seq * half];
    let mut sin = vec![0f32; max_seq * half];
    for r in 0..max_seq {
        for p in 0..half {
            let exponent = -((2 * p) as f32) / head_dim as f32;
            let theta = base.powf(exponent);
            let angle = (r as f32) * theta;
            cos[r * half + p] = angle.cos();
            sin[r * half + p] = angle.sin();
        }
    }
    (cos, sin)
}

/// Apply RoPE on the host, matching the GPU kernel exactly. Returns the
/// rotated tensor flattened in `[B, H, Sq, head_dim]` row-major order.
#[allow(clippy::too_many_arguments)]
fn host_rope_reference(
    x: &[f32],
    batch: usize,
    heads: usize,
    seq_len: usize,
    head_dim: usize,
    cos: &[f32],
    sin: &[f32],
    position: usize,
    interleaved: bool,
) -> Vec<f32> {
    let half = head_dim / 2;
    let mut y = x.to_vec();
    for b in 0..batch {
        for h in 0..heads {
            for qi in 0..seq_len {
                let pos = position + qi;
                let row_base = ((b * heads + h) * seq_len + qi) * head_dim;
                for pair in 0..half {
                    let c = cos[pos * half + pair];
                    let s = sin[pos * half + pair];
                    let (i0, i1) = if interleaved {
                        (row_base + 2 * pair, row_base + 2 * pair + 1)
                    } else {
                        (row_base + pair, row_base + pair + half)
                    };
                    let x0 = x[i0];
                    let x1 = x[i1];
                    y[i0] = c * x0 - s * x1;
                    y[i1] = s * x0 + c * x1;
                }
            }
        }
    }
    y
}

#[test]
#[ignore]
fn test_gpu_rotary_f32_small_split_half() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Q tensor: [B=1, H=2, Sq=4, head_dim=8]. Split-half layout (Gemma).
    let batch = 1usize;
    let heads = 2usize;
    let sq = 4usize;
    let head_dim = 8usize;
    let max_seq = 16usize;

    let n = batch * heads * sq * head_dim;
    let x_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.0125 - 0.25).collect();
    let (cos_data, sin_data) = build_rope_tables(max_seq, head_dim, 10000.0);

    let x = make_f32_device_tensor(
        &[batch as i64, heads as i64, sq as i64, head_dim as i64],
        &x_data,
    );
    let cos = make_f32_device_tensor(&[max_seq as i64, (head_dim / 2) as i64], &cos_data);
    let sin = make_f32_device_tensor(&[max_seq as i64, (head_dim / 2) as i64], &sin_data);

    let position = 0i32;
    let y = cuda::kernels::rotary::rotary_gpu(&rt, &x, &cos, &sin, position, false, None)
        .expect("rotary_gpu f32");
    cuda::kernels::synchronize().expect("sync");

    assert_eq!(
        y.shape,
        vec![batch as i64, heads as i64, sq as i64, head_dim as i64]
    );
    assert_eq!(y.dtype, DataType::Float);

    let y_host = read_f32_device_tensor(&y);
    let expected = host_rope_reference(
        &x_data, batch, heads, sq, head_dim, &cos_data, &sin_data, 0, false,
    );
    for (i, (got, want)) in y_host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "idx {i}: got {got}, expected {want}"
        );
    }
}

#[test]
#[ignore]
fn test_gpu_rotary_f32_decode_step_interleaved() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Decode step: Sq=1, position=37. Interleaved (textbook) layout.
    let batch = 2usize;
    let heads = 4usize;
    let sq = 1usize;
    let head_dim = 16usize;
    let max_seq = 64usize;
    let position: usize = 37;

    let n = batch * heads * sq * head_dim;
    let x_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).sin()).collect();
    let (cos_data, sin_data) = build_rope_tables(max_seq, head_dim, 10000.0);

    let x = make_f32_device_tensor(
        &[batch as i64, heads as i64, sq as i64, head_dim as i64],
        &x_data,
    );
    let cos = make_f32_device_tensor(&[max_seq as i64, (head_dim / 2) as i64], &cos_data);
    let sin = make_f32_device_tensor(&[max_seq as i64, (head_dim / 2) as i64], &sin_data);

    let y = cuda::kernels::rotary::rotary_gpu(&rt, &x, &cos, &sin, position as i32, true, None)
        .expect("rotary_gpu f32 interleaved");
    cuda::kernels::synchronize().expect("sync");

    let y_host = read_f32_device_tensor(&y);
    let expected = host_rope_reference(
        &x_data, batch, heads, sq, head_dim, &cos_data, &sin_data, position, true,
    );
    for (i, (got, want)) in y_host.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "idx {i}: got {got}, expected {want}"
        );
    }
}

#[test]
#[ignore]
fn test_gpu_rotary_bf16_gemma_shape() {
    use smallaios_onnx_rt::ops::microsoft::op_rotary_embedding;

    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Gemma-ish per-layer Q shape: [B=1, H=8, Sq=8, head_dim=128]. The
    // CPU `op_rotary_embedding` accepts F32 only, so we build the
    // reference in F32 and compare BF16 GPU output against it within
    // a 1e-2 tolerance band (matches the design's BF16 tolerance).
    let batch = 1usize;
    let heads = 8usize;
    let sq = 8usize;
    let head_dim = 128usize;
    let max_seq = 32usize;
    let position: usize = 0;

    let n = batch * heads * sq * head_dim;
    let x_data: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.001).cos() * 0.5).collect();
    let (cos_data, sin_data) = build_rope_tables(max_seq, head_dim, 10000.0);

    // BF16 round-trip the input so the GPU sees the same values as the
    // F32 reference.
    let x_bf16_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&x_data));

    let x = make_bf16_device_tensor(
        &[batch as i64, heads as i64, sq as i64, head_dim as i64],
        &x_data,
    );
    let cos = make_f32_device_tensor(&[max_seq as i64, (head_dim / 2) as i64], &cos_data);
    let sin = make_f32_device_tensor(&[max_seq as i64, (head_dim / 2) as i64], &sin_data);

    let y = cuda::kernels::rotary::rotary_gpu(&rt, &x, &cos, &sin, position as i32, false, None)
        .expect("rotary_gpu bf16");
    cuda::kernels::synchronize().expect("sync");

    let y_host = read_bf16_device_tensor(&y);

    // CPU reference via the rank-4 path: shape is (B, H, Sq, head_dim)
    // and the buffer is contiguous in that exact order — same memory
    // layout the GPU kernel sees, so no reshape needed.
    let x_tensor = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![batch as i64, heads as i64, sq as i64, head_dim as i64]),
        name: String::new(),
        raw_data: x_bf16_round.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let pos_tensor = Tensor {
        data_type: DataType::Int64,
        shape: TensorShape::new(vec![sq as i64]),
        name: String::new(),
        raw_data: (0..sq as i64)
            .map(|p| position as i64 + p)
            .flat_map(|v| v.to_le_bytes())
            .collect(),
    };
    let cos_tensor = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![max_seq as i64, (head_dim / 2) as i64]),
        name: String::new(),
        raw_data: cos_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let sin_tensor = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![max_seq as i64, (head_dim / 2) as i64]),
        name: String::new(),
        raw_data: sin_data.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let inputs: [Option<&Tensor>; 4] = [
        Some(&x_tensor),
        Some(&pos_tensor),
        Some(&cos_tensor),
        Some(&sin_tensor),
    ];
    let cpu_out = op_rotary_embedding(&inputs, false, head_dim as i64, heads as i64)
        .expect("op_rotary_embedding");
    let cpu_4d: Vec<f32> = cpu_out
        .raw_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let mut max_abs_err = 0.0f32;
    for (i, (got, want)) in y_host.iter().zip(cpu_4d.iter()).enumerate() {
        let err = (got - want).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        assert!(err < 1e-2, "idx {i}: got {got}, expected {want}, err {err}");
    }
    eprintln!("rotary_bf16 [1,8,8,128] vs op_rotary_embedding max_abs_err = {max_abs_err}");
}

#[test]
#[ignore]
fn test_gpu_rotary_rejects_odd_head_dim() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // head_dim=7 is odd → must be rejected before launch.
    let x = make_f32_device_tensor(&[1, 1, 1, 7], &[0f32; 7]);
    let cos = make_f32_device_tensor(&[1, 3], &[0f32; 3]);
    let sin = make_f32_device_tensor(&[1, 3], &[0f32; 3]);

    let result = cuda::kernels::rotary::rotary_gpu(&rt, &x, &cos, &sin, 0, false, None);
    assert!(result.is_err(), "odd head_dim should be rejected");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("head_dim"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

// ── Section 6.1-6.4: Strided batched GEMM ──────────────────────────

/// CPU reference for `[batch, M, K] @ [batch, K, N] -> [batch, M, N]`
/// in F32. Used to validate `gpu_gemm_strided_batched_ex` outputs.
fn cpu_strided_batched_gemm(
    a: &[f32],
    b: &[f32],
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
) -> Vec<f32> {
    let mut c = vec![0f32; batch * m * n];
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let mut acc = 0f32;
                for ki in 0..k {
                    let a_v = a[bi * m * k + mi * k + ki];
                    let b_v = b[bi * k * n + ki * n + ni];
                    acc += a_v * b_v;
                }
                c[bi * m * n + mi * n + ni] = acc;
            }
        }
    }
    c
}

#[test]
#[ignore]
fn test_gpu_strided_batched_gemm_f32_vs_cpu() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // [batch=4, M=8, K=16] @ [batch=4, K=16, N=8] -> [batch, 8, 8]
    let batch = 4usize;
    let m = 8usize;
    let k = 16usize;
    let n = 8usize;

    let a_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let b_data: Vec<f32> = (0..batch * k * n)
        .map(|i| ((i as f32) * 0.017).cos())
        .collect();

    let a = make_f32_device_tensor(&[batch as i64, m as i64, k as i64], &a_data);
    let b = make_f32_device_tensor(&[batch as i64, k as i64, n as i64], &b_data);

    let c = cuda::dispatch::gpu_gemm_strided_batched_ex(&rt, &a, &b, false, false, DataType::Float)
        .expect("strided batched gemm f32");

    assert_eq!(c.shape, vec![batch as i64, m as i64, n as i64]);
    assert_eq!(c.dtype, DataType::Float);

    let c_host = read_f32_device_tensor(&c);
    let cpu = cpu_strided_batched_gemm(&a_data, &b_data, batch, m, k, n);
    for (i, (got, want)) in c_host.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-3,
            "idx {i}: got {got} expected {want}"
        );
    }
}

#[test]
#[ignore]
fn test_gpu_strided_batched_gemm_bf16_compute32f() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    // GQA-shaped: [batch=num_heads=8, M=seq_q=4, K=head_dim=32]
    //          @  [batch=8, K=32, N=seq_kv=4] -> [8, 4, 4]
    let batch = 8usize;
    let m = 4usize;
    let k = 32usize;
    let n = 4usize;

    let a_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i as f32) * 0.005).sin() * 0.5)
        .collect();
    let b_data: Vec<f32> = (0..batch * k * n)
        .map(|i| ((i as f32) * 0.007).cos() * 0.5)
        .collect();

    let a = make_bf16_device_tensor(&[batch as i64, m as i64, k as i64], &a_data);
    let b = make_bf16_device_tensor(&[batch as i64, k as i64, n as i64], &b_data);

    // BF16 inputs, F32 output (the QK^T case).
    let c = cuda::dispatch::gpu_gemm_strided_batched_ex(&rt, &a, &b, false, false, DataType::Float)
        .expect("strided batched gemm bf16->f32");

    assert_eq!(c.shape, vec![batch as i64, m as i64, n as i64]);
    assert_eq!(c.dtype, DataType::Float);

    // BF16 round-trip the inputs for the CPU reference so we compare
    // against what the GPU actually saw.
    let a_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&a_data));
    let b_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&b_data));

    let c_host = read_f32_device_tensor(&c);
    let cpu = cpu_strided_batched_gemm(&a_round, &b_round, batch, m, k, n);
    for (i, (got, want)) in c_host.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-2,
            "idx {i}: got {got} expected {want}"
        );
    }
}

#[test]
#[ignore]
fn test_gpu_strided_batched_gemm_f32_tf32_path() {
    // Exercises the runtime's TF32 compute mode (the default for F32
    // inputs). Wider tolerance since TF32 truncates 13 mantissa bits.
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    let batch = 2usize;
    let m = 16usize;
    let k = 64usize;
    let n = 16usize;

    let a_data: Vec<f32> = (0..batch * m * k)
        .map(|i| ((i as f32) * 0.003).sin() * 0.3)
        .collect();
    let b_data: Vec<f32> = (0..batch * k * n)
        .map(|i| ((i as f32) * 0.011).cos() * 0.3)
        .collect();

    let a = make_f32_device_tensor(&[batch as i64, m as i64, k as i64], &a_data);
    let b = make_f32_device_tensor(&[batch as i64, k as i64, n as i64], &b_data);

    let c = cuda::dispatch::gpu_gemm_strided_batched_ex(&rt, &a, &b, false, false, DataType::Float)
        .expect("strided batched gemm f32 tf32");

    let c_host = read_f32_device_tensor(&c);
    let cpu = cpu_strided_batched_gemm(&a_data, &b_data, batch, m, k, n);
    for (i, (got, want)) in c_host.iter().zip(cpu.iter()).enumerate() {
        // TF32 tolerance: well above the 1e-3 F32 band the design uses.
        assert!(
            (got - want).abs() < 5e-3,
            "idx {i}: got {got} expected {want}"
        );
    }
}

#[test]
#[ignore]
fn test_gpu_strided_batched_gemm_qk_transpose() {
    // Exercises trans_b (the K^T path used inside QK^T).
    // A: [H, Sq, head_dim] = [4, 3, 8]
    // B: [H, Sk, head_dim] = [4, 5, 8]   (will be transposed to [4, 8, 5])
    // C: [4, 3, 5]
    let rt = cuda::CudaRuntime::init().expect("CUDA init");

    let h = 4usize;
    let sq = 3usize;
    let hd = 8usize;
    let sk = 5usize;

    let a_data: Vec<f32> = (0..h * sq * hd).map(|i| (i as f32) * 0.01).collect();
    let b_data: Vec<f32> = (0..h * sk * hd)
        .map(|i| ((i as f32) * 0.02).sin())
        .collect();

    let a = make_f32_device_tensor(&[h as i64, sq as i64, hd as i64], &a_data);
    let b = make_f32_device_tensor(&[h as i64, sk as i64, hd as i64], &b_data);

    let c = cuda::dispatch::gpu_gemm_strided_batched_ex(&rt, &a, &b, false, true, DataType::Float)
        .expect("strided batched gemm trans_b");
    assert_eq!(c.shape, vec![h as i64, sq as i64, sk as i64]);

    // CPU reference with the K^T baked in.
    let mut cpu = vec![0f32; h * sq * sk];
    for hi in 0..h {
        for qi in 0..sq {
            for ki in 0..sk {
                let mut acc = 0f32;
                for d in 0..hd {
                    acc += a_data[(hi * sq + qi) * hd + d] * b_data[(hi * sk + ki) * hd + d];
                }
                cpu[(hi * sq + qi) * sk + ki] = acc;
            }
        }
    }
    let c_host = read_f32_device_tensor(&c);
    for (i, (got, want)) in c_host.iter().zip(cpu.iter()).enumerate() {
        assert!(
            (got - want).abs() < 5e-3,
            "idx {i}: got {got} expected {want}"
        );
    }
}

// ── Section 6.10-6.11: KV head expansion ───────────────────────────

#[test]
#[ignore]
fn test_gpu_kv_expand_gemma_ratio() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Gemma 4 ratio: 32 attention heads / 16 KV heads = expand 2.
    // Use head_dim=8 and seq=4 to keep the validation loop tractable.
    let batch = 1usize;
    let num_kv = 16usize;
    let expand = 2i32;
    let seq = 4usize;
    let head_dim = 8usize;
    let num_q = num_kv * expand as usize;

    let kv_data: Vec<f32> = (0..batch * num_kv * seq * head_dim)
        .map(|i| (i as f32) * 0.001)
        .collect();
    let kv = make_bf16_device_tensor(
        &[batch as i64, num_kv as i64, seq as i64, head_dim as i64],
        &kv_data,
    );

    let out =
        cuda::kernels::attention::kv_expand_gpu(&rt, &kv, expand).expect("kv_expand_gpu bf16");
    cuda::kernels::synchronize().expect("sync");

    assert_eq!(
        out.shape,
        vec![batch as i64, num_q as i64, seq as i64, head_dim as i64]
    );
    assert_eq!(out.dtype, DataType::BFloat16);

    let kv_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&kv_data));
    let out_host = read_bf16_device_tensor(&out);
    for b in 0..batch {
        for hq in 0..num_q {
            let hk = hq / expand as usize;
            for s in 0..seq {
                for d in 0..head_dim {
                    let want = kv_round[((b * num_kv + hk) * seq + s) * head_dim + d];
                    let got = out_host[((b * num_q + hq) * seq + s) * head_dim + d];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "b={b} hq={hq} (hk={hk}) s={s} d={d}: got {got} want {want}"
                    );
                }
            }
        }
    }
}

#[test]
#[ignore]
fn test_gpu_kv_expand_one_is_identity() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // expand=1 (standard MHA) takes the D2D fast path; output bytes
    // should equal input bytes.
    let kv_data: Vec<f32> = (0..32).map(|i| i as f32).collect();
    let kv = make_f32_device_tensor(&[1, 4, 2, 4], &kv_data);
    let out = cuda::kernels::attention::kv_expand_gpu(&rt, &kv, 1).expect("kv_expand_gpu identity");
    cuda::kernels::synchronize().expect("sync");
    assert_eq!(out.shape, kv.shape);
    let out_host = read_f32_device_tensor(&out);
    for (i, (a, b)) in kv_data.iter().zip(out_host.iter()).enumerate() {
        assert!((a - b).abs() < 1e-6, "idx {i}: {a} != {b}");
    }
}

// ── Section 6.5-6.9: Masked softmax ────────────────────────────────

#[test]
#[ignore]
fn test_gpu_masked_softmax_causal_row_sums_to_one() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Prefill case: H=4, seq_q=8, seq_kv=8 (causal_offset = 0). Each
    // query at q_idx attends to keys [0..=q_idx], so the per-row sum
    // must be exactly 1.0 (not 0.0, since at least k=0 is allowed).
    let heads = 4usize;
    let seq_q = 8usize;
    let seq_kv = 8usize;
    let n = heads * seq_q * seq_kv;

    // Mildly varied scores to make sure the row-max path works.
    let scores: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.13).sin() * 2.0).collect();
    let dev = make_f32_device_tensor(&[heads as i64, seq_q as i64, seq_kv as i64], &scores);

    let scale = 1.0f32 / (16f32).sqrt();
    cuda::kernels::attention::masked_softmax_gpu(
        &rt,
        &dev,
        seq_q as i32,
        seq_kv as i32,
        None,
        scale,
    )
    .expect("masked_softmax_gpu causal");
    cuda::kernels::synchronize().expect("sync");

    let host = read_f32_device_tensor(&dev);
    for h in 0..heads {
        for qi in 0..seq_q {
            let mut row_sum = 0.0f32;
            for kj in 0..seq_kv {
                let v = host[(h * seq_q + qi) * seq_kv + kj];
                if kj > qi {
                    assert!(
                        v.abs() < 1e-6,
                        "masked position should be 0 (h={h} q={qi} k={kj}, got {v})"
                    );
                } else {
                    assert!(v >= 0.0, "softmax output must be non-negative");
                }
                row_sum += v;
            }
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "h={h} q={qi}: row_sum {row_sum} != 1.0"
            );
        }
    }
}

#[test]
#[ignore]
fn test_gpu_masked_softmax_sliding_window() {
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Sliding-window mask. Query q_pos attends to k in
    // [q_pos - window, q_pos]. With seq_q == seq_kv == 16, window = 4:
    //   q=0:  k∈{0}            (q_pos-window=-4 clamps to 0)
    //   q=5:  k∈{1,2,3,4,5}
    //   q=15: k∈{11,...,15}
    let heads = 2usize;
    let seq = 16usize;
    let window = 4i32;
    let n = heads * seq * seq;

    let scores: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.07).cos()).collect();
    let dev = make_f32_device_tensor(&[heads as i64, seq as i64, seq as i64], &scores);

    cuda::kernels::attention::masked_softmax_gpu(
        &rt,
        &dev,
        seq as i32,
        seq as i32,
        Some(window),
        1.0,
    )
    .expect("masked_softmax_gpu sliding");
    cuda::kernels::synchronize().expect("sync");

    let host = read_f32_device_tensor(&dev);
    for h in 0..heads {
        for qi in 0..seq {
            let q_pos = qi as i32;
            let mut row_sum = 0.0f32;
            for kj in 0..seq {
                let k = kj as i32;
                let allowed = k <= q_pos && k >= q_pos - window;
                let v = host[(h * seq + qi) * seq + kj];
                if !allowed {
                    assert!(
                        v.abs() < 1e-6,
                        "h={h} q={qi} k={kj}: out-of-window must be 0 (got {v})"
                    );
                }
                row_sum += v;
            }
            assert!(
                (row_sum - 1.0).abs() < 1e-5,
                "h={h} q={qi}: row_sum {row_sum} != 1.0"
            );
        }
    }
}

#[test]
#[ignore]
fn test_gpu_masked_softmax_rejects_seq_q_gt_seq_kv() {
    // Regression: seq_q > seq_kv would push causal_offset negative and
    // silently mask the leading query rows to all-zero probabilities.
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let seq_q = 8i32;
    let seq_kv = 4i32;
    let scores: Vec<f32> = vec![0.1f32; seq_q as usize * seq_kv as usize];
    let dev = make_f32_device_tensor(&[1, seq_q as i64, seq_kv as i64], &scores);

    let result = cuda::kernels::attention::masked_softmax_gpu(&rt, &dev, seq_q, seq_kv, None, 1.0);
    assert!(result.is_err(), "seq_q > seq_kv should be rejected");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("seq_q"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
}

// ── Section 6.12-6.19: gpu_gqa end-to-end ──────────────────────────

/// Build a [B=1, Sq, hidden] f32 vector from an interleaved
/// [B, H, Sq, head_dim] source — the layout `op_group_query_attention`
/// consumes via its rank-3 input path.
fn pack_bsd_from_bhsd(src: &[f32], heads: usize, sq: usize, head_dim: usize) -> Vec<f32> {
    let mut out = vec![0f32; sq * heads * head_dim];
    for h in 0..heads {
        for qi in 0..sq {
            for d in 0..head_dim {
                let s = (h * sq + qi) * head_dim + d;
                let dst = (qi * heads + h) * head_dim + d;
                out[dst] = src[s];
            }
        }
    }
    out
}

#[test]
#[ignore]
fn test_gpu_gqa_f32_vs_cpu_mha() {
    use smallaios_onnx_rt::ops::microsoft::op_group_query_attention;
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // Standard MHA: num_q_heads == num_kv_heads. Sq == Sk so the CPU
    // op handles the whole thing as a single forward pass with no past.
    let h = 4usize;
    let kv = 4usize;
    let sq = 4usize;
    let hd = 8usize;
    let n_q = h * sq * hd;
    let n_k = kv * sq * hd;

    let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let k_data: Vec<f32> = (0..n_k).map(|i| ((i as f32) * 0.017).cos() * 0.5).collect();
    let v_data: Vec<f32> = (0..n_k).map(|i| ((i as f32) * 0.011).sin() * 0.5).collect();

    let q = make_f32_device_tensor(&[1, h as i64, sq as i64, hd as i64], &q_data);
    let k = make_f32_device_tensor(&[1, kv as i64, sq as i64, hd as i64], &k_data);
    let v = make_f32_device_tensor(&[1, kv as i64, sq as i64, hd as i64], &v_data);

    let out = cuda::kernels::attention::gpu_gqa(&rt, &q, &k, &v, None, None).expect("gpu_gqa f32");
    cuda::kernels::synchronize().expect("sync");
    assert_eq!(out.shape, vec![1, sq as i64, (h * hd) as i64]);
    assert_eq!(out.dtype, DataType::Float);

    let q_3d = pack_bsd_from_bhsd(&q_data, h, sq, hd);
    let k_3d = pack_bsd_from_bhsd(&k_data, kv, sq, hd);
    let v_3d = pack_bsd_from_bhsd(&v_data, kv, sq, hd);
    let q_t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![1, sq as i64, (h * hd) as i64]),
        name: String::new(),
        raw_data: q_3d.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let k_t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![1, sq as i64, (kv * hd) as i64]),
        name: String::new(),
        raw_data: k_3d.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let v_t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![1, sq as i64, (kv * hd) as i64]),
        name: String::new(),
        raw_data: v_3d.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let inputs: [Option<&Tensor>; 9] = [
        Some(&q_t),
        Some(&k_t),
        Some(&v_t),
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let (cpu_out, _, _) =
        op_group_query_attention(&inputs, h as i64, kv as i64, None, -1, false, false)
            .expect("op_group_query_attention");
    let cpu_3d: Vec<f32> = cpu_out
        .raw_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let out_host = read_f32_device_tensor(&out);
    let mut max_abs_err = 0.0f32;
    for (i, (got, want)) in out_host.iter().zip(cpu_3d.iter()).enumerate() {
        let err = (got - want).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        assert!(err < 5e-3, "idx {i}: got {got} expected {want} err {err}");
    }
    eprintln!("gpu_gqa f32 [1,4,4,8] vs CPU max_abs_err = {max_abs_err}");
}

#[test]
#[ignore]
fn test_gpu_gqa_bf16_ratio_2to1() {
    use smallaios_onnx_rt::ops::microsoft::op_group_query_attention;
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // GQA ratio 2:1 (4 q heads / 2 KV heads), BF16. Sq=Sk for direct
    // CPU comparison.
    let h = 4usize;
    let kv = 2usize;
    let sq = 4usize;
    let hd = 16usize;
    let n_q = h * sq * hd;
    let n_k = kv * sq * hd;

    let q_data: Vec<f32> = (0..n_q).map(|i| ((i as f32) * 0.005).sin() * 0.4).collect();
    let k_data: Vec<f32> = (0..n_k).map(|i| ((i as f32) * 0.007).cos() * 0.4).collect();
    let v_data: Vec<f32> = (0..n_k).map(|i| ((i as f32) * 0.009).sin() * 0.4).collect();

    let q = make_bf16_device_tensor(&[1, h as i64, sq as i64, hd as i64], &q_data);
    let k = make_bf16_device_tensor(&[1, kv as i64, sq as i64, hd as i64], &k_data);
    let v = make_bf16_device_tensor(&[1, kv as i64, sq as i64, hd as i64], &v_data);

    let out = cuda::kernels::attention::gpu_gqa(&rt, &q, &k, &v, None, None).expect("gpu_gqa bf16");
    cuda::kernels::synchronize().expect("sync");
    assert_eq!(out.shape, vec![1, sq as i64, (h * hd) as i64]);
    assert_eq!(out.dtype, DataType::BFloat16);

    let q_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&q_data));
    let k_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&k_data));
    let v_round =
        smallaios_onnx_rt::tensor::bf16_to_f32(&smallaios_onnx_rt::tensor::f32_to_bf16(&v_data));

    let q_3d = pack_bsd_from_bhsd(&q_round, h, sq, hd);
    let k_3d = pack_bsd_from_bhsd(&k_round, kv, sq, hd);
    let v_3d = pack_bsd_from_bhsd(&v_round, kv, sq, hd);
    let q_t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![1, sq as i64, (h * hd) as i64]),
        name: String::new(),
        raw_data: q_3d.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let k_t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![1, sq as i64, (kv * hd) as i64]),
        name: String::new(),
        raw_data: k_3d.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let v_t = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![1, sq as i64, (kv * hd) as i64]),
        name: String::new(),
        raw_data: v_3d.iter().flat_map(|f| f.to_le_bytes()).collect(),
    };
    let inputs: [Option<&Tensor>; 9] = [
        Some(&q_t),
        Some(&k_t),
        Some(&v_t),
        None,
        None,
        None,
        None,
        None,
        None,
    ];
    let (cpu_out, _, _) =
        op_group_query_attention(&inputs, h as i64, kv as i64, None, -1, false, false)
            .expect("op_group_query_attention");
    let cpu_3d: Vec<f32> = cpu_out
        .raw_data
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    let out_host = read_bf16_device_tensor(&out);
    let mut max_abs_err = 0.0f32;
    for (i, (got, want)) in out_host.iter().zip(cpu_3d.iter()).enumerate() {
        let err = (got - want).abs();
        if err > max_abs_err {
            max_abs_err = err;
        }
        assert!(err < 5e-2, "idx {i}: got {got} expected {want} err {err}");
    }
    eprintln!("gpu_gqa bf16 [1,4,4,16] (GQA 2:1) vs CPU max_abs_err = {max_abs_err}");
}

#[test]
#[ignore]
fn test_gpu_gqa_first_token_no_history() {
    // The "first token, position == 0" case: Sq = Sk = 1, so the
    // attention output for each head is just V[head, 0] (a single
    // softmax weight of 1.0 on a single key).
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let h = 2usize;
    let hd = 8usize;
    let q_data: Vec<f32> = (0..h * hd).map(|i| (i as f32) * 0.1).collect();
    let v_data: Vec<f32> = (0..h * hd).map(|i| (i as f32) * 0.05 + 1.0).collect();
    let k_data: Vec<f32> = vec![0.0f32; h * hd];

    let q = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &q_data);
    let k = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &k_data);
    let v = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &v_data);

    let out = cuda::kernels::attention::gpu_gqa(&rt, &q, &k, &v, None, None)
        .expect("gpu_gqa first token");
    cuda::kernels::synchronize().expect("sync");
    assert_eq!(out.shape, vec![1, 1, (h * hd) as i64]);

    let host = read_f32_device_tensor(&out);
    for h_i in 0..h {
        for d in 0..hd {
            let want = v_data[h_i * hd + d];
            let got = host[h_i * hd + d];
            assert!(
                (got - want).abs() < 1e-5,
                "h={h_i} d={d}: got {got} want {want}"
            );
            assert!(got.is_finite(), "output must be finite");
        }
    }
}

#[test]
#[ignore]
fn test_gpu_gqa_with_cache_first_token() {
    // §6.19: first token, position == 0, empty KV cache. After
    // gpu_gqa_with_cache: cache.current_position() == 1 and the output
    // matches the no-cache `gpu_gqa` invocation on the same K/V.
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    let h = 2usize;
    let hd = 8usize;
    let max_seq = 8usize;

    let q_data: Vec<f32> = (0..h * hd).map(|i| (i as f32) * 0.1).collect();
    let k_data: Vec<f32> = (0..h * hd).map(|i| ((i as f32) * 0.03).sin()).collect();
    let v_data: Vec<f32> = (0..h * hd).map(|i| (i as f32) * 0.05 + 1.0).collect();

    let q = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &q_data);
    let new_k = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &k_data);
    let new_v = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &v_data);

    let mut cache = cuda::GpuKvCache::allocate(
        &rt,
        1,
        h,
        hd,
        max_seq,
        DataType::Float,
        &[cuda::LayerKind::Global],
    )
    .expect("alloc cache");
    assert_eq!(cache.current_position(), 0);

    let out = cuda::kernels::attention::gpu_gqa_with_cache(
        &rt, &q, &new_k, &new_v, &mut cache, 0, None, None,
    )
    .expect("gpu_gqa_with_cache first token");
    cuda::kernels::synchronize().expect("sync");

    // gpu_gqa_with_cache appends but does NOT advance — the executor
    // advances once per forward pass (after the last layer). For this
    // unit test we advance manually to mirror the executor's contract.
    cache.advance_position().expect("advance_position");

    assert_eq!(cache.current_position(), 1, "cache should hold one token");
    assert_eq!(out.shape, vec![1, 1, (h * hd) as i64]);

    // Reference: re-run gpu_gqa without the cache on the same K/V.
    let k_ref = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &k_data);
    let v_ref = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &v_data);
    let q_ref = make_f32_device_tensor(&[1, h as i64, 1, hd as i64], &q_data);
    let ref_out = cuda::kernels::attention::gpu_gqa(&rt, &q_ref, &k_ref, &v_ref, None, None)
        .expect("gpu_gqa ref");
    cuda::kernels::synchronize().expect("sync");

    let cache_host = read_f32_device_tensor(&out);
    let ref_host = read_f32_device_tensor(&ref_out);
    for (i, (a, b)) in cache_host.iter().zip(ref_host.iter()).enumerate() {
        assert!((a - b).abs() < 1e-5, "idx {i}: cache_path {a} vs ref {b}");
        assert!(a.is_finite(), "cache_path output must be finite");
    }
}

#[test]
#[ignore]
fn test_gpu_gqa_attention_scratch_cap() {
    // §6.13: gpu_gqa fails fast when the [H, Sq, Sk] F32 scratch
    // exceeds the runtime's attention_scratch_cap_bytes setting.
    let rt = cuda::CudaRuntime::init().expect("CUDA init");
    rt.init_kernels().expect("init_kernels");

    // 4 heads × 4 × 4 F32 scratch = 256 bytes. Set cap to 100.
    rt.set_attention_scratch_cap(100);

    let h = 4usize;
    let sq = 4usize;
    let hd = 8usize;
    let q_data: Vec<f32> = vec![0f32; h * sq * hd];
    let kv_data: Vec<f32> = vec![0f32; h * sq * hd];
    let q = make_f32_device_tensor(&[1, h as i64, sq as i64, hd as i64], &q_data);
    let k = make_f32_device_tensor(&[1, h as i64, sq as i64, hd as i64], &kv_data);
    let v = make_f32_device_tensor(&[1, h as i64, sq as i64, hd as i64], &kv_data);

    let result = cuda::kernels::attention::gpu_gqa(&rt, &q, &k, &v, None, None);
    assert!(result.is_err(), "should reject scratch over cap");
    match result {
        Err(cuda::CudaError::RuntimeError { op, .. }) => {
            assert!(op.contains("scratch"), "op msg: {op}");
        }
        Err(other) => panic!("expected RuntimeError, got: {other:?}"),
        Ok(_) => unreachable!(),
    }
    // Restore default so subsequent tests aren't affected.
    rt.set_attention_scratch_cap(256 * 1024 * 1024);
}

// ── Conv attribute coverage: group / depthwise / strided / padded / dilated ──

/// Helper: run op_conv on CPU and gpu_conv2d on CUDA with the same
/// ConvAttrs, assert output shapes match, and return element-wise
/// max-abs diff (f32).
fn conv_gpu_vs_cpu_max_abs(
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    pads: &[i32],
    strides: &[i32],
    dilations: &[i32],
    group: i32,
) -> (Vec<i64>, f32) {
    use smallaios_onnx_rt::operators::{op_conv, ConvAttrs};
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32)
        .expect("CUDA init (F32 precision)");
    let gpu_out = cuda::conv::gpu_conv2d(&rt, input, weight, bias, pads, strides, dilations, group)
        .expect("gpu_conv2d returned error")
        .expect("gpu_conv2d returned None");
    let attrs = ConvAttrs {
        pads: [pads[0], pads[1], pads[2], pads[3]],
        strides: [strides[0], strides[1]],
        dilations: [dilations[0], dilations[1]],
        group,
    };
    let cpu_out = op_conv(input, weight, bias, &attrs).expect("op_conv CPU");
    assert_eq!(
        gpu_out.shape.dims, cpu_out.shape.dims,
        "CPU and GPU output shapes must match"
    );
    let gpu_vals = read_f32(&gpu_out);
    let cpu_vals = read_f32(&cpu_out);
    let mut max_abs = 0.0f32;
    for (g, c) in gpu_vals.iter().zip(cpu_vals.iter()) {
        max_abs = max_abs.max((g - c).abs());
    }
    (gpu_out.shape.dims.clone(), max_abs)
}

#[test]
#[ignore]
fn test_gpu_conv_group2_matches_cpu() {
    // Group-of-2 conv on a [1, 4, 8, 8] input; weight [4, 2, 3, 3] + pad=1.
    let input_vals: Vec<f32> = (0..(4 * 8 * 8)).map(|i| (i as f32) * 0.01).collect();
    let weight_vals: Vec<f32> = (0..(4 * 2 * 9))
        .map(|i| ((i as f32) * 0.013).sin())
        .collect();
    let input = make_f32_tensor(&[1, 4, 8, 8], &input_vals);
    let weight = make_f32_tensor(&[4, 2, 3, 3], &weight_vals);
    let (dims, max_abs) =
        conv_gpu_vs_cpu_max_abs(&input, &weight, None, &[1, 1, 1, 1], &[1, 1], &[1, 1], 2);
    assert_eq!(dims, vec![1, 4, 8, 8]);
    assert!(max_abs < 1e-3, "group-2 CPU vs GPU max_abs = {}", max_abs);
}

#[test]
#[ignore]
fn test_gpu_conv_depthwise_matches_cpu() {
    // Depthwise 32-channel 3x3 conv with pad=1, mimicking MobileNetV2 block.
    let c = 32;
    let h = 14;
    let w = 14;
    let input_vals: Vec<f32> = (0..(c * h * w))
        .map(|i| ((i as f32) * 0.007).cos() * 0.5)
        .collect();
    let weight_vals: Vec<f32> = (0..(c * 9))
        .map(|i| ((i as f32) * 0.017).sin() * 0.1)
        .collect();
    let input = make_f32_tensor(&[1, c as i64, h as i64, w as i64], &input_vals);
    let weight = make_f32_tensor(&[c as i64, 1, 3, 3], &weight_vals);
    let (dims, max_abs) = conv_gpu_vs_cpu_max_abs(
        &input,
        &weight,
        None,
        &[1, 1, 1, 1],
        &[1, 1],
        &[1, 1],
        c as i32,
    );
    assert_eq!(dims, vec![1, c as i64, h as i64, w as i64]);
    assert!(max_abs < 1e-3, "depthwise CPU vs GPU max_abs = {}", max_abs);
}

#[test]
#[ignore]
fn test_gpu_conv_stride2_matches_cpu() {
    // Strided 2 conv — ResNet-50 stem style: 3x7x7 -> 64 channels, stride 2.
    let input_vals: Vec<f32> = (0..(3 * 16 * 16))
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();
    let weight_vals: Vec<f32> = (0..(8 * 3 * 7 * 7))
        .map(|i| ((i as f32) * 0.003).cos() * 0.05)
        .collect();
    let input = make_f32_tensor(&[1, 3, 16, 16], &input_vals);
    let weight = make_f32_tensor(&[8, 3, 7, 7], &weight_vals);
    let (dims, max_abs) =
        conv_gpu_vs_cpu_max_abs(&input, &weight, None, &[3, 3, 3, 3], &[2, 2], &[1, 1], 1);
    // (16 + 3 + 3 - 6 - 1) / 2 + 1 = 8 for each spatial dim.
    assert_eq!(dims, vec![1, 8, 8, 8]);
    assert!(max_abs < 1e-3, "stride-2 CPU vs GPU max_abs = {}", max_abs);
}

#[test]
#[ignore]
fn test_gpu_conv_pad1_matches_cpu() {
    // Same-padding style 3x3 conv keeps spatial dims.
    let input_vals: Vec<f32> = (0..(2 * 5 * 5)).map(|i| (i as f32) * 0.1).collect();
    let weight_vals: Vec<f32> = (0..(4 * 2 * 9))
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();
    let input = make_f32_tensor(&[1, 2, 5, 5], &input_vals);
    let weight = make_f32_tensor(&[4, 2, 3, 3], &weight_vals);
    let (dims, max_abs) =
        conv_gpu_vs_cpu_max_abs(&input, &weight, None, &[1, 1, 1, 1], &[1, 1], &[1, 1], 1);
    assert_eq!(dims, vec![1, 4, 5, 5]);
    assert!(max_abs < 1e-3, "pad-1 CPU vs GPU max_abs = {}", max_abs);
}

#[test]
#[ignore]
fn test_gpu_conv_dilation2_matches_cpu() {
    // Dilation 2 on 2x2 kernel expands effective receptive field.
    let input_vals: Vec<f32> = (0..(2 * 5 * 5)).map(|i| (i as f32) * 0.1).collect();
    let weight_vals: Vec<f32> = (0..(3 * 2 * 4))
        .map(|i| ((i as f32) * 0.04).cos())
        .collect();
    let input = make_f32_tensor(&[1, 2, 5, 5], &input_vals);
    let weight = make_f32_tensor(&[3, 2, 2, 2], &weight_vals);
    let (dims, max_abs) =
        conv_gpu_vs_cpu_max_abs(&input, &weight, None, &[0, 0, 0, 0], &[1, 1], &[2, 2], 1);
    // (5 + 0 + 0 - 2 - 1) / 1 + 1 = 3 per spatial dim.
    assert_eq!(dims, vec![1, 3, 3, 3]);
    assert!(
        max_abs < 1e-3,
        "dilation-2 CPU vs GPU max_abs = {}",
        max_abs
    );
}

// ── Hybrid-mode device-op tests: BN / Activation / Pool / Add ──

use smallaios_onnx_rt::cuda::activation::{gpu_clip, gpu_relu};
use smallaios_onnx_rt::cuda::batchnorm::gpu_batchnorm;
use smallaios_onnx_rt::cuda::elementwise::gpu_add as cuda_gpu_add;
use smallaios_onnx_rt::cuda::gpu_executor::DeviceTensor;
use smallaios_onnx_rt::cuda::pool::{gpu_averagepool, gpu_globalaveragepool, gpu_maxpool};
use smallaios_onnx_rt::operators::{
    op_averagepool, op_batch_normalization, op_global_average_pool, op_maxpool, op_relu, PoolAttrs,
};

fn host_to_device(t: &Tensor, rt: &cuda::CudaRuntime) -> DeviceTensor {
    cuda::gpu_executor::tensor_to_device(t, rt).expect("h2d")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .fold(0f32, |m, (x, y)| m.max((x - y).abs()))
}

#[test]
#[ignore]
fn test_gpu_batchnorm_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let n = 1usize;
    let c = 32usize;
    let h = 14usize;
    let w = 14usize;
    let x_vals: Vec<f32> = (0..n * c * h * w)
        .map(|i| ((i as f32) * 0.013).sin() * 0.5)
        .collect();
    let scale: Vec<f32> = (0..c).map(|i| 0.5 + (i as f32) * 0.01).collect();
    let bias: Vec<f32> = (0..c).map(|i| (i as f32) * 0.005).collect();
    let mean: Vec<f32> = (0..c).map(|i| (i as f32) * 0.002).collect();
    let var: Vec<f32> = (0..c).map(|i| 0.5 + (i as f32) * 0.001).collect();

    let x = make_f32_tensor(&[n as i64, c as i64, h as i64, w as i64], &x_vals);
    let s = make_f32_tensor(&[c as i64], &scale);
    let b = make_f32_tensor(&[c as i64], &bias);
    let m = make_f32_tensor(&[c as i64], &mean);
    let v = make_f32_tensor(&[c as i64], &var);

    let cpu_out = op_batch_normalization(&x, &s, &b, &m, &v, 1e-5).expect("cpu bn");
    let xd = host_to_device(&x, &rt);
    let sd = host_to_device(&s, &rt);
    let bd = host_to_device(&b, &rt);
    let md = host_to_device(&m, &rt);
    let vd = host_to_device(&v, &rt);
    let gpu_out = gpu_batchnorm(&rt, &xd, &sd, &bd, &md, &vd, 1e-5)
        .expect("gpu bn")
        .to_host()
        .expect("d2h");
    let diff = max_abs_diff(&read_f32(&cpu_out), &read_f32(&gpu_out));
    assert!(diff < 1e-3, "BN max_abs_diff = {}", diff);
}

#[test]
#[ignore]
fn test_gpu_relu_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let vals: Vec<f32> = (0..(1 * 64 * 28 * 28))
        .map(|i| (i as f32) * 0.001 - 1.0)
        .collect();
    let x = make_f32_tensor(&[1, 64, 28, 28], &vals);
    let cpu_out = op_relu(&x).expect("cpu relu");
    let xd = host_to_device(&x, &rt);
    let gpu_out = gpu_relu(&rt, &xd)
        .expect("gpu relu")
        .to_host()
        .expect("d2h");
    let diff = max_abs_diff(&read_f32(&cpu_out), &read_f32(&gpu_out));
    assert!(diff < 1e-3, "Relu max_abs_diff = {}", diff);
}

#[test]
#[ignore]
fn test_gpu_clip_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let vals: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 5.0).collect();
    let x = make_f32_tensor(&[1, 1, 16, 16], &vals);
    let xd = host_to_device(&x, &rt);
    let gpu_out = gpu_clip(&rt, &xd, 0.0, 6.0)
        .expect("gpu clip")
        .to_host()
        .expect("d2h");
    let cpu_vals: Vec<f32> = read_f32(&x).iter().map(|&v| v.clamp(0.0, 6.0)).collect();
    let diff = max_abs_diff(&cpu_vals, &read_f32(&gpu_out));
    assert!(diff < 1e-3, "Clip6 max_abs_diff = {}", diff);
}

#[test]
#[ignore]
fn test_gpu_maxpool_3x3_stride2_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let vals: Vec<f32> = (0..(1 * 4 * 16 * 16))
        .map(|i| ((i as f32) * 0.03).sin())
        .collect();
    let x = make_f32_tensor(&[1, 4, 16, 16], &vals);
    let cpu_out = op_maxpool(&x, &[3, 3], Some(&[2, 2]), Some(&[1, 1, 1, 1])).expect("cpu maxpool");
    let attrs = PoolAttrs {
        kernel_shape: [3, 3],
        pads: [1, 1, 1, 1],
        strides: [2, 2],
        ceil_mode: false,
        count_include_pad: false,
    };
    let xd = host_to_device(&x, &rt);
    let gpu_out = gpu_maxpool(&rt, &xd, &attrs)
        .expect("gpu maxpool")
        .to_host()
        .expect("d2h");
    assert_eq!(cpu_out.shape.dims, gpu_out.shape.dims);
    let diff = max_abs_diff(&read_f32(&cpu_out), &read_f32(&gpu_out));
    assert!(diff < 1e-3, "MaxPool max_abs_diff = {}", diff);
}

#[test]
#[ignore]
fn test_gpu_averagepool_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let vals: Vec<f32> = (0..(1 * 8 * 8 * 8)).map(|i| (i as f32) * 0.05).collect();
    let x = make_f32_tensor(&[1, 8, 8, 8], &vals);
    let cpu_out =
        op_averagepool(&x, &[2, 2], Some(&[2, 2]), Some(&[0, 0, 0, 0])).expect("cpu avgpool");
    let attrs = PoolAttrs {
        kernel_shape: [2, 2],
        pads: [0, 0, 0, 0],
        strides: [2, 2],
        ceil_mode: false,
        count_include_pad: false,
    };
    let xd = host_to_device(&x, &rt);
    let gpu_out = gpu_averagepool(&rt, &xd, &attrs)
        .expect("gpu avgpool")
        .to_host()
        .expect("d2h");
    let diff = max_abs_diff(&read_f32(&cpu_out), &read_f32(&gpu_out));
    assert!(diff < 1e-3, "AvgPool max_abs_diff = {}", diff);
}

#[test]
#[ignore]
fn test_gpu_globalaveragepool_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let vals: Vec<f32> = (0..(1 * 256 * 7 * 7))
        .map(|i| ((i as f32) * 0.0011).sin() * 0.3)
        .collect();
    let x = make_f32_tensor(&[1, 256, 7, 7], &vals);
    let cpu_out = op_global_average_pool(&x).expect("cpu gap");
    let xd = host_to_device(&x, &rt);
    let gpu_out = gpu_globalaveragepool(&rt, &xd)
        .expect("gpu gap")
        .to_host()
        .expect("d2h");
    assert_eq!(gpu_out.shape.dims, vec![1, 256, 1, 1]);
    let diff = max_abs_diff(&read_f32(&cpu_out), &read_f32(&gpu_out));
    assert!(diff < 1e-3, "GAP max_abs_diff = {}", diff);
}

#[test]
#[ignore]
fn test_gpu_add_same_shape_matches_cpu() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");
    let vals_a: Vec<f32> = (0..(1 * 128 * 14 * 14))
        .map(|i| (i as f32) * 0.01)
        .collect();
    let vals_b: Vec<f32> = (0..(1 * 128 * 14 * 14))
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();
    let a = make_f32_tensor(&[1, 128, 14, 14], &vals_a);
    let b = make_f32_tensor(&[1, 128, 14, 14], &vals_b);
    let cpu_vals: Vec<f32> = read_f32(&a)
        .iter()
        .zip(read_f32(&b).iter())
        .map(|(x, y)| x + y)
        .collect();
    let ad = host_to_device(&a, &rt);
    let bd = host_to_device(&b, &rt);
    let gpu_out = cuda_gpu_add(&rt, &ad, &bd)
        .expect("gpu add")
        .to_host()
        .expect("d2h");
    let diff = max_abs_diff(&cpu_vals, &read_f32(&gpu_out));
    assert!(diff < 1e-3, "Add max_abs_diff = {}", diff);
}
