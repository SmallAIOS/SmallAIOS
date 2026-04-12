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

    buf.copy_from_host(&data).expect("host-to-device copy failed");

    let mut result = vec![0u8; 256];
    buf.copy_to_host(&mut result).expect("device-to-host copy failed");

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
            i, exp, got
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
            i, exp, got
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
    assert!(
        max_err < 0.05,
        "GPU/CPU mismatch too large: {}",
        max_err
    );
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

    let c_vals: Vec<i32> = c.raw_data
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
    assert!(result.is_none(), "3x3 should fall back to CPU (not 4-aligned)");
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
        if err > max_err { max_err = err; }
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
    let x = make_f32_tensor(&[1, 1, 3, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let w = make_f32_tensor(&[1, 1, 1, 1], &[2.0]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[1, 1])
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 1, 3, 3]);
    let vals = read_f32(&y);
    let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "conv 1x1 [{}]: {} vs {}", i, got, exp);
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[1, 1, 1, 1], &[1, 1], &[1, 1])
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    // Output should be [1,1,4,4] (same spatial due to pad=1)
    assert_eq!(y.shape.dims, vec![1, 1, 4, 4]);
    let vals = read_f32(&y);

    // Spot check: center element y[0,0,1,1] = sum of 3x3 patch around (1,1)
    // = 1+2+3+5+6+7+9+10+11 = 54
    assert!((vals[5] - 54.0).abs() < 1e-4, "conv center: {} vs 54", vals[5]);
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, Some(&bias), &[0, 0], &[1, 1], &[1, 1])
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 2, 2, 2]);
    let vals = read_f32(&y);
    // Chan 0: [1+10, 2+10, 3+10, 4+10] = [11, 12, 13, 14]
    // Chan 1: [-1+20, -2+20, -3+20, -4+20] = [19, 18, 17, 16]
    let expected = [11.0, 12.0, 13.0, 14.0, 19.0, 18.0, 17.0, 16.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-4, "conv+bias [{}]: {} vs {}", i, got, exp);
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[2, 2], &[1, 1])
        .expect("gpu_conv2d failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 1, 2, 2]);
    let vals = read_f32(&y);
    // Stride=2 picks every other element: [1, 3, 9, 11]
    let expected = [1.0, 3.0, 9.0, 11.0];
    for (i, (&got, &exp)) in vals.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-5, "strided conv [{}]: {} vs {}", i, got, exp);
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[2, 2])
        .expect("gpu_conv2d dilated failed");
    assert!(result.is_some());
    let y = result.unwrap();

    assert_eq!(y.shape.dims, vec![1, 1, 1, 1]);
    let vals = read_f32(&y);
    // Dilated 3x3 with dilation=2 picks positions (0,0),(0,2),(0,4),(2,0),(2,2),(2,4),(4,0),(4,2),(4,4)
    // = 1+3+5+11+13+15+21+23+25 = 117
    assert!((vals[0] - 117.0).abs() < 1e-4, "dilated conv: {} vs 117", vals[0]);
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[1, 1, 1, 1], &[1, 1], &[1, 1])
        .expect("gpu_conv2d TF32 failed");
    assert!(result.is_some());
    let y = result.unwrap();
    assert_eq!(y.shape.dims, vec![1, 1, 4, 4]);

    let vals = read_f32(&y);
    // TF32 may have slightly different rounding but should be close.
    // Center element should be ~54.
    assert!((vals[5] - 54.0).abs() < 1.0, "TF32 conv center: {} vs 54", vals[5]);
    eprintln!("gpu_conv2d TF32: OK, center={}", vals[5]);
}

#[test]
#[ignore]
fn test_gpu_conv2d_non4d_falls_back() {
    let rt = cuda::CudaRuntime::init_with_precision(cuda::GpuPrecision::F32).expect("CUDA init");

    // 3D input — should return None (not a 4D conv).
    let x = make_f32_tensor(&[1, 3, 5], &vec![1.0; 15]);
    let w = make_f32_tensor(&[1, 3, 3], &vec![1.0; 9]);

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[1, 1])
        .expect("should not error");
    assert!(result.is_none(), "3D conv should fall back to CPU");
    eprintln!("gpu_conv2d 3D: correctly falls back to CPU");
}

// ── FP8 tests ───────────────────────────────────────────────────────

/// Encode an f32 value to FP8 E4M3 format (1 byte).
/// E4M3: sign(1) + exponent(4) + mantissa(3), bias=7, max=448.
fn f32_to_fp8_e4m3(val: f32) -> u8 {
    // Simplified conversion: only handles positive values 0-448.
    if val == 0.0 { return 0; }
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
        data_type: DataType::Float, // FP8 not yet in DataType enum, use Float as placeholder
        shape: TensorShape::new(vec![n as i64, n as i64]),
        name: String::new(),
        raw_data: a_data,
    };
    let b = Tensor {
        data_type: DataType::Float,
        shape: TensorShape::new(vec![n as i64, n as i64]),
        name: String::new(),
        raw_data: b_data,
    };

    let result = cuda::dispatch::gpu_gemm_fp8(
        &rt, &a, &b,
        cuda::ffi::cudaDataType_t::CUDA_R_8F_E4M3,
    ).expect("gpu_gemm_fp8 failed");

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

    let a_data: [i8; 16] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16];
    let b_data: [i8; 16] = [16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1];

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

    cublas.gemm_ex(
        cuda::ffi::cublasOperation_t::CUBLAS_OP_N,
        cuda::ffi::cublasOperation_t::CUBLAS_OP_N,
        4, 4, 4,
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
    ).expect("cublasGemmEx INT8 failed");

    cuda::synchronize().unwrap();

    let mut result = [0u8; 64];
    c_buf.copy_to_host(&mut result).unwrap();
    let c_vals: Vec<i32> = result.chunks_exact(4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    // Row 0 should be: 80, 70, 60, 50
    assert_eq!(c_vals[0], 80);
    assert_eq!(c_vals[1], 70);
    eprintln!("Raw cublasGemmEx INT8: OK {:?}", &c_vals[..4]);
}
