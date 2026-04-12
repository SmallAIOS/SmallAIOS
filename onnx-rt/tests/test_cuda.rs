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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[1, 1])
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[1, 1, 1, 1], &[1, 1], &[1, 1])
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[2, 2], &[1, 1])
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[0, 0], &[1, 1], &[2, 2])
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

    let result = cuda::conv::gpu_conv2d(&rt, &x, &w, None, &[1, 1, 1, 1], &[1, 1], &[1, 1])
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
    let x_data: Vec<f32> = (0..(1 * 2 * 4 * 4))
        .map(|i| ((i % 11) as f32) * 0.125 - 0.5)
        .collect();
    let w_data: Vec<f32> = (0..(3 * 2 * 3 * 3))
        .map(|i| ((i % 5) as f32) * 0.0625 + 0.125)
        .collect();

    let x_bf = make_bf16_tensor(&[1, 2, 4, 4], &x_data);
    let w_bf = make_bf16_tensor(&[3, 2, 3, 3], &w_data);
    assert_eq!(x_bf.raw_data.len(), 1 * 2 * 4 * 4 * 2);
    assert_eq!(w_bf.raw_data.len(), 3 * 2 * 3 * 3 * 2);

    let result = cuda::conv::gpu_conv2d(&rt, &x_bf, &w_bf, None, &[1, 1, 1, 1], &[1, 1], &[1, 1])
        .expect("gpu_conv2d BF16 failed");
    let y_bf = result.expect("BF16 conv returned None");
    assert_eq!(y_bf.data_type, DataType::BFloat16);
    assert_eq!(y_bf.shape.dims, vec![1, 3, 4, 4]);
    assert_eq!(y_bf.raw_data.len(), 1 * 3 * 4 * 4 * 2);

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
    let kinds = vec![
        cuda::LayerKind::Global,
        cuda::LayerKind::SlidingWindow(4),
    ];

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

// ── Section 9.6: Session::from_safetensors end-to-end ──────────────────
//
// Builds a synthetic 2-layer Gemma-like safetensors directory in a temp
// dir, loads it via Session::from_safetensors, runs a single forward pass,
// and checks the logits shape. The underlying Gemma graph uses several
// operators (Gather, Add, Mul, RMSNormalization, RotaryEmbedding,
// GroupQueryAttention, Silu) that do NOT yet have GPU implementations in
// the Section 5 dispatcher (which only supports MatMul/Gemm/
// MatMulInteger/Conv). As a result this test currently exercises the
// session construction + dispatch plumbing and expects the forward pass
// to bottom out in an "no GPU implementation for {op}" error. When the
// GPU dispatch table is expanded (future sections), this assertion
// should be upgraded to a successful shape+NaN check on the logits.
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
        let mut f = fs::File::create(dir.join("model.safetensors"))
            .expect("create model.safetensors");
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

    // ── Run forward pass ──────────────────────────────────────────────
    //
    // The current GPU dispatcher only implements MatMul/Gemm/
    // MatMulInteger/Conv. The Gemma graph starts with a Gather (for
    // embedding lookup), so we expect an ExecutionFailed error whose
    // message contains "no GPU implementation for Gather". When the GPU
    // dispatcher is extended in a follow-up change, this assertion
    // should flip to an Ok() check on the output shape/dtype.
    let result = session.run(&inputs);
    match result {
        Ok(outputs) => {
            // If a future change makes this path succeed, verify shape.
            assert_eq!(outputs.len(), 1);
            let logits = &outputs[0];
            assert_eq!(
                logits.tensor.shape.dims,
                vec![1i64, 4, vocab as i64],
                "unexpected logits shape"
            );
            assert_eq!(logits.tensor.data_type, DataType::BFloat16);
            eprintln!("test_session_from_safetensors_synthetic_gemma: forward pass OK");
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                msg.contains("no GPU implementation"),
                "unexpected error from session.run: {msg}"
            );
            eprintln!(
                "test_session_from_safetensors_synthetic_gemma: \
                 expected unsupported-op error ({msg})"
            );
        }
    }

    // reset_kv_cache must succeed regardless of forward pass outcome.
    session.reset_kv_cache().expect("reset_kv_cache");

    // Cleanup the temp dir.
    let _ = fs::remove_dir_all(&dir);
}
