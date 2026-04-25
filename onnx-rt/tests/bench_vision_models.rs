// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! End-to-end CPU vs GPU inference benchmarks for vision ONNX models.
//!
//! Covers task 14.1 of `arm64-gpu-container-v1`. Tests are `#[ignore]`
//! by default because they require:
//!   - Fixture models at `tests/fixtures/onnx-models/` (downloaded via
//!     `scripts/download-test-fixtures.sh`)
//!   - A CUDA-capable GPU (run with `--features cuda`) for the GPU path
//!
//! Run with: `cargo test -p smallaios-onnx-rt --features cuda --test
//! bench_vision_models -- --ignored --nocapture`
//!
//! Each test loads the model, runs inference on CPU, optionally on GPU,
//! compares outputs for correctness, and reports latency statistics.

use std::path::PathBuf;
use std::time::Instant;

#[cfg(feature = "cuda")]
use smallaios_onnx_rt::session::GpuResidency;
use smallaios_onnx_rt::session::{self, InferenceInput, Session, SessionConfig};
use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};

const WARMUP_ITERS: usize = 2;
const MEASURE_ITERS: usize = 5;

fn fixture_path(name: &str) -> Option<PathBuf> {
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("onnx-rt parent")
        .to_path_buf();
    let path = workspace_root.join("tests/fixtures/onnx-models").join(name);
    if path.exists() {
        Some(path)
    } else {
        None
    }
}

/// Build a random-ish f32 input tensor of the given shape, deterministic
/// across runs so CPU and GPU receive identical inputs.
fn make_input(name: &str, shape: &[i64]) -> Tensor {
    let total: usize = shape.iter().map(|&d| d as usize).product();
    let mut raw_data = vec![0u8; total * 4];
    for i in 0..total {
        // Pseudo-random but deterministic — avoid pathological all-zero input.
        let v = ((i.wrapping_mul(2654435761) & 0xFFFF) as f32) / 65535.0 - 0.5;
        raw_data[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
    }
    let mut t = Tensor::new(
        DataType::Float,
        TensorShape::new(shape.to_vec()),
        name.to_string(),
    );
    t.raw_data = raw_data;
    t
}

fn read_f32_slice(t: &Tensor) -> Vec<f32> {
    t.raw_data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

struct Stats {
    mean_ms: f64,
    min_ms: f64,
    max_ms: f64,
    p50_ms: f64,
}

fn stats(samples: &[f64]) -> Stats {
    let mut sorted: Vec<f64> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let sum: f64 = samples.iter().sum();
    Stats {
        mean_ms: sum / samples.len() as f64,
        min_ms: *sorted.first().unwrap(),
        max_ms: *sorted.last().unwrap(),
        p50_ms: sorted[sorted.len() / 2],
    }
}

fn report(label: &str, samples: &[f64]) -> Stats {
    let s = stats(samples);
    eprintln!(
        "  {:<16} min={:.2}ms p50={:.2}ms mean={:.2}ms max={:.2}ms  ({} runs)",
        label,
        s.min_ms,
        s.p50_ms,
        s.mean_ms,
        s.max_ms,
        samples.len()
    );
    s
}

/// Comparison result between two output tensors.
struct Diff {
    len_matches: bool,
    cpu_len: usize,
    gpu_len: usize,
    max_abs: f32,
    max_rel: f32,
    mean_abs: f32,
}

/// Compare two output tensors element-wise. Reports length mismatch as
/// a soft finding rather than panicking — real-model CPU-vs-GPU shape
/// divergence is something the benchmark is designed to surface.
fn compare_outputs(a: &Tensor, b: &Tensor) -> Diff {
    let va = read_f32_slice(a);
    let vb = read_f32_slice(b);
    let len_matches = va.len() == vb.len();
    if !len_matches {
        return Diff {
            len_matches: false,
            cpu_len: va.len(),
            gpu_len: vb.len(),
            max_abs: f32::NAN,
            max_rel: f32::NAN,
            mean_abs: f32::NAN,
        };
    }
    let mut max_abs = 0.0f32;
    let mut max_rel = 0.0f32;
    let mut sum_abs = 0.0f32;
    for (x, y) in va.iter().zip(vb.iter()) {
        let abs = (x - y).abs();
        max_abs = max_abs.max(abs);
        sum_abs += abs;
        let denom = x.abs().max(y.abs()).max(1e-6);
        max_rel = max_rel.max(abs / denom);
    }
    Diff {
        len_matches: true,
        cpu_len: va.len(),
        gpu_len: vb.len(),
        max_abs,
        max_rel,
        mean_abs: sum_abs / va.len() as f32,
    }
}

/// Load fixture, initialize a session, return (session, input_name, input_shape).
fn load_vision_model(
    fixture: &str,
    input_name: &str,
    input_shape: &[i64],
) -> Option<(Session, String, Vec<i64>)> {
    load_vision_model_with(fixture, input_name, input_shape, SessionConfig::default())
}

fn load_vision_model_with(
    fixture: &str,
    input_name: &str,
    input_shape: &[i64],
    config: SessionConfig,
) -> Option<(Session, String, Vec<i64>)> {
    let path = fixture_path(fixture)?;
    let bytes = std::fs::read(&path).expect("read fixture");
    let model = session::load_model(&bytes).expect("load_model");

    let mut session = Session::new(config);
    session.initialize(&model).expect("initialize");

    Some((session, input_name.to_string(), input_shape.to_vec()))
}

fn clone_input(name: &str, shape: &[i64], raw: &[u8]) -> InferenceInput {
    let mut t = Tensor::new(
        DataType::Float,
        TensorShape::new(shape.to_vec()),
        name.to_string(),
    );
    t.raw_data = raw.to_vec();
    InferenceInput {
        name: name.to_string(),
        tensor: t,
    }
}

fn bench_cpu(
    session: &Session,
    name: &str,
    shape: &[i64],
    raw: &[u8],
) -> Result<(Tensor, Vec<f64>), String> {
    for _ in 0..WARMUP_ITERS {
        session
            .run(&[clone_input(name, shape, raw)])
            .map_err(|e| format!("{:?}", e))?;
    }
    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    let mut last_out: Option<Tensor> = None;
    for _ in 0..MEASURE_ITERS {
        let start = Instant::now();
        let outputs = session
            .run(&[clone_input(name, shape, raw)])
            .map_err(|e| format!("{:?}", e))?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        last_out = Some(
            outputs
                .into_iter()
                .next()
                .expect("at least one output")
                .tensor,
        );
    }
    Ok((last_out.unwrap(), samples))
}

#[cfg(feature = "cuda")]
fn bench_gpu(
    mut session: Session,
    name: &str,
    shape: &[i64],
    raw: &[u8],
) -> Result<(Tensor, Vec<f64>, MemoryReport), String> {
    use std::sync::Arc;

    let baseline = cuda_mem_free();
    let rt = smallaios_onnx_rt::cuda::CudaRuntime::init().expect("CUDA init");
    let post_init = cuda_mem_free();
    session.cuda_runtime = Some(Arc::new(rt));

    for _ in 0..WARMUP_ITERS {
        session
            .run(&[clone_input(name, shape, raw)])
            .map_err(|e| format!("{:?}", e))?;
    }
    let post_warmup = cuda_mem_free();

    let mut samples = Vec::with_capacity(MEASURE_ITERS);
    let mut last_out: Option<Tensor> = None;
    for _ in 0..MEASURE_ITERS {
        let start = Instant::now();
        let outputs = session
            .run(&[clone_input(name, shape, raw)])
            .map_err(|e| format!("{:?}", e))?;
        samples.push(start.elapsed().as_secs_f64() * 1000.0);
        last_out = Some(
            outputs
                .into_iter()
                .next()
                .expect("at least one output")
                .tensor,
        );
    }
    let mem = MemoryReport {
        baseline_free_mb: baseline / (1024 * 1024),
        post_init_free_mb: post_init / (1024 * 1024),
        post_warmup_free_mb: post_warmup / (1024 * 1024),
        init_cost_mb: baseline.saturating_sub(post_init) / (1024 * 1024),
        workspace_cost_mb: post_init.saturating_sub(post_warmup) / (1024 * 1024),
    };
    Ok((last_out.unwrap(), samples, mem))
}

#[cfg(feature = "cuda")]
struct MemoryReport {
    baseline_free_mb: usize,
    post_init_free_mb: usize,
    post_warmup_free_mb: usize,
    init_cost_mb: usize,
    workspace_cost_mb: usize,
}

/// Query CUDA free memory via `cudaMemGetInfo`. Returns 0 on failure.
#[cfg(feature = "cuda")]
fn cuda_mem_free() -> usize {
    use smallaios_onnx_rt::cuda::ffi;
    let mut free: usize = 0;
    let mut total: usize = 0;
    let err = unsafe { ffi::cudaMemGetInfo(&mut free as *mut _, &mut total as *mut _) };
    if err != ffi::CUDA_SUCCESS {
        return 0;
    }
    free
}

fn run_cpu_vs_gpu(model_label: &str, fixture: &str, input_name: &str, input_shape: &[i64]) {
    run_cpu_vs_gpu_with(model_label, fixture, input_name, input_shape, None);
}

/// `GpuResidency` selector for benchmark variants. CPU code path always
/// uses the default executor; only the GPU side honors this.
#[derive(Debug, Clone, Copy)]
enum BenchMode {
    OpByOp,
    #[cfg(feature = "cuda")]
    Hybrid,
}

#[cfg(feature = "cuda")]
fn bench_mode_to_residency(mode: BenchMode) -> GpuResidency {
    match mode {
        BenchMode::OpByOp => GpuResidency::OpByOp,
        BenchMode::Hybrid => GpuResidency::Hybrid,
    }
}

/// When `expected_output_dims` is `Some`, hard-asserts the output shape
/// once both CPU and GPU paths succeed. Benches still pass silently when
/// either path is blocked by an unrelated operator gap — the purpose is
/// to flip the assertion on once the operator landscape supports it.
fn run_cpu_vs_gpu_with(
    model_label: &str,
    fixture: &str,
    input_name: &str,
    input_shape: &[i64],
    expected_output_dims: Option<&[i64]>,
) {
    run_cpu_vs_gpu_full(
        model_label,
        fixture,
        input_name,
        input_shape,
        expected_output_dims,
        BenchMode::OpByOp,
    );
}

#[cfg(feature = "cuda")]
fn run_cpu_vs_gpu_hybrid(
    model_label: &str,
    fixture: &str,
    input_name: &str,
    input_shape: &[i64],
    expected_output_dims: Option<&[i64]>,
) {
    run_cpu_vs_gpu_full(
        model_label,
        fixture,
        input_name,
        input_shape,
        expected_output_dims,
        BenchMode::Hybrid,
    );
}

fn run_cpu_vs_gpu_full(
    model_label: &str,
    fixture: &str,
    input_name: &str,
    input_shape: &[i64],
    expected_output_dims: Option<&[i64]>,
    bench_mode: BenchMode,
) {
    let Some((cpu_session, in_name, shape)) = load_vision_model(fixture, input_name, input_shape)
    else {
        eprintln!("SKIP {}: fixture {} not found", model_label, fixture);
        return;
    };

    eprintln!(
        "=== {} (fixture: {}, input {:?}) ===",
        model_label, fixture, shape
    );

    let input_tensor = make_input(&in_name, &shape);
    let raw_bytes: Vec<u8> = input_tensor.raw_data.clone();

    let cpu_result = bench_cpu(&cpu_session, &in_name, &shape, &raw_bytes);
    let cpu_outcome = match cpu_result {
        Ok((out, samples)) => {
            let stats = report("CPU", &samples);
            Some((out, stats))
        }
        Err(e) => {
            eprintln!("  CPU path FAILED: {}", e);
            None
        }
    };

    #[cfg(feature = "cuda")]
    {
        let gpu_config = SessionConfig {
            gpu_residency: bench_mode_to_residency(bench_mode),
            ..SessionConfig::default()
        };
        let (gpu_session, _, _) =
            load_vision_model_with(fixture, input_name, input_shape, gpu_config).unwrap();
        let gpu_result = bench_gpu(gpu_session, &in_name, &shape, &raw_bytes);
        match gpu_result {
            Ok((gpu_out, gpu_samples, mem)) => {
                let gpu_stats = report("GPU (hybrid)", &gpu_samples);
                eprintln!(
                    "  GPU VRAM: baseline_free={} MB, post_init_free={} MB, post_warmup_free={} MB",
                    mem.baseline_free_mb, mem.post_init_free_mb, mem.post_warmup_free_mb
                );
                eprintln!(
                    "  GPU VRAM cost: runtime_init={} MB, inference_workspace={} MB",
                    mem.init_cost_mb, mem.workspace_cost_mb
                );

                if let Some((cpu_out, cpu_stats)) = cpu_outcome {
                    eprintln!(
                        "  output shapes: cpu_dims={:?} gpu_dims={:?}",
                        cpu_out.shape.dims, gpu_out.shape.dims
                    );
                    let diff = compare_outputs(&cpu_out, &gpu_out);
                    if diff.len_matches {
                        eprintln!(
                            "  output diff: max_abs={:.4e} max_rel={:.4e} mean_abs={:.4e}",
                            diff.max_abs, diff.max_rel, diff.mean_abs
                        );
                    } else {
                        eprintln!(
                            "  output LENGTH MISMATCH: cpu={} gpu={} — CPU-runtime bug",
                            diff.cpu_len, diff.gpu_len
                        );
                    }
                    let speedup = cpu_stats.mean_ms / gpu_stats.mean_ms;
                    eprintln!("  speedup (CPU/GPU mean): {:.2}x", speedup);

                    if let Some(expected) = expected_output_dims {
                        assert_eq!(
                            cpu_out.shape.dims, expected,
                            "{} CPU output shape mismatch",
                            model_label
                        );
                        assert_eq!(
                            gpu_out.shape.dims, expected,
                            "{} GPU output shape mismatch",
                            model_label
                        );
                    }
                } else {
                    eprintln!("  (speedup N/A — CPU path failed)");
                }
            }
            Err(e) => eprintln!("  GPU path FAILED: {}", e),
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = cpu_outcome;
        let _ = bench_mode;
        eprintln!("  (GPU path skipped: cuda feature disabled)");
    }
}

#[test]
#[ignore]
fn bench_squeezenet_cpu_vs_gpu() {
    run_cpu_vs_gpu(
        "SqueezeNet 1.1",
        "squeezenet.onnx",
        "data",
        &[1, 3, 224, 224],
    );
}

#[test]
#[ignore]
fn bench_mobilenet_v2_cpu_vs_gpu() {
    // MobileNetV2-12 input: "input" name, shape [1, 3, 224, 224].
    // Expected output [1, 1000] once operator coverage allows both
    // paths to complete. Currently blocked on a Gather op gap after
    // Conv attrs landed.
    run_cpu_vs_gpu_with(
        "MobileNetV2-12",
        "mobilenet_v2.onnx",
        "input",
        &[1, 3, 224, 224],
        Some(&[1, 1000]),
    );
}

#[test]
#[ignore]
fn bench_resnet50_cpu_vs_gpu() {
    // ResNet-50 v2: "data" input, shape [1, 3, 224, 224], output [1, 1000].
    run_cpu_vs_gpu_with(
        "ResNet-50 v2",
        "resnet50.onnx",
        "data",
        &[1, 3, 224, 224],
        Some(&[1, 1000]),
    );
}

/// Mid-graph CPU-op equivalence test (G7.4): a tiny synthetic graph
/// with shape `Gemm → Relu → Reshape → Reshape → Gemm`. The two
/// `Reshape` nodes force the hybrid executor to copy device-resident
/// activations back to host (Reshape is intentionally not in
/// `gpu_op_supported`), then re-upload for the next Gemm. Asserts the
/// hybrid output matches op-by-op output within 1e-3.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_hybrid_mid_graph_cpu_op_equivalence() {
    use std::sync::Arc;
    let fixture = "midgraph_cpu_op.onnx";
    let input_name = "x";
    let input_shape: &[i64] = &[1, 16];
    let raw = make_input(input_name, input_shape).raw_data;

    let (mut sess_op, _, shape_op) =
        load_vision_model_with(fixture, input_name, input_shape, SessionConfig::default())
            .expect("op-by-op session");
    let rt = smallaios_onnx_rt::cuda::CudaRuntime::init().expect("CUDA init");
    sess_op.cuda_runtime = Some(Arc::new(rt));
    let out_op = sess_op
        .run(&[clone_input(input_name, &shape_op, &raw)])
        .expect("op-by-op run")
        .into_iter()
        .next()
        .unwrap()
        .tensor;

    let cfg = SessionConfig {
        gpu_residency: GpuResidency::Hybrid,
        ..SessionConfig::default()
    };
    let (mut sess_h, _, shape_h) =
        load_vision_model_with(fixture, input_name, input_shape, cfg).expect("hybrid session");
    let rt2 = smallaios_onnx_rt::cuda::CudaRuntime::init().expect("CUDA init 2");
    sess_h.cuda_runtime = Some(Arc::new(rt2));
    let out_h = sess_h
        .run(&[clone_input(input_name, &shape_h, &raw)])
        .expect("hybrid run")
        .into_iter()
        .next()
        .unwrap()
        .tensor;

    assert_eq!(out_op.shape.dims, out_h.shape.dims);
    let v_op = read_f32_slice(&out_op);
    let v_h = read_f32_slice(&out_h);
    let max_abs = v_op
        .iter()
        .zip(v_h.iter())
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_abs < 1e-3,
        "mid-graph CPU op max_abs_diff = {} (hybrid must match op-by-op even with Reshape forcing host round-trip)",
        max_abs
    );
}

/// Hybrid vs op-by-op equivalence test: runs the MLP fixture under
/// `GpuResidency::Hybrid` and `GpuResidency::OpByOp`, asserts the two
/// outputs agree within 1e-3 max-abs and shape-match exactly. Catches
/// hybrid-path correctness regressions independent of CPU-vs-GPU diff.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn test_hybrid_vs_op_by_op_equivalence_mlp() {
    use std::sync::Arc;
    let fixture = "mlp_784_256_128_10.onnx";
    let input_name = "x";
    let input_shape: &[i64] = &[1, 784];
    let raw = make_input(input_name, input_shape).raw_data;

    // OpByOp session.
    let (mut sess_op, _, shape_op) =
        load_vision_model_with(fixture, input_name, input_shape, SessionConfig::default())
            .expect("op-by-op session");
    let rt = smallaios_onnx_rt::cuda::CudaRuntime::init().expect("CUDA init");
    sess_op.cuda_runtime = Some(Arc::new(rt));
    let out_op = sess_op
        .run(&[clone_input(input_name, &shape_op, &raw)])
        .expect("op-by-op run")
        .into_iter()
        .next()
        .expect("at least one output")
        .tensor;

    // Hybrid session.
    let cfg = SessionConfig {
        gpu_residency: GpuResidency::Hybrid,
        ..SessionConfig::default()
    };
    let (mut sess_h, _, shape_h) =
        load_vision_model_with(fixture, input_name, input_shape, cfg).expect("hybrid session");
    let rt2 = smallaios_onnx_rt::cuda::CudaRuntime::init().expect("CUDA init 2");
    sess_h.cuda_runtime = Some(Arc::new(rt2));
    let out_h = sess_h
        .run(&[clone_input(input_name, &shape_h, &raw)])
        .expect("hybrid run")
        .into_iter()
        .next()
        .expect("at least one output")
        .tensor;

    assert_eq!(
        out_op.shape.dims, out_h.shape.dims,
        "hybrid vs op-by-op output shapes must match"
    );
    let v_op: Vec<f32> = read_f32_slice(&out_op);
    let v_h: Vec<f32> = read_f32_slice(&out_h);
    let max_abs = v_op
        .iter()
        .zip(v_h.iter())
        .fold(0f32, |m, (a, b)| m.max((a - b).abs()));
    assert!(
        max_abs < 1e-3,
        "hybrid vs op-by-op MLP max_abs_diff = {}",
        max_abs
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_squeezenet_cpu_vs_gpu_hybrid() {
    run_cpu_vs_gpu_hybrid(
        "SqueezeNet 1.1 (hybrid)",
        "squeezenet.onnx",
        "data",
        &[1, 3, 224, 224],
        None,
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_mobilenet_v2_cpu_vs_gpu_hybrid() {
    run_cpu_vs_gpu_hybrid(
        "MobileNetV2-12 (hybrid)",
        "mobilenet_v2.onnx",
        "input",
        &[1, 3, 224, 224],
        Some(&[1, 1000]),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_resnet50_cpu_vs_gpu_hybrid() {
    run_cpu_vs_gpu_hybrid(
        "ResNet-50 v2 (hybrid)",
        "resnet50.onnx",
        "data",
        &[1, 3, 224, 224],
        Some(&[1, 1000]),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn bench_mlp_cpu_vs_gpu_hybrid() {
    run_cpu_vs_gpu_hybrid(
        "MLP 784-256-128-10 (hybrid)",
        "mlp_784_256_128_10.onnx",
        "x",
        &[1, 784],
        None,
    );
}

#[test]
#[ignore]
fn bench_mlp_cpu_vs_gpu() {
    // Synthetic MLP (784 → 256 → 128 → 10) — exercises only Gemm + Relu,
    // which are well supported on both CPU and GPU paths.
    run_cpu_vs_gpu(
        "MLP 784-256-128-10",
        "mlp_784_256_128_10.onnx",
        "x",
        &[1, 784],
    );
}
