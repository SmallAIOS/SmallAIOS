// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! CPU-vs-Metal comparison benchmarks for the Tier 1/Tier 2 GPU operators
//! that ship with `metal-operator-kernels-v1`. Built only with the `metal`
//! feature on macOS; on other platforms the bench is empty.
//!
//! Run with: `cargo bench -p smallaios-bench --features metal --bench metal_operators`

extern crate alloc;

#[cfg(all(feature = "metal", target_os = "macos"))]
mod gpu_benches {
    use criterion::{BenchmarkId, Criterion};
    use smallaios_onnx_rt::metal_dispatch::MetalDispatcher;
    use smallaios_onnx_rt::operators::op_matmul;
    use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};
    use std::hint::black_box;

    fn make_float_tensor(dims: &[i64], pattern_mod: usize) -> Tensor {
        let shape = TensorShape::new(dims.to_vec());
        let n = shape.total_elements();
        let mut t = Tensor::new(DataType::Float, shape, alloc::string::String::new());
        t.raw_data = vec![0u8; n * 4];
        for i in 0..n {
            let val: f32 = (i % pattern_mod) as f32 * 0.01;
            t.raw_data[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
        }
        t
    }

    /// Naive CPU SDPA reference matching the GPU kernel's causal-masked
    /// formulation. Not optimized — used only as a wall-clock baseline.
    fn cpu_sdpa(q: &Tensor, k: &Tensor, v: &Tensor, bh: usize, sq: usize, sk: usize, d: usize) {
        let qd: Vec<f32> = q
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let kd: Vec<f32> = k
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let vd: Vec<f32> = v
            .raw_data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        let scale = 1.0_f32 / (d as f32).sqrt();
        let mut out = vec![0.0f32; bh * sq * d];
        for h in 0..bh {
            for qi in 0..sq {
                let q_base = h * sq * d + qi * d;
                let causal_limit = qi + (sk - sq) + 1;
                let mut scores = vec![f32::NEG_INFINITY; sk];
                for j in 0..causal_limit.min(sk) {
                    let mut dot = 0.0f32;
                    for dd in 0..d {
                        dot += qd[q_base + dd] * kd[h * sk * d + j * d + dd];
                    }
                    scores[j] = dot * scale;
                }
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
                let sum_exp: f32 = exps.iter().sum();
                for dd in 0..d {
                    let mut acc = 0.0f32;
                    for j in 0..causal_limit.min(sk) {
                        let prob = exps[j] / sum_exp;
                        acc += prob * vd[h * sk * d + j * d + dd];
                    }
                    out[h * sq * d + qi * d + dd] = acc;
                }
            }
        }
        black_box(out);
    }

    pub fn bench_matmul_1024(c: &mut Criterion) {
        let mut group = c.benchmark_group("MatMul_1024x1024");
        let a = make_float_tensor(&[1024, 1024], 100);
        let b = make_float_tensor(&[1024, 1024], 100);

        group.bench_with_input(BenchmarkId::new("cpu", "f32"), &(), |bench, _| {
            bench.iter(|| op_matmul(black_box(&a), black_box(&b)));
        });

        let mut disp = MetalDispatcher::new().expect("Metal device required");
        // Warm up: compile shader, allocate buffers, populate cache.
        let _ = disp
            .try_execute("MatMul", &[Some(&a), Some(&b)], &[])
            .unwrap()
            .unwrap();
        group.bench_with_input(BenchmarkId::new("metal", "f32"), &(), |bench, _| {
            bench.iter(|| {
                let _ = disp
                    .try_execute(black_box("MatMul"), &[Some(&a), Some(&b)], &[])
                    .unwrap()
                    .unwrap();
            });
        });
        group.finish();
    }

    pub fn bench_sdpa(c: &mut Criterion) {
        // Per spec: batch=1, seq=512, heads=12, dim=64. Q/K/V layout is
        // [batch_heads, S, D] so bh = 1*12 = 12.
        let bh = 12usize;
        let sq = 512usize;
        let sk = 512usize;
        let d = 64usize;
        let q = make_float_tensor(&[bh as i64, sq as i64, d as i64], 100);
        let k = make_float_tensor(&[bh as i64, sk as i64, d as i64], 100);
        let v = make_float_tensor(&[bh as i64, sk as i64, d as i64], 100);

        let mut group = c.benchmark_group("SDPA_b1_s512_h12_d64");
        group.sample_size(10);

        group.bench_with_input(BenchmarkId::new("cpu", "naive"), &(), |bench, _| {
            bench.iter(|| cpu_sdpa(&q, &k, &v, bh, sq, sk, d));
        });

        let mut disp = MetalDispatcher::new().expect("Metal device required");
        let _ = disp
            .try_execute(
                "ScaledDotProductAttention",
                &[Some(&q), Some(&k), Some(&v)],
                &[],
            )
            .unwrap()
            .unwrap();
        group.bench_with_input(BenchmarkId::new("metal", "fused"), &(), |bench, _| {
            bench.iter(|| {
                let _ = disp
                    .try_execute(
                        black_box("ScaledDotProductAttention"),
                        &[Some(&q), Some(&k), Some(&v)],
                        &[],
                    )
                    .unwrap()
                    .unwrap();
            });
        });
        group.finish();
    }
}

#[cfg(all(feature = "metal", target_os = "macos"))]
fn main() {
    let mut criterion = criterion::Criterion::default().configure_from_args();
    gpu_benches::bench_matmul_1024(&mut criterion);
    gpu_benches::bench_sdpa(&mut criterion);
    criterion.final_summary();
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn main() {
    eprintln!(
        "metal_operators bench requires `--features metal` on macOS; skipping on this platform."
    );
}
