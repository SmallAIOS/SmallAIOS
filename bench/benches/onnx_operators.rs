// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for ONNX runtime operators.

extern crate alloc;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallaios_onnx_rt::operators::{op_add, op_matmul, op_relu, op_softmax};
use smallaios_onnx_rt::tensor::{DataType, Tensor, TensorShape};
use std::hint::black_box;

fn make_float_tensor(dims: &[i64]) -> Tensor {
    let shape = TensorShape::new(dims.to_vec());
    let n = shape.total_elements();
    let mut t = Tensor::new(DataType::Float, shape, alloc::string::String::new());
    t.raw_data = vec![0u8; n * 4];
    for i in 0..n {
        let val: f32 = (i % 100) as f32 * 0.01;
        t.raw_data[i * 4..i * 4 + 4].copy_from_slice(&val.to_le_bytes());
    }
    t
}

fn bench_matmul(c: &mut Criterion) {
    let mut group = c.benchmark_group("MatMul");
    for size in [8, 32, 64] {
        let a = make_float_tensor(&[size, size]);
        let b = make_float_tensor(&[size, size]);
        group.bench_with_input(BenchmarkId::new("square", size), &size, |bench, _| {
            bench.iter(|| op_matmul(black_box(&a), black_box(&b)));
        });
    }
    group.finish();
}

fn bench_relu(c: &mut Criterion) {
    let mut group = c.benchmark_group("Relu");
    for n in [64, 1024, 4096] {
        let t = make_float_tensor(&[n]);
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| op_relu(black_box(&t)));
        });
    }
    group.finish();
}

fn bench_softmax(c: &mut Criterion) {
    let mut group = c.benchmark_group("Softmax");
    for n in [10, 100, 1000] {
        let t = make_float_tensor(&[1, n]);
        group.bench_with_input(BenchmarkId::new("classes", n), &n, |bench, _| {
            bench.iter(|| op_softmax(black_box(&t), 1));
        });
    }
    group.finish();
}

fn bench_add(c: &mut Criterion) {
    let mut group = c.benchmark_group("Add");
    for n in [64, 1024, 4096] {
        let a = make_float_tensor(&[n]);
        let b = make_float_tensor(&[n]);
        group.bench_with_input(BenchmarkId::new("elements", n), &n, |bench, _| {
            bench.iter(|| op_add(black_box(&[&a, &b])));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_matmul, bench_relu, bench_softmax, bench_add);
criterion_main!(benches);
