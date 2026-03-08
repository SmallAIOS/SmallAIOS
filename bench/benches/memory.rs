// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for kernel memory allocator.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use smallaios_kernel::mem::tensor::TensorPool;
use smallaios_kernel::mem::PhysAddr;

fn bench_tensor_alloc_free(c: &mut Criterion) {
    let mut group = c.benchmark_group("Memory/tensor_pool");
    // Create pool once outside the benchmark loop to avoid repeated 100KB stack alloc
    for alloc_size in [64, 4096, 65536] {
        group.bench_with_input(
            BenchmarkId::new("alloc_free", alloc_size),
            &alloc_size,
            |bench, &size| {
                let mut pool = Box::new(TensorPool::default());
                pool.init(PhysAddr(0x1000_0000), 256 * 1024 * 1024);
                bench.iter(|| {
                    let handle = pool.allocate(black_box(size)).unwrap();
                    pool.release(handle.0).unwrap();
                });
            },
        );
    }
    group.finish();
}

fn bench_tensor_alloc_burst(c: &mut Criterion) {
    c.bench_function("Memory/tensor_pool/burst_alloc", |bench| {
        let mut pool = Box::new(TensorPool::default());
        bench.iter(|| {
            pool.init(PhysAddr(0x1000_0000), 256 * 1024 * 1024);
            let mut handles = Vec::with_capacity(100);
            for _ in 0..100 {
                if let Ok(h) = pool.allocate(black_box(4096)) {
                    handles.push(h.0);
                }
            }
            for h in handles {
                let _ = pool.release(h);
            }
        });
    });
}

criterion_group!(benches, bench_tensor_alloc_free, bench_tensor_alloc_burst);
criterion_main!(benches);
