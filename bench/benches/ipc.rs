// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for IPC pub/sub message throughput.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallaios_ipc::pubsub::{Message, MessageQueue, PublisherId};
use std::hint::black_box;

fn bench_pubsub_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("IPC/pubsub");
    for payload_size in [64, 1024, 65536] {
        let payload = vec![0xAAu8; payload_size];
        group.bench_with_input(
            BenchmarkId::new("push_pop", payload_size),
            &payload_size,
            |bench, _| {
                bench.iter(|| {
                    let mut queue = MessageQueue::new();
                    let key = smallaios_ipc::key_expr::KeyExpr::new("bench/test").unwrap();
                    for _ in 0..100 {
                        let msg = Message {
                            key: key.clone(),
                            payload: payload.clone(),
                            timestamp: 0,
                            publisher: PublisherId(0),
                        };
                        let _ = queue.push(black_box(msg));
                    }
                    let mut count = 0;
                    while queue.pop().is_some() {
                        count += 1;
                    }
                    count
                });
            },
        );
    }
    group.finish();
}

fn bench_pubsub_push(c: &mut Criterion) {
    let mut group = c.benchmark_group("IPC/pubsub_push");
    let payload = vec![0xBBu8; 64];
    group.bench_function("single_push", |bench| {
        let mut queue = MessageQueue::new();
        let key = smallaios_ipc::key_expr::KeyExpr::new("bench/push").unwrap();
        bench.iter(|| {
            let msg = Message {
                key: key.clone(),
                payload: payload.clone(),
                timestamp: 0,
                publisher: PublisherId(0),
            };
            let _ = queue.push(black_box(msg));
        });
    });
    group.finish();
}

criterion_group!(benches, bench_pubsub_throughput, bench_pubsub_push);
criterion_main!(benches);
