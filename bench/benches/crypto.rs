// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for cryptographic primitives.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use smallaios_security::crypto::aes_gcm::{
    Aes256Gcm, AesKey, GcmNonce, AES256_KEY_LEN, GCM_NONCE_LEN,
};
use smallaios_security::crypto::sha3;
use std::hint::black_box;

fn bench_sha3_256(c: &mut Criterion) {
    let mut group = c.benchmark_group("SHA3-256");
    for size in [64, 1024, 65536] {
        let data = vec![0xABu8; size];
        group.bench_with_input(BenchmarkId::new("bytes", size), &size, |bench, _| {
            bench.iter(|| sha3::sha3_256(black_box(&data)));
        });
    }
    group.finish();
}

fn bench_aes_gcm_encrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("AES-256-GCM/encrypt");
    let key = AesKey::from_bytes([0x42u8; AES256_KEY_LEN]);
    let cipher = Aes256Gcm::new(key);
    let nonce = GcmNonce::from_bytes([0u8; GCM_NONCE_LEN]);
    let aad = b"bench";

    for size in [64, 1024, 65536] {
        let plaintext = vec![0u8; size];
        let mut ciphertext = vec![0u8; size];
        group.bench_with_input(BenchmarkId::new("bytes", size), &size, |bench, _| {
            bench.iter(|| {
                let _ = cipher.encrypt(
                    black_box(&nonce),
                    black_box(aad),
                    black_box(&plaintext),
                    &mut ciphertext,
                );
            });
        });
    }
    group.finish();
}

fn bench_aes_gcm_decrypt(c: &mut Criterion) {
    let mut group = c.benchmark_group("AES-256-GCM/decrypt");
    let key = AesKey::from_bytes([0x42u8; AES256_KEY_LEN]);
    let cipher = Aes256Gcm::new(key);
    let nonce = GcmNonce::from_bytes([0u8; GCM_NONCE_LEN]);
    let aad = b"bench";

    for size in [64, 1024] {
        let plaintext = vec![0u8; size];
        let mut ciphertext = vec![0u8; size];
        let tag = cipher
            .encrypt(&nonce, aad, &plaintext, &mut ciphertext)
            .unwrap();
        let mut decrypted = vec![0u8; size];
        group.bench_with_input(BenchmarkId::new("bytes", size), &size, |bench, _| {
            bench.iter(|| {
                let _ = cipher.decrypt(
                    black_box(&nonce),
                    black_box(aad),
                    black_box(&ciphertext),
                    black_box(&tag),
                    &mut decrypted,
                );
            });
        });
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sha3_256,
    bench_aes_gcm_encrypt,
    bench_aes_gcm_decrypt
);
criterion_main!(benches);
