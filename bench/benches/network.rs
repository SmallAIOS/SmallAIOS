// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Criterion benchmarks for network packet parsing.

use criterion::{criterion_group, criterion_main, Criterion};
use smallaios_net::tcp::TcpHeader;
use smallaios_net::udp::UdpHeader;
use std::hint::black_box;

fn make_tcp_packet() -> Vec<u8> {
    let mut pkt = vec![0u8; 40];
    // Source port: 8080
    pkt[0..2].copy_from_slice(&8080u16.to_be_bytes());
    // Dest port: 443
    pkt[2..4].copy_from_slice(&443u16.to_be_bytes());
    // Seq number
    pkt[4..8].copy_from_slice(&1u32.to_be_bytes());
    // Ack number
    pkt[8..12].copy_from_slice(&0u32.to_be_bytes());
    // Data offset (5 words = 20 bytes) + flags (SYN)
    pkt[12] = 0x50; // 5 << 4
    pkt[13] = 0x02; // SYN
                    // Window size
    pkt[14..16].copy_from_slice(&65535u16.to_be_bytes());
    pkt
}

fn make_udp_packet() -> Vec<u8> {
    let mut pkt = vec![0u8; 16];
    // Source port
    pkt[0..2].copy_from_slice(&5353u16.to_be_bytes());
    // Dest port
    pkt[2..4].copy_from_slice(&5353u16.to_be_bytes());
    // Length (header + 8 bytes payload)
    pkt[4..6].copy_from_slice(&16u16.to_be_bytes());
    // Checksum
    pkt[6..8].copy_from_slice(&0u16.to_be_bytes());
    pkt
}

fn bench_tcp_parse(c: &mut Criterion) {
    let pkt = make_tcp_packet();
    c.bench_function("TCP/parse_header", |b| {
        b.iter(|| TcpHeader::parse(black_box(&pkt)));
    });
}

fn bench_udp_parse(c: &mut Criterion) {
    let pkt = make_udp_packet();
    c.bench_function("UDP/parse_header", |b| {
        b.iter(|| UdpHeader::parse(black_box(&pkt)));
    });
}

criterion_group!(benches, bench_tcp_parse, bench_udp_parse);
criterion_main!(benches);
