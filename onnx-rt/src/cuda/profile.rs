// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Per-op timing + memcpy-byte instrumentation for the hybrid GPU
//! executor.
//!
//! Gated behind the `gpu-profile` Cargo feature. When the feature is
//! disabled, every function in this module compiles to a no-op via
//! `#[cfg]` — the production build pays zero runtime overhead, and
//! `CudaRuntime` doesn't grow new fields.
//!
//! When the feature is on, calls to [`record_op`] / [`record_memcpy`]
//! push events into a per-process ring buffer; on `CudaRuntime::drop`
//! we dump a small summary to stderr. Useful for verifying that the
//! hybrid executor isn't ping-ponging tensors and for spotting which
//! ops dominate inference time.

#![cfg(feature = "gpu-profile")]

extern crate alloc;
extern crate std;

use alloc::collections::{BTreeMap, VecDeque};
use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;
use core::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// Maximum number of events kept in the ring before oldest entries are
/// evicted. Long-running benchmarks can easily exceed a few thousand
/// events; the cap stops a process leaking memory if `dump_to_stderr`
/// is never called.
const RING_CAPACITY: usize = 16_384;

/// Counter of events dropped because the ring was full. Surfaced in the
/// dump so we can tell when the cap is biting.
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// One profile event: an op name + duration in microseconds + bytes
/// transferred (for memcpy events) or 0 (for compute events).
#[derive(Debug, Clone)]
pub struct Event {
    pub label: String,
    pub kind: EventKind,
    pub micros: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EventKind {
    Op,
    HostToDevice,
    DeviceToHost,
}

static ENABLED: AtomicBool = AtomicBool::new(true);

fn ring() -> &'static Mutex<VecDeque<Event>> {
    use std::sync::OnceLock;
    static RING: OnceLock<Mutex<VecDeque<Event>>> = OnceLock::new();
    RING.get_or_init(|| Mutex::new(VecDeque::with_capacity(RING_CAPACITY)))
}

/// Push an event into the bounded ring, evicting the oldest entry if
/// the cap is reached. The eviction count is reported on dump.
fn push_event(r: &mut VecDeque<Event>, e: Event) {
    if r.len() >= RING_CAPACITY {
        r.pop_front();
        DROPPED.fetch_add(1, Ordering::Relaxed);
    }
    r.push_back(e);
}

/// Record a compute event. `start` is the `Instant` captured before
/// the op; we compute duration here.
pub fn record_op(label: &str, start: Instant) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let micros = start.elapsed().as_micros() as u64;
    if let Ok(mut r) = ring().lock() {
        push_event(
            &mut r,
            Event {
                label: String::from(label),
                kind: EventKind::Op,
                micros,
                bytes: 0,
            },
        );
    }
}

/// Record a host↔device memcpy event.
pub fn record_memcpy(kind: EventKind, bytes: u64, start: Instant) {
    if !ENABLED.load(Ordering::Relaxed) {
        return;
    }
    let micros = start.elapsed().as_micros() as u64;
    if let Ok(mut r) = ring().lock() {
        push_event(
            &mut r,
            Event {
                label: String::new(),
                kind,
                micros,
                bytes,
            },
        );
    }
}

/// Disable further recording (useful in tests that don't want to
/// pollute the global ring).
pub fn disable() {
    ENABLED.store(false, Ordering::Relaxed);
}

/// Aggregate the ring buffer into per-op summaries and pretty-print
/// to stderr. Drains the ring afterwards so a subsequent dump only
/// shows new events.
pub fn dump_to_stderr() {
    let events: Vec<Event> = match ring().lock() {
        Ok(mut r) => r.drain(..).collect(),
        Err(_) => return,
    };
    if events.is_empty() {
        return;
    }
    let mut by_label: BTreeMap<String, (u64, u64, u64)> = BTreeMap::new(); // (total_us, count, total_bytes)
    let mut h2d_us = 0u64;
    let mut h2d_bytes = 0u64;
    let mut d2h_us = 0u64;
    let mut d2h_bytes = 0u64;
    for e in &events {
        match e.kind {
            EventKind::Op => {
                let entry = by_label.entry(e.label.clone()).or_insert((0, 0, 0));
                entry.0 += e.micros;
                entry.1 += 1;
            }
            EventKind::HostToDevice => {
                h2d_us += e.micros;
                h2d_bytes += e.bytes;
            }
            EventKind::DeviceToHost => {
                d2h_us += e.micros;
                d2h_bytes += e.bytes;
            }
        }
    }

    let dropped = DROPPED.swap(0, Ordering::Relaxed);
    eprintln!("=== gpu-profile (cumulative) ===");
    if dropped > 0 {
        eprintln!(
            "warning: ring buffer evicted {} oldest events (cap={})",
            dropped, RING_CAPACITY
        );
    }
    eprintln!("host->device: {} bytes in {} us", h2d_bytes, h2d_us);
    eprintln!("device->host: {} bytes in {} us", d2h_bytes, d2h_us);
    eprintln!("op timings (us, count):");
    let mut entries: Vec<(String, u64, u64)> =
        by_label.into_iter().map(|(k, v)| (k, v.0, v.1)).collect();
    entries.sort_by_key(|b| std::cmp::Reverse(b.1));
    for (label, total_us, count) in entries.iter().take(20) {
        eprintln!(
            "  {:<28} total={:>10} us  count={:>5}  avg={} us",
            label,
            total_us,
            count,
            if *count > 0 { total_us / count } else { 0 }
        );
    }
    let _ = format!("{} events flushed", events.len());
}
