// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Generic conformance suite for any [`BlockDevice`] implementation.
//!
//! Per `embedded-filesystem-v1::fs-block-device` spec, every per-arch
//! driver SHALL pass this suite. The suite runs against:
//!
//! - The in-memory `MockBlockDevice` (always — exercised by every CI
//!   run of `cargo test -p smallaios-fs --features fs-on-disk-mounts`).
//! - The virtio-blk driver (added under `fs-block-virtio` once the
//!   QEMU-side test harness lands; see Phase 1 follow-on tickets).
//!
//! The suite uses a minimal Rust-host runner: each test is a top-level
//! `#[test]` fn so `cargo test` picks them up. The file lives at
//! `fs/tests/block_conformance.rs` so it is compiled as an integration
//! test (separate binary).

#![cfg(feature = "fs-on-disk-mounts")]

extern crate alloc;

use core::cell::RefCell;

use smallaios_fs::block::{
    block_error_to_errno, block_error_to_negative_errno, check_buf_alignment, check_lba_in_range,
    mock::MockBlockDevice,
    retry::{
        run_with_retries, AttemptOutcome, PolicyError, PolicyKind, RetryPolicy,
        DEFAULT_READ_TIMEOUT_MS, DEFAULT_RETRY_BACKOFF_MS, DEFAULT_RETRY_COUNT,
        DEFAULT_WRITE_TIMEOUT_MS, MAX_ALLOWED_RETRY_COUNT, RETRY_COUNT_INFINITE_SENTINEL,
    },
    sector::{LegacyPathWarner, SectorAdapter, SectorMode},
    BlockDevice, BlockError, FS_BLOCK_SIZE_BYTES,
};

// ─── Trait-level scenarios from spec ───────────────────────────────────────

/// Spec scenario: "Read returns exactly the requested block".
#[test]
fn read_returns_exactly_the_requested_block() {
    let dev = MockBlockDevice::with_pattern(4096, 16);
    let expected = dev.snapshot(7, 4096);

    let mut buf = alloc::vec![0u8; 4096];
    dev.read_block(7, &mut buf).unwrap();

    assert_eq!(buf, expected);
    assert_eq!(buf.len(), 4096);
}

/// Spec scenario: "Read with mis-sized buffer rejected".
#[test]
fn read_with_mis_sized_buffer_rejected_unaligned() {
    let dev = MockBlockDevice::new(4096, 16);

    // Length not a multiple of 4096 → Unaligned.
    let mut bad = alloc::vec![0u8; 4095];
    assert_eq!(dev.read_block(0, &mut bad), Err(BlockError::Unaligned));

    let mut bad = alloc::vec![0u8; 4097];
    assert_eq!(dev.read_block(0, &mut bad), Err(BlockError::Unaligned));

    // Zero-length is also unaligned (no I/O issued).
    let mut empty: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    assert_eq!(dev.read_block(0, &mut empty), Err(BlockError::Unaligned));

    // Reads do NOT increment the success counter on rejection.
    assert_eq!(dev.read_count(), 0);
}

/// Spec scenario: "Write with mis-sized buffer rejected".
#[test]
fn write_with_mis_sized_buffer_rejected_unaligned() {
    let mut dev = MockBlockDevice::new(4096, 16);
    let bad = alloc::vec![0u8; 4095];
    assert_eq!(dev.write_block(0, &bad), Err(BlockError::Unaligned));

    let bad = alloc::vec![0u8; 4097];
    assert_eq!(dev.write_block(0, &bad), Err(BlockError::Unaligned));

    let bad: alloc::vec::Vec<u8> = alloc::vec::Vec::new();
    assert_eq!(dev.write_block(0, &bad), Err(BlockError::Unaligned));
    assert_eq!(dev.write_count(), 0);
}

/// Spec scenario: "Read past end of device rejected".
#[test]
fn read_past_end_of_device_rejected_outofrange() {
    let dev = MockBlockDevice::new(4096, 4);
    let mut buf = alloc::vec![0u8; 4096];

    // lba == block_count
    assert_eq!(dev.read_block(4, &mut buf), Err(BlockError::OutOfRange));
    // lba > block_count
    assert_eq!(dev.read_block(5, &mut buf), Err(BlockError::OutOfRange));
    assert_eq!(
        dev.read_block(u64::MAX, &mut buf),
        Err(BlockError::OutOfRange)
    );
}

/// Spec scenario: "Read extending past end rejected".
#[test]
fn multi_block_read_extending_past_end_rejected() {
    let dev = MockBlockDevice::new(4096, 4);
    let mut buf = alloc::vec![0u8; 8192]; // 2 blocks
                                          // lba 3 + 2 blocks = covers 3 and 4, but only 0..3 exist → OutOfRange
    assert_eq!(dev.read_block(3, &mut buf), Err(BlockError::OutOfRange));
}

/// Spec scenario: "Write past end of device rejected".
#[test]
fn write_past_end_of_device_rejected_outofrange() {
    let mut dev = MockBlockDevice::new(4096, 4);
    let buf = alloc::vec![0u8; 4096];
    assert_eq!(dev.write_block(4, &buf), Err(BlockError::OutOfRange));
}

/// Multi-block read: 2 contiguous 4 KiB blocks come back as one slab.
#[test]
fn multi_block_read_concatenates_storage() {
    let dev = MockBlockDevice::with_pattern(4096, 4);
    let mut buf = alloc::vec![0u8; 8192];
    dev.read_block(1, &mut buf).unwrap();

    let expected_first = dev.snapshot(1, 4096);
    let expected_second = dev.snapshot(2, 4096);
    assert_eq!(&buf[..4096], &expected_first[..]);
    assert_eq!(&buf[4096..], &expected_second[..]);
}

/// Round-trip: write then read returns the same bytes.
#[test]
fn write_then_read_roundtrip() {
    let mut dev = MockBlockDevice::new(4096, 4);
    let pattern: alloc::vec::Vec<u8> = (0u32..4096).map(|i| (i % 251) as u8).collect();
    dev.write_block(2, &pattern).unwrap();

    let mut readback = alloc::vec![0u8; 4096];
    dev.read_block(2, &mut readback).unwrap();
    assert_eq!(readback, pattern);
}

// ─── Errno mapping spec scenarios ──────────────────────────────────────────

/// Spec scenario: "Timeout maps to ETIMEDOUT at syscall boundary".
#[test]
fn timeout_maps_to_etimedout_at_syscall_boundary() {
    assert_eq!(block_error_to_errno(BlockError::Timeout), 110);
    assert_eq!(block_error_to_negative_errno(BlockError::Timeout), -110);
}

/// Spec scenario: "Bad CRC maps to EIO".
#[test]
fn bad_crc_maps_to_eio() {
    assert_eq!(block_error_to_errno(BlockError::BadCrc), 5);
    assert_eq!(block_error_to_negative_errno(BlockError::BadCrc), -5);
}

/// Bad CRC observed end-to-end: drive injects, errno conversion fires.
#[test]
fn bad_crc_block_observable_end_to_end() {
    let dev = MockBlockDevice::new(4096, 4);
    dev.set_bad_crc_block(2);

    let mut buf = alloc::vec![0u8; 4096];
    let err = dev.read_block(2, &mut buf).unwrap_err();
    assert_eq!(err, BlockError::BadCrc);
    assert_eq!(block_error_to_negative_errno(err), -5); // -EIO
}

/// MediaError surfaces as -EIO (not -EINVAL or anything else).
#[test]
fn media_error_maps_to_eio_at_syscall_boundary() {
    let dev = MockBlockDevice::new(4096, 4);
    dev.set_permanent_failure(BlockError::MediaError);

    let mut buf = alloc::vec![0u8; 4096];
    let err = dev.read_block(0, &mut buf).unwrap_err();
    assert_eq!(err, BlockError::MediaError);
    assert_eq!(block_error_to_negative_errno(err), -5);
}

/// NotPresent surfaces as -ENXIO.
#[test]
fn not_present_maps_to_enxio_at_syscall_boundary() {
    let dev = MockBlockDevice::new(4096, 4);
    dev.set_permanent_failure(BlockError::NotPresent);

    let mut buf = alloc::vec![0u8; 4096];
    let err = dev.read_block(0, &mut buf).unwrap_err();
    assert_eq!(err, BlockError::NotPresent);
    assert_eq!(block_error_to_negative_errno(err), -6);
}

/// DeviceBusy surfaces as -EBUSY.
#[test]
fn device_busy_maps_to_ebusy_at_syscall_boundary() {
    let dev = MockBlockDevice::new(4096, 4);
    dev.set_permanent_failure(BlockError::DeviceBusy);

    let mut buf = alloc::vec![0u8; 4096];
    let err = dev.read_block(0, &mut buf).unwrap_err();
    assert_eq!(err, BlockError::DeviceBusy);
    assert_eq!(block_error_to_negative_errno(err), -16);
}

/// Unaligned surfaces as -EINVAL.
#[test]
fn unaligned_maps_to_einval_at_syscall_boundary() {
    let dev = MockBlockDevice::new(4096, 4);
    let mut buf = alloc::vec![0u8; 4097];
    let err = dev.read_block(0, &mut buf).unwrap_err();
    assert_eq!(err, BlockError::Unaligned);
    assert_eq!(block_error_to_negative_errno(err), -22);
}

/// OutOfRange surfaces as -EINVAL.
#[test]
fn out_of_range_maps_to_einval_at_syscall_boundary() {
    let dev = MockBlockDevice::new(4096, 4);
    let mut buf = alloc::vec![0u8; 4096];
    let err = dev.read_block(4, &mut buf).unwrap_err();
    assert_eq!(err, BlockError::OutOfRange);
    assert_eq!(block_error_to_negative_errno(err), -22);
}

// ─── Retry policy spec scenarios ───────────────────────────────────────────

/// Spec scenario: "Three retries then fail".
#[test]
fn three_retries_then_fail_with_timeout() {
    let policy = RetryPolicy::default_read();
    let calls = core::cell::Cell::new(0u32);
    let sleeps: RefCell<alloc::vec::Vec<u32>> = RefCell::new(alloc::vec::Vec::new());

    let result = run_with_retries(
        &policy,
        |_| {
            calls.set(calls.get() + 1);
            AttemptOutcome::Retry
        },
        |ms| sleeps.borrow_mut().push(ms),
    );

    // 1 initial attempt + 3 retries = 4 total attempts.
    assert_eq!(calls.get(), 4);
    // The 4th failure surfaces as Timeout per the spec.
    assert_eq!(result, Err(BlockError::Timeout));
    // Backoff schedule: 500 / 1000 / 2000.
    assert_eq!(*sleeps.borrow(), alloc::vec![500u32, 1000, 2000]);
}

/// Spec scenario: "Indefinite retry rejected at config write" — the
/// validator returns `-EINVAL`.
#[test]
fn indefinite_retry_rejected_at_config_write() {
    let policy = RetryPolicy {
        timeout_ms: 250,
        retry_count: RETRY_COUNT_INFINITE_SENTINEL,
        backoff_ms: alloc::vec![500],
    };
    assert_eq!(policy.validate(), Err(PolicyError::Indefinite));

    // PolicyError → -EINVAL at the syscall boundary.
    let err_code: i32 = match policy.validate() {
        Err(PolicyError::Indefinite) => -22,
        Err(_) => -22,
        Ok(_) => 0,
    };
    assert_eq!(err_code, -22);
}

/// Even values higher than the explicit sentinel but above the cap
/// are rejected (defense in depth).
#[test]
fn retry_count_above_cap_rejected() {
    let policy = RetryPolicy {
        timeout_ms: 250,
        retry_count: MAX_ALLOWED_RETRY_COUNT + 1,
        backoff_ms: alloc::vec![500],
    };
    assert_eq!(policy.validate(), Err(PolicyError::RetryCountTooHigh));
}

/// Default read policy matches spec values.
#[test]
fn default_read_policy_matches_spec() {
    let p = RetryPolicy::default_read();
    assert_eq!(p.timeout_ms, DEFAULT_READ_TIMEOUT_MS);
    assert_eq!(p.timeout_ms, 250);
    assert_eq!(p.retry_count, DEFAULT_RETRY_COUNT);
    assert_eq!(p.retry_count, 3);
    assert_eq!(p.backoff_ms, DEFAULT_RETRY_BACKOFF_MS.to_vec());
}

/// Default write policy matches spec values.
#[test]
fn default_write_policy_matches_spec() {
    let p = RetryPolicy::default_write();
    assert_eq!(p.timeout_ms, DEFAULT_WRITE_TIMEOUT_MS);
    assert_eq!(p.timeout_ms, 1000);
    assert_eq!(p.retry_count, 3);
    assert_eq!(p.backoff_ms, [500u32, 1000, 2000].to_vec());
}

/// Retry helper: success on retry attempt 2 returns Ok with no further
/// attempts.
#[test]
fn retry_succeeds_on_second_attempt() {
    let policy = RetryPolicy::default_read();
    let calls = core::cell::Cell::new(0u32);
    let sleeps: RefCell<alloc::vec::Vec<u32>> = RefCell::new(alloc::vec::Vec::new());

    let result = run_with_retries(
        &policy,
        |_| {
            let n = calls.get() + 1;
            calls.set(n);
            if n < 2 {
                AttemptOutcome::Retry
            } else {
                AttemptOutcome::Success
            }
        },
        |ms| sleeps.borrow_mut().push(ms),
    );
    assert!(result.is_ok());
    assert_eq!(calls.get(), 2);
    // One backoff before retry #1.
    assert_eq!(*sleeps.borrow(), alloc::vec![500u32]);
}

/// Retry helper: permanent error short-circuits.
#[test]
fn retry_permanent_error_short_circuits() {
    let policy = RetryPolicy::default_read();
    let calls = core::cell::Cell::new(0u32);
    let result = run_with_retries(
        &policy,
        |_| {
            calls.set(calls.get() + 1);
            AttemptOutcome::Permanent(BlockError::OutOfRange)
        },
        |_| panic!("must not sleep on permanent error"),
    );
    assert_eq!(calls.get(), 1);
    assert_eq!(result, Err(BlockError::OutOfRange));
}

/// `from_config` is wired and returns spec defaults today.
#[test]
fn retry_policy_from_config_returns_defaults() {
    assert_eq!(
        RetryPolicy::from_config(PolicyKind::Read),
        RetryPolicy::default_read()
    );
    assert_eq!(
        RetryPolicy::from_config(PolicyKind::Write),
        RetryPolicy::default_write()
    );
}

/// End-to-end: device rejects the first 3 reads with DeviceBusy then
/// succeeds. Combined with the default retry policy, the wrapper
/// returns Ok.
#[test]
fn retry_wrapper_recovers_from_transient_failures() {
    let dev = MockBlockDevice::with_pattern(4096, 4);
    dev.set_transient_failures(3);

    let policy = RetryPolicy::default_read();
    let mut buf = alloc::vec![0u8; 4096];

    let result = run_with_retries(
        &policy,
        |_| match dev.read_block(0, &mut buf) {
            Ok(()) => AttemptOutcome::Success,
            Err(BlockError::DeviceBusy) | Err(BlockError::Timeout) => AttemptOutcome::Retry,
            Err(e) => AttemptOutcome::Permanent(e),
        },
        |_| {},
    );
    assert!(result.is_ok());
    assert_eq!(buf, dev.snapshot(0, 4096));
}

/// End-to-end: device fails the first 4 reads (initial + 3 retries
/// exhaust). Wrapper returns Timeout.
#[test]
fn retry_wrapper_returns_timeout_after_exhausting_retries() {
    let dev = MockBlockDevice::new(4096, 4);
    dev.set_transient_failures(10);

    let policy = RetryPolicy::default_read();
    let mut buf = alloc::vec![0u8; 4096];

    let result = run_with_retries(
        &policy,
        |_| match dev.read_block(0, &mut buf) {
            Ok(()) => AttemptOutcome::Success,
            Err(BlockError::DeviceBusy) | Err(BlockError::Timeout) => AttemptOutcome::Retry,
            Err(e) => AttemptOutcome::Permanent(e),
        },
        |_| {},
    );
    assert_eq!(result, Err(BlockError::Timeout));
    assert_eq!(block_error_to_negative_errno(BlockError::Timeout), -110);
}

// ─── Sector-size spec scenarios ────────────────────────────────────────────

/// Spec scenario: "4 KiB-native device used directly".
#[test]
fn fourk_native_device_used_directly() {
    let dev = MockBlockDevice::with_pattern(4096, 16);
    let adapter = SectorAdapter::new(dev, 4096).unwrap();

    assert_eq!(adapter.mode(), SectorMode::Native4K);
    assert_eq!(adapter.block_size_bytes(), FS_BLOCK_SIZE_BYTES);
    assert_eq!(adapter.block_count(), 16);

    let expected = adapter.inner().snapshot(7, 4096);
    let mut buf = alloc::vec![0u8; 4096];
    adapter.read_block(7, &mut buf).unwrap();
    assert_eq!(buf, expected);
}

/// Spec scenario: "512/4 KiB Advanced Format device aligned".
#[test]
fn advanced_format_512e_device_aligned_to_4k() {
    // 128 * 512 = 64 KiB → exposes 16 FS blocks.
    let dev = MockBlockDevice::with_pattern(512, 128);
    let adapter = SectorAdapter::new(dev, 4096).unwrap();

    assert_eq!(adapter.mode(), SectorMode::AdvancedFormat512e);
    assert_eq!(adapter.block_size_bytes(), FS_BLOCK_SIZE_BYTES);
    assert_eq!(adapter.block_count(), 16);

    // FS block 0 must read inner LBAs 0..8; FS block 5 reads 40..48.
    let mut fs_block = alloc::vec![0u8; 4096];
    adapter.read_block(5, &mut fs_block).unwrap();

    let inner_chunk = adapter.inner().snapshot(40, 4096);
    assert_eq!(fs_block, inner_chunk);
}

/// Spec scenario: "512-byte legacy device emulated".
#[test]
fn legacy_512_device_emulated_with_warning() {
    use core::cell::Cell;

    struct CountingWarner(Cell<u32>);
    impl LegacyPathWarner for CountingWarner {
        fn warn_legacy_slow_path(&self) {
            self.0.set(self.0.get() + 1);
        }
    }

    let dev = MockBlockDevice::with_pattern(512, 128);
    let warner = CountingWarner(Cell::new(0));
    let adapter = smallaios_fs::block::sector::new_with_warning(dev, 512, &warner).unwrap();

    assert_eq!(adapter.mode(), SectorMode::Legacy512);
    assert!(adapter.mode().is_legacy_slow_path());
    // The warning fired exactly once at construction.
    assert_eq!(warner.0.get(), 1);

    // FS-level reads still bundle 8 logical sectors per FS block.
    let mut buf = alloc::vec![0u8; 4096];
    adapter.read_block(0, &mut buf).unwrap();
    assert_eq!(buf.len(), 4096);
}

/// FS reads SHALL never split across two physical sectors on
/// Advanced Format devices.
#[test]
fn fs_block_never_splits_physical_sector_on_advanced_format() {
    // 16 FS blocks via 128 * 512 logical sectors.
    let dev = MockBlockDevice::with_pattern(512, 128);
    let adapter = SectorAdapter::new(dev, 4096).unwrap();

    // Probe every FS block — verify each FS block's contents come from
    // an aligned 8-LBA window (lba * 8 .. lba * 8 + 8) of the inner
    // device, NEVER from a misaligned offset.
    for fs_lba in 0..adapter.block_count() {
        let mut buf = alloc::vec![0u8; 4096];
        adapter.read_block(fs_lba, &mut buf).unwrap();

        let aligned_start = (fs_lba * 8) as usize * 512;
        let inner_storage = adapter
            .inner()
            .snapshot(0, adapter.inner().block_count() as usize * 512);
        assert_eq!(
            buf,
            &inner_storage[aligned_start..aligned_start + 4096],
            "FS block {fs_lba} did not align to inner LBA {}",
            fs_lba * 8
        );
    }
}

/// Unsupported sector-size combinations are rejected.
#[test]
fn unsupported_sector_combinations_rejected() {
    // 2 KiB logical / 4 KiB physical — not supported.
    let dev = MockBlockDevice::new(2048, 32);
    assert!(SectorAdapter::new(dev, 4096).is_none());

    // 4 KiB logical / 512 B physical — nonsensical.
    let dev = MockBlockDevice::new(4096, 16);
    assert!(SectorAdapter::new(dev, 512).is_none());

    // 8 KiB logical — none of the three supported modes.
    let dev = MockBlockDevice::new(8192, 8);
    assert!(SectorAdapter::new(dev, 8192).is_none());
}

/// Adapter rejects unaligned FS-level buffers.
#[test]
fn adapter_rejects_misaligned_fs_buffer() {
    let dev = MockBlockDevice::new(512, 128);
    let adapter = SectorAdapter::new(dev, 4096).unwrap();
    let mut buf = alloc::vec![0u8; 4097];
    assert_eq!(adapter.read_block(0, &mut buf), Err(BlockError::Unaligned));
}

/// Adapter rejects FS-level out-of-range LBAs.
#[test]
fn adapter_rejects_out_of_range_fs_lba() {
    let dev = MockBlockDevice::new(512, 128);
    let adapter = SectorAdapter::new(dev, 4096).unwrap();
    let mut buf = alloc::vec![0u8; 4096];
    assert_eq!(
        adapter.read_block(16, &mut buf),
        Err(BlockError::OutOfRange)
    );
}

/// Round-trip via the legacy 512/512 emulation slow path: writes also
/// bundle 8 logical sectors per FS block.
#[test]
fn legacy_512_roundtrip_through_adapter() {
    let dev = MockBlockDevice::new(512, 128);
    let mut adapter = SectorAdapter::new(dev, 512).unwrap();
    assert_eq!(adapter.mode(), SectorMode::Legacy512);

    let pattern: alloc::vec::Vec<u8> = (0u32..4096)
        .map(|i| (i.wrapping_mul(7) % 251) as u8)
        .collect();
    adapter.write_block(11, &pattern).unwrap();
    let mut readback = alloc::vec![0u8; 4096];
    adapter.read_block(11, &mut readback).unwrap();
    assert_eq!(readback, pattern);
}

// ─── Helpers for module-level invariants ────────────────────────────────────

/// `check_buf_alignment` is the canonical helper used by every per-arch
/// driver. Verify its contract.
#[test]
fn buf_alignment_helper_contract() {
    assert!(check_buf_alignment(4096, 4096).is_ok());
    assert!(check_buf_alignment(8192, 4096).is_ok());
    assert!(check_buf_alignment(512, 512).is_ok());

    assert_eq!(check_buf_alignment(0, 4096), Err(BlockError::Unaligned));
    assert_eq!(check_buf_alignment(1, 4096), Err(BlockError::Unaligned));
    assert_eq!(check_buf_alignment(4097, 4096), Err(BlockError::Unaligned));
}

/// `check_lba_in_range` is the canonical bounds helper.
#[test]
fn lba_in_range_helper_contract() {
    assert!(check_lba_in_range(0, 4096, 4096, 4).is_ok());
    assert!(check_lba_in_range(3, 4096, 4096, 4).is_ok());
    assert_eq!(
        check_lba_in_range(4, 4096, 4096, 4),
        Err(BlockError::OutOfRange)
    );
    assert_eq!(
        check_lba_in_range(0, 4096, 4096, 0),
        Err(BlockError::OutOfRange)
    );
}

/// Flush is reachable through the trait and propagates errors.
#[test]
fn flush_propagates_permanent_failure() {
    let mut dev = MockBlockDevice::new(4096, 4);
    dev.set_permanent_failure(BlockError::MediaError);
    assert_eq!(dev.flush(), Err(BlockError::MediaError));
    dev.clear_permanent_failure();
    assert!(dev.flush().is_ok());
}
