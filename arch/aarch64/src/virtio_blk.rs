// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! virtio-blk-modern (1.0+) MMIO driver implementing
//! [`smallaios_fs::block::BlockDevice`].
//!
//! This is the QEMU CI surface for the on-disk filesystem stack. The
//! driver targets the standardized virtio-MMIO transport — QEMU's
//! `-device virtio-blk-device,drive=...` exposes its registers via a
//! 0x200-byte MMIO window that this module pokes directly.
//!
//! The implementation is intentionally minimal:
//!
//! - Single virtqueue (#0) — virtio-blk has one request queue.
//! - Submit-and-wait per request (no interrupt or async; this is the
//!   smoke-test path).
//! - Three request types: `VIRTIO_BLK_T_IN`, `VIRTIO_BLK_T_OUT`,
//!   `VIRTIO_BLK_T_FLUSH`.
//! - Modern device path only (no legacy device support).
//!
//! Owned by `embedded-filesystem-v1` Phase 1.
//!
//! ## Safety
//!
//! The driver assumes the caller has identified a virtio-MMIO window
//! at `mmio_base` corresponding to a virtio-blk device (DeviceID = 2,
//! Magic = "virt"). The caller passes a physically-contiguous,
//! identity-mapped DMA region for the descriptor/avail/used rings.
//! The driver MUST NOT be used after the underlying device is
//! reset/hot-removed.

#![cfg(feature = "fs-block-virtio")]
// Some MMIO register offsets and feature bits are documented for
// completeness so the module reads as a reference for the
// `embedded-filesystem-v1` follow-on phases (NVMe, AHCI, SDHCI). The
// current submit-and-wait path doesn't use all of them yet.
#![allow(dead_code)]

use core::cell::UnsafeCell;
use core::ptr::{read_volatile, write_volatile};
use core::sync::atomic::{compiler_fence, fence, Ordering};

use smallaios_fs::block::{check_buf_alignment, check_lba_in_range, BlockDevice, BlockError};

// ─── virtio-MMIO register offsets (modern, virtio v1.0+) ────────────────────
// Reference: virtio v1.2 spec § 4.2.2 "MMIO Device Register Layout".

/// Magic value 0x74726976 ("virt" little-endian).
const REG_MAGIC: usize = 0x000;
/// Device version (must be 2 for virtio v1.0+).
const REG_VERSION: usize = 0x004;
/// Subsystem device ID (2 = block).
const REG_DEVICE_ID: usize = 0x008;
const REG_VENDOR_ID: usize = 0x00C;
const REG_DEVICE_FEATURES: usize = 0x010;
const REG_DEVICE_FEATURES_SEL: usize = 0x014;
const REG_DRIVER_FEATURES: usize = 0x020;
const REG_DRIVER_FEATURES_SEL: usize = 0x024;
const REG_QUEUE_SEL: usize = 0x030;
const REG_QUEUE_NUM_MAX: usize = 0x034;
const REG_QUEUE_NUM: usize = 0x038;
const REG_QUEUE_READY: usize = 0x044;
const REG_QUEUE_NOTIFY: usize = 0x050;
const REG_INTERRUPT_STATUS: usize = 0x060;
const REG_INTERRUPT_ACK: usize = 0x064;
const REG_STATUS: usize = 0x070;
const REG_QUEUE_DESC_LOW: usize = 0x080;
const REG_QUEUE_DESC_HIGH: usize = 0x084;
const REG_QUEUE_AVAIL_LOW: usize = 0x090;
const REG_QUEUE_AVAIL_HIGH: usize = 0x094;
const REG_QUEUE_USED_LOW: usize = 0x0A0;
const REG_QUEUE_USED_HIGH: usize = 0x0A4;
const REG_CONFIG: usize = 0x100;

const VIRTIO_MMIO_MAGIC: u32 = 0x7472_6976;
const VIRTIO_DEVICE_ID_BLOCK: u32 = 2;

// virtio-blk feature bits.
const VIRTIO_BLK_F_FLUSH: u32 = 9;
const VIRTIO_F_VERSION_1: u32 = 32;

// Status register bits.
const STATUS_ACKNOWLEDGE: u32 = 1;
const STATUS_DRIVER: u32 = 2;
const STATUS_DRIVER_OK: u32 = 4;
const STATUS_FEATURES_OK: u32 = 8;
const STATUS_FAILED: u32 = 0x80;

// Descriptor flags.
const VRING_DESC_F_NEXT: u16 = 0x1;
const VRING_DESC_F_WRITE: u16 = 0x2;

// virtio-blk request types.
const VIRTIO_BLK_T_IN: u32 = 0;
const VIRTIO_BLK_T_OUT: u32 = 1;
const VIRTIO_BLK_T_FLUSH: u32 = 4;

// virtio-blk request status codes.
const VIRTIO_BLK_S_OK: u8 = 0;
const VIRTIO_BLK_S_IOERR: u8 = 1;
const VIRTIO_BLK_S_UNSUPP: u8 = 2;

// Default queue size — keep small; this is the smoke-test path.
const QUEUE_SIZE: u16 = 16;

// virtio-blk hard-coded sector size (per the spec § 5.2).
const VIRTIO_BLK_SECTOR_SIZE: u32 = 512;

// ─── On-the-wire ring structures ────────────────────────────────────────────

/// Descriptor in the descriptor table.
#[repr(C, align(16))]
#[derive(Clone, Copy, Default)]
struct VringDesc {
    addr: u64,
    len: u32,
    flags: u16,
    next: u16,
}

/// Available ring header + entries (split-virtqueue).
#[repr(C, align(2))]
struct VringAvail {
    flags: u16,
    idx: u16,
    ring: [u16; QUEUE_SIZE as usize],
    used_event: u16,
}

/// Used ring entry.
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VringUsedElem {
    id: u32,
    len: u32,
}

/// Used ring (split-virtqueue).
#[repr(C, align(4))]
struct VringUsed {
    flags: u16,
    idx: u16,
    ring: [VringUsedElem; QUEUE_SIZE as usize],
    avail_event: u16,
}

/// virtio-blk request header (sent on the device-readable descriptor).
#[repr(C)]
#[derive(Clone, Copy, Default)]
struct VirtioBlkReqHeader {
    ty: u32,
    reserved: u32,
    sector: u64,
}

/// virtio-blk readable config space header (subset).
#[repr(C)]
struct VirtioBlkConfig {
    capacity_low: u32,  // sectors, low 32 bits
    capacity_high: u32, // sectors, high 32 bits
}

// ─── Driver state ───────────────────────────────────────────────────────────

/// Storage for one virtqueue. Allocated in identity-mapped, contiguous,
/// aligned DMA memory by the caller via [`VirtioBlk::new`].
#[repr(C, align(4096))]
pub struct QueueStorage {
    descs: [VringDesc; QUEUE_SIZE as usize],
    avail: VringAvail,
    used: VringUsed,
    /// Per-request scratch: 3 descriptors per request (header / data /
    /// status), one slot per descriptor index.
    headers: [VirtioBlkReqHeader; QUEUE_SIZE as usize],
    statuses: [u8; QUEUE_SIZE as usize],
}

impl QueueStorage {
    /// Construct an empty queue storage block. MUST live in
    /// identity-mapped DMA memory.
    pub const fn new() -> Self {
        Self {
            descs: [VringDesc {
                addr: 0,
                len: 0,
                flags: 0,
                next: 0,
            }; QUEUE_SIZE as usize],
            avail: VringAvail {
                flags: 0,
                idx: 0,
                ring: [0; QUEUE_SIZE as usize],
                used_event: 0,
            },
            used: VringUsed {
                flags: 0,
                idx: 0,
                ring: [VringUsedElem { id: 0, len: 0 }; QUEUE_SIZE as usize],
                avail_event: 0,
            },
            headers: [VirtioBlkReqHeader {
                ty: 0,
                reserved: 0,
                sector: 0,
            }; QUEUE_SIZE as usize],
            statuses: [0; QUEUE_SIZE as usize],
        }
    }
}

impl Default for QueueStorage {
    fn default() -> Self {
        Self::new()
    }
}

/// Interior-mutable state of the virtio-blk driver.
///
/// `BlockDevice::read_block` takes `&self` per the trait contract, but
/// the virtio request submission necessarily mutates the rings and
/// the `last_used_idx` watermark. We wrap that mutable state in an
/// `UnsafeCell` and rely on the caller to externally serialize access
/// (the driver is `!Sync` — see [`VirtioBlk`]). Concretely: every
/// SmallAIOS mount path owns its `VirtioBlk` from a single executor
/// task / IRQ context.
struct VirtioBlkInner {
    queue: &'static mut QueueStorage,
    /// Last-seen `used.idx` value, used to detect new completions.
    last_used_idx: u16,
}

/// virtio-blk-modern MMIO driver.
///
/// `!Sync`: callers must serialize access externally (e.g. one device
/// per mount, accessed from the mount's owning executor task).
pub struct VirtioBlk {
    mmio_base: usize,
    inner: UnsafeCell<VirtioBlkInner>,
    /// Capacity in 512-byte sectors as reported by the device.
    capacity_sectors: u64,
    /// Whether the device supports `VIRTIO_BLK_T_FLUSH`.
    has_flush: bool,
}

// `VirtioBlk` is intentionally `!Sync`. The `UnsafeCell` already makes
// it `!Sync`, and we do not provide a manual `unsafe impl Sync`. The
// `Send` bound is also conservative: in practice the driver moves
// between executor tasks freely as long as serialized, but we leave
// that to the kernel scheduler.

impl VirtioBlk {
    /// Probe an MMIO window. Returns `Some(())` if the magic + device
    /// ID match a virtio-blk-modern device.
    ///
    /// # Safety
    /// `mmio_base` must point to a 0x200-byte MMIO window mapped for
    /// volatile access.
    pub unsafe fn probe(mmio_base: usize) -> bool {
        let magic = read_volatile((mmio_base + REG_MAGIC) as *const u32);
        if magic != VIRTIO_MMIO_MAGIC {
            return false;
        }
        let version = read_volatile((mmio_base + REG_VERSION) as *const u32);
        if version != 2 {
            return false;
        }
        let dev_id = read_volatile((mmio_base + REG_DEVICE_ID) as *const u32);
        dev_id == VIRTIO_DEVICE_ID_BLOCK
    }

    /// Initialize a virtio-blk device.
    ///
    /// # Safety
    /// - `mmio_base` must be a virtio-blk-modern MMIO window
    ///   ([`probe`](Self::probe) returned true).
    /// - `queue` must live in identity-mapped, DMA-coherent memory
    ///   for the lifetime of the `VirtioBlk`.
    /// - The caller must externally serialize access to the driver
    ///   (`VirtioBlk` is `!Sync`).
    pub unsafe fn new(
        mmio_base: usize,
        queue: &'static mut QueueStorage,
    ) -> Result<Self, BlockError> {
        // 1. Reset the device.
        write_reg(mmio_base, REG_STATUS, 0);

        // 2. Set ACKNOWLEDGE | DRIVER.
        write_reg(mmio_base, REG_STATUS, STATUS_ACKNOWLEDGE);
        write_reg(mmio_base, REG_STATUS, STATUS_ACKNOWLEDGE | STATUS_DRIVER);

        // 3. Read device features (low 32 + high 32).
        write_reg(mmio_base, REG_DEVICE_FEATURES_SEL, 0);
        let dev_feat_lo = read_reg(mmio_base, REG_DEVICE_FEATURES);
        write_reg(mmio_base, REG_DEVICE_FEATURES_SEL, 1);
        let dev_feat_hi = read_reg(mmio_base, REG_DEVICE_FEATURES);

        // We need VIRTIO_F_VERSION_1 (bit 32 in the high half) to
        // operate in modern mode.
        let want_version_1 = 1u32 << (VIRTIO_F_VERSION_1 - 32);
        if dev_feat_hi & want_version_1 == 0 {
            write_reg(mmio_base, REG_STATUS, STATUS_FAILED);
            return Err(BlockError::NotPresent);
        }

        // Optional: VIRTIO_BLK_F_FLUSH (bit 9 in the low half).
        let has_flush = (dev_feat_lo & (1u32 << VIRTIO_BLK_F_FLUSH)) != 0;

        // Negotiate features: only VERSION_1 (+ FLUSH if available).
        let drv_feat_lo = if has_flush {
            1u32 << VIRTIO_BLK_F_FLUSH
        } else {
            0
        };
        let drv_feat_hi = want_version_1;
        write_reg(mmio_base, REG_DRIVER_FEATURES_SEL, 0);
        write_reg(mmio_base, REG_DRIVER_FEATURES, drv_feat_lo);
        write_reg(mmio_base, REG_DRIVER_FEATURES_SEL, 1);
        write_reg(mmio_base, REG_DRIVER_FEATURES, drv_feat_hi);

        // 4. Set FEATURES_OK and re-read to confirm.
        write_reg(
            mmio_base,
            REG_STATUS,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK,
        );
        let status = read_reg(mmio_base, REG_STATUS);
        if status & STATUS_FEATURES_OK == 0 {
            write_reg(mmio_base, REG_STATUS, status | STATUS_FAILED);
            return Err(BlockError::NotPresent);
        }

        // 5. Set up virtqueue 0.
        write_reg(mmio_base, REG_QUEUE_SEL, 0);
        let max = read_reg(mmio_base, REG_QUEUE_NUM_MAX);
        if max == 0 {
            write_reg(mmio_base, REG_STATUS, status | STATUS_FAILED);
            return Err(BlockError::NotPresent);
        }
        let q_num = if max < QUEUE_SIZE as u32 {
            max as u16
        } else {
            QUEUE_SIZE
        };
        write_reg(mmio_base, REG_QUEUE_NUM, q_num as u32);

        // Program ring physical addresses (identity-mapped → virt == phys).
        let desc_addr = queue.descs.as_ptr() as u64;
        let avail_addr = (&queue.avail as *const _) as u64;
        let used_addr = (&queue.used as *const _) as u64;
        write_reg(mmio_base, REG_QUEUE_DESC_LOW, desc_addr as u32);
        write_reg(mmio_base, REG_QUEUE_DESC_HIGH, (desc_addr >> 32) as u32);
        write_reg(mmio_base, REG_QUEUE_AVAIL_LOW, avail_addr as u32);
        write_reg(mmio_base, REG_QUEUE_AVAIL_HIGH, (avail_addr >> 32) as u32);
        write_reg(mmio_base, REG_QUEUE_USED_LOW, used_addr as u32);
        write_reg(mmio_base, REG_QUEUE_USED_HIGH, (used_addr >> 32) as u32);

        // Mark queue ready.
        write_reg(mmio_base, REG_QUEUE_READY, 1);

        // 6. Read capacity from device-config space.
        let cfg = (mmio_base + REG_CONFIG) as *const VirtioBlkConfig;
        let cap_lo = read_volatile(&(*cfg).capacity_low) as u64;
        let cap_hi = read_volatile(&(*cfg).capacity_high) as u64;
        let capacity_sectors = (cap_hi << 32) | cap_lo;

        // 7. DRIVER_OK → device is live.
        write_reg(
            mmio_base,
            REG_STATUS,
            STATUS_ACKNOWLEDGE | STATUS_DRIVER | STATUS_FEATURES_OK | STATUS_DRIVER_OK,
        );

        Ok(Self {
            mmio_base,
            inner: UnsafeCell::new(VirtioBlkInner {
                queue,
                last_used_idx: 0,
            }),
            capacity_sectors,
            has_flush,
        })
    }

    /// Submit a request and busy-wait for completion.
    ///
    /// Single-queue, three-descriptor chain layout:
    /// - desc[0]: header (device-readable)
    /// - desc[1]: data buffer (device-writable for IN, device-readable for OUT)
    /// - desc[2]: status byte (device-writable)
    ///
    /// For FLUSH there is no data buffer, so the chain is two
    /// descriptors (header + status).
    ///
    /// # Safety
    /// - Caller guarantees `data` (if `Some`) is identity-mapped DMA
    ///   memory for the duration of the call.
    /// - Caller guarantees no other thread/task is calling
    ///   `submit_and_wait` on this `VirtioBlk` concurrently
    ///   (`VirtioBlk` is `!Sync`).
    unsafe fn submit_and_wait(
        &self,
        req_type: u32,
        sector: u64,
        data: Option<(*mut u8, usize, bool)>, // (ptr, len, is_write_to_device)
    ) -> Result<(), BlockError> {
        // SAFETY: `VirtioBlk` is `!Sync` and the caller serializes
        // access. We hold the only active `&mut VirtioBlkInner` for
        // the duration of this call.
        let inner: &mut VirtioBlkInner = &mut *self.inner.get();

        // Use slot 0 deterministically — this is single-threaded
        // submit-and-wait; we never have more than one in flight.
        let slot = 0usize;

        // Populate header.
        inner.queue.headers[slot] = VirtioBlkReqHeader {
            ty: req_type,
            reserved: 0,
            sector,
        };
        inner.queue.statuses[slot] = 0xFFu8; // sentinel "not yet written by device"

        // Build descriptor chain.
        let header_addr = (&inner.queue.headers[slot] as *const _) as u64;
        let status_addr = (&inner.queue.statuses[slot] as *const _) as u64;

        match data {
            Some((data_ptr, data_len, write_to_device)) => {
                // 3-descriptor chain.
                inner.queue.descs[0] = VringDesc {
                    addr: header_addr,
                    len: core::mem::size_of::<VirtioBlkReqHeader>() as u32,
                    flags: VRING_DESC_F_NEXT,
                    next: 1,
                };
                let mut data_flags = VRING_DESC_F_NEXT;
                if write_to_device {
                    // VIRTIO_BLK_T_IN means the *device* writes the buffer
                    // (it's reading from disk, writing into our memory).
                    data_flags |= VRING_DESC_F_WRITE;
                }
                inner.queue.descs[1] = VringDesc {
                    addr: data_ptr as u64,
                    len: data_len as u32,
                    flags: data_flags,
                    next: 2,
                };
                inner.queue.descs[2] = VringDesc {
                    addr: status_addr,
                    len: 1,
                    flags: VRING_DESC_F_WRITE,
                    next: 0,
                };
            }
            None => {
                // FLUSH: 2-descriptor chain (header + status).
                inner.queue.descs[0] = VringDesc {
                    addr: header_addr,
                    len: core::mem::size_of::<VirtioBlkReqHeader>() as u32,
                    flags: VRING_DESC_F_NEXT,
                    next: 1,
                };
                inner.queue.descs[1] = VringDesc {
                    addr: status_addr,
                    len: 1,
                    flags: VRING_DESC_F_WRITE,
                    next: 0,
                };
            }
        }

        // Publish to the available ring.
        let avail_idx = inner.queue.avail.idx;
        inner.queue.avail.ring[(avail_idx as usize) & (QUEUE_SIZE as usize - 1)] = 0; // head desc index
                                                                                      // Memory barrier: ensure descriptor writes are visible before
                                                                                      // we publish the new avail.idx.
        fence(Ordering::Release);
        inner.queue.avail.idx = avail_idx.wrapping_add(1);
        // Another barrier before the device-visible notify.
        fence(Ordering::Release);

        // Notify queue 0.
        write_reg(self.mmio_base, REG_QUEUE_NOTIFY, 0);

        // Busy-wait for used.idx to advance. In CI this returns within
        // microseconds; the production retry-policy wraps this call
        // and converts hangs to BlockError::Timeout.
        let target = inner.last_used_idx.wrapping_add(1);
        let mut spins: u64 = 0;
        while read_volatile(&inner.queue.used.idx) != target {
            core::hint::spin_loop();
            spins = spins.wrapping_add(1);
            // Fail-safe: 100M spins ~ a few hundred ms on a modern CPU.
            // The retry layer handles real timeouts; this is only here
            // so a totally-wedged device surfaces an error.
            if spins > 100_000_000 {
                return Err(BlockError::Timeout);
            }
        }
        inner.last_used_idx = target;

        // Memory barrier before reading device-written status.
        fence(Ordering::Acquire);
        compiler_fence(Ordering::Acquire);

        let status = read_volatile(&inner.queue.statuses[slot]);
        match status {
            VIRTIO_BLK_S_OK => Ok(()),
            VIRTIO_BLK_S_IOERR => Err(BlockError::MediaError),
            VIRTIO_BLK_S_UNSUPP => Err(BlockError::NotPresent),
            // 0xFF means the device never wrote — treat as media error.
            _ => Err(BlockError::MediaError),
        }
    }

    /// Capacity in 512-byte sectors as reported by the device.
    pub fn capacity_sectors(&self) -> u64 {
        self.capacity_sectors
    }

    /// `true` if the device negotiated `VIRTIO_BLK_F_FLUSH`.
    pub fn supports_flush(&self) -> bool {
        self.has_flush
    }
}

impl BlockDevice for VirtioBlk {
    fn read_block(&self, lba: u64, buf: &mut [u8]) -> Result<(), BlockError> {
        check_buf_alignment(buf.len(), VIRTIO_BLK_SECTOR_SIZE)?;
        check_lba_in_range(
            lba,
            buf.len(),
            VIRTIO_BLK_SECTOR_SIZE,
            self.capacity_sectors,
        )?;
        let ptr = buf.as_mut_ptr();
        let len = buf.len();
        // SAFETY: VirtioBlk is `!Sync`; the caller ensures no
        // concurrent submit_and_wait. The buffer is borrowed
        // exclusively (&mut [u8]) for the duration of the call.
        unsafe { self.submit_and_wait(VIRTIO_BLK_T_IN, lba, Some((ptr, len, true))) }
    }

    fn write_block(&mut self, lba: u64, buf: &[u8]) -> Result<(), BlockError> {
        check_buf_alignment(buf.len(), VIRTIO_BLK_SECTOR_SIZE)?;
        check_lba_in_range(
            lba,
            buf.len(),
            VIRTIO_BLK_SECTOR_SIZE,
            self.capacity_sectors,
        )?;
        let ptr = buf.as_ptr() as *mut u8;
        let len = buf.len();
        // SAFETY: same as read_block above; data buffer is read-only
        // by the device for VIRTIO_BLK_T_OUT.
        unsafe { self.submit_and_wait(VIRTIO_BLK_T_OUT, lba, Some((ptr, len, false))) }
    }

    fn block_size_bytes(&self) -> u32 {
        VIRTIO_BLK_SECTOR_SIZE
    }

    fn block_count(&self) -> u64 {
        self.capacity_sectors
    }

    fn flush(&mut self) -> Result<(), BlockError> {
        if !self.has_flush {
            // Spec § 5.2.6.2: "The device MAY support the
            // VIRTIO_BLK_T_FLUSH command [...] If the device does not
            // negotiate VIRTIO_BLK_F_FLUSH, the driver MUST NOT issue
            // a flush." Treat as a successful no-op.
            return Ok(());
        }
        unsafe { self.submit_and_wait(VIRTIO_BLK_T_FLUSH, 0, None) }
    }
}

// ─── Volatile MMIO accessors ────────────────────────────────────────────────

#[inline(always)]
unsafe fn write_reg(base: usize, off: usize, val: u32) {
    write_volatile((base + off) as *mut u32, val);
}

#[inline(always)]
unsafe fn read_reg(base: usize, off: usize) -> u32 {
    read_volatile((base + off) as *const u32)
}
