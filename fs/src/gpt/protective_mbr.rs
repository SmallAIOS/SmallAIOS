// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Protective MBR writer.
//!
//! Per UEFI 2.10 §5.2.3, a GPT-formatted disk SHALL contain a
//! protective MBR at LBA 0. The protective MBR has:
//!
//! - Boot code: zero-filled (446 bytes, offset 0..446).
//! - Partition table: 4 entries × 16 bytes = 64 bytes (offset 446..510).
//!   The first entry is type `0xEE` (GPT protective) covering the whole
//!   disk; the other three are zero.
//! - Boot signature: `0x55AA` little-endian at offset 510..512.
//!
//! The single 0xEE entry's `starting_lba` is 1 and its `size_in_lba`
//! is `device.block_count() - 1`, capped at `0xFFFFFFFF` for disks
//! larger than 2 TiB.
//!
//! This is the only GPT-related write operation v1 SmallAIOS performs;
//! GPT itself is partitioned externally. Once the kernel can read GPT
//! it can also lay down the protective MBR for greenfield systems.

extern crate alloc;

use crate::block::BlockDevice;

use super::{GptError, MBR_SIGNATURE_LE, MBR_TYPE_GPT_PROTECTIVE};

/// Write a protective MBR to LBA 0.
///
/// Per the GPT spec the entry uses CHS values of `0x000200` for start
/// (cylinder 0, head 0, sector 2) and `0xFFFFFF` for end (max), so
/// pre-EFI boot loaders that walk the MBR table see one large
/// protective partition.
pub fn write_protective_mbr(device: &mut dyn BlockDevice) -> Result<(), GptError> {
    let bs = device.block_size_bytes() as usize;
    if bs < 512 {
        // GPT spec requires LBA 0 to be at least 512 bytes; the
        // SmallAIOS sector adapter never emits a sub-512-byte device.
        return Err(GptError::EntryCountOutOfRange);
    }

    let mut block = alloc::vec![0u8; bs];

    // Boot code (offset 0..446) is zeroed by the vec! macro.

    // Partition table starts at offset 446. Build the single 0xEE
    // entry covering LBA 1 .. block_count() - 1.
    let pt = &mut block[446..446 + 64];
    write_protective_entry(pt, device.block_count());

    // Boot signature 0x55AA at offset 510..512 (little-endian).
    block[510] = MBR_SIGNATURE_LE[0];
    block[511] = MBR_SIGNATURE_LE[1];

    device.write_block(0, &block)?;
    device.flush()?;
    Ok(())
}

/// Build a single MBR partition entry of type `0xEE` covering LBAs
/// 1 .. `block_count - 1` into the first 16 bytes of `pt`. The other
/// three entries (offsets 16..32, 32..48, 48..64) are left untouched.
fn write_protective_entry(pt: &mut [u8], block_count: u64) {
    debug_assert!(pt.len() >= 16);
    // Boot indicator: 0x00 (non-bootable).
    pt[0] = 0x00;
    // Starting CHS: cylinder=0, head=0, sector=2 → bytes [0x00, 0x02, 0x00].
    pt[1] = 0x00;
    pt[2] = 0x02;
    pt[3] = 0x00;
    // Partition type: 0xEE (GPT protective).
    pt[4] = MBR_TYPE_GPT_PROTECTIVE;
    // Ending CHS: 0xFFFFFF (max — protective marker).
    pt[5] = 0xFF;
    pt[6] = 0xFF;
    pt[7] = 0xFF;
    // Starting LBA = 1 (LE u32).
    pt[8..12].copy_from_slice(&1u32.to_le_bytes());
    // Size in LBAs = block_count - 1, capped at 0xFFFFFFFF for >2 TiB
    // disks per UEFI 2.10 §5.2.3.
    let size = block_count.saturating_sub(1);
    let size_u32 = if size > 0xFFFF_FFFF {
        0xFFFF_FFFFu32
    } else {
        size as u32
    };
    pt[12..16].copy_from_slice(&size_u32.to_le_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::block::mock::MockBlockDevice;

    #[test]
    fn writes_signature_and_type_byte() {
        let mut dev = MockBlockDevice::new(512, 64);
        write_protective_mbr(&mut dev).unwrap();
        let lba0 = dev.snapshot(0, 512);
        assert_eq!(lba0[510], 0x55);
        assert_eq!(lba0[511], 0xAA);
        // Type byte at offset 446 + 4 = 450.
        assert_eq!(lba0[450], 0xEE);
        // Starting LBA = 1.
        assert_eq!(&lba0[454..458], &1u32.to_le_bytes());
        // Size = block_count - 1 = 63.
        assert_eq!(&lba0[458..462], &63u32.to_le_bytes());
    }

    #[test]
    fn boot_code_zero_filled() {
        let mut dev = MockBlockDevice::new(512, 64);
        // Pre-pollute LBA 0 with non-zero bytes.
        dev.preload(0, &alloc::vec![0xAB; 512]);
        write_protective_mbr(&mut dev).unwrap();
        let lba0 = dev.snapshot(0, 512);
        assert!(lba0[0..446].iter().all(|&b| b == 0));
    }

    #[test]
    fn other_three_entries_zero() {
        let mut dev = MockBlockDevice::new(512, 64);
        dev.preload(0, &alloc::vec![0xAB; 512]);
        write_protective_mbr(&mut dev).unwrap();
        let lba0 = dev.snapshot(0, 512);
        assert!(lba0[462..510].iter().all(|&b| b == 0));
    }

    #[test]
    fn flushes_after_write() {
        let mut dev = MockBlockDevice::new(512, 64);
        assert_eq!(dev.flush_count(), 0);
        write_protective_mbr(&mut dev).unwrap();
        assert_eq!(dev.flush_count(), 1);
    }

    #[test]
    fn caps_size_at_u32_max_for_large_disks() {
        let mut pt = [0u8; 64];
        // 8 TiB at 512 B/sector = 16e9 LBAs > u32::MAX.
        write_protective_entry(&mut pt, 16_000_000_000);
        assert_eq!(pt[4], 0xEE);
        assert_eq!(&pt[12..16], &0xFFFF_FFFFu32.to_le_bytes());
    }

    #[test]
    fn handles_4k_native_devices() {
        let mut dev = MockBlockDevice::new(4096, 16);
        write_protective_mbr(&mut dev).unwrap();
        let lba0 = dev.snapshot(0, 4096);
        // Signature still at offset 510..512.
        assert_eq!(lba0[510], 0x55);
        assert_eq!(lba0[511], 0xAA);
        // Type byte at offset 450.
        assert_eq!(lba0[450], 0xEE);
        // Size = 16 - 1 = 15.
        assert_eq!(&lba0[458..462], &15u32.to_le_bytes());
        // Bytes from offset 512 onward should remain zero (4K block,
        // protective MBR fills the first 512 bytes only).
        assert!(lba0[512..].iter().all(|&b| b == 0));
    }

    #[test]
    fn block_error_propagated() {
        let mut dev = MockBlockDevice::new(512, 64);
        dev.set_permanent_failure(crate::block::BlockError::MediaError);
        let res = write_protective_mbr(&mut dev);
        assert!(matches!(
            res,
            Err(GptError::BlockError(crate::block::BlockError::MediaError))
        ));
    }
}
