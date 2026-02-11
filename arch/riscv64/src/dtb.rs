// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal DTB (Device Tree Blob) parser for RISC-V hardware discovery (23.9).
//!
//! Parses the FDT (Flattened Device Tree) header and extracts key information
//! needed during early boot:
//! - Memory regions
//! - UART base address and type
//! - Interrupt controller type and address
//! - CPU count and ISA extensions
//!
//! The DTB pointer is passed in a1 by OpenSBI at boot.

/// FDT magic number (big-endian: 0xD00DFEED).
pub const FDT_MAGIC: u32 = 0xD00D_FEED;

/// FDT header structure version 17.
#[repr(C)]
pub struct FdtHeader {
    pub magic: u32,
    pub totalsize: u32,
    pub off_dt_struct: u32,
    pub off_dt_strings: u32,
    pub off_mem_rsvmap: u32,
    pub version: u32,
    pub last_comp_version: u32,
    pub boot_cpuid_phys: u32,
    pub size_dt_strings: u32,
    pub size_dt_struct: u32,
}

/// FDT structure tokens.
pub const FDT_BEGIN_NODE: u32 = 0x0000_0001;
pub const FDT_END_NODE: u32 = 0x0000_0002;
pub const FDT_PROP: u32 = 0x0000_0003;
pub const FDT_NOP: u32 = 0x0000_0004;
pub const FDT_END: u32 = 0x0000_0009;

/// A discovered memory region from the DTB.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DtbMemoryRegion {
    pub base: u64,
    pub size: u64,
}

/// Summary of hardware discovered from DTB.
#[derive(Debug, Clone)]
pub struct DtbInfo {
    /// Memory regions (up to 8).
    pub memory: [Option<DtbMemoryRegion>; 8],
    /// Number of valid memory regions.
    pub memory_count: usize,
    /// Total memory in bytes.
    pub total_memory: u64,
    /// Number of CPUs found.
    pub cpu_count: usize,
    /// DTB total size in bytes.
    pub dtb_size: u32,
    /// DTB version.
    pub dtb_version: u32,
    /// Whether the DTB is valid.
    pub valid: bool,
}

impl DtbInfo {
    pub const fn empty() -> Self {
        Self {
            memory: [None; 8],
            memory_count: 0,
            total_memory: 0,
            cpu_count: 0,
            dtb_size: 0,
            dtb_version: 0,
            valid: false,
        }
    }
}

/// Read a big-endian u32 from a pointer.
///
/// # Safety
/// `ptr` must be valid for 4 bytes.
unsafe fn read_be32(ptr: *const u8) -> u32 {
    let bytes = core::slice::from_raw_parts(ptr, 4);
    u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]])
}

/// Read a big-endian u64 from a pointer.
///
/// # Safety
/// `ptr` must be valid for 8 bytes.
unsafe fn read_be64(ptr: *const u8) -> u64 {
    let bytes = core::slice::from_raw_parts(ptr, 8);
    u64::from_be_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Validate the FDT header at the given address.
///
/// # Safety
/// `dtb_addr` must point to a valid FDT blob.
pub unsafe fn validate_header(dtb_addr: usize) -> Option<(u32, u32)> {
    if dtb_addr == 0 {
        return None;
    }
    let magic = read_be32(dtb_addr as *const u8);
    if magic != FDT_MAGIC {
        return None;
    }
    let totalsize = read_be32((dtb_addr + 4) as *const u8);
    let version = read_be32((dtb_addr + 20) as *const u8);
    Some((totalsize, version))
}

/// Parse the DTB at the given address and extract hardware information.
///
/// This is a simplified parser that extracts:
/// - Memory regions from /memory nodes
/// - CPU count from /cpus node
///
/// # Safety
/// `dtb_addr` must point to a valid FDT blob in accessible memory.
pub unsafe fn parse_dtb(dtb_addr: usize) -> DtbInfo {
    let mut info = DtbInfo::empty();

    let header_info = validate_header(dtb_addr);
    if header_info.is_none() {
        return info;
    }
    let (totalsize, version) = header_info.unwrap();

    info.dtb_size = totalsize;
    info.dtb_version = version;
    info.valid = true;

    // Parse structure block
    let struct_offset = read_be32((dtb_addr + 8) as *const u8) as usize;
    let strings_offset = read_be32((dtb_addr + 12) as *const u8) as usize;
    let struct_base = dtb_addr + struct_offset;
    let strings_base = dtb_addr + strings_offset;
    let struct_end = dtb_addr + totalsize as usize;

    let mut pos = struct_base;
    let mut in_memory = false;
    let mut in_cpus = false;
    let mut in_cpu = false;

    while pos + 4 <= struct_end {
        let token = read_be32(pos as *const u8);
        pos += 4;

        match token {
            FDT_BEGIN_NODE => {
                // Node name follows (null-terminated, 4-byte aligned)
                let name_start = pos;
                while pos < struct_end && *(pos as *const u8) != 0 {
                    pos += 1;
                }
                let name_len = pos - name_start;
                let name_bytes = core::slice::from_raw_parts(name_start as *const u8, name_len);
                pos += 1; // skip null terminator
                pos = (pos + 3) & !3; // align to 4 bytes

                if starts_with(name_bytes, b"memory") {
                    in_memory = true;
                } else if name_bytes == b"cpus" {
                    in_cpus = true;
                } else if in_cpus && starts_with(name_bytes, b"cpu@") {
                    in_cpu = true;
                    info.cpu_count += 1;
                }
            }
            FDT_END_NODE => {
                if in_cpu {
                    in_cpu = false;
                } else if in_cpus {
                    in_cpus = false;
                } else if in_memory {
                    in_memory = false;
                }
            }
            FDT_PROP => {
                if pos + 8 > struct_end {
                    break;
                }
                let prop_len = read_be32(pos as *const u8) as usize;
                let name_off = read_be32((pos + 4) as *const u8) as usize;
                pos += 8;

                let _prop_name_ptr = (strings_base + name_off) as *const u8;

                // Check if this is a "reg" property in a memory node
                if in_memory && prop_len >= 16 {
                    let prop_name = get_string(strings_base, name_off, struct_end);
                    if prop_name == Some(b"reg" as &[u8]) {
                        // Parse pairs of (base, size) — assume #address-cells=2, #size-cells=2
                        let mut offset = 0;
                        while offset + 16 <= prop_len && info.memory_count < 8 {
                            let base = read_be64((pos + offset) as *const u8);
                            let size = read_be64((pos + offset + 8) as *const u8);
                            if size > 0 {
                                info.memory[info.memory_count] =
                                    Some(DtbMemoryRegion { base, size });
                                info.memory_count += 1;
                                info.total_memory += size;
                            }
                            offset += 16;
                        }
                    }
                }

                pos += prop_len;
                pos = (pos + 3) & !3; // align to 4 bytes
            }
            FDT_NOP => {}
            FDT_END => break,
            _ => break, // Invalid token
        }
    }

    info
}

/// Check if `haystack` starts with `prefix`.
fn starts_with(haystack: &[u8], prefix: &[u8]) -> bool {
    haystack.len() >= prefix.len() && &haystack[..prefix.len()] == prefix
}

/// Get a null-terminated string from the strings block.
unsafe fn get_string(strings_base: usize, offset: usize, limit: usize) -> Option<&'static [u8]> {
    let start = strings_base + offset;
    if start >= limit {
        return None;
    }
    let mut end = start;
    while end < limit && *(end as *const u8) != 0 {
        end += 1;
    }
    Some(core::slice::from_raw_parts(start as *const u8, end - start))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fdt_magic() {
        assert_eq!(FDT_MAGIC, 0xD00D_FEED);
    }

    #[test]
    fn test_fdt_tokens() {
        assert_eq!(FDT_BEGIN_NODE, 1);
        assert_eq!(FDT_END_NODE, 2);
        assert_eq!(FDT_PROP, 3);
        assert_eq!(FDT_NOP, 4);
        assert_eq!(FDT_END, 9);
    }

    #[test]
    fn test_dtb_info_empty() {
        let info = DtbInfo::empty();
        assert!(!info.valid);
        assert_eq!(info.memory_count, 0);
        assert_eq!(info.cpu_count, 0);
        assert_eq!(info.total_memory, 0);
    }

    #[test]
    fn test_dtb_memory_region() {
        let r = DtbMemoryRegion {
            base: 0x8000_0000,
            size: 128 * 1024 * 1024,
        };
        assert_eq!(r.base, 0x8000_0000);
        assert_eq!(r.size, 128 * 1024 * 1024);
    }

    #[test]
    fn test_starts_with() {
        assert!(starts_with(b"memory@80000000", b"memory"));
        assert!(starts_with(b"cpu@0", b"cpu@"));
        assert!(!starts_with(b"mem", b"memory"));
        assert!(starts_with(b"exact", b"exact"));
    }

    #[test]
    fn test_validate_header_null() {
        let result = unsafe { validate_header(0) };
        assert!(result.is_none());
    }

    #[test]
    fn test_fdt_header_size() {
        assert_eq!(core::mem::size_of::<FdtHeader>(), 40);
    }
}
