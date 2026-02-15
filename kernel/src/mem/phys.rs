// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Physical memory map — boot-time enumeration of usable RAM regions.
//!
//! Two parsers:
//! - Multiboot2 memory map tag (x86-64)
//! - Device Tree Blob (DTB) `/memory` nodes (ARM64)
//!
//! Both produce a flat array of `MemRegion` entries used by the buddy allocator.

use super::PhysAddr;

/// Maximum number of physical memory regions we track.
const MAX_REGIONS: usize = 64;

/// Type of a physical memory region.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegionKind {
    /// Usable RAM available for allocation.
    Usable,
    /// Reserved by firmware or hardware (MMIO, ACPI, etc).
    Reserved,
    /// ACPI reclaimable memory.
    AcpiReclaimable,
    /// Kernel image and boot data — not allocatable.
    Kernel,
}

/// A contiguous physical memory region.
#[derive(Debug, Clone, Copy)]
pub struct MemRegion {
    pub base: PhysAddr,
    pub size: usize,
    pub kind: RegionKind,
}

/// Physical memory map built during boot.
pub struct PhysMemoryMap {
    regions: [MemRegion; MAX_REGIONS],
    count: usize,
}

impl Default for PhysMemoryMap {
    fn default() -> Self {
        Self::new()
    }
}

impl PhysMemoryMap {
    /// Create an empty memory map.
    pub const fn new() -> Self {
        Self {
            regions: [MemRegion {
                base: PhysAddr(0),
                size: 0,
                kind: RegionKind::Reserved,
            }; MAX_REGIONS],
            count: 0,
        }
    }

    /// Add a region to the memory map. Returns false if the map is full.
    pub fn add_region(&mut self, base: PhysAddr, size: usize, kind: RegionKind) -> bool {
        if self.count >= MAX_REGIONS {
            return false;
        }
        self.regions[self.count] = MemRegion { base, size, kind };
        self.count += 1;
        true
    }

    /// Get the number of regions.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Get a region by index.
    pub fn get(&self, index: usize) -> Option<&MemRegion> {
        if index < self.count {
            Some(&self.regions[index])
        } else {
            None
        }
    }

    /// Iterate over all regions.
    pub fn iter(&self) -> impl Iterator<Item = &MemRegion> {
        self.regions[..self.count].iter()
    }

    /// Iterate over usable regions only.
    pub fn usable_regions(&self) -> impl Iterator<Item = &MemRegion> {
        self.iter().filter(|r| r.kind == RegionKind::Usable)
    }

    /// Total usable memory in bytes.
    pub fn total_usable(&self) -> usize {
        self.usable_regions().map(|r| r.size).sum()
    }

    /// Exclude a range from usable regions (e.g., kernel image area).
    /// Splits or shrinks usable regions that overlap [excl_base, excl_base+excl_size).
    pub fn exclude_range(&mut self, excl_base: PhysAddr, excl_size: usize) {
        let excl_end = excl_base.as_usize() + excl_size;
        let mut i = 0;
        while i < self.count {
            let r = &self.regions[i];
            if r.kind != RegionKind::Usable {
                i += 1;
                continue;
            }

            let r_base = r.base.as_usize();
            let r_end = r_base + r.size;

            // No overlap
            if excl_end <= r_base || excl_base.as_usize() >= r_end {
                i += 1;
                continue;
            }

            // Full overlap — mark as kernel
            if excl_base.as_usize() <= r_base && excl_end >= r_end {
                self.regions[i].kind = RegionKind::Kernel;
                i += 1;
                continue;
            }

            // Exclusion splits the region into two parts
            if excl_base.as_usize() > r_base && excl_end < r_end {
                // Shrink current to [r_base, excl_base)
                self.regions[i].size = excl_base.as_usize() - r_base;
                // Insert new region [excl_end, r_end)
                let new_size = r_end - excl_end;
                self.add_region(PhysAddr::new(excl_end), new_size, RegionKind::Usable);
                i += 1;
                continue;
            }

            // Overlap at the start — shrink from the left
            if excl_base.as_usize() <= r_base {
                let new_base = excl_end;
                let new_size = r_end - excl_end;
                self.regions[i].base = PhysAddr::new(new_base);
                self.regions[i].size = new_size;
            } else {
                // Overlap at the end — shrink from the right
                self.regions[i].size = excl_base.as_usize() - r_base;
            }

            i += 1;
        }
    }
}

// ─── Multiboot2 Memory Map Parser (x86-64) ──────────────────────────────────

/// Multiboot2 tag types.
const MB2_TAG_END: u32 = 0;
const MB2_TAG_MEMORY_MAP: u32 = 6;

/// Multiboot2 memory map entry types.
const MB2_MEM_AVAILABLE: u32 = 1;
const _MB2_MEM_RESERVED: u32 = 2;
const MB2_MEM_ACPI_RECLAIMABLE: u32 = 3;

/// Parse a Multiboot2 boot information structure to extract the memory map.
///
/// # Safety
/// `mb_info_addr` must point to a valid Multiboot2 boot information structure.
pub unsafe fn parse_multiboot2(mb_info_addr: usize, map: &mut PhysMemoryMap) {
    let total_size = *(mb_info_addr as *const u32);
    let mut offset: usize = 8; // Skip total_size and reserved fields

    while offset < total_size as usize {
        let tag_ptr = (mb_info_addr + offset) as *const u32;
        let tag_type = *tag_ptr;
        let tag_size = *tag_ptr.add(1);

        if tag_type == MB2_TAG_END {
            break;
        }

        if tag_type == MB2_TAG_MEMORY_MAP {
            parse_mb2_mmap_tag(mb_info_addr + offset, tag_size as usize, map);
        }

        // Tags are 8-byte aligned
        offset += ((tag_size as usize) + 7) & !7;
    }
}

/// Parse a Multiboot2 memory map tag.
///
/// # Safety
/// `tag_addr` must point to a valid Multiboot2 memory map tag.
unsafe fn parse_mb2_mmap_tag(tag_addr: usize, tag_size: usize, map: &mut PhysMemoryMap) {
    // Tag header: type(4) + size(4) + entry_size(4) + entry_version(4)
    let entry_size = *((tag_addr + 8) as *const u32) as usize;
    if entry_size == 0 {
        return;
    }

    let entries_start = tag_addr + 16;
    let entries_end = tag_addr + tag_size;
    let mut entry_addr = entries_start;

    while entry_addr + entry_size <= entries_end {
        // Entry: base_addr(8) + length(8) + type(4) + reserved(4)
        let base = *((entry_addr) as *const u64) as usize;
        let length = *((entry_addr + 8) as *const u64) as usize;
        let mem_type = *((entry_addr + 16) as *const u32);

        let kind = match mem_type {
            MB2_MEM_AVAILABLE => RegionKind::Usable,
            MB2_MEM_ACPI_RECLAIMABLE => RegionKind::AcpiReclaimable,
            _ => RegionKind::Reserved,
        };

        if length > 0 {
            map.add_region(PhysAddr::new(base), length, kind);
        }

        entry_addr += entry_size;
    }
}

// ─── DTB/FDT Memory Parser (ARM64) ──────────────────────────────────────────

/// FDT magic number.
const FDT_MAGIC: u32 = 0xD00DFEED;

/// FDT structure block tokens.
const FDT_BEGIN_NODE: u32 = 0x00000001;
const FDT_END_NODE: u32 = 0x00000002;
const FDT_PROP: u32 = 0x00000003;
const FDT_NOP: u32 = 0x00000004;
const FDT_END: u32 = 0x00000009;

/// Read a big-endian u32 from a pointer.
///
/// # Safety
/// `ptr` must point to a valid, aligned u32.
#[inline]
unsafe fn read_be32(ptr: *const u8) -> u32 {
    let b = core::slice::from_raw_parts(ptr, 4);
    u32::from_be_bytes([b[0], b[1], b[2], b[3]])
}

/// Read a big-endian u64 from a pointer.
///
/// # Safety
/// `ptr` must point to a valid u64 (need not be aligned).
#[inline]
unsafe fn read_be64(ptr: *const u8) -> u64 {
    let b = core::slice::from_raw_parts(ptr, 8);
    u64::from_be_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]])
}

/// Compare a null-terminated C string at `ptr` with a Rust &str.
///
/// # Safety
/// `ptr` must point to a valid null-terminated string.
unsafe fn cstr_eq(ptr: *const u8, s: &str) -> bool {
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        if *ptr.add(i) != byte {
            return false;
        }
    }
    *ptr.add(bytes.len()) == 0
}

/// Length of a null-terminated C string (excluding null).
///
/// # Safety
/// `ptr` must point to a valid null-terminated string.
unsafe fn cstr_len(ptr: *const u8) -> usize {
    let mut len = 0;
    while *ptr.add(len) != 0 {
        len += 1;
    }
    len
}

/// State tracked while walking the FDT structure block.
struct DtbParseState {
    in_memory_node: bool,
    depth: usize,
    memory_depth: usize,
    address_cells: u32,
    size_cells: u32,
}

impl DtbParseState {
    fn new() -> Self {
        Self {
            in_memory_node: false,
            depth: 0,
            memory_depth: 0,
            // Defaults for ARM64
            address_cells: 2,
            size_cells: 2,
        }
    }
}

/// Check whether a node name represents a `/memory` node.
///
/// A memory node is named exactly "memory" or "memory@<unit-address>".
///
/// # Safety
/// `name_ptr` must point to a valid, null-terminated C string of `name_len` bytes
/// (excluding the null terminator).
unsafe fn is_memory_node(name_ptr: *const u8, name_len: usize) -> bool {
    if name_len == 6 {
        return cstr_eq(name_ptr, "memory");
    }
    if name_len > 7 {
        let prefix = core::slice::from_raw_parts(name_ptr, 6);
        return prefix == b"memory" && *name_ptr.add(6) == b'@';
    }
    false
}

/// Process an FDT_BEGIN_NODE token, updating parse state and advancing `pos`.
///
/// # Safety
/// `struct_base.add(pos)` must point to the node name within a valid FDT structure block.
unsafe fn handle_begin_node(struct_base: *const u8, pos: &mut usize, state: &mut DtbParseState) {
    let name_ptr = struct_base.add(*pos);
    let name_len = cstr_len(name_ptr);

    if state.depth == 1 && name_len >= 6 && is_memory_node(name_ptr, name_len) {
        state.in_memory_node = true;
        state.memory_depth = state.depth + 1;
    }

    state.depth += 1;
    // Advance past the null-terminated name, 4-byte aligned
    *pos += (name_len + 1 + 3) & !3;
}

/// Process an FDT_END_NODE token, updating parse state.
fn handle_end_node(state: &mut DtbParseState) {
    if state.in_memory_node && state.depth == state.memory_depth {
        state.in_memory_node = false;
    }
    state.depth = state.depth.saturating_sub(1);
}

/// Process an FDT_PROP token, updating cell sizes and extracting memory regions.
///
/// # Safety
/// `struct_base.add(pos)` must point to the property length field within a valid FDT
/// structure block. `strings_base` must point to the FDT strings block.
unsafe fn handle_prop(
    struct_base: *const u8,
    strings_base: *const u8,
    pos: &mut usize,
    state: &mut DtbParseState,
    map: &mut PhysMemoryMap,
) {
    let prop_len = read_be32(struct_base.add(*pos)) as usize;
    let name_off = read_be32(struct_base.add(*pos + 4)) as usize;
    *pos += 8;

    let prop_name = strings_base.add(name_off);
    let prop_data = struct_base.add(*pos);

    // Parse root-level #address-cells and #size-cells
    if state.depth == 1 && prop_len == 4 {
        if cstr_eq(prop_name, "#address-cells") {
            state.address_cells = read_be32(prop_data);
        } else if cstr_eq(prop_name, "#size-cells") {
            state.size_cells = read_be32(prop_data);
        }
    }

    // Parse "reg" property inside memory nodes
    if state.in_memory_node && cstr_eq(prop_name, "reg") {
        parse_reg_property(
            prop_data,
            prop_len,
            state.address_cells,
            state.size_cells,
            map,
        );
    }

    // Advance past property data, 4-byte aligned
    *pos += (prop_len + 3) & !3;
}

/// Parse a DTB "reg" property into memory regions.
///
/// # Safety
/// `data` must point to valid property data of `len` bytes.
unsafe fn parse_reg_property(
    data: *const u8,
    len: usize,
    address_cells: u32,
    size_cells: u32,
    map: &mut PhysMemoryMap,
) {
    let entry_size = ((address_cells + size_cells) * 4) as usize;
    if entry_size == 0 {
        return;
    }

    let mut off = 0;
    while off + entry_size <= len {
        let base_addr = if address_cells == 2 {
            read_be64(data.add(off)) as usize
        } else {
            read_be32(data.add(off)) as usize
        };

        let addr_bytes = (address_cells * 4) as usize;
        let region_size = if size_cells == 2 {
            read_be64(data.add(off + addr_bytes)) as usize
        } else {
            read_be32(data.add(off + addr_bytes)) as usize
        };

        if region_size > 0 {
            map.add_region(PhysAddr::new(base_addr), region_size, RegionKind::Usable);
        }

        off += entry_size;
    }
}

/// Parse a Flattened Device Tree (FDT) blob to extract `/memory` regions.
///
/// # Safety
/// `dtb_addr` must point to a valid FDT blob.
pub unsafe fn parse_dtb(dtb_addr: usize, map: &mut PhysMemoryMap) {
    let base = dtb_addr as *const u8;

    // Validate magic
    let magic = read_be32(base);
    if magic != FDT_MAGIC {
        return;
    }

    let total_size = read_be32(base.add(4)) as usize;
    let off_dt_struct = read_be32(base.add(8)) as usize;
    let off_dt_strings = read_be32(base.add(12)) as usize;
    let _version = read_be32(base.add(20));

    let struct_base = base.add(off_dt_struct);
    let strings_base = base.add(off_dt_strings);

    let mut pos: usize = 0;
    let struct_end = total_size - off_dt_struct;
    let mut state = DtbParseState::new();

    while pos < struct_end {
        let token = read_be32(struct_base.add(pos));
        pos += 4;

        match token {
            FDT_BEGIN_NODE => handle_begin_node(struct_base, &mut pos, &mut state),
            FDT_END_NODE => handle_end_node(&mut state),
            FDT_PROP => handle_prop(struct_base, strings_base, &mut pos, &mut state, map),
            FDT_NOP => {}
            FDT_END => break,
            _ => break,
        }
    }
}

#[cfg(test)]
mod tests {
    extern crate alloc;
    use super::*;
    use alloc::vec::Vec;

    #[test]
    fn test_empty_map() {
        let map = PhysMemoryMap::new();
        assert_eq!(map.count(), 0);
        assert_eq!(map.total_usable(), 0);
    }

    #[test]
    fn test_add_regions() {
        let mut map = PhysMemoryMap::new();
        assert!(map.add_region(PhysAddr::new(0x1000), 0x1000, RegionKind::Usable));
        assert!(map.add_region(PhysAddr::new(0x10_0000), 0x100_0000, RegionKind::Usable));
        assert!(map.add_region(PhysAddr::new(0xFEE0_0000), 0x1000, RegionKind::Reserved));
        assert_eq!(map.count(), 3);
        assert_eq!(map.total_usable(), 0x1000 + 0x100_0000);
    }

    #[test]
    fn test_exclude_range_full() {
        let mut map = PhysMemoryMap::new();
        map.add_region(PhysAddr::new(0x1000), 0x1000, RegionKind::Usable);
        map.exclude_range(PhysAddr::new(0x1000), 0x1000);
        assert_eq!(map.total_usable(), 0);
    }

    #[test]
    fn test_exclude_range_partial_start() {
        let mut map = PhysMemoryMap::new();
        map.add_region(PhysAddr::new(0x1000), 0x4000, RegionKind::Usable);
        // Exclude [0x1000, 0x2000)
        map.exclude_range(PhysAddr::new(0x1000), 0x1000);
        // Should leave [0x2000, 0x5000)
        let usable: Vec<&MemRegion> = map.usable_regions().collect();
        assert_eq!(usable.len(), 1);
        assert_eq!(usable[0].base.as_usize(), 0x2000);
        assert_eq!(usable[0].size, 0x3000);
    }

    #[test]
    fn test_exclude_range_split() {
        let mut map = PhysMemoryMap::new();
        map.add_region(PhysAddr::new(0x1000), 0x8000, RegionKind::Usable);
        // Exclude [0x3000, 0x5000) — splits into [0x1000, 0x3000) and [0x5000, 0x9000)
        map.exclude_range(PhysAddr::new(0x3000), 0x2000);
        let total = map.total_usable();
        assert_eq!(total, 0x2000 + 0x4000);
    }

    #[test]
    fn test_map_full() {
        let mut map = PhysMemoryMap::new();
        for i in 0..MAX_REGIONS {
            assert!(map.add_region(PhysAddr::new(i * 0x1000), 0x1000, RegionKind::Usable,));
        }
        // 65th should fail
        assert!(!map.add_region(PhysAddr::new(0xFFFF_0000), 0x1000, RegionKind::Usable));
    }
}
