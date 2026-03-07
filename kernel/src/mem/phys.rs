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
    let total_size = core::ptr::read_unaligned(mb_info_addr as *const u32);
    let mut offset: usize = 8; // Skip total_size and reserved fields

    while offset < total_size as usize {
        let tag_ptr = (mb_info_addr + offset) as *const u32;
        let tag_type = core::ptr::read_unaligned(tag_ptr);
        let tag_size = core::ptr::read_unaligned(tag_ptr.add(1));

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
    let entry_size = core::ptr::read_unaligned((tag_addr + 8) as *const u32) as usize;
    if entry_size == 0 {
        return;
    }

    let entries_start = tag_addr + 16;
    let entries_end = tag_addr + tag_size;
    let mut entry_addr = entries_start;

    while entry_addr + entry_size <= entries_end {
        // Entry: base_addr(8) + length(8) + type(4) + reserved(4)
        let base = core::ptr::read_unaligned(entry_addr as *const u64) as usize;
        let length = core::ptr::read_unaligned((entry_addr + 8) as *const u64) as usize;
        let mem_type = core::ptr::read_unaligned((entry_addr + 16) as *const u32);

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

    #[test]
    fn test_default_map() {
        let map = PhysMemoryMap::default();
        assert_eq!(map.count(), 0);
        assert_eq!(map.total_usable(), 0);
        assert!(map.get(0).is_none());
    }

    #[test]
    fn test_get_out_of_bounds() {
        let map = PhysMemoryMap::new();
        assert!(map.get(0).is_none());
        assert!(map.get(1).is_none());
        assert!(map.get(100).is_none());

        let mut map2 = PhysMemoryMap::new();
        map2.add_region(PhysAddr::new(0x1000), 0x1000, RegionKind::Usable);
        assert!(map2.get(0).is_some());
        assert!(map2.get(1).is_none());
    }

    #[test]
    fn test_region_kind_debug() {
        // Exercise Debug formatting on all RegionKind variants
        let kinds = [
            RegionKind::Usable,
            RegionKind::Reserved,
            RegionKind::AcpiReclaimable,
            RegionKind::Kernel,
        ];
        let expected = ["Usable", "Reserved", "AcpiReclaimable", "Kernel"];
        for (kind, exp) in kinds.iter().zip(expected.iter()) {
            let dbg = alloc::format!("{:?}", kind);
            assert_eq!(dbg, *exp);
        }

        // Test Clone and PartialEq
        let a = RegionKind::Usable;
        let b = a;
        assert_eq!(a, b);
        assert_ne!(RegionKind::Usable, RegionKind::Reserved);

        // Test MemRegion Debug
        let region = MemRegion {
            base: PhysAddr(0x1000),
            size: 4096,
            kind: RegionKind::Usable,
        };
        let dbg = alloc::format!("{:?}", region);
        assert!(dbg.contains("Usable"));
    }

    #[test]
    fn test_exclude_range_no_overlap() {
        let mut map = PhysMemoryMap::new();
        map.add_region(PhysAddr::new(0x1000), 0x1000, RegionKind::Usable);
        // Exclude range is completely outside the region
        map.exclude_range(PhysAddr::new(0x8000), 0x1000);
        assert_eq!(map.total_usable(), 0x1000);
        // Exclude below the region
        map.exclude_range(PhysAddr::new(0x0), 0x500);
        assert_eq!(map.total_usable(), 0x1000);
    }

    #[test]
    fn test_exclude_range_partial_end() {
        let mut map = PhysMemoryMap::new();
        // Region: [0x1000, 0x5000)
        map.add_region(PhysAddr::new(0x1000), 0x4000, RegionKind::Usable);
        // Exclude [0x4000, 0x6000) — overlaps the end of the region
        map.exclude_range(PhysAddr::new(0x4000), 0x2000);
        // Should leave [0x1000, 0x4000) = size 0x3000
        let usable: Vec<&MemRegion> = map.usable_regions().collect();
        assert_eq!(usable.len(), 1);
        assert_eq!(usable[0].base.as_usize(), 0x1000);
        assert_eq!(usable[0].size, 0x3000);
    }

    // ── FDT builder helpers ──────────────────────────────────────────────

    fn push_be32(buf: &mut Vec<u8>, val: u32) {
        buf.extend_from_slice(&val.to_be_bytes());
    }

    fn push_be64(buf: &mut Vec<u8>, val: u64) {
        buf.extend_from_slice(&val.to_be_bytes());
    }

    /// Pad buffer to 4-byte alignment.
    fn pad4(buf: &mut Vec<u8>) {
        while buf.len() % 4 != 0 {
            buf.push(0);
        }
    }

    /// Build a minimal valid FDT blob with one memory node.
    ///
    /// `addr_cells` and `size_cells` control the root cell sizes.
    /// `regions` is a list of (base_addr, region_size) pairs to encode in
    /// the `reg` property.
    fn build_fdt(addr_cells: u32, size_cells: u32, regions: &[(u64, u64)]) -> Vec<u8> {
        // ── Strings block ────────────────────────────────────────────
        let mut strings = Vec::new();
        let off_address_cells = strings.len() as u32;
        strings.extend_from_slice(b"#address-cells\0");
        let off_size_cells = strings.len() as u32;
        strings.extend_from_slice(b"#size-cells\0");
        let off_reg = strings.len() as u32;
        strings.extend_from_slice(b"reg\0");
        let off_device_type = strings.len() as u32;
        strings.extend_from_slice(b"device_type\0");
        // Pad strings to 4 bytes
        while strings.len() % 4 != 0 {
            strings.push(0);
        }

        // ── Structure block ──────────────────────────────────────────
        let mut structs = Vec::new();

        // Root node: FDT_BEGIN_NODE + name "\0" (empty root name)
        push_be32(&mut structs, FDT_BEGIN_NODE);
        structs.push(0); // empty root name, null-terminated
        pad4(&mut structs);

        // Root property: #address-cells
        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, 4); // len
        push_be32(&mut structs, off_address_cells); // nameoff
        push_be32(&mut structs, addr_cells); // value

        // Root property: #size-cells
        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, 4); // len
        push_be32(&mut structs, off_size_cells); // nameoff
        push_be32(&mut structs, size_cells); // value

        // Memory node: FDT_BEGIN_NODE + "memory@80000000\0"
        push_be32(&mut structs, FDT_BEGIN_NODE);
        structs.extend_from_slice(b"memory@80000000\0");
        pad4(&mut structs);

        // Memory node property: device_type = "memory\0"
        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, 7); // len = "memory" + null
        push_be32(&mut structs, off_device_type);
        structs.extend_from_slice(b"memory\0");
        pad4(&mut structs);

        // Memory node property: reg
        let entry_bytes = ((addr_cells + size_cells) * 4) as usize;
        let reg_len = entry_bytes * regions.len();
        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, reg_len as u32);
        push_be32(&mut structs, off_reg);
        for &(base, size) in regions {
            if addr_cells == 2 {
                push_be64(&mut structs, base);
            } else {
                push_be32(&mut structs, base as u32);
            }
            if size_cells == 2 {
                push_be64(&mut structs, size);
            } else {
                push_be32(&mut structs, size as u32);
            }
        }
        pad4(&mut structs);

        // End memory node
        push_be32(&mut structs, FDT_END_NODE);

        // End root node
        push_be32(&mut structs, FDT_END_NODE);

        // FDT_END token
        push_be32(&mut structs, FDT_END);

        // ── Header (40 bytes) ────────────────────────────────────────
        let off_dt_struct: u32 = 40; // right after the header
                                     // Memory reservation map: 16 bytes of zeros (one empty entry)
        let off_mem_rsvmap: u32 = 40 + structs.len() as u32;
        let mem_rsvmap_size: u32 = 16;
        let off_dt_strings: u32 = off_mem_rsvmap + mem_rsvmap_size;
        let total_size: u32 = off_dt_strings + strings.len() as u32;

        let mut blob = Vec::new();
        push_be32(&mut blob, FDT_MAGIC); // magic
        push_be32(&mut blob, total_size); // totalsize
        push_be32(&mut blob, off_dt_struct); // off_dt_struct
        push_be32(&mut blob, off_dt_strings); // off_dt_strings
        push_be32(&mut blob, off_mem_rsvmap); // off_mem_rsvmap
        push_be32(&mut blob, 17); // version
        push_be32(&mut blob, 16); // last_comp_version
        push_be32(&mut blob, 0); // boot_cpuid_phys
        push_be32(&mut blob, strings.len() as u32); // size_dt_strings
        push_be32(&mut blob, structs.len() as u32); // size_dt_struct

        // Structure block
        blob.extend_from_slice(&structs);

        // Memory reservation map (one empty entry = 16 zero bytes)
        blob.extend_from_slice(&[0u8; 16]);

        // Strings block
        blob.extend_from_slice(&strings);

        blob
    }

    #[test]
    fn test_parse_dtb_minimal() {
        // One memory region at 0x80000000, size 4 GiB (0x1_0000_0000)
        let fdt = build_fdt(2, 2, &[(0x8000_0000, 0x1_0000_0000)]);
        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_dtb(fdt.as_ptr() as usize, &mut map);
        }
        assert_eq!(map.count(), 1);
        let r = map.get(0).unwrap();
        assert_eq!(r.base.as_usize(), 0x8000_0000);
        assert_eq!(r.size, 0x1_0000_0000);
        assert_eq!(r.kind, RegionKind::Usable);
    }

    #[test]
    fn test_parse_dtb_bad_magic() {
        // A buffer that doesn't start with FDT_MAGIC
        let mut buf = Vec::new();
        push_be32(&mut buf, 0xDEAD_BEEF);
        // Pad enough so we don't access out of bounds
        buf.resize(64, 0);

        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_dtb(buf.as_ptr() as usize, &mut map);
        }
        // Should not have parsed any regions
        assert_eq!(map.count(), 0);
    }

    #[test]
    fn test_parse_dtb_32bit_cells() {
        // address_cells=1, size_cells=1, region at 0x2000_0000, size 256 MiB
        let fdt = build_fdt(1, 1, &[(0x2000_0000, 0x1000_0000)]);
        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_dtb(fdt.as_ptr() as usize, &mut map);
        }
        assert_eq!(map.count(), 1);
        let r = map.get(0).unwrap();
        assert_eq!(r.base.as_usize(), 0x2000_0000);
        assert_eq!(r.size, 0x1000_0000);
        assert_eq!(r.kind, RegionKind::Usable);
    }

    #[test]
    fn test_parse_dtb_multiple_regions() {
        let fdt = build_fdt(
            2,
            2,
            &[(0x8000_0000, 0x4000_0000), (0x1_0000_0000, 0x8000_0000)],
        );
        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_dtb(fdt.as_ptr() as usize, &mut map);
        }
        assert_eq!(map.count(), 2);
        let r0 = map.get(0).unwrap();
        assert_eq!(r0.base.as_usize(), 0x8000_0000);
        assert_eq!(r0.size, 0x4000_0000);
        let r1 = map.get(1).unwrap();
        assert_eq!(r1.base.as_usize(), 0x1_0000_0000);
        assert_eq!(r1.size, 0x8000_0000);
    }

    #[test]
    fn test_parse_multiboot2_minimal() {
        // Build a Multiboot2 info structure with a memory map tag.
        //
        // Layout:
        //   Header: total_size(4) + reserved(4)
        //   Memory map tag: type=6(4) + size(4) + entry_size=24(4) + entry_version=0(4) + entries
        //   Each entry: base_addr(8) + length(8) + type(4) + reserved(4) = 24 bytes
        //   End tag: type=0(4) + size=8(4)
        let mut buf: Vec<u8> = Vec::new();

        // Placeholder for total_size — fill in after building
        let total_size_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // total_size placeholder
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Memory map tag header
        let tag_start = buf.len();
        buf.extend_from_slice(&6u32.to_le_bytes()); // type = MB2_TAG_MEMORY_MAP
        let tag_size_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // tag size placeholder
        buf.extend_from_slice(&24u32.to_le_bytes()); // entry_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // entry_version

        // Entry 1: usable RAM at 0x0010_0000, length 0x0FF0_0000 (type=1)
        buf.extend_from_slice(&0x0010_0000u64.to_le_bytes()); // base_addr
        buf.extend_from_slice(&0x0FF0_0000u64.to_le_bytes()); // length
        buf.extend_from_slice(&1u32.to_le_bytes()); // type = available
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Entry 2: reserved at 0xFEE0_0000, length 0x1000 (type=2)
        buf.extend_from_slice(&0xFEE0_0000u64.to_le_bytes()); // base_addr
        buf.extend_from_slice(&0x1000u64.to_le_bytes()); // length
        buf.extend_from_slice(&2u32.to_le_bytes()); // type = reserved
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Entry 3: ACPI reclaimable at 0xBFFE_0000, length 0x2_0000 (type=3)
        buf.extend_from_slice(&0xBFFE_0000u64.to_le_bytes()); // base_addr
        buf.extend_from_slice(&0x2_0000u64.to_le_bytes()); // length
        buf.extend_from_slice(&3u32.to_le_bytes()); // type = ACPI reclaimable
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Patch tag size
        let tag_size = (buf.len() - tag_start) as u32;
        buf[tag_size_offset..tag_size_offset + 4].copy_from_slice(&tag_size.to_le_bytes());

        // Pad to 8-byte alignment before end tag
        while buf.len() % 8 != 0 {
            buf.push(0);
        }

        // End tag: type=0, size=8
        buf.extend_from_slice(&0u32.to_le_bytes()); // type = end
        buf.extend_from_slice(&8u32.to_le_bytes()); // size

        // Patch total_size
        let total_size = buf.len() as u32;
        buf[total_size_offset..total_size_offset + 4].copy_from_slice(&total_size.to_le_bytes());

        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_multiboot2(buf.as_ptr() as usize, &mut map);
        }

        assert_eq!(map.count(), 3);

        // Entry 1: usable
        let r0 = map.get(0).unwrap();
        assert_eq!(r0.base.as_usize(), 0x0010_0000);
        assert_eq!(r0.size, 0x0FF0_0000);
        assert_eq!(r0.kind, RegionKind::Usable);

        // Entry 2: reserved
        let r1 = map.get(1).unwrap();
        assert_eq!(r1.base.as_usize(), 0xFEE0_0000);
        assert_eq!(r1.size, 0x1000);
        assert_eq!(r1.kind, RegionKind::Reserved);

        // Entry 3: ACPI reclaimable
        let r2 = map.get(2).unwrap();
        assert_eq!(r2.base.as_usize(), 0xBFFE_0000);
        assert_eq!(r2.size, 0x2_0000);
        assert_eq!(r2.kind, RegionKind::AcpiReclaimable);
    }

    #[test]
    fn test_helper_read_be() {
        // Test read_be32
        let data32: [u8; 4] = [0x12, 0x34, 0x56, 0x78];
        let val32 = unsafe { read_be32(data32.as_ptr()) };
        assert_eq!(val32, 0x1234_5678);

        // Test read_be64
        let data64: [u8; 8] = [0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08];
        let val64 = unsafe { read_be64(data64.as_ptr()) };
        assert_eq!(val64, 0x0102_0304_0506_0708);

        // Zero values
        let zero32: [u8; 4] = [0, 0, 0, 0];
        assert_eq!(unsafe { read_be32(zero32.as_ptr()) }, 0);

        let zero64: [u8; 8] = [0; 8];
        assert_eq!(unsafe { read_be64(zero64.as_ptr()) }, 0);

        // Max values
        let max32: [u8; 4] = [0xFF, 0xFF, 0xFF, 0xFF];
        assert_eq!(unsafe { read_be32(max32.as_ptr()) }, u32::MAX);

        let max64: [u8; 8] = [0xFF; 8];
        assert_eq!(unsafe { read_be64(max64.as_ptr()) }, u64::MAX);
    }

    #[test]
    fn test_cstr_helpers() {
        // cstr_len: simple string
        let s = b"hello\0";
        assert_eq!(unsafe { cstr_len(s.as_ptr()) }, 5);

        // cstr_len: empty string
        let empty = b"\0";
        assert_eq!(unsafe { cstr_len(empty.as_ptr()) }, 0);

        // cstr_eq: matching string
        assert!(unsafe { cstr_eq(b"memory\0".as_ptr(), "memory") });

        // cstr_eq: non-matching string
        assert!(!unsafe { cstr_eq(b"memory\0".as_ptr(), "memor") });

        // cstr_eq: longer reference
        assert!(!unsafe { cstr_eq(b"mem\0".as_ptr(), "memory") });

        // cstr_eq: empty strings
        assert!(unsafe { cstr_eq(b"\0".as_ptr(), "") });

        // cstr_eq: prefix that doesn't match due to non-null continuation
        assert!(!unsafe { cstr_eq(b"memory@0\0".as_ptr(), "memory") });

        // cstr_len: longer string
        let long = b"#address-cells\0";
        assert_eq!(unsafe { cstr_len(long.as_ptr()) }, 14);
    }

    #[test]
    fn test_is_memory_node() {
        // Exact "memory" (length 6)
        assert!(unsafe { is_memory_node(b"memory\0".as_ptr(), 6) });

        // "memory@80000000" (length 15)
        assert!(unsafe { is_memory_node(b"memory@80000000\0".as_ptr(), 15) });

        // "memory@0" (length 8)
        assert!(unsafe { is_memory_node(b"memory@0\0".as_ptr(), 8) });

        // Not a memory node: too short
        assert!(!unsafe { is_memory_node(b"mem\0".as_ptr(), 3) });

        // Not a memory node: "memoryX" without '@'
        assert!(!unsafe { is_memory_node(b"memoryX\0".as_ptr(), 7) });

        // Not a memory node: exactly length 7 with '@' — name_len == 7, which is not > 7
        assert!(!unsafe { is_memory_node(b"memory@\0".as_ptr(), 7) });

        // Not a memory node: completely different name
        assert!(!unsafe { is_memory_node(b"chosen\0".as_ptr(), 6) });

        // Not a memory node: "memories" (length 8, but doesn't have '@' at position 6)
        assert!(!unsafe { is_memory_node(b"memories\0".as_ptr(), 8) });
    }

    #[test]
    fn test_handle_end_node_depth() {
        let mut state = DtbParseState::new();
        // Simulate: depth=0 end_node → stays 0 (saturating_sub)
        handle_end_node(&mut state);
        assert_eq!(state.depth, 0);
        assert!(!state.in_memory_node);

        // Simulate: in_memory_node at depth 2
        state.depth = 2;
        state.in_memory_node = true;
        state.memory_depth = 2;
        handle_end_node(&mut state);
        assert!(!state.in_memory_node);
        assert_eq!(state.depth, 1);

        // Simulate: in_memory_node but at wrong depth
        state.depth = 3;
        state.in_memory_node = true;
        state.memory_depth = 2;
        handle_end_node(&mut state);
        // Still in memory node because depth (3) != memory_depth (2)
        assert!(state.in_memory_node);
        assert_eq!(state.depth, 2);
    }

    #[test]
    fn test_exclude_range_skips_reserved() {
        let mut map = PhysMemoryMap::new();
        map.add_region(PhysAddr::new(0x1000), 0x4000, RegionKind::Reserved);
        map.add_region(PhysAddr::new(0x5000), 0x4000, RegionKind::Usable);
        // Exclude [0x1000, 0x9000) — covers both regions
        map.exclude_range(PhysAddr::new(0x1000), 0x8000);
        // Reserved region should be untouched
        let r0 = map.get(0).unwrap();
        assert_eq!(r0.kind, RegionKind::Reserved);
        assert_eq!(r0.size, 0x4000);
        // Usable region should be marked as kernel
        let r1 = map.get(1).unwrap();
        assert_eq!(r1.kind, RegionKind::Kernel);
    }

    #[test]
    fn test_parse_multiboot2_zero_entry_size() {
        // Multiboot2 with a memory map tag that has entry_size=0 → should not crash
        let mut buf: Vec<u8> = Vec::new();

        // Header
        let total_size_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // total_size placeholder
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        // Memory map tag with entry_size=0
        let tag_start = buf.len();
        buf.extend_from_slice(&6u32.to_le_bytes()); // type = MB2_TAG_MEMORY_MAP
        let tag_size_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // tag size placeholder
        buf.extend_from_slice(&0u32.to_le_bytes()); // entry_size = 0
        buf.extend_from_slice(&0u32.to_le_bytes()); // entry_version

        // Patch tag size
        let tag_size = (buf.len() - tag_start) as u32;
        buf[tag_size_offset..tag_size_offset + 4].copy_from_slice(&tag_size.to_le_bytes());

        // Pad to 8-byte alignment
        while buf.len() % 8 != 0 {
            buf.push(0);
        }

        // End tag
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());

        // Patch total_size
        let total_size = buf.len() as u32;
        buf[total_size_offset..total_size_offset + 4].copy_from_slice(&total_size.to_le_bytes());

        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_multiboot2(buf.as_ptr() as usize, &mut map);
        }
        assert_eq!(map.count(), 0);
    }

    #[test]
    fn test_parse_multiboot2_zero_length_entry() {
        // Multiboot2 with an entry whose length is 0 → should be skipped
        let mut buf: Vec<u8> = Vec::new();

        // Header
        let total_size_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]);
        buf.extend_from_slice(&0u32.to_le_bytes());

        // Memory map tag
        let tag_start = buf.len();
        buf.extend_from_slice(&6u32.to_le_bytes()); // type
        let tag_size_offset = buf.len();
        buf.extend_from_slice(&[0u8; 4]); // tag size placeholder
        buf.extend_from_slice(&24u32.to_le_bytes()); // entry_size
        buf.extend_from_slice(&0u32.to_le_bytes()); // entry_version

        // Entry with length=0
        buf.extend_from_slice(&0x1000u64.to_le_bytes()); // base_addr
        buf.extend_from_slice(&0u64.to_le_bytes()); // length = 0
        buf.extend_from_slice(&1u32.to_le_bytes()); // type = available
        buf.extend_from_slice(&0u32.to_le_bytes()); // reserved

        let tag_size = (buf.len() - tag_start) as u32;
        buf[tag_size_offset..tag_size_offset + 4].copy_from_slice(&tag_size.to_le_bytes());

        while buf.len() % 8 != 0 {
            buf.push(0);
        }

        // End tag
        buf.extend_from_slice(&0u32.to_le_bytes());
        buf.extend_from_slice(&8u32.to_le_bytes());

        let total_size = buf.len() as u32;
        buf[total_size_offset..total_size_offset + 4].copy_from_slice(&total_size.to_le_bytes());

        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_multiboot2(buf.as_ptr() as usize, &mut map);
        }
        // Zero-length entry should not be added
        assert_eq!(map.count(), 0);
    }

    #[test]
    fn test_iter_and_usable_regions() {
        let mut map = PhysMemoryMap::new();
        map.add_region(PhysAddr::new(0x1000), 0x1000, RegionKind::Usable);
        map.add_region(PhysAddr::new(0x2000), 0x1000, RegionKind::Reserved);
        map.add_region(PhysAddr::new(0x3000), 0x2000, RegionKind::Usable);
        map.add_region(PhysAddr::new(0x5000), 0x500, RegionKind::AcpiReclaimable);

        // iter should return all 4
        let all: Vec<&MemRegion> = map.iter().collect();
        assert_eq!(all.len(), 4);

        // usable_regions should return 2
        let usable: Vec<&MemRegion> = map.usable_regions().collect();
        assert_eq!(usable.len(), 2);
        assert_eq!(usable[0].base.as_usize(), 0x1000);
        assert_eq!(usable[1].base.as_usize(), 0x3000);

        // total_usable
        assert_eq!(map.total_usable(), 0x1000 + 0x2000);
    }

    #[test]
    fn test_parse_reg_property_zero_entry_size() {
        // If both address_cells and size_cells are 0, entry_size is 0 → early return
        let data: [u8; 8] = [0; 8];
        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_reg_property(data.as_ptr(), 8, 0, 0, &mut map);
        }
        assert_eq!(map.count(), 0);
    }

    #[test]
    fn test_parse_dtb_with_nop_tokens() {
        // Build an FDT that includes NOP tokens to ensure they're skipped.
        // We manually construct a blob similar to build_fdt but with FDT_NOP tokens.
        let mut strings = Vec::new();
        let off_address_cells = strings.len() as u32;
        strings.extend_from_slice(b"#address-cells\0");
        let off_size_cells = strings.len() as u32;
        strings.extend_from_slice(b"#size-cells\0");
        let off_reg = strings.len() as u32;
        strings.extend_from_slice(b"reg\0");
        while strings.len() % 4 != 0 {
            strings.push(0);
        }

        let mut structs = Vec::new();

        // NOP before root
        push_be32(&mut structs, FDT_NOP);

        // Root node
        push_be32(&mut structs, FDT_BEGIN_NODE);
        structs.push(0);
        pad4(&mut structs);

        // Root properties
        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, 4);
        push_be32(&mut structs, off_address_cells);
        push_be32(&mut structs, 2);

        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, 4);
        push_be32(&mut structs, off_size_cells);
        push_be32(&mut structs, 2);

        // NOP between properties
        push_be32(&mut structs, FDT_NOP);

        // Memory node
        push_be32(&mut structs, FDT_BEGIN_NODE);
        structs.extend_from_slice(b"memory\0");
        pad4(&mut structs);

        // reg property: 1 entry at 0x4000_0000, size 0x2000_0000
        push_be32(&mut structs, FDT_PROP);
        push_be32(&mut structs, 16); // 2*4 + 2*4
        push_be32(&mut structs, off_reg);
        push_be64(&mut structs, 0x4000_0000);
        push_be64(&mut structs, 0x2000_0000);

        push_be32(&mut structs, FDT_END_NODE);
        push_be32(&mut structs, FDT_END_NODE);
        push_be32(&mut structs, FDT_END);

        // Build header
        let off_dt_struct: u32 = 40;
        let off_mem_rsvmap: u32 = 40 + structs.len() as u32;
        let mem_rsvmap_size: u32 = 16;
        let off_dt_strings: u32 = off_mem_rsvmap + mem_rsvmap_size;
        let total_size: u32 = off_dt_strings + strings.len() as u32;

        let mut blob = Vec::new();
        push_be32(&mut blob, FDT_MAGIC);
        push_be32(&mut blob, total_size);
        push_be32(&mut blob, off_dt_struct);
        push_be32(&mut blob, off_dt_strings);
        push_be32(&mut blob, off_mem_rsvmap);
        push_be32(&mut blob, 17);
        push_be32(&mut blob, 16);
        push_be32(&mut blob, 0);
        push_be32(&mut blob, strings.len() as u32);
        push_be32(&mut blob, structs.len() as u32);
        blob.extend_from_slice(&structs);
        blob.extend_from_slice(&[0u8; 16]);
        blob.extend_from_slice(&strings);

        let mut map = PhysMemoryMap::new();
        unsafe {
            parse_dtb(blob.as_ptr() as usize, &mut map);
        }
        assert_eq!(map.count(), 1);
        let r = map.get(0).unwrap();
        assert_eq!(r.base.as_usize(), 0x4000_0000);
        assert_eq!(r.size, 0x2000_0000);
    }
}
