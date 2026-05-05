// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UEFI boot path for Tegra234 (Jetson Orin family).
//!
//! Entered via `efi_main(image_handle, system_table) -> Status` from the
//! `aarch64-unknown-uefi` bin target (`smallaios-uefi`).
//!
//! Flow:
//! 1. Print banner via UEFI's `SimpleTextOutputProtocol` (`con_out`).
//! 2. Walk `system_table.configuration_table[]` to find the DTB via
//!    `EFI_DTB_TABLE_GUID` and capture it.
//! 3. Call `GetMemoryMap` + `ExitBootServices` (re-fetching the memory
//!    map if the first ExitBootServices fails with INVALID_PARAMETER —
//!    the spec-mandated retry shape).
//! 4. Jump to `kernel_main(dtb_addr)`. After this point UEFI's services
//!    are gone; output flows through the kernel-side `tegra234_uart`
//!    driver which writes to the TCU directly.
//!
//! Sub-PR 2d (the predecessor) handled steps 1-2 and stopped at a
//! `wfi` halt before ExitBootServices. This sub-PR (2e) adds steps 3
//! and 4 and the matching `tegra234_uart.rs` driver.

#![cfg(feature = "tegra234")]

use crate::uefi::{
    ConfigurationTable, Handle, MemoryDescriptor, Status, SystemTable, EFI_DTB_TABLE_GUID,
};
use core::ffi::c_void;

/// Devicetree blob pointer captured during `efi_main`.
///
/// # Safety
/// Only written from `efi_main`, only once, while still under UEFI's
/// boot-services environment (single-threaded, identity-mapped).
static mut DTB_PTR: *const c_void = core::ptr::null();

/// System-table pointer captured during `efi_main`. The kernel-side
/// UART driver (`crate::tegra234_uart`) reads this to reach
/// `con_out.output_string` for the **interim "kernel uses UEFI con_out
/// for output"** mode. This is set unconditionally; whether the kernel
/// actually uses it depends on whether `tegra234_uart::putc` is built
/// in DRAM-stash or con_out mode.
///
/// # Safety
/// Only written from `efi_main`, only once. Used after kernel_main is
/// called; the system table remains valid as long as we don't call
/// `ExitBootServices`. If we ever do call ExitBootServices and then
/// keep using this pointer, the read becomes UB.
pub(crate) static mut SYSTEM_TABLE: *const SystemTable = core::ptr::null();

/// Memory regions harvested from UEFI's `GetMemoryMap` and exposed
/// to the kernel via [`efi_memory_regions`]. NVIDIA's UEFI rewrites
/// the DTB on Tegra234 and does **not** put `/memory@…` nodes at
/// the standard root depth (the root only holds `/tegra-carveouts`
/// and `/bus@0`); the firmware-blessed source of truth for usable
/// RAM is the EFI memory map, not the FDT. This array is populated
/// during `efi_main` and read from `kernel_main`.
const EFI_MAX_REGIONS: usize = 64;
#[derive(Clone, Copy)]
pub struct EfiMemRegion {
    pub base: u64,
    pub size: u64,
}
static mut EFI_REGIONS: [EfiMemRegion; EFI_MAX_REGIONS] =
    [EfiMemRegion { base: 0, size: 0 }; EFI_MAX_REGIONS];
static mut EFI_REGION_COUNT: usize = 0;

/// `EfiConventionalMemory` (UEFI 2.10 §7.2 Table 7-5) — usable RAM
/// from the kernel's perspective. The other types (BootServicesCode,
/// RuntimeServicesData, ACPI, MemoryMapped IO, Reserved, etc.) are
/// either firmware-owned, MMIO, or reclaimable post-`ExitBootServices`
/// but not yet usable by us. We start by treating ConventionalMemory
/// as the only usable category.
const EFI_CONVENTIONAL_MEMORY: u32 = 7;

/// Walk the system table's configuration entries and return the
/// vendor-table pointer for `EFI_DTB_TABLE_GUID`, or null if not found.
///
/// # Safety
/// Caller must guarantee `system_table` is a valid UEFI system table
/// pointer with `configuration_table` and `number_of_table_entries`
/// set as the firmware provided them. We trust the firmware here —
/// there is no realistic way for the unikernel to validate beyond the
/// header signature, which we don't bother checking.
unsafe fn find_dtb(system_table: *const SystemTable) -> *const c_void {
    let st = unsafe { &*system_table };
    let count = st.number_of_table_entries;
    let table = st.configuration_table;
    for i in 0..count {
        let entry: *const ConfigurationTable = unsafe { table.add(i) };
        let entry = unsafe { &*entry };
        if entry.vendor_guid == EFI_DTB_TABLE_GUID {
            return entry.vendor_table;
        }
    }
    core::ptr::null()
}

/// Maximum length of a single `print` call's message, including the
/// trailing NUL. UEFI strings are UCS-2 (UTF-16), so a stack buffer of
/// `[u16; CON_OUT_BUF]` is twice this many bytes. 256 is plenty for
/// the banner + a hex address; if a future caller wants more they can
/// chunk into multiple `print` calls.
const CON_OUT_BUF: usize = 256;

/// Print a 7-bit ASCII string via UEFI's `con_out`. Converts each byte
/// to a UCS-2 code point (one-to-one for ASCII) and appends a NUL,
/// then calls `output_string`. Caller is responsible for any `\r\n`
/// the terminal expects — UEFI converts neither.
///
/// # Safety
/// `system_table` must be a live, well-formed system table pointer
/// (the same one efi_main was called with). `s` must be ASCII; non-ASCII
/// bytes get widened naively, which produces nonsense for the high
/// half but doesn't violate memory safety.
unsafe fn print(system_table: *const SystemTable, s: &str) {
    let st = unsafe { &*system_table };
    let con_out = st.con_out;
    if con_out.is_null() {
        return;
    }
    let mut buf = [0u16; CON_OUT_BUF];
    let mut i = 0;
    for byte in s.bytes() {
        if i + 1 >= CON_OUT_BUF {
            break; // Reserve last slot for NUL.
        }
        buf[i] = byte as u16;
        i += 1;
    }
    buf[i] = 0;
    let f = unsafe { (*con_out).output_string };
    let _ = unsafe { f(con_out, buf.as_ptr()) };
}

/// Print a non-negative integer as decimal via UEFI's `con_out`. Used
/// for short counts (region counts, attempt indices) where the
/// 16-digit zero-padded hex output of `print_hex64` is more noise
/// than signal.
///
/// # Safety
/// Same preconditions as `print`.
unsafe fn print_dec(system_table: *const SystemTable, val: u64) {
    if val == 0 {
        unsafe { print(system_table, "0") };
        return;
    }
    let mut buf = [0u8; 20];
    let mut pos = 0usize;
    let mut v = val;
    while v > 0 {
        buf[pos] = b'0' + (v % 10) as u8;
        v /= 10;
        pos += 1;
    }
    let mut out = [0u8; 20];
    for i in 0..pos {
        out[i] = buf[pos - 1 - i];
    }
    // SAFETY: out is all ASCII '0'..'9'.
    let s = unsafe { core::str::from_utf8_unchecked(&out[..pos]) };
    unsafe { print(system_table, s) };
}

/// Print a 64-bit value as `0x` + 16 hex digits (uppercase, zero-padded)
/// via UEFI's `con_out`. Mirrors `uart::put_hex` shape. Splits the
/// output into one call per digit which is wasteful — UEFI would
/// happily take a 19-character buffer — but keeps this auxiliary tool
/// simple and dependency-free.
///
/// # Safety
/// Same preconditions as `print`.
unsafe fn print_hex64(system_table: *const SystemTable, val: u64) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    let mut buf = [0u8; 18];
    buf[0] = b'0';
    buf[1] = b'x';
    for i in 0..16 {
        let nibble = ((val >> ((15 - i) * 4)) & 0xF) as usize;
        buf[2 + i] = HEX[nibble];
    }
    // SAFETY: buf is all ASCII bytes (`0`, `x`, `0-9A-F`).
    let s = unsafe { core::str::from_utf8_unchecked(&buf) };
    unsafe { print(system_table, s) };
}

/// Static buffer for the UEFI memory map. UEFI hands you a copy of
/// the in-memory map of every physical region; for Jetson Orin that's
/// typically a few hundred descriptors of 48 bytes each, so 32 KiB is
/// comfortably enough headroom. The buffer must be at least 8-byte
/// aligned because `MemoryDescriptor` contains `u64` fields — wrapping
/// it in `repr(align(8))` guarantees that even though the underlying
/// element type is `u8`. If the firmware reports a larger map
/// `GetMemoryMap` fails with `BUFFER_TOO_SMALL` and we surface the
/// status code on serial.
const MEMORY_MAP_BUF_SIZE: usize = 32 * 1024;
#[repr(C, align(8))]
struct MemoryMapBuf([u8; MEMORY_MAP_BUF_SIZE]);
static mut MEMORY_MAP_BUF: MemoryMapBuf = MemoryMapBuf([0; MEMORY_MAP_BUF_SIZE]);

/// `EFI_INVALID_PARAMETER` — the canonical "memory map changed under
/// you, retry" signal from `ExitBootServices`. Other status codes
/// (`BUFFER_TOO_SMALL`, etc.) are reported by `GetMemoryMap` directly
/// and surface verbatim to the caller, so we don't enumerate them here.
const EFI_INVALID_PARAMETER: usize = 0x8000_0000_0000_0002;

/// Call `GetMemoryMap` then `ExitBootServices`, retrying once if the
/// first `ExitBootServices` fails with `INVALID_PARAMETER` (which means
/// the memory map changed between the two calls). Returns the final
/// `Status` from `ExitBootServices` — `Status::SUCCESS` means UEFI
/// boot services have torn down and the caller now owns the machine.
///
/// Emits diagnostic breadcrumbs via `con_out` between each call so we
/// can see exactly which firmware call faults if anything goes wrong.
///
/// # Safety
/// Caller must guarantee the system table pointer + image handle are
/// the ones UEFI handed `efi_main`. After this returns success the
/// system table's `boot_services` pointer is invalid; callers must
/// not dereference it.
unsafe fn exit_boot_services(image_handle: Handle, system_table: *const SystemTable) -> Status {
    let st = unsafe { &*system_table };
    let bs = unsafe { &*st.boot_services };

    // The UEFI ExitBootServices retry pattern: first try, if it returns
    // INVALID_PARAMETER, refresh the memory map and try once more.
    for attempt in 0..2 {
        unsafe {
            print(system_table, "[boot]   attempt ");
            print_hex64(system_table, attempt as u64);
            print(system_table, " — calling GetMemoryMap\r\n");
        }
        let mut map_size: usize = MEMORY_MAP_BUF_SIZE;
        let mut map_key: usize = 0;
        let mut desc_size: usize = 0;
        let mut desc_version: u32 = 0;
        // SAFETY: MEMORY_MAP_BUF is a static mut owned by us; we serialize
        // calls (single-threaded, called only from efi_main). Use the
        // raw-pointer form so we don't create an aliasing reference to
        // the static (Rust 2024 `static_mut_refs` lint). The
        // `MemoryMapBuf` wrapper guarantees 8-byte alignment so the
        // `*mut MemoryDescriptor` cast doesn't violate alignment.
        let buf_ptr = (&raw mut MEMORY_MAP_BUF) as *mut u8 as *mut MemoryDescriptor;
        let gmm_status = unsafe {
            (bs.get_memory_map)(
                &mut map_size,
                buf_ptr,
                &mut map_key,
                &mut desc_size,
                &mut desc_version,
            )
        };
        unsafe {
            print(system_table, "[boot]   GetMemoryMap returned 0x");
            print_hex64(system_table, gmm_status.0 as u64);
            print(system_table, ", map_size=");
            print_hex64(system_table, map_size as u64);
            print(system_table, ", desc_size=");
            print_hex64(system_table, desc_size as u64);
            print(system_table, ", map_key=");
            print_hex64(system_table, map_key as u64);
            print(system_table, "\r\n");
        }
        if !gmm_status.is_success() {
            // Buffer too small or some other issue — bail. Caller will
            // see the non-success Status and halt with a breadcrumb.
            return gmm_status;
        }

        unsafe {
            print(system_table, "[boot]   calling ExitBootServices(map_key=");
            print_hex64(system_table, map_key as u64);
            print(system_table, ")\r\n");
        }
        let ebs_status = unsafe { (bs.exit_boot_services)(image_handle, map_key) };
        // Note: if ebs succeeded, `con_out` is now invalid — printing
        // anything else here would be undefined behavior. We only print
        // on the *failure* path, before returning.
        if ebs_status.is_success() {
            return ebs_status;
        }
        unsafe {
            print(system_table, "[boot]   ExitBootServices returned 0x");
            print_hex64(system_table, ebs_status.0 as u64);
            print(system_table, "\r\n");
        }
        if ebs_status.0 != EFI_INVALID_PARAMETER {
            // Some other failure — give up.
            return ebs_status;
        }
        // INVALID_PARAMETER → memory map changed; loop and retry once.
    }
    // Two attempts wasn't enough — return the last error code shape.
    Status(EFI_INVALID_PARAMETER)
}

/// UEFI image entry point. Called by the firmware after `LoadImage` +
/// `StartImage`. Spec signature: `EFI_STATUS efi_main(EFI_HANDLE,
/// EFI_SYSTEM_TABLE*)`.
///
/// Returns only on the failure paths (so UEFI can free the image and
/// re-prompt for a boot device); the success path tail-calls
/// `kernel_main` which never returns.
///
/// # Safety
/// Called exactly once by UEFI firmware with `image_handle` and
/// `system_table` set per UEFI 2.10 §4.1.
#[no_mangle]
pub unsafe extern "efiapi" fn efi_main(
    _image_handle: Handle,
    system_table: *const SystemTable,
) -> Status {
    unsafe {
        print(
            system_table,
            "\r\n========================================\r\n",
        );
        print(
            system_table,
            "  Hello, world from SmallAIOS on Tegra234\r\n",
        );
        print(system_table, "========================================\r\n");
    }

    // Locate the Devicetree blob.
    let dtb = unsafe { find_dtb(system_table) };
    unsafe {
        DTB_PTR = dtb;
        SYSTEM_TABLE = system_table;
        if dtb.is_null() {
            print(
                system_table,
                "[boot] DTB lookup via EFI_DTB_TABLE_GUID: not found\r\n",
            );
        } else {
            print(system_table, "[boot] DTB at ");
            print_hex64(system_table, dtb as u64);
            print(system_table, "\r\n");
        }
    }

    // Harvest UEFI's memory map. NVIDIA's UEFI on Tegra234 doesn't
    // expose `/memory@…` nodes in the DTB — only `/tegra-carveouts`
    // and `/bus@0`. The EFI memory map is the firmware-blessed source
    // of truth for usable RAM. The kernel reads
    // [`efi_memory_region_count`] / [`efi_memory_region`] to enumerate.
    unsafe {
        let h = harvest_efi_memory_map(system_table);
        if !h.is_success() {
            print(system_table, "[boot] GetMemoryMap failed, status=");
            print_hex64(system_table, h.0 as u64);
            print(system_table, "\r\n");
        } else {
            print(system_table, "[boot] EFI memory map: ");
            print_dec(system_table, EFI_REGION_COUNT as u64);
            print(system_table, " conventional region(s)\r\n");
        }
    }

    // ── INTERIM: skip ExitBootServices ──────────────────────────────
    //
    // Sub-PR 2e demonstrated ExitBootServices works (silent halt
    // confirmed). But the kernel-side UART path post-EBS is firewalled
    // on Tegra234 and we don't yet have a working post-EBS output
    // mechanism (see `tegra234_uart.rs` for the failed attempts and
    // the candidate follow-up paths).
    //
    // Until we sort that out, we **don't** call ExitBootServices here.
    // We tail-call `kernel_main` while UEFI's services (specifically
    // `con_out`) are still alive, so `tegra234_uart::putc` can route
    // through `SYSTEM_TABLE.con_out` and produce visible output on the
    // J-class carrier's TTL header.
    //
    // This is **not a real kernel boot** — UEFI services persist and
    // we're effectively a fancy UEFI app. But it lets us validate the
    // rest of the kernel logic (memory detection, heap allocator,
    // banner, etc.) on real hardware and gives us a visible
    // proof-of-life that's a real kernel-side print, not a UEFI
    // bootloader print.
    //
    // The follow-up sub-PR will reinstate the ExitBootServices call
    // alongside a working post-EBS UART driver.
    unsafe {
        print(
            system_table,
            "[boot] skipping ExitBootServices (interim mode); calling kernel_main\r\n",
        );
    }
    let kernel_main: extern "C" fn(u64) -> ! = crate::kernel_main;
    kernel_main(dtb as u64);
}

/// Silence dead-code warning for the still-correct `exit_boot_services`
/// helper. The interim mode above doesn't call it; the follow-up
/// sub-PR will reinstate the real handoff.
#[allow(dead_code)]
fn _keep_exit_boot_services_alive() {
    let _ = exit_boot_services;
}

/// Walk UEFI's memory map and record `EfiConventionalMemory` regions
/// into [`EFI_REGIONS`]. Called from `efi_main` before `kernel_main`,
/// while UEFI's boot services (and the memory map) are still valid.
/// On Tegra234 this is the firmware-blessed source of truth for
/// usable RAM (DTB `/memory@…` nodes are absent on this firmware);
/// the kernel reads from [`efi_memory_regions`] in lieu of FDT
/// memory parsing.
///
/// # Safety
/// `system_table` must be a valid UEFI system table pointer with
/// boot services still alive (i.e. before `ExitBootServices`). The
/// caller is responsible for serializing access to the static
/// `MEMORY_MAP_BUF` and `EFI_REGIONS` arrays — this is single-
/// threaded boot code so that's a non-issue.
unsafe fn harvest_efi_memory_map(system_table: *const SystemTable) -> Status {
    let st = unsafe { &*system_table };
    let bs = unsafe { &*st.boot_services };

    let mut map_size: usize = MEMORY_MAP_BUF_SIZE;
    let mut map_key: usize = 0;
    let mut desc_size: usize = 0;
    let mut desc_version: u32 = 0;
    let buf_ptr = (&raw mut MEMORY_MAP_BUF) as *mut u8 as *mut MemoryDescriptor;
    let status = unsafe {
        (bs.get_memory_map)(
            &mut map_size,
            buf_ptr,
            &mut map_key,
            &mut desc_size,
            &mut desc_version,
        )
    };
    if !status.is_success() || desc_size == 0 {
        return status;
    }

    let count = map_size / desc_size;
    let mut written: usize = 0;
    let buf_bytes = (&raw mut MEMORY_MAP_BUF) as *mut u8;
    for i in 0..count {
        // Walk descriptors using `desc_size` as the stride — UEFI may
        // return descriptors larger than `MemoryDescriptor` (extra
        // implementation-defined fields). Reading via the smaller
        // struct is fine because the spec-defined prefix matches.
        let desc_ptr = unsafe { buf_bytes.add(i * desc_size) } as *const MemoryDescriptor;
        let desc = unsafe { &*desc_ptr };
        if desc.typ == EFI_CONVENTIONAL_MEMORY && desc.number_of_pages > 0 {
            if written < EFI_MAX_REGIONS {
                let size = desc.number_of_pages * 4096;
                unsafe {
                    EFI_REGIONS[written] = EfiMemRegion {
                        base: desc.physical_start,
                        size,
                    };
                }
                written += 1;
            }
        }
    }
    unsafe {
        EFI_REGION_COUNT = written;
    }
    Status::SUCCESS
}

/// Iterator-style access to the harvested EFI memory regions. Returns
/// the number of regions populated; pass an index < count to
/// [`efi_memory_region`] to read each entry. Callable from anywhere
/// in the lib once `efi_main` has run.
pub fn efi_memory_region_count() -> usize {
    unsafe { EFI_REGION_COUNT }
}

/// Read one harvested EFI memory region. Returns `None` if `index >=
/// efi_memory_region_count()`.
pub fn efi_memory_region(index: usize) -> Option<EfiMemRegion> {
    unsafe {
        if index < EFI_REGION_COUNT {
            Some(EFI_REGIONS[index])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compile-test only: confirms the function pointer for `find_dtb`
    /// has the expected ABI and that the GUID comparison links cleanly.
    /// No runtime side effect — UEFI types can't be constructed in a
    /// host test environment without violating safety preconditions.
    #[test]
    fn find_dtb_signature_compiles() {
        let _: unsafe fn(*const SystemTable) -> *const c_void = find_dtb;
    }
}
