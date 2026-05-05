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

/// Devicetree blob pointer captured during `efi_main`. Sub-PR 2d-real
/// (the next one) will pass this to `kernel_main` after
/// `ExitBootServices`. Stored as a `static mut` (rather than threading
/// it through the halt loop) so the eventual jump from `efi_main` →
/// `kernel_main` doesn't have to re-derive it.
///
/// # Safety
/// Only written from `efi_main`, only once, while still under UEFI's
/// boot-services environment (single-threaded, identity-mapped). After
/// `ExitBootServices` it becomes effectively immutable.
static mut DTB_PTR: *const c_void = core::ptr::null();

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
    image_handle: Handle,
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
        print(system_table, "[boot] calling ExitBootServices\r\n");
    }

    // Tear down UEFI. After this returns success we own the machine and
    // can no longer use any boot-services calls (including `con_out`).
    let ebs = unsafe { exit_boot_services(image_handle, system_table) };
    if !ebs.is_success() {
        // ExitBootServices failed. We're still under UEFI, so con_out
        // is still valid — print a diagnostic and return. UEFI will
        // either reload the image or re-prompt for a different boot
        // entry depending on its config.
        unsafe {
            print(system_table, "[boot] ExitBootServices FAILED, status=");
            print_hex64(system_table, ebs.0 as u64);
            print(system_table, "\r\n");
        }
        return ebs;
    }

    // ExitBootServices succeeded. UEFI services are torn down. Tail-
    // call `kernel_main(dtb)`, which runs through the standard boot
    // stages and never returns.
    //
    // Verified on Orin NX hardware (P3767-0000 + P3768-0000, JetPack
    // 6.2.1 / L4T R36.4.7): kernel_main runs to its idle loop without
    // exception. There is no observable serial output yet — see
    // `tegra234_uart` for the placeholder driver and the deferred
    // follow-up that lands real UART output (post-EBS access to both
    // UART_1 and the TCU mailbox is blocked by the SoC firewall /
    // page-table state when ExitBootServices is called from EL2).
    let kernel_main: extern "C" fn(u64) -> ! = crate::kernel_main;
    kernel_main(dtb as u64);
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
