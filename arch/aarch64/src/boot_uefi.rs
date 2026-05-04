// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UEFI boot path for Tegra234 (Jetson Orin family).
//!
//! Entered via `efi_main(image_handle, system_table) -> Status` from the
//! `aarch64-unknown-uefi` bin target (`smallaios-uefi`).
//!
//! This sub-PR (2d) adds the **first observable signal**: efi_main
//! prints a banner via UEFI's `SimpleTextOutputProtocol` (the
//! `con_out` slot of the system table) before halting. On Jetson Orin
//! the firmware routes `con_out` through the TCU mailbox, so the
//! banner reaches whatever serial console the user has attached to the
//! J-class carrier's UART header. This validates the .efi-load path
//! end-to-end (PE/COFF parse → image load → entry → DTB lookup →
//! console output) without yet requiring a post-`ExitBootServices`
//! kernel-side TCU UART driver.
//!
//! What's NOT here yet: `ExitBootServices`, the actual jump to
//! `kernel_main`, and the kernel-side TCU UART driver. Those land in
//! the sub-PR after this one. After ExitBootServices `con_out` is no
//! longer valid, so we need the kernel-side UART before we can keep
//! producing output. Splitting the milestones this way isolates "did
//! the .efi load and run?" from "does the post-handoff kernel reach
//! and drive the TCU?" — the failure modes are different and worth
//! catching independently.

#![cfg(feature = "tegra234")]

use crate::uefi::{ConfigurationTable, Handle, Status, SystemTable, EFI_DTB_TABLE_GUID};
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

/// UEFI image entry point. Called by the firmware after `LoadImage` +
/// `StartImage`. Spec signature: `EFI_STATUS efi_main(EFI_HANDLE,
/// EFI_SYSTEM_TABLE*)`.
///
/// # Safety
/// Called exactly once by UEFI firmware with `image_handle` and
/// `system_table` set per UEFI 2.10 §4.1. Until the sub-PR after this
/// wires in `ExitBootServices` and the kernel handoff, this function
/// never returns control to UEFI — it parks in a `wfi` loop after
/// printing the banner and capturing the DTB pointer.
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

    // Locate the Devicetree blob. The firmware on Jetson Orin's UEFI
    // exposes it via EFI_DTB_TABLE_GUID; we record it here so the
    // next sub-PR can pass it to kernel_main, and print it now so the
    // user can confirm the lookup worked over serial.
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
        print(
            system_table,
            "[boot] efi_main reached, halting (Phase 2 first-observable-signal milestone)\r\n",
        );
    }

    loop {
        unsafe {
            core::arch::asm!("wfi");
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
