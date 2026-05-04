// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UEFI boot path for Tegra234 (Jetson Orin family).
//!
//! Entered via `efi_main(image_handle, system_table) -> Status` from the
//! `aarch64-unknown-uefi` bin target (`smallaios-uefi`). This sub-PR (2c)
//! lands the entry-point shape and a minimal configuration-table walk to
//! locate the Devicetree blob via `EFI_DTB_TABLE_GUID`. It deliberately
//! does **not** call `ExitBootServices` yet — that, plus the actual
//! handoff to `kernel_main`, lands in sub-PR 2d alongside the real TCU
//! UART driver (otherwise the kernel would exit boot services and then
//! immediately go silent on a board with no working serial).
//!
//! Until 2d lands, `efi_main` records what it found, then halts in a
//! `wfi` loop. The point of 2c is to (1) prove the .efi PE/COFF artifact
//! builds, (2) prove the configuration-table walk finds the DTB, and (3)
//! give 2d a stable surface to extend.

#![cfg(feature = "tegra234")]

use crate::uefi::{ConfigurationTable, Handle, Status, SystemTable, EFI_DTB_TABLE_GUID};
use core::ffi::c_void;

/// Devicetree blob pointer captured during `efi_main`. Sub-PR 2d will
/// pass this to `kernel_main` after `ExitBootServices`. Stored as a
/// `static mut` (rather than a parameter) so the eventual jump from
/// `efi_main` → `kernel_main` doesn't have to thread it through the
/// halt loop.
///
/// # Safety
/// Only written from `efi_main`, only once, while still under UEFI's
/// boot-services environment (single-threaded, identity-mapped). After
/// `ExitBootServices` (sub-PR 2d) it becomes effectively immutable.
static mut DTB_PTR: *const c_void = core::ptr::null();

/// Walk the system table's configuration entries and return the
/// vendor-table pointer for `EFI_DTB_TABLE_GUID`, or null if not found.
///
/// # Safety
/// Caller must guarantee `system_table` is a valid UEFI system table
/// pointer with `configuration_table` and `number_of_table_entries`
/// set as the firmware provided them. We trust the firmware here —
/// there is no realistic way for the unikernel to validate beyond the
/// header signature, which we don't bother checking in 2c.
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

/// UEFI image entry point. Called by the firmware after `LoadImage` +
/// `StartImage`. Spec signature: `EFI_STATUS efi_main(EFI_HANDLE,
/// EFI_SYSTEM_TABLE*)`.
///
/// # Safety
/// Called exactly once by UEFI firmware with `image_handle` and
/// `system_table` set per UEFI 2.10 §4.1. Until sub-PR 2d wires in
/// `ExitBootServices` and the kernel handoff, this function never
/// returns control to UEFI — it parks in a `wfi` loop after capturing
/// the DTB pointer.
#[no_mangle]
pub unsafe extern "efiapi" fn efi_main(
    _image_handle: Handle,
    system_table: *const SystemTable,
) -> Status {
    // Locate the Devicetree blob. The firmware on Jetson Orin's UEFI
    // exposes it via EFI_DTB_TABLE_GUID; we record it here so sub-PR 2d
    // can pass it to kernel_main.
    let dtb = unsafe { find_dtb(system_table) };
    unsafe {
        DTB_PTR = dtb;
    }

    // Sub-PR 2c stops here: park the CPU. Sub-PR 2d will replace this
    // with the ExitBootServices + jump-to-kernel_main sequence.
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
