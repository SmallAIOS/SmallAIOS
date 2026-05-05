// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Minimal hand-rolled UEFI bindings for the Tegra234 boot path.
//!
//! Only the subset needed by `boot_uefi.rs` is defined: the EFI status
//! type, a few opaque handle/pointer types, the GUID type used for
//! configuration-table lookup, and a flat layout of `EfiSystemTable`
//! covering its first ~13 fields so we can reach `configuration_table`
//! and `boot_services`. Keeping these minimal avoids a `uefi-rs` crate
//! dependency and keeps the bindings auditable from a single file.
//!
//! References:
//!  - UEFI Specification 2.10, §4 (System Table) and §7 (Boot Services)
//!  - Linux ARM64 EFI stub:
//!    `arch/arm64/kernel/efi-header.S` and `drivers/firmware/efi/`
//!  - Devicetree binding GUID:
//!    `b1b621d5-f19c-41a5-830b-d9152c69aae0` (de-facto standard, used by
//!    U-Boot, GRUB, systemd-boot, NVIDIA UEFI on Jetson)

#![cfg(feature = "tegra234")]

use core::ffi::c_void;

/// `EFI_STATUS`. Newtype around `usize`; bit 63 is the error indicator
/// (`EFI_ERROR(x)` macro in the spec). We only check `is_success` and
/// fall through to a halt for any non-zero return — the kernel can't do
/// anything useful if the firmware refuses our calls anyway.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Status(pub usize);

impl Status {
    /// `EFI_SUCCESS` (= 0).
    pub const SUCCESS: Status = Status(0);

    /// True if `self == EFI_SUCCESS`.
    pub fn is_success(self) -> bool {
        self.0 == 0
    }
}

/// `EFI_HANDLE` — opaque pointer.
#[repr(transparent)]
pub struct Handle(pub *mut c_void);

/// `EFI_GUID` — wire format is little-endian for the first three fields.
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct Guid {
    pub a: u32,
    pub b: u16,
    pub c: u16,
    pub d: [u8; 8],
}

/// `EFI_DTB_TABLE_GUID` — the de-facto-standard GUID under which a UEFI
/// bootloader exposes the Devicetree blob to the loaded image. Defined
/// outside the UEFI spec proper but supported by U-Boot, GRUB,
/// systemd-boot, and NVIDIA's Jetson UEFI firmware.
pub const EFI_DTB_TABLE_GUID: Guid = Guid {
    a: 0xb1b6_21d5,
    b: 0xf19c,
    c: 0x41a5,
    d: [0x83, 0x0b, 0xd9, 0x15, 0x2c, 0x69, 0xaa, 0xe0],
};

/// `EFI_TABLE_HEADER` (UEFI 2.10 §4.2). Common header at the start of
/// `EFI_SYSTEM_TABLE`, `EFI_BOOT_SERVICES`, etc.
#[repr(C)]
pub struct TableHeader {
    pub signature: u64,
    pub revision: u32,
    pub header_size: u32,
    pub crc32: u32,
    pub reserved: u32,
}

/// `EFI_CONFIGURATION_TABLE` entry (UEFI 2.10 §4.6). The `SystemTable`
/// has an array of these, keyed by GUID.
#[repr(C)]
pub struct ConfigurationTable {
    pub vendor_guid: Guid,
    pub vendor_table: *const c_void,
}

/// `EFI_SYSTEM_TABLE` (UEFI 2.10 §4.3) — only the prefix we need to
/// reach `boot_services` and `configuration_table`. Fields after
/// `configuration_table` are omitted; pointer arithmetic stays correct
/// because we never index past the declared end.
#[repr(C)]
pub struct SystemTable {
    pub header: TableHeader,
    pub firmware_vendor: *const u16,
    pub firmware_revision: u32,
    pub console_in_handle: Handle,
    pub con_in: *const c_void,
    pub console_out_handle: Handle,
    pub con_out: *mut SimpleTextOutputProtocol,
    pub standard_error_handle: Handle,
    pub std_err: *mut SimpleTextOutputProtocol,
    pub runtime_services: *const c_void,
    pub boot_services: *const BootServices,
    pub number_of_table_entries: usize,
    pub configuration_table: *const ConfigurationTable,
}

/// `EFI_BOOT_SERVICES` (UEFI 2.10 §7) — opaque to us until the sub-PR
/// that lands `ExitBootServices` + memory-map handoff to `kernel_main`.
/// Defining the struct as header-only here keeps the
/// `SystemTable.boot_services` pointer typed without pinning the layout
/// of fields we don't yet use.
#[repr(C)]
pub struct BootServices {
    pub header: TableHeader,
    // ExitBootServices + GetMemoryMap fields land alongside the real
    // post-ExitBootServices kernel handoff.
}

/// `EFI_SIMPLE_TEXT_OUTPUT_PROTOCOL` (UEFI 2.10 §12.4). The
/// `SystemTable.con_out` field points to one of these. On Jetson Orin
/// the firmware routes `con_out` through the TCU mailbox, so calling
/// `output_string` here lands on whatever serial console the user has
/// attached to the J-class carrier's UART header — which is exactly
/// what we need for the "first observable signal" milestone before any
/// kernel-side TCU driver exists.
///
/// Layout: `Reset` is the first slot, `OutputString` the second; spec
/// guarantees stable ordering. We only call `output_string` here.
/// `TestString`, `QueryMode`, `SetMode`, `SetAttribute`, `ClearScreen`,
/// `SetCursorPosition`, `EnableCursor`, and the `Mode` pointer are
/// defined by the spec but unused — declaring the trailing fields would
/// require more spec types we don't need yet, so we deliberately stop
/// at the second fn pointer. Pointer arithmetic stays correct because
/// we never index past the declared end.
#[repr(C)]
pub struct SimpleTextOutputProtocol {
    pub reset: unsafe extern "efiapi" fn(this: *mut Self, extended_verification: bool) -> Status,
    pub output_string: unsafe extern "efiapi" fn(this: *mut Self, string: *const u16) -> Status,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtb_guid_bytes_match_spec_string() {
        // Spec string: b1b621d5-f19c-41a5-830b-d9152c69aae0
        // Little-endian first three fields, big-endian remainder.
        let g = EFI_DTB_TABLE_GUID;
        assert_eq!(g.a, 0xb1b6_21d5);
        assert_eq!(g.b, 0xf19c);
        assert_eq!(g.c, 0x41a5);
        assert_eq!(g.d, [0x83, 0x0b, 0xd9, 0x15, 0x2c, 0x69, 0xaa, 0xe0]);
    }

    #[test]
    fn status_success_is_zero() {
        assert_eq!(Status::SUCCESS.0, 0);
        assert!(Status::SUCCESS.is_success());
        assert!(!Status(1).is_success());
    }
}
