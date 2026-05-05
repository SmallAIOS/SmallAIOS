// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UEFI bin entry point — the .efi loaded by Jetson Orin's UEFI firmware.
//!
//! The actual `efi_main` function lives in
//! `smallaios_arch_aarch64::boot_uefi`. This file just makes the bin
//! crate executable: `extern crate` brings the lib's `efi_main` symbol
//! into the link, and `aarch64-unknown-uefi` emits PE/COFF with that
//! symbol set as the entry point automatically. The lib also supplies
//! the `#[panic_handler]` (currently writes to UART, which is the TCU
//! stub on `--features tegra234` — silent until sub-PR 2d's driver).
//!
//! Built via `just build-kernel-uefi` (target `aarch64-unknown-uefi`,
//! `--features tegra234`). See `arch/aarch64/src/boot_uefi.rs` and
//! `docs/jetson-orin-uefi-boot.md` (sub-PR 2g) for the full flow.

#![no_std]
#![no_main]

extern crate smallaios_arch_aarch64;
