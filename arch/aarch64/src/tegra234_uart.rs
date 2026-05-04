// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra234 kernel-side UART driver — **DRAM-stash placeholder**.
//!
//! ## Status
//!
//! `putc` currently writes bytes into a static DRAM buffer instead of
//! a real UART. **No output reaches the carrier's TTL header.** This
//! is intentional for sub-PR 2e — getting actual MMIO output to the
//! console post-`ExitBootServices` on Tegra234 is harder than expected
//! and is deferred to a follow-up sub-PR.
//!
//! ## Why the placeholder
//!
//! Three real-UART approaches were tried during 2e's bring-up and all
//! faulted from the EL2 context our kernel runs in post-EBS:
//!
//! | Address | Protocol | What happened |
//! |---|---|---|
//! | `0x0310_0000` (UART_1 / UARTA NS16550) | LSR poll + THR write | SError (EC=0x2F) — fabric rejected the access. UEFI gates UART_1 before EBS. |
//! | `0x0C28_0000` (TCU mailbox) | LSR poll + THR write | Sync Exception — page not mapped in UEFI's post-EBS page tables. |
//! | `0x0C28_0000` (TCU mailbox) | Write-only with EOM bit | Sync Exception — same translation fault. |
//!
//! Pre-EBS, UEFI's `con_out` works because UEFI internally reaches the
//! console through a privileged path (probably an SMC into TF-A or a
//! TrustZone-enabled MMIO region). After EBS, that path is gone and
//! direct EL2 access to either `0x0310_0000` or `0x0C28_0000` is
//! blocked at the SoC firewall / page-table level.
//!
//! ## What unblocking this requires (follow-up sub-PR)
//!
//! Likely candidates, none small:
//!   1. Parse the DTB (already at the pointer captured in `boot_uefi`)
//!      to find `/chosen/stdout-path` and confirm the active UART.
//!   2. Set up our own page tables that map the UART region with the
//!      right access attributes from EL2 / lower EL1.
//!   3. Use SMC into NVIDIA's TF-A vendor namespace (SiP function ID
//!      space) to print via the secure-side console.
//!   4. Stay in UEFI Boot Services (don't call ExitBootServices) and
//!      keep using `con_out` — defeats the purpose of "kernel boot"
//!      but lets the kernel print until we sort out a real UART.
//!
//! ## DRAM-stash semantics
//!
//! The placeholder still satisfies the `putc` contract (write a byte,
//! never fault) so the kernel can run through `kernel_main`'s boot
//! stages without issuing MMIO from EL2. Bytes pile up in `DIAG_BUF`
//! up to 1 KiB and then are dropped. The buffer survives a soft
//! reset, so a future debugging tool could recover it by `dd`'ing
//! `/dev/mem` from L4T after a Phase 2 boot.

#![cfg(feature = "tegra234")]

// Reference `crate::platform::UART_BASE` so it stays a published
// constant for the follow-up sub-PR that wires real MMIO. Without
// this the `-D warnings` build would flag the constant as unused
// under `--features tegra234`.
#[allow(unused_imports)]
use crate::platform::UART_BASE as _;

/// Placeholder buffer: `putc` writes here instead of MMIO. Up to 1
/// KiB of buffered output is preserved across kernel boot stages;
/// further bytes are dropped silently. Future debugging tooling can
/// reach the buffer via `/dev/mem` from L4T after a soft reset, since
/// DRAM contents survive the reset.
const DIAG_BUF_SIZE: usize = 1024;

#[repr(C, align(8))]
struct DiagBuf {
    cursor: usize,
    bytes: [u8; DIAG_BUF_SIZE],
}

static mut DIAG_BUF: DiagBuf = DiagBuf {
    cursor: 0,
    bytes: [0; DIAG_BUF_SIZE],
};

/// Stash a byte. Never faults (no MMIO). See module docs for why
/// this is a placeholder rather than a real UART driver.
///
/// # Safety
/// Single-threaded; the static buffer isn't synchronized for concurrent
/// access. Phase 2's kernel boot is single-threaded so this is fine.
pub fn putc(byte: u8) {
    unsafe {
        let buf = &raw mut DIAG_BUF;
        let cursor = (*buf).cursor;
        if cursor < DIAG_BUF_SIZE {
            (*buf).bytes[cursor] = byte;
            (*buf).cursor = cursor + 1;
        }
    }
}

/// No-op. Kept for symmetry with the `pl011::init` / `ns16550a::init`
/// shape in `uart.rs`; a real driver would use this to set baud /
/// pad / clock state if needed.
pub fn init() {}
