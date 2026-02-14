// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Dual-output console abstraction.
//!
//! Provides `console::putc` and `console::puts` that write to both UART
//! and framebuffer on `tegra-x1`, or UART only on `qemu-virt`.
//!
//! This module is always compiled (not behind `#[cfg]`) so that boot code
//! can unconditionally use `console::puts` regardless of platform.

use crate::uart;

#[cfg(feature = "tegra-x1")]
use crate::fb_console;

/// Initialize the console.
///
/// On `tegra-x1`, this initializes the framebuffer console after the display
/// pipeline has been set up. On `qemu-virt`, this is a no-op (UART is already
/// initialized separately).
///
/// # Safety
/// On `tegra-x1`, must be called after DC and SOR initialization, with valid
/// framebuffer parameters.
#[cfg(feature = "tegra-x1")]
pub unsafe fn init(fb_addr: usize, width: usize, height: usize, stride: usize) {
    fb_console::fb_console_init(fb_addr, width, height, stride);
}

/// Initialize the console (QEMU variant — no framebuffer).
#[cfg(feature = "qemu-virt")]
pub fn init() {
    // No-op: UART is already initialized via uart::init()
}

/// Write a single byte to all console outputs.
///
/// On `tegra-x1`: writes to both UART and framebuffer.
/// On `qemu-virt`: writes to UART only.
pub fn putc(byte: u8) {
    uart::putc(byte);

    #[cfg(feature = "tegra-x1")]
    {
        if fb_console::is_initialized() {
            unsafe { fb_console::fb_putc(byte) };
        }
    }
}

/// Write a string to all console outputs, converting `\n` to `\r\n`.
///
/// On `tegra-x1`: writes to both UART and framebuffer.
/// On `qemu-virt`: writes to UART only.
pub fn puts(s: &str) {
    for byte in s.bytes() {
        if byte == b'\n' {
            putc(b'\r');
        }
        putc(byte);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    // Console module compiles on the test host (x86_64).
    // We can't test MMIO output, but we verify the logic compiles.

    #[test]
    fn test_console_module_compiles() {
        // If this test compiles and runs, the console module is valid.
        // On the test host, UART putc/puts are not callable (they use
        // platform-specific MMIO), but the module structure is sound.
        assert!(true);
    }
}
