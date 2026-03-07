// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! UART driver for AArch64 platforms.
//!
//! - **QEMU virt** (`qemu-virt`): PL011 UART @ 0x0900_0000
//! - **Tegra X1** (`tegra-x1`): NS16550A UART-A @ 0x7000_6000 (reg-shift=2)
//!
//! On Tegra X1, U-Boot pre-initializes UART-A at 115200 8N1.
//! The `init()` function is a no-op — only polled TX is needed.

use crate::platform;

// ─── PL011 (QEMU virt) ──────────────────────────────────────────────────────

#[cfg(feature = "qemu-virt")]
mod pl011 {
    use super::platform;

    const UARTDR: usize = 0x000;
    const UARTFR: usize = 0x018;
    const UARTIBRD: usize = 0x024;
    const UARTFBRD: usize = 0x028;
    const UARTLCR_H: usize = 0x02C;
    const UARTCR: usize = 0x030;
    const UARTIMSC: usize = 0x038;

    const FR_TXFF: u32 = 1 << 5;

    #[inline(always)]
    unsafe fn write_reg(offset: usize, val: u32) {
        let ptr = (platform::UART_BASE + offset) as *mut u32;
        core::ptr::write_volatile(ptr, val);
    }

    #[inline(always)]
    unsafe fn read_reg(offset: usize) -> u32 {
        let ptr = (platform::UART_BASE + offset) as *const u32;
        core::ptr::read_volatile(ptr)
    }

    pub fn init() {
        unsafe {
            write_reg(UARTCR, 0);
            write_reg(UARTIMSC, 0);
            // 115200 baud @ 24 MHz: divisor = 13.0208
            write_reg(UARTIBRD, 13);
            write_reg(UARTFBRD, 1);
            // 8N1, FIFO enabled
            write_reg(UARTLCR_H, (0b11 << 5) | (1 << 4));
            // Enable UART, TX, RX
            write_reg(UARTCR, (1 << 0) | (1 << 8) | (1 << 9));
        }
    }

    pub fn putc(byte: u8) {
        unsafe {
            while (read_reg(UARTFR) & FR_TXFF) != 0 {
                core::hint::spin_loop();
            }
            write_reg(UARTDR, byte as u32);
        }
    }
}

// ─── NS16550A (Tegra X1) ────────────────────────────────────────────────────

#[cfg(feature = "tegra-x1")]
mod ns16550a {
    use super::platform;

    /// Transmit Holding Register (offset 0 × 4 = 0x00).
    pub(crate) const THR: usize = 0x00;
    /// Line Status Register (offset 5 × 4 = 0x14, reg-shift=2).
    pub(crate) const LSR: usize = 0x14;
    /// Transmitter Holding Register Empty bit.
    pub(crate) const LSR_THRE: u32 = 1 << 5;

    /// No-op: U-Boot pre-configures UART-A at 115200 8N1.
    pub fn init() {}

    /// Polled character output. Spin-waits on LSR THRE, then writes to THR.
    pub fn putc(byte: u8) {
        unsafe {
            let lsr_ptr = (platform::UART_BASE + LSR) as *const u32;
            let thr_ptr = (platform::UART_BASE + THR) as *mut u32;
            while core::ptr::read_volatile(lsr_ptr) & LSR_THRE == 0 {
                core::hint::spin_loop();
            }
            core::ptr::write_volatile(thr_ptr, byte as u32);
        }
    }
}

// ─── Public API (platform-independent) ───────────────────────────────────────

/// Initialize the UART. No-op on Tegra (U-Boot pre-inits).
pub fn init() {
    #[cfg(feature = "qemu-virt")]
    pl011::init();
    #[cfg(feature = "tegra-x1")]
    ns16550a::init();
}

/// Write a single byte to UART.
pub fn putc(byte: u8) {
    #[cfg(feature = "qemu-virt")]
    pl011::putc(byte);
    #[cfg(feature = "tegra-x1")]
    ns16550a::putc(byte);
}

/// Write a string to UART, converting `\n` to `\r\n`.
pub fn puts(s: &str) {
    for byte in s.bytes() {
        if byte == b'\n' {
            putc(b'\r');
        }
        putc(byte);
    }
}

/// Write a 64-bit value as hex.
pub fn put_hex(val: u64) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    if val == 0 {
        putc(b'0');
        return;
    }

    let mut started = false;
    for i in (0..16).rev() {
        let nibble = ((val >> (i * 4)) & 0xF) as usize;
        if nibble != 0 {
            started = true;
        }
        if started {
            putc(HEX[nibble]);
        }
    }
}

/// Write a 16-bit value as zero-padded 4-digit hex (e.g. `10EC`).
pub fn put_hex16(val: u16) {
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for i in (0..4).rev() {
        let nibble = ((val >> (i * 4)) & 0xF) as usize;
        putc(HEX[nibble]);
    }
}

/// Write an f32 value as decimal with 4 decimal places.
///
/// Uses integer arithmetic only (no libm/formatting).
/// Output format: `[-]integer.dddd`
pub fn put_f32(val: f32) {
    // Handle negative
    if val < 0.0 {
        putc(b'-');
        put_f32(-val);
        return;
    }

    // Integer part
    let int_part = val as u64;
    put_dec(int_part);
    putc(b'.');

    // Fractional part: 4 decimal digits
    let frac = val - int_part as f32;
    let scaled = (frac * 10000.0 + 0.5) as u32;
    // Zero-pad to 4 digits
    if scaled < 1000 {
        putc(b'0');
    }
    if scaled < 100 {
        putc(b'0');
    }
    if scaled < 10 {
        putc(b'0');
    }
    put_dec(scaled as u64);
}

/// Write a 64-bit value as decimal.
pub fn put_dec(val: u64) {
    if val == 0 {
        putc(b'0');
        return;
    }

    let mut buf = [0u8; 20];
    let mut pos = 0;
    let mut v = val;

    while v > 0 {
        buf[pos] = b'0' + (v % 10) as u8;
        v /= 10;
        pos += 1;
    }

    for i in (0..pos).rev() {
        putc(buf[i]);
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "tegra-x1")]
    #[test]
    fn test_ns16550a_register_offsets() {
        // THR at offset 0 (register 0 << 2)
        assert_eq!(super::ns16550a::THR, 0x00);
        // LSR at offset 5 × 4 = 0x14 (register 5 << 2)
        assert_eq!(super::ns16550a::LSR, 0x14);
        // LSR THRE is bit 5
        assert_eq!(super::ns16550a::LSR_THRE, 0x20);
    }
}
