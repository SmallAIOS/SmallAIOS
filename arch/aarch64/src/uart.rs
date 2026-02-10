// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! PL011 UART driver for AArch64 (QEMU virt @ 0x0900_0000).
//!
//! Provides early boot diagnostics output. This is a polled
//! (non-interrupt) driver suitable for boot and panic output.
//!
//! QEMU's PL011 emulation auto-configures baud rate, so we
//! just need to enable TX/RX and the UART itself.

/// PL011 UART base address for QEMU virt machine.
const UART_BASE: usize = 0x0900_0000;

// PL011 register offsets
const UARTDR: usize = 0x000; // Data register
const UARTFR: usize = 0x018; // Flag register
const UARTIBRD: usize = 0x024; // Integer baud rate divisor
const UARTFBRD: usize = 0x028; // Fractional baud rate divisor
const UARTLCR_H: usize = 0x02C; // Line control register
const UARTCR: usize = 0x030; // Control register
const UARTIMSC: usize = 0x038; // Interrupt mask set/clear

// Flag register bits
const FR_TXFF: u32 = 1 << 5; // Transmit FIFO full

/// Write a 32-bit value to a UART register.
#[inline(always)]
unsafe fn write_reg(offset: usize, val: u32) {
    let ptr = (UART_BASE + offset) as *mut u32;
    core::ptr::write_volatile(ptr, val);
}

/// Read a 32-bit value from a UART register.
#[inline(always)]
unsafe fn read_reg(offset: usize) -> u32 {
    let ptr = (UART_BASE + offset) as *const u32;
    core::ptr::read_volatile(ptr)
}

/// Initialize PL011 UART.
///
/// QEMU's PL011 is largely pre-configured, but we perform a
/// proper init sequence for bare-metal correctness:
/// 1. Disable UART
/// 2. Set baud rate (115200 @ 24MHz UARTCLK)
/// 3. Configure 8N1, enable FIFOs
/// 4. Enable TX, RX, and UART
pub fn init() {
    unsafe {
        // Disable UART during configuration
        write_reg(UARTCR, 0);

        // Mask all interrupts (we use polling)
        write_reg(UARTIMSC, 0);

        // Set baud rate: 115200 with 24MHz clock
        // Divisor = 24_000_000 / (16 * 115200) = 13.0208...
        // Integer part = 13, Fractional part = round(0.0208 * 64) = 1
        write_reg(UARTIBRD, 13);
        write_reg(UARTFBRD, 1);

        // 8 bits, no parity, 1 stop bit, enable FIFOs
        write_reg(UARTLCR_H, (0b11 << 5) | (1 << 4)); // WLEN=8bit | FEN

        // Enable UART, TX, RX
        write_reg(UARTCR, (1 << 0) | (1 << 8) | (1 << 9)); // UARTEN | TXE | RXE
    }
}

/// Wait until TX FIFO has space, then send a byte.
pub fn putc(byte: u8) {
    unsafe {
        // Wait while TX FIFO is full
        while (read_reg(UARTFR) & FR_TXFF) != 0 {
            core::hint::spin_loop();
        }
        write_reg(UARTDR, byte as u32);
    }
}

/// Write a string to UART.
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

/// Write a 64-bit value as decimal.
pub fn put_dec(val: u64) {
    if val == 0 {
        putc(b'0');
        return;
    }

    let mut buf = [0u8; 20]; // max 20 digits for u64
    let mut pos = 0;
    let mut v = val;

    while v > 0 {
        buf[pos] = b'0' + (v % 10) as u8;
        v /= 10;
        pos += 1;
    }

    // Print in reverse
    for i in (0..pos).rev() {
        putc(buf[i]);
    }
}
