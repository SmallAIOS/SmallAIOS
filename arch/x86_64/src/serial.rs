// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Serial console driver for x86-64 (COM1 at 0x3F8).
//!
//! Provides early boot diagnostics output. This is a polled
//! (non-interrupt) driver suitable for boot and panic output.

use core::arch::asm;

// COM1 I/O port addresses
const COM1_DATA: u16 = 0x3F8;
const COM1_INT_ENABLE: u16 = 0x3F9;
const COM1_FIFO_CTRL: u16 = 0x3FA;
const COM1_LINE_CTRL: u16 = 0x3FB;
const COM1_MODEM_CTRL: u16 = 0x3FC;
const COM1_LINE_STATUS: u16 = 0x3FD;

/// Write a byte to an I/O port.
#[inline(always)]
unsafe fn outb(port: u16, val: u8) {
    asm!(
        "out dx, al",
        in("dx") port,
        in("al") val,
        options(nomem, nostack, preserves_flags),
    );
}

/// Read a byte from an I/O port.
#[inline(always)]
unsafe fn inb(port: u16) -> u8 {
    let val: u8;
    asm!(
        "in al, dx",
        out("al") val,
        in("dx") port,
        options(nomem, nostack, preserves_flags),
    );
    val
}

/// Initialize COM1 serial port at 115200 baud, 8N1.
pub fn init() {
    unsafe {
        // Disable interrupts
        outb(COM1_INT_ENABLE, 0x00);

        // Enable DLAB (set baud rate divisor)
        outb(COM1_LINE_CTRL, 0x80);

        // Set divisor to 1 (115200 baud)
        outb(COM1_DATA, 0x01); // divisor low byte
        outb(COM1_INT_ENABLE, 0x00); // divisor high byte

        // 8 bits, no parity, one stop bit, DLAB off
        outb(COM1_LINE_CTRL, 0x03);

        // Enable FIFO, clear them, 14-byte threshold
        outb(COM1_FIFO_CTRL, 0xC7);

        // Set RTS/DSR, enable IRQs (for loopback test)
        outb(COM1_MODEM_CTRL, 0x0B);

        // Loopback test: set loopback mode
        outb(COM1_MODEM_CTRL, 0x1E);
        outb(COM1_DATA, 0xAE);

        // Check if serial chip responds
        if inb(COM1_DATA) != 0xAE {
            // Serial port failed — nothing we can do, continue anyway
            return;
        }

        // Disable loopback, set normal operation (OUT1 + OUT2 + RTS + DTR)
        outb(COM1_MODEM_CTRL, 0x0F);
    }
}

/// Wait until the transmit buffer is empty, then send a byte.
pub fn putc(byte: u8) {
    unsafe {
        // Wait for transmit holding register empty (bit 5 of LSR)
        while (inb(COM1_LINE_STATUS) & 0x20) == 0 {
            core::hint::spin_loop();
        }
        outb(COM1_DATA, byte);
    }
}

/// Write a string to serial.
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
