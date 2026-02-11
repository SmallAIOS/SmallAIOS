// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NS16550A UART driver for RISC-V (QEMU virt @ 0x1000_0000).
//!
//! Provides early boot diagnostics output. This is a polled
//! (non-interrupt) driver suitable for boot and panic output.
//!
//! QEMU's 16550 emulation largely auto-configures, but we
//! perform a proper init for bare-metal correctness.

/// NS16550A UART base address for QEMU virt machine.
const UART_BASE: usize = 0x1000_0000;

// NS16550A register offsets
const RBR: usize = 0x00; // Receive Buffer Register (read)
const THR: usize = 0x00; // Transmit Holding Register (write)
const IER: usize = 0x01; // Interrupt Enable Register
const FCR: usize = 0x02; // FIFO Control Register (write)
const LCR: usize = 0x03; // Line Control Register
const MCR: usize = 0x04; // Modem Control Register
const LSR: usize = 0x05; // Line Status Register
const DLL: usize = 0x00; // Divisor Latch Low (DLAB=1)
const DLM: usize = 0x01; // Divisor Latch High (DLAB=1)

// Line Status Register bits
const LSR_DR: u8 = 1 << 0; // Data Ready
const LSR_THRE: u8 = 1 << 5; // Transmit Holding Register Empty

// Line Control Register bits
const LCR_DLAB: u8 = 1 << 7; // Divisor Latch Access Bit
const LCR_8N1: u8 = 0b11; // 8 data bits, no parity, 1 stop bit

/// Write a byte to a UART register.
#[inline(always)]
unsafe fn write_reg(offset: usize, val: u8) {
    let ptr = (UART_BASE + offset) as *mut u8;
    core::ptr::write_volatile(ptr, val);
}

/// Read a byte from a UART register.
#[inline(always)]
unsafe fn read_reg(offset: usize) -> u8 {
    let ptr = (UART_BASE + offset) as *const u8;
    core::ptr::read_volatile(ptr)
}

/// Initialize NS16550A UART.
///
/// Configuration: 115200 baud (with 1.8432 MHz reference clock), 8N1, FIFO enabled.
/// QEMU auto-configures baud rate, but we set divisor for completeness.
pub fn init() {
    unsafe {
        // Disable all interrupts
        write_reg(IER, 0x00);

        // Enable DLAB to set baud rate divisor
        write_reg(LCR, LCR_DLAB);

        // Baud rate 115200 with 1.8432 MHz clock: divisor = 1
        write_reg(DLL, 0x01);
        write_reg(DLM, 0x00);

        // 8 bits, no parity, 1 stop bit (also clears DLAB)
        write_reg(LCR, LCR_8N1);

        // Enable FIFO, clear TX/RX, 14-byte trigger level
        write_reg(FCR, 0xC7);

        // DTR + RTS + OUT2 (OUT2 enables interrupts on some hardware)
        write_reg(MCR, 0x0B);
    }
}

/// Wait until TX holding register is empty, then send a byte.
pub fn putc(byte: u8) {
    unsafe {
        // Wait while THR is not empty
        while (read_reg(LSR) & LSR_THRE) == 0 {
            core::hint::spin_loop();
        }
        write_reg(THR, byte);
    }
}

/// Try to read a byte from the UART. Returns None if no data available.
pub fn getc() -> Option<u8> {
    unsafe {
        if (read_reg(LSR) & LSR_DR) != 0 {
            Some(read_reg(RBR))
        } else {
            None
        }
    }
}

/// Write a string to UART, converting LF to CR+LF.
pub fn puts(s: &str) {
    for byte in s.bytes() {
        if byte == b'\n' {
            putc(b'\r');
        }
        putc(byte);
    }
}

/// Write a 64-bit value as hexadecimal.
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uart_constants() {
        assert_eq!(UART_BASE, 0x1000_0000);
        assert_eq!(LSR_THRE, 0x20);
        assert_eq!(LSR_DR, 0x01);
        assert_eq!(LCR_8N1, 0x03);
        assert_eq!(LCR_DLAB, 0x80);
    }

    #[test]
    fn test_register_offsets() {
        // THR and RBR share offset 0
        assert_eq!(THR, 0x00);
        assert_eq!(RBR, 0x00);
        // DLL and DLM also at 0x00/0x01 (with DLAB=1)
        assert_eq!(DLL, 0x00);
        assert_eq!(DLM, 0x01);
        assert_eq!(LSR, 0x05);
    }
}
