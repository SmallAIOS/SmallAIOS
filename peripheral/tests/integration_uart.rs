// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the UART peripheral subsystem.
//!
//! Tests exercise a mock UART controller implementing the
//! `smallaios_kernel::hal::UartController` trait, covering baud rate
//! configuration, write/read operations, read_until delimiter-based
//! parsing (suitable for NMEA sentence parsing), flush, and error
//! handling.

#![cfg(feature = "uart")]

extern crate alloc;
use alloc::collections::VecDeque;
use alloc::vec::Vec;

use smallaios_kernel::hal::{
    HalError, UartConfig, UartController, UartFlowControl, UartIrqSource, UartParity, UartRxMode,
    UartStopBits,
};
use smallaios_peripheral::PeripheralType;

// ---------------------------------------------------------------------------
// Mock UART controller
// ---------------------------------------------------------------------------

/// Mock UART controller for testing.
struct MockUart {
    initialized: bool,
    config: Option<UartConfig>,
    /// TX buffer: bytes written via write/write_all.
    tx_buf: Vec<u8>,
    /// RX FIFO: pre-loaded data returned by read operations.
    rx_fifo: VecDeque<u8>,
    /// Whether TX has been flushed.
    flushed: bool,
    /// Simulated IRQ source for next irq_handler call.
    pending_irq: Option<UartIrqSource>,
}

impl MockUart {
    fn new() -> Self {
        Self {
            initialized: false,
            config: None,
            tx_buf: Vec::new(),
            rx_fifo: VecDeque::new(),
            flushed: false,
            pending_irq: None,
        }
    }

    /// Inject data into the RX FIFO (simulates received bytes).
    fn inject_rx(&mut self, data: &[u8]) {
        for &b in data {
            self.rx_fifo.push_back(b);
        }
    }
}

impl UartController for MockUart {
    fn init(&mut self, config: UartConfig) -> Result<(), HalError> {
        if config.data_bits != 7 && config.data_bits != 8 {
            return Err(HalError::InvalidConfig);
        }
        self.config = Some(config);
        self.initialized = true;
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        self.tx_buf.extend_from_slice(data);
        self.flushed = false;
        Ok(data.len())
    }

    fn write_all(&mut self, data: &[u8]) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        self.tx_buf.extend_from_slice(data);
        self.flushed = false;
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<usize, HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        let count = buf.len().min(self.rx_fifo.len());
        for item in buf.iter_mut().take(count) {
            *item = self.rx_fifo.pop_front().unwrap();
        }
        Ok(count)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<usize, HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if self.rx_fifo.len() < buf.len() {
            // Read what we can.
            let count = self.rx_fifo.len();
            for item in buf.iter_mut().take(count) {
                *item = self.rx_fifo.pop_front().unwrap();
            }
            Ok(count)
        } else {
            for item in buf.iter_mut() {
                *item = self.rx_fifo.pop_front().unwrap();
            }
            Ok(buf.len())
        }
    }

    fn read_until(&mut self, delimiter: u8, buf: &mut [u8]) -> Result<usize, HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        let mut count = 0;
        while count < buf.len() {
            if let Some(byte) = self.rx_fifo.pop_front() {
                buf[count] = byte;
                count += 1;
                if byte == delimiter {
                    break;
                }
            } else {
                break;
            }
        }
        Ok(count)
    }

    fn flush(&mut self) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        self.flushed = true;
        Ok(())
    }

    fn rx_available(&self) -> usize {
        self.rx_fifo.len()
    }

    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError> {
        if let Some(src) = self.pending_irq.take() {
            Ok(src)
        } else {
            Err(HalError::InterruptError)
        }
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.initialized = false;
        self.config = None;
        self.tx_buf.clear();
        self.rx_fifo.clear();
        self.flushed = false;
        self.pending_irq = None;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_uart_init_115200_8n1() {
    let mut uart = MockUart::new();
    let config = UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    };
    assert!(uart.init(config).is_ok());
    assert!(uart.initialized);
    let cfg = uart.config.unwrap();
    assert_eq!(cfg.baud_rate, 115200);
    assert_eq!(cfg.data_bits, 8);
    assert_eq!(cfg.parity, UartParity::None);
}

#[test]
fn test_uart_init_9600_even_parity() {
    let mut uart = MockUart::new();
    let config = UartConfig {
        baud_rate: 9600,
        data_bits: 8,
        parity: UartParity::Even,
        stop_bits: UartStopBits::Two,
        flow_control: UartFlowControl::RtsCts,
        rx_mode: UartRxMode::Interrupt { fifo_threshold: 8 },
        timeout_us: 10_000,
    };
    assert!(uart.init(config).is_ok());
    let cfg = uart.config.unwrap();
    assert_eq!(cfg.baud_rate, 9600);
    assert_eq!(cfg.parity, UartParity::Even);
    assert_eq!(cfg.stop_bits, UartStopBits::Two);
    assert_eq!(cfg.flow_control, UartFlowControl::RtsCts);
}

#[test]
fn test_uart_write_and_read() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    })
    .unwrap();

    let msg = b"Hello UART";
    let written = uart.write(msg).unwrap();
    assert_eq!(written, msg.len());
    assert_eq!(&uart.tx_buf, msg);

    // Inject response.
    uart.inject_rx(b"OK\r\n");
    let mut buf = [0u8; 4];
    let read = uart.read(&mut buf).unwrap();
    assert_eq!(read, 4);
    assert_eq!(&buf, b"OK\r\n");
}

#[test]
fn test_uart_write_all() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    })
    .unwrap();

    uart.write_all(b"AT+RST\r\n").unwrap();
    assert_eq!(uart.tx_buf, b"AT+RST\r\n");
}

#[test]
fn test_uart_read_until_nmea_sentence() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 9600,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    })
    .unwrap();

    // Inject a NMEA sentence followed by more data.
    let nmea = b"$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,47.0,M,,*47\r\n";
    uart.inject_rx(nmea);
    uart.inject_rx(b"$GPGSA,A,3,04,05,,09,12");

    let mut buf = [0u8; 128];
    let count = uart.read_until(b'\n', &mut buf).unwrap();
    assert_eq!(count, nmea.len());
    assert_eq!(&buf[..count], nmea.as_slice());

    // Verify remaining data is still in FIFO.
    assert_eq!(uart.rx_available(), 23); // "$GPGSA,A,3,04,05,,09,12"
}

#[test]
fn test_uart_read_exact_partial() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 1000,
    })
    .unwrap();

    // Only 3 bytes available but requesting 5.
    uart.inject_rx(&[0xAA, 0xBB, 0xCC]);
    let mut buf = [0u8; 5];
    let count = uart.read_exact(&mut buf).unwrap();
    assert_eq!(count, 3);
    assert_eq!(buf[0], 0xAA);
    assert_eq!(buf[1], 0xBB);
    assert_eq!(buf[2], 0xCC);
}

#[test]
fn test_uart_flush() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    })
    .unwrap();

    uart.write(b"data").unwrap();
    assert!(!uart.flushed);
    uart.flush().unwrap();
    assert!(uart.flushed);
}

#[test]
fn test_uart_irq_handler() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Interrupt { fifo_threshold: 4 },
        timeout_us: 0,
    })
    .unwrap();

    uart.pending_irq = Some(UartIrqSource::RxDataAvailable);
    let src = uart.irq_handler().unwrap();
    assert_eq!(src, UartIrqSource::RxDataAvailable);

    // No more pending.
    assert_eq!(uart.irq_handler(), Err(HalError::InterruptError));
}

#[test]
fn test_uart_uninitialized_error() {
    let mut uart = MockUart::new();
    assert_eq!(uart.write(b"test"), Err(HalError::InitFailed));
    assert_eq!(uart.write_all(b"test"), Err(HalError::InitFailed));
    let mut buf = [0u8; 4];
    assert_eq!(uart.read(&mut buf), Err(HalError::InitFailed));
    assert_eq!(uart.flush(), Err(HalError::InitFailed));
}

#[test]
fn test_uart_reset() {
    let mut uart = MockUart::new();
    uart.init(UartConfig {
        baud_rate: 115200,
        data_bits: 8,
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    })
    .unwrap();

    uart.write(b"hello").unwrap();
    uart.inject_rx(b"world");
    assert!(uart.initialized);

    uart.reset().unwrap();
    assert!(!uart.initialized);
    assert!(uart.config.is_none());
    assert!(uart.tx_buf.is_empty());
    assert_eq!(uart.rx_available(), 0);
}

#[test]
fn test_uart_invalid_data_bits() {
    let mut uart = MockUart::new();
    let config = UartConfig {
        baud_rate: 115200,
        data_bits: 6, // Invalid
        parity: UartParity::None,
        stop_bits: UartStopBits::One,
        flow_control: UartFlowControl::None,
        rx_mode: UartRxMode::Polling,
        timeout_us: 0,
    };
    assert_eq!(uart.init(config), Err(HalError::InvalidConfig));
}

#[test]
fn test_uart_peripheral_type() {
    assert_eq!(PeripheralType::from_u8(3), Some(PeripheralType::Uart));
    assert_eq!(PeripheralType::Uart as u8, 3);
}
