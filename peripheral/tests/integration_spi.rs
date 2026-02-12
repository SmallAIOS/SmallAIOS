// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the SPI peripheral subsystem.
//!
//! Tests exercise a mock SPI controller implementing the
//! `smallaios_kernel::hal::SpiController` trait, covering all four SPI
//! modes, full-duplex transfer, write-only, read-only, chip-select
//! management, and error handling.

#![cfg(feature = "spi")]

extern crate alloc;

use smallaios_kernel::hal::{DmaToken, HalError, SpiBitOrder, SpiConfig, SpiController, SpiMode};
use smallaios_peripheral::PeripheralType;

// ---------------------------------------------------------------------------
// Mock SPI controller
// ---------------------------------------------------------------------------

/// Mock SPI controller for testing.
struct MockSpi {
    initialized: bool,
    config: Option<SpiConfig>,
    /// Simulated MISO response data (returned on next transfer/read).
    miso_data: alloc::vec::Vec<u8>,
    /// Captured MOSI data from last transfer/write.
    last_mosi: alloc::vec::Vec<u8>,
    /// Which chip-select lines are currently asserted.
    cs_asserted: [bool; 4],
    /// Transfer count.
    transfer_count: u32,
}

impl MockSpi {
    fn new() -> Self {
        Self {
            initialized: false,
            config: None,
            miso_data: alloc::vec::Vec::new(),
            last_mosi: alloc::vec::Vec::new(),
            cs_asserted: [false; 4],
            transfer_count: 0,
        }
    }

    /// Pre-load MISO response data.
    fn set_miso_response(&mut self, data: &[u8]) {
        self.miso_data = data.to_vec();
    }
}

impl SpiController for MockSpi {
    fn init(&mut self, config: SpiConfig) -> Result<(), HalError> {
        if config.word_size != 8 && config.word_size != 16 && config.word_size != 32 {
            return Err(HalError::InvalidConfig);
        }
        self.config = Some(config);
        self.initialized = true;
        Ok(())
    }

    fn transfer(&mut self, cs: u8, tx_data: &[u8], rx_buf: &mut [u8]) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if cs >= 4 {
            return Err(HalError::InvalidConfig);
        }
        self.last_mosi = tx_data.to_vec();
        let copy_len = rx_buf.len().min(self.miso_data.len());
        rx_buf[..copy_len].copy_from_slice(&self.miso_data[..copy_len]);
        self.transfer_count += 1;
        Ok(())
    }

    fn write(&mut self, cs: u8, data: &[u8]) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if cs >= 4 {
            return Err(HalError::InvalidConfig);
        }
        self.last_mosi = data.to_vec();
        self.transfer_count += 1;
        Ok(())
    }

    fn read(&mut self, cs: u8, buf: &mut [u8]) -> Result<(), HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if cs >= 4 {
            return Err(HalError::InvalidConfig);
        }
        let copy_len = buf.len().min(self.miso_data.len());
        buf[..copy_len].copy_from_slice(&self.miso_data[..copy_len]);
        // Remaining bytes stay zero (MOSI sends zeros on read).
        self.transfer_count += 1;
        Ok(())
    }

    fn transfer_dma(
        &mut self,
        cs: u8,
        _tx_addr: u64,
        _rx_addr: u64,
        _len: u32,
    ) -> Result<DmaToken, HalError> {
        if !self.initialized {
            return Err(HalError::InitFailed);
        }
        if cs >= 4 {
            return Err(HalError::InvalidConfig);
        }
        Ok(DmaToken {
            channel: cs,
            transfer_id: self.transfer_count,
            complete: true,
            error: false,
        })
    }

    fn cs_assert(&mut self, cs: u8) -> Result<(), HalError> {
        if cs >= 4 {
            return Err(HalError::InvalidConfig);
        }
        self.cs_asserted[cs as usize] = true;
        Ok(())
    }

    fn cs_deassert(&mut self, cs: u8) -> Result<(), HalError> {
        if cs >= 4 {
            return Err(HalError::InvalidConfig);
        }
        self.cs_asserted[cs as usize] = false;
        Ok(())
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.initialized = false;
        self.config = None;
        self.miso_data.clear();
        self.last_mosi.clear();
        self.cs_asserted = [false; 4];
        self.transfer_count = 0;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn test_spi_init_mode0() {
    let mut spi = MockSpi::new();
    let config = SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 1_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 8,
        cs_active_low: true,
        timeout_us: 5000,
    };
    assert!(spi.init(config).is_ok());
    assert!(spi.initialized);
    assert_eq!(spi.config.unwrap().mode, SpiMode::Mode0);
}

#[test]
fn test_spi_all_four_modes() {
    let modes = [
        SpiMode::Mode0,
        SpiMode::Mode1,
        SpiMode::Mode2,
        SpiMode::Mode3,
    ];
    for mode in &modes {
        let mut spi = MockSpi::new();
        let config = SpiConfig {
            mode: *mode,
            clock_hz: 8_000_000,
            bit_order: SpiBitOrder::MsbFirst,
            word_size: 8,
            cs_active_low: true,
            timeout_us: 1000,
        };
        assert!(spi.init(config).is_ok(), "Failed to init mode {:?}", mode);
        assert_eq!(spi.config.unwrap().mode, *mode);
    }
}

#[test]
fn test_spi_full_duplex_transfer() {
    let mut spi = MockSpi::new();
    spi.init(SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 1_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 8,
        cs_active_low: true,
        timeout_us: 5000,
    })
    .unwrap();

    spi.set_miso_response(&[0xDE, 0xAD, 0xBE, 0xEF]);

    let tx = [0x01, 0x02, 0x03, 0x04];
    let mut rx = [0u8; 4];
    spi.transfer(0, &tx, &mut rx).unwrap();

    assert_eq!(rx, [0xDE, 0xAD, 0xBE, 0xEF]);
    assert_eq!(spi.last_mosi, tx);
    assert_eq!(spi.transfer_count, 1);
}

#[test]
fn test_spi_write_only() {
    let mut spi = MockSpi::new();
    spi.init(SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 4_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 8,
        cs_active_low: true,
        timeout_us: 2000,
    })
    .unwrap();

    let cmd = [0x06]; // SPI NOR write enable
    spi.write(0, &cmd).unwrap();
    assert_eq!(spi.last_mosi, cmd);
}

#[test]
fn test_spi_read_only() {
    let mut spi = MockSpi::new();
    spi.init(SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 1_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 8,
        cs_active_low: true,
        timeout_us: 5000,
    })
    .unwrap();

    spi.set_miso_response(&[0x9F, 0xEF, 0x40, 0x18]); // JEDEC ID response
    let mut buf = [0u8; 4];
    spi.read(0, &mut buf).unwrap();
    assert_eq!(buf, [0x9F, 0xEF, 0x40, 0x18]);
}

#[test]
fn test_spi_chip_select_management() {
    let mut spi = MockSpi::new();
    spi.init(SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 1_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 8,
        cs_active_low: true,
        timeout_us: 1000,
    })
    .unwrap();

    // Assert CS0.
    spi.cs_assert(0).unwrap();
    assert!(spi.cs_asserted[0]);
    assert!(!spi.cs_asserted[1]);

    // Assert CS1 too.
    spi.cs_assert(1).unwrap();
    assert!(spi.cs_asserted[0]);
    assert!(spi.cs_asserted[1]);

    // Deassert CS0.
    spi.cs_deassert(0).unwrap();
    assert!(!spi.cs_asserted[0]);
    assert!(spi.cs_asserted[1]);

    // Invalid CS should error.
    assert_eq!(spi.cs_assert(4), Err(HalError::InvalidConfig));
}

#[test]
fn test_spi_invalid_word_size() {
    let mut spi = MockSpi::new();
    let config = SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 1_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 12, // Invalid
        cs_active_low: true,
        timeout_us: 1000,
    };
    assert_eq!(spi.init(config), Err(HalError::InvalidConfig));
}

#[test]
fn test_spi_transfer_uninitialized() {
    let mut spi = MockSpi::new();
    let mut rx = [0u8; 2];
    assert_eq!(spi.transfer(0, &[0x00], &mut rx), Err(HalError::InitFailed));
}

#[test]
fn test_spi_dma_transfer() {
    let mut spi = MockSpi::new();
    spi.init(SpiConfig {
        mode: SpiMode::Mode0,
        clock_hz: 20_000_000,
        bit_order: SpiBitOrder::MsbFirst,
        word_size: 8,
        cs_active_low: true,
        timeout_us: 10_000,
    })
    .unwrap();

    let token = spi.transfer_dma(0, 0x1000_0000, 0x2000_0000, 4096).unwrap();
    assert!(token.complete);
    assert!(!token.error);
    assert_eq!(token.channel, 0);
}

#[test]
fn test_spi_reset() {
    let mut spi = MockSpi::new();
    spi.init(SpiConfig {
        mode: SpiMode::Mode3,
        clock_hz: 10_000_000,
        bit_order: SpiBitOrder::LsbFirst,
        word_size: 16,
        cs_active_low: true,
        timeout_us: 3000,
    })
    .unwrap();

    spi.cs_assert(0).unwrap();
    assert!(spi.initialized);

    spi.reset().unwrap();
    assert!(!spi.initialized);
    assert!(spi.config.is_none());
    assert!(!spi.cs_asserted[0]);
    assert_eq!(spi.transfer_count, 0);
}

#[test]
fn test_spi_peripheral_type() {
    assert_eq!(PeripheralType::from_u8(1), Some(PeripheralType::Spi));
    assert_eq!(PeripheralType::Spi as u8, 1);
}
