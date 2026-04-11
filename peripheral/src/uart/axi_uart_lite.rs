// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Xilinx AXI UART Lite FPGA driver (PG142). Generic over FpgaFabric.

#![allow(dead_code)]

use smallaios_kernel::hal::{FpgaFabric, HalError, UartConfig, UartController, UartIrqSource};

const RX_FIFO: u32 = 0x00;
const TX_FIFO: u32 = 0x04;
const STAT_REG: u32 = 0x08;
const CTRL_REG: u32 = 0x0C;

const STAT_RX_VALID: u32 = 1 << 0;
const STAT_RX_FULL: u32 = 1 << 1;
const STAT_TX_EMPTY: u32 = 1 << 2;
const STAT_TX_FULL: u32 = 1 << 3;
const STAT_INTR: u32 = 1 << 4;
const STAT_OVERRUN: u32 = 1 << 5;
const STAT_FRAME_ERR: u32 = 1 << 6;
const STAT_PARITY_ERR: u32 = 1 << 7;
const CTRL_RST_TX: u32 = 1 << 0;
const CTRL_RST_RX: u32 = 1 << 1;
const CTRL_INTR_EN: u32 = 1 << 4;

pub struct AxiUartLite<F: FpgaFabric> {
    fabric: F,
    base_addr: u64,
    configured: bool,
}

impl<F: FpgaFabric> AxiUartLite<F> {
    pub fn new(fabric: F, base_addr: u64) -> Self {
        Self {
            fabric,
            base_addr,
            configured: false,
        }
    }
    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    fn read_reg(&self, offset: u32) -> Result<u32, HalError> {
        self.fabric.read_reg(self.base_addr, offset)
    }
    fn write_reg(&mut self, offset: u32, value: u32) -> Result<(), HalError> {
        self.fabric.write_reg(self.base_addr, offset, value)
    }
}

impl<F: FpgaFabric> UartController for AxiUartLite<F> {
    fn init(&mut self, config: UartConfig) -> Result<(), HalError> {
        if !crate::uart::validate_baud(config.baud_rate) {
            return Err(HalError::InvalidConfig);
        }
        // AXI UART Lite baud rate is set at synthesis time. Just reset FIFOs.
        self.write_reg(CTRL_REG, CTRL_RST_TX | CTRL_RST_RX)?;
        self.write_reg(CTRL_REG, 0)?;
        self.configured = true;
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        for &byte in data {
            let stat = self.read_reg(STAT_REG)?;
            if stat & STAT_TX_FULL != 0 {
                return Err(HalError::TxFifoFull);
            }
            self.write_reg(TX_FIFO, byte as u32)?;
        }
        Ok(data.len())
    }

    fn write_all(&mut self, data: &[u8]) -> Result<(), HalError> {
        self.write(data)?;
        Ok(())
    }

    fn read(&mut self, buf: &mut [u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let mut count = 0;
        for byte in buf.iter_mut() {
            let stat = self.read_reg(STAT_REG)?;
            if stat & STAT_RX_VALID == 0 {
                break;
            }
            let val = self.read_reg(RX_FIFO)?;
            *byte = (val & 0xFF) as u8;
            count += 1;
        }
        Ok(count)
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<usize, HalError> {
        self.read(buf)
    }

    fn read_until(&mut self, delimiter: u8, buf: &mut [u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        for (i, byte) in buf.iter_mut().enumerate() {
            let stat = self.read_reg(STAT_REG)?;
            if stat & STAT_RX_VALID == 0 {
                return Ok(i);
            }
            let val = self.read_reg(RX_FIFO)?;
            *byte = (val & 0xFF) as u8;
            if *byte == delimiter {
                return Ok(i + 1);
            }
        }
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        let stat = self.read_reg(STAT_REG)?;
        if stat & STAT_TX_EMPTY != 0 {
            Ok(())
        } else {
            Err(HalError::Timeout)
        }
    }

    fn rx_available(&self) -> usize {
        if let Ok(stat) = self.read_reg(STAT_REG) {
            if stat & STAT_RX_VALID != 0 {
                1
            } else {
                0
            }
        } else {
            0
        }
    }

    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError> {
        let stat = self.read_reg(STAT_REG)?;
        if stat & STAT_OVERRUN != 0 {
            return Ok(UartIrqSource::Overrun);
        }
        if stat & STAT_FRAME_ERR != 0 {
            return Ok(UartIrqSource::FramingError);
        }
        if stat & STAT_PARITY_ERR != 0 {
            return Ok(UartIrqSource::ParityError);
        }
        if stat & STAT_RX_VALID != 0 {
            return Ok(UartIrqSource::RxDataAvailable);
        }
        if stat & STAT_TX_EMPTY != 0 {
            return Ok(UartIrqSource::TxEmpty);
        }
        Err(HalError::InterruptError)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(CTRL_REG, CTRL_RST_TX | CTRL_RST_RX)?;
        self.write_reg(CTRL_REG, 0)?;
        self.configured = false;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use smallaios_kernel::hal::{DmaDescriptor, DmaToken, UartParity, UartStopBits};

    struct MockFpgaFabric {
        regs: [u32; 16],
    }
    impl MockFpgaFabric {
        fn new() -> Self {
            Self { regs: [0u32; 16] }
        }
    }
    impl FpgaFabric for MockFpgaFabric {
        fn read_reg(&self, _b: u64, offset: u32) -> Result<u32, HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                Ok(self.regs[idx])
            } else {
                Err(HalError::OutOfRange)
            }
        }
        fn write_reg(&mut self, _b: u64, offset: u32, value: u32) -> Result<(), HalError> {
            let idx = (offset / 4) as usize;
            if idx < self.regs.len() {
                self.regs[idx] = value;
                Ok(())
            } else {
                Err(HalError::OutOfRange)
            }
        }
        fn dma_start(&mut self, _d: DmaDescriptor) -> Result<DmaToken, HalError> {
            Err(HalError::NotSupported)
        }
        fn dma_poll(&self, t: DmaToken) -> Result<DmaToken, HalError> {
            Ok(t)
        }
        fn dma_irq_ack(&mut self) -> Result<u8, HalError> {
            Ok(0)
        }
    }

    fn make_config() -> UartConfig {
        UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        }
    }

    #[test]
    fn test_axi_uart_lite_init() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        assert!(drv.init(make_config()).is_ok());
        assert!(drv.is_configured());
    }

    #[test]
    fn test_axi_uart_lite_write_not_configured() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        assert_eq!(drv.write(&[0x41]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_axi_uart_lite_write() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.write(&[0x41, 0x42]).is_ok());
    }

    #[test]
    fn test_axi_uart_lite_reset() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        drv.reset().unwrap();
        assert!(!drv.is_configured());
    }

    // --- constructor field access ---

    #[test]
    fn test_constructor_base_addr() {
        let drv = AxiUartLite::new(MockFpgaFabric::new(), 0x1234_5678);
        assert_eq!(drv.base_addr(), 0x1234_5678);
        assert!(!drv.is_configured());
    }

    // --- not-configured error checks ---

    #[test]
    fn test_read_not_configured() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read(&mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_exact_not_configured() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read_exact(&mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_until_not_configured() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read_until(b'\n', &mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_flush_not_configured() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        assert_eq!(drv.flush(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_write_all_not_configured() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        assert_eq!(drv.write_all(&[0x41]), Err(HalError::InitFailed));
    }

    // --- baud rate validation ---

    #[test]
    fn test_init_invalid_baud_zero() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let config = UartConfig {
            baud_rate: 0,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_baud_too_low() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let config = UartConfig {
            baud_rate: 100,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_invalid_baud_too_high() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let config = UartConfig {
            baud_rate: 5_000_000,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::InvalidConfig));
    }

    #[test]
    fn test_init_valid_baud_boundary_low() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let config = UartConfig {
            baud_rate: 300, // minimum valid
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert!(drv.init(config).is_ok());
    }

    #[test]
    fn test_init_valid_baud_boundary_high() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        let config = UartConfig {
            baud_rate: 4_000_000, // maximum valid
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert!(drv.init(config).is_ok());
    }

    // --- status polling / flush / rx_available with mock fabric ---

    #[test]
    fn test_flush_tx_empty() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        // Set STAT_TX_EMPTY in STAT_REG (offset 0x08, index 2)
        drv.fabric.regs[2] = STAT_TX_EMPTY;
        assert!(drv.flush().is_ok());
    }

    #[test]
    fn test_flush_tx_not_empty() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        // STAT_REG = 0 means TX not empty
        drv.fabric.regs[2] = 0;
        assert_eq!(drv.flush(), Err(HalError::Timeout));
    }

    #[test]
    fn test_rx_available_with_data() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        // Set STAT_RX_VALID in STAT_REG
        drv.fabric.regs[2] = STAT_RX_VALID;
        assert_eq!(drv.rx_available(), 1);
    }

    #[test]
    fn test_rx_available_no_data() {
        let drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        assert_eq!(drv.rx_available(), 0);
    }

    // --- write with TX_FULL status ---

    #[test]
    fn test_write_tx_full() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        // Set STAT_TX_FULL
        drv.fabric.regs[2] = STAT_TX_FULL;
        assert_eq!(drv.write(&[0x41]), Err(HalError::TxFifoFull));
    }

    // --- read with RX data available ---

    #[test]
    fn test_read_with_data() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        // Set RX_VALID in status and data in RX_FIFO
        drv.fabric.regs[2] = STAT_RX_VALID;
        drv.fabric.regs[0] = 0x42; // RX_FIFO at offset 0x00, index 0
        let mut buf = [0u8; 1];
        let count = drv.read(&mut buf).unwrap();
        assert_eq!(count, 1);
        assert_eq!(buf[0], 0x42);
    }

    #[test]
    fn test_read_no_data() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        // STAT_RX_VALID not set
        drv.fabric.regs[2] = 0;
        let mut buf = [0u8; 4];
        let count = drv.read(&mut buf).unwrap();
        assert_eq!(count, 0);
    }

    // --- irq_handler with mock status ---

    #[test]
    fn test_irq_handler_overrun() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.fabric.regs[2] = STAT_OVERRUN;
        assert_eq!(drv.irq_handler(), Ok(UartIrqSource::Overrun));
    }

    #[test]
    fn test_irq_handler_frame_error() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.fabric.regs[2] = STAT_FRAME_ERR;
        assert_eq!(drv.irq_handler(), Ok(UartIrqSource::FramingError));
    }

    #[test]
    fn test_irq_handler_parity_error() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.fabric.regs[2] = STAT_PARITY_ERR;
        assert_eq!(drv.irq_handler(), Ok(UartIrqSource::ParityError));
    }

    #[test]
    fn test_irq_handler_rx_data_available() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.fabric.regs[2] = STAT_RX_VALID;
        assert_eq!(drv.irq_handler(), Ok(UartIrqSource::RxDataAvailable));
    }

    #[test]
    fn test_irq_handler_tx_empty() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.fabric.regs[2] = STAT_TX_EMPTY;
        assert_eq!(drv.irq_handler(), Ok(UartIrqSource::TxEmpty));
    }

    #[test]
    fn test_irq_handler_no_interrupt() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.fabric.regs[2] = 0;
        assert_eq!(drv.irq_handler(), Err(HalError::InterruptError));
    }

    // --- write_all after init ---

    #[test]
    fn test_write_all_after_init() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.write_all(&[0x41, 0x42, 0x43]).is_ok());
    }

    // --- read_until with delimiter ---

    #[test]
    fn test_read_until_no_data() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        drv.fabric.regs[2] = 0; // no RX_VALID
        let mut buf = [0u8; 4];
        let count = drv.read_until(b'\n', &mut buf).unwrap();
        assert_eq!(count, 0);
    }

    // --- double init ---

    #[test]
    fn test_double_init() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        assert!(drv.is_configured());
        drv.init(make_config()).unwrap();
        assert!(drv.is_configured());
    }

    // --- reset then write fails ---

    #[test]
    fn test_reset_then_write_fails() {
        let mut drv = AxiUartLite::new(MockFpgaFabric::new(), 0x4060_0000);
        drv.init(make_config()).unwrap();
        drv.reset().unwrap();
        assert_eq!(drv.write(&[0x41]), Err(HalError::InitFailed));
    }
}
