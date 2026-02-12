// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SiFive UART controller driver. Reference: FU740-C000 Manual, Chapter 18.

#![allow(dead_code)]

use smallaios_kernel::hal::{
    HalError, UartConfig, UartController, UartIrqSource, UartParity, UartStopBits,
};

const TXDATA: u32 = 0x00;
const RXDATA: u32 = 0x04;
const TXCTRL: u32 = 0x08;
const RXCTRL: u32 = 0x0C;
const IE: u32 = 0x10;
const IP: u32 = 0x14;
const DIV: u32 = 0x18;

const TXDATA_FULL: u32 = 1 << 31;
const RXDATA_EMPTY: u32 = 1 << 31;
const TXCTRL_TXEN: u32 = 1 << 0;
const TXCTRL_NSTOP: u32 = 1 << 1;
const RXCTRL_RXEN: u32 = 1 << 0;
const IE_TXWM: u32 = 1 << 0;
const IE_RXWM: u32 = 1 << 1;
const IP_TXWM: u32 = 1 << 0;
const IP_RXWM: u32 = 1 << 1;

pub struct SiFiveUart {
    base_addr: u64,
    irq: u32,
    configured: bool,
    config: Option<UartConfig>,
    input_clock_hz: u32,
}

impl SiFiveUart {
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: 500_000_000,
        }
    }

    pub fn with_clock(base_addr: u64, irq: u32, clock_hz: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: clock_hz,
        }
    }

    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    fn compute_div(clock_hz: u32, baud: u32) -> u32 {
        if baud == 0 {
            return 0;
        }
        (clock_hz / baud).saturating_sub(1)
    }
}

impl UartController for SiFiveUart {
    fn init(&mut self, config: UartConfig) -> Result<(), HalError> {
        if !crate::uart::validate_baud(config.baud_rate) {
            return Err(HalError::InvalidConfig);
        }
        if config.parity != UartParity::None {
            return Err(HalError::NotSupported);
        }
        if config.data_bits != 8 {
            return Err(HalError::NotSupported);
        }

        let div = Self::compute_div(self.input_clock_hz, config.baud_rate);
        self.write_reg(DIV, div)?;

        let mut txctrl = TXCTRL_TXEN;
        if config.stop_bits == UartStopBits::Two {
            txctrl |= TXCTRL_NSTOP;
        }
        self.write_reg(TXCTRL, txctrl)?;
        self.write_reg(RXCTRL, RXCTRL_RXEN)?;
        self.write_reg(IE, 0)?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        for &byte in data {
            self.write_reg(TXDATA, byte as u32)?;
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
        for byte in buf.iter_mut() {
            let val = self.read_reg(RXDATA)?;
            *byte = (val & 0xFF) as u8;
        }
        Ok(buf.len())
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<usize, HalError> {
        self.read(buf)
    }

    fn read_until(&mut self, delimiter: u8, buf: &mut [u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        for (i, byte) in buf.iter_mut().enumerate() {
            let val = self.read_reg(RXDATA)?;
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
        Err(HalError::MmioError)
    }

    fn rx_available(&self) -> usize {
        0
    }

    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError> {
        let ip = self.read_reg(IP)?;
        if ip & IP_RXWM != 0 {
            return Ok(UartIrqSource::RxDataAvailable);
        }
        if ip & IP_TXWM != 0 {
            return Ok(UartIrqSource::TxEmpty);
        }
        Err(HalError::InterruptError)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(TXCTRL, 0)?;
        self.write_reg(RXCTRL, 0)?;
        self.write_reg(IE, 0)?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sifive_uart_new() {
        let drv = SiFiveUart::new(0x1001_0000, 39);
        assert_eq!(drv.base_addr(), 0x1001_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_compute_div() {
        assert_eq!(
            SiFiveUart::compute_div(500_000_000, 115200),
            500_000_000 / 115200 - 1
        );
    }

    #[test]
    fn test_init_parity_not_supported() {
        let mut drv = SiFiveUart::new(0x1001_0000, 39);
        let config = UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::Even,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::NotSupported));
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = SiFiveUart::new(0x1001_0000, 39);
        assert_eq!(drv.write(&[0x41]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_compute_div_zero() {
        assert_eq!(SiFiveUart::compute_div(500_000_000, 0), 0);
    }
}
