// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! NS16550A UART controller driver.

#![allow(dead_code)]

use smallaios_kernel::hal::{
    HalError, UartConfig, UartController, UartIrqSource, UartParity, UartStopBits,
};

const RBR_THR: u32 = 0x00;
const IER: u32 = 0x01;
const IIR_FCR: u32 = 0x02;
const LCR: u32 = 0x03;
const MCR: u32 = 0x04;
const LSR: u32 = 0x05;
const MSR: u32 = 0x06;
const SCR: u32 = 0x07;
const DLL: u32 = 0x00; // When DLAB=1
const DLM: u32 = 0x01; // When DLAB=1

const LCR_DLAB: u32 = 1 << 7;
const LCR_BREAK: u32 = 1 << 6;
const LCR_PARITY_MASK: u32 = 0x38;
const LCR_EPS: u32 = 1 << 4;
const LCR_PEN: u32 = 1 << 3;
const LCR_STOP: u32 = 1 << 2;
const LCR_WLS_8: u32 = 0x03;
const LCR_WLS_7: u32 = 0x02;

const LSR_DR: u32 = 1 << 0;
const LSR_OE: u32 = 1 << 1;
const LSR_PE: u32 = 1 << 2;
const LSR_FE: u32 = 1 << 3;
const LSR_BI: u32 = 1 << 4;
const LSR_THRE: u32 = 1 << 5;
const LSR_TEMT: u32 = 1 << 6;

const FCR_ENABLE: u32 = 1 << 0;
const FCR_RX_RESET: u32 = 1 << 1;
const FCR_TX_RESET: u32 = 1 << 2;
const FCR_TRIGGER_14: u32 = 3 << 6;

const IER_RDA: u32 = 1 << 0;
const IER_THRE: u32 = 1 << 1;
const IER_RLS: u32 = 1 << 2;

const MCR_RTS: u32 = 1 << 1;

const IIR_PENDING: u32 = 1 << 0;
const IIR_ID_MASK: u32 = 0x0E;
const IIR_RLS: u32 = 0x06;
const IIR_RDA: u32 = 0x04;
const IIR_CTI: u32 = 0x0C;
const IIR_THRE: u32 = 0x02;

pub struct Ns16550a {
    base_addr: u64,
    irq: u32,
    configured: bool,
    config: Option<UartConfig>,
    input_clock_hz: u32,
    reg_shift: u32,
}

impl Ns16550a {
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: 1_843_200,
            reg_shift: 0,
        }
    }

    pub fn with_clock(base_addr: u64, irq: u32, clock_hz: u32, reg_shift: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: clock_hz,
            reg_shift,
        }
    }

    pub fn base_addr(&self) -> u64 {
        self.base_addr
    }
    pub fn is_configured(&self) -> bool {
        self.configured
    }

    fn reg_addr(&self, offset: u32) -> u32 {
        offset << self.reg_shift
    }
    fn read_reg(&self, _offset: u32) -> Result<u32, HalError> {
        Err(HalError::MmioError)
    }
    fn write_reg(&mut self, _offset: u32, _value: u32) -> Result<(), HalError> {
        Err(HalError::MmioError)
    }

    fn compute_divisor(clock_hz: u32, baud: u32) -> u16 {
        if baud == 0 {
            return 1;
        }
        let div = clock_hz / (16 * baud);
        if div == 0 {
            1
        } else {
            div as u16
        }
    }

    fn build_lcr(config: &UartConfig) -> u32 {
        let mut lcr = if config.data_bits == 7 {
            LCR_WLS_7
        } else {
            LCR_WLS_8
        };
        if config.stop_bits == UartStopBits::Two {
            lcr |= LCR_STOP;
        }
        match config.parity {
            UartParity::Even => {
                lcr |= LCR_PEN | LCR_EPS;
            }
            UartParity::Odd => {
                lcr |= LCR_PEN;
            }
            UartParity::None => {}
        }
        lcr
    }
}

impl UartController for Ns16550a {
    fn init(&mut self, config: UartConfig) -> Result<(), HalError> {
        if !crate::uart::validate_baud(config.baud_rate) {
            return Err(HalError::InvalidConfig);
        }

        self.write_reg(self.reg_addr(IER), 0)?;

        // Set DLAB to program baud divisor.
        self.write_reg(self.reg_addr(LCR), LCR_DLAB)?;
        let div = Self::compute_divisor(self.input_clock_hz, config.baud_rate);
        self.write_reg(self.reg_addr(DLL), (div & 0xFF) as u32)?;
        self.write_reg(self.reg_addr(DLM), ((div >> 8) & 0xFF) as u32)?;

        // Set line control (clears DLAB).
        self.write_reg(self.reg_addr(LCR), Self::build_lcr(&config))?;

        // Enable and reset FIFOs.
        self.write_reg(
            self.reg_addr(IIR_FCR),
            FCR_ENABLE | FCR_RX_RESET | FCR_TX_RESET | FCR_TRIGGER_14,
        )?;

        // Set MCR (RTS if flow control).
        let mcr = if config.flow_control == smallaios_kernel::hal::UartFlowControl::RtsCts {
            MCR_RTS
        } else {
            0
        };
        self.write_reg(self.reg_addr(MCR), mcr)?;

        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        for &byte in data {
            self.write_reg(self.reg_addr(RBR_THR), byte as u32)?;
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
            let val = self.read_reg(self.reg_addr(RBR_THR))?;
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
            let val = self.read_reg(self.reg_addr(RBR_THR))?;
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
        let lsr = self.read_reg(self.reg_addr(LSR))?;
        if lsr & LSR_TEMT == 0 {
            return Err(HalError::Timeout);
        }
        Ok(())
    }

    fn rx_available(&self) -> usize {
        0
    }

    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError> {
        let iir = self.read_reg(self.reg_addr(IIR_FCR))?;
        if iir & IIR_PENDING != 0 {
            return Err(HalError::InterruptError);
        }
        match iir & IIR_ID_MASK {
            IIR_RDA => Ok(UartIrqSource::RxDataAvailable),
            IIR_CTI => Ok(UartIrqSource::RxTimeout),
            IIR_THRE => Ok(UartIrqSource::TxEmpty),
            IIR_RLS => {
                let lsr = self.read_reg(self.reg_addr(LSR))?;
                if lsr & LSR_OE != 0 {
                    return Ok(UartIrqSource::Overrun);
                }
                if lsr & LSR_FE != 0 {
                    return Ok(UartIrqSource::FramingError);
                }
                if lsr & LSR_PE != 0 {
                    return Ok(UartIrqSource::ParityError);
                }
                if lsr & LSR_BI != 0 {
                    return Ok(UartIrqSource::Break);
                }
                Err(HalError::InterruptError)
            }
            _ => Err(HalError::InterruptError),
        }
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(self.reg_addr(IER), 0)?;
        self.write_reg(self.reg_addr(IIR_FCR), 0)?;
        self.write_reg(self.reg_addr(LCR), 0)?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ns16550a_new() {
        let drv = Ns16550a::new(0x10000000, 10);
        assert_eq!(drv.base_addr(), 0x10000000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_compute_divisor_115200() {
        let div = Ns16550a::compute_divisor(1_843_200, 115200);
        assert_eq!(div, 1);
    }

    #[test]
    fn test_compute_divisor_9600() {
        let div = Ns16550a::compute_divisor(1_843_200, 9600);
        assert_eq!(div, 12);
    }

    #[test]
    fn test_build_lcr_8n1() {
        let config = UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Ns16550a::build_lcr(&config);
        assert_eq!(lcr & 0x03, LCR_WLS_8);
        assert_eq!(lcr & LCR_STOP, 0);
        assert_eq!(lcr & LCR_PEN, 0);
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = Ns16550a::new(0x10000000, 10);
        assert_eq!(drv.write(&[0x41]), Err(HalError::InitFailed));
    }
}
