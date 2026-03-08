// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ARM PL011 UART controller driver (DDI0183).

#![allow(dead_code)]

use smallaios_kernel::hal::{
    HalError, UartConfig, UartController, UartIrqSource, UartParity, UartStopBits,
};

const DR: u32 = 0x000;
const RSR_ECR: u32 = 0x004;
const FR: u32 = 0x018;
const IBRD: u32 = 0x024;
const FBRD: u32 = 0x028;
const LCR_H: u32 = 0x02C;
const CR: u32 = 0x030;
const IFLS: u32 = 0x034;
const IMSC: u32 = 0x038;
const RIS: u32 = 0x03C;
const MIS: u32 = 0x040;
const ICR: u32 = 0x044;
const DMACR: u32 = 0x048;

const FR_TXFF: u32 = 1 << 5;
const FR_RXFE: u32 = 1 << 4;
const FR_BUSY: u32 = 1 << 3;
const CR_UARTEN: u32 = 1 << 0;
const CR_TXE: u32 = 1 << 8;
const CR_RXE: u32 = 1 << 9;
const CR_RTSEN: u32 = 1 << 14;
const CR_CTSEN: u32 = 1 << 15;
const LCR_H_FEN: u32 = 1 << 4;
const LCR_H_WLEN_8: u32 = 3 << 5;
const LCR_H_WLEN_7: u32 = 2 << 5;
const LCR_H_STP2: u32 = 1 << 3;
const LCR_H_EPS: u32 = 1 << 2;
const LCR_H_PEN: u32 = 1 << 1;
const INT_RX: u32 = 1 << 4;
const INT_TX: u32 = 1 << 5;
const INT_RT: u32 = 1 << 6;
const INT_FE: u32 = 1 << 7;
const INT_PE: u32 = 1 << 8;
const INT_BE: u32 = 1 << 9;
const INT_OE: u32 = 1 << 10;

pub struct Pl011Uart {
    base_addr: u64,
    irq: u32,
    configured: bool,
    config: Option<UartConfig>,
    input_clock_hz: u32,
    rx_count: usize,
}

impl Pl011Uart {
    pub fn new(base_addr: u64, irq: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: 48_000_000,
            rx_count: 0,
        }
    }

    pub fn with_clock(base_addr: u64, irq: u32, clock_hz: u32) -> Self {
        Self {
            base_addr,
            irq,
            configured: false,
            config: None,
            input_clock_hz: clock_hz,
            rx_count: 0,
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

    fn build_lcr_h(config: &UartConfig) -> u32 {
        let mut lcr = LCR_H_FEN;
        lcr |= if config.data_bits == 7 {
            LCR_H_WLEN_7
        } else {
            LCR_H_WLEN_8
        };
        if config.stop_bits == UartStopBits::Two {
            lcr |= LCR_H_STP2;
        }
        match config.parity {
            UartParity::Even => {
                lcr |= LCR_H_PEN | LCR_H_EPS;
            }
            UartParity::Odd => {
                lcr |= LCR_H_PEN;
            }
            UartParity::None => {}
        }
        lcr
    }
}

impl UartController for Pl011Uart {
    fn init(&mut self, config: UartConfig) -> Result<(), HalError> {
        if !crate::uart::validate_baud(config.baud_rate) {
            return Err(HalError::InvalidConfig);
        }
        self.write_reg(CR, 0)?;
        let (ibrd, fbrd) = crate::uart::compute_baud_divisor(self.input_clock_hz, config.baud_rate);
        self.write_reg(IBRD, ibrd as u32)?;
        self.write_reg(FBRD, fbrd as u32)?;
        self.write_reg(LCR_H, Self::build_lcr_h(&config))?;
        self.write_reg(IMSC, 0)?;
        self.write_reg(ICR, 0x7FF)?;
        let mut cr = CR_UARTEN | CR_TXE | CR_RXE;
        if config.flow_control == smallaios_kernel::hal::UartFlowControl::RtsCts {
            cr |= CR_RTSEN | CR_CTSEN;
        }
        self.write_reg(CR, cr)?;
        self.config = Some(config);
        self.configured = true;
        Ok(())
    }

    fn write(&mut self, data: &[u8]) -> Result<usize, HalError> {
        if !self.configured {
            return Err(HalError::InitFailed);
        }
        for &byte in data {
            self.write_reg(DR, byte as u32)?;
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
            let val = self.read_reg(DR)?;
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
            let val = self.read_reg(DR)?;
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
        let fr = self.read_reg(FR)?;
        if fr & FR_BUSY != 0 {
            return Err(HalError::Timeout);
        }
        Ok(())
    }

    fn rx_available(&self) -> usize {
        self.rx_count
    }

    fn irq_handler(&mut self) -> Result<UartIrqSource, HalError> {
        let mis = self.read_reg(MIS)?;
        self.write_reg(ICR, mis)?;
        if mis & INT_RX != 0 {
            return Ok(UartIrqSource::RxDataAvailable);
        }
        if mis & INT_RT != 0 {
            return Ok(UartIrqSource::RxTimeout);
        }
        if mis & INT_TX != 0 {
            return Ok(UartIrqSource::TxEmpty);
        }
        if mis & INT_OE != 0 {
            return Ok(UartIrqSource::Overrun);
        }
        if mis & INT_FE != 0 {
            return Ok(UartIrqSource::FramingError);
        }
        if mis & INT_PE != 0 {
            return Ok(UartIrqSource::ParityError);
        }
        if mis & INT_BE != 0 {
            return Ok(UartIrqSource::Break);
        }
        Err(HalError::InterruptError)
    }

    fn reset(&mut self) -> Result<(), HalError> {
        self.write_reg(CR, 0)?;
        self.write_reg(IMSC, 0)?;
        self.write_reg(ICR, 0x7FF)?;
        self.configured = false;
        self.config = None;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pl011_new() {
        let drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.base_addr(), 0x0900_0000);
        assert!(!drv.is_configured());
    }

    #[test]
    fn test_build_lcr_h_8n1() {
        let config = UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        assert_ne!(lcr & LCR_H_FEN, 0);
        assert_ne!(lcr & LCR_H_WLEN_8, 0);
        assert_eq!(lcr & LCR_H_PEN, 0);
        assert_eq!(lcr & LCR_H_STP2, 0);
    }

    #[test]
    fn test_build_lcr_h_7e2() {
        let config = UartConfig {
            baud_rate: 9600,
            data_bits: 7,
            parity: UartParity::Even,
            stop_bits: UartStopBits::Two,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        assert_ne!(lcr & LCR_H_PEN, 0);
        assert_ne!(lcr & LCR_H_EPS, 0);
        assert_ne!(lcr & LCR_H_STP2, 0);
    }

    #[test]
    fn test_write_not_configured() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.write(&[0x41]), Err(HalError::InitFailed));
    }

    #[test]
    fn test_init_invalid_baud() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
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

    // --- with_clock constructor ---

    #[test]
    fn test_with_clock_constructor() {
        let drv = Pl011Uart::with_clock(0x1000_0000, 5, 24_000_000);
        assert_eq!(drv.base_addr(), 0x1000_0000);
        assert_eq!(drv.irq, 5);
        assert_eq!(drv.input_clock_hz, 24_000_000);
        assert!(!drv.is_configured());
        assert!(drv.config.is_none());
        assert_eq!(drv.rx_count, 0);
    }

    #[test]
    fn test_new_defaults() {
        let drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.input_clock_hz, 48_000_000);
        assert_eq!(drv.rx_count, 0);
        assert!(drv.config.is_none());
    }

    // --- build_lcr_h combinations ---

    #[test]
    fn test_build_lcr_h_8o1() {
        let config = UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::Odd,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        assert_ne!(lcr & LCR_H_WLEN_8, 0);
        assert_ne!(lcr & LCR_H_PEN, 0);
        assert_eq!(lcr & LCR_H_EPS, 0); // odd = PEN without EPS
        assert_eq!(lcr & LCR_H_STP2, 0);
        assert_ne!(lcr & LCR_H_FEN, 0); // FIFO always enabled
    }

    #[test]
    fn test_build_lcr_h_8e2() {
        let config = UartConfig {
            baud_rate: 9600,
            data_bits: 8,
            parity: UartParity::Even,
            stop_bits: UartStopBits::Two,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        assert_ne!(lcr & LCR_H_WLEN_8, 0);
        assert_ne!(lcr & LCR_H_PEN, 0);
        assert_ne!(lcr & LCR_H_EPS, 0);
        assert_ne!(lcr & LCR_H_STP2, 0);
    }

    #[test]
    fn test_build_lcr_h_7n1() {
        let config = UartConfig {
            baud_rate: 9600,
            data_bits: 7,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        // 7-bit word length
        assert_eq!(lcr & (3 << 5), LCR_H_WLEN_7);
        assert_eq!(lcr & LCR_H_PEN, 0);
        assert_eq!(lcr & LCR_H_STP2, 0);
    }

    #[test]
    fn test_build_lcr_h_7o2() {
        let config = UartConfig {
            baud_rate: 9600,
            data_bits: 7,
            parity: UartParity::Odd,
            stop_bits: UartStopBits::Two,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        assert_eq!(lcr & (3 << 5), LCR_H_WLEN_7);
        assert_ne!(lcr & LCR_H_PEN, 0);
        assert_eq!(lcr & LCR_H_EPS, 0);
        assert_ne!(lcr & LCR_H_STP2, 0);
    }

    #[test]
    fn test_build_lcr_h_8n2() {
        let config = UartConfig {
            baud_rate: 9600,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::Two,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        let lcr = Pl011Uart::build_lcr_h(&config);
        assert_ne!(lcr & LCR_H_WLEN_8, 0);
        assert_eq!(lcr & LCR_H_PEN, 0);
        assert_ne!(lcr & LCR_H_STP2, 0);
    }

    #[test]
    fn test_build_lcr_h_fifo_always_enabled() {
        // Verify FEN is set regardless of config
        for parity in [UartParity::None, UartParity::Even, UartParity::Odd] {
            let config = UartConfig {
                baud_rate: 115200,
                data_bits: 8,
                parity,
                stop_bits: UartStopBits::One,
                flow_control: smallaios_kernel::hal::UartFlowControl::None,
                rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
                timeout_us: 1000,
            };
            let lcr = Pl011Uart::build_lcr_h(&config);
            assert_ne!(
                lcr & LCR_H_FEN,
                0,
                "FEN should always be set for parity {:?}",
                parity
            );
        }
    }

    // --- not-configured error checks ---

    #[test]
    fn test_read_not_configured() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read(&mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_exact_not_configured() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read_exact(&mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_read_until_not_configured() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        let mut buf = [0u8; 4];
        assert_eq!(drv.read_until(b'\n', &mut buf), Err(HalError::InitFailed));
    }

    #[test]
    fn test_flush_not_configured() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.flush(), Err(HalError::InitFailed));
    }

    #[test]
    fn test_write_all_not_configured() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.write_all(&[0x41]), Err(HalError::InitFailed));
    }

    // --- invalid baud edge cases ---

    #[test]
    fn test_init_invalid_baud_too_low() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
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
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
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

    // --- init fails on MMIO ---

    #[test]
    fn test_init_fails_on_mmio() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        let config = UartConfig {
            baud_rate: 115200,
            data_bits: 8,
            parity: UartParity::None,
            stop_bits: UartStopBits::One,
            flow_control: smallaios_kernel::hal::UartFlowControl::None,
            rx_mode: smallaios_kernel::hal::UartRxMode::Polling,
            timeout_us: 1000,
        };
        assert_eq!(drv.init(config), Err(HalError::MmioError));
        assert!(!drv.is_configured());
    }

    // --- rx_available ---

    #[test]
    fn test_rx_available_default() {
        let drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.rx_available(), 0);
    }

    // --- reset / irq_handler fail on MMIO ---

    #[test]
    fn test_reset_fails_on_mmio() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.reset(), Err(HalError::MmioError));
    }

    #[test]
    fn test_irq_handler_fails_on_mmio() {
        let mut drv = Pl011Uart::new(0x0900_0000, 33);
        assert_eq!(drv.irq_handler(), Err(HalError::MmioError));
    }

    // --- IBRD/FBRD baud divisor calculations (via the module helper) ---

    #[test]
    fn test_baud_divisor_48mhz_115200() {
        let (ibrd, fbrd) = crate::uart::compute_baud_divisor(48_000_000, 115200);
        // 48M / (16 * 115200) = 26.0416... => IBRD=26
        assert_eq!(ibrd, 26);
        assert!(fbrd < 64); // fractional part is 6-bit
    }

    #[test]
    fn test_baud_divisor_48mhz_9600() {
        let (ibrd, _fbrd) = crate::uart::compute_baud_divisor(48_000_000, 9600);
        // 48M / (16 * 9600) = 312.5 => IBRD=312
        assert_eq!(ibrd, 312);
    }

    #[test]
    fn test_baud_divisor_24mhz_115200() {
        let (ibrd, _fbrd) = crate::uart::compute_baud_divisor(24_000_000, 115200);
        // 24M / (16 * 115200) = 13.02 => IBRD=13
        assert_eq!(ibrd, 13);
    }

    #[test]
    fn test_baud_divisor_zero_baud() {
        let (ibrd, fbrd) = crate::uart::compute_baud_divisor(48_000_000, 0);
        assert_eq!(ibrd, 1);
        assert_eq!(fbrd, 0);
    }

    #[test]
    fn test_baud_divisor_zero_clock() {
        let (ibrd, fbrd) = crate::uart::compute_baud_divisor(0, 115200);
        assert_eq!(ibrd, 1);
        assert_eq!(fbrd, 0);
    }
}
