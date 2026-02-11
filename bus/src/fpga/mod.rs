// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SoC FPGA platform support.
//!
//! Provides AXI bus drivers, DMA engine, DTB-based peripheral discovery,
//! interrupt routing, and platform support packages for Xilinx Zynq
//! UltraScale+ and Microchip PolarFire SoC.
//!
//! FPGA peripherals are treated as standard MMIO devices. No bitstream
//! programming or partial reconfiguration is supported.

pub mod axi_lite;
pub mod axi_full;
pub mod dma;
pub mod discovery;
pub mod interrupt;
pub mod zynq;
pub mod polarfire;
