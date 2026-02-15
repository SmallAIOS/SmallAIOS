// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Tegra T210 GPU HAL — SoC-integrated GM20B (Maxwell, CC 5.3).
//!
//! This module provides bare-metal GPU initialization for the NVIDIA Tegra X1
//! SoC as found in the Jetson Nano. Unlike discrete PCIe GPUs, the GM20B is
//! memory-mapped at fixed addresses and shares system DRAM.
//!
//! # Initialization sequence
//!
//! 1. **Power** — Enable VDD_GPU regulator, remove PMC rail clamp
//! 2. **Clocks** — Enable GPU clock via CAR, configure GPCPLL
//! 3. **Interrupts** — Enable GICv2 SPI 189/190
//! 4. **Firmware** — Load ACR, FECS, GPCCS via Falcon DMA
//! 5. **Engines** — Initialize GR, FIFO, GMMU
//!
//! # Licensing
//!
//! All code in this module is Apache-2.0. Register definitions in `regs.rs`
//! document their provenance from the MIT-licensed nvgpu driver.
//! See `LICENSES/MIT-nvgpu.txt` for the MIT license text.

#![allow(dead_code)]

pub mod clock;
pub mod falcon;
pub mod fifo;
pub mod gmmu;
pub mod gr;
pub mod power;
pub mod regs;

/// Top-level Tegra GPU state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TegraGpuState {
    /// GPU is completely off.
    Off,
    /// Power domain enabled, clocks running.
    PlatformReady,
    /// Firmware loaded, engines initialized.
    Ready,
    /// GPU encountered an error.
    Error,
}

/// Tegra T210 GPU context — orchestrates the full initialization sequence.
pub struct TegraGpu {
    state: TegraGpuState,
    bar0_base: u64,
}

impl TegraGpu {
    /// Create a new Tegra GPU context.
    ///
    /// `bar0_base` is the GPU register base (0x5700_0000 on Tegra X1).
    pub fn new(bar0_base: u64) -> Self {
        Self {
            state: TegraGpuState::Off,
            bar0_base,
        }
    }

    /// Current GPU state.
    pub fn state(&self) -> TegraGpuState {
        self.state
    }

    /// GPU BAR0 base address.
    pub fn bar0_base(&self) -> u64 {
        self.bar0_base
    }

    /// Returns true if the GPU is fully initialized and ready for compute.
    pub fn is_ready(&self) -> bool {
        self.state == TegraGpuState::Ready
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_tegra_gpu_starts_off() {
        let gpu = TegraGpu::new(0x5700_0000);
        assert_eq!(gpu.state(), TegraGpuState::Off);
        assert!(!gpu.is_ready());
    }

    #[test]
    fn bar0_base_is_stored() {
        let gpu = TegraGpu::new(0x5700_0000);
        assert_eq!(gpu.bar0_base(), 0x5700_0000);
    }
}
