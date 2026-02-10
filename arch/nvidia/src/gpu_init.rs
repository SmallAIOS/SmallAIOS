// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GPU initialization sequence.
//!
//! Orchestrates the bring-up of an NVIDIA GPU after PCIe enumeration:
//! BAR mapping, engine initialisation, power-state management, and
//! suspend/resume lifecycle.

#![allow(dead_code)]

use crate::gpu_id::GpuInfo;
use crate::pcie::{BaseAddressRegister, BarType};
use crate::GpuError;

// ---------------------------------------------------------------------------
// GpuState
// ---------------------------------------------------------------------------

/// Lifecycle state of a GPU context.
#[derive(Clone, Debug, PartialEq)]
pub enum GpuState {
    /// GPU discovered but no BARs mapped yet.
    Uninitialized,
    /// BAR0 / BAR1 mapped into the address space.
    BarsMapped,
    /// Compute and copy engines initialised.
    EnginesReady,
    /// GPU is actively running workloads.
    Running,
    /// An unrecoverable error was encountered.
    Error,
    /// GPU power-gated; can be resumed.
    Suspended,
}

// ---------------------------------------------------------------------------
// PowerState
// ---------------------------------------------------------------------------

/// Requested power level for the GPU.
#[derive(Clone, Debug, PartialEq)]
pub enum PowerState {
    /// Maximum performance, all clocks ungated.
    FullPower,
    /// Reduced clocks / partial power gating.
    LowPower,
    /// Deep sleep — minimal leakage, state saved to system RAM.
    Suspended,
}

// ---------------------------------------------------------------------------
// GpuRegisters
// ---------------------------------------------------------------------------

/// Represents the MMIO register apertures obtained from BAR mapping.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuRegisters {
    /// BAR0 base: GPU control/status registers.
    pub bar0_base: u64,
    /// BAR1 base: VRAM aperture (framebuffer / device memory).
    pub bar1_base: u64,
    /// Size of BAR0 in bytes.
    pub bar0_size: u64,
    /// Size of BAR1 in bytes.
    pub bar1_size: u64,
}

// ---------------------------------------------------------------------------
// InitConfig
// ---------------------------------------------------------------------------

/// Configuration knobs for the GPU initialisation sequence.
#[derive(Clone, Debug, PartialEq)]
pub struct InitConfig {
    /// Enable the compute (GR/CE) engine during init.
    pub enable_compute: bool,
    /// Enable the copy (DMA) engine during init.
    pub enable_copy: bool,
    /// Enable MSI-X / interrupt-based completion notification.
    pub enable_interrupts: bool,
    /// Requested power state after init.
    pub power_state: PowerState,
}

impl InitConfig {
    /// Sensible defaults: everything enabled, full power.
    pub fn default() -> Self {
        Self {
            enable_compute: true,
            enable_copy: true,
            enable_interrupts: true,
            power_state: PowerState::FullPower,
        }
    }
}

// ---------------------------------------------------------------------------
// GpuContext
// ---------------------------------------------------------------------------

/// Top-level handle that tracks the complete state of a single GPU.
#[derive(Clone, Debug, PartialEq)]
pub struct GpuContext {
    /// Static information about the GPU (architecture, SM count, etc.).
    info: GpuInfo,
    /// Current lifecycle state.
    state: GpuState,
    /// Mapped register apertures (populated after `map_bars`).
    registers: Option<GpuRegisters>,
    /// Initialisation configuration.
    config: InitConfig,
}

impl GpuContext {
    /// Create a new context in the [`GpuState::Uninitialized`] state.
    pub fn new(info: GpuInfo) -> Self {
        Self {
            info,
            state: GpuState::Uninitialized,
            registers: None,
            config: InitConfig::default(),
        }
    }

    // -- lifecycle transitions --------------------------------------------

    /// Map BAR0 (registers) and BAR1 (VRAM aperture) from the PCI device's
    /// Base Address Registers.
    ///
    /// Expects at least two memory-type BARs. The first memory BAR becomes
    /// BAR0 and the second becomes BAR1.
    ///
    /// Transitions: `Uninitialized` -> `BarsMapped`.
    pub fn map_bars(&mut self, bars: &[BaseAddressRegister]) -> Result<(), GpuError> {
        if self.state != GpuState::Uninitialized {
            return Err(GpuError::InvalidState);
        }

        let memory_bars: alloc::vec::Vec<&BaseAddressRegister> = bars
            .iter()
            .filter(|b| matches!(b.bar_type, BarType::Memory32 | BarType::Memory64))
            .collect();

        if memory_bars.len() < 2 {
            return Err(GpuError::BarNotFound);
        }

        self.registers = Some(GpuRegisters {
            bar0_base: memory_bars[0].address,
            bar1_base: memory_bars[1].address,
            bar0_size: memory_bars[0].size,
            bar1_size: memory_bars[1].size,
        });

        self.state = GpuState::BarsMapped;
        Ok(())
    }

    /// Initialize compute and copy engines.
    ///
    /// On real hardware this programs PGRAPH, CE, and PFIFO registers;
    /// currently a state-machine stub.
    ///
    /// Transitions: `BarsMapped` -> `EnginesReady`.
    pub fn initialize_engines(&mut self) -> Result<(), GpuError> {
        if self.state != GpuState::BarsMapped {
            return Err(GpuError::InvalidState);
        }

        // Stub: real implementation would:
        // 1. Reset GR engine via PMC_ENABLE
        // 2. Load GR context-switch firmware (FECS/GPCCS)
        // 3. Configure CE for host<->device DMA
        // 4. Set up PFIFO channels

        self.state = GpuState::EnginesReady;
        Ok(())
    }

    /// Transition the GPU to the running state.
    ///
    /// Transitions: `EnginesReady` -> `Running`.
    pub fn start(&mut self) -> Result<(), GpuError> {
        if self.state != GpuState::EnginesReady {
            return Err(GpuError::InvalidState);
        }
        self.state = GpuState::Running;
        Ok(())
    }

    /// Suspend the GPU (power gate, save state).
    ///
    /// Transitions: `Running` -> `Suspended`.
    pub fn suspend(&mut self) -> Result<(), GpuError> {
        if self.state != GpuState::Running {
            return Err(GpuError::InvalidState);
        }
        self.state = GpuState::Suspended;
        Ok(())
    }

    /// Resume a previously suspended GPU.
    ///
    /// Transitions: `Suspended` -> `Running`.
    pub fn resume(&mut self) -> Result<(), GpuError> {
        if self.state != GpuState::Suspended {
            return Err(GpuError::InvalidState);
        }
        self.state = GpuState::Running;
        Ok(())
    }

    /// Hard reset — tear everything down and return to uninitialized.
    ///
    /// Transitions: `*` -> `Uninitialized`.
    pub fn reset(&mut self) {
        self.state = GpuState::Uninitialized;
        self.registers = None;
    }

    // -- accessors --------------------------------------------------------

    /// Current lifecycle state.
    pub fn state(&self) -> &GpuState {
        &self.state
    }

    /// Static GPU information.
    pub fn info(&self) -> &GpuInfo {
        &self.info
    }

    /// Mapped register apertures (if any).
    pub fn registers(&self) -> Option<&GpuRegisters> {
        self.registers.as_ref()
    }

    /// Current init configuration.
    pub fn config(&self) -> &InitConfig {
        &self.config
    }

    /// Replace the init configuration.
    pub fn set_config(&mut self, config: InitConfig) {
        self.config = config;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_id::{ComputeCapability, GpuArchitecture, GpuInfo};
    use crate::pcie::{BarType, BaseAddressRegister};

    /// Helper: create a representative `GpuInfo` for test use.
    fn test_gpu_info() -> GpuInfo {
        GpuInfo {
            device_id: 0x20B0,
            architecture: GpuArchitecture::Ampere,
            compute_capability: ComputeCapability::new(8, 0),
            sm_count: 108,
            vram_size_mb: 40960,
            max_threads_per_sm: 2048,
            max_warps_per_sm: 64,
            warp_size: 32,
            max_shared_memory_per_sm: 164 * 1024,
            max_registers_per_sm: 65536,
            name: "NVIDIA A100-SXM4-40GB",
        }
    }

    /// Helper: return two memory-type BARs (BAR0 + BAR1).
    fn test_bars() -> alloc::vec::Vec<BaseAddressRegister> {
        alloc::vec![
            BaseAddressRegister {
                bar_type: BarType::Memory64,
                address: 0xFC00_0000,
                size: 32 * 1024 * 1024,
                prefetchable: false,
            },
            BaseAddressRegister {
                bar_type: BarType::Memory64,
                address: 0x3900_0000_0000,
                size: 512 * 1024 * 1024,
                prefetchable: true,
            },
        ]
    }

    // -- construction & initial state -------------------------------------

    #[test]
    fn new_context_is_uninitialized() {
        let ctx = GpuContext::new(test_gpu_info());
        assert_eq!(*ctx.state(), GpuState::Uninitialized);
        assert!(ctx.registers().is_none());
    }

    #[test]
    fn new_context_preserves_info() {
        let info = test_gpu_info();
        let ctx = GpuContext::new(info.clone());
        assert_eq!(*ctx.info(), info);
    }

    // -- map_bars ---------------------------------------------------------

    #[test]
    fn map_bars_transitions_to_bars_mapped() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        assert_eq!(*ctx.state(), GpuState::BarsMapped);
        let regs = ctx.registers().unwrap();
        assert_eq!(regs.bar0_base, 0xFC00_0000);
        assert_eq!(regs.bar1_base, 0x3900_0000_0000);
    }

    #[test]
    fn map_bars_no_memory_bars_returns_error() {
        let mut ctx = GpuContext::new(test_gpu_info());
        let io_bars = alloc::vec![
            BaseAddressRegister {
                bar_type: BarType::Io,
                address: 0x1000,
                size: 256,
                prefetchable: false,
            },
        ];
        assert_eq!(ctx.map_bars(&io_bars), Err(GpuError::BarNotFound));
        assert_eq!(*ctx.state(), GpuState::Uninitialized);
    }

    #[test]
    fn map_bars_only_one_memory_bar_returns_error() {
        let mut ctx = GpuContext::new(test_gpu_info());
        let single = alloc::vec![
            BaseAddressRegister {
                bar_type: BarType::Memory32,
                address: 0xF000_0000,
                size: 1024,
                prefetchable: false,
            },
        ];
        assert_eq!(ctx.map_bars(&single), Err(GpuError::BarNotFound));
    }

    #[test]
    fn map_bars_requires_uninitialized_state() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        // Already in BarsMapped — should fail if called again.
        assert_eq!(ctx.map_bars(&test_bars()), Err(GpuError::InvalidState));
    }

    // -- initialize_engines -----------------------------------------------

    #[test]
    fn initialize_engines_requires_bars_mapped() {
        let mut ctx = GpuContext::new(test_gpu_info());
        assert_eq!(ctx.initialize_engines(), Err(GpuError::InvalidState));
    }

    #[test]
    fn initialize_engines_transitions_to_engines_ready() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        ctx.initialize_engines().unwrap();
        assert_eq!(*ctx.state(), GpuState::EnginesReady);
    }

    // -- start ------------------------------------------------------------

    #[test]
    fn start_requires_engines_ready() {
        let mut ctx = GpuContext::new(test_gpu_info());
        assert_eq!(ctx.start(), Err(GpuError::InvalidState));
    }

    #[test]
    fn start_transitions_to_running() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        ctx.initialize_engines().unwrap();
        ctx.start().unwrap();
        assert_eq!(*ctx.state(), GpuState::Running);
    }

    #[test]
    fn cannot_start_from_uninitialized() {
        let mut ctx = GpuContext::new(test_gpu_info());
        assert_eq!(ctx.start(), Err(GpuError::InvalidState));
    }

    // -- suspend / resume -------------------------------------------------

    #[test]
    fn suspend_requires_running() {
        let mut ctx = GpuContext::new(test_gpu_info());
        assert_eq!(ctx.suspend(), Err(GpuError::InvalidState));
    }

    #[test]
    fn suspend_transitions_to_suspended() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        ctx.initialize_engines().unwrap();
        ctx.start().unwrap();
        ctx.suspend().unwrap();
        assert_eq!(*ctx.state(), GpuState::Suspended);
    }

    #[test]
    fn resume_requires_suspended() {
        let mut ctx = GpuContext::new(test_gpu_info());
        assert_eq!(ctx.resume(), Err(GpuError::InvalidState));
    }

    #[test]
    fn resume_transitions_to_running() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        ctx.initialize_engines().unwrap();
        ctx.start().unwrap();
        ctx.suspend().unwrap();
        ctx.resume().unwrap();
        assert_eq!(*ctx.state(), GpuState::Running);
    }

    // -- full lifecycle ---------------------------------------------------

    #[test]
    fn full_lifecycle() {
        let mut ctx = GpuContext::new(test_gpu_info());
        assert_eq!(*ctx.state(), GpuState::Uninitialized);

        ctx.map_bars(&test_bars()).unwrap();
        assert_eq!(*ctx.state(), GpuState::BarsMapped);

        ctx.initialize_engines().unwrap();
        assert_eq!(*ctx.state(), GpuState::EnginesReady);

        ctx.start().unwrap();
        assert_eq!(*ctx.state(), GpuState::Running);

        ctx.suspend().unwrap();
        assert_eq!(*ctx.state(), GpuState::Suspended);

        ctx.resume().unwrap();
        assert_eq!(*ctx.state(), GpuState::Running);

        ctx.reset();
        assert_eq!(*ctx.state(), GpuState::Uninitialized);
        assert!(ctx.registers().is_none());
    }

    // -- reset ------------------------------------------------------------

    #[test]
    fn reset_from_bars_mapped() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        ctx.reset();
        assert_eq!(*ctx.state(), GpuState::Uninitialized);
        assert!(ctx.registers().is_none());
    }

    #[test]
    fn reset_from_running() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.map_bars(&test_bars()).unwrap();
        ctx.initialize_engines().unwrap();
        ctx.start().unwrap();
        ctx.reset();
        assert_eq!(*ctx.state(), GpuState::Uninitialized);
    }

    #[test]
    fn reset_from_uninitialized_is_noop() {
        let mut ctx = GpuContext::new(test_gpu_info());
        ctx.reset();
        assert_eq!(*ctx.state(), GpuState::Uninitialized);
    }

    // -- InitConfig -------------------------------------------------------

    #[test]
    fn config_defaults() {
        let cfg = InitConfig::default();
        assert!(cfg.enable_compute);
        assert!(cfg.enable_copy);
        assert!(cfg.enable_interrupts);
        assert_eq!(cfg.power_state, PowerState::FullPower);
    }

    #[test]
    fn config_custom() {
        let cfg = InitConfig {
            enable_compute: true,
            enable_copy: false,
            enable_interrupts: false,
            power_state: PowerState::LowPower,
        };
        assert!(!cfg.enable_copy);
        assert_eq!(cfg.power_state, PowerState::LowPower);
    }

    // -- PowerState -------------------------------------------------------

    #[test]
    fn power_state_variants() {
        assert_eq!(PowerState::FullPower, PowerState::FullPower);
        assert_eq!(PowerState::LowPower, PowerState::LowPower);
        assert_eq!(PowerState::Suspended, PowerState::Suspended);
        assert_ne!(PowerState::FullPower, PowerState::Suspended);
    }

    // -- GpuState ---------------------------------------------------------

    #[test]
    fn gpu_state_variants() {
        let states = [
            GpuState::Uninitialized,
            GpuState::BarsMapped,
            GpuState::EnginesReady,
            GpuState::Running,
            GpuState::Error,
            GpuState::Suspended,
        ];
        // Each variant should be distinct.
        for (i, a) in states.iter().enumerate() {
            for (j, b) in states.iter().enumerate() {
                if i == j {
                    assert_eq!(a, b);
                } else {
                    assert_ne!(a, b);
                }
            }
        }
    }
}
