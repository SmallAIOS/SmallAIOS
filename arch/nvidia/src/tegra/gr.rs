// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! GR (Graphics/Compute) Engine for GM20B (Tegra X1 / T210).
//!
//! The GR engine handles compute kernel execution. GM20B has:
//! - 1 GPC (Graphics Processing Cluster)
//! - 1 TPC (Texture Processing Cluster)
//! - 2 SMs (Streaming Multiprocessors)
//! - 128 CUDA cores total (2 SMs x 64 cores each)

#![allow(dead_code)]

use crate::GpuError;

// PMC and GR register offsets (relative to BAR0)
const NV_PMC_ENABLE: u32 = 0x200;
const NV_PMC_ENABLE_GR: u32 = 1 << 12;
const GR_PRI_GPC0_TPC0_SM_ARCH: u32 = 0x50_4330;

/// MMIO register access trait for hardware interaction.
pub trait MmioAccess {
    fn read32(&self, addr: u64) -> u32;
    fn write32(&self, addr: u64, val: u32);
}

/// GR engine initialization state machine.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum GrState {
    Uninitialized,
    Reset,
    ContextLoaded,
    Ready,
    Error,
}

/// GM20B compute topology description.
#[derive(Clone, Debug, PartialEq)]
pub struct GrTopology {
    pub gpc_count: u32,
    pub tpc_per_gpc: u32,
    pub sm_per_tpc: u32,
    pub max_warps_per_sm: u32,
}

impl GrTopology {
    /// Total number of streaming multiprocessors.
    pub fn total_sms(&self) -> u32 {
        self.gpc_count * self.tpc_per_gpc * self.sm_per_tpc
    }

    /// Total CUDA cores (Maxwell: 128 cores per SM).
    pub fn total_cuda_cores(&self) -> u32 {
        self.total_sms() * 128
    }

    /// Default topology for GM20B (Tegra X1).
    pub fn gm20b_default() -> Self {
        Self {
            gpc_count: 1,
            tpc_per_gpc: 1,
            sm_per_tpc: 2,
            max_warps_per_sm: 64,
        }
    }
}

/// GR (Graphics/Compute) engine controller.
pub struct GrEngine {
    state: GrState,
    topology: GrTopology,
}

impl Default for GrEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl GrEngine {
    /// Create a new uninitialized GR engine.
    pub fn new() -> Self {
        Self {
            state: GrState::Uninitialized,
            topology: GrTopology::gm20b_default(),
        }
    }

    /// Initialize the GR engine.
    ///
    /// 1. Resets the GR engine via NV_PMC_ENABLE.
    /// 2. Reads topology (uses GM20B defaults).
    /// 3. Transitions through ContextLoaded to Ready.
    pub fn init(&mut self, bar0_base: u64, mmio: &impl MmioAccess) -> Result<(), GpuError> {
        if self.state == GrState::Error {
            return Err(GpuError::InvalidState);
        }

        // Step 1: Reset GR engine via PMC
        let pmc_addr = bar0_base + NV_PMC_ENABLE as u64;
        let pmc_val = mmio.read32(pmc_addr);
        // Clear GR enable bit to reset
        mmio.write32(pmc_addr, pmc_val & !NV_PMC_ENABLE_GR);
        // Re-enable GR engine
        mmio.write32(pmc_addr, pmc_val | NV_PMC_ENABLE_GR);
        self.state = GrState::Reset;

        // Step 2: Read topology (use GM20B defaults for integrated GPU)
        self.topology = GrTopology::gm20b_default();

        // Step 3: Transition to ContextLoaded then Ready
        self.state = GrState::ContextLoaded;
        self.state = GrState::Ready;

        Ok(())
    }

    /// Return a reference to the compute topology.
    pub fn topology(&self) -> &GrTopology {
        &self.topology
    }

    /// Check whether the GR engine is initialized and ready.
    pub fn is_ready(&self) -> bool {
        self.state == GrState::Ready
    }

    /// Return the current state.
    pub fn state(&self) -> GrState {
        self.state
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::cell::RefCell;

    /// Mock MMIO that records writes and returns configurable read values.
    struct MockMmio {
        writes: RefCell<alloc::vec::Vec<(u64, u32)>>,
        pmc_value: u32,
    }

    impl MockMmio {
        fn new(pmc_value: u32) -> Self {
            Self {
                writes: RefCell::new(alloc::vec::Vec::new()),
                pmc_value,
            }
        }
    }

    impl MmioAccess for MockMmio {
        fn read32(&self, _addr: u64) -> u32 {
            self.pmc_value
        }

        fn write32(&self, addr: u64, val: u32) {
            self.writes.borrow_mut().push((addr, val));
        }
    }

    extern crate alloc;

    #[test]
    fn test_gm20b_topology_defaults() {
        let topo = GrTopology::gm20b_default();
        assert_eq!(topo.gpc_count, 1);
        assert_eq!(topo.tpc_per_gpc, 1);
        assert_eq!(topo.sm_per_tpc, 2);
        assert_eq!(topo.max_warps_per_sm, 64);
    }

    #[test]
    fn test_total_sms() {
        let topo = GrTopology::gm20b_default();
        assert_eq!(topo.total_sms(), 2);
    }

    #[test]
    fn test_total_cuda_cores() {
        let topo = GrTopology::gm20b_default();
        // 2 SMs * 128 cores/SM = 256 CUDA cores
        assert_eq!(topo.total_cuda_cores(), 256);
    }

    #[test]
    fn test_custom_topology() {
        let topo = GrTopology {
            gpc_count: 4,
            tpc_per_gpc: 5,
            sm_per_tpc: 2,
            max_warps_per_sm: 64,
        };
        assert_eq!(topo.total_sms(), 40);
        assert_eq!(topo.total_cuda_cores(), 5120);
    }

    #[test]
    fn test_gr_engine_initial_state() {
        let engine = GrEngine::new();
        assert_eq!(engine.state(), GrState::Uninitialized);
        assert!(!engine.is_ready());
    }

    #[test]
    fn test_gr_engine_init() {
        let mut engine = GrEngine::new();
        let mmio = MockMmio::new(0x0000_0000);
        let bar0 = 0x5000_0000u64;

        let result = engine.init(bar0, &mmio);
        assert!(result.is_ok());
        assert!(engine.is_ready());
        assert_eq!(engine.state(), GrState::Ready);
    }

    #[test]
    fn test_gr_engine_init_writes_pmc() {
        let mut engine = GrEngine::new();
        let mmio = MockMmio::new(0xFFFF_FFFF);
        let bar0 = 0x5000_0000u64;

        engine.init(bar0, &mmio).unwrap();

        let writes = mmio.writes.borrow();
        let pmc_addr = bar0 + NV_PMC_ENABLE as u64;
        // First write clears GR bit, second write sets it
        assert_eq!(writes[0].0, pmc_addr);
        assert_eq!(writes[0].1 & NV_PMC_ENABLE_GR, 0); // GR bit cleared
        assert_eq!(writes[1].0, pmc_addr);
        assert_ne!(writes[1].1 & NV_PMC_ENABLE_GR, 0); // GR bit set
    }

    #[test]
    fn test_gr_engine_topology_after_init() {
        let mut engine = GrEngine::new();
        let mmio = MockMmio::new(0);
        engine.init(0x5000_0000, &mmio).unwrap();

        let topo = engine.topology();
        assert_eq!(topo.gpc_count, 1);
        assert_eq!(topo.total_cuda_cores(), 256);
    }

    #[test]
    fn test_gr_engine_error_state_rejects_init() {
        let mut engine = GrEngine::new();
        engine.state = GrState::Error;
        let mmio = MockMmio::new(0);

        let result = engine.init(0x5000_0000, &mmio);
        assert_eq!(result, Err(GpuError::InvalidState));
    }
}
