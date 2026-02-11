// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Microchip PolarFire SoC platform support package.
//!
//! The PolarFire SoC combines 4x RISC-V U54 harts + 1x E51 monitor hart
//! with an FPGA fabric. FPGA peripherals are accessed through Fabric
//! Interface Controllers (FIC0-FIC3) that bridge the MSS (Microprocessor
//! SubSystem) AXI bus to the FPGA fabric.
//!
//! Boot flow: HSS (Hart Software Services) initializes the harts and
//! provides a DTB describing FPGA fabric peripherals.

use core::fmt;

/// Number of Fabric Interface Controllers.
pub const NUM_FICS: usize = 4;

/// FIC base addresses (PolarFire SoC MSS memory map).
const FIC_BASE_ADDRS: [u64; NUM_FICS] = [
    0x2000_0000, // FIC0: 32-bit AXI4 to fabric
    0x3000_0000, // FIC1: 32-bit AXI4 to fabric
    0x4000_0000, // FIC2: 32-bit AXI4 to fabric
    0x6000_0000, // FIC3: 64-bit AXI4 to fabric (high performance)
];

/// FIC address space sizes.
const FIC_SIZES: [u64; NUM_FICS] = [
    0x1000_0000, // FIC0: 256 MiB
    0x1000_0000, // FIC1: 256 MiB
    0x1000_0000, // FIC2: 256 MiB
    0x2000_0000, // FIC3: 512 MiB
];

/// Fabric Interface Controller identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FicId {
    /// FIC0: 32-bit AXI4.
    Fic0,
    /// FIC1: 32-bit AXI4.
    Fic1,
    /// FIC2: 32-bit AXI4.
    Fic2,
    /// FIC3: 64-bit AXI4 (high performance).
    Fic3,
}

impl FicId {
    /// Index (0-3).
    pub fn index(&self) -> usize {
        match self {
            FicId::Fic0 => 0,
            FicId::Fic1 => 1,
            FicId::Fic2 => 2,
            FicId::Fic3 => 3,
        }
    }

    /// Base address of this FIC's address window.
    pub fn base_addr(&self) -> u64 {
        FIC_BASE_ADDRS[self.index()]
    }

    /// Address space size.
    pub fn addr_size(&self) -> u64 {
        FIC_SIZES[self.index()]
    }

    /// Data width in bits.
    pub fn data_width_bits(&self) -> u32 {
        match self {
            FicId::Fic0 | FicId::Fic1 | FicId::Fic2 => 32,
            FicId::Fic3 => 64,
        }
    }
}

/// FIC state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FicState {
    /// FIC is not initialized.
    Uninitialized,
    /// FIC is initialized and ready for fabric access.
    Ready,
    /// FIC is in error state.
    Error,
}

/// PolarFire SoC error type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolarFireError {
    /// FIC is not initialized.
    FicNotInitialized,
    /// FIC initialization failed.
    FicInitFailed,
    /// Address is outside the FIC's address window.
    AddressOutOfRange {
        /// FIC that was accessed.
        fic: FicId,
        /// Attempted address.
        address: u64,
    },
    /// MSS interrupt routing failed.
    InterruptRoutingFailed,
    /// HSS boot data not available.
    HssNotAvailable,
    /// FPGA fabric is not programmed or detected.
    FabricNotReady,
}

impl fmt::Display for PolarFireError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolarFireError::FicNotInitialized => write!(f, "FIC not initialized"),
            PolarFireError::FicInitFailed => write!(f, "FIC initialization failed"),
            PolarFireError::AddressOutOfRange { fic, address } => {
                write!(f, "address 0x{address:x} out of range for FIC{}", fic.index())
            }
            PolarFireError::InterruptRoutingFailed => write!(f, "MSS interrupt routing failed"),
            PolarFireError::HssNotAvailable => write!(f, "HSS boot data not available"),
            PolarFireError::FabricNotReady => write!(f, "FPGA fabric not ready"),
        }
    }
}

/// MSS interrupt mapping for fabric interrupts.
///
/// PolarFire SoC routes FPGA fabric interrupts through the MSS
/// interrupt controller to the RISC-V PLIC.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FabricInterruptMapping {
    /// Fabric interrupt line (0-40).
    pub fabric_irq: u8,
    /// PLIC source ID on the RISC-V side.
    pub plic_source: u16,
    /// Whether this mapping is enabled.
    pub enabled: bool,
}

/// Maximum fabric interrupt lines.
pub const MAX_FABRIC_IRQS: usize = 41;

/// PolarFire SoC platform support package.
///
/// Manages FIC initialization, fabric access validation, and MSS-to-PLIC
/// interrupt routing for FPGA peripherals.
#[derive(Debug)]
pub struct PolarFirePlatform {
    /// FIC states.
    fic_states: [FicState; NUM_FICS],
    /// Fabric interrupt mappings.
    irq_mappings: [FabricInterruptMapping; MAX_FABRIC_IRQS],
    /// Number of U54 harts detected.
    u54_hart_count: u8,
    /// Whether HSS boot data was received.
    hss_booted: bool,
    /// Whether the FPGA fabric is programmed.
    fabric_ready: bool,
}

impl Default for PolarFirePlatform {
    fn default() -> Self {
        Self::new()
    }
}

impl PolarFirePlatform {
    /// Create a new PolarFire platform support package.
    pub fn new() -> Self {
        Self {
            fic_states: [FicState::Uninitialized; NUM_FICS],
            irq_mappings: [FabricInterruptMapping {
                fabric_irq: 0,
                plic_source: 0,
                enabled: false,
            }; MAX_FABRIC_IRQS],
            u54_hart_count: 4, // Default for PolarFire SoC
            hss_booted: false,
            fabric_ready: false,
        }
    }

    /// Record that HSS boot completed and provided DTB.
    pub fn on_hss_boot(&mut self) {
        self.hss_booted = true;
    }

    /// Record that FPGA fabric is programmed and ready.
    pub fn on_fabric_ready(&mut self) {
        self.fabric_ready = true;
    }

    /// Whether HSS boot has completed.
    pub fn is_hss_booted(&self) -> bool {
        self.hss_booted
    }

    /// Whether the FPGA fabric is ready.
    pub fn is_fabric_ready(&self) -> bool {
        self.fabric_ready
    }

    /// Number of U54 application harts.
    pub fn u54_hart_count(&self) -> u8 {
        self.u54_hart_count
    }

    /// Set the number of detected U54 harts (from DTB).
    pub fn set_u54_hart_count(&mut self, count: u8) {
        self.u54_hart_count = count.min(4);
    }

    /// Initialize a Fabric Interface Controller.
    ///
    /// Configures the FIC's address mapping before any peripheral access.
    pub fn init_fic(&mut self, fic: FicId) -> Result<(), PolarFireError> {
        if !self.fabric_ready {
            return Err(PolarFireError::FabricNotReady);
        }

        // In real hardware: configure FIC address translation, enable clocking
        // Stub: just mark as ready
        self.fic_states[fic.index()] = FicState::Ready;
        Ok(())
    }

    /// Get the state of a FIC.
    pub fn fic_state(&self, fic: FicId) -> FicState {
        self.fic_states[fic.index()]
    }

    /// Validate that an address is within a FIC's address window.
    pub fn validate_fic_address(&self, fic: FicId, address: u64) -> Result<(), PolarFireError> {
        if self.fic_states[fic.index()] != FicState::Ready {
            return Err(PolarFireError::FicNotInitialized);
        }

        let base = fic.base_addr();
        let end = base + fic.addr_size();
        if address < base || address >= end {
            return Err(PolarFireError::AddressOutOfRange { fic, address });
        }
        Ok(())
    }

    /// Determine which FIC an address belongs to (if any).
    pub fn fic_for_address(&self, address: u64) -> Option<FicId> {
        for fic in [FicId::Fic0, FicId::Fic1, FicId::Fic2, FicId::Fic3] {
            let base = fic.base_addr();
            let end = base + fic.addr_size();
            if address >= base && address < end {
                return Some(fic);
            }
        }
        None
    }

    /// Configure a fabric-to-PLIC interrupt mapping.
    pub fn map_fabric_irq(
        &mut self,
        fabric_irq: u8,
        plic_source: u16,
    ) -> Result<(), PolarFireError> {
        if fabric_irq as usize >= MAX_FABRIC_IRQS {
            return Err(PolarFireError::InterruptRoutingFailed);
        }
        self.irq_mappings[fabric_irq as usize] = FabricInterruptMapping {
            fabric_irq,
            plic_source,
            enabled: true,
        };
        Ok(())
    }

    /// Enable a fabric interrupt mapping.
    pub fn enable_fabric_irq(&mut self, fabric_irq: u8) -> Result<(), PolarFireError> {
        if fabric_irq as usize >= MAX_FABRIC_IRQS {
            return Err(PolarFireError::InterruptRoutingFailed);
        }
        self.irq_mappings[fabric_irq as usize].enabled = true;
        Ok(())
    }

    /// Disable a fabric interrupt mapping.
    pub fn disable_fabric_irq(&mut self, fabric_irq: u8) -> Result<(), PolarFireError> {
        if fabric_irq as usize >= MAX_FABRIC_IRQS {
            return Err(PolarFireError::InterruptRoutingFailed);
        }
        self.irq_mappings[fabric_irq as usize].enabled = false;
        Ok(())
    }

    /// Get the PLIC source for a fabric interrupt.
    pub fn plic_source_for_irq(&self, fabric_irq: u8) -> Option<u16> {
        if fabric_irq as usize >= MAX_FABRIC_IRQS {
            return None;
        }
        let mapping = &self.irq_mappings[fabric_irq as usize];
        if mapping.enabled {
            Some(mapping.plic_source)
        } else {
            None
        }
    }

    /// Get all FIC base addresses and sizes (for DTB-based discovery).
    pub fn fic_regions() -> [(u64, u64); NUM_FICS] {
        let mut regions = [(0u64, 0u64); NUM_FICS];
        for (i, region) in regions.iter_mut().enumerate() {
            *region = (FIC_BASE_ADDRS[i], FIC_SIZES[i]);
        }
        regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_new() {
        let platform = PolarFirePlatform::new();
        assert!(!platform.is_hss_booted());
        assert!(!platform.is_fabric_ready());
        assert_eq!(platform.u54_hart_count(), 4);
        assert_eq!(platform.fic_state(FicId::Fic0), FicState::Uninitialized);
    }

    #[test]
    fn test_hss_boot() {
        let mut platform = PolarFirePlatform::new();
        platform.on_hss_boot();
        assert!(platform.is_hss_booted());
    }

    #[test]
    fn test_fabric_ready() {
        let mut platform = PolarFirePlatform::new();
        platform.on_fabric_ready();
        assert!(platform.is_fabric_ready());
    }

    #[test]
    fn test_init_fic_requires_fabric() {
        let mut platform = PolarFirePlatform::new();
        assert_eq!(
            platform.init_fic(FicId::Fic0),
            Err(PolarFireError::FabricNotReady)
        );
    }

    #[test]
    fn test_init_fic_success() {
        let mut platform = PolarFirePlatform::new();
        platform.on_fabric_ready();
        platform.init_fic(FicId::Fic0).unwrap();
        assert_eq!(platform.fic_state(FicId::Fic0), FicState::Ready);
    }

    #[test]
    fn test_init_all_fics() {
        let mut platform = PolarFirePlatform::new();
        platform.on_fabric_ready();
        for fic in [FicId::Fic0, FicId::Fic1, FicId::Fic2, FicId::Fic3] {
            platform.init_fic(fic).unwrap();
            assert_eq!(platform.fic_state(fic), FicState::Ready);
        }
    }

    #[test]
    fn test_validate_fic_address() {
        let mut platform = PolarFirePlatform::new();
        platform.on_fabric_ready();
        platform.init_fic(FicId::Fic0).unwrap();

        // Valid address in FIC0 range
        platform.validate_fic_address(FicId::Fic0, 0x2000_1000).unwrap();

        // Out of range
        assert!(platform.validate_fic_address(FicId::Fic0, 0x5000_0000).is_err());
    }

    #[test]
    fn test_validate_fic_not_initialized() {
        let platform = PolarFirePlatform::new();
        assert_eq!(
            platform.validate_fic_address(FicId::Fic0, 0x2000_0000),
            Err(PolarFireError::FicNotInitialized)
        );
    }

    #[test]
    fn test_fic_for_address() {
        let platform = PolarFirePlatform::new();
        assert_eq!(platform.fic_for_address(0x2000_5000), Some(FicId::Fic0));
        assert_eq!(platform.fic_for_address(0x3000_5000), Some(FicId::Fic1));
        assert_eq!(platform.fic_for_address(0x4000_5000), Some(FicId::Fic2));
        assert_eq!(platform.fic_for_address(0x6000_5000), Some(FicId::Fic3));
        assert_eq!(platform.fic_for_address(0x1000_0000), None); // Below any FIC
        assert_eq!(platform.fic_for_address(0x9000_0000), None); // Above all FICs
    }

    #[test]
    fn test_fic_properties() {
        assert_eq!(FicId::Fic0.base_addr(), 0x2000_0000);
        assert_eq!(FicId::Fic0.addr_size(), 0x1000_0000);
        assert_eq!(FicId::Fic0.data_width_bits(), 32);

        assert_eq!(FicId::Fic3.base_addr(), 0x6000_0000);
        assert_eq!(FicId::Fic3.addr_size(), 0x2000_0000);
        assert_eq!(FicId::Fic3.data_width_bits(), 64);
    }

    #[test]
    fn test_fic_index() {
        assert_eq!(FicId::Fic0.index(), 0);
        assert_eq!(FicId::Fic1.index(), 1);
        assert_eq!(FicId::Fic2.index(), 2);
        assert_eq!(FicId::Fic3.index(), 3);
    }

    #[test]
    fn test_map_fabric_irq() {
        let mut platform = PolarFirePlatform::new();
        platform.map_fabric_irq(5, 100).unwrap();
        assert_eq!(platform.plic_source_for_irq(5), Some(100));
    }

    #[test]
    fn test_map_fabric_irq_invalid() {
        let mut platform = PolarFirePlatform::new();
        assert_eq!(
            platform.map_fabric_irq(50, 100), // > MAX_FABRIC_IRQS
            Err(PolarFireError::InterruptRoutingFailed)
        );
    }

    #[test]
    fn test_enable_disable_fabric_irq() {
        let mut platform = PolarFirePlatform::new();
        platform.map_fabric_irq(10, 200).unwrap();
        assert_eq!(platform.plic_source_for_irq(10), Some(200));

        platform.disable_fabric_irq(10).unwrap();
        assert_eq!(platform.plic_source_for_irq(10), None);

        platform.enable_fabric_irq(10).unwrap();
        assert_eq!(platform.plic_source_for_irq(10), Some(200));
    }

    #[test]
    fn test_plic_source_unmapped() {
        let platform = PolarFirePlatform::new();
        assert_eq!(platform.plic_source_for_irq(0), None);
    }

    #[test]
    fn test_set_hart_count() {
        let mut platform = PolarFirePlatform::new();
        platform.set_u54_hart_count(2);
        assert_eq!(platform.u54_hart_count(), 2);
        // Clamped to max 4
        platform.set_u54_hart_count(8);
        assert_eq!(platform.u54_hart_count(), 4);
    }

    #[test]
    fn test_fic_regions() {
        let regions = PolarFirePlatform::fic_regions();
        assert_eq!(regions[0], (0x2000_0000, 0x1000_0000));
        assert_eq!(regions[3], (0x6000_0000, 0x2000_0000));
    }

    #[test]
    fn test_fic_state_variants() {
        assert_ne!(FicState::Uninitialized, FicState::Ready);
        assert_ne!(FicState::Ready, FicState::Error);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", PolarFireError::FicNotInitialized), "FIC not initialized");
        assert_eq!(format!("{}", PolarFireError::FicInitFailed), "FIC initialization failed");
        assert_eq!(
            format!(
                "{}",
                PolarFireError::AddressOutOfRange {
                    fic: FicId::Fic2,
                    address: 0x5000_0000,
                }
            ),
            "address 0x50000000 out of range for FIC2"
        );
        assert_eq!(
            format!("{}", PolarFireError::InterruptRoutingFailed),
            "MSS interrupt routing failed"
        );
        assert_eq!(format!("{}", PolarFireError::HssNotAvailable), "HSS boot data not available");
        assert_eq!(format!("{}", PolarFireError::FabricNotReady), "FPGA fabric not ready");
    }
}
