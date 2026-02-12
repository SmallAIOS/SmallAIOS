// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DTB-based FPGA peripheral discovery.
//!
//! Parses the flattened device tree blob (DTB) at boot time to discover
//! FPGA soft-IP peripherals by matching `compatible` strings against
//! known driver names. Extracts MMIO base/size and interrupt numbers.

use alloc::vec::Vec;
use core::fmt;

/// Maximum number of discoverable peripherals.
pub const MAX_PERIPHERALS: usize = 32;

/// Maximum number of known compatible strings to match against.
pub const MAX_KNOWN_DRIVERS: usize = 32;

/// FPGA peripheral descriptor discovered from DTB.
#[derive(Debug, Clone)]
pub struct FpgaPeripheral {
    /// Compatible string from the DTB node (e.g., "xlnx,axi-gpio-2.0").
    pub compatible: PeripheralName,
    /// MMIO base address.
    pub base_addr: u64,
    /// MMIO address range size in bytes.
    pub addr_size: u32,
    /// Interrupt number(s) from the DTB `interrupts` property.
    pub irq_numbers: [u32; 4],
    /// Number of valid interrupt numbers.
    pub irq_count: u8,
    /// Node name from DTB (for diagnostics).
    pub node_name: PeripheralName,
}

/// Fixed-size peripheral name (avoids heap allocation for names).
#[derive(Clone)]
pub struct PeripheralName {
    buf: [u8; 64],
    len: usize,
}

impl PeripheralName {
    /// Create a name from a byte slice (truncates to 64 bytes).
    pub fn from_bytes(s: &[u8]) -> Self {
        let len = s.len().min(64);
        let mut buf = [0u8; 64];
        buf[..len].copy_from_slice(&s[..len]);
        Self { buf, len }
    }

    /// Return the name as a byte slice.
    pub fn as_bytes(&self) -> &[u8] {
        &self.buf[..self.len]
    }

    /// Check if this name matches a given string.
    pub fn matches(&self, other: &[u8]) -> bool {
        self.as_bytes() == other
    }

    /// Length of the name.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the name is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl fmt::Debug for PeripheralName {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Ok(s) = core::str::from_utf8(self.as_bytes()) {
            write!(f, "{s:?}")
        } else {
            write!(f, "{:?}", self.as_bytes())
        }
    }
}

impl PartialEq for PeripheralName {
    fn eq(&self, other: &Self) -> bool {
        self.as_bytes() == other.as_bytes()
    }
}

impl Eq for PeripheralName {}

/// Discovery error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoveryError {
    /// DTB header is invalid (bad magic, version, etc.).
    InvalidDtb,
    /// Peripheral table is full.
    TooManyPeripherals,
    /// Address range overlaps with kernel memory or another peripheral.
    AddressOverlap {
        /// Base address of the conflicting peripheral.
        conflict_addr: u64,
    },
    /// Unknown compatible string (logged as warning, peripheral skipped).
    UnknownCompatible,
    /// DTB parsing encountered a structural error.
    ParseError,
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiscoveryError::InvalidDtb => write!(f, "invalid DTB"),
            DiscoveryError::TooManyPeripherals => write!(f, "too many peripherals"),
            DiscoveryError::AddressOverlap { conflict_addr } => {
                write!(f, "address overlap at 0x{conflict_addr:x}")
            }
            DiscoveryError::UnknownCompatible => write!(f, "unknown compatible string"),
            DiscoveryError::ParseError => write!(f, "DTB parse error"),
        }
    }
}

/// FPGA peripheral discovery engine.
///
/// Maintains a list of known driver compatible strings and discovered
/// peripherals. During boot, the DTB is parsed and peripherals matching
/// known drivers are registered.
#[derive(Debug)]
pub struct PeripheralDiscovery {
    /// Known driver compatible strings.
    known_drivers: [PeripheralName; MAX_KNOWN_DRIVERS],
    /// Number of registered known drivers.
    known_count: usize,
    /// Discovered peripherals.
    peripherals: Vec<FpgaPeripheral>,
    /// Kernel memory region for overlap checking (start, end).
    kernel_region: (u64, u64),
}

impl Default for PeripheralDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PeripheralDiscovery {
    /// Create a new discovery engine.
    pub fn new() -> Self {
        Self {
            known_drivers: core::array::from_fn(|_| PeripheralName::from_bytes(b"")),
            known_count: 0,
            peripherals: Vec::new(),
            kernel_region: (0, 0),
        }
    }

    /// Set the kernel memory region for overlap checking.
    pub fn set_kernel_region(&mut self, start: u64, end: u64) {
        self.kernel_region = (start, end);
    }

    /// Register a known driver compatible string.
    pub fn register_driver(&mut self, compatible: &[u8]) -> bool {
        if self.known_count >= MAX_KNOWN_DRIVERS {
            return false;
        }
        self.known_drivers[self.known_count] = PeripheralName::from_bytes(compatible);
        self.known_count += 1;
        true
    }

    /// Check if a compatible string matches a known driver.
    pub fn is_known_driver(&self, compatible: &[u8]) -> bool {
        for i in 0..self.known_count {
            if self.known_drivers[i].matches(compatible) {
                return true;
            }
        }
        false
    }

    /// Register a discovered peripheral (after DTB parsing).
    ///
    /// Validates the address range doesn't overlap with kernel memory
    /// or other peripherals.
    pub fn register_peripheral(&mut self, periph: FpgaPeripheral) -> Result<(), DiscoveryError> {
        if self.peripherals.len() >= MAX_PERIPHERALS {
            return Err(DiscoveryError::TooManyPeripherals);
        }

        let start = periph.base_addr;
        let end = start + periph.addr_size as u64;

        // Check kernel region overlap
        if self.kernel_region.1 > 0 && start < self.kernel_region.1 && end > self.kernel_region.0 {
            return Err(DiscoveryError::AddressOverlap {
                conflict_addr: start,
            });
        }

        // Check overlap with existing peripherals
        for existing in &self.peripherals {
            let ex_start = existing.base_addr;
            let ex_end = ex_start + existing.addr_size as u64;
            if start < ex_end && end > ex_start {
                return Err(DiscoveryError::AddressOverlap {
                    conflict_addr: ex_start,
                });
            }
        }

        self.peripherals.push(periph);
        Ok(())
    }

    /// Get all discovered peripherals.
    pub fn peripherals(&self) -> &[FpgaPeripheral] {
        &self.peripherals
    }

    /// Find a peripheral by compatible string.
    pub fn find_by_compatible(&self, compatible: &[u8]) -> Option<&FpgaPeripheral> {
        self.peripherals
            .iter()
            .find(|p| p.compatible.matches(compatible))
    }

    /// Find all peripherals matching a compatible string.
    pub fn find_all_by_compatible(&self, compatible: &[u8]) -> Vec<&FpgaPeripheral> {
        self.peripherals
            .iter()
            .filter(|p| p.compatible.matches(compatible))
            .collect()
    }

    /// Number of discovered peripherals.
    pub fn count(&self) -> usize {
        self.peripherals.len()
    }

    /// Clear all discovered peripherals (for re-enumeration).
    pub fn clear(&mut self) {
        self.peripherals.clear();
    }
}

/// Create an FpgaPeripheral from DTB-extracted fields.
pub fn make_peripheral(
    compatible: &[u8],
    node_name: &[u8],
    base_addr: u64,
    addr_size: u32,
    irqs: &[u32],
) -> FpgaPeripheral {
    let mut irq_numbers = [0u32; 4];
    let irq_count = irqs.len().min(4);
    irq_numbers[..irq_count].copy_from_slice(&irqs[..irq_count]);

    FpgaPeripheral {
        compatible: PeripheralName::from_bytes(compatible),
        base_addr,
        addr_size,
        irq_numbers,
        irq_count: irq_count as u8,
        node_name: PeripheralName::from_bytes(node_name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn test_peripheral_name_from_bytes() {
        let name = PeripheralName::from_bytes(b"xlnx,axi-gpio-2.0");
        assert_eq!(name.len(), 17);
        assert!(name.matches(b"xlnx,axi-gpio-2.0"));
        assert!(!name.matches(b"xlnx,axi-uart"));
    }

    #[test]
    fn test_peripheral_name_truncation() {
        let long_name = [b'x'; 100];
        let name = PeripheralName::from_bytes(&long_name);
        assert_eq!(name.len(), 64);
    }

    #[test]
    fn test_peripheral_name_empty() {
        let name = PeripheralName::from_bytes(b"");
        assert!(name.is_empty());
    }

    #[test]
    fn test_peripheral_name_eq() {
        let a = PeripheralName::from_bytes(b"foo");
        let b = PeripheralName::from_bytes(b"foo");
        let c = PeripheralName::from_bytes(b"bar");
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_peripheral_name_debug() {
        let name = PeripheralName::from_bytes(b"test-ip");
        let dbg = format!("{name:?}");
        assert!(dbg.contains("test-ip"));
    }

    #[test]
    fn test_make_peripheral() {
        let p = make_peripheral(
            b"xlnx,axi-gpio-2.0",
            b"gpio@40000000",
            0x4000_0000,
            0x1000,
            &[5, 6],
        );
        assert!(p.compatible.matches(b"xlnx,axi-gpio-2.0"));
        assert!(p.node_name.matches(b"gpio@40000000"));
        assert_eq!(p.base_addr, 0x4000_0000);
        assert_eq!(p.addr_size, 0x1000);
        assert_eq!(p.irq_count, 2);
        assert_eq!(p.irq_numbers[0], 5);
        assert_eq!(p.irq_numbers[1], 6);
    }

    #[test]
    fn test_make_peripheral_no_irqs() {
        let p = make_peripheral(b"custom,ip-1.0", b"ip@50000000", 0x5000_0000, 0x100, &[]);
        assert_eq!(p.irq_count, 0);
    }

    #[test]
    fn test_discovery_register_driver() {
        let mut disc = PeripheralDiscovery::new();
        assert!(disc.register_driver(b"xlnx,axi-gpio-2.0"));
        assert!(disc.register_driver(b"xlnx,axi-uart-1.0"));
        assert!(disc.is_known_driver(b"xlnx,axi-gpio-2.0"));
        assert!(!disc.is_known_driver(b"unknown,device"));
    }

    #[test]
    fn test_discovery_register_peripheral() {
        let mut disc = PeripheralDiscovery::new();
        let p = make_peripheral(
            b"xlnx,axi-gpio-2.0",
            b"gpio@40000000",
            0x4000_0000,
            0x1000,
            &[5],
        );
        disc.register_peripheral(p).unwrap();
        assert_eq!(disc.count(), 1);
    }

    #[test]
    fn test_discovery_find_by_compatible() {
        let mut disc = PeripheralDiscovery::new();
        disc.register_peripheral(make_peripheral(
            b"xlnx,axi-gpio-2.0",
            b"gpio@40000000",
            0x4000_0000,
            0x1000,
            &[5],
        ))
        .unwrap();
        disc.register_peripheral(make_peripheral(
            b"xlnx,axi-uart-1.0",
            b"uart@41000000",
            0x4100_0000,
            0x100,
            &[3],
        ))
        .unwrap();

        let found = disc.find_by_compatible(b"xlnx,axi-uart-1.0").unwrap();
        assert_eq!(found.base_addr, 0x4100_0000);
        assert!(disc.find_by_compatible(b"nonexistent").is_none());
    }

    #[test]
    fn test_discovery_find_all_by_compatible() {
        let mut disc = PeripheralDiscovery::new();
        disc.register_peripheral(make_peripheral(
            b"xlnx,axi-gpio-2.0",
            b"gpio0@40000000",
            0x4000_0000,
            0x1000,
            &[5],
        ))
        .unwrap();
        disc.register_peripheral(make_peripheral(
            b"xlnx,axi-gpio-2.0",
            b"gpio1@40001000",
            0x4000_1000,
            0x1000,
            &[6],
        ))
        .unwrap();
        disc.register_peripheral(make_peripheral(
            b"xlnx,axi-uart-1.0",
            b"uart@41000000",
            0x4100_0000,
            0x100,
            &[3],
        ))
        .unwrap();

        let gpios = disc.find_all_by_compatible(b"xlnx,axi-gpio-2.0");
        assert_eq!(gpios.len(), 2);
    }

    #[test]
    fn test_discovery_address_overlap_kernel() {
        let mut disc = PeripheralDiscovery::new();
        disc.set_kernel_region(0x8000_0000, 0x8100_0000);
        let p = make_peripheral(b"test,ip", b"ip@80800000", 0x8080_0000, 0x1000, &[]);
        assert_eq!(
            disc.register_peripheral(p),
            Err(DiscoveryError::AddressOverlap {
                conflict_addr: 0x8080_0000,
            })
        );
    }

    #[test]
    fn test_discovery_address_overlap_peripheral() {
        let mut disc = PeripheralDiscovery::new();
        disc.register_peripheral(make_peripheral(
            b"test,ip-a",
            b"a@40000000",
            0x4000_0000,
            0x1000,
            &[],
        ))
        .unwrap();

        // Overlaps with existing
        let p = make_peripheral(b"test,ip-b", b"b@40000800", 0x4000_0800, 0x1000, &[]);
        assert_eq!(
            disc.register_peripheral(p),
            Err(DiscoveryError::AddressOverlap {
                conflict_addr: 0x4000_0000,
            })
        );
    }

    #[test]
    fn test_discovery_no_overlap() {
        let mut disc = PeripheralDiscovery::new();
        disc.set_kernel_region(0x8000_0000, 0x8100_0000);
        disc.register_peripheral(make_peripheral(
            b"test,ip-a",
            b"a@40000000",
            0x4000_0000,
            0x1000,
            &[],
        ))
        .unwrap();
        // Non-overlapping
        disc.register_peripheral(make_peripheral(
            b"test,ip-b",
            b"b@40001000",
            0x4000_1000,
            0x1000,
            &[],
        ))
        .unwrap();
        assert_eq!(disc.count(), 2);
    }

    #[test]
    fn test_discovery_clear() {
        let mut disc = PeripheralDiscovery::new();
        disc.register_peripheral(make_peripheral(
            b"test,ip",
            b"ip@40000000",
            0x4000_0000,
            0x1000,
            &[],
        ))
        .unwrap();
        disc.clear();
        assert_eq!(disc.count(), 0);
    }

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", DiscoveryError::InvalidDtb), "invalid DTB");
        assert_eq!(
            format!("{}", DiscoveryError::TooManyPeripherals),
            "too many peripherals"
        );
        assert_eq!(
            format!(
                "{}",
                DiscoveryError::AddressOverlap {
                    conflict_addr: 0x1234
                }
            ),
            "address overlap at 0x1234"
        );
        assert_eq!(
            format!("{}", DiscoveryError::UnknownCompatible),
            "unknown compatible string"
        );
        assert_eq!(format!("{}", DiscoveryError::ParseError), "DTB parse error");
    }
}
