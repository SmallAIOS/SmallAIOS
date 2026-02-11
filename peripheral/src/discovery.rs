// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DTB-based peripheral discovery module.
//!
//! Scans Device Tree Blob (DTB) node information to identify peripherals
//! attached to the system. The arch crates (`smallaios-arch-aarch64`,
//! `smallaios-arch-riscv64`) parse the raw DTB and pass [`DtbNode`] entries
//! to [`PeripheralDiscovery`], which matches `compatible` strings against
//! known peripheral types and produces [`PeripheralDescriptor`] records.
//!
//! This module is always compiled (not feature-gated) because discovery
//! must be available regardless of which specific peripheral drivers are
//! enabled.

use crate::{PeripheralDescriptor, PeripheralType};

// ---------------------------------------------------------------------------
// Maximum limits (fixed-size, no_std friendly)
// ---------------------------------------------------------------------------

/// Maximum number of peripherals that can be discovered in a single scan.
pub const MAX_DISCOVERED: usize = 32;

/// Maximum length of a DTB compatible string we consider.
pub const MAX_COMPATIBLE_LEN: usize = 64;

/// Maximum number of compatible strings per DTB node.
pub const MAX_COMPATIBLE_ENTRIES: usize = 4;

// ---------------------------------------------------------------------------
// DTB node representation
// ---------------------------------------------------------------------------

/// A parsed DTB node passed from the architecture HAL layer.
///
/// The arch crate walks the raw FDT structure block and constructs these
/// lightweight descriptors for each node that has a `compatible` property.
/// The `compatible` field holds one or more null-terminated compatible
/// strings concatenated in order of specificity (most specific first).
#[derive(Debug, Clone, Copy)]
pub struct DtbNode {
    /// MMIO base address from the `reg` property (first entry).
    pub base_addr: u64,
    /// MMIO region size from the `reg` property (first entry).
    pub size: u64,
    /// IRQ number from the `interrupts` property (first entry, 0 if absent).
    pub irq: u32,
    /// Compatible strings, stored as fixed byte arrays.
    /// Each entry is a null-terminated string padded with zeros.
    pub compatible: [[u8; MAX_COMPATIBLE_LEN]; MAX_COMPATIBLE_ENTRIES],
    /// Number of valid compatible string entries.
    pub compatible_count: u8,
}

impl DtbNode {
    /// Create an empty DTB node.
    pub const fn empty() -> Self {
        Self {
            base_addr: 0,
            size: 0,
            irq: 0,
            compatible: [[0u8; MAX_COMPATIBLE_LEN]; MAX_COMPATIBLE_ENTRIES],
            compatible_count: 0,
        }
    }

    /// Create a new DTB node with a single compatible string.
    ///
    /// This is a convenience constructor for testing and simple cases.
    pub fn with_compatible(base_addr: u64, size: u64, irq: u32, compat: &[u8]) -> Self {
        let mut node = Self::empty();
        node.base_addr = base_addr;
        node.size = size;
        node.irq = irq;
        let copy_len = if compat.len() < MAX_COMPATIBLE_LEN {
            compat.len()
        } else {
            MAX_COMPATIBLE_LEN - 1
        };
        node.compatible[0][..copy_len].copy_from_slice(&compat[..copy_len]);
        node.compatible_count = 1;
        node
    }

    /// Add an additional compatible string to this node.
    ///
    /// Returns `true` if the string was added, `false` if the node already
    /// has the maximum number of compatible entries.
    pub fn add_compatible(&mut self, compat: &[u8]) -> bool {
        let idx = self.compatible_count as usize;
        if idx >= MAX_COMPATIBLE_ENTRIES {
            return false;
        }
        let copy_len = if compat.len() < MAX_COMPATIBLE_LEN {
            compat.len()
        } else {
            MAX_COMPATIBLE_LEN - 1
        };
        self.compatible[idx][..copy_len].copy_from_slice(&compat[..copy_len]);
        self.compatible_count += 1;
        true
    }

    /// Get the compatible string at the given index as a byte slice (up to the
    /// first null byte).
    pub fn get_compatible(&self, index: usize) -> Option<&[u8]> {
        if index >= self.compatible_count as usize {
            return None;
        }
        let entry = &self.compatible[index];
        let len = entry
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(MAX_COMPATIBLE_LEN);
        Some(&entry[..len])
    }
}

// ---------------------------------------------------------------------------
// Compatible string tables
// ---------------------------------------------------------------------------

/// I2C controller compatible strings.
const I2C_COMPATIBLES: &[&[u8]] = &[
    b"snps,designware-i2c",
    b"sifive,i2c0",
    b"opencores,i2c-ocores",
    b"xlnx,xps-iic-2.00.a",
];

/// SPI controller compatible strings.
const SPI_COMPATIBLES: &[&[u8]] = &[
    b"arm,pl022",
    b"snps,dw-apb-ssi",
    b"sifive,spi0",
    b"xlnx,xps-spi-2.00.a",
];

/// GPIO controller compatible strings.
const GPIO_COMPATIBLES: &[&[u8]] = &[b"arm,pl061", b"sifive,gpio0", b"xlnx,xps-gpio-1.00.a"];

/// UART controller compatible strings.
const UART_COMPATIBLES: &[&[u8]] = &[
    b"arm,pl011",
    b"ns16550a",
    b"ns16550",
    b"sifive,uart0",
    b"xlnx,xps-uartlite-1.00.a",
];

/// CSI camera receiver compatible strings.
const CSI_COMPATIBLES: &[&[u8]] = &[b"nvidia,tegra-vi", b"brcm,bcm2835-unicam"];

/// I2S audio controller compatible strings.
const I2S_COMPATIBLES: &[&[u8]] = &[b"nvidia,tegra-i2s", b"brcm,bcm2835-i2s"];

/// All compatible tables indexed by [`PeripheralType`] discriminant.
const COMPAT_TABLES: &[&[&[u8]]] = &[
    I2C_COMPATIBLES,  // PeripheralType::I2c  = 0
    SPI_COMPATIBLES,  // PeripheralType::Spi  = 1
    GPIO_COMPATIBLES, // PeripheralType::Gpio = 2
    UART_COMPATIBLES, // PeripheralType::Uart = 3
    CSI_COMPATIBLES,  // PeripheralType::Camera = 4
    I2S_COMPATIBLES,  // PeripheralType::Audio  = 5
];

// ---------------------------------------------------------------------------
// Discovery result
// ---------------------------------------------------------------------------

/// Result of a peripheral discovery scan.
#[derive(Debug, Clone)]
pub struct DiscoveryResult {
    /// Discovered peripheral descriptors.
    pub peripherals: [Option<PeripheralDescriptor>; MAX_DISCOVERED],
    /// Number of peripherals discovered.
    pub count: usize,
}

impl DiscoveryResult {
    /// Create an empty discovery result.
    pub const fn empty() -> Self {
        Self {
            peripherals: [None; MAX_DISCOVERED],
            count: 0,
        }
    }

    /// Get a slice of the discovered peripherals.
    pub fn as_slice(&self) -> &[Option<PeripheralDescriptor>] {
        &self.peripherals[..self.count]
    }

    /// Iterate over discovered peripherals.
    pub fn iter(&self) -> DiscoveryIter<'_> {
        DiscoveryIter {
            result: self,
            index: 0,
        }
    }

    /// Count peripherals of a specific type.
    pub fn count_type(&self, ptype: PeripheralType) -> usize {
        let mut n = 0;
        for i in 0..self.count {
            if let Some(ref desc) = self.peripherals[i] {
                if desc.peripheral_type == ptype {
                    n += 1;
                }
            }
        }
        n
    }

    /// Find the first peripheral of a given type.
    pub fn find_first(&self, ptype: PeripheralType) -> Option<&PeripheralDescriptor> {
        for i in 0..self.count {
            if let Some(ref desc) = self.peripherals[i] {
                if desc.peripheral_type == ptype {
                    return Some(desc);
                }
            }
        }
        None
    }
}

/// Iterator over discovered peripherals.
pub struct DiscoveryIter<'a> {
    result: &'a DiscoveryResult,
    index: usize,
}

impl<'a> Iterator for DiscoveryIter<'a> {
    type Item = &'a PeripheralDescriptor;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.result.count {
            let i = self.index;
            self.index += 1;
            if let Some(ref desc) = self.result.peripherals[i] {
                return Some(desc);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Peripheral discovery engine
// ---------------------------------------------------------------------------

/// DTB-based peripheral discovery engine.
///
/// Accepts a set of [`DtbNode`] entries (typically produced by the arch
/// crate's DTB parser) and matches their `compatible` strings against
/// known peripheral types to produce [`PeripheralDescriptor`] records.
///
/// # Example
///
/// ```ignore
/// let nodes = [
///     DtbNode::with_compatible(0x4000_5400, 0x400, 31, b"snps,designware-i2c"),
///     DtbNode::with_compatible(0x4001_3000, 0x400, 35, b"arm,pl022"),
/// ];
/// let mut discovery = PeripheralDiscovery::new();
/// let result = discovery.discover_all(&nodes);
/// assert_eq!(result.count, 2);
/// ```
pub struct PeripheralDiscovery {
    /// Instance counters per peripheral type (I2C=0..Audio=5).
    instance_counters: [u8; 6],
}

impl Default for PeripheralDiscovery {
    fn default() -> Self {
        Self::new()
    }
}

impl PeripheralDiscovery {
    /// Create a new peripheral discovery engine with zeroed instance counters.
    pub const fn new() -> Self {
        Self {
            instance_counters: [0; 6],
        }
    }

    /// Reset instance counters (useful between successive scans).
    pub fn reset(&mut self) {
        self.instance_counters = [0; 6];
    }

    /// Discover all peripherals from the given DTB nodes.
    ///
    /// Iterates over each node, checks all its compatible strings against
    /// the known tables, and produces a [`PeripheralDescriptor`] for each
    /// match. Instance numbers are assigned sequentially per type.
    pub fn discover_all(&mut self, nodes: &[DtbNode]) -> DiscoveryResult {
        let mut result = DiscoveryResult::empty();

        for node in nodes {
            if result.count >= MAX_DISCOVERED {
                break;
            }
            if let Some(desc) = self.match_node(node) {
                result.peripherals[result.count] = Some(desc);
                result.count += 1;
            }
        }

        result
    }

    /// Try to match a single DTB node against known compatible strings.
    ///
    /// Returns `Some(PeripheralDescriptor)` if a match is found, `None`
    /// otherwise. The first matching compatible string wins (most specific
    /// first, per DTB convention).
    pub fn match_node(&mut self, node: &DtbNode) -> Option<PeripheralDescriptor> {
        for compat_idx in 0..node.compatible_count as usize {
            if let Some(compat_str) = node.get_compatible(compat_idx) {
                if let Some(ptype) = self.match_compatible(compat_str) {
                    let type_idx = ptype as u8 as usize;
                    let instance = self.instance_counters[type_idx];
                    self.instance_counters[type_idx] = instance.saturating_add(1);
                    return Some(PeripheralDescriptor {
                        peripheral_type: ptype,
                        instance,
                        base_addr: node.base_addr,
                        irq: node.irq,
                    });
                }
            }
        }
        None
    }

    /// Match a single compatible string against all known tables.
    fn match_compatible(&self, compat: &[u8]) -> Option<PeripheralType> {
        for (type_idx, table) in COMPAT_TABLES.iter().enumerate() {
            for &known in *table {
                if byte_eq(compat, known) {
                    return PeripheralType::from_u8(type_idx as u8);
                }
            }
        }
        None
    }

    /// Get the current instance count for a peripheral type.
    pub fn instance_count(&self, ptype: PeripheralType) -> u8 {
        self.instance_counters[ptype as u8 as usize]
    }
}

/// Compare two byte slices for equality.
fn byte_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    for i in 0..a.len() {
        if a[i] != b[i] {
            return false;
        }
    }
    true
}

// ---------------------------------------------------------------------------
// Lookup helpers (public API for use by arch crates)
// ---------------------------------------------------------------------------

/// Check if a compatible string matches any known I2C controller.
pub fn is_i2c_compatible(compat: &[u8]) -> bool {
    I2C_COMPATIBLES.iter().any(|&known| byte_eq(compat, known))
}

/// Check if a compatible string matches any known SPI controller.
pub fn is_spi_compatible(compat: &[u8]) -> bool {
    SPI_COMPATIBLES.iter().any(|&known| byte_eq(compat, known))
}

/// Check if a compatible string matches any known GPIO controller.
pub fn is_gpio_compatible(compat: &[u8]) -> bool {
    GPIO_COMPATIBLES.iter().any(|&known| byte_eq(compat, known))
}

/// Check if a compatible string matches any known UART controller.
pub fn is_uart_compatible(compat: &[u8]) -> bool {
    UART_COMPATIBLES.iter().any(|&known| byte_eq(compat, known))
}

/// Check if a compatible string matches any known CSI camera receiver.
pub fn is_csi_compatible(compat: &[u8]) -> bool {
    CSI_COMPATIBLES.iter().any(|&known| byte_eq(compat, known))
}

/// Check if a compatible string matches any known I2S audio controller.
pub fn is_i2s_compatible(compat: &[u8]) -> bool {
    I2S_COMPATIBLES.iter().any(|&known| byte_eq(compat, known))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PeripheralType;

    // -- DtbNode construction -----------------------------------------------

    #[test]
    fn test_dtb_node_empty() {
        let node = DtbNode::empty();
        assert_eq!(node.base_addr, 0);
        assert_eq!(node.size, 0);
        assert_eq!(node.irq, 0);
        assert_eq!(node.compatible_count, 0);
    }

    #[test]
    fn test_dtb_node_with_compatible() {
        let node = DtbNode::with_compatible(0x4000_5400, 0x400, 31, b"snps,designware-i2c");
        assert_eq!(node.base_addr, 0x4000_5400);
        assert_eq!(node.size, 0x400);
        assert_eq!(node.irq, 31);
        assert_eq!(node.compatible_count, 1);
        assert_eq!(
            node.get_compatible(0),
            Some(b"snps,designware-i2c" as &[u8])
        );
    }

    #[test]
    fn test_dtb_node_add_compatible() {
        let mut node = DtbNode::with_compatible(0x1000, 0x100, 10, b"arm,pl022");
        assert!(node.add_compatible(b"arm,primecell"));
        assert_eq!(node.compatible_count, 2);
        assert_eq!(node.get_compatible(0), Some(b"arm,pl022" as &[u8]));
        assert_eq!(node.get_compatible(1), Some(b"arm,primecell" as &[u8]));
    }

    #[test]
    fn test_dtb_node_add_compatible_max() {
        let mut node = DtbNode::empty();
        for i in 0..MAX_COMPATIBLE_ENTRIES {
            assert!(node.add_compatible(b"test"));
            assert_eq!(node.compatible_count, (i + 1) as u8);
        }
        // Should fail on the 5th add.
        assert!(!node.add_compatible(b"overflow"));
        assert_eq!(node.compatible_count, MAX_COMPATIBLE_ENTRIES as u8);
    }

    #[test]
    fn test_dtb_node_get_compatible_out_of_range() {
        let node = DtbNode::with_compatible(0, 0, 0, b"test");
        assert!(node.get_compatible(0).is_some());
        assert!(node.get_compatible(1).is_none());
        assert!(node.get_compatible(99).is_none());
    }

    // -- Compatible string matching -----------------------------------------

    #[test]
    fn test_is_i2c_compatible() {
        assert!(is_i2c_compatible(b"snps,designware-i2c"));
        assert!(is_i2c_compatible(b"sifive,i2c0"));
        assert!(is_i2c_compatible(b"opencores,i2c-ocores"));
        assert!(is_i2c_compatible(b"xlnx,xps-iic-2.00.a"));
        assert!(!is_i2c_compatible(b"arm,pl022"));
        assert!(!is_i2c_compatible(b"unknown,device"));
    }

    #[test]
    fn test_is_spi_compatible() {
        assert!(is_spi_compatible(b"arm,pl022"));
        assert!(is_spi_compatible(b"snps,dw-apb-ssi"));
        assert!(is_spi_compatible(b"sifive,spi0"));
        assert!(is_spi_compatible(b"xlnx,xps-spi-2.00.a"));
        assert!(!is_spi_compatible(b"snps,designware-i2c"));
    }

    #[test]
    fn test_is_gpio_compatible() {
        assert!(is_gpio_compatible(b"arm,pl061"));
        assert!(is_gpio_compatible(b"sifive,gpio0"));
        assert!(is_gpio_compatible(b"xlnx,xps-gpio-1.00.a"));
        assert!(!is_gpio_compatible(b"arm,pl011"));
    }

    #[test]
    fn test_is_uart_compatible() {
        assert!(is_uart_compatible(b"arm,pl011"));
        assert!(is_uart_compatible(b"ns16550a"));
        assert!(is_uart_compatible(b"ns16550"));
        assert!(is_uart_compatible(b"sifive,uart0"));
        assert!(is_uart_compatible(b"xlnx,xps-uartlite-1.00.a"));
        assert!(!is_uart_compatible(b"arm,pl061"));
    }

    #[test]
    fn test_is_csi_compatible() {
        assert!(is_csi_compatible(b"nvidia,tegra-vi"));
        assert!(is_csi_compatible(b"brcm,bcm2835-unicam"));
        assert!(!is_csi_compatible(b"arm,pl011"));
    }

    #[test]
    fn test_is_i2s_compatible() {
        assert!(is_i2s_compatible(b"nvidia,tegra-i2s"));
        assert!(is_i2s_compatible(b"brcm,bcm2835-i2s"));
        assert!(!is_i2s_compatible(b"nvidia,tegra-vi"));
    }

    // -- PeripheralDiscovery ------------------------------------------------

    #[test]
    fn test_discovery_empty_nodes() {
        let mut disc = PeripheralDiscovery::new();
        let result = disc.discover_all(&[]);
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_discovery_single_i2c() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [DtbNode::with_compatible(
            0x4000_5400,
            0x400,
            31,
            b"snps,designware-i2c",
        )];
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, 1);
        let desc = result.peripherals[0].unwrap();
        assert_eq!(desc.peripheral_type, PeripheralType::I2c);
        assert_eq!(desc.instance, 0);
        assert_eq!(desc.base_addr, 0x4000_5400);
        assert_eq!(desc.irq, 31);
    }

    #[test]
    fn test_discovery_multiple_same_type() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [
            DtbNode::with_compatible(0x4000_5400, 0x400, 31, b"snps,designware-i2c"),
            DtbNode::with_compatible(0x4000_5800, 0x400, 32, b"snps,designware-i2c"),
            DtbNode::with_compatible(0x4000_5C00, 0x400, 33, b"sifive,i2c0"),
        ];
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, 3);
        assert_eq!(result.peripherals[0].unwrap().instance, 0);
        assert_eq!(result.peripherals[1].unwrap().instance, 1);
        assert_eq!(result.peripherals[2].unwrap().instance, 2);
        assert_eq!(result.count_type(PeripheralType::I2c), 3);
    }

    #[test]
    fn test_discovery_mixed_types() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [
            DtbNode::with_compatible(0x1000_0000, 0x1000, 10, b"arm,pl011"),
            DtbNode::with_compatible(0x2000_0000, 0x1000, 20, b"arm,pl022"),
            DtbNode::with_compatible(0x3000_0000, 0x1000, 30, b"arm,pl061"),
            DtbNode::with_compatible(0x4000_0000, 0x400, 40, b"snps,designware-i2c"),
        ];
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, 4);
        assert_eq!(result.count_type(PeripheralType::Uart), 1);
        assert_eq!(result.count_type(PeripheralType::Spi), 1);
        assert_eq!(result.count_type(PeripheralType::Gpio), 1);
        assert_eq!(result.count_type(PeripheralType::I2c), 1);
    }

    #[test]
    fn test_discovery_unknown_compatible() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [DtbNode::with_compatible(
            0x1000,
            0x100,
            5,
            b"unknown,device-v1",
        )];
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, 0);
    }

    #[test]
    fn test_discovery_multi_compatible_fallback() {
        let mut disc = PeripheralDiscovery::new();
        let mut node = DtbNode::with_compatible(0x4001_3000, 0x1000, 35, b"vendor,custom-spi-v2");
        // Second compatible is the generic one that matches.
        node.add_compatible(b"arm,pl022");
        let result = disc.discover_all(&[node]);
        assert_eq!(result.count, 1);
        let desc = result.peripherals[0].unwrap();
        assert_eq!(desc.peripheral_type, PeripheralType::Spi);
    }

    #[test]
    fn test_discovery_find_first() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [
            DtbNode::with_compatible(0x1000, 0x100, 10, b"arm,pl011"),
            DtbNode::with_compatible(0x2000, 0x100, 20, b"arm,pl022"),
            DtbNode::with_compatible(0x3000, 0x100, 30, b"ns16550a"),
        ];
        let result = disc.discover_all(&nodes);
        let uart = result.find_first(PeripheralType::Uart).unwrap();
        assert_eq!(uart.base_addr, 0x1000);
        assert_eq!(uart.instance, 0);

        let spi = result.find_first(PeripheralType::Spi).unwrap();
        assert_eq!(spi.base_addr, 0x2000);

        assert!(result.find_first(PeripheralType::Camera).is_none());
    }

    #[test]
    fn test_discovery_reset() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [DtbNode::with_compatible(0x1000, 0x100, 10, b"arm,pl011")];
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, 1);
        assert_eq!(disc.instance_count(PeripheralType::Uart), 1);

        disc.reset();
        assert_eq!(disc.instance_count(PeripheralType::Uart), 0);

        let result2 = disc.discover_all(&nodes);
        assert_eq!(result2.count, 1);
        assert_eq!(result2.peripherals[0].unwrap().instance, 0);
    }

    #[test]
    fn test_discovery_iterator() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [
            DtbNode::with_compatible(0x1000, 0x100, 10, b"arm,pl011"),
            DtbNode::with_compatible(0x2000, 0x100, 20, b"arm,pl022"),
        ];
        let result = disc.discover_all(&nodes);
        let types: alloc::vec::Vec<PeripheralType> =
            result.iter().map(|d| d.peripheral_type).collect();
        assert_eq!(types.len(), 2);
        assert_eq!(types[0], PeripheralType::Uart);
        assert_eq!(types[1], PeripheralType::Spi);
    }

    #[test]
    fn test_discovery_csi_and_i2s() {
        let mut disc = PeripheralDiscovery::new();
        let nodes = [
            DtbNode::with_compatible(0xA000_0000, 0x10000, 100, b"nvidia,tegra-vi"),
            DtbNode::with_compatible(0xB000_0000, 0x1000, 101, b"nvidia,tegra-i2s"),
            DtbNode::with_compatible(0xC000_0000, 0x1000, 102, b"brcm,bcm2835-unicam"),
            DtbNode::with_compatible(0xD000_0000, 0x1000, 103, b"brcm,bcm2835-i2s"),
        ];
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, 4);
        assert_eq!(result.count_type(PeripheralType::Camera), 2);
        assert_eq!(result.count_type(PeripheralType::Audio), 2);
    }

    #[test]
    fn test_discovery_max_limit() {
        let mut disc = PeripheralDiscovery::new();
        // Create more nodes than MAX_DISCOVERED.
        let mut nodes = [DtbNode::empty(); 40];
        for (i, node) in nodes.iter_mut().enumerate() {
            *node = DtbNode::with_compatible((i as u64) * 0x1000, 0x100, i as u32, b"arm,pl011");
        }
        let result = disc.discover_all(&nodes);
        assert_eq!(result.count, MAX_DISCOVERED);
    }

    #[test]
    fn test_discovery_instance_count() {
        let mut disc = PeripheralDiscovery::new();
        assert_eq!(disc.instance_count(PeripheralType::I2c), 0);
        assert_eq!(disc.instance_count(PeripheralType::Uart), 0);

        let nodes = [
            DtbNode::with_compatible(0x1000, 0x100, 1, b"sifive,i2c0"),
            DtbNode::with_compatible(0x2000, 0x100, 2, b"sifive,i2c0"),
        ];
        let _ = disc.discover_all(&nodes);
        assert_eq!(disc.instance_count(PeripheralType::I2c), 2);
        assert_eq!(disc.instance_count(PeripheralType::Uart), 0);
    }

    #[test]
    fn test_byte_eq() {
        assert!(byte_eq(b"hello", b"hello"));
        assert!(!byte_eq(b"hello", b"world"));
        assert!(!byte_eq(b"hello", b"hell"));
        assert!(byte_eq(b"", b""));
    }

    #[test]
    fn test_discovery_result_empty() {
        let result = DiscoveryResult::empty();
        assert_eq!(result.count, 0);
        assert_eq!(result.as_slice().len(), 0);
        assert!(result.find_first(PeripheralType::I2c).is_none());
        assert_eq!(result.count_type(PeripheralType::I2c), 0);
    }

    // -- DtbNode compatible string truncation --------------------------------

    #[test]
    fn test_dtb_node_long_compatible_truncated() {
        let long_str = [b'x'; 128];
        let node = DtbNode::with_compatible(0x1000, 0x100, 5, &long_str);
        let compat = node.get_compatible(0).unwrap();
        // Should be truncated to MAX_COMPATIBLE_LEN - 1.
        assert_eq!(compat.len(), MAX_COMPATIBLE_LEN - 1);
    }
}
