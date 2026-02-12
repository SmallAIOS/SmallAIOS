// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Bus transport adapter registration and cross-transport routing.
//!
//! Extends the Zenoh-style IPC router with support for registering bus protocol
//! transports (CAN, ARINC 429, ARINC 664, MIL-STD-1553, SpaceWire, CCSDS, DDS)
//! and routing messages between heterogeneous transports.
//!
//! # Key Expression Mapping
//!
//! Each bus transport maps its native addressing to Zenoh key expressions:
//! - CAN: `can/{bus_id}/{frame_id}`
//! - ARINC 429: `arinc429/{channel}/{label}`
//! - ARINC 664: `afdx/{vl_id}`
//! - MIL-1553: `mil1553/{bus}/{rt}/{sa}`
//! - SpaceWire: `spw/{link}/{dest}`
//! - CCSDS: `ccsds/{apid}`
//! - DDS: `dds/{domain_id}/{topic}`

#![allow(dead_code)]

use alloc::string::String;
use alloc::vec::Vec;

use crate::key_expr::KeyExpr;
use crate::IpcError;

// ---------------------------------------------------------------------------
// Transport Identifier
// ---------------------------------------------------------------------------

/// Identifies a registered transport type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransportType {
    /// CAN bus transport.
    Can,
    /// ARINC 429 transport.
    Arinc429,
    /// ARINC 664 (AFDX) transport.
    Arinc664,
    /// MIL-STD-1553 transport.
    Mil1553,
    /// SpaceWire transport.
    SpaceWire,
    /// CCSDS Space Packet transport.
    Ccsds,
    /// DDS transport.
    Dds,
    /// TCP transport (built-in).
    Tcp,
    /// Shared memory transport (built-in).
    SharedMemory,
    /// QUIC transport.
    Quic,
}

impl TransportType {
    /// Returns the key expression prefix for this transport type.
    pub fn prefix(&self) -> &str {
        match self {
            TransportType::Can => "can",
            TransportType::Arinc429 => "arinc429",
            TransportType::Arinc664 => "afdx",
            TransportType::Mil1553 => "mil1553",
            TransportType::SpaceWire => "spw",
            TransportType::Ccsds => "ccsds",
            TransportType::Dds => "dds",
            TransportType::Tcp => "tcp",
            TransportType::SharedMemory => "shm",
            TransportType::Quic => "quic",
        }
    }
}

// ---------------------------------------------------------------------------
// Transport Registration
// ---------------------------------------------------------------------------

/// Maximum number of registered transports.
const MAX_TRANSPORTS: usize = 32;

/// Maximum number of bridge rules.
const MAX_BRIDGE_RULES: usize = 128;

/// Maximum number of queued messages for cross-transport routing.
const MAX_PENDING_MESSAGES: usize = 256;

/// A registered transport adapter.
#[derive(Debug, Clone, PartialEq)]
pub struct RegisteredTransport {
    /// Unique transport instance ID.
    pub id: u32,
    /// Transport type.
    pub transport_type: TransportType,
    /// Bus controller index (e.g., CAN bus 0 or 1).
    pub controller_index: u8,
    /// Key expression prefix (e.g., "can/0").
    pub key_prefix: String,
    /// Whether this transport is currently active.
    pub active: bool,
}

/// Discovery source for auto-detected transports.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiscoverySource {
    /// Discovered from Device Tree Blob.
    Dtb,
    /// Discovered from ACPI tables.
    Acpi,
    /// Manually registered.
    Manual,
}

/// A discovered peripheral that can be registered as a transport.
#[derive(Debug, Clone)]
pub struct DiscoveredPeripheral {
    /// Transport type.
    pub transport_type: TransportType,
    /// Controller index (for multiple controllers of the same type).
    pub controller_index: u8,
    /// Base MMIO address.
    pub base_address: u64,
    /// IRQ number.
    pub irq: u32,
    /// Discovery source.
    pub source: DiscoverySource,
    /// Compatible string from DTB (e.g., "xlnx,axi-can-1.00.a").
    pub compatible: String,
}

/// A bridge rule mapping messages between transports.
#[derive(Debug, Clone)]
pub struct BridgeRule {
    /// Source key expression pattern.
    pub source_key: KeyExpr,
    /// Destination key expression.
    pub dest_key: KeyExpr,
    /// Source transport type.
    pub source_transport: TransportType,
    /// Destination transport type.
    pub dest_transport: TransportType,
    /// Whether this rule is bidirectional.
    pub bidirectional: bool,
    /// Whether this rule is active.
    pub active: bool,
}

/// A message routed between transports.
#[derive(Debug, Clone)]
pub struct RoutedMessage {
    /// Source key expression.
    pub source_key: String,
    /// Destination key expression.
    pub dest_key: String,
    /// Message payload.
    pub payload: Vec<u8>,
    /// Source transport type.
    pub source_transport: TransportType,
    /// Destination transport type.
    pub dest_transport: TransportType,
    /// Timestamp in microseconds.
    pub timestamp_us: u64,
}

// ---------------------------------------------------------------------------
// Transport Router
// ---------------------------------------------------------------------------

/// Transport router managing bus protocol registrations, auto-discovery,
/// and cross-transport message routing.
#[derive(Debug)]
pub struct TransportRouter {
    /// Registered transport adapters.
    transports: Vec<RegisteredTransport>,
    /// Auto-discovered peripherals.
    discovered: Vec<DiscoveredPeripheral>,
    /// Bridge rules for cross-transport routing.
    bridge_rules: Vec<BridgeRule>,
    /// Pending messages awaiting delivery.
    pending_messages: Vec<RoutedMessage>,
    /// Next transport ID.
    next_id: u32,
}

impl Default for TransportRouter {
    fn default() -> Self {
        Self::new()
    }
}

impl TransportRouter {
    /// Create a new, empty transport router.
    pub fn new() -> Self {
        Self {
            transports: Vec::new(),
            discovered: Vec::new(),
            bridge_rules: Vec::new(),
            pending_messages: Vec::new(),
            next_id: 1,
        }
    }

    // -- Transport Registration (26.1) ------------------------------------------

    /// Register a bus transport adapter with the router.
    ///
    /// The transport is assigned a unique ID and associated with its key
    /// expression prefix (e.g., `can/0` for CAN bus 0).
    pub fn register_transport(
        &mut self,
        transport_type: TransportType,
        controller_index: u8,
    ) -> Result<u32, IpcError> {
        if self.transports.len() >= MAX_TRANSPORTS {
            return Err(IpcError::TableFull);
        }

        // Build the key prefix.
        let key_prefix = if controller_index > 0
            || matches!(
                transport_type,
                TransportType::Can
                    | TransportType::Arinc429
                    | TransportType::Mil1553
                    | TransportType::SpaceWire
            ) {
            alloc::format!("{}/{}", transport_type.prefix(), controller_index)
        } else {
            String::from(transport_type.prefix())
        };

        let id = self.next_id;
        self.next_id += 1;

        self.transports.push(RegisteredTransport {
            id,
            transport_type,
            controller_index,
            key_prefix,
            active: true,
        });

        Ok(id)
    }

    /// Deregister a transport by ID.
    pub fn deregister_transport(&mut self, id: u32) -> Result<(), IpcError> {
        for t in &mut self.transports {
            if t.id == id && t.active {
                t.active = false;
                return Ok(());
            }
        }
        Err(IpcError::NotFound)
    }

    /// Look up a registered transport by ID.
    pub fn get_transport(&self, id: u32) -> Option<&RegisteredTransport> {
        self.transports.iter().find(|t| t.id == id && t.active)
    }

    /// Check if a transport type with a given controller index is registered.
    pub fn is_transport_available(
        &self,
        transport_type: TransportType,
        controller_index: u8,
    ) -> bool {
        self.transports.iter().any(|t| {
            t.active && t.transport_type == transport_type && t.controller_index == controller_index
        })
    }

    /// Return the number of active registered transports.
    pub fn transport_count(&self) -> usize {
        self.transports.iter().filter(|t| t.active).count()
    }

    /// Find the transport responsible for a given key expression.
    ///
    /// Matches the key prefix against registered transports. Returns
    /// `Err(IpcError::NotFound)` if no transport handles this prefix,
    /// or a `TransportUnavailable`-style error.
    pub fn find_transport_for_key(&self, key: &str) -> Result<&RegisteredTransport, IpcError> {
        for t in &self.transports {
            if t.active && key.starts_with(&t.key_prefix) {
                return Ok(t);
            }
        }
        Err(IpcError::NotFound)
    }

    // -- Transport Auto-Discovery (26.2) ----------------------------------------

    /// Register a discovered peripheral from DTB or ACPI scanning.
    pub fn add_discovered_peripheral(
        &mut self,
        peripheral: DiscoveredPeripheral,
    ) -> Result<(), IpcError> {
        self.discovered.push(peripheral);
        Ok(())
    }

    /// Auto-register all discovered peripherals as transports.
    ///
    /// For each peripheral in the discovery list, registers a transport
    /// if one is not already registered for that type and controller index.
    pub fn auto_register_discovered(&mut self) -> Result<usize, IpcError> {
        let mut count = 0;
        // Collect info first to avoid borrow issues.
        let peripherals: Vec<(TransportType, u8)> = self
            .discovered
            .iter()
            .map(|p| (p.transport_type, p.controller_index))
            .collect();

        for (ttype, idx) in peripherals {
            if !self.is_transport_available(ttype, idx) {
                self.register_transport(ttype, idx)?;
                count += 1;
            }
        }
        Ok(count)
    }

    /// Return the number of discovered peripherals.
    pub fn discovered_count(&self) -> usize {
        self.discovered.len()
    }

    /// Parse a DTB compatible string and return the transport type.
    pub fn identify_peripheral(compatible: &str) -> Option<TransportType> {
        if compatible.contains("can")
            || compatible.contains("mcp2515")
            || compatible.contains("axi-can")
        {
            Some(TransportType::Can)
        } else if compatible.contains("arinc429") {
            Some(TransportType::Arinc429)
        } else if compatible.contains("afdx") || compatible.contains("arinc664") {
            Some(TransportType::Arinc664)
        } else if compatible.contains("mil1553") || compatible.contains("1553") {
            Some(TransportType::Mil1553)
        } else if compatible.contains("spacewire") || compatible.contains("spw") {
            Some(TransportType::SpaceWire)
        } else if compatible.contains("ccsds") {
            Some(TransportType::Ccsds)
        } else {
            None
        }
    }

    // -- Cross-Transport Routing (26.3) -----------------------------------------

    /// Add a bridge rule for cross-transport routing.
    pub fn add_bridge_rule(
        &mut self,
        source_key: KeyExpr,
        dest_key: KeyExpr,
        source_transport: TransportType,
        dest_transport: TransportType,
        bidirectional: bool,
    ) -> Result<(), IpcError> {
        if self.bridge_rules.len() >= MAX_BRIDGE_RULES {
            return Err(IpcError::TableFull);
        }
        self.bridge_rules.push(BridgeRule {
            source_key,
            dest_key,
            source_transport,
            dest_transport,
            bidirectional,
            active: true,
        });
        Ok(())
    }

    /// Remove all bridge rules matching the source key expression.
    pub fn remove_bridge_rules(&mut self, source_key: &KeyExpr) -> usize {
        let mut removed = 0;
        for rule in &mut self.bridge_rules {
            if rule.active && rule.source_key == *source_key {
                rule.active = false;
                removed += 1;
            }
        }
        removed
    }

    /// Return the number of active bridge rules.
    pub fn bridge_rule_count(&self) -> usize {
        self.bridge_rules.iter().filter(|r| r.active).count()
    }

    /// Route a message from a source key expression to all matching bridge rules.
    ///
    /// For each matching rule, creates a `RoutedMessage` in the pending queue
    /// with the destination key and transport type.
    pub fn route_message(
        &mut self,
        source_key: &KeyExpr,
        payload: &[u8],
        source_transport: TransportType,
        timestamp_us: u64,
    ) -> Result<usize, IpcError> {
        if self.pending_messages.len() >= MAX_PENDING_MESSAGES {
            return Err(IpcError::BufferFull);
        }

        let mut routed = 0;

        // Collect matching rules to avoid borrow issues.
        let matches: Vec<(String, String, TransportType)> = self
            .bridge_rules
            .iter()
            .filter(|r| {
                r.active
                    && (r.source_key.matches(source_key)
                        || (r.bidirectional && r.dest_key.matches(source_key)))
            })
            .map(|r| {
                if r.source_key.matches(source_key) {
                    (
                        r.source_key.as_str().into(),
                        r.dest_key.as_str().into(),
                        r.dest_transport,
                    )
                } else {
                    (
                        r.dest_key.as_str().into(),
                        r.source_key.as_str().into(),
                        r.source_transport,
                    )
                }
            })
            .collect();

        for (src, dest, dest_transport) in matches {
            if self.pending_messages.len() >= MAX_PENDING_MESSAGES {
                break;
            }
            self.pending_messages.push(RoutedMessage {
                source_key: src,
                dest_key: dest,
                payload: payload.into(),
                source_transport,
                dest_transport,
                timestamp_us,
            });
            routed += 1;
        }

        Ok(routed)
    }

    /// Drain all pending routed messages.
    pub fn drain_pending(&mut self) -> Vec<RoutedMessage> {
        core::mem::take(&mut self.pending_messages)
    }

    /// Return the number of pending routed messages.
    pub fn pending_count(&self) -> usize {
        self.pending_messages.len()
    }

    /// Find all bridge rules that match a given source key expression.
    pub fn find_bridge_rules(&self, source_key: &KeyExpr) -> Vec<&BridgeRule> {
        self.bridge_rules
            .iter()
            .filter(|r| r.active && r.source_key.matches(source_key))
            .collect()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn ke(s: &str) -> KeyExpr {
        KeyExpr::new(s).unwrap()
    }

    // -- TransportType ------------------------------------------------------

    #[test]
    fn test_transport_type_prefix() {
        assert_eq!(TransportType::Can.prefix(), "can");
        assert_eq!(TransportType::Arinc429.prefix(), "arinc429");
        assert_eq!(TransportType::Arinc664.prefix(), "afdx");
        assert_eq!(TransportType::Mil1553.prefix(), "mil1553");
        assert_eq!(TransportType::SpaceWire.prefix(), "spw");
        assert_eq!(TransportType::Ccsds.prefix(), "ccsds");
        assert_eq!(TransportType::Dds.prefix(), "dds");
        assert_eq!(TransportType::Tcp.prefix(), "tcp");
        assert_eq!(TransportType::SharedMemory.prefix(), "shm");
        assert_eq!(TransportType::Quic.prefix(), "quic");
    }

    // -- Transport Registration (26.1) --------------------------------------

    #[test]
    fn test_register_transport() {
        let mut router = TransportRouter::new();
        let id = router.register_transport(TransportType::Can, 0).unwrap();
        assert_eq!(router.transport_count(), 1);
        let t = router.get_transport(id).unwrap();
        assert_eq!(t.transport_type, TransportType::Can);
        assert_eq!(t.key_prefix, "can/0");
        assert!(t.active);
    }

    #[test]
    fn test_register_multiple_can_controllers() {
        let mut router = TransportRouter::new();
        let id0 = router.register_transport(TransportType::Can, 0).unwrap();
        let id1 = router.register_transport(TransportType::Can, 1).unwrap();
        assert_ne!(id0, id1);
        assert_eq!(router.transport_count(), 2);
        assert_eq!(router.get_transport(id0).unwrap().key_prefix, "can/0");
        assert_eq!(router.get_transport(id1).unwrap().key_prefix, "can/1");
    }

    #[test]
    fn test_register_different_types() {
        let mut router = TransportRouter::new();
        router.register_transport(TransportType::Can, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 0)
            .unwrap();
        router
            .register_transport(TransportType::Mil1553, 0)
            .unwrap();
        assert_eq!(router.transport_count(), 3);
    }

    #[test]
    fn test_deregister_transport() {
        let mut router = TransportRouter::new();
        let id = router.register_transport(TransportType::Can, 0).unwrap();
        assert_eq!(router.transport_count(), 1);
        router.deregister_transport(id).unwrap();
        assert_eq!(router.transport_count(), 0);
        assert!(router.get_transport(id).is_none());
    }

    #[test]
    fn test_deregister_nonexistent() {
        let mut router = TransportRouter::new();
        assert_eq!(router.deregister_transport(99), Err(IpcError::NotFound));
    }

    #[test]
    fn test_is_transport_available() {
        let mut router = TransportRouter::new();
        assert!(!router.is_transport_available(TransportType::Can, 0));
        router.register_transport(TransportType::Can, 0).unwrap();
        assert!(router.is_transport_available(TransportType::Can, 0));
        assert!(!router.is_transport_available(TransportType::Can, 1));
    }

    #[test]
    fn test_find_transport_for_key() {
        let mut router = TransportRouter::new();
        router.register_transport(TransportType::Can, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 1)
            .unwrap();

        let t = router.find_transport_for_key("can/0/0x1A3").unwrap();
        assert_eq!(t.transport_type, TransportType::Can);

        let t = router.find_transport_for_key("arinc429/1/0o215").unwrap();
        assert_eq!(t.transport_type, TransportType::Arinc429);

        assert_eq!(
            router.find_transport_for_key("unknown/x"),
            Err(IpcError::NotFound)
        );
    }

    #[test]
    fn test_transport_unavailable_subscribe_error() {
        let router = TransportRouter::new();
        // No ARINC 429 registered — lookup should fail.
        assert_eq!(
            router.find_transport_for_key("arinc429/0/0o100"),
            Err(IpcError::NotFound)
        );
    }

    // -- Auto-Discovery (26.2) ----------------------------------------------

    #[test]
    fn test_identify_peripheral_can() {
        assert_eq!(
            TransportRouter::identify_peripheral("xlnx,axi-can-1.00.a"),
            Some(TransportType::Can)
        );
        assert_eq!(
            TransportRouter::identify_peripheral("microchip,mcp2515"),
            Some(TransportType::Can)
        );
    }

    #[test]
    fn test_identify_peripheral_other() {
        assert_eq!(
            TransportRouter::identify_peripheral("acme,arinc429-rx"),
            Some(TransportType::Arinc429)
        );
        assert_eq!(
            TransportRouter::identify_peripheral("esa,spacewire-link"),
            Some(TransportType::SpaceWire)
        );
        assert_eq!(
            TransportRouter::identify_peripheral("mil1553-bc"),
            Some(TransportType::Mil1553)
        );
        assert_eq!(
            TransportRouter::identify_peripheral("acme,ccsds-spp"),
            Some(TransportType::Ccsds)
        );
    }

    #[test]
    fn test_identify_peripheral_unknown() {
        assert_eq!(TransportRouter::identify_peripheral("acme,uart"), None);
    }

    #[test]
    fn test_add_discovered_peripheral() {
        let mut router = TransportRouter::new();
        let peripheral = DiscoveredPeripheral {
            transport_type: TransportType::Can,
            controller_index: 0,
            base_address: 0x4000_0000,
            irq: 32,
            source: DiscoverySource::Dtb,
            compatible: String::from("xlnx,axi-can-1.00.a"),
        };
        router.add_discovered_peripheral(peripheral).unwrap();
        assert_eq!(router.discovered_count(), 1);
    }

    #[test]
    fn test_auto_register_discovered() {
        let mut router = TransportRouter::new();

        // Discover two CAN controllers and one ARINC 429.
        router
            .add_discovered_peripheral(DiscoveredPeripheral {
                transport_type: TransportType::Can,
                controller_index: 0,
                base_address: 0x4000_0000,
                irq: 32,
                source: DiscoverySource::Dtb,
                compatible: String::from("xlnx,axi-can-1.00.a"),
            })
            .unwrap();
        router
            .add_discovered_peripheral(DiscoveredPeripheral {
                transport_type: TransportType::Can,
                controller_index: 1,
                base_address: 0x4001_0000,
                irq: 33,
                source: DiscoverySource::Dtb,
                compatible: String::from("xlnx,axi-can-1.00.a"),
            })
            .unwrap();
        router
            .add_discovered_peripheral(DiscoveredPeripheral {
                transport_type: TransportType::Arinc429,
                controller_index: 0,
                base_address: 0x5000_0000,
                irq: 34,
                source: DiscoverySource::Acpi,
                compatible: String::from("acme,arinc429-rx"),
            })
            .unwrap();

        let count = router.auto_register_discovered().unwrap();
        assert_eq!(count, 3);
        assert_eq!(router.transport_count(), 3);
        assert!(router.is_transport_available(TransportType::Can, 0));
        assert!(router.is_transport_available(TransportType::Can, 1));
        assert!(router.is_transport_available(TransportType::Arinc429, 0));
    }

    #[test]
    fn test_auto_register_no_duplicates() {
        let mut router = TransportRouter::new();
        // Manually register CAN 0.
        router.register_transport(TransportType::Can, 0).unwrap();

        // Discover CAN 0 again.
        router
            .add_discovered_peripheral(DiscoveredPeripheral {
                transport_type: TransportType::Can,
                controller_index: 0,
                base_address: 0x4000_0000,
                irq: 32,
                source: DiscoverySource::Dtb,
                compatible: String::from("xlnx,axi-can-1.00.a"),
            })
            .unwrap();

        let count = router.auto_register_discovered().unwrap();
        assert_eq!(count, 0); // Already registered.
        assert_eq!(router.transport_count(), 1);
    }

    #[test]
    fn test_no_hardware_present() {
        let router = TransportRouter::new();
        // No peripherals discovered — ARINC 429 not available.
        assert!(!router.is_transport_available(TransportType::Arinc429, 0));
        assert_eq!(
            router.find_transport_for_key("arinc429/0/0o100"),
            Err(IpcError::NotFound)
        );
    }

    // -- Bridge Rules and Cross-Transport Routing (26.3) --------------------

    #[test]
    fn test_add_bridge_rule() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("mil1553/A/5/3"),
                ke("arinc429/0/0o310"),
                TransportType::Mil1553,
                TransportType::Arinc429,
                false,
            )
            .unwrap();
        assert_eq!(router.bridge_rule_count(), 1);
    }

    #[test]
    fn test_remove_bridge_rules() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("mil1553/A/5/3"),
                ke("arinc429/0/0o310"),
                TransportType::Mil1553,
                TransportType::Arinc429,
                false,
            )
            .unwrap();
        let removed = router.remove_bridge_rules(&ke("mil1553/A/5/3"));
        assert_eq!(removed, 1);
        assert_eq!(router.bridge_rule_count(), 0);
    }

    #[test]
    fn test_route_message_can_to_tcp() {
        let mut router = TransportRouter::new();
        router.register_transport(TransportType::Can, 0).unwrap();
        router.register_transport(TransportType::Tcp, 0).unwrap();

        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("smallaios/v1/sensor/data"),
                TransportType::Can,
                TransportType::Tcp,
                false,
            )
            .unwrap();

        let payload = [0x01, 0x02, 0x03, 0x04];
        let routed = router
            .route_message(&ke("can/0/0x100"), &payload, TransportType::Can, 1000)
            .unwrap();
        assert_eq!(routed, 1);
        assert_eq!(router.pending_count(), 1);

        let messages = router.drain_pending();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].source_key, "can/0/0x100");
        assert_eq!(messages[0].dest_key, "smallaios/v1/sensor/data");
        assert_eq!(messages[0].dest_transport, TransportType::Tcp);
        assert_eq!(messages[0].payload, &[0x01, 0x02, 0x03, 0x04]);
    }

    #[test]
    fn test_route_message_tcp_to_can() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("smallaios/v1/models/detector/output"),
                ke("can/0/0x200"),
                TransportType::Tcp,
                TransportType::Can,
                false,
            )
            .unwrap();

        let payload = [0xDE, 0xAD];
        let routed = router
            .route_message(
                &ke("smallaios/v1/models/detector/output"),
                &payload,
                TransportType::Tcp,
                2000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        let messages = router.drain_pending();
        assert_eq!(messages[0].dest_transport, TransportType::Can);
    }

    #[test]
    fn test_route_message_dds_to_can() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("dds/0/BrakeCommand"),
                ke("can/0/0x200"),
                TransportType::Dds,
                TransportType::Can,
                false,
            )
            .unwrap();

        let payload = [0xFF];
        let routed = router
            .route_message(
                &ke("dds/0/BrakeCommand"),
                &payload,
                TransportType::Dds,
                3000,
            )
            .unwrap();
        assert_eq!(routed, 1);
    }

    #[test]
    fn test_route_message_mil1553_to_arinc429() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("mil1553/A/5/3"),
                ke("arinc429/0/0o310"),
                TransportType::Mil1553,
                TransportType::Arinc429,
                false,
            )
            .unwrap();

        let payload = [0x12, 0x34];
        let routed = router
            .route_message(&ke("mil1553/A/5/3"), &payload, TransportType::Mil1553, 4000)
            .unwrap();
        assert_eq!(routed, 1);

        let messages = router.drain_pending();
        assert_eq!(messages[0].dest_transport, TransportType::Arinc429);
    }

    #[test]
    fn test_route_message_bidirectional() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("dds/0/SensorData"),
                TransportType::Can,
                TransportType::Dds,
                true,
            )
            .unwrap();

        // Forward: CAN -> DDS.
        let routed = router
            .route_message(&ke("can/0/0x100"), &[0x01], TransportType::Can, 1000)
            .unwrap();
        assert_eq!(routed, 1);

        // Reverse: DDS -> CAN.
        let routed = router
            .route_message(&ke("dds/0/SensorData"), &[0x02], TransportType::Dds, 2000)
            .unwrap();
        assert_eq!(routed, 1);

        let messages = router.drain_pending();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].dest_transport, TransportType::Dds);
        assert_eq!(messages[1].dest_transport, TransportType::Can);
    }

    #[test]
    fn test_route_message_no_match() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("dds/0/SensorData"),
                TransportType::Can,
                TransportType::Dds,
                false,
            )
            .unwrap();

        let routed = router
            .route_message(&ke("can/0/0x200"), &[0x01], TransportType::Can, 1000)
            .unwrap();
        assert_eq!(routed, 0);
    }

    #[test]
    fn test_route_message_wildcard_rule() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("smallaios/v1/telemetry/**"),
                ke("spw/0/42"),
                TransportType::SharedMemory,
                TransportType::SpaceWire,
                false,
            )
            .unwrap();

        let routed = router
            .route_message(
                &ke("smallaios/v1/telemetry/attitude"),
                &[0x01],
                TransportType::SharedMemory,
                5000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        let messages = router.drain_pending();
        assert_eq!(messages[0].dest_transport, TransportType::SpaceWire);
    }

    #[test]
    fn test_route_message_multiple_rules() {
        let mut router = TransportRouter::new();
        // Same source, two destinations.
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("dds/0/SensorA"),
                TransportType::Can,
                TransportType::Dds,
                false,
            )
            .unwrap();
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("arinc429/0/0o200"),
                TransportType::Can,
                TransportType::Arinc429,
                false,
            )
            .unwrap();

        let routed = router
            .route_message(&ke("can/0/0x100"), &[0x01], TransportType::Can, 1000)
            .unwrap();
        assert_eq!(routed, 2);
    }

    #[test]
    fn test_drain_pending_clears() {
        let mut router = TransportRouter::new();
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("dds/0/X"),
                TransportType::Can,
                TransportType::Dds,
                false,
            )
            .unwrap();

        router
            .route_message(&ke("can/0/0x100"), &[0x01], TransportType::Can, 1000)
            .unwrap();
        assert_eq!(router.pending_count(), 1);

        let _ = router.drain_pending();
        assert_eq!(router.pending_count(), 0);
    }

    // -- Integration scenario (26.5): inference result over TCP to CAN ------

    #[test]
    fn test_inference_tcp_to_can_integration() {
        let mut router = TransportRouter::new();

        // Register transports.
        router.register_transport(TransportType::Tcp, 0).unwrap();
        router.register_transport(TransportType::Can, 0).unwrap();

        // Bridge: inference output -> CAN frame 0x200.
        router
            .add_bridge_rule(
                ke("smallaios/v1/models/detector/output"),
                ke("can/0/0x200"),
                TransportType::Tcp,
                TransportType::Can,
                false,
            )
            .unwrap();

        // Simulate inference result published over TCP.
        let inference_output = [0x01, 0x00, 0x5A, 0x3C]; // detection result
        let routed = router
            .route_message(
                &ke("smallaios/v1/models/detector/output"),
                &inference_output,
                TransportType::Tcp,
                10_000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        let messages = router.drain_pending();
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0].dest_key, "can/0/0x200");
        assert_eq!(messages[0].dest_transport, TransportType::Can);
        assert_eq!(messages[0].payload, &inference_output);
        assert_eq!(messages[0].timestamp_us, 10_000);
    }

    // -- Mixed transport pub/sub (26.4) -------------------------------------

    #[test]
    fn test_mixed_transport_routing() {
        let mut router = TransportRouter::new();

        // Register all transport types.
        router.register_transport(TransportType::Can, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 0)
            .unwrap();
        router
            .register_transport(TransportType::Mil1553, 0)
            .unwrap();
        router
            .register_transport(TransportType::SpaceWire, 0)
            .unwrap();
        router.register_transport(TransportType::Tcp, 0).unwrap();
        router.register_transport(TransportType::Dds, 0).unwrap();

        // CAN -> DDS.
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("dds/0/SensorData"),
                TransportType::Can,
                TransportType::Dds,
                false,
            )
            .unwrap();

        // DDS -> ARINC 429.
        router
            .add_bridge_rule(
                ke("dds/0/CommandData"),
                ke("arinc429/0/0o215"),
                TransportType::Dds,
                TransportType::Arinc429,
                false,
            )
            .unwrap();

        // MIL-1553 -> SpaceWire.
        router
            .add_bridge_rule(
                ke("mil1553/0/5/3"),
                ke("spw/0/42"),
                TransportType::Mil1553,
                TransportType::SpaceWire,
                false,
            )
            .unwrap();

        // Route CAN message -> should go to DDS.
        router
            .route_message(&ke("can/0/0x100"), &[0x01], TransportType::Can, 1000)
            .unwrap();

        // Route DDS message -> should go to ARINC 429.
        router
            .route_message(&ke("dds/0/CommandData"), &[0x02], TransportType::Dds, 2000)
            .unwrap();

        // Route MIL-1553 message -> should go to SpaceWire.
        router
            .route_message(&ke("mil1553/0/5/3"), &[0x03], TransportType::Mil1553, 3000)
            .unwrap();

        let messages = router.drain_pending();
        assert_eq!(messages.len(), 3);
        assert_eq!(messages[0].dest_transport, TransportType::Dds);
        assert_eq!(messages[1].dest_transport, TransportType::Arinc429);
        assert_eq!(messages[2].dest_transport, TransportType::SpaceWire);
    }

    // -- QUIC session transport (26.7) reference test -----------------------

    #[test]
    fn test_quic_transport_registration() {
        let mut router = TransportRouter::new();
        router.register_transport(TransportType::Quic, 0).unwrap();
        assert!(router.is_transport_available(TransportType::Quic, 0));
        assert!(router.find_transport_for_key("quic/session/1").is_ok());
    }

    // -- Integration test (26.6): DDS topic bridged to ARINC 429 -----------

    #[test]
    fn test_dds_topic_bridged_to_arinc429() {
        let mut router = TransportRouter::new();

        // Register DDS and ARINC 429 transports.
        router.register_transport(TransportType::Dds, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 0)
            .unwrap();

        // Bridge rule: DDS altitude topic -> ARINC 429 label 203 (octal).
        router
            .add_bridge_rule(
                ke("dds/0/FlightData/Altitude"),
                ke("arinc429/0/0o313"),
                TransportType::Dds,
                TransportType::Arinc429,
                false,
            )
            .unwrap();

        // Bridge rule: DDS airspeed topic -> ARINC 429 label 206.
        router
            .add_bridge_rule(
                ke("dds/0/FlightData/Airspeed"),
                ke("arinc429/0/0o316"),
                TransportType::Dds,
                TransportType::Arinc429,
                false,
            )
            .unwrap();

        assert_eq!(router.bridge_rule_count(), 2);

        // Simulate DDS altitude publication (float32 35000.0 in big-endian).
        let altitude_payload = 35000.0f32.to_be_bytes();
        let routed = router
            .route_message(
                &ke("dds/0/FlightData/Altitude"),
                &altitude_payload,
                TransportType::Dds,
                100_000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        // Simulate DDS airspeed publication (float32 250.5 in big-endian).
        let airspeed_payload = 250.5f32.to_be_bytes();
        let routed = router
            .route_message(
                &ke("dds/0/FlightData/Airspeed"),
                &airspeed_payload,
                TransportType::Dds,
                100_001,
            )
            .unwrap();
        assert_eq!(routed, 1);

        // Drain and verify both routed messages.
        let messages = router.drain_pending();
        assert_eq!(messages.len(), 2);

        // First message: altitude -> ARINC 429 label 0o313.
        assert_eq!(messages[0].source_key, "dds/0/FlightData/Altitude");
        assert_eq!(messages[0].dest_key, "arinc429/0/0o313");
        assert_eq!(messages[0].source_transport, TransportType::Dds);
        assert_eq!(messages[0].dest_transport, TransportType::Arinc429);
        assert_eq!(messages[0].payload, altitude_payload);
        assert_eq!(messages[0].timestamp_us, 100_000);

        // Second message: airspeed -> ARINC 429 label 0o316.
        assert_eq!(messages[1].source_key, "dds/0/FlightData/Airspeed");
        assert_eq!(messages[1].dest_key, "arinc429/0/0o316");
        assert_eq!(messages[1].dest_transport, TransportType::Arinc429);
        assert_eq!(messages[1].payload, airspeed_payload);

        // Verify no unmatched DDS topics leak through.
        let routed = router
            .route_message(
                &ke("dds/0/FlightData/Heading"),
                &[0x00; 4],
                TransportType::Dds,
                100_002,
            )
            .unwrap();
        assert_eq!(routed, 0);
        assert_eq!(router.pending_count(), 0);
    }

    #[test]
    fn test_dds_arinc429_bidirectional_bridge() {
        let mut router = TransportRouter::new();

        router.register_transport(TransportType::Dds, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 0)
            .unwrap();

        // Bidirectional bridge: DDS <-> ARINC 429.
        router
            .add_bridge_rule(
                ke("dds/0/NavData/Heading"),
                ke("arinc429/0/0o314"),
                TransportType::Dds,
                TransportType::Arinc429,
                true,
            )
            .unwrap();

        // Forward: DDS -> ARINC 429.
        let routed = router
            .route_message(
                &ke("dds/0/NavData/Heading"),
                &180.0f32.to_be_bytes(),
                TransportType::Dds,
                200_000,
            )
            .unwrap();
        assert_eq!(routed, 1);
        let msgs = router.drain_pending();
        assert_eq!(msgs[0].dest_transport, TransportType::Arinc429);

        // Reverse: ARINC 429 -> DDS.
        let routed = router
            .route_message(
                &ke("arinc429/0/0o314"),
                &[0x12, 0x34, 0x56, 0x78],
                TransportType::Arinc429,
                200_001,
            )
            .unwrap();
        assert_eq!(routed, 1);
        let msgs = router.drain_pending();
        assert_eq!(msgs[0].dest_transport, TransportType::Dds);
        assert_eq!(msgs[0].dest_key, "dds/0/NavData/Heading");
    }

    #[test]
    fn test_dds_arinc429_wildcard_topic_bridge() {
        let mut router = TransportRouter::new();

        router.register_transport(TransportType::Dds, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 0)
            .unwrap();

        // Wildcard: all DDS FlightData subtopics -> single ARINC 429 label.
        router
            .add_bridge_rule(
                ke("dds/0/FlightData/**"),
                ke("arinc429/0/0o300"),
                TransportType::Dds,
                TransportType::Arinc429,
                false,
            )
            .unwrap();

        // Route altitude.
        let routed = router
            .route_message(
                &ke("dds/0/FlightData/Altitude"),
                &[0x01],
                TransportType::Dds,
                300_000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        // Route airspeed.
        let routed = router
            .route_message(
                &ke("dds/0/FlightData/Airspeed"),
                &[0x02],
                TransportType::Dds,
                300_001,
            )
            .unwrap();
        assert_eq!(routed, 1);

        // Route heading.
        let routed = router
            .route_message(
                &ke("dds/0/FlightData/Heading"),
                &[0x03],
                TransportType::Dds,
                300_002,
            )
            .unwrap();
        assert_eq!(routed, 1);

        let msgs = router.drain_pending();
        assert_eq!(msgs.len(), 3);
        for msg in &msgs {
            assert_eq!(msg.dest_key, "arinc429/0/0o300");
            assert_eq!(msg.dest_transport, TransportType::Arinc429);
        }
    }

    // -- Integration test (26.7): Zenoh session over QUIC with migration ---

    #[test]
    fn test_zenoh_quic_session_with_migration() {
        let mut router = TransportRouter::new();

        // Register QUIC transport and a CAN transport as data source.
        router.register_transport(TransportType::Quic, 0).unwrap();
        router.register_transport(TransportType::Can, 0).unwrap();

        // Bridge: CAN sensor data -> QUIC session for remote delivery.
        router
            .add_bridge_rule(
                ke("can/0/0x100"),
                ke("quic/session/sensor_data"),
                TransportType::Can,
                TransportType::Quic,
                false,
            )
            .unwrap();

        // Simulate sensor data arriving on CAN bus.
        let sensor_payload = [0xDE, 0xAD, 0xBE, 0xEF];
        let routed = router
            .route_message(
                &ke("can/0/0x100"),
                &sensor_payload,
                TransportType::Can,
                400_000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        let msgs = router.drain_pending();
        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].dest_key, "quic/session/sensor_data");
        assert_eq!(msgs[0].dest_transport, TransportType::Quic);
        assert_eq!(msgs[0].payload, sensor_payload);

        // Simulate connection migration: deregister old QUIC, register new.
        // In a real scenario, MigrationManager handles path validation.
        // Here we verify the router keeps routing after transport re-registration.
        let quic_id = router
            .find_transport_for_key("quic/session/sensor_data")
            .unwrap()
            .id;
        router.deregister_transport(quic_id).unwrap();
        assert!(!router.is_transport_available(TransportType::Quic, 0));

        // Re-register QUIC (simulating migration to new path).
        router.register_transport(TransportType::Quic, 0).unwrap();
        assert!(router.is_transport_available(TransportType::Quic, 0));

        // Bridge rules survive transport re-registration.
        // Route another sensor message after migration.
        let sensor_payload_2 = [0xCA, 0xFE, 0xBA, 0xBE];
        let routed = router
            .route_message(
                &ke("can/0/0x100"),
                &sensor_payload_2,
                TransportType::Can,
                500_000,
            )
            .unwrap();
        assert_eq!(routed, 1);

        let msgs = router.drain_pending();
        assert_eq!(msgs[0].dest_transport, TransportType::Quic);
        assert_eq!(msgs[0].payload, sensor_payload_2);
    }

    #[test]
    fn test_quic_session_multiple_streams() {
        let mut router = TransportRouter::new();

        router.register_transport(TransportType::Quic, 0).unwrap();
        router.register_transport(TransportType::Dds, 0).unwrap();
        router
            .register_transport(TransportType::Arinc429, 0)
            .unwrap();

        // Multiple streams over QUIC from different sources.
        router
            .add_bridge_rule(
                ke("dds/0/Telemetry"),
                ke("quic/stream/telemetry"),
                TransportType::Dds,
                TransportType::Quic,
                false,
            )
            .unwrap();
        router
            .add_bridge_rule(
                ke("arinc429/0/0o200"),
                ke("quic/stream/nav"),
                TransportType::Arinc429,
                TransportType::Quic,
                false,
            )
            .unwrap();

        // DDS telemetry -> QUIC.
        router
            .route_message(
                &ke("dds/0/Telemetry"),
                &[0x01, 0x02],
                TransportType::Dds,
                600_000,
            )
            .unwrap();

        // ARINC 429 nav -> QUIC.
        router
            .route_message(
                &ke("arinc429/0/0o200"),
                &[0x03, 0x04],
                TransportType::Arinc429,
                600_001,
            )
            .unwrap();

        let msgs = router.drain_pending();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[0].dest_key, "quic/stream/telemetry");
        assert_eq!(msgs[0].dest_transport, TransportType::Quic);
        assert_eq!(msgs[1].dest_key, "quic/stream/nav");
        assert_eq!(msgs[1].dest_transport, TransportType::Quic);
    }
}
