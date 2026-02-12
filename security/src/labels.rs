// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! MAC security labels: classification + integrity + message type.
//!
//! Labels augment the existing capability system — they do not replace it.
//! Every data item crossing a trust boundary carries a label that is checked
//! by the SecurityGate in addition to capability verification.
//!
//! - Classification: Bell-LaPadula (no read-up, no write-down) — reuses
//!   [`crate::compliance::classification::ClassificationLevel`]
//! - Integrity: Biba model (no write-up, no read-down)
//! - Message type: links to a formally verified type in the registry

use crate::compliance::classification::ClassificationLevel;

/// Integrity levels following the Biba integrity model.
///
/// Data at a lower integrity level MUST NOT flow to a higher-integrity
/// destination without passing through an explicit integrity promotion gate.
///
/// - Low: untrusted external source (network inbound)
/// - Medium: authenticated but not cryptographically verified (bus sensors, inference output)
/// - High: kernel-generated or promoted through verification (actuator commands)
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum IntegrityLevel {
    /// Untrusted external source.
    Low = 0,
    /// Authenticated but not fully verified.
    Medium = 1,
    /// Kernel-generated or promoted through verification.
    High = 2,
}

impl IntegrityLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Low),
            1 => Some(Self::Medium),
            2 => Some(Self::High),
            _ => None,
        }
    }

    /// Check whether data at this integrity level may flow to a destination
    /// requiring `dest` integrity (Biba: source >= destination).
    pub fn may_flow_to(self, dest: IntegrityLevel) -> bool {
        self >= dest
    }
}

/// Unique identifier for a verified message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MessageTypeId(pub u32);

impl MessageTypeId {
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }
}

/// MAC security label attached to data crossing trust boundaries.
///
/// Combines confidentiality (classification), integrity (Biba), and
/// an optional verified message type reference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SecurityLabel {
    /// Confidentiality level (Bell-LaPadula).
    pub classification: ClassificationLevel,
    /// Integrity level (Biba).
    pub integrity: IntegrityLevel,
    /// Verified message type (None = untyped).
    pub message_type: Option<MessageTypeId>,
}

impl SecurityLabel {
    pub const fn new(
        classification: ClassificationLevel,
        integrity: IntegrityLevel,
        message_type: Option<MessageTypeId>,
    ) -> Self {
        Self {
            classification,
            integrity,
            message_type,
        }
    }

    /// Label for untyped, untrusted, public data.
    pub const fn untrusted_public() -> Self {
        Self {
            classification: ClassificationLevel::Public,
            integrity: IntegrityLevel::Low,
            message_type: None,
        }
    }

    /// Label for internal data at medium integrity (e.g. inference I/O).
    pub const fn internal_medium() -> Self {
        Self {
            classification: ClassificationLevel::Internal,
            integrity: IntegrityLevel::Medium,
            message_type: None,
        }
    }

    /// Label for kernel-internal high-integrity data.
    pub const fn kernel_internal() -> Self {
        Self {
            classification: ClassificationLevel::Restricted,
            integrity: IntegrityLevel::High,
            message_type: None,
        }
    }

    /// Whether this label has a verified message type.
    pub fn is_typed(&self) -> bool {
        self.message_type.is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── IntegrityLevel ──

    #[test]
    fn integrity_ordering() {
        assert!(IntegrityLevel::Low < IntegrityLevel::Medium);
        assert!(IntegrityLevel::Medium < IntegrityLevel::High);
        assert!(IntegrityLevel::Low < IntegrityLevel::High);
    }

    #[test]
    fn integrity_equality() {
        assert_eq!(IntegrityLevel::Low, IntegrityLevel::Low);
        assert_ne!(IntegrityLevel::Low, IntegrityLevel::High);
    }

    #[test]
    fn integrity_from_u8() {
        assert_eq!(IntegrityLevel::from_u8(0), Some(IntegrityLevel::Low));
        assert_eq!(IntegrityLevel::from_u8(1), Some(IntegrityLevel::Medium));
        assert_eq!(IntegrityLevel::from_u8(2), Some(IntegrityLevel::High));
        assert_eq!(IntegrityLevel::from_u8(3), None);
        assert_eq!(IntegrityLevel::from_u8(255), None);
    }

    #[test]
    fn integrity_as_str() {
        assert_eq!(IntegrityLevel::Low.as_str(), "low");
        assert_eq!(IntegrityLevel::Medium.as_str(), "medium");
        assert_eq!(IntegrityLevel::High.as_str(), "high");
    }

    #[test]
    fn biba_flow_high_to_low_allowed() {
        assert!(IntegrityLevel::High.may_flow_to(IntegrityLevel::Low));
    }

    #[test]
    fn biba_flow_high_to_high_allowed() {
        assert!(IntegrityLevel::High.may_flow_to(IntegrityLevel::High));
    }

    #[test]
    fn biba_flow_low_to_high_denied() {
        assert!(!IntegrityLevel::Low.may_flow_to(IntegrityLevel::High));
    }

    #[test]
    fn biba_flow_low_to_medium_denied() {
        assert!(!IntegrityLevel::Low.may_flow_to(IntegrityLevel::Medium));
    }

    #[test]
    fn biba_flow_medium_to_high_denied() {
        assert!(!IntegrityLevel::Medium.may_flow_to(IntegrityLevel::High));
    }

    #[test]
    fn biba_flow_medium_to_low_allowed() {
        assert!(IntegrityLevel::Medium.may_flow_to(IntegrityLevel::Low));
    }

    #[test]
    fn biba_flow_medium_to_medium_allowed() {
        assert!(IntegrityLevel::Medium.may_flow_to(IntegrityLevel::Medium));
    }

    // ── MessageTypeId ──

    #[test]
    fn message_type_id_new() {
        let id = MessageTypeId::new(0x0001);
        assert_eq!(id.raw(), 0x0001);
    }

    #[test]
    fn message_type_id_equality() {
        assert_eq!(MessageTypeId(1), MessageTypeId(1));
        assert_ne!(MessageTypeId(1), MessageTypeId(2));
    }

    // ── SecurityLabel ──

    #[test]
    fn security_label_new() {
        let label = SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::Medium,
            Some(MessageTypeId(0x0001)),
        );
        assert_eq!(label.classification, ClassificationLevel::Internal);
        assert_eq!(label.integrity, IntegrityLevel::Medium);
        assert_eq!(label.message_type, Some(MessageTypeId(0x0001)));
        assert!(label.is_typed());
    }

    #[test]
    fn security_label_untyped() {
        let label = SecurityLabel::new(ClassificationLevel::Public, IntegrityLevel::Low, None);
        assert!(!label.is_typed());
    }

    #[test]
    fn security_label_presets() {
        let untrusted = SecurityLabel::untrusted_public();
        assert_eq!(untrusted.classification, ClassificationLevel::Public);
        assert_eq!(untrusted.integrity, IntegrityLevel::Low);
        assert!(!untrusted.is_typed());

        let internal = SecurityLabel::internal_medium();
        assert_eq!(internal.classification, ClassificationLevel::Internal);
        assert_eq!(internal.integrity, IntegrityLevel::Medium);

        let kernel = SecurityLabel::kernel_internal();
        assert_eq!(kernel.classification, ClassificationLevel::Restricted);
        assert_eq!(kernel.integrity, IntegrityLevel::High);
    }

    #[test]
    fn security_label_equality() {
        let a = SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::Medium,
            Some(MessageTypeId(1)),
        );
        let b = SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::Medium,
            Some(MessageTypeId(1)),
        );
        let c = SecurityLabel::new(
            ClassificationLevel::Internal,
            IntegrityLevel::High,
            Some(MessageTypeId(1)),
        );
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
