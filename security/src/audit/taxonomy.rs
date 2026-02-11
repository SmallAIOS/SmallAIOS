// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Structured audit event taxonomy.
//!
//! Events are classified into categories and subcategories per NIST AU-2:
//! - **Capability**: grant, revoke, deny
//! - **Authentication**: TLS handshake success/failure
//! - **Resource**: allocation, exhaustion
//! - **Inference**: request, completion, timeout
//! - **System**: boot, shutdown, watchdog

/// Top-level event category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventCategory {
    /// Capability-related events (grant, revoke, deny).
    Capability = 0,
    /// Authentication events (TLS handshake success/failure).
    Authentication = 1,
    /// Resource lifecycle events (allocation, exhaustion).
    Resource = 2,
    /// Inference pipeline events (request, completion, timeout).
    Inference = 3,
    /// System-level events (boot, shutdown, watchdog).
    System = 4,
    /// Security gate events (formal methods firewall decisions).
    Security = 5,
}

impl EventCategory {
    /// Convert from u8 representation.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Capability),
            1 => Some(Self::Authentication),
            2 => Some(Self::Resource),
            3 => Some(Self::Inference),
            4 => Some(Self::System),
            5 => Some(Self::Security),
            _ => None,
        }
    }

    /// Short name for syslog serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Capability => "capability",
            Self::Authentication => "auth",
            Self::Resource => "resource",
            Self::Inference => "inference",
            Self::System => "system",
            Self::Security => "security",
        }
    }
}

/// Event subcategory within a category.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventSubcategory {
    // Capability subcategories
    /// A capability was granted to a task.
    Grant = 0,
    /// A capability was revoked from a task.
    Revoke = 1,
    /// A capability request was denied.
    Deny = 2,

    // Authentication subcategories
    /// TLS handshake succeeded.
    TlsSuccess = 10,
    /// TLS handshake failed.
    TlsFailure = 11,

    // Resource subcategories
    /// Resource was allocated.
    Allocation = 20,
    /// Resource exhaustion occurred.
    Exhaustion = 21,

    // Inference subcategories
    /// Inference request submitted.
    Request = 30,
    /// Inference completed successfully.
    Completion = 31,
    /// Inference timed out.
    Timeout = 32,

    // System subcategories
    /// System boot completed.
    Boot = 40,
    /// System shutdown initiated.
    Shutdown = 41,
    /// Watchdog event (reset/warning).
    Watchdog = 42,

    // Security gate subcategories
    /// Gate check passed — all layers verified.
    GateAllowed = 50,
    /// Gate check denied — a verification layer failed (Enforcing mode).
    GateDenied = 51,
    /// Gate check would have denied but mode is Permissive.
    GatePermissivePass = 52,
    /// Security policy was updated.
    PolicyUpdate = 53,
}

impl EventSubcategory {
    /// Convert from u8 representation.
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Grant),
            1 => Some(Self::Revoke),
            2 => Some(Self::Deny),
            10 => Some(Self::TlsSuccess),
            11 => Some(Self::TlsFailure),
            20 => Some(Self::Allocation),
            21 => Some(Self::Exhaustion),
            30 => Some(Self::Request),
            31 => Some(Self::Completion),
            32 => Some(Self::Timeout),
            40 => Some(Self::Boot),
            41 => Some(Self::Shutdown),
            42 => Some(Self::Watchdog),
            50 => Some(Self::GateAllowed),
            51 => Some(Self::GateDenied),
            52 => Some(Self::GatePermissivePass),
            53 => Some(Self::PolicyUpdate),
            _ => None,
        }
    }

    /// Short name for syslog serialization.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Grant => "grant",
            Self::Revoke => "revoke",
            Self::Deny => "deny",
            Self::TlsSuccess => "tls-success",
            Self::TlsFailure => "tls-failure",
            Self::Allocation => "allocation",
            Self::Exhaustion => "exhaustion",
            Self::Request => "request",
            Self::Completion => "completion",
            Self::Timeout => "timeout",
            Self::Boot => "boot",
            Self::Shutdown => "shutdown",
            Self::Watchdog => "watchdog",
            Self::GateAllowed => "gate-allowed",
            Self::GateDenied => "gate-denied",
            Self::GatePermissivePass => "gate-permissive-pass",
            Self::PolicyUpdate => "policy-update",
        }
    }

    /// The category this subcategory belongs to.
    pub fn category(&self) -> EventCategory {
        match self {
            Self::Grant | Self::Revoke | Self::Deny => EventCategory::Capability,
            Self::TlsSuccess | Self::TlsFailure => EventCategory::Authentication,
            Self::Allocation | Self::Exhaustion => EventCategory::Resource,
            Self::Request | Self::Completion | Self::Timeout => EventCategory::Inference,
            Self::Boot | Self::Shutdown | Self::Watchdog => EventCategory::System,
            Self::GateAllowed
            | Self::GateDenied
            | Self::GatePermissivePass
            | Self::PolicyUpdate => EventCategory::Security,
        }
    }
}

/// Combined event type encoding both category and subcategory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AuditEventType {
    pub category: EventCategory,
    pub subcategory: EventSubcategory,
}

impl AuditEventType {
    /// Create a new event type, validating that the subcategory belongs to the category.
    pub fn new(category: EventCategory, subcategory: EventSubcategory) -> Option<Self> {
        if subcategory.category() == category {
            Some(Self {
                category,
                subcategory,
            })
        } else {
            None
        }
    }

    /// Capability grant event.
    pub const fn capability_grant() -> Self {
        Self {
            category: EventCategory::Capability,
            subcategory: EventSubcategory::Grant,
        }
    }

    /// Capability revoke event.
    pub const fn capability_revoke() -> Self {
        Self {
            category: EventCategory::Capability,
            subcategory: EventSubcategory::Revoke,
        }
    }

    /// Capability deny event.
    pub const fn capability_deny() -> Self {
        Self {
            category: EventCategory::Capability,
            subcategory: EventSubcategory::Deny,
        }
    }

    /// TLS handshake success event.
    pub const fn auth_success() -> Self {
        Self {
            category: EventCategory::Authentication,
            subcategory: EventSubcategory::TlsSuccess,
        }
    }

    /// TLS handshake failure event.
    pub const fn auth_failure() -> Self {
        Self {
            category: EventCategory::Authentication,
            subcategory: EventSubcategory::TlsFailure,
        }
    }

    /// Resource allocation event.
    pub const fn resource_allocation() -> Self {
        Self {
            category: EventCategory::Resource,
            subcategory: EventSubcategory::Allocation,
        }
    }

    /// Resource exhaustion event.
    pub const fn resource_exhaustion() -> Self {
        Self {
            category: EventCategory::Resource,
            subcategory: EventSubcategory::Exhaustion,
        }
    }

    /// Inference request event.
    pub const fn inference_request() -> Self {
        Self {
            category: EventCategory::Inference,
            subcategory: EventSubcategory::Request,
        }
    }

    /// Inference completion event.
    pub const fn inference_completion() -> Self {
        Self {
            category: EventCategory::Inference,
            subcategory: EventSubcategory::Completion,
        }
    }

    /// Inference timeout event.
    pub const fn inference_timeout() -> Self {
        Self {
            category: EventCategory::Inference,
            subcategory: EventSubcategory::Timeout,
        }
    }

    /// System boot event.
    pub const fn system_boot() -> Self {
        Self {
            category: EventCategory::System,
            subcategory: EventSubcategory::Boot,
        }
    }

    /// System shutdown event.
    pub const fn system_shutdown() -> Self {
        Self {
            category: EventCategory::System,
            subcategory: EventSubcategory::Shutdown,
        }
    }

    /// System watchdog event.
    pub const fn system_watchdog() -> Self {
        Self {
            category: EventCategory::System,
            subcategory: EventSubcategory::Watchdog,
        }
    }

    /// Security gate: all layers passed.
    pub const fn gate_allowed() -> Self {
        Self {
            category: EventCategory::Security,
            subcategory: EventSubcategory::GateAllowed,
        }
    }

    /// Security gate: denied in Enforcing mode.
    pub const fn gate_denied() -> Self {
        Self {
            category: EventCategory::Security,
            subcategory: EventSubcategory::GateDenied,
        }
    }

    /// Security gate: would have denied but Permissive mode.
    pub const fn gate_permissive_pass() -> Self {
        Self {
            category: EventCategory::Security,
            subcategory: EventSubcategory::GatePermissivePass,
        }
    }

    /// Security policy update event.
    pub const fn policy_update() -> Self {
        Self {
            category: EventCategory::Security,
            subcategory: EventSubcategory::PolicyUpdate,
        }
    }

    /// Encode as a two-byte value: [category, subcategory].
    pub fn to_bytes(&self) -> [u8; 2] {
        [self.category as u8, self.subcategory as u8]
    }

    /// Decode from a two-byte value.
    pub fn from_bytes(bytes: [u8; 2]) -> Option<Self> {
        let category = EventCategory::from_u8(bytes[0])?;
        let subcategory = EventSubcategory::from_u8(bytes[1])?;
        Self::new(category, subcategory)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn category_from_u8_roundtrip() {
        for i in 0..=5u8 {
            let cat = EventCategory::from_u8(i).unwrap();
            assert_eq!(cat as u8, i);
        }
        assert!(EventCategory::from_u8(6).is_none());
        assert!(EventCategory::from_u8(255).is_none());
    }

    #[test]
    fn subcategory_category_consistency() {
        assert_eq!(
            EventSubcategory::Grant.category(),
            EventCategory::Capability
        );
        assert_eq!(
            EventSubcategory::Revoke.category(),
            EventCategory::Capability
        );
        assert_eq!(EventSubcategory::Deny.category(), EventCategory::Capability);
        assert_eq!(
            EventSubcategory::TlsSuccess.category(),
            EventCategory::Authentication
        );
        assert_eq!(
            EventSubcategory::TlsFailure.category(),
            EventCategory::Authentication
        );
        assert_eq!(
            EventSubcategory::Allocation.category(),
            EventCategory::Resource
        );
        assert_eq!(
            EventSubcategory::Exhaustion.category(),
            EventCategory::Resource
        );
        assert_eq!(
            EventSubcategory::Request.category(),
            EventCategory::Inference
        );
        assert_eq!(
            EventSubcategory::Completion.category(),
            EventCategory::Inference
        );
        assert_eq!(
            EventSubcategory::Timeout.category(),
            EventCategory::Inference
        );
        assert_eq!(EventSubcategory::Boot.category(), EventCategory::System);
        assert_eq!(EventSubcategory::Shutdown.category(), EventCategory::System);
        assert_eq!(EventSubcategory::Watchdog.category(), EventCategory::System);
        assert_eq!(
            EventSubcategory::GateAllowed.category(),
            EventCategory::Security
        );
        assert_eq!(
            EventSubcategory::GateDenied.category(),
            EventCategory::Security
        );
        assert_eq!(
            EventSubcategory::GatePermissivePass.category(),
            EventCategory::Security
        );
        assert_eq!(
            EventSubcategory::PolicyUpdate.category(),
            EventCategory::Security
        );
    }

    #[test]
    fn event_type_new_validates_category() {
        // Valid: Grant belongs to Capability
        assert!(AuditEventType::new(EventCategory::Capability, EventSubcategory::Grant).is_some());
        // Invalid: Grant does not belong to System
        assert!(AuditEventType::new(EventCategory::System, EventSubcategory::Grant).is_none());
        // Invalid: Boot does not belong to Capability
        assert!(AuditEventType::new(EventCategory::Capability, EventSubcategory::Boot).is_none());
    }

    #[test]
    fn event_type_bytes_roundtrip() {
        let et = AuditEventType::capability_deny();
        let bytes = et.to_bytes();
        let decoded = AuditEventType::from_bytes(bytes).unwrap();
        assert_eq!(et, decoded);
    }

    #[test]
    fn event_type_all_constructors() {
        let events = [
            AuditEventType::capability_grant(),
            AuditEventType::capability_revoke(),
            AuditEventType::capability_deny(),
            AuditEventType::auth_success(),
            AuditEventType::auth_failure(),
            AuditEventType::resource_allocation(),
            AuditEventType::resource_exhaustion(),
            AuditEventType::inference_request(),
            AuditEventType::inference_completion(),
            AuditEventType::inference_timeout(),
            AuditEventType::system_boot(),
            AuditEventType::system_shutdown(),
            AuditEventType::system_watchdog(),
            AuditEventType::gate_allowed(),
            AuditEventType::gate_denied(),
            AuditEventType::gate_permissive_pass(),
            AuditEventType::policy_update(),
        ];
        for et in &events {
            assert_eq!(et.subcategory.category(), et.category);
            let bytes = et.to_bytes();
            let decoded = AuditEventType::from_bytes(bytes).unwrap();
            assert_eq!(*et, decoded);
        }
    }

    #[test]
    fn subcategory_from_u8_all_valid() {
        let valid = [0, 1, 2, 10, 11, 20, 21, 30, 31, 32, 40, 41, 42, 50, 51, 52, 53];
        for v in valid {
            assert!(
                EventSubcategory::from_u8(v).is_some(),
                "expected Some for {}",
                v
            );
        }
    }

    #[test]
    fn subcategory_from_u8_invalid() {
        let invalid = [3, 5, 12, 22, 33, 43, 54, 100, 255];
        for v in invalid {
            assert!(
                EventSubcategory::from_u8(v).is_none(),
                "expected None for {}",
                v
            );
        }
    }

    #[test]
    fn category_as_str() {
        assert_eq!(EventCategory::Capability.as_str(), "capability");
        assert_eq!(EventCategory::Authentication.as_str(), "auth");
        assert_eq!(EventCategory::Resource.as_str(), "resource");
        assert_eq!(EventCategory::Inference.as_str(), "inference");
        assert_eq!(EventCategory::System.as_str(), "system");
        assert_eq!(EventCategory::Security.as_str(), "security");
    }

    #[test]
    fn subcategory_as_str() {
        assert_eq!(EventSubcategory::Grant.as_str(), "grant");
        assert_eq!(EventSubcategory::TlsFailure.as_str(), "tls-failure");
        assert_eq!(EventSubcategory::Exhaustion.as_str(), "exhaustion");
        assert_eq!(EventSubcategory::Timeout.as_str(), "timeout");
        assert_eq!(EventSubcategory::Watchdog.as_str(), "watchdog");
        assert_eq!(EventSubcategory::GateAllowed.as_str(), "gate-allowed");
        assert_eq!(EventSubcategory::GateDenied.as_str(), "gate-denied");
        assert_eq!(EventSubcategory::GatePermissivePass.as_str(), "gate-permissive-pass");
        assert_eq!(EventSubcategory::PolicyUpdate.as_str(), "policy-update");
    }
}
