// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! DDS-Security authentication and access control plugin interfaces.
//!
//! Defines the plugin traits and types per OMG DDS-Security 1.1.
//! SmallAIOS uses ML-DSA-65 identity certificates instead of
//! classical RSA/ECDSA.

use crate::dds::types::{DomainId, GuidPrefix};
use crate::BusError;
use alloc::string::String;
use alloc::vec::Vec;

/// Identity handle — opaque reference to an authenticated identity.
pub type IdentityHandle = u64;

/// Permissions handle — opaque reference to access control permissions.
pub type PermissionsHandle = u64;

/// Authentication status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthStatus {
    /// Authentication is pending (handshake in progress).
    Pending,
    /// Authentication succeeded — identities mutually validated.
    Authenticated,
    /// Authentication failed — certificate invalid, expired, or untrusted.
    Failed,
}

/// Authentication handshake token (simplified).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HandshakeToken {
    /// Token class ID (identifies the authentication method).
    pub class_id: String,
    /// Binary token data.
    pub data: Vec<u8>,
}

/// Identity certificate (ML-DSA-65 based).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdentityCertificate {
    /// Subject name.
    pub subject: String,
    /// Issuer name (CA).
    pub issuer: String,
    /// Serial number.
    pub serial: u64,
    /// Public key bytes (ML-DSA-65 public key).
    pub public_key: Vec<u8>,
    /// Signature bytes (ML-DSA-65 signature over the certificate fields).
    pub signature: Vec<u8>,
    /// Validity: not-before timestamp (Unix seconds).
    pub not_before: u64,
    /// Validity: not-after timestamp (Unix seconds).
    pub not_after: u64,
}

impl IdentityCertificate {
    /// Check if the certificate is expired.
    pub fn is_expired(&self, now_unix_sec: u64) -> bool {
        now_unix_sec > self.not_after || now_unix_sec < self.not_before
    }

    /// Check if the certificate has a valid time window.
    pub fn is_time_valid(&self, now_unix_sec: u64) -> bool {
        now_unix_sec >= self.not_before && now_unix_sec <= self.not_after
    }
}

/// Authentication plugin trait.
///
/// Per DDS-Security, the authentication plugin handles mutual
/// authentication between DDS participants.
pub trait AuthenticationPlugin {
    /// Validate a local identity (check own certificate and private key).
    fn validate_local_identity(
        &self,
        cert: &IdentityCertificate,
        now_unix_sec: u64,
    ) -> Result<IdentityHandle, BusError>;

    /// Begin handshake with a remote participant.
    fn begin_handshake(
        &mut self,
        local_handle: IdentityHandle,
        remote_cert: &IdentityCertificate,
    ) -> Result<HandshakeToken, BusError>;

    /// Process a received handshake token.
    fn process_handshake(
        &mut self,
        local_handle: IdentityHandle,
        token: &HandshakeToken,
    ) -> Result<AuthStatus, BusError>;

    /// Get the authentication status for a remote participant.
    fn get_status(&self, remote_prefix: &GuidPrefix) -> AuthStatus;
}

/// Access control permission rule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionRule {
    /// Topic name pattern (supports wildcards with '*').
    pub topic_pattern: String,
    /// Allowed operations.
    pub allow_publish: bool,
    /// Allow subscribe.
    pub allow_subscribe: bool,
}

/// Access control governance for a domain.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DomainGovernance {
    /// Domain ID this governance applies to.
    pub domain_id: DomainId,
    /// Whether authentication is required.
    pub require_authentication: bool,
    /// Whether access control is enforced.
    pub enable_access_control: bool,
    /// Whether data protection (encryption) is required.
    pub enable_data_protection: bool,
}

/// Access control permissions document.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PermissionsDocument {
    /// Subject name this document applies to.
    pub subject: String,
    /// Authorized domains.
    pub allowed_domains: Vec<DomainId>,
    /// Per-topic permission rules.
    pub rules: Vec<PermissionRule>,
}

/// Access control plugin trait.
///
/// Per DDS-Security, the access control plugin validates that
/// participants and endpoints have the required permissions.
pub trait AccessControlPlugin {
    /// Check if a participant is authorized to join a domain.
    fn check_domain_access(
        &self,
        permissions: &PermissionsDocument,
        domain_id: DomainId,
    ) -> Result<PermissionsHandle, BusError>;

    /// Check if a DataWriter is authorized to publish on a topic.
    fn check_publish(
        &self,
        permissions_handle: PermissionsHandle,
        topic_name: &str,
    ) -> Result<bool, BusError>;

    /// Check if a DataReader is authorized to subscribe to a topic.
    fn check_subscribe(
        &self,
        permissions_handle: PermissionsHandle,
        topic_name: &str,
    ) -> Result<bool, BusError>;
}

/// Built-in authentication plugin (stub implementation).
///
/// Validates certificates by time window only. Full ML-DSA-65
/// signature verification requires the security crate's crypto
/// primitives to be wired in.
#[derive(Debug, Default)]
pub struct BuiltinAuthPlugin {
    next_handle: u64,
    authenticated: Vec<(GuidPrefix, AuthStatus)>,
}

impl BuiltinAuthPlugin {
    /// Create a new built-in authentication plugin.
    pub fn new() -> Self {
        Self {
            next_handle: 1,
            authenticated: Vec::new(),
        }
    }
}

impl AuthenticationPlugin for BuiltinAuthPlugin {
    fn validate_local_identity(
        &self,
        cert: &IdentityCertificate,
        now_unix_sec: u64,
    ) -> Result<IdentityHandle, BusError> {
        if cert.is_expired(now_unix_sec) {
            return Err(BusError::SecurityError);
        }
        Ok(self.next_handle)
    }

    fn begin_handshake(
        &mut self,
        _local_handle: IdentityHandle,
        remote_cert: &IdentityCertificate,
    ) -> Result<HandshakeToken, BusError> {
        // Stub: generate a minimal handshake token
        Ok(HandshakeToken {
            class_id: String::from("dds.sec.auth"),
            data: remote_cert.public_key.clone(),
        })
    }

    fn process_handshake(
        &mut self,
        _local_handle: IdentityHandle,
        token: &HandshakeToken,
    ) -> Result<AuthStatus, BusError> {
        if token.data.is_empty() {
            return Ok(AuthStatus::Failed);
        }
        // Stub: always succeeds if token has data
        // Real implementation would verify ML-DSA-65 signature
        Ok(AuthStatus::Authenticated)
    }

    fn get_status(&self, remote_prefix: &GuidPrefix) -> AuthStatus {
        self.authenticated
            .iter()
            .find(|(p, _)| p == remote_prefix)
            .map(|(_, s)| *s)
            .unwrap_or(AuthStatus::Pending)
    }
}

/// Built-in access control plugin.
#[derive(Debug, Default)]
pub struct BuiltinAccessPlugin {
    next_handle: u64,
}

impl BuiltinAccessPlugin {
    /// Create a new built-in access control plugin.
    pub fn new() -> Self {
        Self { next_handle: 1 }
    }
}

impl AccessControlPlugin for BuiltinAccessPlugin {
    fn check_domain_access(
        &self,
        permissions: &PermissionsDocument,
        domain_id: DomainId,
    ) -> Result<PermissionsHandle, BusError> {
        if permissions.allowed_domains.contains(&domain_id) {
            Ok(self.next_handle)
        } else {
            Err(BusError::SecurityError)
        }
    }

    fn check_publish(
        &self,
        _permissions_handle: PermissionsHandle,
        _topic_name: &str,
    ) -> Result<bool, BusError> {
        // In a real implementation, we'd look up the permissions handle
        // and match topic_name against the rules. Stub: allow all.
        Ok(true)
    }

    fn check_subscribe(
        &self,
        _permissions_handle: PermissionsHandle,
        _topic_name: &str,
    ) -> Result<bool, BusError> {
        Ok(true)
    }
}

/// Check a permissions document for topic-level authorization.
pub fn check_topic_permission(
    doc: &PermissionsDocument,
    topic_name: &str,
    is_publish: bool,
) -> bool {
    for rule in &doc.rules {
        if topic_matches(&rule.topic_pattern, topic_name) {
            if is_publish {
                return rule.allow_publish;
            } else {
                return rule.allow_subscribe;
            }
        }
    }
    false // No matching rule = deny
}

/// Simple topic pattern matching (supports trailing '*' wildcard).
fn topic_matches(pattern: &str, topic: &str) -> bool {
    if pattern == "*" {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        topic.starts_with(prefix)
    } else {
        pattern == topic
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    fn make_cert(subject: &str, not_before: u64, not_after: u64) -> IdentityCertificate {
        IdentityCertificate {
            subject: String::from(subject),
            issuer: String::from("TestCA"),
            serial: 1,
            public_key: vec![0xAA; 32],
            signature: vec![0xBB; 64],
            not_before,
            not_after,
        }
    }

    #[test]
    fn test_certificate_time_valid() {
        let cert = make_cert("test", 1000, 2000);
        assert!(cert.is_time_valid(1500));
        assert!(!cert.is_time_valid(500));
        assert!(!cert.is_time_valid(2500));
    }

    #[test]
    fn test_certificate_expired() {
        let cert = make_cert("test", 1000, 2000);
        assert!(!cert.is_expired(1500));
        assert!(cert.is_expired(2500));
        assert!(cert.is_expired(500));
    }

    #[test]
    fn test_builtin_auth_validate_local() {
        let plugin = BuiltinAuthPlugin::new();
        let cert = make_cert("local", 1000, 2000);
        let handle = plugin.validate_local_identity(&cert, 1500).unwrap();
        assert!(handle > 0);
    }

    #[test]
    fn test_builtin_auth_validate_expired() {
        let plugin = BuiltinAuthPlugin::new();
        let cert = make_cert("local", 1000, 2000);
        assert_eq!(
            plugin.validate_local_identity(&cert, 3000).err(),
            Some(BusError::SecurityError)
        );
    }

    #[test]
    fn test_builtin_auth_handshake() {
        let mut plugin = BuiltinAuthPlugin::new();
        let cert = make_cert("remote", 1000, 2000);
        let token = plugin.begin_handshake(1, &cert).unwrap();
        assert_eq!(token.class_id, "dds.sec.auth");
        assert!(!token.data.is_empty());
    }

    #[test]
    fn test_builtin_auth_process_handshake() {
        let mut plugin = BuiltinAuthPlugin::new();
        let token = HandshakeToken {
            class_id: String::from("dds.sec.auth"),
            data: vec![0xAA; 32],
        };
        let status = plugin.process_handshake(1, &token).unwrap();
        assert_eq!(status, AuthStatus::Authenticated);
    }

    #[test]
    fn test_builtin_auth_process_empty_token() {
        let mut plugin = BuiltinAuthPlugin::new();
        let token = HandshakeToken {
            class_id: String::from("dds.sec.auth"),
            data: vec![],
        };
        let status = plugin.process_handshake(1, &token).unwrap();
        assert_eq!(status, AuthStatus::Failed);
    }

    #[test]
    fn test_builtin_auth_get_status_unknown() {
        let plugin = BuiltinAuthPlugin::new();
        assert_eq!(
            plugin.get_status(&GuidPrefix::new([0x01; 12])),
            AuthStatus::Pending
        );
    }

    #[test]
    fn test_builtin_access_domain_allowed() {
        let plugin = BuiltinAccessPlugin::new();
        let doc = PermissionsDocument {
            subject: String::from("test"),
            allowed_domains: vec![0, 1, 2],
            rules: vec![],
        };
        let handle = plugin.check_domain_access(&doc, 0).unwrap();
        assert!(handle > 0);
    }

    #[test]
    fn test_builtin_access_domain_denied() {
        let plugin = BuiltinAccessPlugin::new();
        let doc = PermissionsDocument {
            subject: String::from("test"),
            allowed_domains: vec![0, 1],
            rules: vec![],
        };
        assert_eq!(
            plugin.check_domain_access(&doc, 42).err(),
            Some(BusError::SecurityError)
        );
    }

    #[test]
    fn test_topic_matches_exact() {
        assert!(topic_matches("SensorData", "SensorData"));
        assert!(!topic_matches("SensorData", "OtherData"));
    }

    #[test]
    fn test_topic_matches_wildcard_all() {
        assert!(topic_matches("*", "SensorData"));
        assert!(topic_matches("*", "anything"));
    }

    #[test]
    fn test_topic_matches_wildcard_prefix() {
        assert!(topic_matches("Sensor*", "SensorData"));
        assert!(topic_matches("Sensor*", "SensorCommand"));
        assert!(!topic_matches("Sensor*", "OtherData"));
    }

    #[test]
    fn test_check_topic_permission_publish() {
        let doc = PermissionsDocument {
            subject: String::from("test"),
            allowed_domains: vec![0],
            rules: vec![PermissionRule {
                topic_pattern: String::from("Sensor*"),
                allow_publish: true,
                allow_subscribe: false,
            }],
        };
        assert!(check_topic_permission(&doc, "SensorData", true));
        assert!(!check_topic_permission(&doc, "SensorData", false));
    }

    #[test]
    fn test_check_topic_permission_no_rule() {
        let doc = PermissionsDocument {
            subject: String::from("test"),
            allowed_domains: vec![0],
            rules: vec![],
        };
        // No matching rule = deny
        assert!(!check_topic_permission(&doc, "SensorData", true));
    }

    #[test]
    fn test_domain_governance() {
        let gov = DomainGovernance {
            domain_id: 0,
            require_authentication: true,
            enable_access_control: true,
            enable_data_protection: false,
        };
        assert!(gov.require_authentication);
        assert_eq!(gov.domain_id, 0);
    }

    #[test]
    fn test_permission_rule() {
        let rule = PermissionRule {
            topic_pattern: String::from("Classified*"),
            allow_publish: false,
            allow_subscribe: true,
        };
        assert!(!rule.allow_publish);
        assert!(rule.allow_subscribe);
    }

    #[test]
    fn test_auth_status_variants() {
        assert_ne!(AuthStatus::Pending, AuthStatus::Authenticated);
        assert_ne!(AuthStatus::Authenticated, AuthStatus::Failed);
    }
}
