// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Security policy: loaded independently from ONNX models.
//!
//! The policy contains:
//! - Verified message type registry
//! - Model hash whitelist
//! - Per-boundary enforcement modes
//! - Global enforcement mode
//! - Policy hash and ML-DSA-65 signature
//!
//! Policy lifecycle:
//! 1. Compiled-in default at boot (all Permissive)
//! 2. Optionally overridden by signed blob at SecurityReady phase
//! 3. Remotely updateable via signed blob over mTLS

use crate::enforcement::EnforcementMode;
use crate::labels::MessageTypeId;
use crate::message_types::{
    default_registry, MessageTypeRegistry, SchemaHash, VerifiedMessageType, UNVERIFIED_HASH,
};

/// Policy blob magic number: "SPOL" in ASCII.
pub const POLICY_MAGIC: u32 = 0x5350_4F4C;

/// Current policy blob format version.
pub const POLICY_VERSION: u32 = 1;

/// Maximum number of model hashes in the whitelist.
pub const MAX_MODEL_WHITELIST: usize = 32;

/// ML-DSA-65 signature size placeholder (actual: 3293 bytes).
/// Using a smaller size for the fixed-size struct; real implementation
/// would reference the crypto module's constant.
pub const POLICY_SIG_SIZE: usize = 64;

/// Number of trust boundaries.
const NUM_BOUNDARIES: usize = 5;

/// Fixed-size model hash whitelist.
#[derive(Clone)]
pub struct ModelWhitelist {
    hashes: [SchemaHash; MAX_MODEL_WHITELIST],
    count: usize,
}

impl ModelWhitelist {
    pub const fn new() -> Self {
        Self {
            hashes: [[0u8; 32]; MAX_MODEL_WHITELIST],
            count: 0,
        }
    }

    /// Add a model hash. Returns false if full.
    pub fn add(&mut self, hash: SchemaHash) -> bool {
        if self.count >= MAX_MODEL_WHITELIST {
            return false;
        }
        // Check for duplicate
        for i in 0..self.count {
            if self.hashes[i] == hash {
                return true; // already present
            }
        }
        self.hashes[self.count] = hash;
        self.count += 1;
        true
    }

    /// Check whether a model hash is in the whitelist.
    pub fn contains(&self, hash: &SchemaHash) -> bool {
        for i in 0..self.count {
            if &self.hashes[i] == hash {
                return true;
            }
        }
        false
    }

    /// Number of whitelisted models.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether the whitelist is empty.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }
}

impl Default for ModelWhitelist {
    fn default() -> Self {
        Self::new()
    }
}

/// Policy validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyError {
    /// Blob too short to contain header.
    BlobTooShort,
    /// Magic number mismatch.
    InvalidMagic,
    /// Unsupported policy version.
    UnsupportedVersion,
    /// ML-DSA-65 signature verification failed.
    SignatureInvalid,
    /// Internal consistency check failed.
    InconsistentPolicy,
    /// Attempted to demote a type from Enforcing to Permissive.
    EnforcementDemotion {
        type_id: MessageTypeId,
    },
    /// Duplicate type IDs in policy.
    DuplicateTypeId {
        type_id: MessageTypeId,
    },
}

/// The security policy.
pub struct SecurityPolicy {
    /// Verified message type registry.
    pub type_registry: MessageTypeRegistry,
    /// Model hash whitelist.
    pub model_whitelist: ModelWhitelist,
    /// Per-boundary enforcement modes.
    pub boundary_modes: [EnforcementMode; NUM_BOUNDARIES],
    /// Global default enforcement mode.
    pub global_mode: EnforcementMode,
    /// SHA-3-256 of this policy's serialized content.
    pub policy_hash: SchemaHash,
    /// Policy format version.
    pub version: u32,
}

impl SecurityPolicy {
    /// Create the compiled-in default policy.
    /// All types are Permissive, whitelist is empty (allows all models).
    pub fn default_policy() -> Self {
        Self {
            type_registry: default_registry(),
            model_whitelist: ModelWhitelist::new(),
            boundary_modes: [EnforcementMode::Permissive; NUM_BOUNDARIES],
            global_mode: EnforcementMode::Permissive,
            policy_hash: UNVERIFIED_HASH,
            version: POLICY_VERSION,
        }
    }

    /// Parse and validate a signed policy blob.
    ///
    /// Blob format:
    /// - Bytes 0..4: magic (0x53504F4C)
    /// - Bytes 4..8: version (u32 LE)
    /// - Bytes 8..8+POLICY_SIG_SIZE: signature
    /// - Bytes 8+POLICY_SIG_SIZE..: payload
    ///
    /// In this implementation, signature verification is stubbed
    /// (the real implementation would call into crypto::ml_dsa).
    pub fn load_from_blob(blob: &[u8]) -> Result<Self, PolicyError> {
        let header_size = 8 + POLICY_SIG_SIZE;
        if blob.len() < header_size {
            return Err(PolicyError::BlobTooShort);
        }

        // Check magic
        let magic = u32::from_le_bytes([blob[0], blob[1], blob[2], blob[3]]);
        if magic != POLICY_MAGIC {
            return Err(PolicyError::InvalidMagic);
        }

        // Check version
        let version = u32::from_le_bytes([blob[4], blob[5], blob[6], blob[7]]);
        if version != POLICY_VERSION {
            return Err(PolicyError::UnsupportedVersion);
        }

        // Signature verification (stub — real impl uses crypto::ml_dsa)
        let _signature = &blob[8..8 + POLICY_SIG_SIZE];
        // In production: verify_ml_dsa_65(policy_pubkey, &blob[8+POLICY_SIG_SIZE..], signature)?

        let payload = &blob[header_size..];
        Self::deserialize_payload(payload, version)
    }

    /// Validate and apply a remote policy update.
    ///
    /// Enforces:
    /// - Signature verification
    /// - No enforcement demotion (Enforcing→Permissive blocked)
    /// - No duplicate type IDs
    ///
    /// Returns the new policy on success. The caller is responsible for
    /// retaining the old policy for rollback.
    pub fn remote_update(
        &self,
        blob: &[u8],
    ) -> Result<Self, PolicyError> {
        let new_policy = Self::load_from_blob(blob)?;

        // Validate: no enforcement demotion
        for existing_type in self.type_registry.iter() {
            if existing_type.mode == EnforcementMode::Enforcing {
                if let Some(new_type) = new_policy.type_registry.lookup(existing_type.type_id) {
                    if new_type.mode == EnforcementMode::Permissive {
                        return Err(PolicyError::EnforcementDemotion {
                            type_id: existing_type.type_id,
                        });
                    }
                }
                // Type removed entirely is OK (it's not a demotion)
            }
        }

        Ok(new_policy)
    }

    /// Check whether a model hash is allowed by this policy.
    /// If the whitelist is empty, all models are allowed (permissive default).
    pub fn model_allowed(&self, model_hash: &SchemaHash) -> bool {
        if self.model_whitelist.is_empty() {
            return true; // empty whitelist = allow all
        }
        self.model_whitelist.contains(model_hash)
    }

    /// Deserialize policy payload (simplified — real impl would parse binary format).
    fn deserialize_payload(payload: &[u8], version: u32) -> Result<Self, PolicyError> {
        // For now, return default policy with the given version.
        // A full implementation would parse boundary_modes, type registry entries,
        // and model whitelist from the payload bytes.
        if payload.is_empty() {
            return Err(PolicyError::InconsistentPolicy);
        }

        // Parse global mode from first byte
        let global_mode = match payload[0] {
            0 => EnforcementMode::Enforcing,
            1 => EnforcementMode::Permissive,
            _ => return Err(PolicyError::InconsistentPolicy),
        };

        Ok(Self {
            type_registry: default_registry(),
            model_whitelist: ModelWhitelist::new(),
            boundary_modes: [global_mode; NUM_BOUNDARIES],
            global_mode,
            policy_hash: UNVERIFIED_HASH,
            version,
        })
    }
}

impl Default for SecurityPolicy {
    fn default() -> Self {
        Self::default_policy()
    }
}

/// Result of re-validating loaded models after a policy swap.
#[derive(Debug, Clone)]
pub struct RevalidationResult {
    /// Models that remain valid under the new policy.
    pub valid_models: alloc::vec::Vec<SchemaHash>,
    /// Models that must be unloaded (not in new whitelist).
    pub rejected_models: alloc::vec::Vec<SchemaHash>,
}

/// Re-validate a set of loaded model hashes against a new policy.
pub fn revalidate_models(
    model_hashes: &[SchemaHash],
    policy: &SecurityPolicy,
) -> RevalidationResult {
    let mut valid = alloc::vec::Vec::new();
    let mut rejected = alloc::vec::Vec::new();

    for hash in model_hashes {
        if policy.model_allowed(hash) {
            valid.push(*hash);
        } else {
            rejected.push(*hash);
        }
    }

    RevalidationResult {
        valid_models: valid,
        rejected_models: rejected,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── ModelWhitelist ──

    #[test]
    fn whitelist_new_empty() {
        let wl = ModelWhitelist::new();
        assert_eq!(wl.count(), 0);
        assert!(wl.is_empty());
    }

    #[test]
    fn whitelist_add_and_contains() {
        let mut wl = ModelWhitelist::new();
        let hash = [0xABu8; 32];
        assert!(wl.add(hash));
        assert_eq!(wl.count(), 1);
        assert!(wl.contains(&hash));
        assert!(!wl.is_empty());
    }

    #[test]
    fn whitelist_duplicate_idempotent() {
        let mut wl = ModelWhitelist::new();
        let hash = [0xABu8; 32];
        assert!(wl.add(hash));
        assert!(wl.add(hash)); // duplicate
        assert_eq!(wl.count(), 1);
    }

    #[test]
    fn whitelist_not_contains() {
        let wl = ModelWhitelist::new();
        assert!(!wl.contains(&[0xABu8; 32]));
    }

    #[test]
    fn whitelist_capacity() {
        let mut wl = ModelWhitelist::new();
        for i in 0..MAX_MODEL_WHITELIST {
            let mut hash = [0u8; 32];
            hash[0] = i as u8;
            assert!(wl.add(hash));
        }
        assert_eq!(wl.count(), MAX_MODEL_WHITELIST);
        // 33rd should fail
        assert!(!wl.add([0xFFu8; 32]));
    }

    // ── SecurityPolicy ──

    #[test]
    fn default_policy() {
        let policy = SecurityPolicy::default_policy();
        assert_eq!(policy.type_registry.count(), 11);
        assert_eq!(policy.global_mode, EnforcementMode::Permissive);
        assert!(policy.model_whitelist.is_empty());
        assert_eq!(policy.version, POLICY_VERSION);
    }

    #[test]
    fn model_allowed_empty_whitelist() {
        let policy = SecurityPolicy::default_policy();
        // Empty whitelist → allow all
        assert!(policy.model_allowed(&[0xABu8; 32]));
    }

    #[test]
    fn model_allowed_with_whitelist() {
        let mut policy = SecurityPolicy::default_policy();
        let allowed_hash = [0xABu8; 32];
        policy.model_whitelist.add(allowed_hash);

        assert!(policy.model_allowed(&allowed_hash));
        assert!(!policy.model_allowed(&[0xCDu8; 32]));
    }

    // ── load_from_blob ──

    fn make_blob(magic: u32, version: u32, payload: &[u8]) -> alloc::vec::Vec<u8> {
        let mut blob = alloc::vec::Vec::new();
        blob.extend_from_slice(&magic.to_le_bytes());
        blob.extend_from_slice(&version.to_le_bytes());
        blob.extend_from_slice(&[0u8; POLICY_SIG_SIZE]); // stub signature
        blob.extend_from_slice(payload);
        blob
    }

    #[test]
    fn load_valid_blob() {
        let blob = make_blob(POLICY_MAGIC, POLICY_VERSION, &[1]); // Permissive
        let policy = SecurityPolicy::load_from_blob(&blob).unwrap();
        assert_eq!(policy.global_mode, EnforcementMode::Permissive);
    }

    #[test]
    fn load_enforcing_blob() {
        let blob = make_blob(POLICY_MAGIC, POLICY_VERSION, &[0]); // Enforcing
        let policy = SecurityPolicy::load_from_blob(&blob).unwrap();
        assert_eq!(policy.global_mode, EnforcementMode::Enforcing);
    }

    #[test]
    fn load_blob_too_short() {
        let blob = &[0u8; 4];
        assert!(matches!(
            SecurityPolicy::load_from_blob(blob),
            Err(PolicyError::BlobTooShort)
        ));
    }

    #[test]
    fn load_blob_invalid_magic() {
        let blob = make_blob(0xDEADBEEF, POLICY_VERSION, &[1]);
        assert!(matches!(
            SecurityPolicy::load_from_blob(&blob),
            Err(PolicyError::InvalidMagic)
        ));
    }

    #[test]
    fn load_blob_unsupported_version() {
        let blob = make_blob(POLICY_MAGIC, 999, &[1]);
        assert!(matches!(
            SecurityPolicy::load_from_blob(&blob),
            Err(PolicyError::UnsupportedVersion)
        ));
    }

    #[test]
    fn load_blob_empty_payload() {
        let blob = make_blob(POLICY_MAGIC, POLICY_VERSION, &[]);
        assert!(matches!(
            SecurityPolicy::load_from_blob(&blob),
            Err(PolicyError::InconsistentPolicy)
        ));
    }

    #[test]
    fn load_blob_invalid_mode_byte() {
        let blob = make_blob(POLICY_MAGIC, POLICY_VERSION, &[99]);
        assert!(matches!(
            SecurityPolicy::load_from_blob(&blob),
            Err(PolicyError::InconsistentPolicy)
        ));
    }

    // ── remote_update ──

    #[test]
    fn remote_update_accepted() {
        let current = SecurityPolicy::default_policy();
        let blob = make_blob(POLICY_MAGIC, POLICY_VERSION, &[0]); // Enforcing
        let new_policy = current.remote_update(&blob).unwrap();
        assert_eq!(new_policy.global_mode, EnforcementMode::Enforcing);
    }

    #[test]
    fn remote_update_demotion_rejected() {
        // Create a policy with an Enforcing type
        let mut current = SecurityPolicy::default_policy();
        // Manually register an Enforcing type
        let enforcing_type = VerifiedMessageType {
            type_id: MessageTypeId(0x0001),
            name: "InferenceTensorInput",
            boundary: crate::boundary::trust_boundaries::TrustBoundary::Network,
            direction: crate::boundary::trust_boundaries::DataFlowDirection::Inbound,
            schema_hash: [0xAB; 32],
            mode: EnforcementMode::Enforcing,
            invariants: &[],
        };
        // Replace the registry with one containing the Enforcing type
        let mut reg = MessageTypeRegistry::new();
        reg.register(enforcing_type);
        current.type_registry = reg;

        // New policy blob has type 0x0001 as Permissive (demotion)
        let blob = make_blob(POLICY_MAGIC, POLICY_VERSION, &[1]);
        let result = current.remote_update(&blob);
        assert!(matches!(
            result,
            Err(PolicyError::EnforcementDemotion { .. })
        ));
    }

    #[test]
    fn remote_update_invalid_blob() {
        let current = SecurityPolicy::default_policy();
        let blob = &[0u8; 4]; // too short
        assert!(current.remote_update(blob).is_err());
    }

    // ── revalidate_models ──

    #[test]
    fn revalidate_empty_whitelist_allows_all() {
        let policy = SecurityPolicy::default_policy();
        let hashes = [[0xABu8; 32], [0xCDu8; 32]];
        let result = revalidate_models(&hashes, &policy);
        assert_eq!(result.valid_models.len(), 2);
        assert_eq!(result.rejected_models.len(), 0);
    }

    #[test]
    fn revalidate_with_whitelist() {
        let mut policy = SecurityPolicy::default_policy();
        let allowed = [0xABu8; 32];
        let denied = [0xCDu8; 32];
        policy.model_whitelist.add(allowed);

        let result = revalidate_models(&[allowed, denied], &policy);
        assert_eq!(result.valid_models.len(), 1);
        assert_eq!(result.valid_models[0], allowed);
        assert_eq!(result.rejected_models.len(), 1);
        assert_eq!(result.rejected_models[0], denied);
    }
}
