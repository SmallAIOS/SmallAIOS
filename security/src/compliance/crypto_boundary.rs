// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! FIPS 140-3 Level 1 crypto module boundary definition.
//!
//! Defines the approved algorithms, their FIPS references, and provides
//! boundary enforcement that ensures only approved algorithms are used.
//! Also implements key rotation scheduling (time-based and usage-based).

/// FIPS-approved cryptographic algorithms in the SmallAIOS crypto module.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ApprovedAlgorithm {
    /// SHA-3-256 hash (FIPS 202).
    Sha3_256 = 0,
    /// SHAKE256 extendable output (FIPS 202).
    Shake256 = 1,
    /// AES-256-GCM authenticated encryption (FIPS 197 + SP 800-38D).
    Aes256Gcm = 2,
    /// ML-KEM-768 key encapsulation (FIPS 203).
    MlKem768 = 3,
    /// ML-DSA-65 digital signature (FIPS 204).
    MlDsa65 = 4,
    /// HKDF-SHA3-256 key derivation (SP 800-56C).
    HkdfSha3 = 5,
}

impl ApprovedAlgorithm {
    /// FIPS document reference for this algorithm.
    pub fn fips_reference(&self) -> &'static str {
        match self {
            Self::Sha3_256 => "FIPS 202",
            Self::Shake256 => "FIPS 202",
            Self::Aes256Gcm => "FIPS 197 + SP 800-38D",
            Self::MlKem768 => "FIPS 203",
            Self::MlDsa65 => "FIPS 204",
            Self::HkdfSha3 => "SP 800-56C",
        }
    }

    /// Human-readable algorithm name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Sha3_256 => "SHA-3-256",
            Self::Shake256 => "SHAKE256",
            Self::Aes256Gcm => "AES-256-GCM",
            Self::MlKem768 => "ML-KEM-768",
            Self::MlDsa65 => "ML-DSA-65",
            Self::HkdfSha3 => "HKDF-SHA3-256",
        }
    }

    /// Security strength in bits (NIST SP 800-57).
    pub fn security_strength_bits(&self) -> u32 {
        match self {
            Self::Sha3_256 => 128,  // Collision resistance
            Self::Shake256 => 128,  // Security level
            Self::Aes256Gcm => 256, // Key size
            Self::MlKem768 => 192,  // NIST security level 3
            Self::MlDsa65 => 192,   // NIST security level 3
            Self::HkdfSha3 => 128,  // Derived from SHA-3-256
        }
    }

    /// Whether this algorithm is fully implemented (vs stub).
    pub fn is_implemented(&self) -> bool {
        match self {
            Self::Sha3_256 => true,   // Production implementation
            Self::Shake256 => true,   // Production implementation
            Self::Aes256Gcm => false, // Stub
            Self::MlKem768 => false,  // Stub
            Self::MlDsa65 => false,   // Stub
            Self::HkdfSha3 => false,  // Stub
        }
    }

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::Sha3_256),
            1 => Some(Self::Shake256),
            2 => Some(Self::Aes256Gcm),
            3 => Some(Self::MlKem768),
            4 => Some(Self::MlDsa65),
            5 => Some(Self::HkdfSha3),
            _ => None,
        }
    }
}

/// All approved algorithms in the crypto module boundary.
pub const ALL_APPROVED: &[ApprovedAlgorithm] = &[
    ApprovedAlgorithm::Sha3_256,
    ApprovedAlgorithm::Shake256,
    ApprovedAlgorithm::Aes256Gcm,
    ApprovedAlgorithm::MlKem768,
    ApprovedAlgorithm::MlDsa65,
    ApprovedAlgorithm::HkdfSha3,
];

/// Check whether an algorithm is within the approved boundary.
pub fn is_approved(algorithm: ApprovedAlgorithm) -> bool {
    ALL_APPROVED.contains(&algorithm)
}

/// Count of implemented vs stub algorithms.
pub fn implementation_status() -> (usize, usize) {
    let implemented = ALL_APPROVED.iter().filter(|a| a.is_implemented()).count();
    let stub = ALL_APPROVED.len() - implemented;
    (implemented, stub)
}

// ─── Key Rotation Scheduling ────────────────────────────────────────────────

/// Nanoseconds per hour (for rotation scheduling).
const NS_PER_HOUR: u64 = 3_600_000_000_000;

/// Key rotation trigger criteria.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RotationTrigger {
    /// Time-based rotation (maximum key lifetime exceeded).
    TimeBased { age_ns: u64, max_age_ns: u64 },
    /// Usage-based rotation (operation count exceeded threshold).
    UsageBased { usage: u64, threshold: u64 },
    /// Manual rotation requested by operator.
    Manual,
}

/// Key rotation policy per key type.
#[derive(Debug, Clone, Copy)]
pub struct RotationPolicy {
    /// Maximum key lifetime in nanoseconds (0 = no time limit).
    pub max_age_ns: u64,
    /// Maximum operations before rotation (0 = no usage limit).
    pub max_usage: u64,
}

impl RotationPolicy {
    /// Default rotation policies per NIST SP 800-57.
    pub const fn for_session_key() -> Self {
        Self {
            max_age_ns: 24 * NS_PER_HOUR, // 24 hours
            max_usage: 1u64 << 32,        // 2^32 operations
        }
    }

    pub const fn for_signing_key() -> Self {
        Self {
            max_age_ns: 0, // Rotated on reboot only
            max_usage: 1u64 << 32,
        }
    }

    pub const fn for_kem_key() -> Self {
        Self {
            max_age_ns: 0, // Rotated on reboot only
            max_usage: 1u64 << 32,
        }
    }

    /// Check whether rotation is needed based on this policy.
    pub fn check(&self, created_ns: u64, now_ns: u64, usage_count: u64) -> Option<RotationTrigger> {
        // Check time-based rotation
        if self.max_age_ns > 0 {
            let age = now_ns.saturating_sub(created_ns);
            if age >= self.max_age_ns {
                return Some(RotationTrigger::TimeBased {
                    age_ns: age,
                    max_age_ns: self.max_age_ns,
                });
            }
        }
        // Check usage-based rotation
        if self.max_usage > 0 && usage_count >= self.max_usage {
            return Some(RotationTrigger::UsageBased {
                usage: usage_count,
                threshold: self.max_usage,
            });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithm_fips_references() {
        assert_eq!(ApprovedAlgorithm::Sha3_256.fips_reference(), "FIPS 202");
        assert_eq!(
            ApprovedAlgorithm::Aes256Gcm.fips_reference(),
            "FIPS 197 + SP 800-38D"
        );
        assert_eq!(ApprovedAlgorithm::MlKem768.fips_reference(), "FIPS 203");
        assert_eq!(ApprovedAlgorithm::MlDsa65.fips_reference(), "FIPS 204");
    }

    #[test]
    fn algorithm_names() {
        assert_eq!(ApprovedAlgorithm::Sha3_256.name(), "SHA-3-256");
        assert_eq!(ApprovedAlgorithm::Aes256Gcm.name(), "AES-256-GCM");
        assert_eq!(ApprovedAlgorithm::MlKem768.name(), "ML-KEM-768");
    }

    #[test]
    fn security_strengths() {
        assert_eq!(ApprovedAlgorithm::Sha3_256.security_strength_bits(), 128);
        assert_eq!(ApprovedAlgorithm::Aes256Gcm.security_strength_bits(), 256);
        assert_eq!(ApprovedAlgorithm::MlKem768.security_strength_bits(), 192);
    }

    #[test]
    fn implementation_status_count() {
        let (implemented, stub) = implementation_status();
        assert_eq!(implemented, 2); // SHA-3 and SHAKE256
        assert_eq!(stub, 4); // AES-GCM, ML-KEM, ML-DSA, HKDF
        assert_eq!(implemented + stub, ALL_APPROVED.len());
    }

    #[test]
    fn all_algorithms_approved() {
        for alg in ALL_APPROVED {
            assert!(is_approved(*alg));
        }
    }

    #[test]
    fn algorithm_roundtrip() {
        for i in 0..=5u8 {
            let alg = ApprovedAlgorithm::from_u8(i).unwrap();
            assert_eq!(alg as u8, i);
        }
        assert!(ApprovedAlgorithm::from_u8(6).is_none());
    }

    #[test]
    fn rotation_policy_time_based() {
        let policy = RotationPolicy::for_session_key();
        // Key created at time 0, checked after 25 hours
        let trigger = policy.check(0, 25 * NS_PER_HOUR, 0);
        assert!(matches!(trigger, Some(RotationTrigger::TimeBased { .. })));
    }

    #[test]
    fn rotation_policy_no_trigger_before_deadline() {
        let policy = RotationPolicy::for_session_key();
        let trigger = policy.check(0, 12 * NS_PER_HOUR, 100);
        assert!(trigger.is_none());
    }

    #[test]
    fn rotation_policy_usage_based() {
        let policy = RotationPolicy::for_signing_key();
        let trigger = policy.check(0, 0, 1u64 << 32);
        assert!(matches!(trigger, Some(RotationTrigger::UsageBased { .. })));
    }

    #[test]
    fn rotation_policy_no_time_limit() {
        let policy = RotationPolicy::for_signing_key();
        // max_age_ns is 0, so no time-based rotation
        let trigger = policy.check(0, 999 * NS_PER_HOUR, 0);
        assert!(trigger.is_none());
    }

    #[test]
    fn rotation_policy_time_priority() {
        // Time-based triggers before usage-based
        let policy = RotationPolicy {
            max_age_ns: 10 * NS_PER_HOUR,
            max_usage: 100,
        };
        let trigger = policy.check(0, 11 * NS_PER_HOUR, 200);
        assert!(matches!(trigger, Some(RotationTrigger::TimeBased { .. })));
    }
}
