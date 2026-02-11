// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Key lifecycle management per NIST SP 800-53 SC-12/SC-13.
//!
//! Provides:
//! - Key lifecycle management (generation, rotation, destruction)
//! - Key zeroization with `core::ptr::write_volatile` (not optimized away)
//! - Crypto module self-test (Known Answer Tests for SHA-3-256)
//! - Key usage tracking with rotation warning at 2^32 operations
//!
//! All key material is volatile-only (no persistent storage without TPM).

/// Maximum number of managed keys.
const MAX_KEYS: usize = 32;

/// Usage threshold for rotation warning (2^32 operations).
const ROTATION_WARNING_THRESHOLD: u64 = 1u64 << 32;

// ─── Key Types ──────────────────────────────────────────────────────────────

/// Type of cryptographic key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyType {
    /// ML-KEM-768 key pair (encapsulation/decapsulation).
    MlKem768 = 0,
    /// ML-DSA-65 signing key pair.
    MlDsa65 = 1,
    /// AES-256 symmetric key.
    Aes256 = 2,
    /// TLS 1.3 session key.
    Tls13Session = 3,
    /// HKDF-derived key.
    HkdfDerived = 4,
    /// CSPRNG internal state.
    CsprngState = 5,
    /// Audit log signing key.
    AuditSigning = 6,
}

impl KeyType {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            0 => Some(Self::MlKem768),
            1 => Some(Self::MlDsa65),
            2 => Some(Self::Aes256),
            3 => Some(Self::Tls13Session),
            4 => Some(Self::HkdfDerived),
            5 => Some(Self::CsprngState),
            6 => Some(Self::AuditSigning),
            _ => None,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::MlKem768 => "ml-kem-768",
            Self::MlDsa65 => "ml-dsa-65",
            Self::Aes256 => "aes-256",
            Self::Tls13Session => "tls13-session",
            Self::HkdfDerived => "hkdf-derived",
            Self::CsprngState => "csprng-state",
            Self::AuditSigning => "audit-signing",
        }
    }

    /// Maximum key material size for this key type.
    pub fn max_key_size(&self) -> usize {
        match self {
            Self::MlKem768 => 64,    // Seed/shared secret size
            Self::MlDsa65 => 64,     // Seed size
            Self::Aes256 => 32,      // 256-bit key
            Self::Tls13Session => 48, // Key + IV
            Self::HkdfDerived => 64,  // Up to 512 bits
            Self::CsprngState => 64,  // Internal state
            Self::AuditSigning => 64, // Signing seed
        }
    }
}

/// Key lifecycle state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum KeyState {
    /// Slot is empty.
    Empty = 0,
    /// Key is generated and active.
    Active = 1,
    /// Key is scheduled for rotation.
    PendingRotation = 2,
    /// Key has been zeroized (destroyed).
    Destroyed = 3,
}

// ─── Key Slot ───────────────────────────────────────────────────────────────

/// Maximum key material storage per slot.
const MAX_KEY_MATERIAL: usize = 64;

/// A managed key slot in the key registry.
#[derive(Clone)]
pub struct KeySlot {
    /// Unique key identifier.
    key_id: u64,
    /// Type of key.
    key_type: KeyType,
    /// Current lifecycle state.
    state: KeyState,
    /// Key material (zeroized on destruction).
    material: [u8; MAX_KEY_MATERIAL],
    /// Actual length of key material in use.
    material_len: usize,
    /// Timestamp of key creation (nanoseconds).
    created_ns: u64,
    /// Number of cryptographic operations performed with this key.
    usage_count: u64,
}

impl KeySlot {
    const fn empty() -> Self {
        Self {
            key_id: 0,
            key_type: KeyType::Aes256,
            state: KeyState::Empty,
            material: [0u8; MAX_KEY_MATERIAL],
            material_len: 0,
            created_ns: 0,
            usage_count: 0,
        }
    }

    /// Zeroize the key material using volatile writes.
    ///
    /// Uses `core::ptr::write_volatile` to ensure the compiler does not
    /// optimize away the zeroing operation.
    fn zeroize(&mut self) {
        for byte in self.material.iter_mut() {
            // SAFETY: byte is a valid, aligned, dereferenceable pointer to a u8
            // within our owned array. volatile write prevents optimization.
            unsafe {
                core::ptr::write_volatile(byte, 0u8);
            }
        }
        // Also zero the length via volatile write
        unsafe {
            core::ptr::write_volatile(&mut self.material_len, 0);
        }
        self.state = KeyState::Destroyed;
    }

    /// Verify that key material is actually zeroed after zeroization.
    fn verify_zeroized(&self) -> bool {
        // Read each byte via volatile read to prevent optimization
        let mut acc: u8 = 0;
        for byte in &self.material {
            let val = unsafe { core::ptr::read_volatile(byte) };
            acc |= val;
        }
        acc == 0
    }

    /// Check whether this key needs rotation based on usage count.
    pub fn needs_rotation(&self) -> bool {
        self.state == KeyState::Active && self.usage_count >= ROTATION_WARNING_THRESHOLD
    }
}

// ─── Key Manager ────────────────────────────────────────────────────────────

/// Errors from key management operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyManagerError {
    /// No free key slots available.
    RegistryFull,
    /// Key not found by ID.
    KeyNotFound,
    /// Key is not in the Active state.
    KeyNotActive,
    /// Key material too large for slot.
    MaterialTooLarge,
    /// Zeroization verification failed.
    ZeroizationFailed,
    /// Self-test failed.
    SelfTestFailed,
    /// Key type mismatch.
    TypeMismatch,
}

/// Key lifecycle manager per NIST SP 800-53 SC-12.
///
/// Manages key generation, usage tracking, rotation warnings,
/// and secure destruction (zeroization) for all cryptographic keys.
pub struct KeyManager {
    /// Key slots.
    slots: [KeySlot; MAX_KEYS],
    /// Next key ID to assign.
    next_id: u64,
    /// Total keys generated since boot.
    total_generated: u64,
    /// Total keys destroyed since boot.
    total_destroyed: u64,
    /// Whether the crypto self-test has passed.
    self_test_passed: bool,
}

impl KeyManager {
    /// Create a new key manager.
    pub fn new() -> Self {
        Self {
            slots: core::array::from_fn(|_| KeySlot::empty()),
            next_id: 1,
            total_generated: 0,
            total_destroyed: 0,
            self_test_passed: false,
        }
    }

    /// Register a new key with the given material.
    ///
    /// Returns the assigned key ID on success.
    pub fn register_key(
        &mut self,
        key_type: KeyType,
        material: &[u8],
        timestamp_ns: u64,
    ) -> Result<u64, KeyManagerError> {
        if material.len() > MAX_KEY_MATERIAL {
            return Err(KeyManagerError::MaterialTooLarge);
        }

        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.state == KeyState::Empty || s.state == KeyState::Destroyed)
            .ok_or(KeyManagerError::RegistryFull)?;

        let key_id = self.next_id;
        self.next_id += 1;

        slot.key_id = key_id;
        slot.key_type = key_type;
        slot.state = KeyState::Active;
        slot.material[..material.len()].copy_from_slice(material);
        slot.material_len = material.len();
        slot.created_ns = timestamp_ns;
        slot.usage_count = 0;

        self.total_generated += 1;

        Ok(key_id)
    }

    /// Record a usage of the key (increment counter).
    ///
    /// Returns `true` if the key now needs rotation (threshold reached).
    pub fn record_usage(&mut self, key_id: u64) -> Result<bool, KeyManagerError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.key_id == key_id && s.state == KeyState::Active)
            .ok_or(KeyManagerError::KeyNotActive)?;

        slot.usage_count = slot.usage_count.saturating_add(1);
        Ok(slot.needs_rotation())
    }

    /// Get the current usage count for a key.
    pub fn usage_count(&self, key_id: u64) -> Result<u64, KeyManagerError> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.key_id == key_id && s.state != KeyState::Empty)
            .ok_or(KeyManagerError::KeyNotFound)?;
        Ok(slot.usage_count)
    }

    /// Mark a key as pending rotation.
    pub fn mark_for_rotation(&mut self, key_id: u64) -> Result<(), KeyManagerError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.key_id == key_id && s.state == KeyState::Active)
            .ok_or(KeyManagerError::KeyNotActive)?;

        slot.state = KeyState::PendingRotation;
        Ok(())
    }

    /// Destroy a key: zeroize material and mark as destroyed.
    ///
    /// Performs a verification pass after zeroization.
    pub fn destroy_key(&mut self, key_id: u64) -> Result<(), KeyManagerError> {
        let slot = self
            .slots
            .iter_mut()
            .find(|s| s.key_id == key_id && s.state != KeyState::Empty && s.state != KeyState::Destroyed)
            .ok_or(KeyManagerError::KeyNotFound)?;

        slot.zeroize();

        if !slot.verify_zeroized() {
            return Err(KeyManagerError::ZeroizationFailed);
        }

        self.total_destroyed += 1;
        Ok(())
    }

    /// Destroy all active keys (shutdown zeroization).
    ///
    /// Returns the number of keys zeroized and whether all passed verification.
    pub fn destroy_all(&mut self) -> (u32, bool) {
        let mut count = 0u32;
        let mut all_verified = true;

        for slot in self.slots.iter_mut() {
            if slot.state == KeyState::Active || slot.state == KeyState::PendingRotation {
                slot.zeroize();
                if !slot.verify_zeroized() {
                    all_verified = false;
                }
                count += 1;
                self.total_destroyed += 1;
            }
        }

        (count, all_verified)
    }

    /// Get the key type for a given key ID.
    pub fn key_type(&self, key_id: u64) -> Result<KeyType, KeyManagerError> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.key_id == key_id && s.state != KeyState::Empty)
            .ok_or(KeyManagerError::KeyNotFound)?;
        Ok(slot.key_type)
    }

    /// Get the lifecycle state for a given key ID.
    pub fn key_state(&self, key_id: u64) -> Result<KeyState, KeyManagerError> {
        let slot = self
            .slots
            .iter()
            .find(|s| s.key_id == key_id)
            .ok_or(KeyManagerError::KeyNotFound)?;
        Ok(slot.state)
    }

    /// Number of currently active keys.
    pub fn active_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state == KeyState::Active)
            .count()
    }

    /// Number of keys pending rotation.
    pub fn pending_rotation_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.state == KeyState::PendingRotation)
            .count()
    }

    /// Total keys generated since boot.
    pub fn total_generated(&self) -> u64 {
        self.total_generated
    }

    /// Total keys destroyed since boot.
    pub fn total_destroyed(&self) -> u64 {
        self.total_destroyed
    }

    /// List key IDs that need rotation (usage >= threshold).
    pub fn keys_needing_rotation(&self, buf: &mut [u64]) -> usize {
        let mut count = 0;
        for slot in &self.slots {
            if slot.needs_rotation() && count < buf.len() {
                buf[count] = slot.key_id;
                count += 1;
            }
        }
        count
    }

    /// Whether the crypto self-test has passed.
    pub fn self_test_passed(&self) -> bool {
        self.self_test_passed
    }

    /// Run FIPS 140-3 Level 1 power-on self-tests.
    ///
    /// Executes Known Answer Tests (KATs) for:
    /// - SHA-3-256 (production implementation)
    /// - AES-256-GCM (stub — always passes)
    ///
    /// Returns `Ok(())` if all tests pass, `Err(SelfTestFailed)` otherwise.
    /// The crypto module MUST NOT perform operations if self-test fails.
    pub fn run_self_test(&mut self) -> Result<(), KeyManagerError> {
        // KAT 1: SHA-3-256 on empty input
        // NIST test vector: SHA-3-256("") = a7ffc6f8bf1ed76651c14756a061d662f580ff4de43b49fa82d80a4b80f8434a
        let empty_hash = crate::crypto::sha3::sha3_256(b"");
        let expected_empty: [u8; 32] = [
            0xa7, 0xff, 0xc6, 0xf8, 0xbf, 0x1e, 0xd7, 0x66,
            0x51, 0xc1, 0x47, 0x56, 0xa0, 0x61, 0xd6, 0x62,
            0xf5, 0x80, 0xff, 0x4d, 0xe4, 0x3b, 0x49, 0xfa,
            0x82, 0xd8, 0x0a, 0x4b, 0x80, 0xf8, 0x43, 0x4a,
        ];
        if empty_hash.as_bytes() != &expected_empty {
            self.self_test_passed = false;
            return Err(KeyManagerError::SelfTestFailed);
        }

        // KAT 2: SHA-3-256 on "abc"
        // NIST test vector: SHA-3-256("abc") = 3a985da74fe225b2045c172d6bd390bd855f086e3e9d525b46bfe24511431532
        let abc_hash = crate::crypto::sha3::sha3_256(b"abc");
        let expected_abc: [u8; 32] = [
            0x3a, 0x98, 0x5d, 0xa7, 0x4f, 0xe2, 0x25, 0xb2,
            0x04, 0x5c, 0x17, 0x2d, 0x6b, 0xd3, 0x90, 0xbd,
            0x85, 0x5f, 0x08, 0x6e, 0x3e, 0x9d, 0x52, 0x5b,
            0x46, 0xbf, 0xe2, 0x45, 0x11, 0x43, 0x15, 0x32,
        ];
        if abc_hash.as_bytes() != &expected_abc {
            self.self_test_passed = false;
            return Err(KeyManagerError::SelfTestFailed);
        }

        // KAT 3: AES-256-GCM (stub - other crypto is not yet implemented)
        // When AES-GCM is implemented, this will use a NIST test vector.
        // For now, this is a placeholder that passes.

        self.self_test_passed = true;
        Ok(())
    }
}

impl Default for KeyManager {
    fn default() -> Self {
        Self::new()
    }
}

// ─── Standalone Zeroize ─────────────────────────────────────────────────────

/// Securely zeroize a byte slice using volatile writes.
///
/// Guaranteed not to be optimized away by the compiler.
/// Must be called before releasing memory containing key material.
pub fn zeroize_bytes(buf: &mut [u8]) {
    for byte in buf.iter_mut() {
        unsafe {
            core::ptr::write_volatile(byte, 0u8);
        }
    }
}

/// Verify that a byte slice has been properly zeroized.
///
/// Uses volatile reads to prevent the compiler from optimizing
/// away the verification.
pub fn verify_zeroized(buf: &[u8]) -> bool {
    let mut acc: u8 = 0;
    for byte in buf {
        let val = unsafe { core::ptr::read_volatile(byte) };
        acc |= val;
    }
    acc == 0
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn register_and_destroy() {
        let mut mgr = KeyManager::new();
        let material = [0xAA; 32];
        let id = mgr.register_key(KeyType::Aes256, &material, 1000).unwrap();
        assert_eq!(id, 1);
        assert_eq!(mgr.active_count(), 1);
        assert_eq!(mgr.key_type(id).unwrap(), KeyType::Aes256);
        assert_eq!(mgr.key_state(id).unwrap(), KeyState::Active);

        mgr.destroy_key(id).unwrap();
        assert_eq!(mgr.active_count(), 0);
        assert_eq!(mgr.key_state(id).unwrap(), KeyState::Destroyed);
        assert_eq!(mgr.total_destroyed(), 1);
    }

    #[test]
    fn usage_tracking() {
        let mut mgr = KeyManager::new();
        let id = mgr.register_key(KeyType::Aes256, &[0x42; 32], 1000).unwrap();

        // First usage should not trigger rotation
        let needs_rotation = mgr.record_usage(id).unwrap();
        assert!(!needs_rotation);
        assert_eq!(mgr.usage_count(id).unwrap(), 1);

        // Record many usages
        for _ in 0..99 {
            mgr.record_usage(id).unwrap();
        }
        assert_eq!(mgr.usage_count(id).unwrap(), 100);
    }

    #[test]
    fn rotation_warning() {
        let mut mgr = KeyManager::new();
        let id = mgr.register_key(KeyType::Aes256, &[0x42; 32], 1000).unwrap();

        // Directly set usage count near threshold via multiple registrations
        // We can't easily hit 2^32, but we can test the needs_rotation logic
        // by accessing the slot directly in the test
        let slot = mgr.slots.iter_mut().find(|s| s.key_id == id).unwrap();
        slot.usage_count = ROTATION_WARNING_THRESHOLD;
        assert!(slot.needs_rotation());

        let mut buf = [0u64; 4];
        let count = mgr.keys_needing_rotation(&mut buf);
        assert_eq!(count, 1);
        assert_eq!(buf[0], id);
    }

    #[test]
    fn mark_for_rotation() {
        let mut mgr = KeyManager::new();
        let id = mgr.register_key(KeyType::MlDsa65, &[0x11; 64], 1000).unwrap();
        mgr.mark_for_rotation(id).unwrap();
        assert_eq!(mgr.key_state(id).unwrap(), KeyState::PendingRotation);
        assert_eq!(mgr.pending_rotation_count(), 1);
    }

    #[test]
    fn destroy_all() {
        let mut mgr = KeyManager::new();
        mgr.register_key(KeyType::Aes256, &[0xAA; 32], 1000).unwrap();
        mgr.register_key(KeyType::MlDsa65, &[0xBB; 64], 2000).unwrap();
        mgr.register_key(KeyType::Tls13Session, &[0xCC; 48], 3000).unwrap();

        assert_eq!(mgr.active_count(), 3);
        let (count, verified) = mgr.destroy_all();
        assert_eq!(count, 3);
        assert!(verified);
        assert_eq!(mgr.active_count(), 0);
    }

    #[test]
    fn registry_full() {
        let mut mgr = KeyManager::new();
        for i in 0..MAX_KEYS {
            mgr.register_key(KeyType::Aes256, &[i as u8; 32], i as u64 * 100)
                .unwrap();
        }
        let result = mgr.register_key(KeyType::Aes256, &[0xFF; 32], 99999);
        assert_eq!(result, Err(KeyManagerError::RegistryFull));
    }

    #[test]
    fn reuse_destroyed_slot() {
        let mut mgr = KeyManager::new();
        let id1 = mgr.register_key(KeyType::Aes256, &[0xAA; 32], 1000).unwrap();
        mgr.destroy_key(id1).unwrap();

        // Should be able to register a new key in the freed slot
        let id2 = mgr.register_key(KeyType::MlKem768, &[0xBB; 64], 2000).unwrap();
        assert_ne!(id1, id2); // Different ID
        assert_eq!(mgr.active_count(), 1);
    }

    #[test]
    fn material_too_large() {
        let mut mgr = KeyManager::new();
        let big = [0u8; MAX_KEY_MATERIAL + 1];
        let result = mgr.register_key(KeyType::Aes256, &big, 1000);
        assert_eq!(result, Err(KeyManagerError::MaterialTooLarge));
    }

    #[test]
    fn destroy_nonexistent() {
        let mut mgr = KeyManager::new();
        let result = mgr.destroy_key(999);
        assert_eq!(result, Err(KeyManagerError::KeyNotFound));
    }

    #[test]
    fn usage_on_inactive_key() {
        let mut mgr = KeyManager::new();
        let id = mgr.register_key(KeyType::Aes256, &[0xAA; 32], 1000).unwrap();
        mgr.destroy_key(id).unwrap();
        let result = mgr.record_usage(id);
        assert_eq!(result, Err(KeyManagerError::KeyNotActive));
    }

    #[test]
    fn zeroize_bytes_works() {
        let mut buf = [0xAA; 64];
        zeroize_bytes(&mut buf);
        assert!(verify_zeroized(&buf));
    }

    #[test]
    fn verify_zeroized_detects_nonzero() {
        let buf = [0x01; 4];
        assert!(!verify_zeroized(&buf));
    }

    #[test]
    fn verify_zeroized_empty() {
        let buf: [u8; 0] = [];
        assert!(verify_zeroized(&buf));
    }

    #[test]
    fn self_test_sha3() {
        let mut mgr = KeyManager::new();
        assert!(!mgr.self_test_passed());
        mgr.run_self_test().unwrap();
        assert!(mgr.self_test_passed());
    }

    #[test]
    fn key_type_roundtrip() {
        for i in 0..=6u8 {
            let kt = KeyType::from_u8(i).unwrap();
            assert_eq!(kt as u8, i);
        }
        assert!(KeyType::from_u8(7).is_none());
    }

    #[test]
    fn key_type_as_str() {
        assert_eq!(KeyType::MlKem768.as_str(), "ml-kem-768");
        assert_eq!(KeyType::MlDsa65.as_str(), "ml-dsa-65");
        assert_eq!(KeyType::Aes256.as_str(), "aes-256");
        assert_eq!(KeyType::Tls13Session.as_str(), "tls13-session");
        assert_eq!(KeyType::HkdfDerived.as_str(), "hkdf-derived");
        assert_eq!(KeyType::CsprngState.as_str(), "csprng-state");
        assert_eq!(KeyType::AuditSigning.as_str(), "audit-signing");
    }

    #[test]
    fn key_type_max_size() {
        assert_eq!(KeyType::Aes256.max_key_size(), 32);
        assert_eq!(KeyType::Tls13Session.max_key_size(), 48);
        assert_eq!(KeyType::MlKem768.max_key_size(), 64);
    }

    #[test]
    fn multiple_key_types() {
        let mut mgr = KeyManager::new();
        let id1 = mgr.register_key(KeyType::Aes256, &[0x11; 32], 1000).unwrap();
        let id2 = mgr.register_key(KeyType::MlKem768, &[0x22; 64], 2000).unwrap();
        let id3 = mgr.register_key(KeyType::AuditSigning, &[0x33; 64], 3000).unwrap();

        assert_eq!(mgr.key_type(id1).unwrap(), KeyType::Aes256);
        assert_eq!(mgr.key_type(id2).unwrap(), KeyType::MlKem768);
        assert_eq!(mgr.key_type(id3).unwrap(), KeyType::AuditSigning);
        assert_eq!(mgr.total_generated(), 3);
    }
}
