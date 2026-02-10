// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! SHAKE256-based cryptographically secure pseudo-random number generator.
//!
//! This CSPRNG uses SHAKE256 as its core extendable-output function. It is:
//!
//! - Seeded from hardware entropy (RDRAND on x86-64, RNDR on AArch64)
//! - Backtracking-resistant (reseeding supported)
//! - Deterministic from a given seed (for testing)
//!
//! # Design
//!
//! The CSPRNG maintains an internal SHAKE256 XOF state. Bytes are produced
//! by squeezing the XOF. Reseeding is accomplished by absorbing additional
//! entropy into a new SHAKE256 instance mixed with the previous state.
//!
//! # Security Properties
//!
//! - 256-bit security level (SHAKE256)
//! - Forward secrecy via periodic reseeding
//! - No state recovery from output (XOF is one-way)

#![allow(unused)]

use core::fmt;

/// Seed length in bytes.
pub const CSPRNG_SEED_LEN: usize = 32;

/// Maximum bytes to generate before mandatory reseed.
pub const CSPRNG_RESEED_INTERVAL: u64 = 1 << 20; // 1 MiB

// ─── Error Type ──────────────────────────────────────────────────────────────

/// Errors from CSPRNG operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CsprngError {
    /// Invalid seed length (must be 32 bytes).
    InvalidSeedLength,
    /// Reseed required (generation limit reached).
    ReseedRequired,
    /// Requested output length is zero.
    ZeroLength,
    /// CSPRNG not yet seeded.
    NotSeeded,
    /// Operation not yet implemented.
    NotImplemented,
}

impl fmt::Display for CsprngError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSeedLength => write!(
                f,
                "invalid seed length (expected {} bytes)",
                CSPRNG_SEED_LEN
            ),
            Self::ReseedRequired => write!(f, "reseed required: generation limit reached"),
            Self::ZeroLength => write!(f, "requested output length is zero"),
            Self::NotSeeded => write!(f, "CSPRNG not seeded"),
            Self::NotImplemented => write!(f, "CSPRNG not yet implemented"),
        }
    }
}

// ─── Types ───────────────────────────────────────────────────────────────────

/// A 32-byte CSPRNG seed.
#[derive(Clone, PartialEq, Eq)]
pub struct CsprngSeed {
    bytes: [u8; CSPRNG_SEED_LEN],
}

impl CsprngSeed {
    /// Create a seed from a 32-byte array.
    pub fn from_bytes(bytes: [u8; CSPRNG_SEED_LEN]) -> Self {
        Self { bytes }
    }

    /// Create a seed from a byte slice, validating length.
    pub fn from_slice(slice: &[u8]) -> Result<Self, CsprngError> {
        if slice.len() != CSPRNG_SEED_LEN {
            return Err(CsprngError::InvalidSeedLength);
        }
        let mut bytes = [0u8; CSPRNG_SEED_LEN];
        bytes.copy_from_slice(slice);
        Ok(Self { bytes })
    }

    /// Return the seed bytes.
    pub fn as_bytes(&self) -> &[u8; CSPRNG_SEED_LEN] {
        &self.bytes
    }
}

impl fmt::Debug for CsprngSeed {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "CsprngSeed([REDACTED])")
    }
}

/// SHAKE256-based CSPRNG state.
///
/// Generates cryptographically secure random bytes by squeezing a
/// SHAKE256 XOF seeded from hardware entropy or an explicit seed.
pub struct Csprng {
    /// Whether the CSPRNG has been seeded.
    seeded: bool,
    /// Bytes generated since last (re)seed.
    bytes_generated: u64,
    /// Internal state placeholder (will hold SHAKE256 XOF state).
    /// Using a fixed-size buffer to represent the Keccak state (200 bytes = 1600 bits).
    state: [u8; 200],
}

impl Csprng {
    /// Create a new, unseeded CSPRNG.
    ///
    /// Must call `seed()` or `seed_from_hardware()` before generating bytes.
    pub fn new() -> Self {
        Self {
            seeded: false,
            bytes_generated: 0,
            state: [0u8; 200],
        }
    }

    /// Seed the CSPRNG with a 32-byte seed.
    ///
    /// This initializes the internal SHAKE256 XOF by absorbing the seed.
    pub fn seed(&mut self, seed: &CsprngSeed) -> Result<(), CsprngError> {
        // Stub: will initialize SHAKE256 with seed
        self.seeded = true;
        self.bytes_generated = 0;
        // Mix seed into state placeholder
        for (i, &b) in seed.as_bytes().iter().enumerate() {
            self.state[i] = b;
        }
        Err(CsprngError::NotImplemented)
    }

    /// Seed from hardware entropy source (RDRAND/RNDR).
    ///
    /// This is the preferred way to initialize the CSPRNG in production.
    pub fn seed_from_hardware(&mut self) -> Result<(), CsprngError> {
        // Stub: will read from RDRAND (x86-64) or RNDR (AArch64)
        Err(CsprngError::NotImplemented)
    }

    /// Generate random bytes into the output buffer.
    ///
    /// # Errors
    ///
    /// - `NotSeeded` if `seed()` has not been called
    /// - `ReseedRequired` if the generation limit has been reached
    /// - `ZeroLength` if `out` is empty
    pub fn generate(&mut self, out: &mut [u8]) -> Result<(), CsprngError> {
        if !self.seeded {
            return Err(CsprngError::NotSeeded);
        }
        if out.is_empty() {
            return Err(CsprngError::ZeroLength);
        }
        if self.bytes_generated >= CSPRNG_RESEED_INTERVAL {
            return Err(CsprngError::ReseedRequired);
        }
        // Stub: will squeeze SHAKE256 XOF
        Err(CsprngError::NotImplemented)
    }

    /// Reseed the CSPRNG by mixing additional entropy into the state.
    ///
    /// Combines the current state with new entropy to provide forward secrecy.
    pub fn reseed(&mut self, entropy: &CsprngSeed) -> Result<(), CsprngError> {
        if !self.seeded {
            return Err(CsprngError::NotSeeded);
        }
        // Stub: will create new SHAKE256, absorb old state + new entropy
        self.bytes_generated = 0;
        Err(CsprngError::NotImplemented)
    }

    /// Return the number of bytes generated since the last (re)seed.
    pub fn bytes_generated(&self) -> u64 {
        self.bytes_generated
    }

    /// Check if the CSPRNG has been seeded.
    pub fn is_seeded(&self) -> bool {
        self.seeded
    }

    /// Check if reseeding is required.
    pub fn needs_reseed(&self) -> bool {
        self.bytes_generated >= CSPRNG_RESEED_INTERVAL
    }
}

impl Default for Csprng {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Csprng {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Csprng")
            .field("seeded", &self.seeded)
            .field("bytes_generated", &self.bytes_generated)
            .field("state", &"[REDACTED]")
            .finish()
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    #[test]
    fn csprng_seed_from_bytes() {
        let seed = CsprngSeed::from_bytes([0x42; CSPRNG_SEED_LEN]);
        assert_eq!(seed.as_bytes()[0], 0x42);
        assert_eq!(seed.as_bytes().len(), CSPRNG_SEED_LEN);
    }

    #[test]
    fn csprng_seed_from_slice_valid() {
        let bytes = [0xAB; CSPRNG_SEED_LEN];
        let seed = CsprngSeed::from_slice(&bytes).unwrap();
        assert_eq!(seed.as_bytes(), &bytes);
    }

    #[test]
    fn csprng_seed_from_slice_invalid() {
        assert_eq!(
            CsprngSeed::from_slice(&[0u8; 16]),
            Err(CsprngError::InvalidSeedLength)
        );
    }

    #[test]
    fn csprng_seed_debug_redacted() {
        let seed = CsprngSeed::from_bytes([0xFF; CSPRNG_SEED_LEN]);
        let debug = format!("{:?}", seed);
        assert_eq!(debug, "CsprngSeed([REDACTED])");
    }

    #[test]
    fn csprng_new_is_unseeded() {
        let rng = Csprng::new();
        assert!(!rng.is_seeded());
        assert_eq!(rng.bytes_generated(), 0);
    }

    #[test]
    fn csprng_generate_without_seed_fails() {
        let mut rng = Csprng::new();
        let mut buf = [0u8; 32];
        assert_eq!(rng.generate(&mut buf), Err(CsprngError::NotSeeded));
    }

    #[test]
    fn csprng_reseed_without_seed_fails() {
        let mut rng = Csprng::new();
        let entropy = CsprngSeed::from_bytes([0; CSPRNG_SEED_LEN]);
        assert_eq!(rng.reseed(&entropy), Err(CsprngError::NotSeeded));
    }

    #[test]
    fn csprng_seed_returns_not_implemented() {
        let mut rng = Csprng::new();
        let seed = CsprngSeed::from_bytes([0x01; CSPRNG_SEED_LEN]);
        assert_eq!(rng.seed(&seed), Err(CsprngError::NotImplemented));
    }

    #[test]
    fn csprng_debug_format() {
        let rng = Csprng::new();
        let debug = format!("{:?}", rng);
        assert!(debug.contains("Csprng"));
        assert!(debug.contains("REDACTED"));
        assert!(!debug.contains("00"), "state must not leak in debug output");
    }

    #[test]
    fn csprng_constants() {
        assert_eq!(CSPRNG_SEED_LEN, 32);
        assert_eq!(CSPRNG_RESEED_INTERVAL, 1 << 20);
    }
}
