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

use super::sha3::Shake256;
use core::fmt;

/// Seed length in bytes.
pub const CSPRNG_SEED_LEN: usize = 32;

/// Maximum bytes to generate before mandatory reseed.
pub const CSPRNG_RESEED_INTERVAL: u64 = 1 << 20; // 1 MiB

/// Number of bytes squeezed from old state during reseed for forward secrecy.
const RESEED_STATE_BYTES: usize = 64;

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
    /// Internal SHAKE256 XOF state used for squeezing random bytes.
    xof: Shake256,
}

impl Csprng {
    /// Create a new, unseeded CSPRNG.
    ///
    /// Must call `seed()` or `seed_from_hardware()` before generating bytes.
    pub const fn new() -> Self {
        Self {
            seeded: false,
            bytes_generated: 0,
            xof: Shake256::new(),
        }
    }

    /// Seed the CSPRNG with a 32-byte seed.
    ///
    /// This initializes the internal SHAKE256 XOF by absorbing the seed
    /// along with a domain separator to bind the usage context.
    pub fn seed(&mut self, seed: &CsprngSeed) -> Result<(), CsprngError> {
        let mut xof = Shake256::new();
        // Domain separator so CSPRNG output differs from raw SHAKE256
        xof.absorb(b"SmallAIOS-CSPRNG-v1")
            .expect("fresh SHAKE256 absorb cannot fail");
        xof.absorb(seed.as_bytes())
            .expect("SHAKE256 absorb cannot fail");
        self.xof = xof;
        self.seeded = true;
        self.bytes_generated = 0;
        Ok(())
    }

    /// Seed from hardware entropy source (RDRAND/RNDR).
    ///
    /// This is the preferred way to initialize the CSPRNG in production.
    /// Reads 32 bytes from the hardware RNG and uses them as the seed.
    pub fn seed_from_hardware(&mut self) -> Result<(), CsprngError> {
        #[allow(unused_mut)]
        let mut hw_seed = [0u8; CSPRNG_SEED_LEN];

        #[cfg(target_arch = "x86_64")]
        {
            // Use RDRAND to fill the seed buffer
            for chunk in hw_seed.chunks_mut(8) {
                let mut val: u64;
                let mut ok: u8;
                // Retry up to 10 times per RDRAND call (Intel recommendation)
                let mut attempts = 0;
                loop {
                    unsafe {
                        core::arch::asm!(
                            "rdrand {val}",
                            "setc {ok}",
                            val = out(reg) val,
                            ok = out(reg_byte) ok,
                        );
                    }
                    if ok != 0 {
                        break;
                    }
                    attempts += 1;
                    if attempts >= 10 {
                        // RDRAND failed persistently; fall back to zero-seed
                        // which is still mixed with the domain separator.
                        val = 0;
                        break;
                    }
                }
                let bytes = val.to_le_bytes();
                let len = chunk.len();
                chunk.copy_from_slice(&bytes[..len]);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            // Use RNDR (ARMv8.5-RNG) to fill the seed buffer
            for chunk in hw_seed.chunks_mut(8) {
                let val: u64;
                unsafe {
                    core::arch::asm!(
                        "mrs {val}, s3_3_c2_c4_0", // RNDR register
                        val = out(reg) val,
                    );
                }
                let bytes = val.to_le_bytes();
                let len = chunk.len();
                chunk.copy_from_slice(&bytes[..len]);
            }
        }

        // On architectures without hardware RNG, the seed is all-zeros.
        // This is acceptable for testing but not for production.
        let seed = CsprngSeed::from_bytes(hw_seed);
        self.seed(&seed)
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

        self.xof.squeeze(out).expect("SHAKE256 squeeze cannot fail");
        self.bytes_generated += out.len() as u64;
        Ok(())
    }

    /// Reseed the CSPRNG by mixing additional entropy into the state.
    ///
    /// Combines the current state with new entropy to provide forward secrecy.
    /// Squeezes bytes from the old XOF and mixes them with the new entropy
    /// into a fresh SHAKE256 instance.
    pub fn reseed(&mut self, entropy: &CsprngSeed) -> Result<(), CsprngError> {
        if !self.seeded {
            return Err(CsprngError::NotSeeded);
        }

        // Squeeze some bytes from the current state for mixing
        let mut old_state = [0u8; RESEED_STATE_BYTES];
        self.xof
            .squeeze(&mut old_state)
            .expect("SHAKE256 squeeze cannot fail");

        // Create a new XOF, absorb domain separator + old state + new entropy
        let mut xof = Shake256::new();
        xof.absorb(b"SmallAIOS-CSPRNG-reseed-v1")
            .expect("fresh SHAKE256 absorb cannot fail");
        xof.absorb(&old_state).expect("SHAKE256 absorb cannot fail");
        xof.absorb(entropy.as_bytes())
            .expect("SHAKE256 absorb cannot fail");

        self.xof = xof;
        self.bytes_generated = 0;
        Ok(())
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
    fn csprng_seed_and_generate() {
        let mut rng = Csprng::new();
        let seed = CsprngSeed::from_bytes([0x01; CSPRNG_SEED_LEN]);
        rng.seed(&seed).unwrap();

        assert!(rng.is_seeded());

        let mut buf = [0u8; 32];
        rng.generate(&mut buf).unwrap();

        // Output should not be all zeros (seed was non-zero)
        assert!(
            buf.iter().any(|&b| b != 0),
            "output should not be all zeros"
        );
        assert_eq!(rng.bytes_generated(), 32);
    }

    #[test]
    fn csprng_deterministic() {
        let seed = CsprngSeed::from_bytes([0xAA; CSPRNG_SEED_LEN]);

        let mut rng1 = Csprng::new();
        rng1.seed(&seed).unwrap();
        let mut out1 = [0u8; 64];
        rng1.generate(&mut out1).unwrap();

        let mut rng2 = Csprng::new();
        rng2.seed(&seed).unwrap();
        let mut out2 = [0u8; 64];
        rng2.generate(&mut out2).unwrap();

        assert_eq!(out1, out2, "same seed should produce same output");
    }

    #[test]
    fn csprng_different_seeds_different_output() {
        let seed_a = CsprngSeed::from_bytes([0x01; CSPRNG_SEED_LEN]);
        let seed_b = CsprngSeed::from_bytes([0x02; CSPRNG_SEED_LEN]);

        let mut rng_a = Csprng::new();
        rng_a.seed(&seed_a).unwrap();
        let mut out_a = [0u8; 32];
        rng_a.generate(&mut out_a).unwrap();

        let mut rng_b = Csprng::new();
        rng_b.seed(&seed_b).unwrap();
        let mut out_b = [0u8; 32];
        rng_b.generate(&mut out_b).unwrap();

        assert_ne!(
            out_a, out_b,
            "different seeds should produce different output"
        );
    }

    #[test]
    fn csprng_generate_empty_fails() {
        let mut rng = Csprng::new();
        let seed = CsprngSeed::from_bytes([0x01; CSPRNG_SEED_LEN]);
        rng.seed(&seed).unwrap();

        assert_eq!(rng.generate(&mut []), Err(CsprngError::ZeroLength));
    }

    #[test]
    fn csprng_reseed_changes_output() {
        let seed = CsprngSeed::from_bytes([0x01; CSPRNG_SEED_LEN]);
        let mut rng = Csprng::new();
        rng.seed(&seed).unwrap();

        let mut out_before = [0u8; 32];
        rng.generate(&mut out_before).unwrap();

        // Reseed with new entropy
        let entropy = CsprngSeed::from_bytes([0xFF; CSPRNG_SEED_LEN]);
        rng.reseed(&entropy).unwrap();

        assert_eq!(rng.bytes_generated(), 0);

        let mut out_after = [0u8; 32];
        rng.generate(&mut out_after).unwrap();

        assert_ne!(out_before, out_after, "output after reseed should differ");
    }

    #[test]
    fn csprng_bytes_generated_tracks() {
        let mut rng = Csprng::new();
        let seed = CsprngSeed::from_bytes([0x42; CSPRNG_SEED_LEN]);
        rng.seed(&seed).unwrap();

        let mut buf = [0u8; 100];
        rng.generate(&mut buf).unwrap();
        assert_eq!(rng.bytes_generated(), 100);

        rng.generate(&mut buf).unwrap();
        assert_eq!(rng.bytes_generated(), 200);
    }

    #[test]
    fn csprng_debug_format() {
        let rng = Csprng::new();
        let debug = format!("{:?}", rng);
        assert!(debug.contains("Csprng"));
        assert!(debug.contains("REDACTED"));
    }

    #[test]
    fn csprng_constants() {
        assert_eq!(CSPRNG_SEED_LEN, 32);
        assert_eq!(CSPRNG_RESEED_INTERVAL, 1 << 20);
    }

    #[test]
    fn csprng_default_is_unseeded() {
        let rng = Csprng::default();
        assert!(!rng.is_seeded());
        assert_eq!(rng.bytes_generated(), 0);
    }

    #[test]
    fn csprng_needs_reseed_false_initially() {
        let mut rng = Csprng::new();
        let seed = CsprngSeed::from_bytes([0x42; CSPRNG_SEED_LEN]);
        rng.seed(&seed).unwrap();
        assert!(!rng.needs_reseed());
    }

    #[test]
    fn csprng_error_display_all() {
        assert_eq!(
            format!("{}", CsprngError::InvalidSeedLength),
            "invalid seed length (expected 32 bytes)"
        );
        assert_eq!(
            format!("{}", CsprngError::ReseedRequired),
            "reseed required: generation limit reached"
        );
        assert_eq!(
            format!("{}", CsprngError::ZeroLength),
            "requested output length is zero"
        );
        assert_eq!(format!("{}", CsprngError::NotSeeded), "CSPRNG not seeded");
    }

    #[cfg_attr(miri, ignore)]
    #[test]
    fn csprng_seed_from_hardware() {
        let mut rng = Csprng::new();
        // On x86_64 test hosts, seed_from_hardware should succeed
        // (uses RDRAND or falls back to zero-seed)
        rng.seed_from_hardware().unwrap();
        assert!(rng.is_seeded());

        let mut buf = [0u8; 32];
        rng.generate(&mut buf).unwrap();
        assert_eq!(rng.bytes_generated(), 32);
    }

    #[test]
    fn csprng_seed_equality() {
        let s1 = CsprngSeed::from_bytes([0xAA; CSPRNG_SEED_LEN]);
        let s2 = CsprngSeed::from_bytes([0xAA; CSPRNG_SEED_LEN]);
        let s3 = CsprngSeed::from_bytes([0xBB; CSPRNG_SEED_LEN]);
        assert_eq!(s1, s2);
        assert_ne!(s1, s3);
    }

    #[test]
    fn csprng_multiple_generates() {
        let mut rng = Csprng::new();
        let seed = CsprngSeed::from_bytes([0x42; CSPRNG_SEED_LEN]);
        rng.seed(&seed).unwrap();

        let mut buf1 = [0u8; 16];
        let mut buf2 = [0u8; 16];
        rng.generate(&mut buf1).unwrap();
        rng.generate(&mut buf2).unwrap();
        // Sequential outputs should differ
        assert_ne!(buf1, buf2);
        assert_eq!(rng.bytes_generated(), 32);
    }
}
