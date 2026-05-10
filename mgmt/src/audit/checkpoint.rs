// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ML-DSA-65 signed audit checkpoints (opt-in per
//! `audit.signed_checkpoints.enabled`).
//!
//! Phase 10 ships the trait + a no-op signer; the production-side
//! ML-DSA-65 signer that signs every Nth chain head with the kernel-
//! held key is wired via the existing `security::crypto::ml_dsa::*`
//! API. Off-box verifiers consume the signed-checkpoint stream to
//! detect tampering even if the kernel's signing key is later
//! compromised.

extern crate alloc;
use alloc::vec::Vec;

/// Signs a 32-byte chain head; production wiring uses ML-DSA-65.
pub trait CheckpointSigner {
    /// Sign the chain head. Returns the raw signature bytes (≤ 4096
    /// for ML-DSA-65).
    fn sign(&self, chain_head: &[u8; 32]) -> Result<Vec<u8>, CheckpointError>;
}

/// Verifies a checkpoint signature; symmetric with [`CheckpointSigner`].
pub trait CheckpointVerifier {
    fn verify(&self, chain_head: &[u8; 32], signature: &[u8]) -> Result<(), CheckpointError>;
}

/// No-op signer: returns an empty vector. Used when
/// `audit.signed_checkpoints.enabled = false` (default) and in tests.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoopSigner;

impl CheckpointSigner for NoopSigner {
    fn sign(&self, _chain_head: &[u8; 32]) -> Result<Vec<u8>, CheckpointError> {
        Ok(Vec::new())
    }
}

/// Errors raised by the checkpoint subsystem.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointError {
    /// Signing key is not loaded or not available.
    SignerUnavailable,
    /// Signature verification failed.
    SignatureInvalid,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_signer_returns_empty() {
        let s = NoopSigner;
        let head = [0u8; 32];
        assert_eq!(s.sign(&head).unwrap(), Vec::<u8>::new());
    }
}
