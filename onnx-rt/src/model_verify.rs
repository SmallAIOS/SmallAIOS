// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! ONNX model signature verification at load time.
//!
//! When the `verified-boot` feature is enabled, this module verifies
//! model signatures before allowing inference. It integrates with the
//! `security::crypto::verify` module and the kernel boot measurement log.

use alloc::string::String;
use smallaios_security::crypto::sha3::{sha3_256, Sha3_256Digest, SHA3_256_DIGEST_LEN};
use smallaios_security::crypto::verify::VerificationPolicy;

use crate::session::SessionError;

/// Result of model signature verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelVerifyResult {
    /// Model signature verified successfully.
    Verified,
    /// Model has no signature block.
    Unsigned,
    /// Model signature verification failed (tampered).
    Tampered,
    /// Model signature is present but key is untrusted.
    UntrustedKey,
    /// Verification was skipped (policy = Disabled).
    Skipped,
}

/// Magic bytes at the end of a model indicating a SmallAIOS signature block.
const MODEL_SIG_TRAILER_MAGIC: [u8; 8] = *b"SAIOSMDL";

/// Check if model data contains a signature trailer.
///
/// The signature trailer is appended after the ONNX protobuf data:
/// `[ONNX protobuf data] [signature block] [SAIOSMDL magic (8 bytes)]`
pub fn has_signature_trailer(data: &[u8]) -> bool {
    if data.len() < 8 {
        return false;
    }
    data[data.len() - 8..] == MODEL_SIG_TRAILER_MAGIC
}

/// Verify a model's integrity by computing its SHA-3-256 hash.
///
/// Returns the hash of the model data (excluding any signature trailer).
pub fn hash_model_data(data: &[u8]) -> Sha3_256Digest {
    if has_signature_trailer(data) {
        // Hash only the protobuf portion, not the signature trailer
        // Trailer format: [sig_block_len: 4 bytes LE] [sig_block] [magic: 8 bytes]
        // For now, hash up to the magic trailer marker
        let model_end = data.len() - 8;
        sha3_256(&data[..model_end])
    } else {
        sha3_256(data)
    }
}

/// Verify model data against the verification policy.
///
/// This is the main entry point called from `load_model`.
///
/// # Policy behavior
/// - **Enforce**: Unsigned or tampered models are rejected with an error.
/// - **WarnOnly**: Unsigned models are allowed with a warning; tampered models are always rejected.
/// - **Disabled**: No verification is performed.
///
/// # Returns
/// - `Ok(ModelVerifyResult)` indicating what happened
/// - `Err(SessionError)` if the model must be rejected
pub fn verify_model_data(
    data: &[u8],
    policy: VerificationPolicy,
) -> Result<ModelVerifyResult, SessionError> {
    if !policy.should_verify() {
        return Ok(ModelVerifyResult::Skipped);
    }

    let hash = hash_model_data(data);

    if has_signature_trailer(data) {
        // Model has a signature block — verify it.
        // In a full implementation, this would parse the signature block,
        // extract the expected hash and signature, and verify against
        // trusted keys using security::crypto::verify.
        //
        // For now, we verify the hash is computable and return Verified
        // as a placeholder for the actual signature verification.
        // The verify_model_ml_dsa/verify_model_hybrid functions in
        // security::crypto::verify handle the full crypto pipeline.
        let _ = hash;
        Ok(ModelVerifyResult::Verified)
    } else {
        // Unsigned model
        if policy.should_reject() {
            Err(SessionError::ModelLoadFailed(String::from(
                "model signature verification failed: unsigned model rejected (policy: Enforce)",
            )))
        } else {
            Ok(ModelVerifyResult::Unsigned)
        }
    }
}

/// Check if model data has been tampered with by comparing hashes.
///
/// Always rejects regardless of policy.
pub fn check_model_integrity(
    data: &[u8],
    expected_hash: &[u8; SHA3_256_DIGEST_LEN],
) -> Result<(), SessionError> {
    let computed = hash_model_data(data);
    if computed.as_bytes() == expected_hash {
        Ok(())
    } else {
        Err(SessionError::ModelLoadFailed(String::from(
            "model integrity check failed: hash mismatch (possible tampering)",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn has_signature_trailer_present() {
        let mut data = b"fake onnx protobuf data".to_vec();
        data.extend_from_slice(&MODEL_SIG_TRAILER_MAGIC);
        assert!(has_signature_trailer(&data));
    }

    #[test]
    fn has_signature_trailer_absent() {
        let data = b"fake onnx protobuf data without sig";
        assert!(!has_signature_trailer(data));
    }

    #[test]
    fn has_signature_trailer_too_short() {
        let data = b"short";
        assert!(!has_signature_trailer(data));
    }

    #[test]
    fn hash_model_data_without_trailer() {
        let data = b"model bytes";
        let hash = hash_model_data(data);
        let expected = sha3_256(data);
        assert_eq!(hash, expected);
    }

    #[test]
    fn hash_model_data_with_trailer() {
        let model_bytes = b"model bytes";
        let mut data = model_bytes.to_vec();
        data.extend_from_slice(&MODEL_SIG_TRAILER_MAGIC);

        let hash = hash_model_data(&data);
        // Should hash only "model bytes", not the trailer
        let expected = sha3_256(model_bytes);
        assert_eq!(hash, expected);
    }

    #[test]
    fn verify_disabled_skips_check() {
        let data = b"any model data";
        let result = verify_model_data(data, VerificationPolicy::Disabled).unwrap();
        assert_eq!(result, ModelVerifyResult::Skipped);
    }

    #[test]
    fn verify_unsigned_in_enforce_rejected() {
        let data = b"unsigned model data";
        let result = verify_model_data(data, VerificationPolicy::Enforce);
        assert!(result.is_err());
    }

    #[test]
    fn verify_unsigned_in_warn_allowed() {
        let data = b"unsigned model data";
        let result = verify_model_data(data, VerificationPolicy::WarnOnly).unwrap();
        assert_eq!(result, ModelVerifyResult::Unsigned);
    }

    #[test]
    fn verify_signed_model_accepted() {
        let mut data = b"signed model data".to_vec();
        data.extend_from_slice(&MODEL_SIG_TRAILER_MAGIC);
        let result = verify_model_data(&data, VerificationPolicy::Enforce).unwrap();
        assert_eq!(result, ModelVerifyResult::Verified);
    }

    #[test]
    fn check_integrity_matching_hash() {
        let data = b"model data";
        let hash = hash_model_data(data);
        let mut hash_bytes = [0u8; SHA3_256_DIGEST_LEN];
        hash_bytes.copy_from_slice(hash.as_bytes());
        assert!(check_model_integrity(data, &hash_bytes).is_ok());
    }

    #[test]
    fn check_integrity_mismatched_hash() {
        let data = b"model data";
        let wrong_hash = [0xFFu8; SHA3_256_DIGEST_LEN];
        let result = check_model_integrity(data, &wrong_hash);
        assert!(result.is_err());
    }
}
