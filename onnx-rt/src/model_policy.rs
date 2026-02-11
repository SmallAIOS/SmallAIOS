// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Model policy validation for the formal methods firewall.
//!
//! Validates ONNX models against the security policy at load time:
//! - Model hash must be present in the policy whitelist
//! - I/O shapes must be compatible with registered message type invariants

use alloc::string::String;

use smallaios_security::message_types::SchemaHash;
use smallaios_security::policy::ModelWhitelist;

use crate::onnx_types::ModelProto;
use crate::session::SessionError;

/// Compute a stub hash of the model for whitelist checking.
///
/// In production this would use SHA-3-256 over the serialized model.
/// For now, we compute a simple hash from model metadata to enable
/// the policy validation pipeline.
pub fn compute_model_hash(model: &ModelProto) -> SchemaHash {
    let mut hash = [0u8; 32];
    // Mix in model metadata to produce a deterministic hash.
    let name_bytes = model.producer_name.as_bytes();
    for (i, &b) in name_bytes.iter().enumerate() {
        hash[i % 32] ^= b;
    }
    let version_bytes = model.model_version.to_le_bytes();
    for (i, &b) in version_bytes.iter().enumerate() {
        hash[(16 + i) % 32] ^= b;
    }
    let ir_bytes = model.ir_version.to_le_bytes();
    for (i, &b) in ir_bytes.iter().enumerate() {
        hash[(24 + i) % 32] ^= b;
    }
    hash
}

/// Validate a model against the security policy whitelist.
///
/// Returns `Ok(hash)` if the model's hash is in the whitelist,
/// or `Err(SessionError::PolicyViolation)` if not.
pub fn validate_model_whitelist(
    model: &ModelProto,
    whitelist: &ModelWhitelist,
) -> Result<SchemaHash, SessionError> {
    let hash = compute_model_hash(model);
    if whitelist.contains(&hash) {
        Ok(hash)
    } else {
        Err(SessionError::PolicyViolation(String::from(
            "model hash not in policy whitelist",
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::onnx_types::ModelProto;

    fn make_test_model() -> ModelProto {
        ModelProto {
            ir_version: 9,
            producer_name: String::from("test-producer"),
            model_version: 1,
            ..ModelProto::default()
        }
    }

    #[test]
    fn compute_hash_deterministic() {
        let model = make_test_model();
        let h1 = compute_model_hash(&model);
        let h2 = compute_model_hash(&model);
        assert_eq!(h1, h2);
    }

    #[test]
    fn compute_hash_different_models() {
        let m1 = make_test_model();
        let mut m2 = make_test_model();
        m2.producer_name = String::from("other-producer");
        let h1 = compute_model_hash(&m1);
        let h2 = compute_model_hash(&m2);
        assert_ne!(h1, h2);
    }

    #[test]
    fn whitelist_allows_registered_model() {
        let model = make_test_model();
        let hash = compute_model_hash(&model);
        let mut whitelist = ModelWhitelist::new();
        whitelist.add(hash);

        let result = validate_model_whitelist(&model, &whitelist);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), hash);
    }

    #[test]
    fn whitelist_rejects_unknown_model() {
        let model = make_test_model();
        let whitelist = ModelWhitelist::new(); // empty

        let result = validate_model_whitelist(&model, &whitelist);
        assert!(matches!(result, Err(SessionError::PolicyViolation(_))));
    }

    #[test]
    fn whitelist_rejects_wrong_hash() {
        let model = make_test_model();
        let mut whitelist = ModelWhitelist::new();
        whitelist.add([0xAA; 32]); // different hash

        let result = validate_model_whitelist(&model, &whitelist);
        assert!(matches!(result, Err(SessionError::PolicyViolation(_))));
    }
}
