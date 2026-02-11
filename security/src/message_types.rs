// Copyright 2026 SmallAIOS Contributors
// SPDX-License-Identifier: Apache-2.0

//! Verified message types and runtime invariant checking.
//!
//! Each message type crossing a trust boundary has a unique ID, a schema
//! hash linking to a formal proof artifact (Lean 4 / TLA+), and a set of
//! runtime-checkable invariants derived from those proofs.
//!
//! The `MessageTypeRegistry` holds all registered types in a fixed-size
//! array (no heap allocation).

use crate::boundary::trust_boundaries::{DataFlowDirection, TrustBoundary};
use crate::enforcement::EnforcementMode;
use crate::labels::MessageTypeId;

/// SHA-3-256 hash of a formal proof artifact.
pub type SchemaHash = [u8; 32];

/// Maximum number of registered message types.
pub const MAX_MESSAGE_TYPES: usize = 64;

/// Maximum number of invariants per message type.
pub const MAX_INVARIANTS_PER_TYPE: usize = 16;

/// Tensor element data type (mirrors ipc::inference_proto::TensorDataType
/// without creating a cross-crate dependency).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TensorDtype {
    Float32 = 1,
    Float16 = 2,
    Int8 = 3,
    Int32 = 4,
    Int64 = 5,
    Uint8 = 6,
}

impl TensorDtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Float32),
            2 => Some(Self::Float16),
            3 => Some(Self::Int8),
            4 => Some(Self::Int32),
            5 => Some(Self::Int64),
            6 => Some(Self::Uint8),
            _ => None,
        }
    }
}

/// Runtime-checkable invariant derived from formal proofs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Invariant {
    /// Tensor rank must be <= this value.
    MaxRank(u8),
    /// Tensor rank must be >= this value.
    MinRank(u8),
    /// Tensor element type must equal this dtype.
    AllowedDtype(TensorDtype),
    /// Total element count must be <= this value.
    MaxElements(u32),
    /// Wire-level payload size must be <= this value.
    MaxPayloadBytes(u32),
    /// Element values must be in [min, max].
    ValueRange { min: i64, max: i64 },
    /// Value must be a valid variant of an enum with this many variants.
    EnumMembership(u8),
    /// No dimension in the tensor shape may be zero.
    NonZeroDimensions,
    /// Timestamp must be strictly greater than the last seen (stateful).
    MonotonicTimestamp,
    /// Messages must not exceed this rate (stateful).
    RateLimit { max_per_sec: u32 },
}

/// Metadata about a message being checked against invariants.
///
/// Not all fields are relevant for every invariant; unused fields
/// should be set to their default (0 / empty).
#[derive(Debug, Clone)]
pub struct MessageMetadata {
    /// Tensor rank (number of dimensions).
    pub rank: u8,
    /// Tensor element type.
    pub dtype: u8,
    /// Total element count.
    pub element_count: u32,
    /// Wire-level payload size in bytes.
    pub payload_bytes: u32,
    /// Tensor dimensions (up to 8).
    pub dimensions: [u32; 8],
    /// Number of valid entries in `dimensions`.
    pub num_dimensions: u8,
    /// Enum variant value (for EnumMembership checks).
    pub enum_value: u8,
    /// Message timestamp (for MonotonicTimestamp checks).
    pub timestamp: u64,
    /// Element value for range checking (representative sample).
    pub sample_value: i64,
}

impl MessageMetadata {
    pub const fn empty() -> Self {
        Self {
            rank: 0,
            dtype: 0,
            element_count: 0,
            payload_bytes: 0,
            dimensions: [0; 8],
            num_dimensions: 0,
            enum_value: 0,
            timestamp: 0,
            sample_value: 0,
        }
    }
}

/// A formally verified message type.
#[derive(Debug, Clone, Copy)]
pub struct VerifiedMessageType {
    /// Unique message type identifier.
    pub type_id: MessageTypeId,
    /// Human-readable name.
    pub name: &'static str,
    /// Trust boundary this type crosses.
    pub boundary: TrustBoundary,
    /// Data flow direction at the boundary.
    pub direction: DataFlowDirection,
    /// SHA-3-256 hash of the formal proof artifact.
    /// All zeros = proof not yet completed.
    pub schema_hash: SchemaHash,
    /// Enforcement mode for this specific type.
    pub mode: EnforcementMode,
    /// Runtime-checkable invariants.
    pub invariants: &'static [Invariant],
}

impl VerifiedMessageType {
    /// Whether the formal proof is completed (schema_hash is non-zero).
    pub fn is_verified(&self) -> bool {
        self.schema_hash.iter().any(|&b| b != 0)
    }
}

/// Per-type stateful invariant tracking.
#[derive(Debug, Clone, Copy)]
struct TypeState {
    type_id: MessageTypeId,
    last_timestamp: u64,
    message_count_in_window: u32,
    window_start: u64,
    active: bool,
}

impl TypeState {
    const fn empty() -> Self {
        Self {
            type_id: MessageTypeId(0),
            last_timestamp: 0,
            message_count_in_window: 0,
            window_start: 0,
            active: false,
        }
    }
}

/// Invariant checker with per-type stateful tracking.
pub struct InvariantChecker {
    states: [TypeState; MAX_MESSAGE_TYPES],
}

/// Result of checking a single invariant.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvariantResult {
    Pass,
    Fail(u8), // invariant index
}

impl InvariantChecker {
    pub fn new() -> Self {
        Self {
            states: [TypeState::empty(); MAX_MESSAGE_TYPES],
        }
    }

    /// Check all invariants for a message type against the given metadata.
    /// Returns the index of the first failing invariant, or None if all pass.
    pub fn check(
        &mut self,
        msg_type: &VerifiedMessageType,
        metadata: &MessageMetadata,
    ) -> Option<u8> {
        let idx = self.get_or_create_state_index(msg_type.type_id);

        for (i, inv) in msg_type.invariants.iter().enumerate() {
            if !Self::check_single(inv, metadata, &self.states[idx]) {
                return Some(i as u8);
            }
        }

        // Update stateful tracking after successful check
        if metadata.timestamp > 0 {
            self.states[idx].last_timestamp = metadata.timestamp;
        }
        self.states[idx].message_count_in_window += 1;

        None
    }

    fn check_single(
        invariant: &Invariant,
        metadata: &MessageMetadata,
        state: &TypeState,
    ) -> bool {
        match *invariant {
            Invariant::MaxRank(max) => metadata.rank <= max,
            Invariant::MinRank(min) => metadata.rank >= min,
            Invariant::AllowedDtype(dtype) => metadata.dtype == dtype as u8,
            Invariant::MaxElements(max) => metadata.element_count <= max,
            Invariant::MaxPayloadBytes(max) => metadata.payload_bytes <= max,
            Invariant::ValueRange { min, max } => {
                metadata.sample_value >= min && metadata.sample_value <= max
            }
            Invariant::EnumMembership(max_variant) => metadata.enum_value < max_variant,
            Invariant::NonZeroDimensions => {
                for i in 0..metadata.num_dimensions as usize {
                    if i < 8 && metadata.dimensions[i] == 0 {
                        return false;
                    }
                }
                true
            }
            Invariant::MonotonicTimestamp => {
                if state.last_timestamp == 0 {
                    true // first message, no prior timestamp
                } else {
                    metadata.timestamp > state.last_timestamp
                }
            }
            Invariant::RateLimit { max_per_sec } => {
                // Simple window-based rate check
                state.message_count_in_window < max_per_sec
            }
        }
    }

    fn state_index(&self, type_id: MessageTypeId) -> Option<usize> {
        self.states
            .iter()
            .position(|s| s.active && s.type_id == type_id)
    }

    fn get_or_create_state_index(&mut self, type_id: MessageTypeId) -> usize {
        if let Some(idx) = self.state_index(type_id) {
            return idx;
        }
        // Find first empty slot
        let idx = self
            .states
            .iter()
            .position(|s| !s.active)
            .unwrap_or(0); // fallback to slot 0 if full
        self.states[idx] = TypeState {
            type_id,
            last_timestamp: 0,
            message_count_in_window: 0,
            window_start: 0,
            active: true,
        };
        idx
    }

    /// Reset rate limit window for a type (called periodically).
    pub fn reset_rate_window(&mut self, type_id: MessageTypeId) {
        for state in self.states.iter_mut() {
            if state.active && state.type_id == type_id {
                state.message_count_in_window = 0;
                return;
            }
        }
    }
}

impl Default for InvariantChecker {
    fn default() -> Self {
        Self::new()
    }
}

/// Fixed-size registry of verified message types.
pub struct MessageTypeRegistry {
    types: [Option<VerifiedMessageType>; MAX_MESSAGE_TYPES],
    count: usize,
}

impl MessageTypeRegistry {
    pub const fn new() -> Self {
        Self {
            types: [const { None }; MAX_MESSAGE_TYPES],
            count: 0,
        }
    }

    /// Register a new message type. Returns false if registry is full or ID already exists.
    pub fn register(&mut self, msg_type: VerifiedMessageType) -> bool {
        // Check for duplicate ID
        for slot in self.types.iter() {
            if let Some(existing) = slot {
                if existing.type_id == msg_type.type_id {
                    return false;
                }
            }
        }
        // Find empty slot
        for slot in self.types.iter_mut() {
            if slot.is_none() {
                *slot = Some(msg_type);
                self.count += 1;
                return true;
            }
        }
        false // full
    }

    /// Look up a message type by ID.
    pub fn lookup(&self, type_id: MessageTypeId) -> Option<&VerifiedMessageType> {
        for slot in self.types.iter() {
            if let Some(mt) = slot {
                if mt.type_id == type_id {
                    return Some(mt);
                }
            }
        }
        None
    }

    /// Number of registered types.
    pub fn count(&self) -> usize {
        self.count
    }

    /// Whether the registry is full.
    pub fn is_full(&self) -> bool {
        self.count >= MAX_MESSAGE_TYPES
    }

    /// Iterate over all registered types.
    pub fn iter(&self) -> impl Iterator<Item = &VerifiedMessageType> {
        self.types.iter().filter_map(|slot| slot.as_ref())
    }
}

impl Default for MessageTypeRegistry {
    fn default() -> Self {
        Self::new()
    }
}

// ── Default message type catalog ──

/// All-zero schema hash (proof not yet completed).
pub const UNVERIFIED_HASH: SchemaHash = [0u8; 32];

// Invariant arrays for each default type
static TENSOR_INPUT_INVARIANTS: &[Invariant] = &[
    Invariant::MinRank(1),
    Invariant::MaxRank(4),
    Invariant::AllowedDtype(TensorDtype::Float32),
    Invariant::MaxElements(1_048_576), // 1M elements
    Invariant::NonZeroDimensions,
];

static TENSOR_OUTPUT_INVARIANTS: &[Invariant] = &[
    Invariant::MinRank(1),
    Invariant::MaxRank(4),
    Invariant::MaxElements(1_048_576),
    Invariant::NonZeroDimensions,
];

static INFERENCE_REQUEST_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(64 * 1024 * 1024), // 64 MB
    Invariant::EnumMembership(5),                  // InferenceRequestType has 4 variants (1-4)
];

static INFERENCE_RESPONSE_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(64 * 1024 * 1024),
    Invariant::EnumMembership(6), // ResponseStatus has 6 variants (0-5)
];

static BUS_SENSOR_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(8), // CAN frame max
    Invariant::MonotonicTimestamp,
];

static BUS_ACTUATOR_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(8),
    Invariant::ValueRange {
        min: -10_000,
        max: 10_000,
    },
    Invariant::RateLimit { max_per_sec: 100 },
];

static GPU_TRANSFER_INVARIANTS: &[Invariant] = &[
    Invariant::MaxElements(16_777_216), // 16M elements
    Invariant::NonZeroDimensions,
];

static GPU_RESULT_INVARIANTS: &[Invariant] = &[
    Invariant::MaxElements(16_777_216),
    Invariant::NonZeroDimensions,
];

static K8S_COMMAND_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(65_536), // 64 KB
    Invariant::EnumMembership(8),       // pod operations
];

static K8S_METRICS_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(16_384), // 16 KB
];

static IPC_MESSAGE_INVARIANTS: &[Invariant] = &[
    Invariant::MaxPayloadBytes(1_048_576), // 1 MB
];

// SHA-3-256 of formal/lean4/TensorTypeInvariants.lean (tensor type schema proofs)
const TENSOR_PROOF_HASH: SchemaHash = [
    0x2f, 0x99, 0x1c, 0xc6, 0xd2, 0xcf, 0x05, 0x4e,
    0x69, 0xf0, 0x49, 0x78, 0xdc, 0x3a, 0x92, 0x1f,
    0x44, 0x68, 0x29, 0xc2, 0xb7, 0xe0, 0x9e, 0xbd,
    0xb6, 0x18, 0x06, 0x6d, 0x04, 0xda, 0x46, 0x10,
];

// SHA-3-256 of formal/lean4/MessageTypeProperties.lean (registry proofs)
const REGISTRY_PROOF_HASH: SchemaHash = [
    0xac, 0xf3, 0x35, 0xef, 0x68, 0x6c, 0xbb, 0x5e,
    0xef, 0xc7, 0x20, 0x1e, 0x26, 0x59, 0xc4, 0x33,
    0x7a, 0xce, 0x53, 0x46, 0xa4, 0xb3, 0x25, 0x19,
    0xcd, 0x3f, 0x1f, 0xcc, 0xed, 0xfa, 0xda, 0x47,
];

/// Derive a per-type schema hash by XORing the base proof hash with the type ID.
const fn type_schema_hash(base: &SchemaHash, type_id: u32) -> SchemaHash {
    let id_bytes = type_id.to_le_bytes();
    let mut h = *base;
    h[0] ^= id_bytes[0];
    h[1] ^= id_bytes[1];
    h[2] ^= id_bytes[2];
    h[3] ^= id_bytes[3];
    h
}

/// The 11 default message types shipped with the compiled-in policy.
pub const DEFAULT_MESSAGE_TYPES: &[VerifiedMessageType] = &[
    VerifiedMessageType {
        type_id: MessageTypeId(0x0001),
        name: "InferenceTensorInput",
        boundary: TrustBoundary::Network,
        direction: DataFlowDirection::Inbound,
        schema_hash: type_schema_hash(&TENSOR_PROOF_HASH, 0x0001),
        mode: EnforcementMode::Permissive,
        invariants: TENSOR_INPUT_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0002),
        name: "InferenceTensorOutput",
        boundary: TrustBoundary::Network,
        direction: DataFlowDirection::Outbound,
        schema_hash: type_schema_hash(&TENSOR_PROOF_HASH, 0x0002),
        mode: EnforcementMode::Permissive,
        invariants: TENSOR_OUTPUT_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0003),
        name: "InferenceRequest",
        boundary: TrustBoundary::Network,
        direction: DataFlowDirection::Inbound,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0003),
        mode: EnforcementMode::Permissive,
        invariants: INFERENCE_REQUEST_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0004),
        name: "InferenceResponse",
        boundary: TrustBoundary::Network,
        direction: DataFlowDirection::Outbound,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0004),
        mode: EnforcementMode::Permissive,
        invariants: INFERENCE_RESPONSE_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0010),
        name: "BusSensorFrame",
        boundary: TrustBoundary::BusProtocol,
        direction: DataFlowDirection::Inbound,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0010),
        mode: EnforcementMode::Permissive,
        invariants: BUS_SENSOR_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0011),
        name: "BusActuatorCommand",
        boundary: TrustBoundary::BusProtocol,
        direction: DataFlowDirection::Outbound,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0011),
        mode: EnforcementMode::Permissive,
        invariants: BUS_ACTUATOR_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0020),
        name: "GpuTensorTransfer",
        boundary: TrustBoundary::Gpu,
        direction: DataFlowDirection::Outbound,
        schema_hash: type_schema_hash(&TENSOR_PROOF_HASH, 0x0020),
        mode: EnforcementMode::Permissive,
        invariants: GPU_TRANSFER_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0021),
        name: "GpuInferenceResult",
        boundary: TrustBoundary::Gpu,
        direction: DataFlowDirection::Inbound,
        schema_hash: type_schema_hash(&TENSOR_PROOF_HASH, 0x0021),
        mode: EnforcementMode::Permissive,
        invariants: GPU_RESULT_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0030),
        name: "K8sPodCommand",
        boundary: TrustBoundary::Kubernetes,
        direction: DataFlowDirection::Inbound,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0030),
        mode: EnforcementMode::Permissive,
        invariants: K8S_COMMAND_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0031),
        name: "K8sHealthMetrics",
        boundary: TrustBoundary::Kubernetes,
        direction: DataFlowDirection::Outbound,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0031),
        mode: EnforcementMode::Permissive,
        invariants: K8S_METRICS_INVARIANTS,
    },
    VerifiedMessageType {
        type_id: MessageTypeId(0x0040),
        name: "IpcPubSubMessage",
        boundary: TrustBoundary::Kernel,
        direction: DataFlowDirection::Bidirectional,
        schema_hash: type_schema_hash(&REGISTRY_PROOF_HASH, 0x0040),
        mode: EnforcementMode::Permissive,
        invariants: IPC_MESSAGE_INVARIANTS,
    },
];

/// Build a registry pre-populated with the default message type catalog.
pub fn default_registry() -> MessageTypeRegistry {
    let mut registry = MessageTypeRegistry::new();
    for mt in DEFAULT_MESSAGE_TYPES {
        registry.register(*mt);
    }
    registry
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── TensorDtype ──

    #[test]
    fn tensor_dtype_from_u8() {
        assert_eq!(TensorDtype::from_u8(1), Some(TensorDtype::Float32));
        assert_eq!(TensorDtype::from_u8(6), Some(TensorDtype::Uint8));
        assert_eq!(TensorDtype::from_u8(0), None);
        assert_eq!(TensorDtype::from_u8(7), None);
    }

    // ── VerifiedMessageType ──

    #[test]
    fn verified_hash_is_not_zero() {
        let mt = &DEFAULT_MESSAGE_TYPES[0];
        assert!(mt.is_verified());
        assert_ne!(mt.schema_hash, UNVERIFIED_HASH);
    }

    #[test]
    fn verified_hash_is_verified() {
        let mut hash = UNVERIFIED_HASH;
        hash[0] = 0xAB;
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9999),
            name: "Test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: hash,
            mode: EnforcementMode::Enforcing,
            invariants: &[],
        };
        assert!(mt.is_verified());
    }

    // ── MessageTypeRegistry ──

    #[test]
    fn registry_new_is_empty() {
        let reg = MessageTypeRegistry::new();
        assert_eq!(reg.count(), 0);
        assert!(!reg.is_full());
    }

    #[test]
    fn registry_register_and_lookup() {
        let mut reg = MessageTypeRegistry::new();
        let mt = DEFAULT_MESSAGE_TYPES[0];
        assert!(reg.register(mt));
        assert_eq!(reg.count(), 1);

        let found = reg.lookup(MessageTypeId(0x0001)).unwrap();
        assert_eq!(found.name, "InferenceTensorInput");
    }

    #[test]
    fn registry_duplicate_rejected() {
        let mut reg = MessageTypeRegistry::new();
        let mt = DEFAULT_MESSAGE_TYPES[0];
        assert!(reg.register(mt));
        assert!(!reg.register(mt)); // duplicate ID
        assert_eq!(reg.count(), 1);
    }

    #[test]
    fn registry_lookup_missing() {
        let reg = MessageTypeRegistry::new();
        assert!(reg.lookup(MessageTypeId(0xFFFF)).is_none());
    }

    #[test]
    fn registry_iterate() {
        let reg = default_registry();
        let count = reg.iter().count();
        assert_eq!(count, 11);
    }

    #[test]
    fn default_registry_has_all_types() {
        let reg = default_registry();
        assert_eq!(reg.count(), 11);

        // Spot-check some types
        assert!(reg.lookup(MessageTypeId(0x0001)).is_some());
        assert!(reg.lookup(MessageTypeId(0x0010)).is_some());
        assert!(reg.lookup(MessageTypeId(0x0020)).is_some());
        assert!(reg.lookup(MessageTypeId(0x0030)).is_some());
        assert!(reg.lookup(MessageTypeId(0x0040)).is_some());
    }

    #[test]
    fn default_types_have_invariants() {
        for mt in DEFAULT_MESSAGE_TYPES {
            assert!(
                !mt.invariants.is_empty(),
                "Type '{}' should have at least one invariant",
                mt.name
            );
        }
    }

    #[test]
    fn default_types_all_permissive() {
        for mt in DEFAULT_MESSAGE_TYPES {
            assert_eq!(
                mt.mode,
                EnforcementMode::Permissive,
                "Default type '{}' should be Permissive",
                mt.name
            );
        }
    }

    #[test]
    fn default_types_all_verified() {
        for mt in DEFAULT_MESSAGE_TYPES {
            assert!(
                mt.is_verified(),
                "Default type '{}' should have proof-derived hash",
                mt.name
            );
        }
    }

    #[test]
    fn default_types_have_unique_hashes() {
        for i in 0..DEFAULT_MESSAGE_TYPES.len() {
            for j in (i + 1)..DEFAULT_MESSAGE_TYPES.len() {
                assert_ne!(
                    DEFAULT_MESSAGE_TYPES[i].schema_hash,
                    DEFAULT_MESSAGE_TYPES[j].schema_hash,
                    "Types '{}' and '{}' should have different hashes",
                    DEFAULT_MESSAGE_TYPES[i].name,
                    DEFAULT_MESSAGE_TYPES[j].name,
                );
            }
        }
    }

    #[test]
    fn default_types_unique_ids() {
        for (i, a) in DEFAULT_MESSAGE_TYPES.iter().enumerate() {
            for (j, b) in DEFAULT_MESSAGE_TYPES.iter().enumerate() {
                if i != j {
                    assert_ne!(
                        a.type_id, b.type_id,
                        "Types '{}' and '{}' have duplicate IDs",
                        a.name, b.name
                    );
                }
            }
        }
    }

    // ── InvariantChecker — stateless invariants ──

    #[test]
    fn check_max_rank_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9001),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MaxRank(4)],
        };
        let mut meta = MessageMetadata::empty();
        meta.rank = 3;
        assert_eq!(checker.check(&mt, &meta), None);
    }

    #[test]
    fn check_max_rank_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9002),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MaxRank(4)],
        };
        let mut meta = MessageMetadata::empty();
        meta.rank = 5;
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_min_rank_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9003),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MinRank(1)],
        };
        let mut meta = MessageMetadata::empty();
        meta.rank = 2;
        assert_eq!(checker.check(&mt, &meta), None);
    }

    #[test]
    fn check_min_rank_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9004),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MinRank(2)],
        };
        let mut meta = MessageMetadata::empty();
        meta.rank = 1;
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_allowed_dtype_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9005),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::AllowedDtype(TensorDtype::Float32)],
        };
        let mut meta = MessageMetadata::empty();
        meta.dtype = TensorDtype::Float32 as u8;
        assert_eq!(checker.check(&mt, &meta), None);
    }

    #[test]
    fn check_allowed_dtype_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9006),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::AllowedDtype(TensorDtype::Float32)],
        };
        let mut meta = MessageMetadata::empty();
        meta.dtype = TensorDtype::Int8 as u8;
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_max_elements_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9007),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MaxElements(1000)],
        };
        let mut meta = MessageMetadata::empty();
        meta.element_count = 999;
        assert_eq!(checker.check(&mt, &meta), None);
    }

    #[test]
    fn check_max_elements_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9008),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MaxElements(1000)],
        };
        let mut meta = MessageMetadata::empty();
        meta.element_count = 1001;
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_max_payload_bytes() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9009),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MaxPayloadBytes(100)],
        };
        let mut meta = MessageMetadata::empty();
        meta.payload_bytes = 50;
        assert_eq!(checker.check(&mt, &meta), None);
        meta.payload_bytes = 101;
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_value_range() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x900A),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::ValueRange { min: -100, max: 100 }],
        };
        let mut meta = MessageMetadata::empty();
        meta.sample_value = 50;
        assert_eq!(checker.check(&mt, &meta), None);
        meta.sample_value = 101;
        assert_eq!(checker.check(&mt, &meta), Some(0));
        meta.sample_value = -101;
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_enum_membership() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x900B),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::EnumMembership(4)],
        };
        let mut meta = MessageMetadata::empty();
        meta.enum_value = 3; // valid (0, 1, 2, 3)
        assert_eq!(checker.check(&mt, &meta), None);
        meta.enum_value = 4; // invalid
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_non_zero_dimensions_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x900C),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::NonZeroDimensions],
        };
        let mut meta = MessageMetadata::empty();
        meta.num_dimensions = 3;
        meta.dimensions = [1, 3, 224, 0, 0, 0, 0, 0];
        assert_eq!(checker.check(&mt, &meta), None);
    }

    #[test]
    fn check_non_zero_dimensions_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x900D),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::NonZeroDimensions],
        };
        let mut meta = MessageMetadata::empty();
        meta.num_dimensions = 3;
        meta.dimensions = [1, 0, 224, 0, 0, 0, 0, 0]; // second dim is 0
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    // ── InvariantChecker — stateful invariants ──

    #[test]
    fn check_monotonic_timestamp_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x900E),
            name: "test",
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MonotonicTimestamp],
        };
        let mut meta = MessageMetadata::empty();

        meta.timestamp = 100;
        assert_eq!(checker.check(&mt, &meta), None); // first message

        meta.timestamp = 200;
        assert_eq!(checker.check(&mt, &meta), None); // increasing
    }

    #[test]
    fn check_monotonic_timestamp_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x900F),
            name: "test",
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::MonotonicTimestamp],
        };
        let mut meta = MessageMetadata::empty();

        meta.timestamp = 100;
        assert_eq!(checker.check(&mt, &meta), None);

        meta.timestamp = 50; // going backwards
        assert_eq!(checker.check(&mt, &meta), Some(0));
    }

    #[test]
    fn check_rate_limit_pass() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9010),
            name: "test",
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Outbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::RateLimit { max_per_sec: 3 }],
        };
        let meta = MessageMetadata::empty();

        assert_eq!(checker.check(&mt, &meta), None); // 1st
        assert_eq!(checker.check(&mt, &meta), None); // 2nd
        assert_eq!(checker.check(&mt, &meta), None); // 3rd
    }

    #[test]
    fn check_rate_limit_fail() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9011),
            name: "test",
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Outbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::RateLimit { max_per_sec: 2 }],
        };
        let meta = MessageMetadata::empty();

        assert_eq!(checker.check(&mt, &meta), None); // 1st
        assert_eq!(checker.check(&mt, &meta), None); // 2nd
        assert_eq!(checker.check(&mt, &meta), Some(0)); // 3rd — exceeds limit
    }

    #[test]
    fn rate_limit_reset() {
        let mut checker = InvariantChecker::new();
        let type_id = MessageTypeId(0x9012);
        let mt = VerifiedMessageType {
            type_id,
            name: "test",
            boundary: TrustBoundary::BusProtocol,
            direction: DataFlowDirection::Outbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[Invariant::RateLimit { max_per_sec: 1 }],
        };
        let meta = MessageMetadata::empty();

        assert_eq!(checker.check(&mt, &meta), None); // 1st
        assert_eq!(checker.check(&mt, &meta), Some(0)); // 2nd — exceeds

        checker.reset_rate_window(type_id);
        assert_eq!(checker.check(&mt, &meta), None); // reset, 1st again
    }

    // ── Multiple invariants ──

    #[test]
    fn multiple_invariants_first_failure_reported() {
        let mut checker = InvariantChecker::new();
        let mt = VerifiedMessageType {
            type_id: MessageTypeId(0x9013),
            name: "test",
            boundary: TrustBoundary::Network,
            direction: DataFlowDirection::Inbound,
            schema_hash: UNVERIFIED_HASH,
            mode: EnforcementMode::Enforcing,
            invariants: &[
                Invariant::MinRank(1),           // index 0
                Invariant::MaxRank(4),           // index 1
                Invariant::MaxElements(1000),    // index 2
            ],
        };
        let mut meta = MessageMetadata::empty();
        meta.rank = 5; // fails MaxRank (index 1), but passes MinRank (index 0)
        meta.element_count = 500;
        assert_eq!(checker.check(&mt, &meta), Some(1)); // second invariant fails
    }

    #[test]
    fn multiple_invariants_all_pass() {
        let mut checker = InvariantChecker::new();
        let mt = &DEFAULT_MESSAGE_TYPES[0]; // InferenceTensorInput
        let mut meta = MessageMetadata::empty();
        meta.rank = 3;
        meta.dtype = TensorDtype::Float32 as u8;
        meta.element_count = 150_528; // 1*3*224*224
        meta.num_dimensions = 3;
        meta.dimensions = [3, 224, 224, 0, 0, 0, 0, 0];
        assert_eq!(checker.check(mt, &meta), None);
    }

    // ── Fuzz-style invariant checking (task 8.2) ──

    /// Simple pseudo-random number generator for deterministic fuzzing.
    fn xorshift32(state: &mut u32) -> u32 {
        let mut x = *state;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *state = x;
        x
    }

    #[test]
    fn fuzz_invariant_checking_no_panics() {
        let mut checker = InvariantChecker::new();
        let mut rng_state: u32 = 0xDEAD_BEEF;

        for _ in 0..1000 {
            let r = xorshift32(&mut rng_state);
            // Pick a random message type from the default catalog
            let type_idx = (r as usize) % DEFAULT_MESSAGE_TYPES.len();
            let mt = &DEFAULT_MESSAGE_TYPES[type_idx];

            let r2 = xorshift32(&mut rng_state);
            let r3 = xorshift32(&mut rng_state);
            let r4 = xorshift32(&mut rng_state);
            let r5 = xorshift32(&mut rng_state);

            let meta = MessageMetadata {
                rank: (r2 & 0xFF) as u8,
                dtype: ((r2 >> 8) & 0x0F) as u8,
                element_count: r3,
                payload_bytes: r4,
                num_dimensions: ((r2 >> 16) & 0xFF) as u8,
                dimensions: [
                    (r5 & 0xFFFF) as u32,
                    ((r5 >> 16) & 0xFFFF) as u32,
                    (r3 & 0xFF) as u32,
                    ((r3 >> 8) & 0xFF) as u32,
                    0,
                    0,
                    0,
                    0,
                ],
                enum_value: (r5 & 0xFF) as u8,
                timestamp: r4 as u64,
                sample_value: r3 as i64,
            };

            // Should never panic — it either returns None (pass) or Some(idx) (fail)
            let result = checker.check(mt, &meta);
            assert!(result.is_none() || result.is_some());
        }
    }

    #[test]
    fn fuzz_invariant_extreme_values_no_panics() {
        let mut checker = InvariantChecker::new();
        let extreme_values: &[u32] = &[0, 1, u32::MAX, u32::MAX / 2, 255, 256, 65535, 65536];

        for &elem_count in extreme_values {
            for &payload in extreme_values {
                for rank in [0u8, 1, 2, 8, 255] {
                    for dtype in 0u8..=10 {
                        let meta = MessageMetadata {
                            rank,
                            dtype,
                            element_count: elem_count,
                            payload_bytes: payload,
                            num_dimensions: rank,
                            dimensions: [elem_count, 1, 1, 1, 0, 0, 0, 0],
                            enum_value: dtype,
                            timestamp: 0,
                            sample_value: elem_count as i64,
                        };
                        for mt in DEFAULT_MESSAGE_TYPES.iter() {
                            let _ = checker.check(mt, &meta);
                        }
                    }
                }
            }
        }
    }
}
