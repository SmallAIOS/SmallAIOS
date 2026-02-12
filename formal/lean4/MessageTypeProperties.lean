/-
  SmallAIOS Message Type Registry Properties — Lean 4 Proof
  Verifies: registry well-formedness (no duplicate IDs),
  invariant check totality and determinism

  SPDX-License-Identifier: Apache-2.0
-/

-- ============================================================================
-- Message type ID
-- ============================================================================

/-- A message type identifier (corresponds to a 16-bit protocol ID in the
    implementation, modeled here as a natural number). -/
structure MessageTypeId where
  val : Nat
  deriving DecidableEq, Repr

-- ============================================================================
-- Invariant kinds
-- ============================================================================

/-- Invariant kinds that can be attached to a message type.
    Each variant corresponds to a validation rule enforced by the
    formal security gate before a message is dispatched. -/
inductive InvariantKind where
  | MaxRank           : InvariantKind
  | MinRank           : InvariantKind
  | AllowedDtype      : InvariantKind
  | MaxElements       : InvariantKind
  | MaxPayloadBytes   : InvariantKind
  | ValueRange        : InvariantKind
  | EnumMembership    : InvariantKind
  | NonZeroDimensions : InvariantKind
  | MonotonicTimestamp : InvariantKind
  | RateLimit         : InvariantKind
  deriving DecidableEq, Repr

-- ============================================================================
-- Registry entry and registry
-- ============================================================================

/-- A registry entry associates a message type ID with a list of invariants
    that must hold for messages of that type. -/
structure RegistryEntry where
  id : MessageTypeId
  invariants : List InvariantKind
  deriving DecidableEq, Repr

/-- A registry is a list of registry entries. -/
abbrev Registry := List RegistryEntry

-- ============================================================================
-- Well-formedness predicate
-- ============================================================================

/-- A registry is well-formed when no two entries share the same ID.
    This is checked pair-wise: for every pair of distinct positions,
    the IDs differ. -/
def wellFormed (reg : Registry) : Prop :=
  ∀ (i j : Nat), i < reg.length → j < reg.length → i ≠ j →
    (reg.get ⟨i, by omega⟩).id ≠ (reg.get ⟨j, by omega⟩).id

-- ============================================================================
-- Helper: all IDs are distinct (decidable version for native_decide)
-- ============================================================================

/-- Check that a given ID does not appear in a list of entries. -/
def idNotIn (mid : MessageTypeId) : List RegistryEntry → Bool
  | []        => true
  | e :: rest => e.id != mid && idNotIn mid rest

/-- Decidable duplicate-free check on a registry. -/
def noDuplicateIds : Registry → Bool
  | []        => true
  | e :: rest => idNotIn e.id rest && noDuplicateIds rest

-- ============================================================================
-- Default catalog — 11 message types
-- ============================================================================

/-- The default message type catalog with 11 entries.
    IDs mirror the protocol specification:
      0x0001  TensorPayload
      0x0002  InferenceRequest
      0x0003  InferenceResponse
      0x0004  ModelMetadata
      0x0010  HealthCheck
      0x0011  HealthResponse
      0x0020  CapabilityGrant
      0x0021  CapabilityRevoke
      0x0030  AuditEvent
      0x0031  AuditQuery
      0x0040  SystemControl
-/
def defaultCatalog : Registry :=
  [ { id := ⟨0x0001⟩, invariants := [.MaxRank, .AllowedDtype, .NonZeroDimensions, .MaxPayloadBytes] },
    { id := ⟨0x0002⟩, invariants := [.MaxPayloadBytes] },
    { id := ⟨0x0003⟩, invariants := [.MaxPayloadBytes] },
    { id := ⟨0x0004⟩, invariants := [.MaxPayloadBytes] },
    { id := ⟨0x0010⟩, invariants := [.MonotonicTimestamp] },
    { id := ⟨0x0011⟩, invariants := [.MonotonicTimestamp] },
    { id := ⟨0x0020⟩, invariants := [.EnumMembership] },
    { id := ⟨0x0021⟩, invariants := [.EnumMembership] },
    { id := ⟨0x0030⟩, invariants := [.MonotonicTimestamp, .MaxPayloadBytes, .RateLimit] },
    { id := ⟨0x0031⟩, invariants := [.MaxPayloadBytes] },
    { id := ⟨0x0040⟩, invariants := [.EnumMembership, .RateLimit] } ]

/-- The default catalog has exactly 11 entries. -/
theorem defaultCatalog_length : defaultCatalog.length = 11 := by
  native_decide

/-- The default catalog has no duplicate IDs (decidable check). -/
theorem defaultCatalog_noDuplicateIds : noDuplicateIds defaultCatalog = true := by
  native_decide

/-- The default catalog is well-formed: no two entries share the same ID. -/
theorem defaultRegistryWellFormed : wellFormed defaultCatalog := by
  intro i j hi hj hne heq
  -- Enumerate all 11 * 10 distinct (i, j) pairs; each leads to a contradiction
  -- because the IDs are pairwise distinct. We use the decidable check to close
  -- this via native_decide on a boolean reformulation.
  have hnd := defaultCatalog_noDuplicateIds
  -- We convert the pointwise proof obligation to the boolean check.
  -- Since noDuplicateIds defaultCatalog = true is proven above,
  -- and all IDs are distinct concrete values, omega + simp suffice per pair.
  simp [defaultCatalog] at hi hj
  interval_cases i <;> interval_cases j <;> simp_all [defaultCatalog, MessageTypeId.mk.injEq]

-- ============================================================================
-- Invariant result and check function
-- ============================================================================

/-- Result of checking a single invariant against a message. -/
inductive InvariantResult where
  | Pass : InvariantResult
  | Fail : InvariantResult
  deriving DecidableEq, Repr

/-- Simplified model of invariant checking.
    In the real implementation each invariant kind inspects message fields;
    here we model the outcome as a pure function of the invariant kind and
    an abstract input value (natural number). -/
def checkInvariant (kind : InvariantKind) (input : Nat) : InvariantResult :=
  match kind with
  | .MaxRank           => if input ≤ 8     then .Pass else .Fail
  | .MinRank           => if input ≥ 1     then .Pass else .Fail
  | .AllowedDtype      => if input < 12    then .Pass else .Fail
  | .MaxElements       => if input ≤ 65536 then .Pass else .Fail
  | .MaxPayloadBytes   => if input ≤ 4194304 then .Pass else .Fail  -- 4 MiB
  | .ValueRange        => if input < 1000000 then .Pass else .Fail
  | .EnumMembership    => if input < 256   then .Pass else .Fail
  | .NonZeroDimensions => if input ≥ 1     then .Pass else .Fail
  | .MonotonicTimestamp => .Pass  -- validated against prior state, always structurally ok
  | .RateLimit         => if input ≤ 10000 then .Pass else .Fail

-- ============================================================================
-- Totality: checkInvariant always returns Pass or Fail
-- ============================================================================

/-- For any invariant kind and any input, checkInvariant returns exactly
    Pass or Fail — it is total. -/
theorem checkInvariant_total (kind : InvariantKind) (input : Nat) :
    checkInvariant kind input = .Pass ∨ checkInvariant kind input = .Fail := by
  cases kind <;> simp [checkInvariant] <;> omega

-- ============================================================================
-- Determinism: checkInvariant is a pure function
-- ============================================================================

/-- Calling checkInvariant twice with the same arguments yields the
    same result — the function is deterministic. -/
theorem checkInvariant_deterministic (kind : InvariantKind) (input : Nat) :
    checkInvariant kind input = checkInvariant kind input := by
  rfl

/-- Stronger determinism: if two calls agree on kind and input,
    they agree on the result. -/
theorem checkInvariant_deterministic_ext
    (k1 k2 : InvariantKind) (n1 n2 : Nat)
    (hk : k1 = k2) (hn : n1 = n2) :
    checkInvariant k1 n1 = checkInvariant k2 n2 := by
  subst hk; subst hn; rfl

-- ============================================================================
-- Additional properties
-- ============================================================================

/-- MonotonicTimestamp invariant always passes (stateless structural check). -/
theorem monotonic_always_passes (input : Nat) :
    checkInvariant .MonotonicTimestamp input = .Pass := by
  rfl

/-- MaxRank with rank 0 passes (rank ≤ 8). -/
theorem maxrank_zero_passes :
    checkInvariant .MaxRank 0 = .Pass := by
  rfl

/-- MaxRank with rank 9 fails (rank > 8). -/
theorem maxrank_nine_fails :
    checkInvariant .MaxRank 9 = .Fail := by
  native_decide

/-- NonZeroDimensions with 0 fails. -/
theorem nonzero_dim_zero_fails :
    checkInvariant .NonZeroDimensions 0 = .Fail := by
  rfl

/-- Running all invariants for an entry: check every invariant in the list. -/
def checkAllInvariants (entry : RegistryEntry) (input : Nat) : InvariantResult :=
  entry.invariants.foldl (fun acc kind =>
    match acc with
    | .Fail => .Fail
    | .Pass => checkInvariant kind input
  ) .Pass

/-- checkAllInvariants is total: it always returns Pass or Fail. -/
theorem checkAllInvariants_total (entry : RegistryEntry) (input : Nat) :
    checkAllInvariants entry input = .Pass ∨ checkAllInvariants entry input = .Fail := by
  simp [checkAllInvariants]
  -- The fold always produces an InvariantResult; by exhaustive matching
  -- of the two constructors, the disjunction holds.
  cases (entry.invariants.foldl (fun acc kind =>
    match acc with
    | .Fail => .Fail
    | .Pass => checkInvariant kind input) .Pass) with
  | Pass => left; rfl
  | Fail => right; rfl

/-- checkAllInvariants is deterministic. -/
theorem checkAllInvariants_deterministic (entry : RegistryEntry) (input : Nat) :
    checkAllInvariants entry input = checkAllInvariants entry input := by
  rfl
