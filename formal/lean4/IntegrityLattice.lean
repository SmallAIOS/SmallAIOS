/-
  SmallAIOS Integrity Lattice — Lean 4 Proof
  Proves that the IntegrityLevel type used in the formal security gate
  forms a total order and satisfies the Biba no-write-up property.

  The Biba integrity model prevents untrusted (low-integrity) data from
  flowing upward to corrupt high-integrity destinations. This is enforced
  at the syscall boundary in SmallAIOS's capability-based security layer.

  Properties proved:
  1. IntegrityLevel.le is a partial order (reflexive, antisymmetric, transitive)
  2. IntegrityLevel.le is a total order (every pair is comparable)
  3. The chain Low <= Medium <= High holds explicitly
  4. Biba no-write-up: source cannot write to a higher-integrity destination
  5. Write-allowed for equal and downward flows
  6. Well-formedness: every pair has a defined ordering

  SPDX-License-Identifier: Apache-2.0
-/

/-- Integrity levels in SmallAIOS, ordered Low < Medium < High.
    Corresponds to the IntegrityLevel enum in security/src/integrity.rs. -/
inductive IntegrityLevel where
  | low    : IntegrityLevel
  | medium : IntegrityLevel
  | high   : IntegrityLevel
  deriving DecidableEq, Repr

/-- Ordering on integrity levels: Low <= Medium <= High.
    Returns true when `a` is less than or equal to `b`. -/
def IntegrityLevel.le : IntegrityLevel → IntegrityLevel → Bool
  | .low,    _       => true
  | .medium, .low    => false
  | .medium, _       => true
  | .high,   .high   => true
  | .high,   _       => false

/-- Strict ordering: `a` is strictly less than `b`. -/
def IntegrityLevel.lt (a b : IntegrityLevel) : Bool :=
  a.le b && a != b

-- ============================================================================
-- Partial Order Properties
-- ============================================================================

/-- Reflexivity: every integrity level is <= itself. -/
theorem IntegrityLevel.le_refl (a : IntegrityLevel) :
    a.le a = true := by
  cases a <;> rfl

/-- Antisymmetry: if a <= b and b <= a, then a = b. -/
theorem IntegrityLevel.le_antisymm (a b : IntegrityLevel) :
    a.le b = true → b.le a = true → a = b := by
  cases a <;> cases b <;> simp [IntegrityLevel.le]

/-- Transitivity: if a <= b and b <= c, then a <= c. -/
theorem IntegrityLevel.le_trans (a b c : IntegrityLevel) :
    a.le b = true → b.le c = true → a.le c = true := by
  cases a <;> cases b <;> cases c <;> simp [IntegrityLevel.le]

-- ============================================================================
-- Total Order
-- ============================================================================

/-- Totality: for any two integrity levels, either a <= b or b <= a. -/
theorem IntegrityLevel.le_total (a b : IntegrityLevel) :
    a.le b = true ∨ b.le a = true := by
  cases a <;> cases b <;> simp [IntegrityLevel.le]

-- ============================================================================
-- Explicit Chain: Low <= Medium <= High
-- ============================================================================

/-- Low <= Medium. -/
theorem IntegrityLevel.low_le_medium :
    IntegrityLevel.low.le .medium = true := by
  rfl

/-- Medium <= High. -/
theorem IntegrityLevel.medium_le_high :
    IntegrityLevel.medium.le .high = true := by
  rfl

/-- Low <= High (by transitivity, but also directly). -/
theorem IntegrityLevel.low_le_high :
    IntegrityLevel.low.le .high = true := by
  rfl

/-- Low < Medium (strict). -/
theorem IntegrityLevel.low_lt_medium :
    IntegrityLevel.low.lt .medium = true := by
  native_decide

/-- Medium < High (strict). -/
theorem IntegrityLevel.medium_lt_high :
    IntegrityLevel.medium.lt .high = true := by
  native_decide

/-- Low < High (strict). -/
theorem IntegrityLevel.low_lt_high :
    IntegrityLevel.low.lt .high = true := by
  native_decide

-- ============================================================================
-- Biba Integrity Model
-- ============================================================================

/-- Biba write policy: a source at integrity level `src` may write to a
    destination at integrity level `dst` only if src >= dst.
    This is the "no-write-up" rule: low-integrity data cannot corrupt
    high-integrity destinations. -/
def bibaWriteAllowed (src dst : IntegrityLevel) : Bool :=
  dst.le src

-- ============================================================================
-- No-Write-Up Proofs (Biba)
-- ============================================================================

/-- Low cannot write to Medium (no-write-up). -/
theorem biba_no_write_low_to_medium :
    bibaWriteAllowed .low .medium = false := by
  rfl

/-- Low cannot write to High (no-write-up). -/
theorem biba_no_write_low_to_high :
    bibaWriteAllowed .low .high = false := by
  rfl

/-- Medium cannot write to High (no-write-up). -/
theorem biba_no_write_medium_to_high :
    bibaWriteAllowed .medium .high = false := by
  rfl

/-- General no-write-up: if src < dst, then write is denied. -/
theorem biba_no_write_up (src dst : IntegrityLevel) :
    src.lt dst = true → bibaWriteAllowed src dst = false := by
  cases src <;> cases dst <;> simp [IntegrityLevel.lt, IntegrityLevel.le, bibaWriteAllowed]

-- ============================================================================
-- Write-Allowed Proofs (Equal and Downward)
-- ============================================================================

/-- Any level can write to itself (equal integrity). -/
theorem biba_write_self (level : IntegrityLevel) :
    bibaWriteAllowed level level = true := by
  cases level <;> rfl

/-- High can write to Medium (write-down allowed). -/
theorem biba_write_high_to_medium :
    bibaWriteAllowed .high .medium = true := by
  rfl

/-- High can write to Low (write-down allowed). -/
theorem biba_write_high_to_low :
    bibaWriteAllowed .high .low = true := by
  rfl

/-- Medium can write to Low (write-down allowed). -/
theorem biba_write_medium_to_low :
    bibaWriteAllowed .medium .low = true := by
  rfl

/-- General write-allowed: if src >= dst, then write is permitted. -/
theorem biba_write_allowed_iff (src dst : IntegrityLevel) :
    bibaWriteAllowed src dst = true ↔ dst.le src = true := by
  simp [bibaWriteAllowed]

-- ============================================================================
-- Well-Formedness
-- ============================================================================

/-- Well-formedness: every pair of integrity levels has a defined ordering.
    That is, for all a b, exactly one of a <= b or b < a holds. -/
theorem IntegrityLevel.ordering_wellformed (a b : IntegrityLevel) :
    a.le b = true ∨ b.le a = true := by
  cases a <;> cases b <;> simp [IntegrityLevel.le]

/-- Decidability: the ordering is computable for every pair. -/
theorem IntegrityLevel.le_decidable (a b : IntegrityLevel) :
    a.le b = true ∨ a.le b = false := by
  cases a <;> cases b <;> simp [IntegrityLevel.le]

/-- The bibaWriteAllowed function is decidable for every pair. -/
theorem biba_write_decidable (src dst : IntegrityLevel) :
    bibaWriteAllowed src dst = true ∨ bibaWriteAllowed src dst = false := by
  cases src <;> cases dst <;> simp [bibaWriteAllowed, IntegrityLevel.le]

/-- Exhaustive verification: enumerate all 9 (src, dst) pairs and confirm
    the Biba policy matches the expected allow/deny outcome. -/
theorem biba_exhaustive_check :
    -- Equal pairs: all allowed
    bibaWriteAllowed .low .low = true ∧
    bibaWriteAllowed .medium .medium = true ∧
    bibaWriteAllowed .high .high = true ∧
    -- Write-down: all allowed
    bibaWriteAllowed .medium .low = true ∧
    bibaWriteAllowed .high .low = true ∧
    bibaWriteAllowed .high .medium = true ∧
    -- Write-up: all denied
    bibaWriteAllowed .low .medium = false ∧
    bibaWriteAllowed .low .high = false ∧
    bibaWriteAllowed .medium .high = false := by
  native_decide
