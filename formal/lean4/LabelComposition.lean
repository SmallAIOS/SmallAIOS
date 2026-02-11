/-
  SmallAIOS Security Label Composition — Lean 4 Proof
  Verifies: SecurityLabel comparison composes correctly with
  ClassificationLevel and IntegrityLevel orderings

  The Bell-LaPadula model enforces "no read up" (simple security property)
  and the Biba model enforces integrity ordering. This proof demonstrates:
  1. ClassificationLevel and IntegrityLevel each form total orders
  2. The composed SecurityLabel dominance relation is reflexive, transitive,
     and antisymmetric (a partial order)
  3. Bell-LaPadula no-read-up follows from dominance failure
  4. Biba integrity composition holds when classification is equal

  SPDX-License-Identifier: Apache-2.0
-/

-- ============================================================================
-- Classification Level (ascending confidentiality)
-- ============================================================================

/-- Classification levels in SmallAIOS, ordered by ascending confidentiality.
    Public < Internal < Confidential < Restricted -/
inductive ClassificationLevel where
  | Public       : ClassificationLevel
  | Internal     : ClassificationLevel
  | Confidential : ClassificationLevel
  | Restricted   : ClassificationLevel
  deriving DecidableEq, Repr

/-- Numeric encoding of classification levels for ordering comparison. -/
def ClassificationLevel.toNat : ClassificationLevel → Nat
  | .Public       => 0
  | .Internal     => 1
  | .Confidential => 2
  | .Restricted   => 3

/-- Classification ordering: a ≤ b iff a's confidentiality rank ≤ b's rank. -/
def classLe (a b : ClassificationLevel) : Bool :=
  a.toNat ≤ b.toNat

instance : DecidableEq ClassificationLevel := ClassificationLevel.decEq

-- ============================================================================
-- Integrity Level
-- ============================================================================

/-- Integrity levels in SmallAIOS: Low < Medium < High. -/
inductive IntegrityLevel where
  | Low    : IntegrityLevel
  | Medium : IntegrityLevel
  | High   : IntegrityLevel
  deriving DecidableEq, Repr

/-- Numeric encoding of integrity levels for ordering comparison. -/
def IntegrityLevel.toNat : IntegrityLevel → Nat
  | .Low    => 0
  | .Medium => 1
  | .High   => 2

/-- Integrity ordering: a ≤ b iff a's integrity rank ≤ b's rank. -/
def intLe (a b : IntegrityLevel) : Bool :=
  a.toNat ≤ b.toNat

instance : DecidableEq IntegrityLevel := IntegrityLevel.decEq

-- ============================================================================
-- Security Label (composition of classification and integrity)
-- ============================================================================

/-- A security label combines a classification level with an integrity level.
    This is the core lattice element for Bell-LaPadula / Biba enforcement. -/
structure SecurityLabel where
  classification : ClassificationLevel
  integrity      : IntegrityLevel
  deriving DecidableEq, Repr

/-- Label dominance: label A dominates label B iff A's classification is at
    least as high as B's AND A's integrity is at least as high as B's.
    This implements the Bell-LaPadula "read-down" check: a subject at label A
    may read an object at label B only if A dominates B. -/
def labelDominates (a b : SecurityLabel) : Bool :=
  classLe b.classification a.classification &&
  intLe b.integrity a.integrity

-- ============================================================================
-- Proofs: Sub-ordering properties
-- ============================================================================

/-- classLe is reflexive. -/
theorem classLe_refl (c : ClassificationLevel) : classLe c c = true := by
  cases c <;> native_decide

/-- intLe is reflexive. -/
theorem intLe_refl (i : IntegrityLevel) : intLe i i = true := by
  cases i <;> native_decide

/-- classLe is transitive. -/
theorem classLe_trans (a b c : ClassificationLevel) :
    classLe a b = true → classLe b c = true → classLe a c = true := by
  cases a <;> cases b <;> cases c <;> simp [classLe, ClassificationLevel.toNat] <;> omega

/-- intLe is transitive. -/
theorem intLe_trans (a b c : IntegrityLevel) :
    intLe a b = true → intLe b c = true → intLe a c = true := by
  cases a <;> cases b <;> cases c <;> simp [intLe, IntegrityLevel.toNat] <;> omega

/-- classLe is antisymmetric: if a ≤ b and b ≤ a then a = b. -/
theorem classLe_antisymm (a b : ClassificationLevel) :
    classLe a b = true → classLe b a = true → a = b := by
  cases a <;> cases b <;> simp [classLe, ClassificationLevel.toNat] <;> omega

/-- intLe is antisymmetric: if a ≤ b and b ≤ a then a = b. -/
theorem intLe_antisymm (a b : IntegrityLevel) :
    intLe a b = true → intLe b a = true → a = b := by
  cases a <;> cases b <;> simp [intLe, IntegrityLevel.toNat] <;> omega

/-- classLe is total: for any two classification levels, a ≤ b or b ≤ a. -/
theorem classLe_total (a b : ClassificationLevel) :
    classLe a b = true ∨ classLe b a = true := by
  cases a <;> cases b <;> simp [classLe, ClassificationLevel.toNat] <;> omega

/-- intLe is total: for any two integrity levels, a ≤ b or b ≤ a. -/
theorem intLe_total (a b : IntegrityLevel) :
    intLe a b = true ∨ intLe b a = true := by
  cases a <;> cases b <;> simp [intLe, IntegrityLevel.toNat] <;> omega

-- ============================================================================
-- Proofs: Label dominance is a partial order
-- ============================================================================

/-- labelDominates is reflexive: every label dominates itself. -/
theorem labelDominates_reflexive (l : SecurityLabel) :
    labelDominates l l = true := by
  simp [labelDominates, classLe_refl, intLe_refl]

/-- labelDominates is transitive: if A dominates B and B dominates C,
    then A dominates C. -/
theorem labelDominates_transitive (a b c : SecurityLabel) :
    labelDominates a b = true → labelDominates b c = true →
    labelDominates a c = true := by
  simp [labelDominates, Bool.and_eq_true]
  intro hab_c hab_i hbc_c hbc_i
  exact ⟨classLe_trans c.classification b.classification a.classification hbc_c hab_c,
         intLe_trans c.integrity b.integrity a.integrity hbc_i hab_i⟩

/-- labelDominates is antisymmetric: if A dominates B and B dominates A,
    then A = B. -/
theorem labelDominates_antisymmetric (a b : SecurityLabel) :
    labelDominates a b = true → labelDominates b a = true → a = b := by
  simp [labelDominates, Bool.and_eq_true]
  intro hab_c hab_i hba_c hba_i
  have hc := classLe_antisymm a.classification b.classification hba_c hab_c
  have hi := intLe_antisymm a.integrity b.integrity hba_i hab_i
  cases a; cases b; simp_all

-- ============================================================================
-- Proofs: Composition — sub-orderings compose into dominance
-- ============================================================================

/-- Composition theorem: if classLe a.classification b.classification AND
    intLe a.integrity b.integrity then labelDominates b a.
    That is, component-wise ordering on (class, integrity) yields dominance. -/
theorem labelDominates_composition (a b : SecurityLabel) :
    classLe a.classification b.classification = true →
    intLe a.integrity b.integrity = true →
    labelDominates b a = true := by
  intro hc hi
  simp [labelDominates, hc, hi]

-- ============================================================================
-- Proofs: Bell-LaPadula no-read-up
-- ============================================================================

/-- Access decision based on Bell-LaPadula simple security property. -/
def bellLaPadulaRead (reader writer : SecurityLabel) : Bool :=
  labelDominates reader writer

/-- Bell-LaPadula no-read-up: if the reader does NOT dominate the writer's
    label, the read is denied. A subject cannot read an object at a
    higher security label. -/
theorem bellLaPadula_no_read_up (reader writer : SecurityLabel) :
    labelDominates reader writer = false →
    bellLaPadulaRead reader writer = false := by
  intro h
  simp [bellLaPadulaRead, h]

/-- Concrete example: a Public/Low reader cannot read a Restricted/High object. -/
theorem no_read_up_example :
    bellLaPadulaRead
      { classification := .Public, integrity := .Low }
      { classification := .Restricted, integrity := .High } = false := by
  native_decide

-- ============================================================================
-- Proofs: Biba integrity composition
-- ============================================================================

/-- Biba composition: given equal classification, a label with higher (or equal)
    integrity dominates one with lower integrity. -/
theorem biba_integrity_dominance (c : ClassificationLevel) (i1 i2 : IntegrityLevel) :
    intLe i1 i2 = true →
    labelDominates
      { classification := c, integrity := i2 }
      { classification := c, integrity := i1 } = true := by
  intro hi
  simp [labelDominates, classLe_refl, hi]

/-- Concrete example: High integrity dominates Low integrity at the same
    classification level. -/
theorem biba_high_dominates_low :
    labelDominates
      { classification := .Internal, integrity := .High }
      { classification := .Internal, integrity := .Low } = true := by
  native_decide

/-- Biba integrity ordering is strict when levels differ: Medium does not
    dominate High (at equal classification). -/
theorem biba_medium_not_dominates_high :
    labelDominates
      { classification := .Confidential, integrity := .Medium }
      { classification := .Confidential, integrity := .High } = false := by
  native_decide

-- ============================================================================
-- Proofs: Totality of sub-orderings (independently)
-- ============================================================================

/-- Classification ordering is total: any two levels are comparable. -/
theorem classLe_totality :
    ∀ (a b : ClassificationLevel), classLe a b = true ∨ classLe b a = true := by
  intro a b
  exact classLe_total a b

/-- Integrity ordering is total: any two levels are comparable. -/
theorem intLe_totality :
    ∀ (a b : IntegrityLevel), intLe a b = true ∨ intLe b a = true := by
  intro a b
  exact intLe_total a b

/-- Note: labelDominates is NOT total — it is a partial order.
    Counter-example: (Public, High) and (Restricted, Low) are incomparable. -/
theorem labelDominates_not_total :
    labelDominates
      { classification := .Public, integrity := .High }
      { classification := .Restricted, integrity := .Low } = false ∧
    labelDominates
      { classification := .Restricted, integrity := .Low }
      { classification := .Public, integrity := .High } = false := by
  constructor <;> native_decide
