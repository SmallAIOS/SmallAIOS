## ADDED Requirements

### Requirement: Crypto Primitives Replay an Official Test Corpus

Every cryptographic primitive in the `security` crate SHALL be
validated by replaying an official public test corpus —
Wycheproof/C2SP, NIST CAVP/ACVP, or the test vectors of the defining
RFC/FIPS document — as part of the crate's standard `cargo test` run.
The corpus SHALL be checked into the repository (generated modules or
fixture files), every vector SHALL be executed (no sampling), and a
failing vector SHALL identify itself (test-case id or vector index)
in the failure output. New primitive changes SHALL name their corpus
in the change proposal before implementation starts.

#### Scenario: Existing primitives carry their corpus in-tree

- **WHEN** a reviewer inspects `security/src/` for the shipped
  primitives (SHA-2/SHA-3, ChaCha20-Poly1305, AES-256-GCM, Ed25519,
  X25519, ML-KEM-768, ML-DSA-65)
- **THEN** each SHALL have vector-replay tests sourced from its
  defining standard or an official corpus
- **AND** the vectors SHALL execute in `cargo test -p smallaios-security`
  without network access

#### Scenario: A new primitive without a named corpus is rejected in review

- **WHEN** a change proposal adds a crypto primitive to `security/`
  without naming the official corpus it will replay
- **THEN** the proposal SHALL be considered incomplete against this
  policy
- **AND** the corpus SHALL be named before implementation tasks begin

#### Scenario: Corpus failure output identifies the vector

- **WHEN** any replayed vector produces the wrong outcome
- **THEN** the test failure output SHALL include the corpus's
  identifier for that vector (e.g. Wycheproof `tc_id`, CAVP case
  number)

### Requirement: No C or C++ Crypto Libraries in the Workspace

The workspace SHALL NOT link C or C++ cryptographic libraries —
including wolfSSL/wolfCrypt, OpenSSL, BoringSSL, mbedTLS, and
libsodium — in any crate, at any layer. Layer-0 crypto SHALL remain
clean-room `#![no_std]` Rust. This rule SHALL be enforced
mechanically: `deny.toml` SHALL carry `[bans]` entries for the known
binding/sys crates (at minimum `openssl-sys`, `openssl`,
`wolfssl-sys`, `wolfssl`, `boring-sys`, `boring`, `mbedtls-sys-auto`,
`mbedtls`, `libsodium-sys`, `sodiumoxide`), so the existing
Supply Chain Security CI gate fails any PR introducing one.

#### Scenario: Adding a banned crypto crate fails the supply-chain gate

- **WHEN** a PR adds `wolfssl-sys` (or any other banned C-crypto
  binding crate) to any workspace `Cargo.toml`
- **THEN** `cargo deny check bans` SHALL fail
- **AND** the Supply Chain Security CI gate SHALL block the PR

#### Scenario: Ban list is active in deny.toml

- **WHEN** a reviewer reads `deny.toml`
- **THEN** the `[bans]` section SHALL list the C-crypto binding
  crates named by this requirement
- **AND** `cargo deny check bans` SHALL pass on the unmodified
  workspace (proving the bans are compatible with the existing
  dependency tree)

### Requirement: FIPS Path Decision Record

The repository SHALL carry `docs/crypto-validation.md` recording the
crypto-validation strategy: the corpus-replay policy, the rationale
for rejecting C crypto libraries (memory-safety-by-construction,
GPL-or-commercial licensing, non-transferable FIPS 140-3 operational
environment boundaries, DO-178C evidence economics), the enumerated
future options should FIPS or Common Criteria validation become a
hard requirement (commercial wolfCrypt FIPS as a feature-gated
container-mode-only backend; CMVP validation of SmallAIOS's own
modules; contractual acceptance of corpus-tested clean-room crypto),
and the explicit revisit triggers. The document SHALL be linked from
CLAUDE.md's Key Design Decisions.

#### Scenario: Decision record exists with revisit triggers

- **WHEN** a reviewer reads `docs/crypto-validation.md`
- **THEN** it SHALL state the no-C-crypto decision with its four
  rationale points
- **AND** it SHALL enumerate the future FIPS options
- **AND** it SHALL list concrete revisit triggers (a deployment
  contract requiring FIPS 140-3 validation; a certification
  authority rejecting corpus-vector evidence; a required primitive
  judged too complex to clean-room safely)

#### Scenario: CLAUDE.md points at the decision record

- **WHEN** a reviewer reads CLAUDE.md's Key Design Decisions section
- **THEN** it SHALL reference `docs/crypto-validation.md` for the
  crypto validation strategy
