# security Specification

## Purpose
TBD - created by archiving change management-login-v1. Update Purpose after archive.
## Requirements
### Requirement: Argon2id KDF
The `security` crate SHALL provide an Argon2id KDF as a `#![no_std]` clean-room Rust implementation in `security/src/argon2id.rs`. The implementation SHALL conform to RFC 9106 and SHALL be tested against the RFC 9106 KAT vectors. Per-arch SIMD shims (NEON for AArch64, AVX2 for x86-64) SHALL be available behind cargo features and SHALL fall back to the portable path when not enabled. The external `argon2` crate MAY appear in `dev-dependencies` only as a validation oracle; it SHALL NOT enter the production dependency graph.

The crate SHALL expose:

```rust
pub fn argon2id_hash(
    password: &[u8],
    salt: &[u8],
    params: Argon2idParams,
) -> [u8; 32];

pub fn argon2id_verify(
    password: &[u8],
    phc_string: &str,
) -> Result<bool, Error>;

pub fn argon2id_format_phc(
    salt: &[u8],
    tag: &[u8; 32],
    params: Argon2idParams,
) -> heapless::String<256>;

pub struct Argon2idParams { pub m_cost_kib: u32, pub t_cost: u32, pub p_cost: u32 }
```

#### Scenario: RFC 9106 vector matches
- **WHEN** `argon2id_hash` is called with the RFC 9106 test vector inputs
- **THEN** the returned tag SHALL equal the RFC 9106 expected output

#### Scenario: PHC round-trip
- **WHEN** a hash is formatted to PHC and re-parsed via `argon2id_verify`
- **THEN** verification with the original password SHALL succeed
- **AND** verification with a different password SHALL fail

#### Scenario: SIMD shim matches portable path
- **WHEN** the same input is hashed with and without the SIMD feature
- **THEN** the resulting tags SHALL be byte-identical

### Requirement: TOTP (RFC 6238) module
The `security` crate SHALL provide an RFC 6238 TOTP module supporting the standard SHA-1 HMAC variant for interoperability with common authenticator apps. The implementation SHALL be `#![no_std]` and SHALL be tested against the RFC 6238 test vectors. The module SHALL expose:

```rust
pub fn totp_generate(
    secret: &[u8],
    unix_time: u64,
    digits: u32,    // typically 6
    period: u32,    // typically 30 seconds
) -> u32;

pub fn totp_verify(
    secret: &[u8],
    code: u32,
    unix_time: u64,
    digits: u32,
    period: u32,
    window: u32,    // accepted clock-skew steps
) -> bool;
```

A SHA-3-based variant MAY be added in a future change without breaking the SHA-1 path.

#### Scenario: RFC 6238 vector matches
- **WHEN** `totp_generate` is called with the RFC 6238 reference key and timestamp 59
- **THEN** the result SHALL equal the documented RFC 6238 6-digit output

#### Scenario: Clock-skew tolerance
- **WHEN** a code generated for time t is verified at time t+25 (within a 30-s period and window=1)
- **THEN** verification SHALL succeed

#### Scenario: Out-of-window code rejected
- **WHEN** a code generated for time t is verified at time t+90 (window=1)
- **THEN** verification SHALL fail

