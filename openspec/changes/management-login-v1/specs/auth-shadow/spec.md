## ADDED Requirements

### Requirement: Shadow file record format
The shadow file at `/data/auth/shadow` SHALL store one user per line in colon-separated extensible format with a PHC-encoded Argon2id hash and append-only trailing fields. The canonical form is:

```text
username:$argon2id$v=19$m=<m>,t=<t>,p=<p>$<base64-salt>$<base64-tag>:role=<root|operator|viewer>:flags=<u32>:last_changed=<unix-day>:totp_secret=<base32-or-empty>:lockout_until=<unix-or-zero>
```

Parsers SHALL ignore unknown trailing fields so future extensions do not break existing shadows. The `flags` field is a `u32` bitmask whose bit 0 is `must_change_password_on_login`; remaining bits are reserved.

#### Scenario: Round-trip parse and serialize
- **WHEN** a `User` struct is serialized then re-parsed
- **THEN** every declared field SHALL match exactly, and any unknown trailing field SHALL be preserved verbatim

#### Scenario: Forward-compatible parse
- **WHEN** a parser encounters a record with an unknown trailing field (e.g., `mfa_kind=fido2`)
- **THEN** the record SHALL parse successfully and the unknown field SHALL be preserved on rewrite

#### Scenario: Reject malformed PHC string
- **WHEN** the Argon2id PHC string is missing required parameters or has an unrecognized version
- **THEN** parsing SHALL return `-EINVAL` and the file SHALL be treated as corrupt per the boot-state requirement

### Requirement: Argon2id per-tier parameter selection
At first boot, the kernel SHALL measure available RAM and select Argon2id parameters from one of three tiers, with the parameters embedded in each generated PHC string so verification is self-describing:

| Tier | RAM | m_cost | t_cost | p_cost |
|------|-----|--------|--------|--------|
| `tiny` | ≤ 256 MiB | 8 MiB | 3 | 1 |
| `default` | > 256 MiB and < 4 GiB | 64 MiB | 3 | 1 |
| `strong` | ≥ 4 GiB | 128 MiB | 4 | 2 |

#### Scenario: Tiny board selects tiny tier
- **WHEN** the kernel boots on a target with 256 MiB of RAM
- **THEN** new password hashes SHALL be generated with `m=8 MiB, t=3, p=1`
- **AND** the PHC string SHALL record those parameters

#### Scenario: Jetson-class selects strong tier
- **WHEN** the kernel boots on a Jetson Orin NX (16 GiB)
- **THEN** new password hashes SHALL be generated with `m=128 MiB, t=4, p=2`

#### Scenario: Verification reads parameters from hash
- **WHEN** verifying a password against a stored hash whose tier differs from the runtime tier
- **THEN** verification SHALL use the parameters in the stored PHC string, not the runtime tier

### Requirement: Atomic shadow rewrite
Every write to `/data/auth/shadow` SHALL stage to `<path>.tmp`, `fsync` the staged file, then `rename` atomically over the target. The rename SHALL be the only operation visible to other readers; partial writes SHALL never appear at the canonical path.

#### Scenario: Successful rewrite
- **WHEN** a user record is updated
- **THEN** `<path>.tmp` SHALL be created and fsync'd, then renamed to `<path>` in a single step

#### Scenario: Crash mid-rename leaves shadow intact
- **WHEN** the system crashes after the staged write but before the rename
- **THEN** on next boot the original `<path>` SHALL be intact and the orphan `<path>.tmp` SHALL be removed before any login is permitted

### Requirement: Shadow file permission defense
The shadow file SHALL be mode 0600 owned by the kernel. The loader SHALL refuse to read a shadow file whose mode is laxer than declared (defense in depth against an accidental world-readable shadow).

#### Scenario: Mode 0644 shadow rejected
- **WHEN** the kernel boots and `/data/auth/shadow` has mode 0644
- **THEN** the loader SHALL refuse to read it and treat the file as corrupt per the boot-state requirement

#### Scenario: Mode 0600 shadow accepted
- **WHEN** the kernel boots and `/data/auth/shadow` has mode 0600
- **THEN** the loader SHALL parse it normally

### Requirement: Shadow boot states
When `/data/auth/shadow` is missing, the kernel SHALL enter first-boot setup. When the file exists but is corrupt (parse failure, mode laxer than declared, truncated, malformed PHC string, mismatched field count after the colon-trailing convention), the kernel SHALL halt with an explicit recovery message instructing the operator to boot with `auth.skip-firstboot` to regenerate.

#### Scenario: Missing shadow enters first-boot
- **WHEN** the kernel boots and `/data/auth/shadow` does not exist
- **THEN** the kernel SHALL prompt `Set initial root password:` on the console

#### Scenario: Corrupt shadow halts with recovery hint
- **WHEN** the kernel boots and `/data/auth/shadow` fails to parse
- **THEN** the kernel SHALL print a message naming the failure (`parse error`, `mode laxer than 0600`, `truncated`) and the recovery boot argument
- **AND** the kernel SHALL NOT silently overwrite the corrupt shadow
