## ADDED Requirements

### Requirement: `/data/audit_export/` directory

On first boot of an image that carries the `audit-export` capability compiled in, the kernel SHALL ensure `/data/audit_export/` exists with mode 0700 owned by kernel. The directory SHALL be created atomically alongside the rest of the `/data/` tree per the existing first-boot directory-tree-creation requirement.

#### Scenario: Directory created on first boot
- **WHEN** the kernel formats `/data/` per `embedded-filesystem-v1` and the `audit-export` capability is compiled in
- **THEN** `/data/audit_export/` SHALL exist with mode 0700
- **AND** an audit record `audit_export_directory_initialized` SHALL be appended

#### Scenario: Directory not created without capability
- **WHEN** `audit-export` is not compiled in
- **THEN** `/data/audit_export/` SHALL NOT be created
- **AND** the absence SHALL not block boot

### Requirement: Per-file permission table — audit-export entries

The per-file permission declaration table SHALL include entries for the audit-export files:

| File | Mode | Owner |
|------|:----:|:-----:|
| `/data/audit_export/immudb.toml` | 0640 | kernel |
| `/data/audit_export/immudb.token` | 0600 | kernel |
| `/data/audit_export/last_state.bin` | 0600 | kernel |
| `/data/audit_export/last_state.bin.tmp` | 0600 | kernel |

The loader SHALL refuse to read any of these files whose mode is laxer than declared, identically to the existing per-file permission enforcement rule.

#### Scenario: Token mode 0644 rejected
- **WHEN** `/data/audit_export/immudb.token` exists with mode 0644 (declared 0600)
- **THEN** the loader SHALL refuse to read the file
- **AND** the exporter SHALL refuse to start

#### Scenario: TOML mode 0600 (stricter than declared) accepted
- **WHEN** `/data/audit_export/immudb.toml` exists with mode 0600 (declared 0640)
- **THEN** the loader SHALL accept the file per the stricter-than-declared rule

### Requirement: Token-bytes secret redaction

The secret-redaction rule applied to audit records' `before` and `after` JSON SHALL be extended to cover the contents of `/data/audit_export/immudb.token`. Any audit record that references this path SHALL render its `before` and `after` values as fixed-length placeholders (e.g. `"<redacted:64>"`), never as the literal bytes.

#### Scenario: Token rotation audit redacted
- **WHEN** `Role::Root` writes a new token to `/data/audit_export/immudb.token`
- **THEN** the resulting `config_write` audit record SHALL show `before` and `after` as redacted placeholders
- **AND** SHALL NOT contain any byte of either token

#### Scenario: Read-only inspection redacted
- **WHEN** a debug tool emits the contents of `immudb.token` into an audit-visible buffer
- **THEN** the byte sequence SHALL be redacted before the buffer is hashed into the chain
