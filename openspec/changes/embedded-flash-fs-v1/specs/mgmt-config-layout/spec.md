## ADDED Requirements

### Requirement: /flash/ directory tree on first boot
On first boot of an image carrying the `fs-flash` cargo feature enabled with a mounted `/flash/`, the kernel SHALL ensure the canonical `/flash/` directory tree exists with the declared modes:

```text
/flash/secrets/         mode 0700  -- update-signing public key, attestation state
/flash/secure-config/   mode 0700  -- power-fail-critical configuration
```

For flash-only targets (no F2FS `/data/`), the kernel SHALL also create:

```text
/flash/auth/            mode 0700
/flash/audit/           mode 0700
/flash/mgmt/            mode 0700
```

Directory creation SHALL be atomic across the set: either the entire tree exists or the kernel halts.

#### Scenario: Both-substrate target creates only flash-specific paths
- **WHEN** the system has both /data/ (F2FS) and /flash/ (littlefs)
- **THEN** `/flash/secrets/` and `/flash/secure-config/` SHALL exist
- **AND** `/flash/auth/`, `/flash/audit/`, `/flash/mgmt/` SHALL NOT be created (those live on /data/)

#### Scenario: Flash-only target creates full tree
- **WHEN** the system has only /flash/ (no /data/)
- **THEN** all five directories SHALL exist
- **AND** `auth_login` SHALL read `/flash/auth/shadow` instead of `/data/auth/shadow`

#### Scenario: Atomic creation
- **WHEN** the kernel begins creating the /flash/ tree and a power-loss occurs after creating only /flash/secrets/
- **THEN** on next boot the kernel SHALL detect the partial tree
- **AND** SHALL complete the missing directories
- **AND** SHALL append a `flash_tree_repaired` audit record

### Requirement: Per-file permission table extended for /flash/
The per-file permission declaration table SHALL include declarations for the canonical /flash/ paths:

| File | Mode | Owner |
|------|:----:|:-----:|
| `/flash/secrets/update-key.pub` | 0644 | kernel (public key, world-readable for verification) |
| `/flash/secrets/sign-key.priv` | 0600 | kernel (signing key, never readable from user space) |
| `/flash/secure-config/*.toml` | 0640 | kernel |
| `/flash/auth/shadow` (flash-only) | 0600 | kernel |
| `/flash/audit/log.jsonl` (flash-only) | 0640 | kernel |
| `/flash/mgmt/policy.toml` (flash-only) | 0640 | kernel |

The same mode-stricter-than-declared loader rule from `mgmt-config-layout` SHALL apply.

#### Scenario: Lax mode on signing key rejected
- **WHEN** `/flash/secrets/sign-key.priv` exists with mode 0644 (declared 0600)
- **THEN** the loader SHALL refuse to read it
- **AND** the kernel SHALL halt with a recovery message
