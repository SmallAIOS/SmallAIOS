## ADDED Requirements

### Requirement: /data/ directory layout
All operator-tunable configuration SHALL live under `/data/` in the documented hybrid layout: a top-level `system.toml` for cross-cutting knobs and per-subsystem files for substantive configuration. v1 layout:

```text
/data/
├── system.toml              # hostname, time zone, log level, mDNS default
├── auth/
│   └── shadow               # 0600 root — passwords, role table
├── network/                 # populated by network-management-v1
│   ├── eth0.toml
│   ├── eth1.toml
│   └── bond0.toml
├── mgmt/
│   ├── zenoh.toml           # listen endpoints, PSK paths
│   └── policy.toml          # role defs, rate limits, lockout, idle, audit, password policy, metrics cadence
├── update/                  # populated by remote-update-v1
│   └── policy.toml
└── automotive/              # populated by automotive-bus-management-v1
    └── uds.toml
```

The hybrid layout SHALL be preferred over a monolithic file because it: (a) gives permission granularity (`auth/shadow` 0600, `network/*.toml` viewer-readable); (b) prevents a partial write-failure on one subsystem from corrupting another's config.

#### Scenario: Fresh /data/ contains expected v1 paths
- **WHEN** the system runs first-boot completion
- **THEN** `/data/system.toml`, `/data/auth/shadow`, `/data/mgmt/zenoh.toml`, and `/data/mgmt/policy.toml` SHALL exist with conservative defaults

### Requirement: Per-file permission declaration
Each declared file SHALL have a per-file permission declared in the schema. The loader SHALL refuse to read a file whose mode is laxer than declared.

| File | Mode | Owner |
|------|:----:|:-----:|
| `/data/system.toml` | 0644 | kernel |
| `/data/auth/shadow` | 0600 | kernel |
| `/data/mgmt/zenoh.toml` | 0640 | kernel |
| `/data/mgmt/policy.toml` | 0640 | kernel |
| `/data/network/*.toml` | 0644 | kernel |
| `/data/update/policy.toml` | 0640 | kernel |
| `/data/automotive/uds.toml` | 0640 | kernel |

#### Scenario: Stricter-than-declared mode accepted
- **WHEN** `/data/mgmt/zenoh.toml` exists with mode 0600 (declared 0640)
- **THEN** the loader SHALL accept the file

#### Scenario: Laxer-than-declared mode rejected
- **WHEN** `/data/auth/shadow` exists with mode 0644 (declared 0600)
- **THEN** the loader SHALL refuse and SHALL treat the file as corrupt
