## ADDED Requirements

### Requirement: `update/policy.toml` joins the Config model

The typed `Config` model SHALL gain an `update/policy.toml` section carrying the update policy: the watchdog confirm window (v1 default 60 seconds), slot retention, and the transport allowlist. The fields SHALL be served through the existing TOML, TTY, and Zenoh `ConfigSurface` implementations and the standard `/data/` layout and audit log from `management-login-v1` — no update-specific configuration plumbing SHALL be added.

#### Scenario: Update policy readable through every existing surface

- **WHEN** the watchdog confirm window is read via the TOML loader and via the Zenoh admin surface
- **THEN** both reads SHALL return the identical value from the same in-memory `Config` instance
- **AND** the default SHALL be 60 seconds

#### Scenario: Policy writes follow the standard apply lifecycle

- **WHEN** Root updates a field in `update/policy.toml` through any surface
- **THEN** the write SHALL pass through the existing parse → validate → stage → atomic-rename → notify → audit lifecycle
- **AND** an audit record naming the surface and the before/after values SHALL be appended
