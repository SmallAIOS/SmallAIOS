## ADDED Requirements

### Requirement: Non-Isolated Bus Refusal

The v1 default security posture SHALL be "physically-isolated diagnostic bus only". A boot-time check SHALL enforce it: when the configuration marks the bus `isolated = false`, the ISO-TP management listener SHALL refuse to bind unless either a SecOC key is configured or an explicit override flag is set. A bus marked `isolated = true` SHALL bind normally.

#### Scenario: Non-isolated bus without SecOC or override refuses to bind

- **WHEN** the configuration has `isolated = false`, no SecOC key file, and no override flag
- **THEN** the boot-time check SHALL refuse to bind the ISO-TP management listener to the bus

#### Scenario: Isolated bus binds

- **WHEN** the configuration asserts `isolated = true`
- **THEN** the ISO-TP management listener SHALL bind to the configured interface

#### Scenario: Non-isolated bus with a SecOC key binds protected

- **WHEN** the configuration has `isolated = false` and a SecOC key file is configured
- **THEN** the listener SHALL bind
- **AND** every UDS payload SHALL be protected by the SecOC-equivalent MAC layer

#### Scenario: Explicit override binds unprotected

- **WHEN** the configuration has `isolated = false`, no SecOC key, and the explicit override flag set
- **THEN** the listener SHALL bind

### Requirement: Boot-Time Security Mode Logging

Boot SHALL emit a clear log line stating which bus security mode is in effect (isolated bus, SecOC-equivalent MAC, or explicit unprotected override), so operators who expected SecOC by default are not surprised by the conservative isolated-only default. When the listener refuses to bind, the log SHALL state why and which configuration would permit binding.

#### Scenario: Selected mode is logged at boot

- **WHEN** the ISO-TP management listener binds
- **THEN** a boot log line SHALL name the active mode (isolated bus, SecOC, or override)

#### Scenario: Refusal is logged with remediation

- **WHEN** the boot-time check refuses to bind on a non-isolated bus
- **THEN** the log SHALL state that the bus is marked non-isolated
- **AND** SHALL state that configuring a SecOC key or setting the override flag would permit binding
