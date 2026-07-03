## ADDED Requirements

### Requirement: Viewer read-only yardstick

`Role::Viewer` SHALL be the primary user of the console monitor, and the monitor's observed surface SHALL be the v1 yardstick for what "read-only" means: everything the monitor displays SHALL be accessible to a `Role::Viewer` session, and nothing reachable from within the monitor SHALL mutate system state for any role.

#### Scenario: Viewer can observe the full monitor surface

- **WHEN** a `Role::Viewer` session runs the console monitor with every section enabled
- **THEN** every displayed metric SHALL be delivered without a permission error

#### Scenario: The monitor surface stays read-only for Viewer

- **WHEN** a `Role::Viewer` monitor session submits any state-mutating request (e.g., `model_unload`)
- **THEN** the kernel SHALL return `-EPERM` per the existing role-vs-syscall partition
- **AND** no state SHALL be mutated
