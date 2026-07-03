## ADDED Requirements

### Requirement: `top` built-in command registration

The console shell SHALL register `top` as a built-in command available to every role (`Role::Viewer`, `Role::Operator`, `Role::Root`). The read-only/writable distinction SHALL be enforced inside the monitor, not at the command boundary: every role can run `top`. Launching the monitor SHALL append an audit record for the session so the "session ran the monitor" event is traceable.

#### Scenario: top appears as a built-in for a Viewer

- **WHEN** an authenticated `Role::Viewer` session enters `top` at the shell prompt
- **THEN** the shell SHALL dispatch to the console monitor rather than reporting an unknown command

#### Scenario: No role check at the command boundary

- **WHEN** sessions with each of the three roles enter `top`
- **THEN** the shell SHALL launch the monitor for all three
- **AND** any role-dependent behavior SHALL be applied inside the monitor

#### Scenario: Launch is audited

- **WHEN** a session launches `top`
- **THEN** an audit record naming the session and the monitor launch SHALL be appended
