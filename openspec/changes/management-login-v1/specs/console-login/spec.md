## ADDED Requirements

### Requirement: First-boot initial root setup
On a system whose `/data/auth/shadow` does not exist, the kernel SHALL prompt for an initial root password before any service that handles untrusted data starts. The prompt SHALL: print `Set initial root password:`, read with echo suppressed, prompt `Confirm:`, verify the two entries match, hash with the runtime-tier Argon2id parameters, and atomically write the resulting shadow record. The created record SHALL have `role=root` and `flags=0` (no forced rotation; the operator just chose the password).

#### Scenario: First-boot creates root entry
- **WHEN** the kernel boots and `/data/auth/shadow` does not exist
- **THEN** the kernel SHALL prompt for and confirm a root password
- **AND** SHALL atomically write a shadow file containing exactly one record with `role=root`

#### Scenario: Mismatched confirmation re-prompts
- **WHEN** the operator's confirmation does not match the first entry
- **THEN** the kernel SHALL print a mismatch message and re-prompt up to 3 attempts
- **AND** if 3 attempts fail, the kernel SHALL halt with a recovery message

### Requirement: TTY login flow
On every subsequent boot, the kernel SHALL prompt `username:` then `password:` (with echo suppressed). On success, the kernel SHALL establish a session token bound to the caller's role and emit a banner naming the user and role. On failure, the kernel SHALL re-prompt subject to the lockout requirement.

#### Scenario: Successful login establishes session
- **WHEN** the operator submits the correct username and password
- **THEN** the kernel SHALL create a session token, set the active role on the TTY control plane, and append an `auth_login` audit record

#### Scenario: Wrong password re-prompts
- **WHEN** the operator submits an incorrect password (and lockout is not yet armed)
- **THEN** the kernel SHALL print `Authentication failed.` and SHALL re-prompt for `username:`

### Requirement: TTY line editing during password prompt
Password prompts SHALL honor backspace, ^U (kill line), and ^C (abort prompt) and SHALL NOT echo any character (not even `*`) to avoid leaking length. Newline SHALL submit.

#### Scenario: Backspace removes a typed character
- **WHEN** the operator types `a`, `b`, then backspace
- **THEN** the buffered password SHALL contain only `a`

#### Scenario: ^C aborts the prompt
- **WHEN** the operator presses ^C during a password prompt
- **THEN** the kernel SHALL return to `username:` prompt and SHALL NOT count the abort as a failed attempt

### Requirement: Lockout policy
After 5 consecutive failed login attempts from a single source, the kernel SHALL refuse further attempts from that source for 60 seconds. A successful login SHALL reset the counter. The TTY console SHALL count as one source; each Zenoh remote PQC peer identity SHALL be its own source. A locked remote source SHALL NOT lock the local TTY.

#### Scenario: Five fails locks for 60 seconds
- **WHEN** the TTY source records 5 consecutive failures
- **THEN** the next 60 seconds of TTY login attempts SHALL be rejected with a lockout message

#### Scenario: Successful login resets the counter
- **WHEN** the TTY source has 4 failures and then a successful login
- **THEN** the failure counter SHALL be 0

#### Scenario: Remote lockout does not affect local
- **WHEN** a remote Zenoh peer triggers a 60-second lockout
- **THEN** the TTY source SHALL still accept login attempts

### Requirement: Recovery skip-firstboot boot argument
The kernel SHALL honor an `auth.skip-firstboot` kernel boot argument **only** when a `PhysicalPresenceProvider` registered for the current architecture asserts presence. When honored, the kernel SHALL: generate a cryptographically random 16-character root password, hash it with the runtime-tier Argon2id parameters, write a fresh shadow record with `must_change_password_on_login` set, print the cleartext password once on the serial console, and append an audit record naming the recovery event. An alternative form `auth.skip-firstboot=<argon2id-phc-string>` SHALL accept a pre-computed PHC hash instead of generating a random password; the same `must_change_password_on_login` flag SHALL be set.

#### Scenario: Skip-firstboot honored with presence asserted
- **WHEN** the kernel boots with `auth.skip-firstboot` and the GPIO presence pin is asserted
- **THEN** a fresh root entry SHALL be created with the must-change flag and a one-time password printed

#### Scenario: Skip-firstboot ignored without presence
- **WHEN** the kernel boots with `auth.skip-firstboot` but no presence provider asserts
- **THEN** the boot argument SHALL be ignored and the kernel SHALL proceed with normal boot

#### Scenario: Pre-baked hash form
- **WHEN** the kernel boots with `auth.skip-firstboot=$argon2id$v=19$m=...` and presence is asserted
- **THEN** the shadow file SHALL be created with that exact hash
- **AND** `must_change_password_on_login` SHALL be set

### Requirement: Explicit logout and Ctrl-D
The TTY console SHALL accept the commands `logout` and `exit` and SHALL treat Ctrl-D (EOF) at a shell prompt the same way. Each SHALL invalidate the session token, clear the audit identity on the control plane, redraw the login prompt, and append a `logout` audit record.

#### Scenario: Logout invalidates session
- **WHEN** an authenticated session issues `logout`
- **THEN** the session token SHALL be removed from the kernel session table
- **AND** the next syscall on the TTY control plane SHALL fail with `-EAUTH`
- **AND** the kernel SHALL redraw `username:` prompt

#### Scenario: Ctrl-D at shell prompt logs out
- **WHEN** the operator presses Ctrl-D at a fresh shell prompt
- **THEN** the kernel SHALL behave identically to `logout`

### Requirement: Idle auto-logout with keypress reset
The console session SHALL be auto-logged-out per the per-role idle window from `auth-roles`. Any keypress (including arrow keys and refresh in `console-monitor-v1`) SHALL reset the timer.

#### Scenario: Idle Root session times out
- **WHEN** a Root TTY session sees no keypress for 15 minutes
- **THEN** the kernel SHALL invalidate the session, append an `auto_logout` audit record, and redraw `username:`

#### Scenario: Keypress in console-monitor refreshes timer
- **WHEN** a Viewer session is running `console-monitor-v1` and presses any key at minute 59
- **THEN** the timer SHALL reset to zero
