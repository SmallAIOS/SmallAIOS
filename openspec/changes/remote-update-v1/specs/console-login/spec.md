## ADDED Requirements

### Requirement: `update` console command

The console SHALL accept a new `update` command from an authenticated root serial-console session. Issuing `update` SHALL put the kernel into YMODEM-1K receive mode on the same TTY; the operator then triggers the upload from their terminal emulator's YMODEM send function. When the transfer completes, fails, or is aborted, control SHALL return to the console session.

#### Scenario: Root session enters receive mode

- **WHEN** an operator logged in as root over the serial console runs `update`
- **THEN** the kernel SHALL enter YMODEM-1K receive mode on that same TTY
- **AND** SHALL begin the `<C>`-mode handshake awaiting the sender

#### Scenario: update requires root login

- **WHEN** `update` is issued without an authenticated root session
- **THEN** the command SHALL be refused
- **AND** the kernel SHALL NOT enter YMODEM receive mode

#### Scenario: Failed transfer returns to the console

- **WHEN** a transfer started via `update` fails signature verification or is cancelled by the operator
- **THEN** the console prompt SHALL be restored on the same TTY with an error message
- **AND** the boot pointer SHALL be untouched
