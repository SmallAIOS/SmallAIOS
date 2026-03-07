## ADDED Requirements

### Requirement: QEMU GDB debugging mode
The Makefile SHALL provide a `debug-x86` target that builds the debug kernel and launches QEMU with GDB stub support.

#### Scenario: Launch debug session
- **WHEN** `make debug-x86` is run
- **THEN** it SHALL depend on `build-kernel-x86-debug`
- **AND** QEMU SHALL be launched with `-s -S` flags (GDB stub on port 1234, paused at reset vector)
- **AND** QEMU SHALL use the debug kernel at `target/x86_64-unknown-none/debug/smallaios-x86_64`
- **AND** serial output SHALL go to both stdio and `build/serial-debug.log`
- **AND** the target SHALL print: `GDB: gdb target/x86_64-unknown-none/debug/smallaios-x86_64 -ex "target remote :1234"`

#### Scenario: GDB helper file
- **WHEN** the repository is cloned
- **THEN** a `.gdbinit-x86` file SHALL exist at the repository root
- **AND** it SHALL contain commands to connect to `localhost:1234`, set a breakpoint on `kernel_main`, and continue execution

#### Scenario: Debug kernel has symbols
- **WHEN** the debug kernel is loaded into GDB
- **THEN** GDB SHALL resolve function names (e.g., `kernel_main`, `_start`) to source file and line number
- **AND** GDB SHALL support `break`, `step`, `next`, `print` commands on the running kernel

### Requirement: QEMU network-enabled mode
The Makefile SHALL provide a `run-x86-net` target that boots the kernel with a virtio-net NIC and user-mode networking.

#### Scenario: Launch with networking
- **WHEN** `make run-x86-net` is run
- **THEN** QEMU SHALL be launched with `-device virtio-net-pci,netdev=net0 -netdev user,id=net0,hostfwd=tcp::8080-:8080`
- **AND** the kernel SHALL have a PCI device visible on the virtual bus (vendor `0x1AF4`, device `0x1000`)
- **AND** host port 8080 SHALL be forwarded to guest port 8080

#### Scenario: Network mode uses release kernel
- **WHEN** `make run-x86-net` is run
- **THEN** it SHALL depend on `build-kernel-x86` (release build)

### Requirement: Serial log capture
All QEMU invocation targets SHALL capture serial output to a log file in the `build/` directory.

#### Scenario: Serial log for run-x86
- **WHEN** `make run-x86` is run
- **THEN** serial output SHALL go to both stdio (interactive) and `build/serial.log` (file)
- **AND** `build/serial.log` SHALL be overwritten on each invocation

#### Scenario: Serial log for debug-x86
- **WHEN** `make debug-x86` is run
- **THEN** serial output SHALL go to both stdio and `build/serial-debug.log`

#### Scenario: Serial log for run-x86-net
- **WHEN** `make run-x86-net` is run
- **THEN** serial output SHALL go to both stdio and `build/serial-net.log`

### Requirement: QEMU monitor access
All QEMU invocation targets SHALL provide access to the QEMU monitor console.

#### Scenario: Monitor via telnet
- **WHEN** any QEMU target is run
- **THEN** QEMU SHALL be launched with `-monitor telnet:localhost:4444,server,nowait`
- **AND** the developer SHALL be able to connect via `telnet localhost 4444` to access the QEMU monitor
- **AND** the target SHALL print the monitor connection instruction

### Requirement: Documentation for local testing workflows
The repository SHALL provide a `docs/local-testing.md` file documenting all local testing paths.

#### Scenario: Documentation content
- **WHEN** a developer reads `docs/local-testing.md`
- **THEN** it SHALL cover: prerequisites, Docker CPU-only quickstart, Docker GPU quickstart, QEMU bare-metal boot, QEMU GDB debugging, QEMU networking, VMware image creation, and troubleshooting
- **AND** each section SHALL include the exact `make` command to run
- **AND** the prerequisites section SHALL list required host packages for Ubuntu and Fedora
