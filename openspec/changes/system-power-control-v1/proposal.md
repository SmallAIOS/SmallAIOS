## Why

Once `management-login-v1` lands, an operator can log in (locally
or via Zenoh) but still has no way to **reboot or shut down** the
machine remotely or from the console. Today the only way to power-
cycle a SmallAIOS box is to physically pull the plug, which is
fine on a developer desk but unworkable for any deployed
appliance — Jetson Orin in a rack, an industrial controller in a
cabinet, a bench unit running long inference soak tests, or
anything reached only over a serial cable.

The reset paths themselves already exist in the architecture
crates (PSCI on AArch64, ACPI / triple-fault on x86-64, SBI on
RISC-V) — what is missing is the **management surface that
invokes them**: an authenticated, audited, role-checked entry
point reachable from both the console session and the Zenoh
admin keyspace.

## What Changes

### New verbs

- `system_reboot()` — graceful shutdown of the inference scheduler,
  flush of audit log, then platform reset.
- `system_shutdown()` — same as reboot but halts at the lowest
  power state the platform supports without resuming (PSCI
  `SYSTEM_OFF`, ACPI S5).
- `system_status()` — uptime, boot-slot, power-state, watchdog
  state. Read-only; available to `Role::Viewer`.

All three flow through the same role gate as the rest of the
admin surface: `reboot` and `shutdown` are root-only,
`status` is viewer-OK.

### Console surface

- `reboot` and `shutdown` commands available in the post-login
  console shell (the small command parser added by
  `management-login-v1`). Both prompt `Confirm? [y/N]` before
  executing — no flag override; physical-presence operators can
  still mistype.

### Zenoh surface

- `smallaios/admin/system/reboot` — request/response. Body is the
  bearer token + a fresh confirmation nonce returned from a prior
  GET on `smallaios/admin/system/reboot/nonce`. Two-step to
  prevent a stale token from triggering a reboot during a network
  partition.
- `smallaios/admin/system/shutdown` — same pattern.
- `smallaios/admin/system/status` — single-shot query, viewer-OK.

### Platform reset paths

- AArch64: `psci::system_reset()` and `psci::system_off()` via
  `HVC #0` (already wrapped in `arch/aarch64`). Verified on
  Jetson Orin under KVM and on QEMU virt.
- x86-64: ACPI `\_S5` for shutdown; reboot via the 8042 keyboard
  controller (`port 0x64 ← 0xFE`) with a triple-fault fallback.
  Already prototyped in `arch/x86_64`.
- RISC-V: SBI HSM extension (`SBI_HSM_HART_STOP`) for shutdown,
  SBI SRST for reset. (No hardware to test on; QEMU validation
  only for v1.)

### New syscall

One new syscall (46) — `system_power(action: u8) -> 0|err` —
where `action` is one of `{REBOOT=1, SHUTDOWN=2, STATUS=3}`.
Wraps the platform-specific calls behind a single ABI entry.
Root-only at the kernel boundary; the console / Zenoh layers
re-check before invoking.

### Audit

Every successful `reboot` / `shutdown` writes a final record to
the in-kernel audit ring (`who`, `when`, `transport`, `nonce`)
that survives the reset because it lives on disk before the
platform call returns. On the next boot, the first telemetry
publish includes "last shutdown was clean / by user X via
transport Y."

### Out of scope for v1 (flagged)

- Wake-on-LAN / IPMI / BMC integration.
- Scheduled reboots (cron-style).
- Graceful inference-job draining beyond a fixed 10-second
  timeout (long-running training jobs would need their own
  drain protocol — separate change).
- Suspend / sleep / S3. The unikernel has no resume path today.
- Reboot into a specific boot slot (covered in
  `remote-update-v1` once A/B slots exist).

## Capabilities

### New Capabilities

- `system-power`: definitions of the three verbs, the
  confirmation-nonce protocol, the role partition (root for
  reboot/shutdown, viewer for status), and the audit-record
  format that survives the reset.

### Modified Capabilities

- `kernel-syscalls`: adds `system_power` (#46).
- `mgmt-zenoh-admin`: adds the `smallaios/admin/system/**`
  keyspace (gated on `management-login-v1`).
- `auth-roles`: extends the role-vs-syscall matrix to cover
  `system_power`.
- `arch-aarch64` / `arch-x86_64` / `arch-riscv64`: each gains a
  `platform_reset()` / `platform_off()` contract their HAL
  must satisfy.

## Impact

- **Code:** ~150 lines kernel-side (syscall + role gate + audit
  record), ~80 lines Zenoh handler, ~40 lines console command,
  thin wrappers per arch crate.
- **Tests:** unit tests for nonce expiry, role rejection,
  audit-record persistence; QEMU integration tests that issue
  a Zenoh reboot and observe the second boot. Aim for ~30 new
  passing tests.
- **Boot footprint:** negligible (<2 KB).
- **Downstream:** unblocks operating remote/headless boxes
  without physical access. Required prerequisite for
  `remote-update-v1` (the update flow ends with a
  reboot-into-new-slot).
- **Dependencies:** `management-login-v1` — provides auth, roles,
  the Zenoh admin keyspace, **and the management surface
  convention** (`Config` model, `ConfigSurface` trait, audit log,
  `/data/` layout). This change does not add fields to `Config`
  (verbs only, no persistent settings) but its `reboot` /
  `shutdown` / `status` verbs are exposed through the same TTY,
  Zenoh, and (later) UDS surfaces by implementing handlers, not
  by adding new transport plumbing.
- **Risks:** (1) A bug that triggers a reboot loop is
  catastrophic on a remote box — the watchdog-rollback machinery
  doesn't exist until `remote-update-v1`, so for v1 we must
  *guarantee* `system_power` cannot be reached without an
  authenticated root session. (2) RISC-V SBI path is QEMU-only;
  documented as such.

## Open Questions

1. Should `system_power` block until the inference scheduler has
   drained, or fire-and-forget once the audit record is flushed?
   Block-until-drained is safer; fire-and-forget is more
   responsive. Leaning block with a 10 s ceiling.
2. Should the confirmation nonce live in volatile memory (lost
   across crashes — fail safe) or persist to disk (survives a
   network blip — fail open)? Leaning volatile.
