## ADDED Requirements

### Requirement: KVM-hosted unikernel boot on Jetson Orin

The repository SHALL provide a documented and CI-gated procedure for booting the existing `aarch64-unknown-none` SmallAIOS kernel under `qemu-system-aarch64 -accel kvm -cpu host` running on a Jetson Orin host (J4012 / Orin NX or any Tegra234 device with JetPack 6 / L4T R36.4+).

#### Scenario: One-command run from a developer workstation

- **GIVEN** a developer with `cargo`, `rustup`, and ssh access to a Jetson Orin running JetPack 6
- **WHEN** they run `just run-jetson-kvm <ssh-host>` from the workspace root
- **THEN** the recipe SHALL cross-build the SmallAIOS kernel for `aarch64-unknown-none`, copy the artifact to the Jetson host, launch `qemu-system-aarch64 -M virt,gic-version=3 -cpu host -accel kvm -nographic -kernel <artifact>`, and stream the PL011 UART output back to the developer's terminal
- **AND** the boot banner SHALL appear within 30 seconds of recipe invocation
- **AND** the kernel SHALL not panic before reaching the cooperative scheduler's idle loop

#### Scenario: Smoke test wrapper for non-interactive use

- **GIVEN** the same prerequisites as the developer flow
- **WHEN** a CI / cron job invokes `scripts/test-jetson-kvm.sh <ssh-host>`
- **THEN** the script SHALL grep the captured serial output for the documented boot banner, exit 0 on a hit, and exit non-zero with a captured-output dump on a miss
- **AND** the script SHALL accept an `SSH_HOST` argument or environment variable so it can be parameterized per-runner without code changes

#### Scenario: CI-side smoke build (no self-hosted Jetson runner required)

- **GIVEN** a PR that touches the `aarch64-unknown-none` kernel build path or the `arch/aarch64` crate
- **WHEN** the PR pipeline runs
- **THEN** a `kvm-smoke-build` CI job SHALL build the kernel for `aarch64-unknown-none --release`, run it under TCG-emulated `qemu-system-aarch64 -M virt,gic-version=3 -nographic -kernel <bin>` for at most 30 seconds, and assert the documented boot banner appears
- **AND** the `kvm-smoke-build` job SHALL be wired into the `change-gates` meta-job and SHALL block merge on failure
- **AND** a comment block in the workflow file SHALL note that the job uses TCG (not KVM) because GitHub-hosted runners do not provide nested virtualization, and that on-Jetson KVM execution is verified by the `scripts/test-jetson-kvm.sh` script run manually or on a future self-hosted Jetson runner

### Requirement: UEFI USB-bootable unikernel image for Jetson Orin

The repository SHALL produce a UEFI-bootable USB image (`build-jetson-usb.img`) that boots the SmallAIOS unikernel directly on a Jetson Orin device (J4012 / Orin NX) without writing to the device's eMMC and without requiring NVIDIA's `flash.sh` recovery toolchain.

#### Scenario: Image is a standard UEFI ESP layout

- **GIVEN** a developer who has built `smallaios.efi` from `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234`
- **WHEN** they run `just build-jetson-usb-image`
- **THEN** the recipe SHALL produce a FAT32 disk image at `build-jetson-usb.img` containing `EFI/BOOT/BOOTAA64.EFI` (a copy of `smallaios.efi`) at the root of the FAT32 filesystem
- **AND** the image SHALL be `dd`-able to a USB stick of size ≥ 256 MB without further preparation
- **AND** no other partitions, no GPT, no EFI System Partition GUID gymnastics SHALL be required — the firmware SHALL detect the FAT32 ESP-style layout directly

#### Scenario: USB-stick boot does not modify the J4012's eMMC

- **GIVEN** a J4012 with stock JetPack 6 / L4T installed on its eMMC
- **WHEN** the developer inserts a USB stick imaged with `build-jetson-usb.img` and selects it in the J4012's UEFI boot menu
- **THEN** the J4012 SHALL boot into the SmallAIOS unikernel
- **AND** removing the USB stick and power-cycling SHALL restore the original L4T boot
- **AND** the workflow SHALL NOT modify EFI variables, EFI BootOrder, eMMC contents, or QSPI bootloader state at any point — verified by `efibootmgr -v` showing identical output before and after the boot test

#### Scenario: Tegra234 UART produces observable boot output

- **GIVEN** a J4012 booted from the USB stick image with a USB-to-TTL serial cable connected to the Tegra234 UART header
- **WHEN** the SmallAIOS unikernel reaches its main entry point
- **THEN** the boot banner SHALL appear over the Tegra Combined UART (TCU) at the documented MMIO base (`0x0c280000`) at the documented baud rate
- **AND** the GICv3 distributor + redistributor init SHALL log success
- **AND** the cooperative scheduler SHALL reach its idle loop and emit at least one heartbeat tick that is observable on the serial console

### Requirement: tegra234 Cargo feature is distinct from existing Tegra features

The `tegra234` Cargo feature on `smallaios-arch-aarch64` SHALL be the sole feature for the bare-metal Tegra234 / Orin-family BSP, and SHALL NOT overlap or conflict with the existing `tegra-x1` feature on the same crate or the existing `tegra-orin` feature on `smallaios-arch-nvidia`.

#### Scenario: Three feature names, three meanings

- **GIVEN** a developer reading the workspace `Cargo.toml` files
- **THEN** `smallaios-arch-aarch64`'s `tegra-x1` feature SHALL select the Tegra X1 / Tegra210 / cc 5.3 bare-metal HAL via `aarch64-unknown-none`
- **AND** `smallaios-arch-aarch64`'s `tegra234` feature SHALL select the Tegra234 / Orin-family bare-metal HAL via `aarch64-unknown-uefi`
- **AND** `smallaios-arch-nvidia`'s `tegra-orin` feature SHALL continue to select cc 8.7 for the **userspace CUDA** container path (Tegra Orin via NVIDIA Container Toolkit)
- **AND** the Cargo doc-comments on each feature SHALL explicitly call out the distinction so future contributors are not confused by the overlapping family names

#### Scenario: tegra234 build excludes X1-specific drivers

- **GIVEN** a build invocation `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234`
- **THEN** the `tegra_dc.rs`, `tegra_sor.rs`, `tegra_edid.rs`, `tegra_pcie.rs`, and `gicv2.rs` files SHALL NOT be compiled (they are Tegra X1 specific and not relevant to Phase 2)
- **AND** the `tegra234_uart.rs` and `gicv3.rs` files SHALL be compiled in their place
- **AND** `arch/nvidia/src/tegra/` (Maxwell GM20B GPU HAL) SHALL NOT be compiled — GPU support for Orin is explicitly out of scope for this change

### Requirement: CI build of the Tegra234 UEFI image

A CI job SHALL build the `tegra234`-feature unikernel and exercise it under OVMF UEFI on QEMU on every PR, advisory initially.

#### Scenario: PR build of the UEFI artifact

- **GIVEN** a PR that touches the `aarch64-unknown-uefi` build path or the `arch/aarch64` crate under the `tegra234` feature
- **WHEN** the PR pipeline runs
- **THEN** a `Build Jetson Orin UEFI` CI job SHALL run `cargo build --target aarch64-unknown-uefi -p smallaios-kernel --features tegra234` and produce a `smallaios.efi` artifact
- **AND** the job SHALL run with `continue-on-error: true` (advisory) until a self-hosted Jetson runner is available, at which point a follow-up change SHALL flip the gate to blocking
- **AND** the workflow comment block SHALL document the promote-to-gate criterion

#### Scenario: OVMF QEMU smoke test catches PE/COFF and UEFI entry regressions

- **GIVEN** the same PR as above
- **WHEN** the `Build Jetson Orin UEFI` job (or a sibling job) runs the produced `smallaios.efi` under `qemu-system-aarch64 -M virt -cpu cortex-a78 -bios edk2-aarch64-code.fd -drive file=fat:rw:<dir-with-EFI-BOOT-BOOTAA64.EFI>`
- **THEN** the job SHALL assert the documented boot banner appears within 60 seconds
- **AND** a regression that breaks the PE/COFF header, the UEFI entry signature, or the early UART init SHALL cause this job to fail
