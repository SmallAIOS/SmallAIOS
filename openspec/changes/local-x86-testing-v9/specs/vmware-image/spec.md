## ADDED Requirements

### Requirement: VMware VMDK image builder
The repository SHALL provide a script `scripts/make-vmware-x86.sh` that creates a VMware-compatible VMDK disk image containing the bare-metal x86 SmallAIOS kernel with a GRUB2 bootloader.

#### Scenario: Build VMDK image
- **WHEN** `make vmware-x86` is run
- **THEN** the script SHALL first build the x86 bare-metal kernel via `build-kernel-x86`
- **AND** create a 64 MB raw disk image with GPT partition table
- **AND** create a BIOS boot partition (1 MB) and an ext4 data partition
- **AND** install GRUB2 with Multiboot2 support into the BIOS boot partition
- **AND** copy the kernel binary to `/boot/smallaios-x86_64` on the ext4 partition
- **AND** write a GRUB configuration that boots the SmallAIOS kernel via Multiboot2
- **AND** convert the raw image to VMDK format using `qemu-img convert -O vmdk`
- **AND** output the VMDK to `build/smallaios-x86.vmdk`

#### Scenario: VMDK image size
- **WHEN** the VMDK image is created
- **THEN** the image SHALL be less than 64 MB

#### Scenario: Missing host tools
- **WHEN** required tools (`grub-install`, `sgdisk`, `mkfs.ext4`, `qemu-img`, `losetup`) are not found
- **THEN** the script SHALL print an error message listing the missing tools and their package names for Ubuntu and Fedora
- **AND** the script SHALL exit with a non-zero status

### Requirement: VMX configuration template
The repository SHALL provide a VMX template at `scripts/vmware-template.vmx` that configures a VMware virtual machine for SmallAIOS.

#### Scenario: Default VM configuration
- **WHEN** the VMX template is generated alongside the VMDK
- **THEN** the VMX SHALL configure: `guestOS = "other-64"`, 512 MB RAM, 2 vCPUs, LSI Logic SCSI controller, the VMDK as the boot disk, and a serial port mapped to `build/vmware-serial.log`

#### Scenario: Open in VMware
- **WHEN** the user opens `build/smallaios-x86.vmx` in VMware Workstation or Fusion
- **THEN** VMware SHALL recognize the VM configuration and boot from the VMDK
- **AND** GRUB SHALL load the SmallAIOS kernel via Multiboot2
- **AND** kernel serial output SHALL be captured to `build/vmware-serial.log`

### Requirement: GRUB configuration for Multiboot2
The GRUB configuration installed in the VMDK SHALL boot SmallAIOS using the Multiboot2 protocol.

#### Scenario: GRUB boot entry
- **WHEN** GRUB loads
- **THEN** the default boot entry SHALL be `SmallAIOS`
- **AND** the entry SHALL use `multiboot2 /boot/smallaios-x86_64`
- **AND** the boot timeout SHALL be 3 seconds

### Requirement: Makefile target for VMware image
The Makefile SHALL provide a `vmware-x86` target.

#### Scenario: Build VMware image
- **WHEN** `make vmware-x86` is run
- **THEN** it SHALL depend on `build-kernel-x86`
- **AND** it SHALL invoke `scripts/make-vmware-x86.sh`
- **AND** it SHALL print the path to the output VMDK and VMX files
