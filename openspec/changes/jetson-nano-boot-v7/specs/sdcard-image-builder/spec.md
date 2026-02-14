## ADDED Requirements

### Requirement: SD card image build target
The build system SHALL provide a `make sdcard-jetson` target that produces a bootable microSD card image for the Jetson Nano.

#### Scenario: Build SD card image
- **WHEN** the user runs `make sdcard-jetson`
- **THEN** the system SHALL produce a file `build/sdcard-jetson.img` containing a GPT-partitioned disk image with one ext4 partition

#### Scenario: Image contains required boot files
- **WHEN** the SD card image is mounted
- **THEN** it SHALL contain `/boot/Image` (the SmallAIOS kernel), `/boot/tegra210-p3450-0000.dtb` (device tree blob), and `/boot/extlinux/extlinux.conf` (U-Boot boot configuration)

### Requirement: extlinux.conf boot configuration
The SD card image SHALL include an `extlinux.conf` that directs U-Boot to load the SmallAIOS kernel with UART console output.

#### Scenario: Boot config contents
- **WHEN** U-Boot reads `/boot/extlinux/extlinux.conf` from the SD card
- **THEN** the config SHALL specify `LINUX /boot/Image`, `FDT /boot/tegra210-p3450-0000.dtb`, and `APPEND console=ttyS0,115200n8`

#### Scenario: Auto-boot without user interaction
- **WHEN** the Jetson Nano powers on with the SD card inserted
- **THEN** U-Boot SHALL automatically load and boot SmallAIOS after the default timeout (no key press required)

### Requirement: dd-writable image
The output image SHALL be writable to a physical microSD card using standard `dd` or GUI tools (e.g., balenaEtcher).

#### Scenario: Flash to microSD
- **WHEN** the user runs `dd if=build/sdcard-jetson.img of=/dev/sdX bs=4M status=progress`
- **THEN** the resulting microSD card SHALL boot the Jetson Nano to SmallAIOS

### Requirement: Image size
The SD card image SHALL be no larger than 64 MB (the kernel binary is < 15 MB; the image includes partition table overhead, filesystem metadata, and DTB).

#### Scenario: Image fits on any microSD card
- **WHEN** the image is built
- **THEN** the file size SHALL be at most 64 MB

### Requirement: DTB vendored in repository
The Tegra 210 device tree blob (`tegra210-p3450-0000.dtb`) SHALL be vendored in the repository at `arch/aarch64/dtb/tegra210-p3450-0000.dtb` for reproducible builds.

#### Scenario: DTB available without external downloads
- **WHEN** `make sdcard-jetson` is run on a fresh clone
- **THEN** the build SHALL NOT require downloading external files — all inputs are in the repository
