# Delta for Container Interface

## ADDED Requirements

### Requirement: UEFI Secure Boot Support

SmallAIOS bare metal and MicroVM kernel images SHALL support UEFI Secure Boot with ML-DSA-65 post-quantum digital signatures. The SmallAIOS UEFI application binary MUST be signed during the build process and the UEFI firmware MUST verify the signature before executing the kernel. A fallback to traditional RSA-2048/SHA-256 Secure Boot signatures SHALL be provided for firmware that does not yet support post-quantum algorithms.

#### Scenario: Successful Secure Boot with ML-DSA-65 signed kernel

- WHEN a SmallAIOS kernel image signed with ML-DSA-65 is loaded by UEFI firmware that supports post-quantum signature verification
- THEN the firmware MUST verify the ML-DSA-65 signature against the enrolled public key in the UEFI Secure Boot database (db)
- AND the firmware MUST proceed to execute the SmallAIOS kernel entry point upon successful verification
- AND the kernel MUST log "Secure Boot: verified (ML-DSA-65)" during early boot

#### Scenario: Secure Boot verification failure rejects unsigned kernel

- WHEN an unsigned or tampered SmallAIOS kernel image is presented to UEFI firmware with Secure Boot enabled
- THEN the firmware MUST refuse to execute the kernel
- AND MUST display a Secure Boot violation error to the operator

#### Scenario: Fallback to RSA-2048 Secure Boot signature

- WHEN the UEFI firmware does not support ML-DSA-65 signature verification
- AND the kernel image carries both ML-DSA-65 and RSA-2048 signatures (dual-signed)
- THEN the firmware MUST verify the RSA-2048 signature
- AND the kernel MUST log "Secure Boot: verified (RSA-2048, PQ fallback)" during early boot

### Requirement: Bare Metal Provisioning via Network Boot

SmallAIOS SHALL support bare metal provisioning via PXE and iPXE network boot protocols. The provisioning flow SHALL deliver the SmallAIOS kernel image, ONNX model files, and runtime configuration to target hardware over the network without requiring local storage media. The provisioning server SHALL serve images over TFTP (PXE) or HTTP/HTTPS (iPXE).

#### Scenario: PXE boot of SmallAIOS on x86-64 hardware

- WHEN a target machine with PXE-capable NIC performs a network boot
- AND the DHCP server provides the SmallAIOS PXE boot filename and TFTP server address
- THEN the PXE client MUST download the SmallAIOS kernel image via TFTP
- AND MUST download the initial ramdisk containing ONNX models and configuration
- AND SmallAIOS MUST boot and reach ready state within 5 seconds of image download completion

#### Scenario: iPXE boot with HTTPS image delivery

- WHEN a target machine boots iPXE firmware and chains to a SmallAIOS iPXE script
- THEN iPXE MUST download the SmallAIOS kernel image via HTTPS from the provisioning server
- AND the downloaded image MUST be verified against an ML-DSA-65 signature before execution
- AND the iPXE script MUST pass kernel parameters (model URL, IPC port, node ID) via the kernel command line

#### Scenario: ARM64 bare metal network boot via UEFI HTTP Boot

- WHEN an ARM64 target with UEFI HTTP Boot support performs a network boot
- THEN the UEFI firmware MUST discover the SmallAIOS boot image URL via DHCP option 59 (Boot File URL)
- AND MUST download and execute the SmallAIOS UEFI application
- AND SmallAIOS MUST parse the kernel command line to locate model and configuration URLs for subsequent download

### Requirement: VM Image Generation

The SmallAIOS build system SHALL generate virtual machine images in raw disk, qcow2 (QEMU/KVM), and VMDK (VMware) formats. Each generated image SHALL contain the SmallAIOS kernel, a read-only filesystem with ONNX models and configuration, and a GPT partition table with an EFI System Partition. The build process MUST be reproducible: identical source and configuration inputs MUST produce bit-for-bit identical images.

#### Scenario: Generate raw disk image

- WHEN the build system is invoked with `--format raw`
- THEN it MUST produce a raw disk image containing a GPT partition table with an EFI System Partition (ESP) and a data partition
- AND the ESP MUST contain the SmallAIOS UEFI application binary at `/EFI/BOOT/BOOTX64.EFI` (x86-64) or `/EFI/BOOT/BOOTAA64.EFI` (ARM64)
- AND the data partition MUST contain the ONNX models and `smallaios.toml` configuration

#### Scenario: Generate qcow2 image from raw image

- WHEN the build system is invoked with `--format qcow2`
- THEN it MUST produce a qcow2 image with the same logical contents as the raw image
- AND the qcow2 image file size MUST be smaller than the raw image due to sparse allocation
- AND the image MUST boot successfully in QEMU with `qemu-system-x86_64 -drive file=smallaios.qcow2,format=qcow2 -bios /usr/share/OVMF/OVMF_CODE.fd`

#### Scenario: Generate VMDK image for VMware

- WHEN the build system is invoked with `--format vmdk`
- THEN it MUST produce a VMDK image compatible with VMware ESXi 7.0+ and VMware Workstation 16+
- AND the VMDK descriptor MUST specify the virtual hardware version and disk adapter type

#### Scenario: Reproducible image generation

- WHEN the build system generates a VM image twice from the same source tree, configuration, and toolchain version
- THEN the two output images MUST be byte-for-byte identical
- AND all timestamps within the image filesystem MUST be set to a fixed epoch value (2026-01-01T00:00:00Z)

### Requirement: Image Signing with ML-DSA-65 Post-Quantum Signatures

All SmallAIOS release artifacts (OCI container images, VM images, bare metal kernel binaries) SHALL be signed with ML-DSA-65 post-quantum digital signatures. The signing key SHALL be stored in a hardware security module (HSM) or equivalent secure key storage. Signature verification tooling SHALL be provided for operators to verify image authenticity before deployment.

#### Scenario: Sign OCI container image with ML-DSA-65

- WHEN the CI/CD pipeline produces a SmallAIOS OCI container image
- THEN the pipeline MUST compute the ML-DSA-65 signature over the image manifest digest
- AND MUST attach the signature as a Sigstore/cosign-compatible annotation
- AND the signature MUST be verifiable using the published SmallAIOS public key

#### Scenario: Sign VM image with ML-DSA-65

- WHEN the build system produces a VM image (raw, qcow2, or VMDK)
- THEN it MUST compute the ML-DSA-65 signature over the SHA3-256 hash of the image file
- AND MUST produce a detached signature file (`.sig`) alongside the image
- AND the signature MUST include the image filename, hash algorithm, and signing timestamp in a structured header

#### Scenario: Verify image signature before deployment

- WHEN an operator runs `smallaios-verify --image smallaios.qcow2 --sig smallaios.qcow2.sig --pubkey release.pub`
- THEN the tool MUST compute the SHA3-256 hash of the image file
- AND MUST verify the ML-DSA-65 signature against the provided public key
- AND MUST print "Signature valid" with the signing timestamp on success
- AND MUST exit with a non-zero status code and print "Signature INVALID" on verification failure

#### Scenario: Reject image with expired or revoked signing key

- WHEN an operator attempts to verify an image signed with a revoked key
- AND the revocation list is available (via CRL or embedded revocation timestamp)
- THEN the verification tool MUST reject the signature
- AND MUST print a warning indicating the signing key has been revoked with the revocation date

> **Note**: Kubernetes orchestration (Virtual Kubelet provider, K3s/K8s integration, pod spec translation) has been moved to the new `kubernetes-integration` capability and is no longer part of this specification.
