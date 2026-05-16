# Design — confidential-compute-v1

## Goal

A SmallAIOS instance booting as a hardware-isolated, memory-encrypted enclave on at least one production confidential-compute platform, observable as:

1. Boot under `qemu-system-aarch64 -cpu max,rme=on -machine virt` with tf-rmm as the Realm Management Monitor.
2. SmallAIOS in the Realm initializes, requests an attestation token via `RSI_ATTEST_TOKEN_*`, returns it via the `remote-attestation-v1` surface with `Backend: "cca-realm"`.
3. An ONNX model bundle, encrypted with a key the model owner releases to the Realm after verifying the attestation, decrypts into Realm-private memory.
4. An inference call across the Realm boundary (input in shared memory → encrypted compute → output in shared memory) succeeds with expected output.
5. Verifier confirms (a) the Realm Attestation Token signature against the RMM attestation root, (b) the SmallAIOS counter-signature, (c) the measurement matches the certified release.

The same observable shape applies to Phase 2 (TDX) and Phase 3 (SEV-SNP); only the inner attestation report format differs.

## Cross-platform abstraction (Phase 1 design output)

The largest single design effort is the **abstraction layer**: a Rust trait that all three platform paths implement, so the inference runtime and the attestation surface don't carry platform-specific code.

```rust
// security/src/confidential/mod.rs

/// A memory region that the hardware enclave has marked private.
/// Reads/writes by a non-enclave context (hypervisor, host OS, co-tenant)
/// see either ciphertext or hardware-enforced abort, depending on platform.
pub trait SealedRegion: Sized {
    /// Allocate a private region of `pages` 4KB pages.
    fn allocate(pages: usize) -> Result<Self, ConfidentialError>;

    /// Get a Rust slice view of the private region. Only callable from
    /// inside the enclave.
    fn as_slice(&self) -> &[u8];
    fn as_slice_mut(&mut self) -> &mut [u8];

    /// Free the region back to the enclave's heap.
    fn free(self);
}

/// An enclave that can produce a platform-specific attestation report.
pub trait AttestableEnclave {
    type Report: AsRef<[u8]>;  // CBOR / DER / vendor-specific bytes

    /// Returns the platform tag for the `AttestResponse::Backend` field.
    fn backend_tag() -> &'static str;

    /// Produces an attestation report covering the enclave's identity
    /// (measurement of its loaded code), the supplied claims (typically
    /// the nonce + SmallAIOS measurement bundle), and a hardware signature.
    fn produce_report(claims: &[u8]) -> Result<Self::Report, ConfidentialError>;
}

/// Shared-memory window between enclave and host. Used for I/O.
pub trait SharedWindow: Sized {
    fn allocate(pages: usize) -> Result<Self, ConfidentialError>;
    fn as_slice(&self) -> &[u8];           // visible to host (DON'T trust it for confidential data)
    fn as_slice_mut(&mut self) -> &mut [u8];
    fn free(self);
}
```

Phase 1 implements `SealedRegion`, `AttestableEnclave`, `SharedWindow` for CCA Realms. Phase 2 reimplements for TDX. Phase 3 reimplements for SEV-SNP. The `onnx-rt` confidential loader and the `remote-attestation-v1` glue speak to the traits, not the concrete implementations.

## Alternatives considered

### 1. Skip the cross-platform abstraction; build separate kernels per platform

Rejected. Three platform-specific kernels triples the maintenance burden, balkanizes the test surface, and prevents the verifier crate from working uniformly. The trait abstraction is ~150 LOC of Rust and saves multi-thousand-LOC duplication downstream.

### 2. Start with TDX (highest commercial deployment) instead of CCA

Considered. TDX shipped first in volume on Sapphire Rapids (2023), and the cloud-vendor offerings (Azure DCsv5, GCP Confidential VMs Gen 2) have the biggest existing customer base.

Rejected for Phase 1 because (a) SmallAIOS's AArch64 surface (Jetson Orin, planned Neoverse server) is heavier than its x86-64 datacenter surface today; (b) CCA's open RMM (`tf-rmm`) lets us CI in pure-open-source — TDX requires proprietary Intel TDX-module artifacts which (while signed and freely-distributable) are a separate trust-anchor story we don't want in the abstraction-design phase; (c) the CCA threat model is simpler to walk through first (one CPU vendor, one specification, one RMM implementation) — Phase 1's design output is cleaner if we don't have to defend "but TDX does it differently" decisions while still iterating on the abstraction.

Phase 2 picks up TDX as the natural extension once Phase 1's abstractions are stable. TDX-first deployment customers are not blocked — they wait for Phase 2 just as Phase 1 customers wait for CCA silicon.

### 3. Use a process-level enclave (Intel SGX) instead of a VM-level enclave (TDX/CCA/SNP)

Rejected. SGX is a different deployment model — application-as-enclave with the OS outside. SmallAIOS is a unikernel; the whole runtime IS the application. VM-level enclaves (TDX, CCA Realms, SNP) match the deployment shape: one SmallAIOS unikernel per enclave, the host hypervisor untrusted. SGX would require a different kernel architecture (running atop a host kernel, exposing SGX-tailored syscalls to the application) — incompatible with the unikernel design.

SGX is also being deprecated by Intel for server platforms (DCAP-style fleet attestation is officially supported but the SGX server roadmap is effectively maintenance-only). Going TDX-first when we extend to x86-64 is the future-facing choice.

### 4. Build a "confidential mode" feature flag and gate the existing kernel

Rejected. The existing kernel's memory model assumes a single address space, freely-allocatable DRAM, no encryption boundaries. Confidential compute fundamentally changes the memory model: pages have *states* (private vs shared), state transitions are mediated by the platform's trusted firmware (RMM/TDX-module/SEV-firmware), and some operations (debug breakpoints, performance counters) are restricted. Trying to retrofit the existing kernel would force every memory-touching subsystem to learn about enclave states, polluting code that has no business knowing about confidential compute.

Cleaner: a separate build path (`--features cca-realm` etc.) that links a `confidential/` runtime alongside the standard runtime. The standard build is unchanged; the confidential build adds the page-state and attestation paths. Most of the kernel (scheduler, ONNX runtime, networking) is shared.

### 5. ARM CCA via Phase 1 only; skip x86-64 entirely

Rejected. x86-64 is the largest deployment surface for SmallAIOS today (cloud datacenter, NVIDIA-CUDA-on-x86-64 container path). Confidential compute on x86-64 is the most-asked-for feature in customer conversations. CCA-only would be technically sound but commercially incomplete.

Phases 2 and 3 are committed in this proposal even though they're multi-quarter deliverables. The OpenSpec change covers the multi-quarter plan; landing happens phase-by-phase.

## Phase 1 implementation details

### Realm build target

There's no canonical `aarch64-unknown-realm` Rust target. Phase 1 uses `aarch64-unknown-none` (the existing target for bare-metal AArch64) plus:

- The `cca-realm` Cargo feature enables Realm-specific modules and removes incompatible bare-metal HAL bits.
- A custom linker script `arch/aarch64/linker-cca-realm.ld` sets the Realm's image base per the CCA spec (the RMM places the initial Realm image at an RMM-determined address; the linker script matches).
- A custom Cargo `[[bin]] required-features = ["cca-realm"]` produces a `smallaios-cca-realm` binary.

The build command becomes `cargo build --release --target aarch64-unknown-none -p smallaios-arch-aarch64 --bin smallaios-cca-realm --features cca-realm` — pattern-consistent with the `tegra234` and `tegra-x1` paths from `unikernel-orin-bringup-v1`.

### RMM interface client

The Realm Services Interface (RSI) is the API a Realm uses to ask the RMM for services. RSI calls are issued via the `SMC` instruction, same calling-convention as the OP-TEE bridge (`op-tee-bridge-v1`) — so the `arch/aarch64/src/smc.rs` module is reused.

Phase 1 implements the minimum RSI subset:

| RSI call | Purpose |
|----------|---------|
| `RSI_REALM_CONFIG` | Query Realm's IPA size, hash algo, attestation algo. |
| `RSI_IPA_STATE_GET` | Query a page's state (RAM-private / RAM-shared / unassigned). |
| `RSI_IPA_STATE_SET` | Transition a page between states. |
| `RSI_ATTEST_TOKEN_INIT` | Begin generating an attestation token; supplies challenge nonce. |
| `RSI_ATTEST_TOKEN_CONTINUE` | Fetch the next chunk of the token (tokens are large; multi-call). |
| `RSI_HOST_CALL` | Yield to host for I/O (similar to TDX's `TDG.VP.VMCALL`). |

The RSI is a stable Arm spec; the FIDs are pinned in `security/src/confidential/cca_rsi_ids.rs`. ~30 const definitions.

### Granule (page) state management

A Realm sees memory in three states:

- **RAM-private**: encrypted, only Realm reads/writes succeed. Host sees ciphertext (or possibly an abort, platform-defined).
- **RAM-shared**: cleartext shared region. Used for I/O — host can read inputs and write outputs.
- **Unassigned**: not part of the Realm. Reads/writes generate exceptions.

`arch/aarch64/src/cca/granule.rs` exposes `transition_to_shared(ipa, count)` and `transition_to_private(ipa, count)` wrapping `RSI_IPA_STATE_SET`. Higher-level types (`SealedRegion`, `SharedWindow`) wrap the transitions in RAII.

The transition is *destructive*: moving a page from private → shared zeroes the page. Moving shared → private zeroes the page. This is hardware-enforced — there's no way to "leak" data across the transition by accident. The Rust API enforces it via type-state: a `SealedRegion` becomes a `SharedWindow` only by consuming the `SealedRegion` and explicitly zeroizing.

### Confidential ONNX model loader

```
[Model Owner]                    [Cloud / Untrusted Host]            [SmallAIOS Realm]
    │                                     │                                 │
    │ 1. Encrypt model.onnx with K_model   │                                 │
    │ 2. Ship encrypted blob + AAD         │                                 │
    │────────────────────────────────────► │                                 │
    │                                     │  3. Host hands blob to Realm    │
    │                                     │     via shared-memory window    │
    │                                     │ ──────────────────────────────► │
    │                                     │                                 │  4. Realm requests
    │                                     │                                 │     attestation token
    │                                     │                                 │     (claims = K_model
    │                                     │                                 │     wrap params, nonce)
    │ ◄─────────────────────────────────────────────────────────────────────│
    │ 5. Model Owner verifies attestation │                                 │
    │    matches expected Realm identity, │                                 │
    │    expected SmallAIOS measurement,  │                                 │
    │    expected nonce                   │                                 │
    │                                     │                                 │
    │ 6. Releases K_model wrapped to      │                                 │
    │    Realm's attest-key-pair-wrap-pub │                                 │
    │──────────────────────────────────────────────────────────────────────►│
    │                                     │                                 │  7. Realm unwraps
    │                                     │                                 │     K_model, decrypts
    │                                     │                                 │     model bytes into
    │                                     │                                 │     private memory.
```

The flow is the standard "attested key release" pattern: model owner only releases the decryption key after verifying the Realm is the certified SmallAIOS running on a certified RMM. The verification piggybacks on `remote-attestation-v1`'s verifier crate, extended with a `--release-key-to-attested-target` subcommand in Phase 1.

### Realm attestation token (RAT)

A CCA RAT is a CBOR-encoded structure with two sub-tokens:

- **Platform token**: signed by a CPU-resident attestation key, attests to the platform's CCA configuration and the RMM identity.
- **Realm token**: signed by an RMM-derived key, attests to the Realm's measurement (initial code hash + extended measurements like model hashes).

Together they form a chain rooted at a CPU-vendor key publicly announced by Arm (the "CCA Platform Attestation Key" — Arm publishes the root of trust per silicon generation). SmallAIOS wraps the RAT in the existing `HybridQuote` envelope from `remote-attestation-v1`, adding the SmallAIOS counter-signature as the PQC half.

The verifier crate gains:

- A `cca-platform-roots/` directory under `trust-anchors/` for the Arm-published CCA Platform Attestation Key roots per silicon generation.
- RAT-specific verification logic — CBOR parse, signature chain check, measurement extraction.

### CI strategy

`qemu-system-aarch64 -cpu max,rme=on -machine virt,gic-version=3 -bios tf-a.bin -kernel smallaios-cca-realm` boots a Realm under tf-rmm. The CI job:

1. Builds tf-rmm from the upstream `tf-rmm` repository (pinned commit).
2. Builds SmallAIOS with `--features cca-realm`.
3. Boots the combined image under QEMU.
4. Runs `attest-verifier verify --backend cca-realm` against the booted Realm.
5. Asserts PASS.

This is software-only verification — QEMU emulates the RME extension and tf-rmm runs as a software RMM. It does *not* validate hardware-side correctness, only software-side correctness. Hardware verification follows when CCA-capable silicon is generally available; the CI gate is documented as covering only the software side until then.

## Threat model captures (cross-phase)

### Adversary classes (recurring)

The same six adversary classes from the proposal's sketch table apply across all three phases. Phase 1 produces the full threat model document; Phases 2 and 3 add columns:

| Adversary | CCA defense | TDX defense | SEV-SNP defense |
|-----------|-------------|-------------|-----------------|
| Co-tenant VM (Spectre-class) | RME boundary + Arm branch-history isolation | TDX page-key separation + Intel branch-history clearing on TD-entry | SEV-SNP page-state metadata + AMD's IBPB on entry |
| Hypervisor (passive read) | RMM-mediated enclave entry — hypervisor sees ciphertext | TDX-module-mediated — same | SEV firmware-mediated — same |
| Datacenter operator (physical) | Memory encryption (per-Realm key) | Memory encryption (per-TD key) | Memory encryption (per-VM key) |
| Compromised firmware | None (firmware trusted) | None (firmware trusted) | None (firmware trusted) |
| Compromised CPU | None | None | None |
| Side-channel timing | App-level only (constant-time ONNX) | Same | Same |

### Residual risks (documented honestly)

- **Speculative execution gadgets**. All three platforms have published-and-patched gadgets historically. SmallAIOS's mitigation is to keep the in-enclave codebase small (the unikernel surface is much smaller than a Linux guest), and to follow each platform's prescribed entry/exit barrier sequence. Documented as residual.
- **Memory-encryption replay attacks**. CCA, TDX, and SEV-SNP all use AES-XTS or similar with per-page tweaks; replay attacks at the physical-DRAM layer are mitigated by anti-replay metadata. Documented.
- **DMA from untrusted I/O devices** (pre-TDISP). Phase 1-3 don't address PCIe TEE-IO. DMA-capable peripherals can read enclave memory if the platform doesn't enforce IOMMU separation. Documented as a known limitation; deployment guidance is "treat all I/O as crossing the trust boundary".

## Build / CI surface (Phase 1)

- New: `arch/aarch64/src/cca/{mod.rs, granule.rs, rsi_ids.rs}`.
- New: `security/src/confidential/{mod.rs, cca_backend.rs}`.
- New: `onnx-rt/src/confidential.rs`.
- New: `arch/aarch64/linker-cca-realm.ld`.
- New Cargo feature: `cca-realm` on `smallaios-arch-aarch64` + `smallaios-onnx-rt`.
- New `[[bin]]`: `smallaios-cca-realm` (in `arch/aarch64/Cargo.toml`).
- New CI job: `cca-realm-qemu-smoke` (advisory at land).
- New `tools/attest-verifier/` extensions: CCA RAT parsing, `--release-key-to-attested-target` subcommand.
- New docs: `docs/cca-realm-deployment.md`, `docs/confidential-compute-threat-model.md`, `docs/cca-attestation-key-release.md`.

## What this change does NOT do

- Does not modify the non-confidential SmallAIOS build paths. Standard `tegra234`, `tegra-x1`, x86-64, RISC-V builds are unchanged. Confidential builds are additive.
- Does not require any non-confidential deployment to enable any new feature flag. The default-features build is unchanged.
- Does not commit to a specific cloud-vendor confidential-VM service. The change targets the open standards (CCA, TDX, SNP); cloud-vendor specifics are deployment notes only.
- Does not bring up any specific TPM-or-equivalent inside the enclave. Each platform's attestation primitive replaces a TPM for confidential-compute purposes.
- Does not bring confidential GPU compute. CPU-side ONNX only in Phase 1-3. Confidential GPU is a future change.
