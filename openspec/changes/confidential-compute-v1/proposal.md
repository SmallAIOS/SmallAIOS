# confidential-compute-v1

## Summary

Confidential AI inference is the SmallAIOS deployment model where **(a)** model weights are licensed intellectual property and the model owner does not want them readable by the host infrastructure provider, and **(b)** customer inputs to inference are confidential and the customer does not want them readable by either the model owner or the infrastructure provider. Today SmallAIOS provides neither: model weights live as plaintext bytes in DRAM and on disk; customer tensor data is plaintext in DRAM during inference. A privileged host (hypervisor, fabric admin, datacenter operator) with sufficient access can read both.

This change targets that gap via **hardware-enforced memory confidentiality** on three platform paths, sequenced as a multi-quarter, multi-phase effort. Phase 1 (~6-8 weeks) is design and threat-model work plus a Tier-1 ARM CCA (Confidential Compute Architecture, Realm Management Extension) targeted bring-up. Phase 2 (~8-10 weeks) extends to Intel TDX (Trust Domain Extensions). Phase 3 (~8-10 weeks) extends to AMD SEV-SNP. Each phase produces an independently-deployable confidential SmallAIOS variant on its target platform.

ARM CCA is Phase 1 because:

- SmallAIOS already has heavy AArch64 investment (`unikernel-orin-bringup-v1` on Tegra Orin, `boot-root-of-trust-v1` Phase 2 TF-A integration, `op-tee-bridge-v1`, `remote-attestation-v1` PSA-IA backend). ARM CCA's RME extends the TrustZone trust model SmallAIOS will already understand by the time this change starts.
- Production silicon ships in 2026 on Neoverse N3 / V3 datacenter cores and Cortex-X / A-series for client. SmallAIOS is positioned for a clean "production-silicon, day-1 software" story on the platform.
- The Realm Management Monitor (RMM) is open source (Arm releases reference RMM as the `tf-rmm` project under TF-A), so SmallAIOS can develop and CI-test against a real RMM implementation in QEMU `-cpu max,rme=on` without proprietary firmware dependencies.

Intel TDX and AMD SEV-SNP follow as Phase 2 and Phase 3 because (a) they target datacenter x86-64, which is the highest-volume confidential-compute deployment surface; (b) both are production-shipping in 2026 (Sapphire Rapids + TDX, Genoa + SNP); (c) the SmallAIOS-side abstractions defined in Phase 1 (`SealedRegion`, `AttestableEnclave`, `ConfidentialOnnxRuntime`) transfer to either x86-64 platform with platform-specific glue.

Capability: `security-confidential-compute` (new). Cargo features: `cca-realm` (AArch64, RME-based confidential VM), `intel-tdx`, `amd-sev-snp` (x86-64). Each is mutually exclusive with the others at build time (a CCA-realm kernel doesn't link the TDX driver, etc.).

## Why

- **AI weights are the deployment IP of 2026.** Model owners (foundation model providers, fine-tuned domain experts) are increasingly unwilling to deploy weights to infrastructure they don't fully control. The current options are (a) self-host the weights yourself, expensive at scale, or (b) trust the infrastructure operator, which doesn't work for high-value IP. Hardware confidential compute is the third option: weights are decrypted only inside an attestable, memory-encrypted enclave, never visible to the hypervisor or host OS. SmallAIOS is purpose-built for AI inference and is therefore the natural unikernel to package as a confidential workload.
- **Customer data is the deployment IP of every other workload.** A financial-services customer sending tensor inputs to an inference endpoint doesn't want the cloud operator (or another tenant on the same physical host) reading those tensors. Confidential compute is the *only* technology that gives customers cryptographic proof their data is protected even from the cloud operator.
- **The three platforms diverge in detail but converge in shape.** ARM CCA, Intel TDX, and AMD SEV-SNP all give you: (1) hardware-encrypted memory pages tagged with a per-enclave key, (2) a small trusted firmware component (RMM / TDX module / SEV firmware) that mediates enclave entry/exit, (3) hardware-rooted attestation reporting the enclave's identity to a remote verifier. SmallAIOS designs the abstraction layer in Phase 1 (CCA-anchored) so Phases 2 and 3 are platform glue, not redesign.
- **Aligns with the existing PQC and attestation work.** Confidential compute attestation reports are signed by hardware keys (classical). SmallAIOS's PQC-default stance means the reports SHALL be carry-counter-signable with ML-DSA-65 + Ed25519 hybrid — extending the `remote-attestation-v1` `HybridQuote` format to add a `ConfidentialEvidence` field for the platform-specific report. The same verifier crate consumes confidential-compute evidence transparently.
- **DO-178C DAL A becomes more tractable when the certified workload is isolated.** A confidential SmallAIOS enclave with a sharply-defined boundary (the enclave's measured pages) is easier to certify than a workload that shares DRAM with arbitrary co-tenants. The enclave's measurement IS the certification artifact: anything inside is certified; anything outside is irrelevant to the certification scope.
- **Phase 1 is independently valuable.** Even without Phases 2 and 3, ARM CCA support on production Neoverse silicon is a deployable product. CCA datacenter platforms (NVIDIA Grace, Ampere Altra Max successors, AWS Graviton 4 / 5 — once ARM CCA-capable cores reach those platforms) are SmallAIOS's clearest fit. Phases 2 and 3 widen the platform matrix but don't gate Phase 1's value.

## Threat model (Phase 1 design output)

The threat model is the largest single design artifact this change produces, because it determines what each phase's enclave actually defends against. Captured in `docs/confidential-compute-threat-model.md` (new), reviewed independently from the implementation work. Sketch:

| Adversary | Capability | Phase 1 (CCA) defense | Notes |
|-----------|------------|---------------------|-------|
| Co-tenant VM | Read DRAM via cache side channels (Spectre-class) | Memory encryption + Arm's branch-history isolation (`CSV2`, `BHB` invalidation on enclave-entry) | Speculative-execution gadgets remain a residual risk; mitigations are best-effort. |
| Host hypervisor | Read enclave DRAM | Hardware encryption (per-realm key) — hypervisor sees ciphertext | Strong. RMM-mediated enclave entry/exit prevents register exfiltration. |
| Datacenter operator with physical access (cold-boot, bus probing) | Read enclave DRAM via physical attack | Memory encryption (per-realm key) | Strong against cold-boot. Bus probing requires DRAM-controller compromise; out of scope. |
| Compromised RMM | Read or modify enclave state | None — RMM is trusted | RMM is open source (tf-rmm), verifiable by inspection. |
| Compromised CPU microcode | Anything | None — CPU is trusted | Mitigated by vendor signing of microcode; out of scope. |
| Side-channel via timing of inference | Infer input properties from inference time | Out of scope — application-level mitigations only (constant-time ONNX ops in `onnx-rt`) | Documented as a separate concern. |

Phase 2 (TDX) and Phase 3 (SEV-SNP) re-evaluate against the same adversary list — different mitigations, similar coverage. Each phase's deliverable includes its specific threat-model entry in the matrix.

## Phase 1 — ARM CCA Realm bring-up

### Build target

A new build target `aarch64-unknown-none-realm` is a synthetic name in the SmallAIOS workspace mapping to `aarch64-unknown-none` plus a `cca-realm` Cargo feature plus a CCA-specific linker script. The artifact is a Realm-compatible flat binary loaded by the Realm Management Monitor (`tf-rmm`) at Realm-mode entry.

### Realm boot flow

```
[Host VM (untrusted Linux KVM)] requests Realm creation
       │
       │  RMI_REALM_CREATE  (Realm Management Interface)
       ▼
[RMM at EL3] allocates Realm structures, measures Realm image
       │
       ▼
[RMM] sets initial Realm Personalization Value (RPV)
       │
       ▼
[SmallAIOS Realm at R-EL1] starts execution in encrypted memory
       │
       ▼
[SmallAIOS] runs ONNX inference; model weights and tensors live in
encrypted DRAM. Inference traffic crosses the Realm boundary only
through encrypted shared memory windows ("Granule Inhibit" pages).
```

The Realm sees memory as if it were a normal VM. The hypervisor (host KVM) cannot read Realm memory — even though KVM scheduled the Realm and provisioned its initial state. RMM enforces all of this; SmallAIOS just runs.

### What ships in Phase 1

- `arch/aarch64/src/cca/mod.rs` — Realm-side RMM interface client. Implements the small subset of RMI calls SmallAIOS issues from inside the Realm (RSI: Realm Services Interface): `RSI_ATTEST_TOKEN_INIT`, `RSI_ATTEST_TOKEN_CONTINUE`, `RSI_HOST_CALL`, `RSI_REALM_CONFIG`, `RSI_IPA_STATE_*` (page state).
- `arch/aarch64/src/cca/granule.rs` — Granule (4 KB Realm page) state management. SmallAIOS asks RMM to mark pages as `RAM` (private to Realm) or `SHARED` (visible to host, used for I/O).
- `security/src/confidential/mod.rs` — generic abstraction layer (`SealedRegion`, `AttestableEnclave`). Platform-specific backends plug in here; Phase 1 provides the CCA backend.
- `security/src/confidential/cca_backend.rs` — CCA-specific attestation report fetching via `RSI_ATTEST_TOKEN_*`. Report shape: a CBOR-encoded Realm Attestation Token (RAT) per Arm's CCA attestation specification. Integrates with `remote-attestation-v1` so the existing attestation surface returns a CCA report when the kernel runs as a Realm.
- `onnx-rt/src/confidential.rs` — encrypted-model-weights loading. Model bytes arrive in shared (host-visible) memory, are decrypted into Realm-private memory using a key the model owner provisioned via attested key release.
- Documentation: `docs/cca-realm-deployment.md`, `docs/confidential-compute-threat-model.md`, `docs/cca-attestation-key-release.md`.
- CI: `cca-realm-qemu-smoke` job using `qemu-system-aarch64 -cpu max,rme=on -machine virt,gic-version=3` plus tf-rmm as the RMM payload. Boots SmallAIOS as a Realm, runs an attestation round-trip, asserts PASS.

### Phase 1 success criterion

A SmallAIOS Realm boots under QEMU+tf-rmm, returns an attestation report via `remote-attestation-v1`'s extended surface (`Backend: "cca-realm"`), and the verifier validates the report against tf-rmm's reference attestation public key. Model weights load into Realm-private DRAM via an attested-key-release pattern; a smoke ONNX inference (vector-add) succeeds with inputs/outputs flowing through shared-memory windows.

## Phase 2 — Intel TDX

Intel TDX (Trust Domain Extensions, Sapphire Rapids+) provides a similar shape with Intel-specific glue. The TDX module mediates TD (Trust Domain) creation; SmallAIOS in a TD calls `tdcall(TDG.VP.VMCALL)` for host I/O and `tdcall(TDG.MR.REPORT)` for attestation. Phase 2 ports the `SealedRegion` / `AttestableEnclave` abstractions to TDX-specific implementations and produces an `intel-tdx` Cargo feature.

The Phase 2 deliverable mirrors Phase 1: kernel patches, attestation backend, ONNX confidential loader, docs, CI. The threat-model column for TDX is filled in. Estimated 8-10 weeks (more than Phase 1 because we need to do x86-64 enclave boot, which is a different shape than the existing x86-64 SmallAIOS boot).

## Phase 3 — AMD SEV-SNP

AMD SEV-SNP (Secure Encrypted Virtualization — Secure Nested Paging, Genoa+) is the third datacenter confidential-compute platform. Architecturally closer to CCA in some ways (page-state metadata enforced by hardware, similar to CCA's granule states); operationally closer to TDX in others (x86-64 with a small TDX-module-like SEV firmware). Phase 3 adds the `amd-sev-snp` Cargo feature.

Estimated 8-10 weeks. The cross-platform `SealedRegion` abstraction designed in Phase 1 is the value-multiplier here: porting to a third platform is mostly platform-glue work.

## Out of scope (across all phases)

- **Confidential GPU compute.** NVIDIA's H100 / H200 / B100 expose confidential-compute modes (CC mode + UVM encryption), but SmallAIOS's GPU integration is currently CPU-side cuDNN-backed (`Dockerfile.jetson`). Confidential GPU is its own follow-up change (`confidential-gpu-v1`) after Phase 1 / Phase 2 land.
- **Confidential I/O (TDISP).** PCIe TEE-IO (TDISP) is too new (production silicon 2027+) and bundles NIC / NVMe vendors. Documented as a future-future story.
- **Side-channel hardening of ONNX operators.** Constant-time matmul, attention, and softmax are valuable but orthogonal — they belong with the inference runtime, not the confidential-compute boot path. Tracked separately as `onnx-constant-time-ops-v1` (future).
- **Trusted I/O between confidential enclaves.** Multi-enclave coordination is a research topic; for SmallAIOS, one inference workload per enclave is the deployment model.
- **Vendor cloud confidential services** (AWS Nitro Enclaves, Azure Confidential VMs, GCP Confidential Computing). SmallAIOS targets the *standards* (CCA, TDX, SNP); cloud-vendor wrappers can adapt by speaking those.
- **Pre-silicon / FPGA-emulated CCA.** Phase 1 CI uses QEMU `+rme=on` with tf-rmm; FPGA platforms (Arm's CCA Reference Platform on the Juno SoC FPGA) are valuable for hardware-team validation but not for SmallAIOS-side CI.

## Sequencing

| Phase | Scope | Estimate | Depends on |
|-------|-------|----------|------------|
| Design + threat model | `docs/confidential-compute-threat-model.md`, abstraction layer design, build-target wiring | 2-3 weeks (within Phase 1) | None |
| 1: ARM CCA | Realm boot, RMM interface, attestation backend, ONNX confidential loader, QEMU+tf-rmm CI | 4-5 weeks (after design) | `op-tee-bridge-v1` (for SMC-related infrastructure), `remote-attestation-v1` (for the protocol surface to extend) |
| 2: Intel TDX | TD boot, tdcall driver, attestation backend, ONNX integration, CI | 8-10 weeks | Phase 1 abstractions |
| 3: AMD SEV-SNP | SNP enclave boot, GHCB driver, attestation backend, CI | 8-10 weeks | Phase 1 abstractions |
| **Total** | | **~12-16 weeks for Phase 1 alone; 28-36 weeks for all three** | |

Phases 2 and 3 are independent of each other and can land in parallel after Phase 1.

The 12-16 week range for Phase 1 reflects the unknown of "production CCA silicon availability for end-to-end verification on real hardware". QEMU + tf-rmm covers software-side CI; hardware-side verification slips into when CCA silicon ships in volume. The change ships against software-only verification first and adds hardware verification as silicon arrives.

## DO-178C alignment

Confidential compute is the strongest **isolation** claim DAL A can make. The certified kernel runs inside a hardware-enclosed boundary; any DAL A objective relating to "freedom from interference" by co-running workloads is structurally satisfied. The attestation report is the auditor's proof that a given inference result was produced inside the enclave.

Specific claims unlocked per phase:

- **Phase 1**: "The Realm executing SmallAIOS for tenant T on ARM CCA platform P is bit-identical to certified release X.Y.Z, isolated from co-tenant interference by the Realm Management Monitor." Evidence: Realm Attestation Token signed by the RMM's attestation key, plus the SmallAIOS counter-signature.
- **Phase 2**: same claim, TDX platform, TD Report signed by Intel's TDX attestation chain.
- **Phase 3**: same claim, SEV-SNP platform, SNP attestation report signed by AMD's SEV firmware.

## PQC stance

Each phase's attestation report integrates with `remote-attestation-v1`'s `HybridQuote` envelope: the platform-specific report (RAT, TD Report, SNP Report) is the inner hardware signature, and the SmallAIOS counter-signature (ML-DSA-65 + Ed25519) is the outer PQC half. Verifiers can require PQC, classical, or both per existing `PqcMode` semantics. No new protocol surface is needed; this change extends the existing one.
