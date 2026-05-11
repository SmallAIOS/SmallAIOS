# Tasks — ecc-scrubbing-v1

## 0. Reference reading + hardware verification

- [ ] 0.1 Read Tegra234 TRM section "External Memory Controller (EMC) — ECC and Scrubbing". Document the `EMC_ECC_SCRUB_*`, `EMC_ECC_STATUS`, `EMC_ECC_CONTROL` register fields, scrub modes (patrol vs. demand), and completion-polling semantics.
- [ ] 0.2 On an Orin NX 16 GB host, confirm ECC is configured: read `cat /proc/meminfo | grep -i ecc`, `dmesg | grep -i 'ecc\|emc'`. Paste in PR description.
- [ ] 0.3 Inspect L4T's `tegra-mc` driver source for register-access patterns and any undocumented-field hints.
- [ ] 0.4 Identify a representative LLM workload for the post-merge soak test (e.g., `Qwen 3 8B instruct` quantized, continuous inference). Document the workload definition in `docs/ecc-scrubbing.md`.

## 1. Phase 1 — Scrub service core

### 1a. Module scaffolding

- [ ] 1.1 Create `kernel/src/mem/scrub/mod.rs` with the public API: `pub fn init(config: ScrubConfig)`, `pub fn add_region(name, base, size, interval)`, `pub fn pause(name)`, `pub fn resume(name)`, `pub fn cursor(name) -> Position`, `pub fn stats(name) -> ScrubStats`.
- [ ] 1.2 Create `kernel/src/mem/scrub/config.rs` with `ScrubConfig`, `Region`, `Interval` types. Support a TOML-loaded form (`scrub::Config::from_toml(&str)`) and a programmatic builder form.
- [ ] 1.3 Create `kernel/src/mem/scrub/stats.rs` with `ScrubStats { cycles_completed, last_cycle_duration, correctable_errors, uncorrectable_errors, cursor_position, advanced_at }`. All counters atomic so telemetry can read concurrently.
- [ ] 1.4 Add `ecc-scrub` Cargo feature on `smallaios-kernel`. Off by default. Doc-comment notes that hardware-accelerated scrub requires platform-specific backend features.

### 1b. Scrub task

- [ ] 1.5 Create `kernel/src/mem/scrub/task.rs` with the async task entry point: a loop that walks each registered region's interval timer, picks the next region due, drives one cycle via the active backend, updates stats, yields between chunks.
- [ ] 1.6 Implement chunked iteration: configurable `chunk_size` (default 64 KiB), yield-now after each chunk to keep the cooperative scheduler responsive.
- [ ] 1.7 Wire `scrub::init` into `kernel_main` after `mem::init` completes — register default regions (heap, ONNX weight region when known, kernel `.bss` / `.data`).
- [ ] 1.8 Unit tests: fake backend (in-memory counter), drive a 1 MB region, assert correct cursor advance, correct cycle counting, correct yield behavior.

## 2. Phase 2 — Tegra234 EMC backend

- [ ] 2.1 Create `arch/aarch64/src/scrub/tegra_emc.rs` implementing the `ScrubBackend` trait via Tegra234's EMC.
- [ ] 2.2 Implement `probe()` that walks the DTB for the EMC node (`compatible = "nvidia,tegra234-emc"`), reads the MMIO base, returns `Some(TegraEmcBackend)` on success.
- [ ] 2.3 Implement `scrub_region`: program `EMC_ECC_SCRUB_REGION_LO/HI` for the region descriptor, set `EMC_ECC_SCRUB_CONTROL.SCRUB_EN`, poll `EMC_ECC_SCRUB_STATUS.DONE` (with cooperative yield between polls), read `EMC_ECC_STATUS` for the correctable / uncorrectable counts since last reset, return them.
- [ ] 2.4 Implement boot-time demand-mode wipe: configure `SCRUB_MODE = demand`, kick off, block-poll until done, log the baseline error counts. Verifies the EMC responds and establishes a known-good baseline.
- [ ] 2.5 Implement fallback detection: if `EMC_ECC_STATUS` reads back as `0xFFFF_FFFF` (a typical "register unimplemented" pattern) or otherwise fails sanity, return `None` from `probe` and fall through to the software backend with a `[scrub] EMC probe failed, falling back to software` log line.

## 3. Phase 3 — Software-fallback backend

- [ ] 3.1 Create `kernel/src/mem/scrub/sw_backend.rs` implementing the `ScrubBackend` trait via `core::ptr::read_volatile` / `write_volatile`.
- [ ] 3.2 Implement chunked walk: read every `usize` in a 4 KiB page, write the same value back. Yield between pages.
- [ ] 3.3 Surface correctable / uncorrectable counts via the DRAM controller's RAS interface where exposed (`ras_counters` API on x86 via PCIe config; on aarch64 via `ERR<n>FR_EL1` system registers if the silicon exposes the RAS extension). When unavailable, return `(0, 0)` and document that.
- [ ] 3.4 Unit-test the software backend against a known memory region with synthetic content; assert content unchanged after scrub.

## 4. Phase 4 — Watchdog integration

- [ ] 4.1 Add `scrub-watchdog-aggressive` and `scrub-watchdog-permissive` Cargo features (mutually exclusive, default = aggressive on `--features ecc-scrub`).
- [ ] 4.2 Modify the scrub task to call `watchdog::feed()` per loop iteration in permissive mode, and per cursor-advance in aggressive mode.
- [ ] 4.3 Add a watchdog reset reason code `WdReason::ScrubStall` (or extend existing reasons). On boot, if the kernel detects it woke from a `ScrubStall` reset, log it prominently.
- [ ] 4.4 Test: inject a scheduler-block scenario (a deliberately-uncooperative task) and confirm the watchdog fires within `watchdog_threshold` seconds.

## 5. Phase 5 — Telemetry + docs

- [ ] 5.1 Surface `ScrubStats` per region via the existing telemetry path (or, if `telemetry-otel-export-v1` hasn't landed, via a periodic boot-log line).
- [ ] 5.2 Create `docs/ecc-scrubbing.md` covering: what SEUs are and why they matter, the DO-178C DAL A motivation, supported platforms / backends, the configuration knobs (region, interval, chunk size, watchdog mode), recommended intervals for different operational environments (ground / cruise / orbital), failure modes, and the soak-test procedure.
- [ ] 5.3 Update `docs/architecture.md` to note the scrub service as a Layer 0 (Foundation) service.
- [ ] 5.4 Update `CLAUDE.md` "Current state" to note ECC scrubbing is available on `tegra234`-feature builds.

## 6. Verify

- [ ] 6.1 Run `openspec validate ecc-scrubbing-v1 --strict`.
- [ ] 6.2 Run the `just ecc-scrub-test` recipe locally under QEMU; assert clean exit.
- [ ] 6.3 On Orin NX hardware: boot with `--features tegra234,ecc-scrub`, observe one boot-time demand wipe, observe at least one patrol cycle, capture the telemetry counters. Paste in PR description.
- [ ] 6.4 Schedule the post-merge 7-day soak test as a follow-up activity (not gating, but tracked in the change archive note).

## 7. Archive

- [ ] 7.1 Open archive PR moving the change to `openspec/changes/archive/YYYY-MM-DD-ecc-scrubbing-v1` and sync the spec deltas to main specs. Include the soak-test pointer in the archive note.
