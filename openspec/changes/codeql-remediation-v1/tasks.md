## 1. CI Workflow Permissions

- [x] 1.1 Add `permissions:` block to Format Check job (`contents: read`)
- [x] 1.2 Add `permissions:` block to Clippy Lint job (`contents: read`)
- [x] 1.3 Add `permissions:` block to Unit Tests job (`contents: read`)
- [x] 1.4 Add `permissions:` block to all Build jobs (x86, AArch64, RISC-V, Tegra) (`contents: read`)
- [x] 1.5 Add `permissions:` block to RISC-V QEMU Smoke Test job (`contents: read`)
- [x] 1.6 Add `permissions:` block to Image Size Check job (`contents: read`)
- [x] 1.7 Add `permissions:` block to Docker Build job (`contents: read`)
- [x] 1.8 Add `permissions:` block to Code Coverage job (`contents: read`, `checks: write`)
- [x] 1.9 Add `permissions:` block to SonarCloud Analysis job (`contents: read`, `pull-requests: read`)
- [x] 1.10 Add `permissions:` block to TLA+ Verification job (`contents: read`)
- [x] 1.11 Add `permissions:` block to Change Gates job (`contents: read`)
- [x] 1.12 Add `permissions:` block to Semver PR Title Check job (`contents: read`, `pull-requests: read`)
- [x] 1.13 Add `permissions:` block to Verified Boot Test job (`contents: read`)

## 2. Release Workflow Permissions

- [x] 2.1 Add `permissions:` block to build-and-release job (`contents: write`)
- [x] 2.2 Add `permissions:` block to Docker publish job (`packages: write`, `contents: read`)

## 3. Remove Cleartext Key Logging

- [x] 3.1 Suppress cleartext-logging false positives at `net/src/quic/protection.rs` lines 67-69 (struct field assignments, not logging)
- [x] 3.2 Verify no other debug/log statements in QUIC protection output raw key material

## 4. Suppress False-Positive Crypto Alerts

- [x] 4.1 Add `lgtm` suppression comment on `sample_uniform(&rho, ...)` in `ml_dsa.rs` KeyGen (line ~1069)
- [x] 4.2 Add `lgtm` suppression comment on `sample_uniform(rho, ...)` in `ml_dsa.rs` sign helper (line ~1155)
- [x] 4.3 Add `lgtm` suppression comment on `sample_mask(&rho_pp, ...)` in `ml_dsa.rs` sign loop (line ~1349)
- [x] 4.4 Add `lgtm` suppression comment on `sample_uniform(&rho, ...)` in `ml_dsa.rs` verify (line ~1476)
- [x] 4.5 Add `lgtm` suppression comment on `prf(random_coins, ...)` in `ml_kem.rs` encrypt (line ~803)
- [x] 4.6 Add `lgtm` suppression comments on test fixture keys in `protection.rs` `test_keys()` (line ~225) and `wrong_keys` (line ~278)

## 5. Enable CodeQL as Merge Gate

- [ ] 5.1 Verify zero open CodeQL alerts after all fixes are merged
- [ ] 5.2 Add CodeQL as a required status check on branch protection for `main` and `develop`
