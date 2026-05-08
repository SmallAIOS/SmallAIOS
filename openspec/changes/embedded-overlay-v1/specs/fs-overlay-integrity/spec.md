## ADDED Requirements

### Requirement: SHA-3-256 fingerprint sidecar
Every successful `model_add` SHALL produce a sidecar file `<name>.sha3` in the upper layer containing the hex-encoded SHA-3-256 of the model file's bytes. The sidecar SHALL be written atomically alongside the model file (staged + renamed in the same `model_add` transaction). Reads from `/models/` SHALL hash-verify the upper-layer file against the sidecar before bytes flow into ONNX runtime; mismatch SHALL fail closed with `-EIO`.

#### Scenario: Sidecar produced on add
- **WHEN** `model_add("foo.onnx", fd)` succeeds with content of length N
- **THEN** `/data/models-upper/foo.onnx.sha3` SHALL contain the hex-encoded SHA-3-256 of those N bytes

#### Scenario: Mismatch fails closed
- **WHEN** `/data/models-upper/foo.onnx` is corrupted post-add (single byte flipped)
- **AND** ONNX runtime calls `read("/models/foo.onnx", ...)`
- **THEN** the read SHALL return `-EIO` after the hash check
- **AND** zero bytes of corrupted content SHALL be visible to ONNX runtime
- **AND** an audit record `model_hash_mismatch` SHALL be appended

#### Scenario: Missing sidecar fails closed
- **WHEN** an upper-layer model file exists without the corresponding `.sha3` sidecar
- **THEN** reads SHALL return `-EIO` with reason `missing fingerprint sidecar`
- **AND** an audit record SHALL be appended

### Requirement: Optional ML-DSA-65 signature sidecar
When `fs.overlay.require_signed = true`, every `model_load` from the upper layer SHALL find and verify a `<name>.sig` sidecar containing an ML-DSA-65 signature over the file's SHA-3-256 fingerprint, signed by the SmallAIOS model-signing key. Missing sidecar SHALL return `-EAUTH`. Invalid signature SHALL return `-EAUTH` and append an audit record. When `fs.overlay.require_signed = false` (default), the sidecar SHALL be ignored (verification still runs if a sig is present, but is non-fatal on mismatch).

#### Scenario: Required-signed missing rejected
- **WHEN** `fs.overlay.require_signed = true` and `/data/models-upper/foo.onnx.sig` is absent
- **AND** ONNX runtime calls `model_load("/models/foo.onnx")`
- **THEN** the load SHALL fail with `-EAUTH`
- **AND** an audit record `model_load_unsigned` SHALL be appended

#### Scenario: Required-signed invalid rejected
- **WHEN** `fs.overlay.require_signed = true` and the sig sidecar fails ML-DSA-65 verification
- **THEN** the load SHALL fail with `-EAUTH`
- **AND** an audit record `model_signature_invalid` SHALL be appended

#### Scenario: Default-off allows unsigned
- **WHEN** `fs.overlay.require_signed = false` and a model has no sig sidecar
- **THEN** the load SHALL succeed (subject to fingerprint verify)

#### Scenario: Policy flip rejects existing unsigned
- **WHEN** an unsigned model exists in the upper
- **AND** an operator flips `fs.overlay.require_signed` to `true`
- **THEN** subsequent `model_load` of that model SHALL fail with `-EAUTH`
- **AND** the file remains in place — re-uploading with a sig restores access

### Requirement: Lower-layer integrity unchanged
Files read from the lower (squashfs) layer SHALL continue to use `embedded-filesystem-v1`'s per-block SHA-3-256 manifest verification. The overlay SHALL NOT add any second-layer hash check on top of the lower; the manifest is sufficient. The merged read path SHALL apply the appropriate verification per-source.

#### Scenario: Lower file uses block-level manifest
- **WHEN** a read targets a lower-only file
- **THEN** verification SHALL be the per-block SHA-3-256 manifest check from `embedded-filesystem-v1`
- **AND** the overlay SHALL NOT add a second check

#### Scenario: Upper file uses sidecar fingerprint
- **WHEN** a read targets an upper-layer file
- **THEN** verification SHALL be the file-level SHA-3-256 sidecar check
- **AND** the lower's manifest SHALL NOT be consulted for this file
