# Verifiable Audit Log — Operator Setup Guide

Status: prototype (`verifiable-audit-log-v1`).
This guide assumes you have a SmallAIOS image built with
`--features audit-export` and an immudb server running off-box.

## What you get

Every audit record produced on a SmallAIOS box is appended to
the local SHA-3-256 hash chain (`mgmt-audit-log` does this
unconditionally) and, when the exporter is enabled, also
shipped to an external immudb instance as a `verifiableSet`.
The immudb server signs every transaction; the SmallAIOS
client verifies the signature against a fingerprint you
pinned, and persists the new state to
`/data/audit_export/last_state.bin`.

Two integrity fingerprints become visible on every box:

- **Local**: SHA-3-256 chain head of the audit ring.
- **Remote**: SHA-256 signed state from immudb.

Both are published on `smallaios/metrics/audit_fingerprint`
and surfaced in the console-monitor `top` header line.

## Threat model in one paragraph

An attacker with on-box read/write access can tamper with
`/data/audit/log.jsonl`, but the local SHA-3 chain detects
that. They cannot retroactively change what immudb already
holds — every record shipped pre-tampering remains
SHA-256-Merkle-anchored in the off-box ledger. The signed
state response is verified against a pubkey **fingerprint
pinned in the TOML**, so an attacker who substitutes a
different immudb server cannot pass verification. Historic
CVEs in immudb's first-party SDKs were verifier bugs; the
SmallAIOS verifier returns a hard error on every mismatch
class and refuses to advance silently.

## Prerequisites

1. An immudb server, version 1.11.x recommended.
   The vendored proto in `audit-export/vendor/schema.proto`
   pins this protocol version (commit
   `f07d3ac01c068e3d6e760afaaf1f1db20b36d0bc`,
   tag `v1.11.0`).
2. A bearer token for the database. Create it on the
   immudb side:
   ```
   immuadmin user create <username> readwrite smallaios_audit
   ```
3. The Ed25519 public key the server uses to sign state.
   Find it under your immudb data directory; compute its
   SHA-256 fingerprint:
   ```
   sha256sum /path/to/immudb/server.pubkey
   ```
   The hex digest goes into `tls.server_pubkey_fingerprint`.

## Step 1 — Drop the token on the box

```
ssh root@orin-01
cat > /data/audit_export/immudb.token <<<'<the-bearer-token>'
chmod 0600 /data/audit_export/immudb.token
```

The loader refuses any mode laxer than 0600. The file MUST
NOT have a trailing newline; redirect with `printf` if your
shell adds one.

## Step 2 — Configure the TOML

Copy `examples/immudb.toml` to `/data/audit_export/immudb.toml`:

```toml
[exporter]
enabled  = true
endpoint = "https://immudb.example.com:3322"
database = "smallaios_audit"

[tls]
require_pqc = false
server_pubkey_fingerprint = "<64 hex chars of SHA-256(pubkey)>"
```

The validator rejects:

- `enabled = true && endpoint = ""`
- `enabled = true && server_pubkey_fingerprint = ""`
- `endpoint` not starting with `https://`
- Fingerprint that isn't 64 hex chars
- `buffer_bytes` outside [1 MiB, 64 MiB]
- `batch_size` outside [1, 10000]
- `backoff_cap_ms < backoff_initial_ms`
- `auth_mode = "mtls"` (v2 deferred — D3 in design.md)

## Step 3 — Verify it's running

On any TTY logged in as Viewer or above:

```
audit-export status
```

Should report `enabled=true endpoint=... last=<timestamp>`.

The `top` header now carries one of:

- `IMMUDB off` — exporter disabled
- `IMMUDB pending` — enabled, no successful export yet
- `IMMUDB ok tx=NNN last 4s` — healthy
- `IMMUDB stale tx=NNN last 120s` — enabled but not making
  progress (network outage, transient endpoint failure)
- `IMMUDB HALT proof_failure` — verifier rejected a proof.
  Pipeline is stopped; operator must investigate.

## Step 4 — Verify off-box with `immuclient`

The canonical off-box verifier is immudb's own CLI:

```
immuclient -d smallaios_audit audit
```

If `immuclient audit` reports zero divergence, the chain is
intact. If it reports tampering, both SmallAIOS's local
audit ring AND the immudb tree disagree with each other —
investigate immediately.

## Halt-class failures: what they mean

| Verb in audit ring          | Reason                                                                  |
|-----------------------------|-------------------------------------------------------------------------|
| `audit_export_attempt`      | gRPC PERMISSION_DENIED (7) or UNAUTHENTICATED (16). Rotate the token.   |
| `audit_export_proof_failure`| Verifier rejected the server's response. Investigate immediately.       |
| `audit_export_rollback_suspected` | Server's `txId` regressed below our local value. Tampering signal.|
| `audit_export_decode_failure` | Server response didn't decode as protobuf. Likely a version mismatch. |

Halt-class failures stop the exporter. To recover:

1. Investigate the underlying cause.
2. Restart the SmallAIOS box (or, when implemented,
   `audit-export reset` as Root from the TTY shell).

## Rotating the token

1. Create a new token on the immudb side.
2. `chmod 0600 /data/audit_export/immudb.token.new`
3. Write the new bytes.
4. `mv -f /data/audit_export/immudb.token.new /data/audit_export/immudb.token`

The exporter re-reads the token on its next attempt; the
`config_write` audit record automatically redacts both the
old and new token bytes to `<redacted:N>`.

## Disabling the exporter

```toml
[exporter]
enabled = false
```

After save + apply, the exporter unregisters its tap on
the audit ring. Local audit logging continues unchanged.

## Removing the feature entirely

Build the container or unikernel image without
`--features audit-export`. Zero exporter code is linked,
zero size delta versus a build that never knew the feature
existed. See design.md D10 for the two-layer opt-in
matrix.

## See also

- `openspec/changes/verifiable-audit-log-v1/` — full design + specs.
- `docs/audit-export-ci.md` — CI fixture-replay + nightly E2E flow.
- `audit-export/vendor/IMMUDB_SCHEMA_SHA` — pinned upstream commit.
