## ADDED Requirements

### Requirement: Operator-controlled trust store path

The trust store path SHALL be configured via the `tls.trust_store_path` key in `/data/audit_export/immudb.toml`. The path SHALL point to a PEM-encoded bundle file containing one or more X.509 certificates. The loader SHALL refuse to start the exporter when the path is empty AND `[exporter] enabled = true`.

#### Scenario: Empty trust_store_path with enabled exporter rejected
- **WHEN** `[exporter] enabled = true` AND `tls.trust_store_path = ""`
- **THEN** `Config::validate` SHALL return `ConfigError::TrustStoreRequired`

#### Scenario: Disabled exporter ignores empty trust store
- **WHEN** `[exporter] enabled = false` AND `tls.trust_store_path = ""`
- **THEN** `Config::validate` SHALL accept the config

### Requirement: PEM bundle loader

The trust-store loader SHALL parse a PEM-encoded file containing zero or more `-----BEGIN CERTIFICATE-----` / `-----END CERTIFICATE-----` blocks. Each block SHALL be base64-decoded into a DER cert which SHALL then pass the X.509v3 parser used by chain verification. The loader SHALL refuse a bundle file that contains:

- Zero certificate blocks (an empty bundle never anchors any chain).
- A certificate whose `BasicConstraints.CA` is `false` (only CA certs may anchor).
- Two distinct certificates with the same `Subject` (ambiguous trust anchor).

The loader SHALL accept a bundle file containing one or more valid CA certs.

#### Scenario: Empty bundle rejected
- **WHEN** the PEM file contains no CERTIFICATE blocks
- **THEN** the loader SHALL return an error
- **AND** the exporter SHALL refuse to start

#### Scenario: Non-CA cert in bundle rejected
- **WHEN** a PEM block decodes to a cert with `BasicConstraints.CA = false`
- **THEN** the loader SHALL return an error

#### Scenario: Bundle with two valid CAs accepted
- **WHEN** the PEM file contains two distinct CA certs with different Subjects
- **THEN** the loader SHALL accept both

### Requirement: Optional per-CA fingerprint pinning

The TOML SHALL accept an optional `tls.trust_store_pin` field carrying the lowercase-hex SHA-256 fingerprint of the *root* certificate of the trust chain the operator expects. When set, the verifier SHALL refuse any leaf whose chain does NOT anchor at a root with that exact fingerprint, even if a different root in the bundle would otherwise validate the chain.

#### Scenario: Pinned fingerprint mismatch rejected
- **WHEN** `tls.trust_store_pin = "<sha256 of root A>"` AND the server's chain anchors at root B
- **THEN** the verifier SHALL emit `TlsClientError::ChainUntrusted`

#### Scenario: Pinned fingerprint match accepted
- **WHEN** `tls.trust_store_pin = "<sha256 of root A>"` AND the server's chain anchors at root A
- **THEN** the verifier SHALL accept

#### Scenario: No pin → any bundle root is acceptable
- **WHEN** `tls.trust_store_pin` is unset AND the bundle contains roots A and B AND the server's chain anchors at B
- **THEN** the verifier SHALL accept
