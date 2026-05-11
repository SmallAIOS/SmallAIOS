## ADDED Requirements

### Requirement: TLS 1.3 record framing

The `tls-client` crate SHALL implement the TLS 1.3 record layer per RFC 8446 § 5. Every inbound and outbound byte SHALL pass through a record whose header is exactly 5 bytes (`ContentType: u8`, `LegacyVersion: u16`, `Length: u16`). The `Length` field SHALL be enforced at parse time: `TLSPlaintext.length` MUST NOT exceed 16,384 (2^14); `TLSCiphertext.length` MUST NOT exceed 16,640 (2^14 + 256). Records that declare a length above the cap SHALL be rejected with `TlsClientError::BadRecord` BEFORE any allocation is sized to the declared length.

#### Scenario: Oversized inbound record rejected without allocation
- **WHEN** a record header declares `length = 30000`
- **THEN** the parser SHALL return `TlsClientError::BadRecord`
- **AND** the implementation SHALL NOT allocate a buffer larger than the cap

#### Scenario: Plaintext at the 16,384 boundary accepted
- **WHEN** a `TLSPlaintext` record declares `length = 16384`
- **THEN** the parser SHALL accept the record

### Requirement: AEAD-protected application records use the negotiated cipher suite

Once the handshake-traffic keys are derived, the record layer SHALL AEAD-encrypt every outbound record (other than the legacy `ChangeCipherSpec` echo permitted by RFC 8446 § 5.1) and AEAD-decrypt every inbound `application_data` record. The AEAD primitive SHALL be the one negotiated in the cipher suite: one of `TLS_AES_128_GCM_SHA256`, `TLS_AES_256_GCM_SHA384`, `TLS_CHACHA20_POLY1305_SHA256`. Authentication failure on a decrypt SHALL be reported as `TlsClientError::Aead`; the implementation SHALL NOT surface the plaintext.

#### Scenario: AEAD decrypt failure surfaces as TlsClientError::Aead
- **WHEN** an inbound record's authentication tag does not validate under the current decryption keys
- **THEN** the record layer SHALL return `TlsClientError::Aead`
- **AND** the connection SHALL be terminated

#### Scenario: AES-256-GCM negotiated; record sealed under SHA-384 key schedule
- **WHEN** the cipher suite negotiated is `TLS_AES_256_GCM_SHA384`
- **THEN** outbound records SHALL be sealed using AES-256-GCM
- **AND** the traffic keys SHALL be derived from the SHA-384 transcript hash

### Requirement: Legacy record version refused

After the ServerHello, the record layer SHALL reject any record whose `LegacyVersion` is not `0x0303` (TLS 1.2 — the on-the-wire compatibility value mandated for TLS 1.3) with `TlsClientError::Version`. Records carrying `0x0301` (TLS 1.0), `0x0302` (TLS 1.1), or `0x0300` (SSL 3.0) at any point SHALL trigger the same error.

#### Scenario: TLS 1.1 record rejected
- **WHEN** an inbound record carries `LegacyVersion = 0x0302`
- **THEN** the record layer SHALL return `TlsClientError::Version`
