## ADDED Requirements

### Requirement: No cleartext logging of cryptographic key material
The QUIC packet protection module SHALL NOT log, print, or debug-format any cryptographic key, IV, nonce, or header protection key bytes. This applies to both production and test builds.

#### Scenario: PacketProtectionKeys construction does not log keys
- **WHEN** `PacketProtectionKeys::new()` is called with key, IV, and HP key parameters
- **THEN** no log output, debug print, or format string SHALL contain the raw byte values of those parameters

#### Scenario: No sensitive material in debug trait output
- **WHEN** a `PacketProtectionKeys` value is formatted via `Debug` or `Display`
- **THEN** the output SHALL NOT contain raw key bytes (redacted output or derived-Debug with opaque fields is acceptable)

### Requirement: Cryptographic material redaction in error messages
Error messages and panic messages in cryptographic modules SHALL NOT include raw key material. Only lengths, algorithm names, or error codes are acceptable.

#### Scenario: AEAD error does not leak key
- **WHEN** `aead_encrypt` or `aead_decrypt` returns an error
- **THEN** the error value SHALL NOT contain any portion of the key, IV, or plaintext
