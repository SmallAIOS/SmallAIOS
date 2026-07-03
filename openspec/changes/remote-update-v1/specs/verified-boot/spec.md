## ADDED Requirements

### Requirement: Signature Verification At Update-Commit Time

The existing `verified-boot` boot-time ML-DSA-65 signature check SHALL extend to in-field updates: the same key and the same algorithm SHALL be applied at update-commit time, before an image is written as bootable. The system SHALL NOT be able to run an unsigned image via the update path any more than via the boot path.

#### Scenario: Same key verifies at boot and at commit

- **WHEN** an image signed with the `verified-boot` ML-DSA-65 signing key is committed through any update transport
- **THEN** the update-commit verifier SHALL accept it using the same key and algorithm as the boot-time check
- **AND** no separate update-signing key SHALL exist

#### Scenario: Update path cannot smuggle an unsigned image

- **WHEN** an image whose ML-DSA-65 signature does not verify is submitted through the YMODEM or Zenoh transport
- **THEN** the commit SHALL be refused before any slot is marked bootable
- **AND** the boot pointer SHALL be untouched
- **AND** the previously active image SHALL boot unchanged on next reset
