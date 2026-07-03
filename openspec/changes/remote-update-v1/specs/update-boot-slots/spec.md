## ADDED Requirements

### Requirement: A/B On-Disk Image Slots

The on-disk layout SHALL provide two image slots, `slot_a` and `slot_b`, each sized for the unikernel with ~16 MB headroom. Updates SHALL always be flashed to the inactive slot; the active slot's bytes SHALL never be modified by an in-progress update. The doubled image storage (~16 MB → ~32 MB) is the accepted disk-footprint cost.

#### Scenario: Update writes only the inactive slot

- **WHEN** `slot_a` is the active slot and a verified image is committed
- **THEN** the payload SHALL be written to `slot_b`
- **AND** the bytes of `slot_a` SHALL be identical before and after the write

#### Scenario: Insufficient slot space aborts cleanly

- **WHEN** an image's `payload_len` exceeds the inactive slot's capacity
- **THEN** the update SHALL abort with an insufficient-slot-space failure
- **AND** the boot pointer SHALL be untouched

### Requirement: Boot-Pointer Record

The update pipeline SHALL NOT introduce a new boot-pointer format. The boot pointer SHALL be the existing `fs-ab-boot` double-buffered `BootConfigRecord` (dedicated GPT boot-config partition, SHA3-256 `record_hash`, monotonically increasing `generation`), extended by this change's `fs-ab-boot` delta with `tries_remaining: u8` and `manifest_hash: [u8; 32]`. The proposal's abstract fields SHALL map onto that record as follows: `active: A | B` → `active_slot`; `pending = Some(<slot>)` ⇔ `tentative = 1` (the pending slot is the record's `active_slot`) and `pending = None` ⇔ `tentative = 0`; `tries_remaining` → the new `tries_remaining` field (decremented on each boot attempt while `tentative = 1`; exhaustion triggers reversion to the prior slot); `manifest_hash` → the new `manifest_hash` field carrying the SHA3-256 hash over the active image's manifest. The boot loader (UEFI app on Orin and x86-EFI, board-specific on other bare-metal targets) SHALL read this record before each boot and select the slot it names.

#### Scenario: Boot loader selects the slot named by the record

- **WHEN** the boot loader starts and the highest-generation valid record reads `active_slot = B, tentative = 0`
- **THEN** the boot loader SHALL load and boot the image in `slot_b`

#### Scenario: Pending boot decrements the try counter

- **WHEN** the record reads `active_slot = B, tentative = 1, tries_remaining = 3` and the boot loader attempts a boot
- **THEN** the boot loader SHALL decrement `tries_remaining` to 2 — via the `fs-ab-boot` atomic write-to-inactive-record path — before handing off to the slot-B image

#### Scenario: Exhausted tries revert to the prior slot

- **WHEN** the record reads `active_slot = B, tentative = 1, tries_remaining = 0`
- **THEN** the boot loader SHALL fall back to the prior slot (`slot_a`)
- **AND** the pending update SHALL be considered failed

### Requirement: Corruption-Safe Boot-Pointer Writes

Because boot-pointer corruption is fatal, every boot-pointer write performed by the update pipeline SHALL go through the journaled-replace scheme `fs-ab-boot` already specifies: the new record is written to the inactive record buffer with `generation = max + 1`, integrity-protected by its SHA3-256 `record_hash`, and flushed to durable storage before it supersedes the old record — so a crash or power loss at any point leaves a valid record readable. The update pipeline SHALL NOT add a separate CRC or journaling mechanism for the boot pointer.

#### Scenario: Power loss mid-write leaves a valid record

- **WHEN** power is lost partway through a boot-pointer update
- **THEN** on next boot the boot loader SHALL find a record whose `record_hash` validates
- **AND** that record SHALL be either the complete old record or the complete new record — never a torn mixture

#### Scenario: Hash-invalid record is not trusted

- **WHEN** the boot loader reads a boot-pointer record buffer whose `record_hash` does not validate
- **THEN** the boot loader SHALL NOT act on its contents
- **AND** SHALL use the other, valid record buffer per `fs-ab-boot` (halting with the `fs-ab-boot` recovery message only when both buffers are invalid)

### Requirement: Failed Updates Never Touch The Boot Pointer

Every update failure mode occurring before commit — transfer CRC mismatch, payload-hash mismatch, signature failure, wrong arch, insufficient slot space, or an explicit abort — SHALL leave the boot-pointer record byte-for-byte unchanged, so an interrupted or rejected update can never affect the next boot.

#### Scenario: Signature failure leaves the pointer unchanged

- **WHEN** an image fails ML-DSA-65 verification after a complete transfer
- **THEN** the boot-pointer record SHALL be byte-identical to its pre-update state
- **AND** the next boot SHALL use the same slot as if no update had been attempted

#### Scenario: Aborted transfer leaves the pointer unchanged

- **WHEN** a transfer is aborted mid-stream (YMODEM cancel or Zenoh `abort`)
- **THEN** staged bytes SHALL be discarded
- **AND** the boot-pointer record SHALL be unchanged
