## ADDED Requirements

### Requirement: Raw binary I/O mode for bulk transfers

The UART driver (`peripheral/src/uart.rs`) SHALL provide a raw binary I/O mode with no terminal cooking, as required by the YMODEM-1K receiver: all 256 byte values — including control bytes such as `SOH`, `STX`, `ACK`, `NAK`, `EOT`, and `0x00` — SHALL pass verbatim in both directions, with no echo, no line buffering, and no line-discipline interpretation. Entering and leaving raw binary mode SHALL be explicit, and the TTY SHALL return to its normal cooked behavior when the transfer ends or aborts.

#### Scenario: Control bytes pass verbatim

- **WHEN** the YMODEM receiver reads a 1024-byte data block in raw binary mode whose payload contains `0x00`, `0x03` (^C), and `0x04` (^D) bytes
- **THEN** every byte SHALL be delivered to the caller unmodified and uninterpreted
- **AND** no byte SHALL be echoed back to the terminal

#### Scenario: Cooked behavior restored after the transfer

- **WHEN** a YMODEM session completes or aborts and raw binary mode is exited
- **THEN** subsequent console reads SHALL exhibit the normal echo and line-discipline behavior
- **AND** the raw mode SHALL leave no lingering driver state
