## ADDED Requirements

### Requirement: Session Configuration Is Validated Eagerly at Construction

`Session` construction SHALL validate its `SessionConfig` before
returning, rejecting invalid configurations with an error rather than
deferring the failure to first inference. `SessionConfig` SHALL provide
a `validate(&self) -> Result<(), SessionError>` method that is the
single validation authority and is **independent of any Cargo feature**
(in particular, it SHALL NOT be gated behind the `cuda` feature).
`Session::new` SHALL call `validate` and return
`Result<Self, SessionError>`, propagating the error. In v1 the validated
invariant is `StreamConfig::Overlap { transfer_streams }` with
`transfer_streams <= 2`; `validate` SHALL return
`SessionError::InvalidConfig` when it exceeds 2.

#### Scenario: Invalid transfer_streams is rejected at construction

- **WHEN** `Session::new` is called with `StreamConfig::Overlap {
  transfer_streams: 5 }`
- **THEN** it SHALL return `Err(SessionError::InvalidConfig(..))`
- **AND** no `Session` value SHALL be produced

#### Scenario: Validation is independent of the cuda feature

- **WHEN** `SessionConfig::validate` is called on a config with
  `transfer_streams > 2`
- **THEN** it SHALL return `Err(SessionError::InvalidConfig(..))`
  whether or not the `cuda` feature is enabled

#### Scenario: Valid configuration constructs successfully

- **WHEN** `Session::new` is called with `StreamConfig::SingleStream` or
  `StreamConfig::Overlap { transfer_streams: 2 }`
- **THEN** it SHALL return `Ok(session)`

#### Scenario: The stream-pool check remains as a backstop

- **WHEN** `ensure_stream_pool` runs on a session whose config passed
  construction validation
- **THEN** its `transfer_streams <= 2` check SHALL still be present
- **AND** SHALL not reject any config that construction accepted
