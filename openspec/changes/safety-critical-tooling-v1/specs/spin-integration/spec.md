## ADDED Requirements

### Requirement: SPIN/Promela models for liveness properties
The project SHALL maintain SPIN/Promela models in `formal/promela/` for verifying liveness properties (LTL) of concurrent protocols, complementing the existing TLA+ safety models.

#### Scenario: QUIC handshake liveness verified
- **WHEN** SPIN verifies the QUIC handshake Promela model
- **THEN** the LTL property "every ClientHello eventually receives a ServerHello or timeout" SHALL be satisfied

#### Scenario: IPC delivery guarantee verified
- **WHEN** SPIN verifies the IPC pub/sub Promela model
- **THEN** the LTL property "every published message is eventually delivered to all subscribers" SHALL be satisfied

### Requirement: SPIN verification in CI
SPIN models SHALL be verified in CI alongside TLA+ models, with a timeout per model.

#### Scenario: SPIN CI job runs all models
- **WHEN** CI runs the SPIN verification job
- **THEN** all Promela models in `formal/promela/` SHALL be compiled and verified

#### Scenario: SPIN model failure blocks PR
- **WHEN** a code change causes a SPIN model to find a counterexample
- **THEN** the SPIN CI job SHALL fail and report the trail (counterexample trace)
