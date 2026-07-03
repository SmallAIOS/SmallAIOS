## ADDED Requirements

### Requirement: Pattern B Project-Run Relay Documented as the Recommendation

The change SHALL document Pattern B — a project-run relay behind a public, unauthenticated, write-only ingest endpoint that re-stamps requests with the project's real backend credential and forwards them — as the recommended ingest pattern, together with its rationale: backend optionality (switching backends is a relay-config change, no client update), schema enforcement at the edge (reject over-schema requests, strip unexpected attributes, drop oversized payloads), cost control, and audit-friendliness. Patterns A (SaaS-with-DSN) and C (Grafana Faro) SHALL be recorded as viable fallbacks, and the shipped wire format SHALL be retargetable to them.

#### Scenario: Recommendation and rationale recorded

- **WHEN** a reviewer reads the change's pattern-decision documentation
- **THEN** Pattern B (project-run relay) SHALL be named as the recommendation
- **AND** the four rationale points — optionality, schema enforcement at the edge, cost control, audit-friendliness — SHALL be stated
- **AND** Patterns A and C SHALL be recorded as fallbacks

#### Scenario: No embedded shared secret

- **WHEN** a reviewer inspects the client-side usage-telemetry wiring
- **THEN** the ingest endpoint SHALL be public and write-only
- **AND** no shared secret or credential SHALL be embedded in the SmallAIOS binary for this channel

### Requirement: Deliberate Non-Decisions Recorded

The pattern decision SHALL explicitly record what it does not decide: which backend runs behind the relay, which hosting platform runs the relay, and the relay source code itself. Operating the relay is out of scope — the relay lives in a separate repository with separate review; this change ships only the client-side wiring. Server-side aggregation tooling, dashboards, and alerting are likewise out of scope.

#### Scenario: Non-decisions listed as out of scope

- **WHEN** a reviewer reads the pattern-decision documentation
- **THEN** the backend behind the relay, the relay hosting platform, and the relay source code SHALL each be listed as deliberate non-decisions out of this change's scope

#### Scenario: No relay code in this repository

- **WHEN** the change's implementation is reviewed
- **THEN** it SHALL contain only client-side wiring
- **AND** no relay deployment code SHALL be added to the SmallAIOS repository by this change

### Requirement: Build-Time Endpoint Constant

The usage-telemetry endpoint SHALL be a build-time constant `USAGE_TELEMETRY_ENDPOINT` defaulting to a project-controlled hostname (the relay URL, e.g. `https://usage.smallaios.invalid/v1/ingest` until the real relay DNS name is assigned). The relay does not need to exist in v1 — the URL SHALL be wired so that launching collection is a relay deployment plus the gated default-flip policy event, not a client release.

#### Scenario: Constant points at a project-controlled hostname

- **WHEN** a reviewer reads the `USAGE_TELEMETRY_ENDPOINT` definition
- **THEN** it SHALL be a compile-time constant
- **AND** its value SHALL be a project-controlled DNS name

#### Scenario: Flipping the switch requires no client release

- **WHEN** the project later deploys the relay at the baked-in hostname
- **THEN** already-shipped binaries with recorded consent SHALL be able to export without a client code change

### Requirement: One-Way Channel

The usage-telemetry channel SHALL be one-way, client to relay. No A/B experiment assignment, feature-flag delivery, or any other remote configuration of SmallAIOS SHALL travel back over this channel.

#### Scenario: No configuration back-channel

- **WHEN** a reviewer reads the usage-telemetry transport code
- **THEN** there SHALL be no code path that parses a relay response into SmallAIOS configuration, feature flags, or experiment assignments
