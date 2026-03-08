# Delta for Tamper-Evident Audit

## ADDED Requirements

### Requirement: Signed Audit Log Batches
The kernel audit subsystem SHALL sign log batches using ML-DSA-65 with a hash chain linking consecutive batches.

#### Scenario: Sign a completed audit batch
- WHEN the audit subsystem seals a batch of log entries
- THEN the subsystem MUST produce an ML-DSA-65 signature over the batch content
- AND the signature MUST be verifiable with the corresponding public key
- AND the batch MUST include a hash chain reference linking it to the previous batch

#### Scenario: Hash chain continuity across batches
- WHEN a new batch is sealed after a previous batch exists
- THEN the new batch MUST include the SHA-3-256 hash of the previous batch
- AND an auditor MUST be able to walk the hash chain from any batch back to the genesis batch
- AND any gap or inconsistency in the chain MUST be detectable

### Requirement: Audit Log Entry Structure
Each audit log entry SHALL include: timestamp (nanosecond), event type, task ID, resource reference, operation, result (success/failure), and capability ID used.

#### Scenario: Record a capability grant event
- WHEN a capability is granted to a task
- THEN the audit entry MUST contain a nanosecond-precision timestamp
- AND the entry MUST contain the event type set to capability grant
- AND the entry MUST contain the task ID of the receiving task
- AND the entry MUST contain the resource reference for the granted resource
- AND the entry MUST contain the operation (grant)
- AND the entry MUST contain the result (success or failure)
- AND the entry MUST contain the capability ID that was granted

#### Scenario: Record an inference request event
- WHEN an inference request is submitted
- THEN the audit entry MUST contain all required fields: timestamp, event type, task ID, resource reference, operation, result, and capability ID
- AND no field MUST be omitted or left as a default sentinel value

### Requirement: Batch Sealing Policy
The audit subsystem SHALL seal a batch when either 256 entries accumulate or 1 second elapses, whichever comes first.

#### Scenario: Seal batch on entry count threshold
- WHEN the audit subsystem accumulates 256 log entries in the current batch
- THEN the subsystem MUST immediately seal the batch
- AND MUST begin a new batch for subsequent entries

#### Scenario: Seal batch on time threshold
- WHEN 1 second elapses since the current batch was opened and fewer than 256 entries have accumulated
- THEN the subsystem MUST seal the batch with whatever entries have accumulated
- AND MUST begin a new batch with a fresh 1-second timer

#### Scenario: Empty batch suppression
- WHEN 1 second elapses and zero entries have accumulated
- THEN the subsystem MUST NOT produce an empty sealed batch
- AND MUST reset the timer for the next interval

### Requirement: Batch Seal Computation
The batch seal SHALL compute SHA-3-256(previous_batch_hash || serialized_entries) and sign with ML-DSA-65.

#### Scenario: Compute batch digest
- WHEN the audit subsystem seals a batch
- THEN the subsystem MUST serialize all entries in the batch into a deterministic byte representation
- AND MUST compute SHA-3-256 over the concatenation of the previous batch hash and the serialized entries
- AND MUST sign the resulting digest with ML-DSA-65

#### Scenario: First batch has no predecessor
- WHEN the first batch is sealed after system boot
- THEN the previous batch hash MUST be set to a well-known genesis value (32 zero bytes)
- AND the batch MUST be signed following the same procedure as subsequent batches

### Requirement: Audit Export via Zenoh IPC
Signed audit batches SHALL be exported via Zenoh IPC on key expression `smallaios/v1/audit`.

#### Scenario: Publish sealed batch to Zenoh
- WHEN a batch is sealed and signed
- THEN the audit subsystem MUST publish the signed batch (entries + signature + batch hash) on Zenoh key expression `smallaios/v1/audit`
- AND a Zenoh subscriber on that key expression MUST receive the complete signed batch

#### Scenario: Subscriber receives batches in order
- WHEN a Zenoh subscriber is connected to `smallaios/v1/audit`
- THEN the subscriber MUST receive batches in the same order they were sealed
- AND the hash chain MUST be verifiable from the received sequence of batches

### Requirement: Structured Event Taxonomy
The audit subsystem SHALL define a structured event taxonomy with categories: capability (grant, revoke, deny), authentication (TLS handshake success/failure), resource (allocation, exhaustion), inference (request, completion, timeout), system (boot, shutdown, watchdog).

#### Scenario: Classify a capability denial event
- WHEN a task is denied a capability
- THEN the audit entry MUST be classified under category capability with subcategory deny
- AND the event type field MUST encode both the category and subcategory

#### Scenario: Classify a TLS handshake failure
- WHEN a TLS 1.3 handshake fails
- THEN the audit entry MUST be classified under category authentication with subcategory failure
- AND the entry MUST include the peer address and failure reason

#### Scenario: Classify a system boot event
- WHEN the system completes boot initialization
- THEN the audit subsystem MUST emit an entry classified under category system with subcategory boot
- AND this MUST be the first entry in the first audit batch

#### Scenario: Classify an inference timeout event
- WHEN an inference request exceeds its deadline
- THEN the audit entry MUST be classified under category inference with subcategory timeout
- AND the entry MUST include the task ID and the elapsed time

#### Scenario: Classify a resource exhaustion event
- WHEN a memory allocation fails due to resource exhaustion
- THEN the audit entry MUST be classified under category resource with subcategory exhaustion
- AND the entry MUST include the requested allocation size and the resource type

### Requirement: Audit Log Integrity Verification
Audit log integrity SHALL be verifiable: given a batch and its signature, any party with the public key can verify the batch was not tampered with.

#### Scenario: Verify an unmodified batch
- WHEN an auditor receives a signed batch and the ML-DSA-65 public key
- THEN the auditor MUST be able to verify the signature against the batch content
- AND verification MUST succeed for an unmodified batch

#### Scenario: Detect a tampered batch
- WHEN a signed batch is modified after signing (even by one bit)
- THEN signature verification MUST fail
- AND the auditor MUST be able to determine that the batch was tampered with

#### Scenario: Verify hash chain integrity across batches
- WHEN an auditor receives a sequence of signed batches
- THEN the auditor MUST be able to recompute each batch hash and verify it matches the previous_batch_hash field in the next batch
- AND any break in the chain MUST be detectable

### Requirement: Configurable Log Retention Policy
Log retention policy SHALL be configurable: minimum 7 days for edge deployments, 90 days for datacenter, 1 year for safety-critical.

#### Scenario: Edge deployment retention
- WHEN the system is configured for edge deployment
- THEN the audit subsystem MUST retain signed batches for a minimum of 7 days
- AND batches older than the configured retention period MAY be pruned

#### Scenario: Datacenter deployment retention
- WHEN the system is configured for datacenter deployment
- THEN the audit subsystem MUST retain signed batches for a minimum of 90 days

#### Scenario: Safety-critical deployment retention
- WHEN the system is configured for safety-critical deployment
- THEN the audit subsystem MUST retain signed batches for a minimum of 1 year
- AND pruning MUST NOT occur until the retention period has elapsed

#### Scenario: Custom retention override
- WHEN an operator specifies a custom retention period in the system configuration
- THEN the audit subsystem MUST use the custom period provided it meets or exceeds the minimum for the deployment class

### Requirement: Non-Blocking Audit Logging
Audit logging SHALL NOT block the inference path: signing occurs asynchronously after batch accumulation.

#### Scenario: Inference proceeds without waiting for batch signing
- WHEN an audit log entry is generated during inference processing
- THEN the entry MUST be appended to the current batch without blocking the inference task
- AND the ML-DSA-65 signing operation MUST occur asynchronously after batch accumulation
- AND the inference task MUST NOT wait for the signing operation to complete

#### Scenario: Signing latency does not affect inference latency
- WHEN the audit subsystem is under high load with frequent batch seals
- THEN the p99 inference latency MUST NOT increase by more than 1% compared to audit-disabled operation
- AND the signing workload MUST be scheduled at a lower priority than inference tasks
