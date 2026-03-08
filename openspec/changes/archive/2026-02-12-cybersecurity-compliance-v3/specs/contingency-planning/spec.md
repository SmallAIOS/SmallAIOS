# Delta for Contingency Planning

## ADDED Requirements

### Requirement: Recovery Time Objective Targets
The system SHALL define RTO (Recovery Time Objective) targets per deployment class: datacenter (30 seconds), edge (5 seconds), and safety-critical (watchdog-bounded, typically 100ms).

#### Scenario: Datacenter deployment RTO
- WHEN a SmallAIOS instance in a datacenter deployment experiences a failure (crash, hang, or resource exhaustion)
- THEN the system MUST recover to a fully operational state within 30 seconds
- AND recovery MUST include: process restart, model reload from local cache or OCI registry, IPC reconnection, and resumption of inference serving

#### Scenario: Edge deployment RTO
- WHEN a SmallAIOS instance in an edge deployment (e.g., Jetson Orin Nano, Jetson Nano, RPi 4/5) experiences a failure
- THEN the system MUST recover to a fully operational state within 5 seconds
- AND recovery MUST prioritize fast boot using pre-loaded model state from local storage to minimize reload time

#### Scenario: Safety-critical deployment RTO
- WHEN a SmallAIOS instance in a safety-critical deployment (avionics, automotive, industrial control) experiences a failure
- THEN the system MUST recover within the watchdog timer bound, typically 100ms
- AND recovery MUST transition through a documented fail-safe state before resuming normal operation
- AND the fail-safe state MUST be defined per deployment domain (e.g., safe error output for avionics, controlled stop for automotive)

#### Scenario: RTO measurement and verification
- WHEN RTO compliance is being verified
- THEN the recovery time MUST be measured from the moment of failure detection (watchdog timeout, health check failure, crash signal) to the moment the system reports ready status
- AND measurement MUST be performed under realistic load conditions representative of production workloads

### Requirement: Recovery Point Objective
The system SHALL define RPO (Recovery Point Objective): zero data loss for configuration state, and model state recoverable from OCI registry.

#### Scenario: Zero data loss for configuration state
- WHEN a system failure occurs
- THEN all configuration state (capability policies, scheduler parameters, network policies, crypto parameters) MUST be recoverable with zero data loss
- AND configuration state MUST be persisted to durable storage (local disk, network storage, or Kubernetes ConfigMap/Secret) before being applied

#### Scenario: Model state recoverable from OCI registry
- WHEN a system failure results in loss of the loaded ONNX model
- THEN the model MUST be recoverable from the OCI registry without manual intervention
- AND the recovery process MUST verify model integrity (SHA-256 hash check) before loading the recovered model into the inference runtime

#### Scenario: Audit log persistence
- WHEN a system failure occurs
- THEN all audit log entries generated before the failure MUST be preserved
- AND the audit log MUST be flushed to durable storage or transmitted to the remote syslog receiver before being acknowledged as persisted
- AND no acknowledged audit log entries SHALL be lost due to the failure

#### Scenario: Inference state is non-persistent
- WHEN a system failure occurs during active inference
- THEN in-flight inference requests MAY be lost (inference state is ephemeral)
- AND the system documentation MUST clearly state that inference requests are not guaranteed to survive a failure and clients MUST implement retry logic

### Requirement: Automatic Recovery via Watchdog
The system SHALL support automatic recovery via watchdog timer reset on bare-metal and pod restart in Kubernetes/K3s.

#### Scenario: Bare-metal watchdog timer recovery
- WHEN the SmallAIOS kernel fails to service the hardware watchdog timer within the configured timeout period
- THEN the watchdog MUST trigger a hardware reset of the system
- AND upon reboot, the kernel MUST detect the watchdog-triggered reset, log the event (previous boot crash), and resume normal operation

#### Scenario: Kubernetes pod restart recovery
- WHEN a SmallAIOS pod in Kubernetes or K3s fails its liveness probe
- THEN the Kubernetes kubelet MUST restart the pod according to the configured restart policy (Always)
- AND the restarted pod MUST reload its configuration from ConfigMap/Secret and model from OCI registry
- AND the pod MUST report ready status via the readiness probe before receiving traffic

#### Scenario: Container runtime restart recovery
- WHEN a SmallAIOS container instance crashes outside of Kubernetes (e.g., standalone Docker/Podman)
- THEN the container runtime MUST restart the container according to the configured restart policy (unless-stopped or always)
- AND the restarted container MUST follow the same recovery sequence as Kubernetes pod restart

#### Scenario: Watchdog timeout configuration
- WHEN the watchdog timer is configured for a deployment
- THEN the timeout period MUST be configurable per deployment class: safety-critical (100ms default), edge (1 second default), datacenter (5 seconds default)
- AND the kernel MUST service the watchdog at an interval no greater than half the configured timeout to prevent spurious resets

### Requirement: Failover Procedures per Deployment Mode
The system SHALL document failover procedures for each deployment mode: bare-metal watchdog reset, container restart, and Kubernetes pod rescheduling.

#### Scenario: Bare-metal failover procedure
- WHEN a bare-metal SmallAIOS instance fails
- THEN the documented failover procedure MUST specify: (1) watchdog triggers hardware reset, (2) bootloader reloads kernel image, (3) kernel detects crash recovery mode, (4) configuration is restored from persistent storage, (5) model is reloaded, (6) system reports operational status
- AND each step MUST include expected duration and failure handling if the step itself fails

#### Scenario: Container failover procedure
- WHEN a containerized SmallAIOS instance fails
- THEN the documented failover procedure MUST specify: (1) container runtime detects exit, (2) restart policy triggers new container instance, (3) container mounts persistent configuration volume, (4) model is pulled from OCI registry or loaded from local cache, (5) health check passes, (6) container reports ready
- AND the procedure MUST document the maximum number of restart attempts before alerting an operator

#### Scenario: Kubernetes pod rescheduling procedure
- WHEN a SmallAIOS pod fails and cannot be restarted on the same node
- THEN the documented failover procedure MUST specify: (1) kubelet reports pod failure, (2) scheduler selects a new node meeting resource requirements and affinity rules, (3) pod is scheduled on the new node, (4) container image is pulled (or served from cache), (5) configuration and model are loaded, (6) readiness probe passes, (7) service endpoints are updated
- AND the procedure MUST document expected pod rescheduling time and any data locality considerations

#### Scenario: Failover notification
- WHEN any failover procedure is executed (bare-metal, container, or Kubernetes)
- THEN the system MUST generate a notification (log entry, Zenoh event, or Prometheus alert) indicating the failover event, deployment mode, timestamp, and recovery status
- AND the notification MUST be delivered within 5 seconds of recovery completion

### Requirement: Quarterly Recovery Testing
Recovery procedures SHALL be tested quarterly with documented results per NIST CP-4.

#### Scenario: Quarterly test execution
- WHEN a calendar quarter boundary is reached
- THEN recovery procedures MUST be tested for each active deployment mode (bare-metal, container, Kubernetes) within that quarter
- AND the test MUST simulate realistic failure conditions (kernel crash, watchdog timeout, pod eviction, node failure)

#### Scenario: Test documentation requirements
- WHEN a recovery test is executed
- THEN the test results MUST be documented with: test date, test executor, deployment mode tested, failure scenario simulated, actual recovery time measured, comparison against RTO target, any deviations or issues discovered, and corrective actions assigned
- AND the test documentation MUST be retained for at least three years for audit purposes

#### Scenario: RTO target validation
- WHEN the quarterly recovery test measures actual recovery time
- THEN the measured recovery time MUST be compared against the defined RTO target for the deployment class
- AND if the measured recovery time exceeds the RTO target, a corrective action MUST be opened in the POA&M with a remediation plan and target date

#### Scenario: Test plan review and update
- WHEN recovery test results reveal gaps in the test plan (e.g., untested failure modes, new deployment configurations)
- THEN the recovery test plan MUST be updated to address the identified gaps before the next quarterly test
- AND the updated test plan MUST be reviewed and approved by the Security Lead

#### Scenario: Stakeholder reporting
- WHEN quarterly recovery testing is complete
- THEN a summary report MUST be presented to the security steering committee at the next scheduled meeting
- AND the report MUST include pass/fail status for each deployment mode, trend analysis against previous quarters, and any open corrective actions
