# Delta for Continuous Monitoring

## ADDED Requirements

### Requirement: Capability Denial Rate Tracking
The system SHALL track capability denial rate per second per task and generate an alert when rate exceeds configurable threshold (default: 10 denials/second).

#### Scenario: Alert on excessive capability denials
- WHEN a single task accumulates more than 10 capability denials within a 1-second window
- THEN the monitoring subsystem MUST generate a security alert
- AND the alert MUST identify the task ID, the denial count, and the 1-second window in which the denials occurred

#### Scenario: No alert below threshold
- WHEN a task accumulates fewer than the configured threshold of capability denials within a 1-second window
- THEN the monitoring subsystem MUST NOT generate an alert for that task in that window

#### Scenario: Custom denial rate threshold
- WHEN an operator configures a custom capability denial rate threshold at boot time
- THEN the monitoring subsystem MUST use the custom threshold instead of the default 10 denials/second

### Requirement: Memory Allocation Failure Rate Tracking
The system SHALL track memory allocation failure rate and alert when failures exceed configurable threshold.

#### Scenario: Alert on allocation failure spike
- WHEN the memory allocation failure rate exceeds the configured threshold within a measurement window
- THEN the monitoring subsystem MUST generate an alert
- AND the alert MUST include the failure count, the measurement window, and the allocator region (buddy, slab, or tensor)

#### Scenario: Track failures per allocator region
- WHEN memory allocation failures occur
- THEN the monitoring subsystem MUST attribute each failure to the specific allocator region (buddy, slab, or tensor)
- AND MUST maintain per-region failure counters

### Requirement: Inference Latency Anomaly Detection
The system SHALL compute rolling inference latency statistics (p50, p99, p999) and alert when any request exceeds 3-sigma deviation from rolling p50.

#### Scenario: Alert on latency outlier
- WHEN an inference request completes with latency exceeding 3 standard deviations above the rolling p50
- THEN the monitoring subsystem MUST generate a latency anomaly alert
- AND the alert MUST include the observed latency, the rolling p50, the rolling standard deviation, and the task ID

#### Scenario: Compute rolling statistics
- WHEN inference requests complete
- THEN the monitoring subsystem MUST maintain rolling p50, p99, and p999 latency statistics
- AND the rolling window MUST contain at least the most recent 1000 observations or 60 seconds of data, whichever is larger

#### Scenario: No false alert during warmup
- WHEN fewer than 100 inference requests have completed since boot
- THEN the monitoring subsystem MUST NOT generate latency anomaly alerts
- AND MUST accumulate baseline statistics until the minimum sample size is reached

### Requirement: Watchdog Time Remaining Tracking
The system SHALL track watchdog time remaining and alert when remaining time drops below 50% of timeout period.

#### Scenario: Alert on watchdog time pressure
- WHEN the watchdog timer remaining time drops below 50% of the configured timeout period
- THEN the monitoring subsystem MUST generate a watchdog warning alert
- AND the alert MUST include the remaining time, the timeout period, and the task currently executing

#### Scenario: Clear alert after watchdog refresh
- WHEN the watchdog timer is refreshed and remaining time returns above 50% of the timeout period
- THEN the monitoring subsystem MUST clear the watchdog warning condition
- AND MUST NOT generate repeated alerts for the same watchdog cycle once cleared

### Requirement: Network Connection Rate Tracking
The system SHALL track network connection rate and alert on suspected SYN flood (>100 new connections/second).

#### Scenario: Alert on SYN flood detection
- WHEN the rate of new network connections exceeds 100 per second
- THEN the monitoring subsystem MUST generate a SYN flood alert
- AND the alert MUST include the connection rate, the source address distribution, and the measurement window

#### Scenario: Distinguish legitimate traffic bursts
- WHEN the connection rate exceeds 100 per second but all connections complete the TLS handshake within normal parameters
- THEN the monitoring subsystem MUST still generate the alert
- AND the alert MUST include the handshake completion rate to assist operator triage

### Requirement: Security Metrics Export
Security metrics SHALL be exported via Prometheus endpoint (`GET /metrics`) and Zenoh IPC (`smallaios/v1/metrics`).

#### Scenario: Export metrics via Prometheus endpoint
- WHEN an HTTP client sends a GET request to the `/metrics` endpoint
- THEN the system MUST respond with metrics in Prometheus exposition format
- AND the response MUST include capability denial rate, memory allocation failure count, inference latency percentiles (p50, p99, p999), watchdog time remaining, and network connection rate

#### Scenario: Export metrics via Zenoh IPC
- WHEN the monitoring subsystem updates security metrics
- THEN the subsystem MUST publish the updated metrics on Zenoh key expression `smallaios/v1/metrics`
- AND a Zenoh subscriber MUST receive the metrics in a structured serialization format

#### Scenario: Metrics consistency between export channels
- WHEN metrics are exported via both Prometheus and Zenoh within the same collection interval
- THEN the metric values MUST be consistent between the two channels for the same measurement window

### Requirement: Configurable Alert Thresholds
All alert thresholds SHALL be configurable at boot time via system configuration.

#### Scenario: Override default thresholds at boot
- WHEN the system boots with a configuration file specifying custom alert thresholds
- THEN the monitoring subsystem MUST apply the configured thresholds for capability denial rate, memory allocation failure rate, inference latency deviation, watchdog warning percentage, and network connection rate

#### Scenario: Fall back to defaults when not configured
- WHEN the system boots without explicit threshold configuration for a specific metric
- THEN the monitoring subsystem MUST use the documented default threshold for that metric

#### Scenario: Reject invalid threshold values
- WHEN the system configuration specifies a threshold value that is negative or zero
- THEN the monitoring subsystem MUST reject the configuration
- AND MUST log a configuration error and fall back to the default threshold

### Requirement: Statistical Anomaly Detection Methods
Anomaly detection SHALL use statistical methods (moving average, standard deviation) and SHALL NOT use machine learning models.

#### Scenario: Use moving average for baseline computation
- WHEN the monitoring subsystem computes a baseline for any metric
- THEN it MUST use a moving average over a configurable window
- AND MUST NOT use neural networks, decision trees, or any machine learning model

#### Scenario: Use standard deviation for outlier detection
- WHEN the monitoring subsystem determines whether a metric value is anomalous
- THEN it MUST compute the standard deviation over the rolling window
- AND MUST flag values exceeding the configured sigma threshold as anomalous

#### Scenario: Verify no ML model dependency
- WHEN the monitoring subsystem is built
- THEN the build MUST NOT link against any machine learning inference library
- AND the anomaly detection code MUST be implementable using only arithmetic operations on scalar values

### Requirement: Automated Vulnerability Scanning in CI
The CI pipeline SHALL run cargo-audit automatically and report results as part of continuous vulnerability assessment.

#### Scenario: Run cargo-audit on every CI build
- WHEN the CI pipeline executes for a commit or pull request
- THEN cargo-audit MUST be run against the current Cargo.lock
- AND the pipeline MUST report all known vulnerabilities found in dependencies

#### Scenario: Fail CI on critical vulnerabilities
- WHEN cargo-audit detects a vulnerability with severity Critical or High
- THEN the CI pipeline MUST fail the build
- AND the failure report MUST include the advisory ID, affected crate, and recommended remediation

#### Scenario: Track vulnerability history
- WHEN cargo-audit runs in CI
- THEN the results MUST be stored as a build artifact
- AND historical vulnerability data MUST be available for trend analysis across builds
