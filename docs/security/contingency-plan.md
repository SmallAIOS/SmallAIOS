# SmallAIOS Contingency Plan

**Document Version:** 1.0
**Date:** 2026-02-10
**Classification:** Internal

---

## 1. Recovery Time Objectives (RTO)

| Deployment Class | RTO | Rationale |
|-----------------|-----|-----------|
| Datacenter | 30 seconds | Container restart + model reload from local cache/OCI registry |
| Edge | 5 seconds | Fast boot with pre-loaded model state from local storage |
| Safety-Critical | 100 ms (watchdog-bounded) | Hardware watchdog reset; fail-safe state before resume |

### RTO Measurement
- **Start:** Moment of failure detection (watchdog timeout, health check failure, crash signal)
- **End:** Moment system reports ready status (health probe passes, inference serving resumes)
- **Conditions:** Measured under realistic load representative of production workloads

---

## 2. Recovery Point Objectives (RPO)

| Data Type | RPO | Mechanism |
|-----------|-----|-----------|
| Configuration state | Zero data loss | Persisted to durable storage before being applied (local disk, ConfigMap/Secret) |
| Audit logs | Zero acknowledged loss | Flushed to durable storage or transmitted to remote syslog before acknowledgment |
| Model state | Recoverable | OCI registry stores all model versions; integrity verified on reload (SHA-256) |
| Inference state | Ephemeral | In-flight inference requests may be lost; clients must implement retry logic |

---

## 3. Automatic Recovery Mechanisms

### 3.1 Bare-Metal Watchdog Recovery

**Trigger:** Hardware watchdog timer expires (kernel failed to service within timeout).

**Sequence:**
1. Watchdog triggers hardware reset (GPIO/ACPI)
2. Bootloader reloads kernel image from persistent storage
3. Kernel detects watchdog-triggered reset (reads reset cause register)
4. Kernel logs previous-boot crash event to audit log
5. Configuration restored from persistent storage
6. ONNX model reloaded from local storage
7. System reports operational status

**Timeout Configuration:**
- Safety-critical: 100 ms (default)
- Edge: 1 second (default)
- Datacenter: 5 seconds (default)
- Service interval: <= half the configured timeout

### 3.2 Container Restart Recovery (Docker/Podman)

**Trigger:** Container process exits (crash, OOM kill, explicit termination).

**Sequence:**
1. Container runtime detects exit status
2. Restart policy triggers new container instance (`restart: unless-stopped` or `always`)
3. Container mounts persistent configuration volume
4. Model pulled from OCI registry or loaded from local cache (layer cache preferred)
5. Health check passes (`/healthz` endpoint returns 200)
6. Container reports ready

**Configuration:**
- Max restart attempts: 5 within 5 minutes (exponential backoff)
- If exceeded: alert operator via monitoring; container remains stopped

### 3.3 Kubernetes Pod Rescheduling

**Trigger:** Pod fails liveness probe or node failure detected.

**Sequence:**
1. Kubelet reports pod failure to API server
2. Controller manager marks pod for rescheduling
3. Scheduler selects new node meeting: resource requirements, affinity rules, topology constraints
4. Container image pulled on new node (or served from node cache)
5. Configuration loaded from ConfigMap/Secret
6. Model loaded from OCI registry (or node-local PersistentVolume)
7. Readiness probe passes
8. Service endpoints updated to include new pod

**Expected Timing:**
- Image cached on node: 5-10 seconds
- Image pull required: 30-60 seconds (depends on image size and network)
- Model reload: 5-15 seconds (depends on model size)

---

## 4. Failover Procedures

### 4.1 Bare-Metal Failover

| Step | Action | Expected Duration | If Step Fails |
|------|--------|-------------------|---------------|
| 1 | Watchdog triggers hardware reset | < 1 ms | Hardware fault; manual power cycle |
| 2 | Bootloader loads kernel image | 10-50 ms | Bootloader fallback partition; manual re-flash |
| 3 | Kernel detects crash recovery mode | < 5 ms | Clean boot if crash flag unreadable |
| 4 | Configuration restored from persistent storage | 5-10 ms | Boot with compiled-in defaults |
| 5 | Model reloaded from local storage | 10-500 ms (size-dependent) | Alert operator; operate in no-model mode |
| 6 | System reports operational | < 1 ms | If health check fails, re-enter recovery |

### 4.2 Container Failover

| Step | Action | Expected Duration | If Step Fails |
|------|--------|-------------------|---------------|
| 1 | Runtime detects container exit | < 100 ms | Runtime monitoring/health check |
| 2 | Restart policy creates new instance | < 500 ms | Alert after max restarts |
| 3 | Mount persistent config volume | < 100 ms | Fallback to environment variables |
| 4 | Pull/cache model from OCI registry | 1-30 s | Use local cache; alert if unavailable |
| 5 | Health check passes | < 1 s | Restart again; count toward max |
| 6 | Container reports ready | < 100 ms | Mark unhealthy; alert operator |

### 4.3 Kubernetes Pod Rescheduling

| Step | Action | Expected Duration | If Step Fails |
|------|--------|-------------------|---------------|
| 1 | Kubelet reports pod failure | < 5 s | Node-level kubelet monitoring |
| 2 | Scheduler selects new node | < 2 s | No schedulable nodes: pod Pending; alert |
| 3 | Image pulled (or from cache) | 5-60 s | ImagePullBackOff: check registry access |
| 4 | ConfigMap/Secret mounted | < 1 s | Pod fails to start; check RBAC |
| 5 | Model loaded from registry/PV | 5-15 s | Init container retry; alert |
| 6 | Readiness probe passes | < 5 s | Pod not added to service; CrashLoopBackOff |
| 7 | Service endpoints updated | < 2 s | Endpoint controller reconciliation |

### 4.4 Failover Notification

When any failover procedure executes:
1. Audit log entry generated: `system_boot` event with `recovery=true` flag
2. Zenoh event published: `smallaios/v1/incidents` with severity=Medium, type=failover
3. Prometheus counter incremented: `smallaios_recovery_total{mode="<deployment>"}`
4. Notification delivered within 5 seconds of recovery completion

---

## 5. Model and Configuration Backup/Restore

### 5.1 Model Backup
- All ONNX models stored in OCI registry with immutable tags
- Each model version tagged with SHA-256 content hash
- Registry replication across availability zones (datacenter deployments)
- Local model cache on persistent storage (edge deployments)

### 5.2 Model Restore
1. Identify target model version (OCI tag or content hash)
2. Pull model from registry: `oci pull <registry>/<model>:<tag>`
3. Verify integrity: SHA-256 hash match
4. Verify signature: ML-DSA-65 signature check against trusted public key
5. Load into ONNX runtime
6. Run inference sanity check (if test inputs configured)

### 5.3 Configuration Backup
- All configuration stored in version control (git)
- Kubernetes: ConfigMap and Secret objects versioned via GitOps
- Bare-metal: Configuration partition on persistent storage (read-only mount)

### 5.4 Configuration Restore
1. Identify target configuration version (git tag or ConfigMap revision)
2. Apply configuration via deployment pipeline or manual restore
3. Restart affected services to apply
4. Verify system health and configuration correctness

---

## 6. Recovery Testing Plan

### 6.1 Testing Cadence
- Quarterly testing for each active deployment mode
- Testing aligned with calendar quarters (Q1: Jan-Mar, Q2: Apr-Jun, Q3: Jul-Sep, Q4: Oct-Dec)

### 6.2 Test Scenarios

| Scenario | Deployment Mode | Simulated Failure | Success Criteria |
|----------|----------------|-------------------|-----------------|
| Watchdog reset | Bare-metal | Block watchdog service; wait for timeout | System recovers within RTO; audit log contains crash event |
| Container crash | Docker/Podman | `docker kill <container>` | Container restarts within RTO; health check passes |
| Pod eviction | Kubernetes | `kubectl delete pod <name>` | New pod scheduled and ready within RTO |
| Node failure | Kubernetes | Cordon and drain node | Pod rescheduled to different node within RTO |
| Model corruption | All | Replace model file with invalid data | System detects corruption; loads previous version or reports error |
| Config rollback | All | Apply invalid configuration | System detects invalid config or operator rolls back; previous config restored |

### 6.3 Test Documentation

Each test execution produces a report containing:
- Test date and executor
- Deployment mode tested
- Failure scenario simulated
- Actual recovery time measured
- Comparison against RTO target (pass/fail)
- Deviations or issues discovered
- Corrective actions assigned (if any)

### 6.4 RTO Validation
- If measured recovery time exceeds RTO: corrective action opened in POA&M
- Trend analysis across quarters to detect degradation

### 6.5 Retention
- Test documentation retained for at least 3 years for audit purposes
- Stored in `docs/security/recovery-tests/` directory (git-tracked)

### 6.6 Reporting
- Summary report presented to security steering committee at next scheduled meeting
- Report includes: pass/fail per deployment mode, trend analysis, open corrective actions
