# Spec 13: Formal Verification

## Overview

SmallAIOS uses formal verification to provide machine-checked proofs of critical
system properties. Three complementary tools are used based on their strengths:

| Tool | Best For | Language | Verification Method |
|---|---|---|---|
| **TLA+** | Concurrent systems | TLA+ | TLC model checker (exhaustive state exploration) |
| **Lean 4** | Type-level invariants | Lean 4 | Dependent types, machine-checked proofs |
| **SPIN** | Protocol verification | Promela | LTL model checking |

## TLA+ Models

### Scheduler Model (`formal/tla/Scheduler.tla`)

Models the work-stealing cooperative scheduler with N workers and M tasks.

**Variables:**
- `workers`: Array of worker states (Idle, Running, Stealing)
- `queues`: Per-worker task queues
- `tasks`: Set of task states (Ready, Running, Completed)

**Properties verified:**
- **Deadlock freedom**: The system always has at least one enabled action
- **Starvation freedom**: Every ready task is eventually executed (under weak fairness)
- **Work conservation**: If tasks are ready and a worker is idle, work progresses
- **Termination**: If no new tasks arrive, all tasks eventually complete

**Model parameters:** N=4 workers, M=8 tasks (bounded model checking)

### Memory Allocator Model (`formal/tla/BuddyAllocator.tla`)

Models the buddy allocator with split and merge operations.

**Variables:**
- `pages`: Map from physical page number to state (Free, Allocated, Split)
- `free_lists`: Per-order free lists
- `allocated_count`: Total allocated pages

**Properties verified:**
- **Conservation**: `free_pages + allocated_pages = total_pages` (invariant)
- **No double-free**: A free operation on a non-allocated page is impossible
- **No double-alloc**: An allocated page cannot be allocated again
- **Merge correctness**: When both buddies are free, they merge to the next order
- **No leak**: After free, the page is eventually available for allocation

### Capability Model (`formal/tla/Capabilities.tla`)

Models the capability-based access control system.

**Properties verified:**
- **Non-forgery**: No operation sequence creates capabilities with more permissions than the root set
- **Revocation completeness**: Revoking a capability also revokes all delegated children
- **No privilege escalation**: Delegation can only produce equal or lesser permissions

## Lean 4 Proofs

### Tensor Shape Algebra (`formal/lean4/TensorShape.lean`)

Proves that tensor operations produce correctly shaped outputs.

```lean
theorem matmul_shape_correct (a : Tensor [m, k]) (b : Tensor [k, n]) :
    shape (matmul a b) = [m, n] := by
  simp [matmul, shape]
```

**Properties proven:**
- MatMul output shape is [M, N] given inputs [M, K] and [K, N]
- Reshape preserves total element count
- Transpose swaps specified dimensions
- Concat increases the specified dimension by the sum of inputs
- Broadcasting rules produce correct output shapes

### Capability Type Safety (`formal/lean4/Capability.lean`)

```lean
theorem delegation_no_escalation (cap : Capability) (subset : Permissions) :
    subset ⊆ cap.permissions →
    (delegate cap subset).permissions ⊆ cap.permissions := by
  intro h
  exact h
```

**Properties proven:**
- Delegation produces equal or lesser permissions
- Revocation is transitive (revoking parent revokes children)
- Permission intersection is the maximum delegatable set

### Reference Count Correctness (`formal/lean4/RefCount.lean`)

**Properties proven:**
- Refcount equals number of live references
- Refcount reaches zero iff all references are dropped
- Buffer is freed exactly once when refcount reaches zero

## SPIN Models

### IPC Pub/Sub (`formal/spin/pubsub.pml`)

Models the pub/sub messaging system with publishers, subscribers, and the message router.

```promela
// Simplified model structure
active proctype publisher() {
    do :: router ! MSG, key, data
    od
}

active [N] proctype subscriber() {
    do :: router ? MSG, key, data -> assert(key_matches(key, subscription))
    od
}
```

**LTL Properties verified:**
- `[]<>delivered`: Every published message is eventually delivered to all matching subscribers
- `[](published -> <>delivered)`: Publication implies eventual delivery
- `[](!misrouted)`: No message is delivered to a non-matching subscriber

### TCP State Machine (`formal/spin/tcp.pml`)

Models the TCP connection state machine from RFC 9293.

**LTL Properties verified:**
- `[](close_requested -> <>CLOSED)`: Close request eventually leads to CLOSED state
- `[](!invalid_transition)`: No invalid state transitions
- `[](ESTABLISHED -> <>data_can_flow)`: Established connections can transfer data
- Correct TIME_WAIT behavior (remains in TIME_WAIT for 2MSL before CLOSED)

### Request/Reply (`formal/spin/reqreply.pml`)

**LTL Properties verified:**
- `[](query_sent -> <>(reply_received || timeout))`: Every query gets a reply or times out
- `[](!duplicate_reply)`: No query receives more than one reply
- Timeout correctly fires when queryable is unresponsive

## CI Integration

### TLA+ in CI

```yaml
- name: TLA+ Model Checking
  run: |
    java -jar tla2tools.jar -workers auto \
      -config formal/tla/Scheduler.cfg \
      formal/tla/Scheduler.tla
    java -jar tla2tools.jar -workers auto \
      -config formal/tla/BuddyAllocator.cfg \
      formal/tla/BuddyAllocator.tla
    java -jar tla2tools.jar -workers auto \
      -config formal/tla/Capabilities.cfg \
      formal/tla/Capabilities.tla
```

### SPIN in CI

```yaml
- name: SPIN Protocol Verification
  run: |
    spin -a formal/spin/pubsub.pml
    gcc -o pan pan.c -DSAFETY -DREACH
    ./pan -m100000
    spin -a formal/spin/tcp.pml
    gcc -o pan pan.c -DSAFETY -DREACH
    ./pan -m100000
```

### Lean 4 in CI

```yaml
- name: Lean 4 Proofs
  run: |
    cd formal/lean4
    lake build
```

## Directory Structure

```
formal/
├── tla/
│   ├── Scheduler.tla          # Scheduler concurrency model
│   ├── Scheduler.cfg          # TLC configuration
│   ├── BuddyAllocator.tla    # Memory allocator model
│   ├── BuddyAllocator.cfg
│   ├── Capabilities.tla       # Access control model
│   └── Capabilities.cfg
├── lean4/
│   ├── lakefile.lean          # Lean 4 project file
│   ├── SmallAIOS/
│   │   ├── TensorShape.lean   # Tensor shape algebra proofs
│   │   ├── Capability.lean    # Capability type safety proofs
│   │   └── RefCount.lean      # Reference counting proofs
│   └── lean-toolchain          # Lean version pin
└── spin/
    ├── pubsub.pml             # IPC pub/sub model
    ├── tcp.pml                # TCP state machine model
    └── reqreply.pml           # Request/reply model
```
