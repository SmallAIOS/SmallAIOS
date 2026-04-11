# SmallAIOS Formal Verification

This directory contains formal models for verifying safety and liveness properties of SmallAIOS protocols and algorithms.

## Tool Division of Responsibility

| Tool | Property Type | Models | Location |
|------|--------------|--------|----------|
| **TLA+** (TLC) | Safety invariants, deadlock freedom, bounded state | 22 models | `formal/tla/` |
| **SPIN** (Promela) | Liveness (LTL), protocol conformance, fairness | 6 models | `formal/spin/`, `formal/promela/` |
| **Lean 4** | Theorem proving (mathematical proofs) | Experimental | `formal/lean4/` |

### When to Use TLA+

- **Safety properties**: "Nothing bad ever happens" — invariants that must hold in every reachable state
- **Deadlock freedom**: Verify that the system never reaches a state where no process can progress
- **Bounded model checking**: State spaces that can be fully explored with small constants
- **Examples**: Memory allocator correctness, protocol state machine validity, arbitration fairness

### When to Use SPIN

- **Liveness properties**: "Something good eventually happens" — expressed as LTL formulas
- **Protocol conformance**: Verify message exchange sequences match specifications
- **Fairness**: Verify that no process is starved indefinitely
- **Examples**: QUIC handshake completion, IPC message delivery, scheduler fairness

### Overlap and Complementarity

Both tools can check some of the same properties, but excel in different areas:

- TLA+ is better for abstract protocol design and invariant checking
- SPIN is better for LTL liveness checking and counterexample generation
- Using both provides defense-in-depth: TLA+ catches safety violations, SPIN catches liveness violations

## Directory Structure

```
formal/
├── tla/              # TLA+ models and configs
│   ├── *.tla         # TLA+ specifications
│   └── *.cfg         # TLC model checker configs
├── spin/             # SPIN/Promela models (legacy location)
│   ├── ipc_pubsub.pml
│   ├── PubSubRouting.pml
│   ├── TcpStateMachine.pml
│   └── InferencePipeline.pml
├── promela/          # SPIN/Promela models (new)
│   ├── quic_handshake.pml
│   └── scheduler_fairness.pml
└── lean4/            # Lean 4 proofs (experimental)
```

## Running Verification

### TLA+
```bash
make tla-verify  # Runs TLC on all 22 models (5 min timeout each)
```

### SPIN
```bash
# Single model
spin -a formal/promela/quic_handshake.pml
cc -o pan pan.c
./pan -a  # Check LTL properties

# All models (CI)
make spin-verify
```

## CI Integration

- **TLA+**: Runs on every PR and push to main/develop (`.github/workflows/ci.yml`)
- **SPIN**: Runs on every PR and push (`.github/workflows/ci.yml`)
- Both have 5-minute timeouts per model; timeouts are warnings, not failures
