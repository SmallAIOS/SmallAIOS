# Spec 14: Documentation System

## Overview

SmallAIOS uses **Sphinx-needs** for requirements traceability and **PlantUML** for
architecture diagrams, integrated into a single documentation build that produces
all DO-178C required artifacts.

## Sphinx-needs Configuration

### Need Types

| Type | Prefix | Description | Example |
|---|---|---|---|
| REQ | REQ_ | High-level requirement | REQ_MEM_001: Buddy allocator SHALL manage physical memory |
| SPEC | SPEC_ | Low-level specification | SPEC_MEM_001_01: Buddy split at order N creates two order N-1 blocks |
| IMPL | IMPL_ | Implementation reference | IMPL_MEM_001_01: kernel/src/mem/buddy.rs:split() |
| TEST | TEST_ | Test case | TEST_MEM_001_01: test_buddy_split_creates_two_blocks |
| VERIFY | VER_ | Verification result | VER_MEM_001_01: PASS, MC/DC 100%, 2026-02-15 |
| HAZARD | HAZ_ | Hazard from ARP4761 | HAZ_001: Memory corruption during inference |
| FORMAL | FORMAL_ | Formal verification model | FORMAL_SCHED_001: TLA+ deadlock freedom proof |

### Traceability Links

```
REQ ──→ SPEC ──→ IMPL ──→ TEST ──→ VERIFY
  │                          │
  └──→ HAZARD               └──→ FORMAL
```

Every requirement must have a complete chain. The documentation build fails
if any link in the chain is missing.

### Sphinx-needs Configuration (`conf.py`)

```python
needs_types = [
    dict(directive="req", title="Requirement", prefix="REQ_", color="#BFD8D2"),
    dict(directive="spec", title="Specification", prefix="SPEC_", color="#DEDCE6"),
    dict(directive="impl", title="Implementation", prefix="IMPL_", color="#FFE5B4"),
    dict(directive="test", title="Test Case", prefix="TEST_", color="#D4EDDA"),
    dict(directive="verify", title="Verification", prefix="VER_", color="#CCE5FF"),
    dict(directive="hazard", title="Hazard", prefix="HAZ_", color="#F8D7DA"),
    dict(directive="formal", title="Formal Proof", prefix="FORMAL_", color="#E2D9F3"),
]

needs_extra_options = {
    "dal": dict(description="Design Assurance Level", default="A"),
    "coverage": dict(description="MC/DC Coverage %", default=""),
    "module": dict(description="Rust module path", default=""),
}
```

## PlantUML Diagrams

### Required Diagrams

| Diagram | Type | Shows |
|---|---|---|
| System architecture | Component | All crates, dependencies, HAL boundary |
| Inference data flow | Sequence | Client → IPC → ONNX → EP → response |
| Boot sequence | Sequence | Firmware → HAL init → kernel init → ready |
| Task lifecycle | State machine | Created → Ready → Running → Waiting → Completed |
| TCP state machine | State machine | All TCP states and transitions |
| Memory hierarchy | Component | Buddy → Slab → Heap → Tensor Pool |
| Capability flow | Sequence | Root → delegation → check → revocation |
| Container deployment | Deployment | Docker/K8s pod with SmallAIOS container |
| Bare metal deployment | Deployment | UEFI → SmallAIOS → hardware |
| GPU data flow | Sequence | CPU tensor → DMA → GPU VRAM → kernel → DMA → CPU |

### PlantUML Integration

Diagrams are stored as `.puml` files in `docs/plantuml/` and rendered during
Sphinx build via the `sphinxcontrib-plantuml` extension.

## DO-178C Document Mapping

Each DO-178C required document maps to a Sphinx document:

| DO-178C Document | Sphinx Location | Content Source |
|---|---|---|
| PSAC | docs/source/do178c/psac.rst | Manual |
| SDP | docs/source/do178c/sdp.rst | Manual + auto-generated from CI config |
| SVP | docs/source/do178c/svp.rst | Manual + auto-generated coverage config |
| SCMP | docs/source/do178c/scmp.rst | Auto-generated from git configuration |
| SRS | docs/source/do178c/srs.rst | Auto-extracted from Sphinx-needs REQ items |
| SDD | docs/source/do178c/sdd.rst | Auto-extracted from Sphinx-needs SPEC/IMPL items |
| SCS | docs/source/do178c/scs.rst | MISRA-Rust coding standard |
| SVCP | docs/source/do178c/svcp.rst | Auto-extracted from Sphinx-needs TEST items |
| SVR | docs/source/do178c/svr.rst | Auto-generated from CI test results |
| SCI | docs/source/do178c/sci.rst | Auto-generated from cargo metadata + git |
| SAS | docs/source/do178c/sas.rst | Summary of all verification evidence |

## Directory Structure

```
docs/
├── Makefile                   # Sphinx build
├── source/
│   ├── conf.py               # Sphinx configuration (needs, plantuml)
│   ├── index.rst             # Documentation root
│   ├── requirements/
│   │   ├── kernel.rst        # Kernel requirements (REQ, SPEC)
│   │   ├── onnx.rst          # ONNX runtime requirements
│   │   ├── ipc.rst           # IPC requirements
│   │   ├── security.rst      # Security requirements
│   │   └── ...
│   ├── architecture/
│   │   ├── overview.rst      # Architecture overview (embeds PlantUML)
│   │   ├── memory.rst        # Memory model
│   │   └── ...
│   ├── do178c/
│   │   ├── psac.rst
│   │   ├── sdp.rst
│   │   ├── svp.rst
│   │   └── ...
│   └── traceability/
│       ├── matrix.rst        # Auto-generated traceability matrix
│       └── coverage.rst      # MC/DC coverage reports
├── plantuml/
│   ├── architecture.puml
│   ├── inference-flow.puml
│   ├── boot-sequence.puml
│   ├── task-lifecycle.puml
│   ├── tcp-state.puml
│   ├── memory-hierarchy.puml
│   ├── capability-flow.puml
│   ├── container-deploy.puml
│   ├── bare-metal-deploy.puml
│   └── gpu-data-flow.puml
└── do178c/
    ├── templates/            # Document templates
    └── evidence/             # Collected verification evidence
```

## Documentation CI

```yaml
- name: Build Documentation
  run: |
    pip install sphinx sphinx-needs sphinxcontrib-plantuml
    cd docs && make html
    # Fail if any warnings
    make html 2>&1 | grep -c "WARNING" | xargs test 0 -eq

- name: Traceability Check
  run: |
    cd docs && make needs-check
    # Fails if any orphan REQ, SPEC, IMPL, or TEST
```
