# SmallAIOS Architecture (4+1 View Model)

This directory contains architectural documentation following the Kruchten 4+1 View Model, rendered in PlantUML.

## Views

| View | File | Description |
|------|------|-------------|
| **Logical** | `logical-view.puml` | Crate decomposition, trait interfaces, dependency relationships |
| **Process** | `process-view.puml` | Boot sequence, cooperative scheduler, async task flow |
| **Physical** | `physical-view.puml` | Deployment targets: bare metal, VM, container, Jetson |
| **Development** | `development-view.puml` | CI pipeline, build matrix, workspace structure |
| **Scenarios** | `scenarios.puml` | Key use cases: boot-to-inference, QUIC handshake, tensor lifecycle |

## Framework

The 4+1 View Model (Kruchten, 1995) is ISO/IEC/IEEE 42010 compatible and maps naturally to safety-critical documentation requirements:

- **Logical View** supports AUTOSAR-style software component modeling
- **Process View** documents timing and concurrency for WCET analysis
- **Physical View** maps to DO-178C deployment documentation
- **Development View** traces to CI/CD qualification evidence
- **Scenarios** provide validation test cases for each view

## Rendering

PlantUML diagrams can be rendered with:

```bash
# Local rendering (requires Java + PlantUML)
java -jar plantuml.jar docs/architecture/*.puml

# VS Code: Install PlantUML extension for live preview
```

Diagrams are also rendered automatically on the GitHub Pages documentation site via sphinx-needs integration.
