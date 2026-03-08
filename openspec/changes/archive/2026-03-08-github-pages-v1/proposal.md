## Why

The project lacks a public documentation site. Technical documentation, architecture diagrams, API references, and requirement traceability exist in RST files but are not published anywhere accessible. A GitHub Pages site provides zero-cost hosting with automatic deployment from the repo.

## What Changes

- Create a GitHub Pages documentation site deployed via GitHub Actions
- Use a modern dark-mode theme with glass/translucent UI elements and Read the Docs style sidebar navigation
- Auto-deploy documentation on pushes to main
- Integrate existing Sphinx/sphinx-needs RST documentation
- Render PlantUML diagrams as part of the build
- Include architecture overview, API docs, getting started guide, and requirement traceability matrix

## Capabilities

### New Capabilities
- `docs-site`: GitHub Pages documentation site — covers theme selection/customization (dark glass UI, RTD-style nav), Sphinx build configuration, page structure, and content organization
- `docs-ci-deploy`: CI/CD pipeline for documentation — covers GitHub Actions workflow for building and deploying docs to GitHub Pages on pushes to main, PlantUML rendering, and sphinx-needs integration

### Modified Capabilities
None — this change adds a new documentation deployment without modifying existing specs.

## Impact

- `docs/` — restructure and expand Sphinx configuration
- `docs/conf.py` — theme configuration (likely Furo or sphinx-immaterial with custom CSS)
- `.github/workflows/` — new docs deployment workflow
- `requirements.txt` or `docs/requirements.txt` — Python dependencies for Sphinx build
- GitHub Pages settings — configure to deploy from GitHub Actions
