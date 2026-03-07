## Context

SmallAIOS has an existing Sphinx documentation setup in `docs/` with sphinx-needs traceability, PlantUML diagram support, and RST content covering architecture, requirements, and traceability matrices. The current theme is Alabaster (default), the build is local-only (`docs/_build/`), and there is no deployment pipeline. The site is not published anywhere.

The project uses GitHub Actions for CI (`.github/workflows/ci.yml`) with established patterns for caching, multi-step jobs, and artifact management. GitHub Pages is available at no cost for public repositories.

Existing docs content:
- `docs/conf.py` — Sphinx config with sphinx-needs types (req, spec, impl, test, design), PlantUML, autodoc
- `docs/index.rst` — landing page with toctree (architecture, requirements, traceability)
- `docs/architecture.rst` + `docs/architecture.puml` — system architecture with PlantUML diagram
- `docs/requirements.rst` — DO-178C requirement definitions with sphinx-needs directives
- `docs/traceability.rst` — requirement traceability matrix
- `docs/security/` — security documentation
- Several markdown files: `bare-metal-deployment.md`, `boot-security-matrix.md`, `local-testing.md`, `misra-rust-policy.md`

## Goals / Non-Goals

**Goals:**
- Replace the Alabaster theme with a dark-mode glass/translucent theme using Furo as the base
- Add custom CSS for glass UI effects (backdrop-filter, semi-transparent backgrounds, frosted borders)
- Configure RTD-style sidebar with collapsible sections and responsive mobile layout
- Set up GitHub Actions workflow to build Sphinx and deploy to GitHub Pages on pushes to main
- Manage Python dependencies in `docs/requirements.txt` for reproducible builds
- Render PlantUML diagrams via the PlantUML public server (no local JAR required)
- Expand the page structure: landing page, architecture overview, getting started, API reference, requirements traceability, changelog

**Non-Goals:**
- Auto-generating Rust API docs from doc comments (requires `cargo doc` integration, deferred to future work; placeholder page only)
- Writing new documentation content beyond structural pages (this change sets up the site, not content authoring)
- Custom Sphinx extensions or plugins beyond sphinx-needs and sphinxcontrib-plantuml
- Multi-version documentation (future work)

## Decisions

### 1. Furo theme with custom glass CSS overlay

**Decision**: Use the Furo theme as the base and apply custom CSS for the dark-mode glass/translucent effect.

**Rationale**: Furo is the most popular modern Sphinx theme with built-in dark mode support, responsive design, RTD-style sidebar with collapsible sections, and clean typography. It supports `html_css_files` for custom overrides without forking the theme. The glass effect is achieved purely with CSS: `backdrop-filter: blur(12px)`, `background: rgba(30, 30, 40, 0.75)`, `border: 1px solid rgba(255, 255, 255, 0.08)`.

**Alternative considered**: sphinx-immaterial (Material Design) — rejected because the visual language is opaque/flat, making the glass/translucent effect fight the theme's design intent. Furo's clean, minimal base makes glass overlays natural.

### 2. PlantUML rendering via public server

**Decision**: Use the PlantUML public rendering server (`https://www.plantuml.com/plantuml/svg/`) instead of a local JAR.

**Rationale**: Avoids requiring a Java runtime in the CI environment, eliminates a ~60MB dependency, and keeps the GitHub Actions workflow simple. The public server is reliable for build-time rendering. The `plantuml` config in `conf.py` changes to use `sphinxcontrib.plantuml` with the server URL.

**Alternative considered**: Local JAR via `apt-get install plantuml` in CI — rejected for build time and image size. A self-hosted server is a future option if the public server becomes unreliable.

### 3. GitHub Actions with pages artifact deployment

**Decision**: Use the modern `actions/deploy-pages` workflow pattern: build Sphinx in a job, upload as a GitHub Pages artifact, then deploy via the `deploy` environment.

**Rationale**: This is GitHub's recommended approach. It avoids the `gh-pages` branch pattern (which requires force-pushes and pollutes git history). The artifact-based approach integrates with GitHub's deployment protection rules and environment approvals.

### 4. Python dependencies in docs/requirements.txt

**Decision**: Pin all Python dependencies in `docs/requirements.txt` with exact versions for reproducible builds.

**Rationale**: Sphinx and its extensions have frequent releases that can break builds. Pinning ensures CI reproducibility. The file is co-located with `docs/conf.py` for discoverability.

### 5. Expanded page structure with myst-parser for markdown

**Decision**: Add `myst-parser` to support the existing markdown files alongside RST, and expand the toctree to include: landing page, architecture, getting started, API reference (placeholder), requirements/traceability, and changelog.

**Rationale**: The docs directory already contains several `.md` files (bare-metal-deployment, boot-security-matrix, local-testing, misra-rust-policy) that are not included in the Sphinx build. Adding myst-parser lets these be included without converting to RST. The root `CHANGELOG.md` can be included via a symlink or myst include.

### 6. Build caching with pip cache

**Decision**: Cache the pip download directory in the GitHub Actions workflow using `actions/setup-python` built-in caching.

**Rationale**: The `actions/setup-python` action natively supports pip caching via `cache: 'pip'` with `cache-dependency-path: 'docs/requirements.txt'`. This avoids re-downloading ~50MB of Python packages on every build, reducing deploy time by 20-30 seconds.

## Risks / Trade-offs

- **[PlantUML server availability]** The public PlantUML server could be temporarily unavailable during a docs build, causing diagram rendering failures. Mitigation: diagrams that fail to render produce a text fallback, not a build failure. The workflow can be re-run.
- **[Theme CSS fragility]** Custom glass CSS depends on Furo's internal DOM structure, which could change across versions. Mitigation: pin the Furo version in requirements.txt; the CSS targets generic selectors (`.sidebar-brand`, `.sidebar-scroll`, `article.bd-article`) that are part of Furo's public API.
- **[GitHub Pages limitations]** GitHub Pages has a 1GB size limit and no server-side processing. Mitigation: Sphinx generates static HTML, and documentation sites are typically <50MB. PlantUML SVGs are small.
- **[Markdown/RST mixing]** Using both RST and markdown (via myst-parser) in one Sphinx project can create confusion about which format to use. Mitigation: establish convention that new docs use RST for sphinx-needs content and markdown for narrative/guide content.
