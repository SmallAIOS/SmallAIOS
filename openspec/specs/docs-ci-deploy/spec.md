# docs-ci-deploy Specification

## Purpose
TBD - created by archiving change github-pages-v1. Update Purpose after archive.
## Requirements
### Requirement: Auto-deployment on push to main
A GitHub Actions workflow SHALL build the Sphinx documentation and deploy it to GitHub Pages on every push to the `main` branch. The workflow SHALL use the `actions/deploy-pages` action pattern (artifact upload + deploy) rather than the legacy `gh-pages` branch approach.

#### Scenario: Push to main triggers docs build and deploy
- **WHEN** a commit is pushed to the `main` branch
- **THEN** the GitHub Actions workflow triggers, builds the Sphinx documentation, uploads the HTML output as a GitHub Pages artifact, and deploys it to the configured GitHub Pages URL

#### Scenario: Push to non-main branch does not deploy
- **WHEN** a commit is pushed to `develop` or a feature branch
- **THEN** the docs deployment workflow does not trigger

#### Scenario: PR builds docs without deploying
- **WHEN** a pull request targets the `main` branch
- **THEN** the workflow builds the documentation to verify it compiles without errors, but does not deploy to GitHub Pages

### Requirement: Sphinx build produces complete HTML output
The CI workflow SHALL run `sphinx-build -b html docs/ docs/_build/html` with warnings treated as non-fatal. The build SHALL produce a complete static HTML site suitable for GitHub Pages hosting, including all CSS, JavaScript, images, and rendered diagrams.

#### Scenario: Build produces deployable HTML
- **WHEN** the Sphinx build step completes successfully
- **THEN** the `docs/_build/html/` directory contains `index.html`, all linked pages, static assets (CSS, JS), and rendered PlantUML SVGs

#### Scenario: Build fails on RST syntax errors
- **WHEN** a documentation source file contains invalid RST or markdown syntax
- **THEN** the build step fails with a non-zero exit code and the workflow reports the error

#### Scenario: Sphinx-needs directives are processed
- **WHEN** the build runs
- **THEN** all `.. req::`, `.. spec::`, `.. impl::`, `.. test::`, and `.. design::` directives are processed and rendered, and `.. needtable::` / `.. needlist::` / `.. needflow::` directives produce their expected output

### Requirement: Python dependencies managed via requirements file
The CI workflow SHALL install Python dependencies from `docs/requirements.txt` using pip. The requirements file SHALL pin exact versions of all direct dependencies: Sphinx, Furo, sphinx-needs, sphinxcontrib-plantuml, and myst-parser.

#### Scenario: Dependencies install from requirements.txt
- **WHEN** the CI workflow runs the pip install step
- **THEN** all packages listed in `docs/requirements.txt` are installed at the pinned versions

#### Scenario: Requirements file is self-contained
- **WHEN** a developer runs `pip install -r docs/requirements.txt` locally
- **THEN** all dependencies needed to build the documentation are installed without additional manual steps

### Requirement: Build caching for faster deploys
The CI workflow SHALL cache Python pip packages between runs using the `actions/setup-python` cache mechanism with `cache: 'pip'` and `cache-dependency-path: 'docs/requirements.txt'`. Cache hits SHALL reduce workflow runtime by avoiding redundant package downloads.

#### Scenario: Cache is populated on first run
- **WHEN** the workflow runs for the first time (no cache exists)
- **THEN** pip packages are downloaded, installed, and the pip cache is saved for future runs

#### Scenario: Cache is reused on subsequent runs
- **WHEN** the workflow runs and `docs/requirements.txt` has not changed since the last run
- **THEN** the pip cache is restored, skipping package downloads and reducing the install step duration

#### Scenario: Cache is invalidated on dependency change
- **WHEN** `docs/requirements.txt` is modified (version bump, new package)
- **THEN** the old cache is not used, packages are re-downloaded at the new versions, and the updated cache is saved

### Requirement: PlantUML rendering in CI via public server
The CI workflow SHALL render PlantUML diagrams using the PlantUML public server (`https://www.plantuml.com/plantuml/svg/`). No Java runtime or local PlantUML JAR SHALL be required in the CI environment.

#### Scenario: Diagrams render without Java
- **WHEN** the Sphinx build runs in CI
- **THEN** PlantUML diagrams are rendered via HTTP requests to the public PlantUML server, producing SVG output, without requiring `java` or `plantuml` to be installed on the runner

### Requirement: Workflow uses GitHub Pages environment
The deployment step SHALL use the `github-pages` environment with the `id-token: write` and `pages: write` permissions. The workflow SHALL use `actions/upload-pages-artifact` to package the build output and `actions/deploy-pages` to publish it.

#### Scenario: Deployment uses proper permissions
- **WHEN** the deploy job runs
- **THEN** it uses the `github-pages` environment, requests `id-token: write` and `pages: write` permissions, and successfully publishes to the repository's GitHub Pages URL

#### Scenario: Deployment URL is reported
- **WHEN** the deployment completes successfully
- **THEN** the workflow output includes the URL where the documentation is accessible

