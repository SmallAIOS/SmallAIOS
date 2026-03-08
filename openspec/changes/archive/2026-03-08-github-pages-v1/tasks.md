## 1. Python Dependencies

- [x] 1.1 Create `docs/requirements.txt` with pinned versions: `Sphinx`, `furo`, `sphinx-needs`, `sphinxcontrib-plantuml`, `myst-parser`
- [x] 1.2 Verify all dependencies install cleanly: `pip install -r docs/requirements.txt`

## 2. Sphinx Configuration Update

- [x] 2.1 Update `docs/conf.py`: change `html_theme` from `"alabaster"` to `"furo"`
- [x] 2.2 Add `myst_parser` to `extensions` list in `docs/conf.py`
- [x] 2.3 Configure Furo dark mode as default: set `html_theme_options` with `light_css_variables` and `dark_css_variables` color overrides (dark backgrounds: #1a1a2e base, #16213e secondary, light text: #e0e0e0)
- [x] 2.4 Update `plantuml` setting in `docs/conf.py` to use the public server: `plantuml = "java -jar"` replaced with server URL config via `plantuml_server = "https://www.plantuml.com/plantuml"` and `plantuml_output_format = "svg"`
- [x] 2.5 Add `myst_enable_extensions` config for markdown features (colon_fence, deflist, substitution)
- [x] 2.6 Configure `html_title`, `html_favicon`, and `html_logo` if project assets exist
- [x] 2.7 Add `suppress_warnings = ["myst.header"]` to avoid noisy myst-parser warnings for existing markdown files

## 3. Glass Theme Custom CSS

- [x] 3.1 Create `docs/_static/css/glass-theme.css` with glass/translucent styles:
  - Sidebar: `backdrop-filter: blur(12px)`, `background: rgba(30, 30, 40, 0.75)`, `border-right: 1px solid rgba(255, 255, 255, 0.08)`
  - Content area: `background: rgba(25, 25, 45, 0.65)` with subtle border
  - Cards/admonitions: semi-transparent backgrounds with blur
  - Code blocks: slightly lighter translucent background
  - Links: accent color (#64ffda or similar cyan-green)
  - Scrollbar: thin, translucent track
- [x] 3.2 Add `@supports` fallback for browsers without `backdrop-filter` support (opaque dark backgrounds)
- [x] 3.3 Register custom CSS in `docs/conf.py` via `html_css_files = ["css/glass-theme.css"]`
- [x] 3.4 Verify glass effect renders correctly: build locally with `sphinx-build -b html docs/ docs/_build/html` and inspect in browser

## 4. Page Structure and Navigation

- [x] 4.1 Rewrite `docs/index.rst` as expanded landing page: project name, tagline, key features list, quick links to subsections, badges (CI status, coverage, license)
- [x] 4.2 Create `docs/getting-started.rst` with build instructions (from CLAUDE.md), deployment modes (container, bare-metal, QEMU), and first-inference walkthrough
- [x] 4.3 Create `docs/api-reference.rst` as a placeholder page explaining that API docs will be auto-generated from Rust doc comments in a future iteration
- [x] 4.4 Create `docs/changelog.rst` that includes the root `CHANGELOG.md` via myst-parser `include` directive or a symlink
- [x] 4.5 Update toctree in `docs/index.rst` to include all pages: architecture, getting-started, api-reference, requirements, traceability, changelog
- [x] 4.6 Add existing markdown files to toctree or a "Guides" subsection: bare-metal-deployment, local-testing, boot-security-matrix, misra-rust-policy
- [x] 4.7 Organize sidebar into logical sections using toctree captions: "Overview" (index, architecture), "User Guide" (getting-started, guides), "Reference" (api-reference, requirements, traceability), "Project" (changelog)

## 5. Sphinx-Needs Traceability Styling

- [x] 5.1 Verify sphinx-needs directives render with correct colors in the Furo dark theme
- [x] 5.2 Add custom CSS overrides in `glass-theme.css` for sphinx-needs cards if default colors clash with dark glass background
- [x] 5.3 Verify `needtable`, `needlist`, and `needflow` render correctly in built output
- [x] 5.4 Test that extra options (`safety_level`, `coverage`, `verification_method`) display in rendered needs items

## 6. GitHub Actions Docs Workflow

- [x] 6.1 Create `.github/workflows/docs.yml` with trigger on push to `main` and on pull requests targeting `main`
- [x] 6.2 Add `build` job: checkout, setup Python 3.12 with pip cache (`cache: 'pip'`, `cache-dependency-path: 'docs/requirements.txt'`), install dependencies, run `sphinx-build -b html docs/ docs/_build/html`
- [x] 6.3 Add conditional deploy logic: on push to main, upload artifact with `actions/upload-pages-artifact` (path: `docs/_build/html`) and deploy with `actions/deploy-pages`
- [x] 6.4 Set workflow permissions: `contents: read`, `pages: write`, `id-token: write`
- [x] 6.5 Configure the `deploy` job to use `environment: github-pages` with `url: ${{ steps.deployment.outputs.page_url }}`
- [x] 6.6 Add concurrency group `pages` with `cancel-in-progress: false` to prevent parallel deployments
- [x] 6.7 Ensure PR builds only run the `build` job (no deploy) to validate docs compile

## 7. GitHub Pages Repository Settings

- [x] 7.1 Configure GitHub Pages source to "GitHub Actions" in repository settings (Settings > Pages > Source > GitHub Actions)

## 8. Local Build Verification

- [x] 8.1 Run `sphinx-build -b html docs/ docs/_build/html` locally and verify: all pages build without errors, glass theme renders, PlantUML diagrams appear, sphinx-needs items display correctly, sidebar navigation works
- [x] 8.2 Verify mobile responsiveness by resizing browser window below 768px: sidebar collapses, hamburger menu works, content fills viewport
- [x] 8.3 Add `docs/_build/` to `.gitignore` if not already present
- [x] 8.4 Update `docs/` entry in root `.gitignore` or verify `_build` exclusion in `conf.py` `exclude_patterns`
