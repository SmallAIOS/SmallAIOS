# SmallAIOS Documentation Configuration
# Sphinx-needs + PlantUML for DO-178C traceability
# SPDX-License-Identifier: Apache-2.0

project = "SmallAIOS"
copyright = "2026, SmallAIOS Contributors"
author = "SmallAIOS Contributors"
release = "0.1.0"

extensions = [
    "sphinx_needs",
    "sphinxcontrib.plantuml",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "myst_parser",
]

# MyST-Parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "substitution",
]
suppress_warnings = ["myst.header"]

# Sphinx-needs configuration for DO-178C traceability
needs_types = [
    {
        "directive": "req",
        "title": "Requirement",
        "prefix": "REQ_",
        "color": "#BFD8D2",
        "style": "node",
    },
    {
        "directive": "spec",
        "title": "Specification",
        "prefix": "SPEC_",
        "color": "#FEDCD2",
        "style": "node",
    },
    {
        "directive": "impl",
        "title": "Implementation",
        "prefix": "IMPL_",
        "color": "#DF744A",
        "style": "node",
    },
    {
        "directive": "test",
        "title": "Test Case",
        "prefix": "TEST_",
        "color": "#DCB239",
        "style": "node",
    },
    {
        "directive": "design",
        "title": "Design Decision",
        "prefix": "DES_",
        "color": "#9CAFB7",
        "style": "node",
    },
    {
        "directive": "nist_control",
        "title": "NIST SP 800-53 Control",
        "prefix": "NIST_",
        "color": "#C3E6CB",
        "style": "node",
    },
]

needs_extra_options = ["safety_level", "coverage", "verification_method", "impl_status"]

needs_extra_links = [
    {"option": "satisfies", "incoming": "satisfied_by", "outgoing": "satisfies", "copy": False},
    {"option": "implements", "incoming": "implemented_by", "outgoing": "implements", "copy": False},
    {"option": "verifies", "incoming": "verified_by", "outgoing": "verifies", "copy": False},
    {"option": "traces_to", "incoming": "traced_from", "outgoing": "traces_to", "copy": False},
]

# PlantUML configuration — use public server (no local Java required)
plantuml_server = "https://www.plantuml.com/plantuml"
plantuml_output_format = "svg"

# HTML output — Furo theme with dark mode
html_theme = "furo"
html_title = "SmallAIOS"
html_static_path = ["_static"]
html_css_files = ["css/glass-theme.css"]

html_theme_options = {
    "dark_css_variables": {
        "color-brand-primary": "#64ffda",
        "color-brand-content": "#64ffda",
        "color-background-primary": "#1a1a2e",
        "color-background-secondary": "#16213e",
        "color-foreground-primary": "#e0e0e0",
        "color-foreground-secondary": "#b0b0c0",
    },
    "light_css_variables": {
        "color-brand-primary": "#00897b",
        "color-brand-content": "#00897b",
    },
}

# Build options
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
