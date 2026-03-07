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
]

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

needs_extra_options = ["safety_level", "coverage", "verification_method", "implements", "status"]

needs_extra_links = [
    {"option": "satisfies", "incoming": "satisfied_by", "copy": False},
    {"option": "implements", "incoming": "implemented_by", "copy": False},
    {"option": "verifies", "incoming": "verified_by", "copy": False},
    {"option": "traces_to", "incoming": "traced_from", "copy": False},
]

# PlantUML configuration
plantuml = "plantuml"
plantuml_output_format = "svg"

# HTML output
html_theme = "alabaster"
html_static_path = ["_static"]

# Build options
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
templates_path = ["_templates"]
