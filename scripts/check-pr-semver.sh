#!/usr/bin/env bash
# check-pr-semver.sh — Validate PR title follows conventional commits and
# output the implied semver bump level.
#
# Usage:
#   ./scripts/check-pr-semver.sh "feat(net): add QUIC 0-RTT support"
#   ./scripts/check-pr-semver.sh "$PR_TITLE"
#
# Exit codes:
#   0 — valid, bump level printed to stdout
#   1 — invalid title format
#
# Conventional Commits reference: https://www.conventionalcommits.org/
#
# Bump rules (pre-1.0):
#   BREAKING (feat!, fix!, refactor!, or "BREAKING CHANGE" in body) → minor
#   feat                                                             → minor
#   fix, perf                                                        → patch
#   docs, chore, ci, test, refactor, style, build                    → none
#
# Post-1.0 (future):
#   BREAKING → major
#   feat     → minor
#   fix      → patch

set -euo pipefail

TITLE="${1:-}"

if [ -z "$TITLE" ]; then
  echo "error: no PR title provided" >&2
  echo "usage: $0 \"<PR title>\"" >&2
  exit 1
fi

# Pattern: type(optional-scope)optional-!: description
# Allowed types (lowercase)
TYPES="feat|fix|docs|chore|ci|test|refactor|style|perf|build|revert"
PATTERN="^(${TYPES})(\([a-z0-9_/ -]+\))?(!)?: .+"

if ! echo "$TITLE" | grep -qE "$PATTERN"; then
  echo "error: PR title does not follow conventional commits format" >&2
  echo "" >&2
  echo "Expected: <type>[optional scope][!]: <description>" >&2
  echo "" >&2
  echo "Types: feat, fix, docs, chore, ci, test, refactor, style, perf, build, revert" >&2
  echo "" >&2
  echo "Examples:" >&2
  echo "  feat(net): add QUIC 0-RTT support" >&2
  echo "  fix(crypto): correct ML-DSA hint coefficient" >&2
  echo "  feat!: redesign syscall interface" >&2
  echo "  chore: update CI workflow" >&2
  exit 1
fi

# Extract type and breaking indicator
TYPE=$(echo "$TITLE" | sed -E "s/^(${TYPES}).*/\1/")
BREAKING=$(echo "$TITLE" | grep -qE "^(${TYPES})(\([a-z0-9_/ -]+\))?!:" && echo "yes" || echo "no")

# Current version epoch (pre-1.0 vs post-1.0)
WORKSPACE_VERSION=$(grep -m1 'version = ' Cargo.toml | head -1 | sed 's/.*"\(.*\)".*/\1/' || echo "0.1.0")
MAJOR=$(echo "$WORKSPACE_VERSION" | cut -d. -f1)

# Determine bump level
if [ "$BREAKING" = "yes" ]; then
  if [ "$MAJOR" -eq 0 ]; then
    BUMP="minor"
  else
    BUMP="major"
  fi
elif [ "$TYPE" = "feat" ]; then
  BUMP="minor"
elif [ "$TYPE" = "fix" ] || [ "$TYPE" = "perf" ] || [ "$TYPE" = "revert" ]; then
  BUMP="patch"
else
  BUMP="none"
fi

echo "$BUMP"
