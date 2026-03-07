#!/usr/bin/env bash
# suggest-release-bump.sh — Suggest the semver bump level for the next release.
#
# Scans all commit messages since the last v* tag, applies conventional commit
# bump rules (pre-1.0), and outputs the highest bump level to stdout.
# Individual commit classifications are printed to stderr for review.
#
# Usage: ./scripts/suggest-release-bump.sh
# Output (stdout): "minor", "patch", or "none"
#
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

# Find the most recent v* tag
LAST_TAG=$(git describe --tags --match "v[0-9]*" --abbrev=0 2>/dev/null || echo "")

if [ -z "$LAST_TAG" ]; then
    echo "No existing v* tag found — first release." >&2
    echo "minor"
    exit 0
fi

# Get commit messages between last tag and HEAD
RANGE="${LAST_TAG}..HEAD"
COMMITS=$(git log "$RANGE" --pretty=format:"%s" 2>/dev/null)

if [ -z "$COMMITS" ]; then
    echo "No commits since ${LAST_TAG}." >&2
    echo "none"
    exit 0
fi

echo "Commits since ${LAST_TAG}:" >&2
echo "---" >&2

max_bump="none"

while IFS= read -r msg; do
    # Extract type and check for breaking change indicator
    if echo "$msg" | grep -qE '^(feat|fix|docs|chore|ci|test|refactor|style|perf|build|revert)(\([a-z0-9_/ -]+\))?!:'; then
        bump="minor"
    elif echo "$msg" | grep -qE '^feat(\([a-z0-9_/ -]+\))?:'; then
        bump="minor"
    elif echo "$msg" | grep -qE '^(fix|perf|revert)(\([a-z0-9_/ -]+\))?:'; then
        bump="patch"
    elif echo "$msg" | grep -qE '^(docs|chore|ci|test|refactor|style|build)(\([a-z0-9_/ -]+\))?:'; then
        bump="none"
    else
        bump="none"
    fi

    printf "  %-7s  %s\n" "[$bump]" "$msg" >&2

    # Track maximum bump level
    case "$bump" in
        minor) max_bump="minor" ;;
        patch) [ "$max_bump" != "minor" ] && max_bump="patch" ;;
    esac
done <<< "$COMMITS"

echo "---" >&2
echo "Suggested bump: ${max_bump}" >&2
echo "$max_bump"
