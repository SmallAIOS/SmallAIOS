#!/usr/bin/env bash
# bump-version.sh — Bump the workspace version across all crates.
#
# Usage:
#   ./scripts/bump-version.sh patch   # 0.1.0 → 0.1.1
#   ./scripts/bump-version.sh minor   # 0.1.0 → 0.2.0
#   ./scripts/bump-version.sh major   # 0.1.0 → 1.0.0
#   ./scripts/bump-version.sh 0.3.0   # explicit version
#
# Updates Cargo.toml workspace version and runs cargo check to validate.

set -euo pipefail

BUMP="${1:-}"
CARGO_TOML="Cargo.toml"

if [ -z "$BUMP" ]; then
  echo "usage: $0 <major|minor|patch|X.Y.Z>" >&2
  exit 1
fi

# Read current version
CURRENT=$(grep -A1 '\[workspace.package\]' "$CARGO_TOML" | grep 'version' | sed 's/.*"\(.*\)".*/\1/')
if [ -z "$CURRENT" ]; then
  echo "error: could not read workspace version from $CARGO_TOML" >&2
  exit 1
fi

IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "$BUMP" in
  major)
    MAJOR=$((MAJOR + 1))
    MINOR=0
    PATCH=0
    ;;
  minor)
    MINOR=$((MINOR + 1))
    PATCH=0
    ;;
  patch)
    PATCH=$((PATCH + 1))
    ;;
  [0-9]*)
    # Explicit version
    if ! echo "$BUMP" | grep -qE '^[0-9]+\.[0-9]+\.[0-9]+$'; then
      echo "error: invalid version format '$BUMP' — expected X.Y.Z" >&2
      exit 1
    fi
    IFS='.' read -r MAJOR MINOR PATCH <<< "$BUMP"
    ;;
  *)
    echo "error: unknown bump level '$BUMP'" >&2
    echo "usage: $0 <major|minor|patch|X.Y.Z>" >&2
    exit 1
    ;;
esac

NEW_VERSION="${MAJOR}.${MINOR}.${PATCH}"

if [ "$NEW_VERSION" = "$CURRENT" ]; then
  echo "Version is already $CURRENT — nothing to do."
  exit 0
fi

echo "Bumping version: $CURRENT → $NEW_VERSION"

# Update workspace Cargo.toml
sed -i "s/^version = \"$CURRENT\"/version = \"$NEW_VERSION\"/" "$CARGO_TOML"

echo "Updated $CARGO_TOML"
echo ""
echo "Version bumped to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  1. Run 'cargo check' to validate"
echo "  2. Update Cargo.lock: 'cargo update --workspace'"
echo "  3. Commit: git commit -am \"chore: bump version to $NEW_VERSION\""
echo "  4. Tag: git tag v$NEW_VERSION"
