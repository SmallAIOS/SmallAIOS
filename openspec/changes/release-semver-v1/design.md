## Context

SmallAIOS uses Gitflow branching (feature branches -> `develop` -> `main`), conventional commit PR titles with semver enforcement, `cargo-release` for version bumping, and a GitHub Actions release workflow that triggers on `v*` tags. The 0.1.0 alpha was released manually: the CHANGELOG.md was hand-written, the develop->main merge was a squash-merge followed by a main->develop sync, and cargo-release handled the version bump + tag.

What is missing:

1. **No documented release checklist** -- the steps live in contributor memory, not in a runbook.
2. **Manual changelog** -- CHANGELOG.md is hand-maintained in Keep a Changelog format, which does not scale as PR volume increases.
3. **Ambiguous version bump rules** -- CLAUDE.md documents the mapping but there is no automated decision tool beyond `check-pr-semver.sh` (which only validates individual PR titles, not the aggregate bump for a release).
4. **No post-release sync enforcement** -- after squash-merging develop->main, the main->develop sync-back must happen immediately to avoid divergent histories, but nothing enforces this.

Current tooling inventory:

| Tool | Role | Status |
|------|------|--------|
| `scripts/check-pr-semver.sh` | Validates PR title, prints bump level | Working |
| `release.toml` | cargo-release config (shared version, push=false, pre-release-hook=make test) | Working |
| `.github/workflows/release.yml` | Builds kernels, creates GitHub Release, publishes Docker on `v*` tag | Working |
| `.github/workflows/ci.yml` | Format, clippy, test, build, TLA+, coverage, SonarCloud | Working |
| `CHANGELOG.md` | Keep a Changelog format, manual entries | Working but manual |

## Goals / Non-Goals

**Goals:**
- Document the full develop->main->release->sync-back workflow as a repeatable runbook
- Automate CHANGELOG.md generation from merged PR titles using git-cliff
- Provide a version bump decision tree that aggregates all PR titles since last release
- Generate GitHub Release notes from the corresponding CHANGELOG.md section (already partially done in release.yml)
- Integrate changelog automation with cargo-release via pre-release hooks
- Keep the workflow simple enough for a single maintainer

**Non-Goals:**
- Automated release triggering (releases remain manually initiated via `make release`)
- Branch protection rule changes (those are GitHub settings, not code)
- crates.io publishing (all crates have `publish = false`)
- Post-1.0 versioning rules (will be addressed when 1.0 is reached)
- Release signing with GPG/sigstore (may be added later but out of scope here)

## Decisions

### Decision 1: git-cliff for changelog automation

**Choice**: Use [git-cliff](https://git-cliff.org/) to generate CHANGELOG.md entries from conventional commit PR titles (squash-merge commit messages).

**Rationale**: git-cliff is a standalone Rust binary with no runtime dependencies, supports conventional commits natively, outputs Keep a Changelog format, and can be configured via `cliff.toml`. It reads git history directly -- no GitHub API token required for local use. The existing CHANGELOG.md format (Keep a Changelog) is already compatible with git-cliff's output templates.

**Alternatives considered**:
- **conventional-changelog (Node.js)**: Requires Node.js runtime, heavier dependency. SmallAIOS is a Rust project with no Node.js tooling.
- **Manual changelog**: Current approach. Does not scale and is error-prone.
- **GitHub auto-generated release notes**: Only covers release notes, not CHANGELOG.md. Cannot be committed to the repo.

**Integration**: git-cliff runs as a cargo-release `pre-release-hook` replacement (or addition). Before the version bump commit, git-cliff regenerates the `[Unreleased]` section of CHANGELOG.md with entries since the last tag.

### Decision 2: Squash-merge develop->main with immediate sync-back

**Choice**: The develop->main merge uses GitHub's squash-merge (via PR). Immediately after the squash-merge lands, main is merged back into develop with a regular merge commit to synchronize histories.

**Step-by-step workflow**:

1. **Prepare release PR**: Create PR from `develop` to `main` with title following conventional commits (e.g., `feat: release 0.2.0`)
2. **Squash-merge**: Merge the PR using GitHub's squash-merge button. This creates a single commit on main.
3. **Sync-back**: Immediately run `git checkout develop && git merge main && git push origin develop`. This creates a merge commit on develop that ties the histories together.
4. **Version bump**: On main, run `make release BUMP=<level>`. cargo-release bumps versions, runs pre-release hook (make test + git-cliff), commits, and tags.
5. **Push**: Review the commit and tag, then `git push origin main --follow-tags` to trigger release.yml.
6. **Post-release sync**: Merge main back into develop again to pick up the version bump commit and tag.

**Rationale**: Squash-merge keeps main's history clean (one commit per release). The sync-back prevents develop from diverging. Two sync-backs per release (post-squash and post-version-bump) is slightly redundant but keeps histories aligned at every step.

### Decision 3: Aggregate bump level from PR titles

**Choice**: The release bump level is determined by scanning all squash-merge commit messages on develop since the last release tag, extracting the conventional commit type from each, and taking the highest bump level.

**Decision tree (pre-1.0)**:
```
For each PR merged since last tag:
  - Has `!` (breaking change)? -> minor
  - Type is `feat`?             -> minor
  - Type is `fix`/`perf`/`revert`? -> patch
  - Otherwise                   -> none

Release bump = max(all individual bumps)
  - Any minor? -> minor
  - Any patch? -> patch
  - All none?  -> no release needed (skip or force patch)
```

**Implementation**: A new script `scripts/suggest-release-bump.sh` reads git log from the last `v*` tag to HEAD, applies `check-pr-semver.sh` logic to each commit message, and outputs the suggested bump level.

**Rationale**: This removes ambiguity about which bump level to use. The maintainer runs the script, reviews the suggestion, and passes it to `make release BUMP=<level>`.

### Decision 4: Release checklist as a markdown runbook

**Choice**: Store the release checklist as `docs/release-runbook.md` rather than a GitHub issue template.

**Rationale**: A runbook in the repo is version-controlled, searchable, and always available (even offline). GitHub issue templates are useful for recurring issues but add ceremony (creating/closing issues) that is unnecessary for a single-maintainer project. The runbook can be promoted to an issue template later if the team grows.

**Contents**: The runbook will cover pre-release checks, the merge/bump/push sequence, post-release verification, and rollback procedures.

### Decision 5: Integrate git-cliff into cargo-release flow

**Choice**: Replace the existing `pre-release-hook` in `release.toml` with a script that runs both `make test` and `git-cliff` to update CHANGELOG.md before the version bump commit.

**Current hook**: `pre-release-hook = ["bash", "-c", "make -C $(git rev-parse --show-toplevel) test"]`

**New hook**: `pre-release-hook = ["bash", "-c", "make -C $(git rev-parse --show-toplevel) test && make -C $(git rev-parse --show-toplevel) changelog"]`

This means `make release BUMP=minor` will:
1. Run tests (existing)
2. Generate changelog entries via git-cliff (new)
3. Stage CHANGELOG.md changes (handled by the script)
4. Bump version in all 18 Cargo.toml files (cargo-release)
5. Create commit: "chore: release v0.2.0" (cargo-release)
6. Create tag: v0.2.0 (cargo-release)

### Decision 6: cliff.toml configuration

**Choice**: Use a `cliff.toml` at the repo root with a Keep a Changelog-compatible template that groups entries by conventional commit type.

**Group mapping**:
```
feat     -> ### Added
fix      -> ### Fixed
perf     -> ### Performance
refactor -> ### Changed
docs     -> ### Documentation
revert   -> ### Reverted
```

Types `chore`, `ci`, `test`, `style`, `build` are excluded from the changelog (they produce `semver:none` bumps and are internal).

**Rationale**: This matches the existing CHANGELOG.md format and conventional commit types. Excluding non-functional changes keeps the changelog focused on user-facing changes.

## Risks / Trade-offs

**[Risk] git-cliff misses PRs with unconventional titles** -> Mitigation: CI already enforces conventional commit format on PR titles (`check-pr-semver.sh`). Non-conforming titles are rejected before merge.

**[Risk] Sync-back step is forgotten after squash-merge** -> Mitigation: The runbook documents it explicitly. A future CI check could verify develop is not behind main, but that is out of scope for this change.

**[Risk] git-cliff version drift** -> Mitigation: Pin git-cliff version in CI and document the required version in the runbook. Use `cargo install git-cliff --version <pinned>` or a GitHub Action.

**[Risk] CHANGELOG.md conflicts during sync-back** -> Mitigation: Since git-cliff regenerates the file, conflicts can be resolved by re-running git-cliff after the merge. The runbook will include conflict resolution steps.

**[Risk] Pre-release hook failure blocks release** -> Mitigation: cargo-release does not commit if the hook fails. The maintainer fixes the issue and re-runs. This is existing behavior (make test already runs as a hook).

## Open Questions

1. **Should the `[Unreleased]` section be auto-generated on every develop commit, or only at release time?** Current decision: only at release time, via the pre-release hook. Continuous generation would require CI to commit back to develop, adding complexity.

2. **Should the release runbook include Docker image verification steps (pull + health check)?** Leaning yes, but depends on whether GHCR access is available in the release environment.
