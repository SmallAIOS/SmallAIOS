## 1. git-cliff Configuration

- [ ] 1.1 Add `cliff.toml` at the repository root with conventional commit type mappings: feat->Added, fix->Fixed, perf->Performance, refactor->Changed, revert->Reverted; exclude chore, ci, test, style, build, docs
- [ ] 1.2 Configure cliff.toml output template to match Keep a Changelog 1.1.0 format with `## [version] - date` headers and `### Group` subheaders
- [ ] 1.3 Configure cliff.toml to prefix changelog entries with the commit scope in bold (e.g., `**net:** description`) when scope is present
- [ ] 1.4 Verify git-cliff generates correct output by running it against existing git history and comparing with the current CHANGELOG.md 0.1.0 section

## 2. Makefile Changelog Target

- [ ] 2.1 Add `changelog` target to Makefile that invokes `git-cliff --config cliff.toml -o CHANGELOG.md`
- [ ] 2.2 Verify `make changelog` preserves existing versioned sections while updating the `[Unreleased]` section
- [ ] 2.3 Verify `make changelog` produces empty `[Unreleased]` when no new commits exist since the last tag

## 3. Version Bump Decision Script

- [ ] 3.1 Create `scripts/suggest-release-bump.sh` that finds the most recent `v*` tag and extracts commit messages between that tag and HEAD
- [ ] 3.2 Implement pre-1.0 bump rule aggregation: scan each commit message for conventional commit type, apply bump rules, output the maximum bump level to stdout
- [ ] 3.3 Print individual PR title classifications to stderr for maintainer review
- [ ] 3.4 Handle edge case: no commits since last tag outputs `none` with an informational message
- [ ] 3.5 Handle edge case: no existing tags (first release) defaults to `minor`

## 4. cargo-release Integration

- [ ] 4.1 Update `release.toml` pre-release-hook to run both `make test` and `make changelog` before the version bump commit
- [ ] 4.2 Create a `scripts/pre-release.sh` wrapper script that runs tests, generates changelog, and stages CHANGELOG.md for the version bump commit
- [ ] 4.3 Verify that `make release-dry-run BUMP=patch` correctly invokes the updated hook chain without modifying files
- [ ] 4.4 Verify that a hook failure (e.g., test failure) aborts the release without creating a commit or tag

## 5. Release Runbook

- [ ] 5.1 Create `docs/release-runbook.md` with pre-release checklist: CI green, no open blockers, run `suggest-release-bump.sh`, review suggested level
- [ ] 5.2 Document the develop-to-main merge workflow: create PR, squash-merge, sync main back to develop
- [ ] 5.3 Document the version bump and tagging steps: checkout main, `make release BUMP=<level>`, review commit/tag, push with `--follow-tags`
- [ ] 5.4 Document post-release verification: GitHub Release created, kernel artifacts present, Docker image on GHCR, merge main back to develop
- [ ] 5.5 Document rollback procedure: delete tag locally and remotely, revert version bump commit, sync to develop

## 6. CLAUDE.md Update

- [ ] 6.1 Update the Releasing section in CLAUDE.md to reference the release runbook and `make changelog` target
- [ ] 6.2 Add `suggest-release-bump.sh` to the Scripts section in CLAUDE.md
- [ ] 6.3 Document that git-cliff is a development dependency required for releases

## 7. Testing and Validation

- [ ] 7.1 Test full release dry-run cycle: `suggest-release-bump.sh` -> `make changelog` -> `make release-dry-run BUMP=<suggested>`
- [ ] 7.2 Verify CHANGELOG.md output format matches Keep a Changelog with correct group headers and scope prefixes
- [ ] 7.3 Verify release.yml changelog extraction (`awk` command) works with git-cliff-generated sections
- [ ] 7.4 Verify the sync-back workflow: squash-merge develop->main, merge main->develop, confirm `git log develop..main` returns zero commits
