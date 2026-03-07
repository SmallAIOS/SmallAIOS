# Release Runbook

Step-by-step guide for releasing a new version of SmallAIOS.

## Pre-release Checklist

- [ ] CI is green on `develop` (all checks pass)
- [ ] No open blockers or critical issues
- [ ] All planned PRs for this release are merged to `develop`
- [ ] Run `./scripts/suggest-release-bump.sh` and review the suggested bump level
- [ ] Confirm the bump level matches expectations (review individual PR classifications)

## 1. Create Release PR (develop → main)

```bash
# Ensure develop is up to date
git checkout develop && git pull origin develop

# Create PR
gh pr create --base main --head develop \
  --title "chore: release develop to main" \
  --body "Release PR. Squash-merge to main."
```

Wait for CI to pass on the PR.

## 2. Squash-merge to main

Use GitHub's **Squash and merge** button on the PR, or:

```bash
gh pr merge <PR_NUMBER> --squash --subject "chore: release develop to main"
```

## 3. Sync main back to develop

**This step is critical.** Without it, develop and main histories diverge.

```bash
git checkout main && git pull origin main
git checkout develop && git pull origin develop
git merge main
git push origin develop
```

Verify sync: `git log develop..main` should return zero commits.

## 4. Version bump and tag

```bash
git checkout main && git pull origin main

# Check suggested bump
./scripts/suggest-release-bump.sh

# Dry run first
make release-dry-run BUMP=<patch|minor>

# Execute (runs tests, generates changelog, bumps version, commits, tags)
make release BUMP=<patch|minor>
```

Review the commit and tag:
```bash
git log -1
git tag -l --sort=-v:refname | head -3
```

## 5. Push to trigger release workflow

```bash
git push origin main --follow-tags
```

This triggers `.github/workflows/release.yml` which:
- Builds x86-64, AArch64, RISC-V kernels
- Creates a GitHub Release with binary assets
- Publishes Docker images to GHCR

## 6. Post-release verification

- [ ] GitHub Release exists with correct version and release notes
- [ ] Kernel binaries are attached (x86_64, aarch64, riscv64)
- [ ] Docker image is published: `docker pull ghcr.io/smallaios/smallaios:<version>`
- [ ] Docker health check passes: `docker run --rm ghcr.io/smallaios/smallaios:<version> --health-check`

## 7. Post-release sync back

Merge the version bump commit back to develop:

```bash
git checkout develop && git pull origin develop
git merge main
git push origin develop
```

## Rollback

If a release needs to be reverted:

```bash
# Delete the tag locally and remotely
git tag -d v<version>
git push origin :refs/tags/v<version>

# Revert the version bump commit on main
git checkout main
git revert HEAD
git push origin main

# Sync to develop
git checkout develop && git merge main && git push origin develop
```

Then delete the GitHub Release manually via the web UI or:
```bash
gh release delete v<version> --yes
```
