#!/usr/bin/env bash
# Copyright 2026 SmallAIOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Gate parity check: reconcile the two independent PR-gating mechanisms.
#
# SmallAIOS gates merges through two lists that are maintained separately and
# can silently drift apart:
#
#   1. The `change-gates` meta-job in .github/workflows/ci.yml, whose `needs:`
#      array names the jobs the pipeline *intends* to be mandatory.
#   2. GitHub branch protection's required-status-check list, which is what
#      actually stops a merge.
#
# Only (2) is enforced. A job in (1) but not (2) looks mandatory in the workflow
# and in review, yet cannot block anything -- the failure mode this script
# exists to catch. `change-gates` itself is only meaningful if it appears in (2).
#
# A job that sets `continue-on-error: true` always reports success, so listing
# it in (2) is equally toothless; those are flagged separately.
#
# Requires: gh (authenticated), python3.
# Exits 0 when the two lists agree, 1 on drift, 2 on operational failure.

set -euo pipefail

BRANCH="${1:-develop}"
REPO="${REPO:-SmallAIOS/SmallAIOS}"
WORKFLOW=".github/workflows/ci.yml"

command -v gh >/dev/null 2>&1 || { echo "error: gh not found on PATH" >&2; exit 2; }
[ -f "$WORKFLOW" ] || { echo "error: $WORKFLOW not found (run from the repo root)" >&2; exit 2; }

if ! required_json=$(gh api "repos/${REPO}/branches/${BRANCH}/protection" \
        --jq '.required_status_checks.contexts' 2>/dev/null); then
  echo "error: could not read branch protection for ${REPO}@${BRANCH}." >&2
  echo "       Needs a token with admin:repo scope, and the branch must be protected." >&2
  exit 2
fi

WORKFLOW="$WORKFLOW" REQUIRED_JSON="$required_json" BRANCH="$BRANCH" python3 <<'PY'
import json, os, re, sys

wf = open(os.environ["WORKFLOW"]).read()
required = set(json.loads(os.environ["REQUIRED_JSON"]) or [])
branch = os.environ["BRANCH"]

# job id -> (display name, is_advisory)
jobs = {}
for blk in re.split(r"\n  (?=[A-Za-z0-9_-]+:\n)", wf.split("jobs:", 1)[1]):
    m = re.match(r"\s*([A-Za-z0-9_-]+):", blk)
    if not m:
        continue
    name = re.search(r"^\s{4}name:\s*(.+)$", blk, re.M)
    coe = re.search(r"^\s{4}continue-on-error:\s*(\S+)", blk, re.M)
    jobs[m.group(1)] = (name.group(1).strip() if name else m.group(1),
                        bool(coe) and coe.group(1) == "true")

needs_m = re.search(r"change-gates:.*?needs:\s*\[(.*?)\]", wf, re.S)
if not needs_m:
    print("error: could not locate the change-gates `needs:` array", file=sys.stderr)
    sys.exit(2)
needs = [n.strip() for n in needs_m.group(1).split(",") if n.strip()]

nm = lambda jid: jobs.get(jid, (jid, False))[0]
gate_names = {nm(n) for n in needs}

intended_not_enforced = [(n, nm(n)) for n in needs if nm(n) not in required]
enforced_not_intended = sorted(required - gate_names)
toothless = sorted(n for n in required
                   for jid, (disp, adv) in jobs.items() if disp == n and adv)
meta_enforced = "Change Gates" in required

print(f"branch:              {branch}")
print(f"change-gates needs:  {len(needs)}")
print(f"required checks:     {len(required)}")
print(f"'Change Gates' itself required: {'yes' if meta_enforced else 'NO'}")

if intended_not_enforced:
    print(f"\nIntended as mandatory but CANNOT block a merge ({len(intended_not_enforced)}):")
    for jid, disp in intended_not_enforced:
        print(f"  {jid:24} {disp}")
    if not meta_enforced:
        print("\n  Fix: add 'Change Gates' to the required-status-check list and all of the")
        print("  above become enforcing at once, without listing each individually.")

if enforced_not_intended:
    print(f"\nRequired but absent from change-gates' needs ({len(enforced_not_intended)}):")
    for n in enforced_not_intended:
        print(f"  {n}")
    print("  (Fine when deliberate -- they are enforced directly. Listed for review.)")

if toothless:
    print(f"\nRequired but continue-on-error: true -- always reports green ({len(toothless)}):")
    for n in toothless:
        print(f"  {n}")
    print("  Drop continue-on-error to make these real gates.")

drift = bool(intended_not_enforced or toothless)
print("\n" + ("DRIFT: the two gate lists disagree." if drift
              else "OK: gate lists agree and no required check is continue-on-error."))
sys.exit(1 if drift else 0)
PY
