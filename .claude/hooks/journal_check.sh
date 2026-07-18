#!/usr/bin/env bash
# Advisory JOURNAL discipline check (CLAUDE.md §Journal): if src/ changed but
# docs/JOURNAL.md was not touched, real progress may be going unrecorded.
#
# Prints a one-line verdict and exits:
#   0  = OK (no src change, or JOURNAL.md is among the changes)
#   0 + "JOURNAL-MISSING" on stdout = src changed with no JOURNAL.md edit
#
# Never blocks — it is meant to be read by /commit-esper, not to fail a build.
# Considers both staged and unstaged changes plus untracked files.
set -euo pipefail

root="${CLAUDE_PROJECT_DIR:-$(git rev-parse --show-toplevel 2>/dev/null || pwd)}"
cd "$root"

changed="$(git status --porcelain | awk '{print $2}')"

src_touched="$(printf '%s\n' "$changed" | grep -E '^src/.*\.mojo$' || true)"
journal_touched="$(printf '%s\n' "$changed" | grep -E '^docs/JOURNAL\.md$' || true)"

if [[ -n "$src_touched" && -z "$journal_touched" ]]; then
  echo "JOURNAL-MISSING: src/ changed but docs/JOURNAL.md was not updated."
  echo "  Run /journal to append an HH:MM entry (the *why*) before committing."
  exit 0
fi

echo "JOURNAL-OK"
exit 0
