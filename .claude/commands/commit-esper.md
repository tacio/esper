---
description: Sanctioned Esper commit flow — format, gate, journal-check, then commit to master (never push).
allowed-tools: Bash(./esper fmt*), Bash(./esper fast), Bash(git add*), Bash(git commit*), Bash(git status*), Bash(git diff*), Bash(git log*), Bash(.claude/hooks/journal_check.sh)
---

Commit the current work to `master` following Esper's discipline. Do **not** push
(the git_guard hook enforces this; do not try to work around it).

Run these steps in order and report what happened at each:

1. **Format (CI parity):** run `./esper fmt`. Then `git diff --stat` — if formatting
   changed files, that's expected; they'll be included in the commit.

2. **Fast gate:** run `./esper fast`. This is the ~2-min local subset at full budget.
   - If it **fails**, STOP: show the failing output and do not commit. Fixing the
     failure (or recording an honest negative) comes first.
   - Skip this step only if the change touches no `.mojo` under `src/`/`tests/` (e.g.
     docs-only) — say so explicitly when you skip it.

3. **Journal discipline:** run `.claude/hooks/journal_check.sh`.
   - If it prints `JOURNAL-MISSING`, pause and run `/journal` to append an `HH:MM`
     entry (the *why*) before committing. Do not commit src changes with no journal.

4. **Stage & commit to master:**
   - Confirm `git rev-parse --abbrev-ref HEAD` is `master` (git_guard blocks otherwise).
   - `git add -A` the intended files (review `git status` first; don't sweep in stray
     scratch files).
   - Write a **milestone-style, verdict-first** commit title matching the repo's house
     style — e.g. `Route X: <what> — GO` / `STOP` / `PARTIAL`, or `<Area>: <outcome>`.
     The body is a short narrative: what was tried, the number that moved, and which
     docs were touched (JOURNAL/ROADMAP/RESEARCH-NOTES). Reference the pre-registered
     gate if one applies.
   - End the message with the trailer:
     `Co-Authored-By: Claude <current-model> <noreply@anthropic.com>`
     (use the model actually driving this session).

5. **Confirm:** show `git log --oneline -1` and `git status`. Never run `git push`.

$ARGUMENTS may contain a commit title/summary to use; if empty, derive the title from
the diff and the latest JOURNAL entry.
