---
description: Local CI parity — batch idiom lint + mojo format + git-diff check across src/ and tests/.
allowed-tools: Bash(./esper fmt*), Bash(git diff*), Bash(git status*), Bash(python3 .claude/hooks/mojo_lint.py)
---

Run the on-demand, whole-repo counterpart to the per-edit hook — mirror what CI enforces
so nothing is a surprise at commit time.

1. **Idiom audit:** delegate to the `mojo-linter` agent (Haiku) to scan `src/*.mojo` and
   `tests/*.mojo` for the 1.0.0b2 idiom violations and hot-loop/POD smells (same rule set
   as `.claude/hooks/mojo_lint.py`). Report HARD findings first, then advisory.
2. **Format (CI parity):** run `./esper fmt`, then `git diff --stat`. CI runs
   `mojo format src/*.mojo tests/*.mojo` and fails on any resulting diff — so if the
   formatter changed files, they need to be committed (or the change reverted).
3. **Summary:** report whether the repo is CI-clean (no HARD idiom findings, no
   post-format diff) or list exactly what needs fixing.

This does not run the test suite — use `./esper fast` (quick gate) or `./esper suite`
(full) for that. $ARGUMENTS may narrow to specific files.
