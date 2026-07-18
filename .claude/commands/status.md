---
description: One-glance Esper state — active rung, latest journal entry, recent commits, working tree.
allowed-tools: Bash(git log*), Bash(git status*), Bash(tail*), Bash(sed*), Read
---

Give a compact status of where Esper is right now. Gather and summarize:

1. **Active direction / rung** — the current status line(s) from `docs/ROADMAP.md`. Read
   the relevant section (the active Phase / rung and its last GO/STOP/PARTIAL), don't
   dump the whole 900-line file.
2. **Latest journal entries** — the last 1–2 entries of `docs/JOURNAL.md` (tail it), to
   see what was last worked on and why.
3. **Recent commits** — `git log --oneline -5`.
4. **Working tree** — `git status --short` (uncommitted work, untracked files).
5. **Open gates** — if `docs/gates/` exists, list any gate whose Status is still
   `REGISTERED` (unresolved experiment).

Then write a 4–6 line summary: what rung is active, what the last verdict was, what's
uncommitted, and what the obvious next action is. Keep it scannable.
