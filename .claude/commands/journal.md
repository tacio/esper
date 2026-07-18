---
description: Append a timestamped why-not-what entry to docs/JOURNAL.md.
allowed-tools: Bash(date*), Bash(git log*), Bash(git diff*), Edit, Read
---

Append a new entry to `docs/JOURNAL.md` per the discipline in CLAUDE.md §Journal.

Rules:
- **Newest at the bottom** — append, never prepend or overwrite prior entries.
- Timestamp with the local time in **America/Bahia** (−03). Get it with
  `TZ=America/Bahia date '+%H:%M'` and match the surrounding entries' heading format
  (read the last ~30 lines of `docs/JOURNAL.md` first to mirror the exact style).
- Narrate the **why**, not the what — git records the what. A good entry explains the
  motivation, what was tried, what the result/number was, and (for a blocker) the
  **diagnosis and the fix**. Discoveries and dead-ends belong here too.
- Keep it to the real progress of this step; don't summarize the whole session.

Steps:
1. `Read` the tail of `docs/JOURNAL.md` to see the latest entry and exact heading format.
2. Get the timestamp: `TZ=America/Bahia date '+%H:%M'` (and the date if the format uses it).
3. Draft the entry from what actually happened this session (use `git diff --stat` /
   `git log --oneline -5` for grounding if helpful).
4. `Edit` `docs/JOURNAL.md` to append the entry at the end.

$ARGUMENTS, if present, is the gist of what to record — expand it into a proper entry.
