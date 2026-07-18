---
description: Propose the next Esper rung/step per the ROADMAP discipline (roadmap is source of truth).
allowed-tools: Bash(git log*), Bash(tail*), Read, Grep
---

Propose what to work on next, grounded strictly in the project's own direction.

1. Read the current state from `docs/ROADMAP.md` — the active Phase, the last
   GO/STOP/PARTIAL, and any explicitly-named unscheduled candidates or SHARPENED-but-
   unbuilt items. Cross-check the tail of `docs/JOURNAL.md` for the most recent thinking.
2. Propose **1–3 candidate next steps**, each with: the hypothesis, why it's the right
   next lever (per the roadmap's logic, e.g. a wall that was hit and the reroute), and
   what its GO/STOP gate would roughly test. Prefer candidates the roadmap already
   surfaces over inventing new direction.
3. If a candidate is a real experiment, suggest registering it with `/gate register`
   before building.

Remember: **`docs/ROADMAP.md` is the source of truth; per-session plan files are
ephemeral.** A genuine change of direction is a user decision, not something to assume —
surface it as a question rather than silently picking a new mission. $ARGUMENTS may
narrow the scope (e.g. a specific axis: Vision A / transfer / open-endedness).
