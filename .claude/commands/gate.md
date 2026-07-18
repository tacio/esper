---
description: Pre-register (register) or score (eval) an Esper GO/STOP experiment gate honestly.
allowed-tools: Read, Write, Edit, Bash(ls*), Bash(cat*), Bash(git log*)
---

Manage a pre-registered experiment gate. Invoke the `esper-gate` skill for the full
discipline, then act on the sub-command in $ARGUMENTS:

- **`/gate register <slug or hypothesis>`** — create `docs/gates/NNN-<slug>.md` (next
  number; `ls docs/gates/` to find it) with the registered criteria: hypothesis, metric
  + its consumer, GO/STOP/PARTIAL thresholds, ≥2–3 seeds, a refuting control, and the
  anti-stone-soup clause. Use the `gate-designer` agent (Opus) to design the
  metric/control/thresholds. Set Status: `REGISTERED` + today's date. Do **not** run the
  experiment here.

- **`/gate eval <NNN or slug>`** — read the registered gate, score the actual result
  against exactly those criteria (per-seed + aggregate), declare GO/STOP/PARTIAL, confirm
  the control and no scaffolding, and update the file's Status. Then prompt to journal it,
  reflect it in `docs/ROADMAP.md`, and `/commit-esper`.

- **`/gate list`** — `ls docs/gates/` and summarize each gate's slug + Status.

If $ARGUMENTS is empty, ask which sub-command. Never edit a gate's registered criteria
during `eval` — moving the goalposts after seeing the result is the exact failure this
guards against.
