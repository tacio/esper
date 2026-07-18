---
name: esper-gate
description: Pre-register and evaluate a GO/STOP experiment gate for Esper the honest way — write the success criteria (metric, its consumer, seed count, refuting control) BEFORE the run, then score the result against exactly those criteria. Use when starting a milestone/rung experiment or deciding GO/STOP/PARTIAL on one. Encodes the "held-out generalization is the only currency; a scaffolded pass is a negative result" discipline.
---

# Esper experiment gate

A gate is a **pre-registered** commitment: you write down what would count as success
*before* you see the result, then you are bound to it. This is what makes a milestone
honest instead of a post-hoc rationalization.

Gates live in `docs/gates/` as `NNN-<slug>.md` (create the dir if absent). They are the
durable record; the JOURNAL references them and the commit title states the verdict.

## `/gate register` — before the run
Produce a gate file with these fields, and nothing vague:
- **Hypothesis** — the one sentence being tested (what should carry / emerge / transfer).
- **Metric** — the exact number, and **its consumer** (the downstream code/decision that
  reads it). Held-out generalization by default: fit on train pairs, score the unseen
  test pair.
- **GO / STOP / PARTIAL thresholds** — concrete numbers, decided now.
- **Seeds** — the ≥2–3 independent draws the number will be pinned across (not just
  `seed(0)`; that is repeatability, not robustness).
- **Refuting control** — the condition under which the hypothesis would produce a "no".
  If you can't name one, the gate is unfalsifiable — fix it before running.
- **Anti-stone-soup clause** — the fit must be a single COLD fit: no per-task hand-staging,
  no mid-fit boosts, no bespoke tuning. Note explicitly what would count as scaffolding
  here, so a later scaffolded pass is recognized as a **negative result**.
- **Status:** `REGISTERED` (+ date).

## `/gate eval` — after the run
Score the actual result against the registered criteria, changing nothing about them:
- Report the metric per seed and the aggregate; state GO / STOP / PARTIAL by the
  pre-committed thresholds.
- Confirm the refuting control was in place and what it showed.
- Confirm no scaffolding crept in. If it did, the verdict is a **documented negative**,
  regardless of the raw number.
- Update the gate file's **Status** to the verdict (+ date) and append a one-line
  rationale. Then prompt to record it in `docs/JOURNAL.md` and reflect it in
  `docs/ROADMAP.md`, and to commit with a verdict-first title via `/commit-esper`.

## Delegate the design
For designing the metric/control/thresholds (the part that needs real judgment about what
would refute the hypothesis), use the `gate-designer` agent (Opus). Keep scoring tied
strictly to the registered file — do not move the goalposts after seeing the result.
