---
name: esper-discipline-review
description: Review an Esper code diff against the project's judgment-level discipline rules (no runtime DSL / memory-selector, no backprop, POD-over-arena, no hot-loop alloc, stone-soup / single-seed / unfalsifiable-claim smells, journal + local-first git). Use before committing a non-trivial change, or when asked to review Esper work for discipline. Complements the mojo_lint.py hook, which already blocks the deterministic idiom violations.
---

# Esper discipline review

You are reviewing a diff for the discipline that keeps Esper honest and emergent. The
deterministic Mojo-idiom rules are already enforced by the `mojo_lint.py` PreToolUse
hook — your job is the **judgment-level** rules a regex can't decide.

## Do this

1. **Get the diff.** Prefer the working-tree/staged diff (`git diff`, `git diff --staged`)
   or the range the user names. Focus the review on changed lines and their blast radius.
2. **Read the rule catalog:** `references/rules.md` (this skill's directory). Check the
   diff against every rule that its files/hunks could plausibly touch. Use
   `references/mojo-idioms.md` for anything the hook might have missed on unchanged lines.
3. **For a deep or high-stakes review, delegate** to the `discipline-reviewer` agent
   (Opus) via the Agent tool — it is scoped to the module map and reasons about
   emergence/measurement violations. Do this when the diff touches the ES core, a memory
   family, the `Domain`/`Memory` seams, or claims a milestone/gate result. For a small,
   localized diff, review inline.
4. **Report findings ranked by severity.** For each: the rule (by number/name from
   `rules.md`), the file:line, why it violates, and the concrete fix. Separate **blocking**
   (a real violation) from **advisory** (a smell worth a second look).
5. **If the diff claims an empirical result** (a milestone, a GO/STOP, a number), apply
   the honest-measurement rules hardest: is it a single COLD fit (no stone soup)? pinned
   across ≥2–3 seeds? gated on the metric its consumer reads? does a refuting control
   exist? A scaffolded pass must be reported as a **negative result**, not a win.

## The spirit
The two north-star violations to never let through:
- **A DSL on the runtime path** (hand-coded transform selection, or a runtime
  memory-selector) — the engine must *learn* the mapping, not choose among primitives.
- **A dishonest measurement** (stone soup, single seed, consumer-mismatched metric, or a
  claim with no control that could refute it) — held-out generalization is the only currency.

Be specific and cite the rule. If the diff is clean, say so plainly — don't invent findings.
