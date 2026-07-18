# Esper discipline rules — the review catalog

The deterministic idiom rules (see `mojo-idioms.md`) are caught by the `mojo_lint.py`
hook. This file catalogs the **judgment-level** rules a regex can't decide — the ones a
reviewer must reason about on a diff. Grouped by the value they defend
(CLAUDE.md §Hard constraints, ROADMAP.md §Values / §Working principles).

## A. Emergence over hand-coding (no DSL on the runtime path)
1. **No symbolic DSL / hand-coded primitive selection on the inference path.** A
   `prim_*` / `apply_primitive`-style branch, or any hand-written case-split that *is*
   the transform, is a violation. The grid→grid mapping must be a **learned** operator
   fit by ES on demonstrations. Symbolic transforms live only in offline
   `tools/synth_tasks.py` (ground truth to rediscover), never in `src/`.
2. **No runtime memory-selector.** A switch/dispatch that picks among memories at
   runtime is "a DSL over memories" — itself a violation. Growth is **additive and
   trait-based**: each memory is a compile-time choice measured on the subset it
   expresses. The generic ES / two-timescale core stays untouched.
3. **No backprop / gradient-descent / autodiff.** Learning is Evolution Strategies. An
   analytic gradient of the loss w.r.t. weights, or an ML framework, is out.

## B. Honest measurement (held-out generalization is the only currency)
4. **The bar is a single COLD fit.** Per-task hand-staging, mid-fit parameter boosts, or
   bespoke per-task tuning = **"stone soup"** → it must be recorded as a documented
   **negative result**, not a milestone. A scaffolded pass is a *negative*.
5. **A margin must survive a different draw.** Every gated number is pinned from **≥2–3
   independent seeds/draws**. `seed(0)` gives repeatability, not robustness. A claim on
   a single seed is not yet a claim.
6. **Match the metric to its consumer.** Gate on the number that the downstream code
   actually reads — not a proxy that looks similar.
7. **Every claim needs a control that could refute it.** If nothing in the experiment
   could have produced a "no," the "yes" is unfalsifiable and doesn't count.
8. **Report the real number, including 0%.** Held-out generalization (fit on train
   pairs, score the unseen test pair) is the only success metric.

## C. Architecture integrity
9. **POD over the arena.** A struct placed into `HopeArena` (e.g. `HopeNode`) that gains
   an owning member (String, List, owned pointer) breaks the move-only bump-allocation
   invariant. Reference other nodes by arena index.
10. **No dynamic allocation in hot loops.** `alloc` / `.free()` / `List(...)` inside an
    ES iteration (`evolve_fast_weights` / `fitness*` / `update_fast_weights` /
    `calculate_fitness`) must become a reused `ESWorkspace` slot.
11. **Move the search off the fast weights.** ES fits only small/slow params; fast
    adaptation is the memory's own write rule over the demos, not an ES search.
12. **Domain-agnostic core.** The ES reaches metrics only through a `Domain`
    (`arc_io.mojo`), never ARC directly; `Domain` / `Memory` traits stay separable.

## D. Process / governance
13. **Journal every step.** Real progress / a discovery / a blocker with no matching
    `HH:MM` entry appended to `docs/JOURNAL.md` (newest-at-bottom, *why* not *what*) is a
    process miss.
14. **Local-first git.** Commit **directly to `master`** — no feature branch, no PR,
    **never push**. Keep the `Co-Authored-By` trailer. (Enforced by the git_guard hook.)
15. **ROADMAP is the source of truth.** Direction changes land in `docs/ROADMAP.md`;
    per-session plan files are ephemeral and not authoritative. At a real wall, pause and
    survey the literature into `docs/RESEARCH-NOTES.md`, then reroute.
