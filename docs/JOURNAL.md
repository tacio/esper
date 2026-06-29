# Esper development journal

A running, timestamped narrative of progress, discoveries, and blockers — so the
project has a complete story later. Newest entries at the bottom. Times are local
(America/Bahia, −03).

Companion docs: the strategy/roadmap lives in the plan file
(`~/.claude/plans/let-s-create-a-roadmap-ticklish-adleman.md`, user-local); the
theory is distilled in `docs/NL-summary.md`.

---

## 2026-06-29

**18:24 — Repo housekeeping.** Committed the vendored Modular/Mojo agent skills,
lockfiles, and the roadmap docs. Removed `NL.pdf` from git (kept locally, gitignored)
and switched doc references to its source URL (https://abehrouz.github.io/files/NL.pdf).

**~18:40 — Roadmap → implementation design approved.** Decided the spine: fast weights
parameterize a *learned* grid→grid operator fit in-context by the ES on demonstration
pairs (no hand-coded DSL); slow weights are the meta-learned prior. Foundational chunk =
M1–M4 + migration. Two design refinements fell out of reading the source: (1) operator
*execution* goes in `hope.mojo` but the demo *fitness* and `forward_with_learning` must
move to `esper_evolution.mojo` to keep the import DAG acyclic; (2) `apply_operator` needs
no scratch buffer (it writes a separate output, no aliasing).

**~18:45 — Decision: CPU-only for Phase A.** Confirmed the codebase has zero GPU code;
everything is CPU SIMD (`simd_width_of`, host `alloc`/arena). The parallelism that would
benefit from GPU is *across ES samples*, not inside the per-cell gather — and at Phase-A
sizes (16-param vectors, ≤30×30 grids) GPU launch/transfer overhead would dominate.
Staying CPU; revisit only when batching many tasks / scaling N hard.

**~18:50 — M1 done.** Added the operator layout (`OP_DIM=16` = 6-param centered affine +
10-entry colour LUT), `apply_operator`, `seed_identity_operator` to `hope.mojo`.
`tests/test_operator.mojo` passes: hand-set weights reproduce identity/flip_h/transpose/
recolor exactly. No regression in existing tests.

**~18:55 — M2+M3 code landed, then hit the keystone blocker.** Reworked `ESWorkspace`
(param-sized ES vectors + a grid-sized `op_output` scratch), added `operator_fitness`
(−MSE over demos + L2 anchor), changed `evolve_fast_weights` to take `(slow, demos, …)`,
deleted the `evaluate_primitives` surrogate. First `test_demo_fitness` run: fitness
improved (−21.7 → −6.3) but **held-out match = 0** — stuck in an intermediate plateau.

**DISCOVERY — the rounded gather is the enemy.** Nearest-neighbour rounding in the
operator creates wide flat plateaus where the ES has no gradient: params random-walk and
either get stuck or diverge to millions, and nothing pins `a0` at 1 (so the row mapping
scrambles). Fix: made the operator **smooth** — bilinear interpolation for the geometry
gather + linear interpolation over the colour LUT. Integer-valued params still reproduce
the transforms exactly (verified: `test_operator` still passes), but now every parameter
has a real gradient. Smoothing alone didn't fix divergence; small step sizes
(`alpha≈0.02–0.1`) stopped the explosion.

**BLOCKER (current) — test data, not the optimizer.** With small steps the ES now
converges *stably* but to a consistent **wrong** matrix (`A≈[[0.45,0.55],[−0.87,0.19]]`)
regardless of step size. Diagnosis: my demo grids are linear ramps (`k*3 % 10`), which
have affine symmetries — flip_h isn't uniquely determined by them. Next: switch the demos
to random colour grids (as `synth_tasks.py` already does) so flip_h is identifiable, then
re-tune.

**19:05 — M3 keystone RESOLVED.** Two fixes on top of the smooth operator: (1) random demo
grids (flip_h identifiable); (2) **annealed ES** — decay sigma 0.3→0.01 and alpha 0.1→0.003
over ~4000 iters: explore wide early to find the transform's structure, then settle precisely
onto the integer parameters that reproduce it exactly. With fixed sigma the params hovered at
~0.984 (held-out ~0.87); annealing reaches a0≈0.999, a3≈−0.990 and **held-out flip_h = 0.99**.
Centralized the recipe as `fit_operator` (the shared annealed loop) so every caller uses one
schedule. `tests/test_demo_fitness.mojo` passes (final fitness −0.02, held-out 0.994, ~5s).
Lesson learned: the ES needs a smooth landscape *and* an exploration→exploitation schedule;
neither alone was enough.

**19:12 — M4 done.** Moved `forward_with_learning` into `esper_evolution.mojo` (import-DAG
reasons), rewrote it to fit the operator via `fit_operator` then apply it to the test input
(dropped the old `op: Int` primitive selector). Updated `main.mojo` to build an `OP_DIM` node,
seed both slow (prior) and fast (init) to identity, and learn flip_h end-to-end. New
`tests/test_forward_learning.mojo` uses fit-once / generalize-to-many: held-out average 0.99.

**19:15 — M5 migration done; foundational chunk GREEN.** Deleted the `evaluate_primitives`
surrogate's consumers: removed `src/benchmark.mojo` (memorized a known target — exactly the
thing we killed; replaced by `src/arc_solve.mojo` at M7) and the two old target-based ES tests
(`test_es`, `test_es_convergence`, subsumed by the new operator tests), and dropped the
benchmark step from `run_tests.sh`. Full suite passes in ~15s: arena, demo-fitness, fitness,
forward-learning, io, operator, plus the end-to-end driver.

**Status:** M1–M4 + migration complete. The engine now genuinely *learns* a transformation
from demonstrations and generalizes to held-out inputs — the first real proof-of-learning.
Next up: M5 task loader, M6 shape handling, M7 held-out generalization benchmark over the full
expressible subset.
