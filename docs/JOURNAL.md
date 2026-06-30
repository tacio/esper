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

**~19:40 — M5 done (task bundles + loader).** Chose **per-task binary bundles** (`.task`) over the
roadmap's per-grid-files-plus-text-manifest, because Mojo is weak at both directory globbing and
text parsing — the shell globs `*.task` into the driver's argv instead (like the old benchmark).
`arc_compiler.py`: factored `_write_grid` out of `_save_grid`, added `_save_task`. `hope.mojo`:
`ArcTask{train,test: List[ArcTaskPair]}`. `arc_io.mojo`: factored `_read_grid_block` (shared by
`load_arc_grid`) + `load_arc_task`. `tests/test_task_loader.mojo` round-trips a fixture bundle.

**~19:55 — M6 done (shape guard).** The operator is same-shape; `operator_fitness` now penalizes a
demo whose output area differs from its input area (instead of an OOB compare), steering the ES
away from inexpressible shape-changing tasks. `tests/test_shape.mojo` covers it.

**20:36 — M7 convergence: two structural fixes before the benchmark.** Before wiring the benchmark
I checked whether the flip_h-tuned schedule fits the *whole* expressible subset. flip_h/flip_v/
transpose converged (~0.97–1.0, even transpose's four-param move), but **recolor failed (~0.18)**.
Two distinct discoveries:

1. *Parameter-scale mismatch → preconditioned ES.* The colour-LUT entries need to travel much
   further than the affine entries, so one global step size can't serve both (small steps starve
   colour; large steps destabilize geometry). Normalizing the colour LUT to ~unit scale flipped the
   problem (then sigma was too coarse for the tight 1/9 palette spacing and scrambled colour). Real
   fix: a **per-group step `scale`** in `ESWorkspace` (colour group gets `COLOR_SCALE`), applied to
   both the ES perturbation and update — a diagonal preconditioner. Geometry went to a clean 1.0;
   recolor improved but stuck at ~0.8.

2. *Geometry↔colour coupling → colour-then-gather.* recolor's last two palette entries (8→9, 9→0,
   the extreme/wrap colours) stayed wrong because the affine never settled to *exactly* identity, so
   the bilinear gather blended neighbours and corrupted which palette entry each cell read.
   Restructured `apply_operator` to **apply the colour LUT to the four integer input corners BEFORE
   the bilinear blend** (instead of colouring the blended output). Now each LUT entry fits
   independently of geometry precision, while the gather stays bilinear-smooth for the flips — and
   it still needs no scratch (colour the corner reads inline). Result: **all four transforms reach
   1.0 held-out with the default 4000-iter schedule.** Lesson: the operator's two sub-systems
   (geometry, colour) must be decoupled in *both* the optimizer (per-group scale) and the forward
   pass (colour before gather).

Next: finish M7 — `generate_task_groups` (synth bundles), `arc_solve.mojo` (held-out solve rate +
train/test gap), `test_generalization.mojo` (whole subset), wire into `run_tests.sh`.

**20:47 — M7 done; Phase A (M1–M7) COMPLETE.** Built `generate_task_groups` (emits `.task`
bundles), `src/arc_solve.mojo` (fits each task on train, scores held-out test, reports solve rate +
train/test gap, raises on 0), and `tests/test_generalization.mojo` (self-contained whole-subset
proof). One last fragility surfaced: **transpose was seed-dependent** — it converged in isolation
but landed in a local optimum from some RNG states (5/6 across seeds), because all four affine
params must move together (a 90° change). Fix: widen the shared exploration, `FIT_SIGMA0` 0.3 → 0.5
— transpose went to 6/6 across seeds and the easy transforms were unaffected. Full suite green
(~62s): the whole expressible subset {flip_h, flip_v, transpose, recolor} is learned to **held-out
1.0 with train/test gap 0.0** — genuine generalization, no overfit. The suite is heavier now (ES
fits in 3 tests + main + the driver); acceptable, trimmable later if needed.

**Where Phase A stands:** the engine learns every transform it can express, purely from
demonstrations, scored only on unseen inputs. Remaining roadmap: M8 (ingest real ARC-AGI 2 JSON →
bundles + honest eval) and M9 (meta-learn the slow prior — the second timescale). The expressible
subset is geometry+colour; real ARC-AGI 2 will mostly fall *outside* it (objects, counting,
symmetry, shape change) — M8's honest number is expected to be low, and that's the point of the
held-out metric.

**21:34 — M8 done: real ARC-AGI 2 ingest + the honest held-out number.** User cloned the
ARC-AGI-2 repo and symlinked its `data/` into the repo (`arg-agi-2-data` → training/ 1000 +
evaluation/ 120 task JSONs, both splits ship the test output, so held-out scoring is real). Added a
batch ingest to `arc_compiler.py`: `compile_task_to_bundle` (one task JSON → one `.task` bundle via
the existing `_save_task`; drops a task that has no scorable test output) + `compile_arc_dir`
(directory → `{task_id}.task`) + a `__main__` CLI (`python src/arc_compiler.py <json_dir>
<out_dir>`). Compiled both splits (0 skipped — the corpus has outputs).

Two fixes the synth subset never exercised:
- *Shape-change OOB guard in `arc_solve.mojo`.* The operator is same-shape, but real ARC is often
  not (only 680/1000 training tasks are fully same-shape). The driver scored
  `exact_match(pred, test.output, rows*cols)` using the **input** dims — on a shrinking task that
  reads past the (smaller) output buffer. Guard: a test/train pair whose output area ≠ input area
  honestly scores 0 (skip the compare) instead of crashing. This is where the honest number stays
  low, by construction.
- *`--report` honest-eval mode.* The driver `raise`s on 0 solved as a CI regression signal for the
  synth bundles (which MUST solve some). On the real corpus 0 solved is a legitimate honest result,
  not a crash — `--report` (first arg) suppresses the raise. `run_tests.sh`'s synth path is
  unchanged (no flag → still raises).

**The honest ARC-AGI-2 evaluation number (120 public-eval tasks, fit on train / scored on the
unseen test):**

    Solved (held-out >= 0.99):  0 / 120   (0.0%)
    mean held-out exact-match:  0.033
    held-out >= 0.5:   2 tasks      0 < held-out < 0.5:  19 tasks      held-out == 0:  99 tasks

`exact_match` is *fraction of cells correct*, not all-or-nothing — so the mean 0.033 with **0
tasks fully solved** is the truthful reading: the engine never reproduces a whole held-out grid.
The two outliers are honest non-solves, not near-misses: `5545f144` (held 0.91 / train 0.0 / gap
−0.91) is a near-identity *test* pair the seeded identity operator matches by luck while failing
its own demos; `7c66cb00` (held 0.59, gap 0.036) is the closest thing to a genuine in-subset task.
9 tasks overfit (train fit > 0.3, held-out < 0.1) — the ES finds an affine/colour that fits the
demos but doesn't transfer. ~7s/task; the full eval ran in ~14 min, no crashes across all 120
(the shape guard held). This is exactly the predicted outcome: ARC-AGI-2 is almost entirely
outside the geometry+colour subset, and the held-out metric reports that with no inflation.
Raw dump: `scratch/arc2_eval_results.txt` (gitignored, reproducible). Next: M9 (meta-learn the
slow prior — the second timescale) and Phase B (strip the structural priors).

**23:39 — M9 done: the slow prior is now LEARNED (the second timescale turns on).** Through M8
`slow` was a fixed identity anchor — HOPE's outer timescale was inert. M9 adds **Reptile-style
meta-learning** (chosen over an ES-on-slow, which would be ES-in-ES and blow the cost budget):
`reptile_meta_train` loops over a family of tasks, fits each task's `fast` in-context FROM the
current prior (reusing `fit_operator` unchanged), then nudges `slow += META_LR·(fast − slow)`. Two
small SIMD/FMA weight helpers (`copy_weights`, `reptile_update`); one `fast` buffer allocated once,
the caller's `ESWorkspace` reused — no per-iteration alloc. Import DAG untouched.

**DISCOVERY — a warm start is worthless under a wide-exploration schedule.** First test attempt
asserted "meta beats identity at a fixed budget" using the *default* schedule (`FIT_SIGMA0=0.5`,
`EVAL_ITERS=400`). Both priors scored **1.0** — gap 0.0. Two reasons: (1) flip_h is a one-parameter
move the annealed ES nails fast even from cold; (2) more fundamentally, the wide `sigma0=0.5`
*explores the whole space first*, washing out wherever you started — so a good init confers no
advantage. A prior only helps if you can then afford a CHEAP, LOCAL (exploit) fit. So the eval must
be a *narrow* fit, and the claim becomes: with the prior a narrow local fit lands the answer;
without it the same narrow fit can't reach the transform's basin. Probed the regime (scratch,
since deleted): at `sigma0=0.12 / 300 iters`, an exact-flip_h prior fits to 1.0 while a cold prior
gets ~0; widen to `sigma0≥0.2` and cold solves too (gap gone) — the narrowness *is* the mechanism.

Split the schedules accordingly: **meta-train keeps the wide `FIT_SIGMA0=0.5`** (it must *discover*
the family's operator from cold), **eval uses `EVAL_SIGMA0=0.12`, `EVAL_ITERS=300`** (fast
in-context adaptation). `tests/test_meta_prior.mojo`: meta-train `slow` on 8 flip_h tasks, then fit
a fresh unseen-grid flip_h task at the narrow eval budget from both priors — **identity-prior
held-out 0.0, meta-prior 1.0, gap 1.0** (asserts ≥0.95 and a ≥0.3 margin; both clear by a mile).
The fit only ever sees train; held-out scored after — no leak. Full suite green (~58s).

**Honesty note (why single-family).** Within the 16-param affine+LUT operator, every task in one
transform family shares the *same* fitted operator, so `slow` converges to it and genuinely
accelerates fresh-grid tasks of that family — the clean two-timescale speedup. A *mixed* family
would average to an unhelpful centroid (flip_h+flip_v ≈ a non-transform), helping no member: the
structured operator is too rigid to host a useful cross-task sub-manifold. That richer "prior over
related-but-different tasks" needs a more expressive memory — exactly Phase B's job. M9 proves the
*mechanism* (slow is learned, on the slow timescale, and demonstrably helps) honestly, on the case
this operator can support. **Phase A (M1–M9) COMPLETE.**
