# Esper development journal

A running, timestamped narrative of progress, discoveries, and blockers — so the
project has a complete story later. Newest entries at the bottom. Times are local
(America/Bahia, −03).

Companion docs: the canonical roadmap (status + direction) lives in
`docs/ROADMAP.md`; the theory is distilled in `docs/NL-summary.md`.

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

---

## 2026-06-30

**08:47 — Full ARC-AGI 2 training-split eval (the larger honest sample).** Ran the held-out driver
(`--report`) over all **1000 training-split tasks** overnight (~2h, ~7s/task, no crashes). Result:

    Solved (held-out >= 0.99):  5 / 1000   (0.5%)
    mean held-out exact-match:  0.0265
    held-out >= 0.5:  26      0 < held-out < 0.5:  44      held-out == 0:  930

**The 5 solves are GENUINE, not flukes** — each has held-out 1.0, train-fit 1.0, gap 0.0 (fit on
the task's demos, reproduced the *unseen* test grid exactly): `0d3d703e`, `67a3c6ac`, `68b16354`,
`c8f0f002`, `d511f180`. Spot-checked `0d3d703e`: a pure colour-permutation task (5→1, 8→9, 6→2, …)
— the operator learned the LUT from the demonstrations and applied it to the held-out test. These
are real ARC-AGI 2 tasks that happen to fall inside the geometry+colour subset, **rediscovered by
the learned operator with no symbolic DSL given** — exactly the spine of the project. This clears
Phase A done-criterion (2): *raw ARC-AGI 2 solve rate > 0% end-to-end with a held-out test* (met on
the training split; the 120-task eval split is harder — 0/120, mean 0.033).

The shape of the rest is the same honest story as the eval split: ~93% of tasks score 0 (shape
change / objects / counting / symmetry — outside the operator), a thin band partially fits but
doesn't transfer (the ~15 overfit: train > 0.3, held-out < 0.1), and ~20 show genuine partial
transfer (held-out ≥ 0.3, |gap| < 0.15 — near-subset tasks). Combined across both splits the engine
solves **5 / 1120** real ARC-AGI 2 tasks end-to-end. Raw dump: `scratch/arc2_train_results.txt`
(gitignored, reproducible). The path to a higher number is Phase B (a more expressive in-context
memory), not tuning — the 0.5% is the honest ceiling of a 16-param affine+LUT operator on real ARC.

**10:52 — Parallel eval across all cores + a determinism fix (the number is now reproducible).**
The full-corpus run was sequential (~7s/task → ~2h for 1000). Each task is independent (own fit, own
buffers), so `eval_parallel.sh` shards the `.task` list round-robin across `nproc` worker processes
(one `mojo run src/arc_solve.mojo --report` per shard) and re-aggregates the per-task lines —
**~12× on this 12-core box** (1000 tasks: ~2h → **758s**; the 120 eval: ~14min → ~3.5min). Pure
harness; no in-process threading (the `ESWorkspace` scratch and global RNG aren't thread-safe).

**DISCOVERY — the benchmark number depended on task ORDERING.** Validating the parallel runner
against the sequential result, 16/120 eval tasks came out *different* (aggregate stayed 0/120 but
the mean drifted). Cause: `arc_solve.mojo` seeded the RNG **once in `main()`**, so the one shared
stream made each task's stochastic ES fit depend on its position in argv — and sharding changes
positions. Fix: seed **per task** (`SOLVE_SEED`, before each fit), so a task's result depends only
on the task itself — invariant to ordering and to how the corpus is split. Proved it: N=1 vs N=10
workers now give byte-identical per-task held-out values.

**Canonical reproducible numbers (per-task seeded, shard-invariant):**

    Training split (1000):  5 / 1000 solved (0.5%),  mean held-out 0.0268
    Public-eval split (120): 0 / 120 solved,          mean held-out 0.0426

The **same 5 tasks** solve as the overnight sequential run (`0d3d703e`, `67a3c6ac`, `68b16354`,
`c8f0f002`, `d511f180`) — held-out 1.0, gap 0.0 — confirming they are robust genuine in-subset
solves, not an artifact of one lucky seed. The means shifted slightly (different RNG stream); the
headline (5/1000, 0/120) is unchanged and now reproducible regardless of worker count. Re-run with
`./eval_parallel.sh data_bin/arc2_train` (or `…/arc2_eval`).

**11:47 — Phase B / B1 begins: probed Mojo 1.0 trait mechanics before refactoring.** B1 lifts the
ES core onto `Domain` + `Memory` abstractions, so first I probed what Mojo 1.0.0b2 actually supports
(four throwaway scratch probes, since deleted). Findings that shaped the design:
- **`fn` is fully removed** — `def` everywhere, including trait methods and generic functions. `alias`
  is deprecated → `comptime`. A parameter cannot be named `out` (reserved by `def __init__(out self)`).
- Traits support **associated types** (`comptime Example: Movable`), **static methods**, and **nested
  associated access** (`M.Dom.Example`). Generic functions `def f[M: Memory](...)` and generic
  **structs** `struct ESWorkspace[M: Memory]` work (struct params referenced as `Self.M`).
- **Trait declarations do NOT take parameters** — `trait Memory[D: Domain]` is rejected. So the planned
  dual-parameter `[D: Domain, M: Memory[D]]` is impossible. Resolved it *better*: `Memory` carries its
  domain as an **associated type** (`comptime Dom: Domain`), so the ES core is single-parameter
  `[M: Memory]` and reaches the domain via `M.Dom` (`M.Dom.Example`, `M.Dom.distance(...)`). Cleaner
  than dual params. Validated end-to-end with a probe (generic `fit[M]` + generic `ESWorkspace[M]`
  calling `M.fill_scale` in `__init__`). Proceeding to the refactor (operator wrapped as
  `OperatorMemory`, suite must stay green) then the emergent `MLPMemory`.

**14:37 — B1 DONE: the Memory/Domain seam + the first emergent memory.** Lifted the ES /
two-timescale core onto two traits so it never names ARC specifics:
- `Domain` (in `arc_io.mojo`): associated `Example` type + `distance`/`score`/`capacity` (GridDomain
  wraps ArcGrid + the existing metrics).
- `Memory` (new `src/memory.mojo`): `param_dim`/`seed`/`fill_scale`/`apply`, with an associated
  `comptime Dom: Domain` reached as `M.Dom` (traits can't take parameters, so the domain is an
  associated type, not `Memory[D]`). The ES core is single-parameter generic `[M: Memory]` — static
  dispatch, zero overhead.
The demo containers became generic too (`ExamplePair[E]`, `Task[E]` in `hope.mojo`, with
`ArcTaskPair`/`ArcTask` as grid `comptime` aliases): Mojo type-checks generic bodies *eagerly*, so a
concrete `ArcGrid` can't be passed where the abstract `M.Dom.Example` is expected — the containers
must carry the abstract type. `ESWorkspace` → `ESWorkspace[M]` (fills the preconditioner via
`M.fill_scale`, retiring the hardcoded `COLOR_SCALE` branch); `OP_DIM` → `M.param_dim()` throughout;
`forward_with_learning` stays a concrete grid+operator wrapper (a fully generic forward needs a
Domain-provided output constructor — deferred). **`OperatorMemory` wraps the existing operator with
zero behaviour change — the whole suite stays green through the generic path** (held-out 1.0
everywhere, meta-prior gap 1.0, synth 3/3, real ARC unaffected).

**`MLPMemory` — the first training-wheel removal (emergent recolor, no LUT).** A per-cell 1->H->1
tanh MLP over the centre cell (K=1, H=16, 49 params): `x = in/9`, a fixed steep-tanh basis (W1=20)
tiling the colour range, output layer (W2,b2) fit by the *same* annealed ES, squashed to [0,9].
Learns recolor `(c+1)%10` to **held-out 1.0** — the colour permutation emerges from data, nothing
symbolic given. `tests/test_mlp_memory.mojo`.

**DISCOVERY — three landscape fixes, each found by `scratch/probe_mlp.mojo` (per-colour readout):**
1. *Unbounded output diverges.* A raw linear MLP output is unbounded, so a large W2 explodes the MSE
   and the ES blows up (held-out 0.0). The operator never has this — its gather output is always a
   bounded input value. Fix: squash the output (bounded MSE → stable ES).
2. *Basis resolution.* Gentle ramps (W1≈4) span ~0.25 in x, wider than the 1/9≈0.11 colour spacing,
   so adjacent colours aren't separable. Steep ramps (W1=20, near a soft lookup table) fixed it; the
   tiling also extends just past [0,1] so the edge colours sit inside the basis.
3. *Squash range must be EXACTLY [0,9] (the subtle one).* I first widened it to [-2,11] reasoning the
   extremes shouldn't sit at tanh's tails — but that makes a *saturated* output land on -2/11, which
   round to invalid colours that never match. With [0,9] a saturated output lands on a valid extreme
   colour (0 or 9), and the exact-match ±0.5 tolerance covers it. This also makes the 9->0 recolor
   **wrap fit for free**: colour 9 saturates the squash low, which is exactly 0. With [0,9], all ten
   colours (wrap included) land at the standard 4000 iters.

Honest scope: the MLP is a *local* memory (K=1) — it cannot express global geometry (flip/transpose
are coordinate permutations); that's B3's emergent-global-addressing memory. There is no
memory-selector — `OperatorMemory` and `MLPMemory` are compile-time choices, each measured on the
subset it expresses. Also added a dev harness `./esper` (run/test/main/solve/fmt/suite) so commands
don't need cd + venv-activate + `mojo run -I src`. Full suite green (~72s).

**18:39 — B2 DONE: the second (non-grid) domain — the Domain seam carries cross-domain.** Plugged a
1-D integer-sequence domain into the B1 seam and proved the **ES / two-timescale core is unchanged**:
`src/esper_evolution.mojo` was not touched at all — the whole of B2 is additive. That non-change *is*
the proof the abstraction isn't secretly grid-coupled.

- New Example type `Sequence` (`src/hope.mojo`, next to `ArcGrid`): an owned 1-D Float32 array,
  same lifecycle (alloc+zero / unconditional free), `Movable`+implicitly-deletable so it slots into
  the existing generic `ExamplePair[E]`/`Task[E]` containers with no container changes.
- `SeqDomain` (`src/arc_io.mojo`, next to `GridDomain`): metrics delegate to the **same**
  `calculate_fitness`/`exact_match` — they already take a flat `(ptr, ptr, n)`, so the sequence
  domain reuses them with **zero new metric code** (the "honest metric transfer" B2 set out to show).
- Two memories (`src/memory.mojo`), both `Dom = SeqDomain`, fit by the unchanged ES: `SeqOperatorMemory`
  (structured, the 1-D analog of `OperatorMemory` — a 2-param centered position affine + a 10-entry
  value LUT, value-then-gather with 1-D linear interpolation; reverse=`a=-1,t=0`, shift=`a=1,t=k`,
  increment via the LUT, all exact at integer params) and `SeqMLPMemory` (emergent, the per-element
  MLP — reuses `MLP_DIM`/`seed`/`fill_scale` and a factored-out shared `_mlp_cell` forward verbatim,
  iterating the sequence instead of the grid).
- `tests/test_seq_domain.mojo` (self-contained, mirrors `test_mlp_memory`): fit on random sequences,
  score on **held-out** ones. **All three reach held-out 1.0** — `SeqOperatorMemory` reverse (a
  genuinely new 1-D *geometry* the local MLP can't express) and increment, and `SeqMLPMemory`
  increment (emergent, no LUT).

Two implementation notes. (1) The generic test helper `held_out[M]` must take
`List[ExamplePair[M.Dom.Example]]` and score via `M.Dom.score(pred, target, n)`, **not** touch
`.data` — Mojo type-checks generic bodies eagerly, so inside `[M]` the abstract `M.Dom.Example`
exposes no fields; reaching the data only through the Domain seam is exactly the point (and the first
caught-by-the-compiler reminder that the core never names the concrete type). (2) No new ES tuning was
needed: reverse's `a:1→−1` is the same through-zero sign flip as grid `flip_h`, and increment mirrors
recolor, so the proven `FIT_*` schedule fit both off the shelf. Both the structured and the emergent
paths transfer to a new domain unchanged. Full suite green (the grid suite is untouched — held-out 1.0
everywhere, synth 3/3, meta-prior gap 1.0).

**19:10 — B3 DONE: emergent global addressing — geometry re-earned with NO hand-coded affine.** The
per-cell MLP (B1) is local and provably can't express geometry; only `OperatorMemory`'s hand-coded
affine could, by computing one source address and gathering exactly there. B3 replaces that explicit
single-address gather with a learned **global read** — a position-attention gather — and rediscovers
flip_h/flip_v/transpose through it. Additive once more: only `src/memory.mojo` + a new test;
`esper_evolution.mojo`/`hope.mojo`/`arc_io.mojo` untouched (B1/B2/B3 all extend capability with zero
core changes — the genericity holds).

- `AttnGatherMemory` (`src/memory.mojo`, `Dom = GridDomain`, **7 params**): a 2×2 coordinate
  projection `M`, translation `t`, and a learned temperature β. Each output cell `i` (centered coord
  `v_i`) reads from **all** input cells `j` weighted by `softmax_j(-β·‖M·v_i + t − v_j‖²)` — a
  two-pass streaming softmax (max-subtract for stability, then weighted gather; no per-cell buffer, no
  hot-loop alloc). As β grows the softmax → one-hot at the cell nearest `M·v_i + t`, so an integer `M`
  reproduces a permutation exactly (flip_h `[[1,0],[0,-1]]`, flip_v `[[-1,0],[0,1]]`, transpose
  `[[0,1],[1,0]]`). Scalar-per-cell like `apply_operator`; the SIMD/FMA weight-space ES update is
  generic and untouched. `param_dim` fixed (grid-size-independent).
- `tests/test_attn_memory.mojo` (mirrors the geometry portion of `test_generalization`): fit on random
  demos, score held-out. **flip_h / flip_v / transpose all reach held-out 1.0 — first try, no
  landscape iteration.**

**Why it converged cleanly (the design call that mattered): β must be seeded MODERATE, not small.**
The instinct was "seed β small for a smooth landscape," but β→0 makes the attention uniform and the
gather equals the grid mean — *independent of M*, so ∂(gather)/∂M = 0 (a flat, gradient-free plateau,
the same failure mode as the nearest-neighbour rounding back at M3). β→∞ is also flat (a locked
one-hot). The gradient in `M` is largest at **moderate** β, where the soft ~1-cell peak actually
*moves* when `M` moves — the attention analog of the bilinear gather's smooth geometry gradient.
Seeded β ≈ 2 (stored as `raw²` for ≥0 without `exp` overflow, β its own preconditioner group), the
wide `FIT_SIGMA0=0.5` explored `M` while the anneal let β climb to sharpen onto the exact integer
permutation. transpose's 4-param `M` move (the M7 fragility) was a non-issue here.

Honest framing/scope: this is the next training-wheel removal — the explicit single-address gather (a
strong structural prior) is gone, replaced by a learned global similarity read, the substrate B4's
self-modifying memory builds on; the residual linear coord projection is what later milestones
dissolve. **Geometry-only by decision** — recolor stays `MLPMemory`'s job, so `OperatorMemory` does
**not** retire yet; folding the colour "local map" (the value MLP) onto this global read into one
combined whole-subset "general grid memory" is the deferred follow-up. Cost: the gather is O(N²) per
cell-grid, so the suite went ~72s → ~116s (still < 2 min at 4×4 grids); trimmable via iters/grid size
if it grows. Full suite green; all prior paths unchanged (operator/MLP/seq/meta held-out 1.0, synth
3/3).

**NEGATIVE RESULT — the combined "general grid memory" (attention geometry + value MLP colour) was
attempted and abandoned as honestly unlearnable by a single ES fit; OperatorMemory stays the
geometry+colour baseline.** Goal: fold the emergent colour MLP onto the attention global-read into ONE
memory (`GridMemory`, ~56 params) covering {flip, transpose, recolor} and combinations, to retire
`OperatorMemory`. It does not work as clean emergence, and the *way* it fails is the lesson:

- A cold JOINT ES fit over the 56-D coupled memory never converged — different seeds landed different
  transforms, none exact. ES estimator variance grows with dimension, and β-sharpening couples the
  geometry and colour stages, so the joint landscape is too hard for the derivative-free ES.
- Every fix that moved the number was **hand-engineering the solution path**, not emergence: a staged
  fit (fit geometry, then colour — via zeroing each group's preconditioner `scale`), a mid-fit β-boost
  to a hand-picked sharpness so the colour phase sees integer gathers, a per-phase L2-anchor bump to
  stop the colour-blind geometry phase from drifting off identity on pure-colour tasks, a colour-
  output preconditioner, a residual (`gathered + 9·tanh(MLP)`) colour parameterisation seeded to exact
  identity. With the whole stack, geometry hit 1.0 and 4/6 transforms reached 1.0 — but pure `recolor`
  / `flip_h+recolor` stayed ~0.1–0.4 and brittle.
- **The decisive realisation (the user's "stone soup" call):** even if one more tweak turned all six
  green, the green would come from the *hand-choreographed fitting recipe*, not from the memory
  emergently learning the transform — exactly the "put the reasoning in the human" the project's spine
  forbids. By contrast B1 (recolor) and B3-geometry each emerged from a single COLD JOINT fit, no
  staging — that is the bar for an emergence claim, and the combined memory cannot clear it.
- Why the operator's joint fit works but this doesn't: M1–M7 *co-designed* the operator (smooth
  bilinear, colour-then-gather, per-group preconditioner) for ES tractability. The operator IS the
  round hole. A fully-emergent attention+MLP is strictly more general but ES-intractable as one fit.

**Conclusion / routing.** Retiring `OperatorMemory` via one emergent memory is a real research step,
not a tuning session — it needs a better adaptation mechanism (precisely B4's self-modifying memory:
the memory generating its own update/sharpening dynamics) or a fundamentally cleaner architecture, not
more scaffolding. Reverted all combined-memory code (`GridMemory`, `fit_staged`, residual colour,
`_mlp_z`/base-offset refactor); shipped the clean geometry-only `AttnGatherMemory`. `OperatorMemory`
remains the honest geometry+colour baseline. **B3 = emergent global addressing for geometry, DONE.**

**23:03 — Compute scaling, phase 1: parallelized the ES inner loop across CPU cores (numerically
identical, ~3.5× on the heavy attention fit).** B3's O(N²) attention pushed the suite to ~116s and the
user wanted a higher training budget. Investigated GPU first: the RTX 2060 Mobile (compute 7.5) is
present, but the **pinned slim `mojo` 1.0.0b2 wheel ships only `std` + `layout` — the Mojo GPU API
(`gpu.host.DeviceContext`, kernels) lives in the MAX packages, not installed.** GPU would mean
migrating off the hard version pin to MAX, and a 2060 Mobile wouldn't beat 6–12 CPU cores on these
tiny kernels anyway. `std.algorithm.parallelize`, by contrast, IS in-toolchain and runs across cores.
Decision (user): **phased — CPU now, GPU/MAX later** (revisit when batching many tasks or scaling N
huge).

The ES does **2·N independent fitness evals per iteration** — embarrassingly parallel. Restructured
`evolve_fast_weights` into serial → parallel → serial: (1) draw all N epsilons up front in the exact
sequential RNG order; (2) `parallelize[sample](N)` — each sample builds its ±perturbation and
evaluates F+/F− in its **own disjoint workspace stripe** (`ESWorkspace` now holds `n_samples`× the
`eps`/`perturbed`/`op_output` buffers + a `coeff` array; `n_samples` defaults to `FIT_N`, so all 8
construction sites are unchanged); (3) reduce `grad = Σ coeff[s]·eps[s]` serially in sample order. The
SIMD/FMA bodies and `fitness[M]` are reused verbatim.

**Bit-identical by construction** — same epsilons (serial RNG), deterministic per-sample fitness,
same serial reduction order → same `grad` → same weights. Verified: the whole suite stays green with
the SAME numbers, including `main`'s exact float dump `4.0731115 9.093359 0.92952275 0.91821253`
unchanged. So this is pure throughput, not behaviour.

Results: **`test_attn_memory` 54.9s → 15.5s (3.5×)**; full suite ~116s → ~87s (`user` 7m52s ≫ `real`
87s confirms heavy parallel use). The feared cheap-fit regression (parallelize dispatch overhead on
4000 trivial-eval iterations) **did not materialise** — even the operator/MLP fits net-sped-up — so
**no work-size gate was needed** (kept the code simple; revisit only if a future cheap memory
regresses). Only blast-radius fix: `test_demo_fitness` poked `workspace.op_output` directly (renamed
to `op_output_all`; its base pointer is still a valid capacity-sized scratch). Pure Mojo, no new
deps, no hot-loop alloc (per-sample buffers pre-allocated once), SIMD/FMA preserved. **Phase 2 (GPU
via MAX) deferred** — documented in the plan; out of scope until the workload outgrows 12 cores.

**23:42 — B4 first step DONE: a self-modifying memory that writes its own state in-context (recolor),
read projections meta-learned cold.** The B3 lesson was that the ES can't fit coupled/high-dim FAST
weights. HOPE's self-modifying memory (NL §8) is the reframe-fix: instead of the ES searching the
fast weights, the memory runs its OWN update rule that WRITES its fast state from the demos in one
forward pass; the ES fits only the small SLOW rule params. Built it on recolor, in two checkpoints
(the user's validate-first choice).

- **Checkpoint 1 — mechanism (fixed projections, no ES).** `RecolorSelfWrite` (`memory.mojo`):
  `adapt(demos)` accumulates a per-colour value table from the demonstration (in,out) cells in ONE
  pass (one-hot keys, count-normalised); `apply` reads it. `tests/test_selfmod_memory.mojo`: held-out
  recolor **1.0** from a single in-context pass, ~3s, no ES. The self-write generalises — gate passed.
- **Checkpoint 2 — emergent (meta-learned read).** `RecolorSelfModMemory`: the write is unchanged but
  the READ is a learned softmax attention over **meta-learned colour embeddings + temperature**
  (81 slow params) — `pred(q) = Σ_c softmax(β·E[q]·E[c])·value[c]`. `meta_fit_selfmod`
  (`esper_evolution.mojo`) fits the slow vector by the SAME antithetic *parallel* ES — but its
  dimension is just the 81 slow params (the fast state is written by `adapt`, never ES-searched),
  which is exactly why it's tractable where fitting fast weights wasn't. Meta-objective: across a
  family of random recolor *permutations*, adapt from each task's train demos → score its held-out
  (continuous −MSE).

**Result (cold, no scaffolding — honesty guard NOT triggered):** a generic-seed read scores **0.125**
held-out on a fresh permutation; after one ~12s cold meta-fit (2000 iters), a **fresh, unseen recolor
permutation is solved to held-out 1.0 by a SINGLE `adapt` forward pass** (avg over 5 fresh perms) — no
per-task ES fit. The before/after gap (0.125 → 1.0) is asserted in the test, so the emergence is
non-vacuous (the generic seed must fail). Contrast: B1's `MLPMemory` needs a ~4000-iteration ES fit
*per* recolor task; this adapts in one pass after a one-time meta-fit — the HOPE fast-adaptation win,
and a direct payoff of the B3-lesson reframe (move the search off the fast weights).

Honest scope: kept the self-mod memory **concrete** (no `SelfModMemory` trait yet — YAGNI with one
instance; abstract when a second lands, as B1 did). The read is meta-learned but the *write* is still
a fixed accumulate, and recolor is a case where hard indexing trivially works — so this proves the
*mechanism is learnable cold*, not that it beats the operator on geometry. The fuller HOPE block
(self-generated η/forget-gates + internal objective) and the self-write applied to **geometry** (the
honest route to finally retiring `OperatorMemory`, B3's open thread) are the deferred next steps.
Additive: `hope.mojo`/`arc_io.mojo` and the existing `Memory`/ES path untouched; full suite green
(~104s, all prior numbers unchanged).

---

## 2026-07-01

**12:25 — B4 FULLER BLOCK DONE: self-generated key/η/α + a gated delta-rule self-write, meta-learned
cold; solves arbitrary local-context rules in one pass.** B4-step-1's *write* was a fixed accumulate;
the fuller HOPE block (NL §8, Eqs 83–97) makes the write a learned, self-generated update rule. Proven
on a sequence local-context map `out[i] = f(in[i], in[i-1])` (circular left neighbour) — a
neighbour-dependent rule the per-cell MLP / colour-LUT **cannot** express; the memory must key on the
(cell, neighbour) *context*.

- **Abstractions.** Added the `SelfModMemory` trait (`Dom`/`slow_dim`/`state_dim`/`seed_slow`/
  `fill_scale`/`adapt`/`apply`) — justified now at ≥2 instances; retro-conformed `RecolorSelfModMemory`
  and made `meta_fit_selfmod` generic `[M: SelfModMemory]` (scores via `M.Dom.distance`; the parallel
  ES body unchanged). RecolorSelfMod stayed bit-identical through the generic path.
- **Checkpoint A (fixed one-hot keys, no ES).** `DeltaSelfWrite`: the gated delta rule
  `S ← (1−α)S + η(v−S·k)k` with one-hot context keys builds the per-context table in one pass →
  held-out **1.0**. Mechanism validated (fail-fast gate cleared).
- **Checkpoint B (emergent, meta-learned).** `DeltaSelfModMemory`: the key is the **outer product** of
  learned per-symbol embeddings `E[c0]⊗E[c1]` (Dk=25 = A², so a linear `S·k` can represent an ARBITRARY
  `f(c0,c1)` — a *concat* key would only give an additive `g(c0)+h(c1)`), and η/α are self-generated
  (sigmoid of a learned projection of the key). `meta_fit_selfmod[DeltaSelfModMemory]` fits the slow
  projections (embeddings + gate weights, ~77 params) cold. **Result: generic seed 0.231 → after a cold
  meta-fit, a fresh unseen rule solves to held-out 0.984 (avg of 10 fresh rules) by a SINGLE adapt
  pass.** The before/after gap is asserted (non-vacuous).

**DISCOVERY — key normalisation is essential for the delta rule's stability.** First meta-fit went
BELOW the random seed (0.104 → 0.0): with an unnormalised outer-product key the ES wandered the
embeddings into regions where the recurrent write `η(v−S·k)k` diverges (η·|k|² unbounded → the state
blows up → NaN fitness → the ES chases garbage). Fix: **normalise the key to unit norm** inside `_key`,
which bounds the effective step size η·|k|² ≤ η < 2 regardless of embedding scale (the same reason
attention scales its dot-products — a general architectural choice, not task scaffolding; a no-op for
the one-hot sanity case). With it, cold meta-fit climbs 0.231 → 0.984, no scaffolding — honesty guard
NOT triggered.

Honest scope: unlike recolor's clean 1.0, this harder (arbitrary local-context, A=5) task lands right
around the bar (~0.95–0.98 depending on budget/embedding dim) — a genuine cold pass but a noisier,
near-bar result, so the test uses a comfortable meta-budget (10 tasks, 4500 iters) and averages 10
fresh rules for a stable ~0.984. Cost: the meta-fit is the heaviest test (~74s; suite ~104s→~169s).
Additive — `hope.mojo`/`arc_io.mojo` and the existing `Memory`/ES path untouched; full suite green.
**Next (a later phase, per plan): the path to full ARC-AGI 2** — 2-D context keys (grid neighbourhoods),
composing this content self-write with the B3 attention geometry (the honest route to retiring
`OperatorMemory`), shape-change, and a multi-block CMS chain.

**14:44 — 2-D CONTEXT KEYS DONE (first ARC-AGI-2 block): the self-modifying memory keys on a grid
NEIGHBOURHOOD and learns a local pattern→colour rule in-context.** Lifted the B4 fuller block from a
1-D sequence neighbour to a 2-D toroidal grid neighbourhood. Proving task (user's chosen class):
`out[r,c] = h1(center, up) + h2(center, left)` — the ADDITIVE center↔neighbour class, genuinely 2-D (a
per-cell MLP, a colour-LUT, and the 1-D sequence memory all provably fail it; it needs *both*
neighbours). Additive because the read `S·k` is linear — arbitrary neighbourhood rules are
combinatorial (A^9) and out of scope (a nonlinear read is a later block).

- **`GridContextSelfModMemory`** (`SelfModMemory`, `Dom = GridDomain`): per toroidal cell the key is
  `concat(E[center]⊗E[up], E[center]⊗E[left])` (Dk = 2·De² = 50), unit-normalised (the B4 stability
  lesson); η/α self-generated; the gated delta write runs a few EPOCHS over the demo cells (the
  additive read is a 2-way decomposition SGD solves — one "adapt" is multi-epoch, still a forward pass,
  not ES). ~127 slow params. Reuses `meta_fit_selfmod[GridContextSelfModMemory]` with **zero core
  change** (the meta-fit was already `Domain`-generic — a grid self-mod memory just plugs in).
- **Checkpoint A** (`GridContextSelfWrite`, fixed one-hot keys, no ES): the gated delta write solves the
  additive decomposition from the demo cells (eta 0.2, 20 epochs) → held-out **1.0**, mechanism +
  convergence validated.
- **Checkpoint B** (meta-learned): generic seed **0.114 → 0.985** (avg of 8 fresh unseen rules) after a
  cold meta-fit (4×4 grids, 8 tasks, 2500 iters), each fresh rule solved by a single adapt pass; the
  before/after gap is asserted.

Sized the meta-fit against the 10-minute budget before running (a short 500-iter probe = 27s → the
full 2500-iter fit ≈ 2m51s, comfortably under). Honest scope: like the 1-D fuller block this lands
near the bar (~0.98) — a genuine cold pass on a harder task. Cost: heaviest test yet (~3.2 min; suite
now ~5.4 min). Additive — `esper_evolution.mojo`/`hope.mojo`/`arc_io.mojo` and the whole existing
`Memory`/ES/self-mod core untouched; full suite green, all prior numbers unchanged. **Next: compose
this content self-write with the B3 attention geometry (retire `OperatorMemory`), richer neighbourhoods
/ nonlinear read, shape-change, CMS chain.**

**15:05 — Fast/full suite tiers.** The suite has crept to ~5.4 min as each meta-fit proof landed, too
slow for a routine local gate. Timed every test individually to see where the cost lives — it is
brutally lopsided: `test_grid_context_selfmod` (178s) + `test_delta_selfmod` (64s) = **79% of the whole
suite**; the other 14 tests + `main` + `arc_solve` total ~65s. So the split is really about deferring
those two. Added a `run_tests.sh [full|fast]` tier (default `full`, so CI and bare `./esper suite` are
unchanged) + a `./esper fast` alias (~80s). A test opts *out* of `fast` by self-tagging a
`# suite-tier: full` comment line (grepped by the runner); untagged → runs in both. Only the two heavy
proofs are tagged. **Design discipline:** `fast` is a *strict subset at full budget* — no reduced
iterations, no relaxed `≥0.95` gates (a weakened smoke test would be exactly the stone-soup failure
mode). It still covers every code path once at full budget, including the self-mod `meta_fit_selfmod`
core via `test_selfmod_memory` (11s); the two deferred tests are the same mechanism at larger scale, so
`fast` loses no coverage type, only the large-scale milestone confirmations. Chose self-tagging over a
central list so growth stays additive (new heavy proofs declare themselves; nothing to keep in sync).
CI stays `full` (correctness gate; runner time is cheap). Verified: `./esper fast` = 69s, skips exactly
the two, green; default banner reports `tier: full`; bad arg exits 2; `mojo format` leaves the tagged
files clean (marker-only additions).

**16:05 — RICHER NEIGHBOURHOODS + NONLINEAR READ DONE (ARC-AGI-2 block 2): the disjunctive/count
class.** GridContext reads `pred = S·k` — LINEAR — so it does only additive positional rules. This block
crosses the nonlinear barrier: `GridNbhdSelfModMemory` keys on a Moore-8 neighbour-count histogram and
reads through a sigmoid THRESHOLD, expressing `out = C1 if (#neighbours == P) ≥ t else C2` — an OR /
count that no linear read of the count can be (a sharp step). Zero core change: it's a `SelfModMemory`
over `GridDomain`, so `meta_fit_selfmod` fits it untouched.

Getting there took three design corrections, each empirically forced (validate-first paid off — all on
the cheap fixed-key Ckpt A, before any meta-fit):
1. **Centre-free key.** My first key was the GridContext-style outer product `E[centre]⊗Σ_n E[n]`.
   Held-out stuck at ~0.52 (majority baseline): tying the histogram to the centre fragments the
   per-colour weight across 5 centre-rows and injects irrelevant centre-dependence. The disjunctive/count
   class is centre-INDEPENDENT, so the key must be the bare neighbour histogram `k = (1/8)Σ_n E[n]` (+ a
   bias slot). Then every cell trains ONE small weight vector — clean logistic-regression-style learning.
2. **Mean, not unit-norm.** The B4 lesson is "unit-normalise the delta key or it diverges" — but here the
   count magnitude IS the signal the threshold reads; normalising divides it out. The MEAN (÷8) keeps
   counts proportional while still bounding ‖k‖ for stability. (Different memory, different right answer.)
3. **Feature scale vs the bias.** With one-hot `E` the count feature (`count_P/8` ≤ 0.5) is dwarfed by
   the bias slot (1.0), so the write just fit the mean → constant collapse. Scaling `E` (×3) so the
   feature is O(1) fixed it; a **constant bias slot** in the key lets the write self-calibrate the
   threshold offset regardless of the learned magnitude (and makes `t` itself learnable).

Read: `pred = LO + (HI−LO)·σ(g·(S·k) + c)`; write: the perceptron-style gated delta
`S ← (1−α)S + η·e·(HI−LO)σ(1−σ)g·k`, η/α self-generated, 12 epochs over the demo cells. The boundary
cells (count exactly = t) need a fairly SHARP σ or they mis-round; 12 epochs + moderate `g` on a
balanced `t=2` rule (Binom(8,1/5): P(≥2)≈0.50) gets there. **Honesty control (cheap, decisive):** the
SAME write with a LINEAR read (identity, even clamped — the strongest linear baseline) reaches only 0.61
vs the nonlinear 0.995 — a linear read cannot sharply threshold a count. Ckpt A: nonlinear **0.995**,
linear **0.611**. Ckpt B (cold `meta_fit_selfmod`, generic seed → fresh unseen rules): before **0.0**
(generic seed predicts constant → non-vacuous), after **1.0** — a fresh disjunctive/count rule (unseen
predicate colour P) solved to held-out 1.0 in ONE adapt pass.

Scope (honest): the balanced `t=2`, fixed output-pair `{0,4}` family; the predicate colour P is inferred
in-context (analogous to RecolorSelfModMemory inferring a permutation). Varying `t` (imbalanced, harder)
and multi-bin count → later. Cost: sized against the 10-min rule — the full 326s fit (N=96/1500) hit
after=1.0 with huge headroom, so I trimmed to N=64/1000 iters (**157s**, still after=1.0). The test is
`# suite-tier: full`, so `./esper fast` is unchanged (71s); the full suite rises to ~8 min (under the
line). Additive: only `memory.mojo` grew + the new test; the ES/self-mod core untouched.

**17:20 — BROADENED THE NONLINEAR CLASS (ARC-AGI-2 block 3): infer the WHOLE 2-level count rule
in-context.** Block 2 fixed the output colours (`{0,4}`) and threshold (`t=2`) in meta params. This
generalises `GridNbhdSelfModMemory` (in place — the fixed case is a strict subset) so the memory infers
`out = C1 if (#Moore-8 == P) ≥ t else C2` with **P, t, AND both colours C1/C2 all varying per task**,
keeping the single-sigmoid read.

The first attempt was a **written 2-unit output head** `pred = v0(1−σ)+v1·σ` with `v0/v1` delta-written
and `min/max` symmetry-break init. It worked for ~half the rules but was marginal on **inverted** rules
(fire → the *smaller* colour) and erratic at higher `g`: the value-coupled head and the salience *fight*
— `min/max` pins `v1=max`, but an inverted rule wants the high-count regime to output the *min* colour,
forcing a large wrong correction. Diagnosed on the cheap fixed-key probe (0.24–1.0 scatter).

Fix = **decouple colour from threshold**. Read the two colours straight off the demos (`v0/v1 = min/max`
output, written, no delta), and train the salience `S` as a **binary classifier** of *which* colour a
cell outputs (`y = output > mid`) — not by regressing the raw target. So `S` inherits block 2's proven
logistic count-write, and the classifier's **learned sign** handles inversion for free; the bias slot
self-calibrates any `t`. One more correction: the classifier's cross-entropy update had lost block 2's
`(HI−LO)=4` gain, so it under-trained (~16× too small a step); restored via a `GRIDNBHD_LR` gain +
dropping the vanishing `σ(1−σ)` factor (cross-entropy, doesn't stall under a sharp read). After that:
fixed-config Ckpt A = **1.0 on normal, inverted, and t=3** rules; linear-read control **0.48**.

Ckpt B honesty subtlety (worth recording): with a `sin` embedding seed the generic seed *already* scored
**0.92** — because block 3 reads the colours off the demos, a decent classifier + a separable-enough
generic `E` generalises on its own, making the "generic seed fails" guard vacuous. The honest fix is a
**zero-embedding seed** (the canonical "no learned representation" prior): then `k=0`, the read is
constant, and `before` = best-constant baseline **0.74**, while the meta-fit must *discover* separable
colour embeddings from scratch → **after 1.0**. So the ES genuinely grows `E` from zero (antithetic noise
breaks the 5-way colour symmetry). Cost ~164s (in place → replaces block 2's test; full suite stays
~8.4 min, under the 10-min line). Scope now: any 2-level count rule; multi-bin count (3+ outputs) is the
next broadening. Zero core change — `meta_fit_selfmod` reads `state_dim`/`slow_dim`.

**18:30 — MULTI-BIN COUNT MAP DONE (ARC-AGI-2 block 4): an ARBITRARY function of a colour's
neighbour-count, incl. non-monotone.** `GridCountMapSelfModMemory` reads `out = M(count_P)` for an
arbitrary map `M: count → colour` (non-contiguous, ≥3 output colours); both predicate `P` and map `M`
inferred per task. (Execution budget raised to 15 min mid-block, so the 4th heavy full-tier test is fine.)

**The instructive part — a documented NEGATIVE RESULT, then the fix.** I first built the "obvious"
generalisation of block 3: a soft count-BIN value table `pred = Σ_j φ_j(z)·V[j]` with a COUPLED gradient
self-write (learn the salience `S` and the value table `V` together by the reconstruction gradient). It
**failed**: on non-monotone maps held-out was erratic 0.03–0.85 across every LR, and even fine monotone
maps only reached ~0.66–0.85. Root cause (verified with a pinned-salience read-check that itself hits
**1.0**): the READ is fine, but at `S=0` every cell shares the count-score `z`, so the only signal
bootstrapping the salience is the **linear covariance** of `count_c` with the output — which **vanishes
for non-monotone maps**. Identifying *which* colour to count is a discrete SELECTION, not a smooth
gradient target. (Same family as block 3's coupled-head fight; a hard-won recurring lesson.)

**The fix (held the emergent bar): a META-LEARNED SCORING salience.** Instead of gradient-writing `S`,
the memory computes per-colour demo STATISTICS of how `count_c` relates to the output — the variance
REDUCTION `Var(out) − E_j[Var(out|count_c=j)]` (captures ANY functional dependence, monotone or not), the
linear correlation, and the mean count — and a meta-learned linear score `a = softmax(τ·(w·features))`
picks the predicate colour. `z = Σ_c a_c·count_c` (a→onehot(P) ⇒ `z = count_P`, integer-scaled, so the
bin grid needs no calibration); the value table is written by a delta keyed by the soft bin. The meta-fit
**learns to weight variance-reduction over correlation** — i.e. it discovers that the linear feature is
exactly the one that fails on non-monotone maps. `w=0` seed ⇒ uniform selection ⇒ constant read ⇒ fails
(non-vacuous). Results: Ckpt A (fixed scoring) solves arbitrary non-monotone maps at **1.0**, while the
block-3 2-level memory on the same ≥3-colour map gets **0.41** (it can emit only two colours — multi-bin
is load-bearing). Ckpt B (cold meta-fit): before **0.319** → after **1.0** (fresh unseen maps, one adapt
pass). `slow_dim=12` so the fit is fast (~56s test). Additive new struct (block 3 stays green); zero core
change. Scope: `A=5`, Moore-8, observable counts ~0..4; rare high counts are out-of-distribution.

## 2026-07-02

**12:30 — BLOCK 5 HIT A REAL WALL (single composed geometry+colour memory): the soft/sharp coupling.
Recorded as a NEGATIVE RESULT; development paused for a literature pass.** The block-5 plan was the
anti-stone-soup design: ES searches ONLY the 7 AttnGather geometry params, the colour table is solved
CLOSED-FORM from the demos per candidate (a soft-binned value table over the gathered colour) — never a
joint 17-D search. It still failed, and the failure is instructive enough to record in full.

~7 mechanism iterations, every one trading transpose against recolor (flip_h/flip_v were 1.0 throughout):
soft colour read at test → recolor 0.89 (soft blending mixes adjacent colours that map far apart in a
permutation); hard lookup fixed recolor to 1.0 but transpose collapsed (0.11–0.25) — root cause **colour
absorption**: a free colour table explains away geometry error at WRONG geometries, flattening the
geometry ES's fitness contrast (control: colour forced ≡ identity → transpose 1.0, M≈[[0,1],[1,0]] — the
ES *can* find it, the colour table was hiding it). Every containment then broke the other side: a
constant identity-ridge → transpose 1.0 / recolor 0.84 (breaks map entries far from identity, e.g. the
9→0 wrap); a variance-gated shrink → recolor 0.16 (the soft gather blurs recolor's bins → spurious
variance → over-shrink); hard-bin solve + gate → 0.25; decoupling sharpness by ROLE (sharp gather for
solve/test, learned soft beta for the fitness gradient) → recolor 0.13 (soft-fitness/hard-test mismatch —
the fitness optimum drifts off identity). The diagnosis is structural, not a tuning matter: **the
geometry ES needs a SOFT gather (a gradient in M) while colour needs a SHARP gather (clean bins), and
one temperature + one colour table cannot serve both.** Per the stone-soup discipline I stopped rather
than stage the fit by hand. The WIP composed code (`ComposedGeomColorMemory`/`fit_composed`) sits
uncommitted, not wired into the suite, pending the reroute.

**The pause (user call, and the right one): stop building, survey the field** — both for this blocker
and for the long-term vision (an architecture for neuro-symbolic + continual-learning problems well
beyond ARC; the blocker's fix should serve that vision, not just this block). Findings recorded in the
new **`docs/RESEARCH-NOTES.md`** (a durable doc: literature mapped onto Esper's concepts at decision
points), summarized: (1) **energy-based compositional inference** (Compositional Energy Minimization
2025; IRED, ICML 2024) dissolves the wall — recast memories as energies `E(out|in,demos)`, train
geometry/colour energies SEPARATELY, compose at inference by SUMMATION, solve by annealed minimization
over the output; the soft→hard schedule lives on the SOLVER, so no shared temperature exists to fight
over, and `+`-composition is selector-free. (2) **Identifiability of modular structure** (Schug et al.,
ICLR 2024): hypernetwork task-weights = per-task code × shared templates (exactly our Reptile fast/slow
split) provably recover primitive modules from O(M) combinations — IFF compositional+connected task
support and NO over-parameterization (excess capacity ⇒ memorize-per-task instead of factoring; a live
caution for our memory sizes). This is the theory for the user's "pipeline of emergent memories whose
composition is itself learned". (3) Learned decomposition scheduling (LCC-CMA-ES 2025) — our decouple
lesson made adaptive; parked. (4) Latent program spaces (LPN 2024) — search a smooth latent that DECODES
to a valid operator; the reserve pivot if energies stall. Context check (ARC Prize 2025 report): the
frontier is test-time refinement loops in WEIGHT space, not symbolic space — Esper's core bet is aligned.

Roadmap updated accordingly: Next #1 rerouted from "one composed memory" to **energy composition**; a
"Beyond ARC" horizon item added for the emergent pipeline under the identifiability conditions; two new
working principles ("sharpness belongs to the solver, not the module"; "at a real wall, pause and
survey"). Block 5's retirement of `OperatorMemory` is deferred to the energy-composition block.

**12:56 — BLOCK 5 DONE (the energy-composition reroute works): `OperatorMemory` retired emergently by
`GeomColorComposedMemory` — the whole subset 1.0 cold, PLUS a composed flip∘recolor no single memory
expresses.** The design straight from the research pass: two modules, each fit on a signal INVARIANT to
the other's factor, composed additively, each keeping its own sharpness.

The invariance is the whole trick. **Colour:** for position-permutation geometry, per-demo colour COUNTS
are position-free (`cnt_out[V(c)] = cnt_in[c]` whatever the permutation), so the colour table V is
**written closed-form from count signatures** — match each input colour's across-demo count vector
against every output colour's, sharp softmax (τ=32), done. Probe: V exact on recolor **including the
9→0 wrap**, and exactly identity on geometry tasks — with zero geometry knowledge. A write rule in the
self-mod family (one forward pass, never ES-searched). **Geometry:** the proven AttnGather ES — but run
on demos whose inputs are PRE-MAPPED through V (**colour-then-gather**, the same decoupling hope.mojo
chose in Phase A): V is cellwise, the gather positional, so they commute, and with V applied first the
7-param search runs on the exact B3 fitness landscape (integer values, no colour-table cliff).

Two probe-driven design corrections worth recording:
1. **Fitting geometry THROUGH a soft colour read fails** (probe v1: composed task 0.10 vs 1.0 with the
   pre-map): interp(V, gather) with V's 9→0 cliff turns colour blends near 9 into wild values and the
   ES collapses into a soft-blur optimum. The pre-map is load-bearing, not a convenience. (This also
   closes the block-5 negative-result loop with a constructive fix: the soft/sharp conflict is
   dissolved by moving the colour read out of the geometry fitness entirely.)
2. **Per-task RNG seeding in the proof test** (the arc_solve protocol): the first full-test run had
   Ckpt C fail at 0.21 purely from its RNG stream position after five prior fits — a fresh-seed probe
   showed **12/12 arbitrary seeds solve the composed task at 1.0** (both eval orders; seed 9 even found
   the negative-beta flip variant, β=raw² making the sign free). Re-seeding per task makes each fit
   order-invariant and attributable; not seed-shopping, and the honest failure-rate evidence is journaled.

Implementation is almost embarrassingly additive: `GeomColorComposedMemory` (17 = 7 attn | 10 V;
`fill_scale` zeroes the V group — the perturbation AND update are scale-multiplied, so even a generic ES
fit can never move V) + `fit_geomcolor` = write_color → pre-map the demo list once (per task, not per
iteration — the hot loop stays alloc-free) → the standard `fit_operator[AttnGatherMemory]` on the
state's first 7 slots. Zero core change; the composed fit IS the proven attention fit on transformed
data. `apply` = sharp fitted gather + hard V lookup (equal to colour-then-gather at the fitted β≈9,
probe-verified both orders).

`test_composed_generalization` (full tier): Ckpt A the write recovers the full recolor permutation
exactly; Ckpt B the retirement bar — {flip_h, flip_v, transpose, recolor} each **held-out 1.0, cold**
(test_generalization's exact bar); Ckpt C **flip_h∘recolor 1.0 cold** — the composition payoff, a task
NO existing single memory expresses (and one the structured operator was never even tested on); control:
geometry-only ablation on recolor fails at 0.23 (the colour module is load-bearing). `OperatorMemory` is
now DORMANT (kept solely as the arc_solve/M8 baseline; removal is a later cleanup). This is the
first working instance of the composition pattern the research pass predicted — the fixed 2-stage
pipeline is v1; making the pipeline itself emergently learned is the horizon item (Schug conditions).

**13:33 — RESTRUCTURE: memory.mojo split per-family; src/ is now pure Mojo.** Two moves, zero semantic
change (verified: full suite green with identical numbers). (1) `memory.mojo` (1811 lines, 11 structs)
is now the trait seam only (`Memory` + `SelfModMemory`, ~90 lines); the families moved verbatim into
flat `-I src` modules — `memory_es.mojo` (the ES-fit forward family: dormant OperatorMemory, MLP, the
seq pair, AttnGather), `memory_composed.mojo` (GeomColorComposedMemory), `memory_selfmod.mojo` (B4
core: Recolor + Delta self-writes), `memory_selfmod_grid.mojo` (the ARC-AGI-2 grid blocks). Direct
imports everywhere (no re-export façade — explicit, and avoids betting on b2 transitive-import
semantics); flat modules rather than a package dir (unverified b2 package semantics; the `memory_*`
prefix gives the grouping anyway). 15 import sites retargeted mechanically. A neat find for future
refactors: `mojo build --emit object -I src <file>` compile-checks a file in seconds without linking
(the nix env lacks `-lm` for full links) — all 19 entry points checked before any test ran. (2) The
offline Python toolchain moved `src/` → `tools/` (arc_compiler.py, synth_tasks.py): the "Python is
offline-only" hard constraint is now directory structure, not convention — `src/` is pure Mojo by
construction. Deliberately NOT restructured (evaluated): `esper_evolution.mojo` (685 lines, cohesive
— revisit when the CMS chain grows the meta layer), hope.mojo's dormant `prim_*` (tied to the
OperatorMemory-removal cleanup), tests/ layout, root scripts. CI needed zero changes (globs).

**17:31 — ROADMAP.md gets a Mission/Vision/Values framing; a second, still-unshaped Vision surfaces.**
Pure discussion, no code — but it changes how future milestones get sized, so it's recorded here.
Used the standard business frame to separate three altitudes that had been getting flattened into one
"direction" narrative: **Mission** = the unreachable holy grail, redefined only on a pivot (never
"achieved," never itself measured) — an artificial reasoner that gets more intelligent through its own
experience, emergent rather than hand-installed, eventually learning *how to learn* and not just what.
**Vision** = the current, objectively measurable horizon, which recedes toward the mission as it's
closed on. **Values** = the self-imposed constraints on *how* the mission is pursued — these bind
across both visions below.

Talking it through surfaced that Esper actually has **two visions from two different starting points
toward the same mission**, not one:
- **Vision A (current, active)** is the existing ARC-AGI 2 path — held-out solve rate via emergent,
  composable, self-modifying memories fit in-context by ES (~5/1000 today, M8 operator ceiling). Named
  explicitly for the first time: this vision still hands the engine *goals* (a task's demo pairs are
  compressed supervision), even though it never hands it a DSL.
- **Vision B (WIP — not previously written down anywhere)** is inspired by Random Network Distillation,
  open-endedness, and unsupervised RL: an agent that masters its environment with **zero hand-coded
  goals** at all — self-generated novelty (prediction-error against a fixed random network) replaces
  the demonstration pairs. This is a *stricter* reading of the mission than Vision A — it drops the last
  hand-out (the goal itself), not just the DSL. Concretely it will need a new `Domain` (Example =
  trajectory, not grid pair) and a fitness signal the agent generates itself rather than one supplied
  externally — ES-for-RL (Salimans et al. 2017) is the natural fit for the existing derivative-free core,
  so this isn't a new learning mechanism, just a new `Domain`/`Memory` instance and a different fitness
  source. No design work done yet; explicitly a placeholder until shaped into milestones (likely first
  rung: the persistent-slow-weights-across-a-task-stream idea already tabled for later).
- **Convergence hypothesis**, stated but not yet tested: primitives discovered by open-ended exploration
  (B) become the reusable vocabulary that few-shot composition (A) draws on — unsupervised
  "pretraining" feeding few-shot composition, via ES/CMS instead of gradient descent on a corpus.

Also added a 6th value, **frugal by design**: every milestone must be provable on commodity hardware
with synthetic/tiny corpora; needing a cluster or a petabyte corpus to even attempt a claim is a signal
the substrate isn't emergent/efficient enough yet, not a green light to add compute. Sits alongside the
existing emergence / honest-measurement / bare-metal / domain-agnostic-core / narrated-evidence values,
all of which now explicitly bind Vision B too, not just Vision A. The existing "Beyond ARC-AGI 2"
section is left as-is but now cross-referenced as Vision B's seed, so it's clear nothing in it is
scheduled yet.

**19:15 — ROADMAP: "Next" section expanded (user-driven, after the MVV framing).** The path to full
ARC-AGI 2 now carries six evidence-ordered rungs instead of four: the funnel facts measured today
(68% same-shape ⇒ expressiveness binds before shape; median 3 demos ⇒ few-demo robustness is a real
block, promoted to #3; grid scale ⇒ compute, already addressed by the windowed gather) are stated in
the section as the basis for the ordering, the in-flight v2 re-measure moved to #1 with its
deliverable spelled out (the breakdown, not just the number), and the tabled **persistent slow
weights + task-stream** idea is now rung #6 with concrete measurables — (a) solve rate at a narrow
budget improves with stream position, (b) no catastrophic forgetting, (c) frozen-prior + shuffled-
stream controls — plus its known hazard (M9: flat priors wash out across heterogeneous families →
Schug hypernetwork route / CMS hierarchy) and its dual role as Vision B's likely first rung.

## 2026-07-03

**07:00 — REAL ARC-AGI-2 RE-MEASURE v2 (the emergent stack on the real corpus): train 10/1000
(M8 operator: 5), eval 0/120, and a decisive diagnostic verdict.** The block (started 2026-07-02):
windowed attention gather (a (2W+1)²=13² window centred on q — bit-identical on synth grids by
construction since every synth grid is ≤6 wide, ~3-5× cheaper at real 30×30; the composed full-tier
test actually got FASTER, 61s→33s), `arc_solve` switched from the dormant operator to the emergent
`GeomColorComposedMemory` (`fit_geomcolor`), a documented `--fit N ITERS` corpus budget (full-budget
fits at 30×30 are compute-prohibitive; CI's synth path keeps the full proven budget), an exact
compute skip when every test pair is shape-changing (held-out 0 by construction, ~⅓ of corpus
compute saved), and `tools/corpus_stats.py` (funnel + train-vs-held-out breakdown). Corpus runs at
budget 64/1500, 6 workers pinned to half the cores (user needed the machine): eval split 3390s,
train split 14878s.

**The numbers (budget 64/1500, seed-per-task, reproducible):**
- **train: 10/1000 solved** (mean held-out 0.402) vs the M8 operator's 5/1000 (0.027) at FULL
  budget — the emergent composed memory **doubles the operator ceiling at 1/5.3 the ES budget**.
  4 of M8's 5 retained; **6 NEW solves** (0b17323b, 18286ef8, 9f5f939b, 9f8de559, b1948b0a,
  ed36ccf7) — every one scored **0.0 under M8**, so this is new capability (the attention geometry +
  composed mechanics), not budget luck.
- **eval: 0/120** (mean held-out 0.319 vs 0.027) — the eval split stays unsolved, as expected;
  its top task (135a2760) sits at held-out **0.989**, a hair under the 0.99 bar.
- **One loss, and it's instructive: d511f180** (M8 1.0 → v2 0.75/0.80 train-fit): a colour-map task
  where the ES-fit LUT at full budget succeeded but the count-signature colour write partially
  mis-assigns — at the corpus's **median 3 demos** count signatures can tie. The few-demo-robustness
  rung (roadmap #3) now has its first real-corpus exhibit.
- Honest caveat: the big mean-held-out jump (0.03→0.40) partly reflects partial credit on
  sparse-edit tasks (identity geometry + colour table get the unchanged background right); the
  solve count is the headline, the breakdown is the value.

**The breakdown (the block's real deliverable — it re-prioritizes the roadmap with evidence):**
train same-shape subset (680): expressiveness gap (train<0.3) only **12.5%**, fits-but-no-generalize
**27**, near misses (0.5≤held-out<0.99) **437**. Eval same-shape (81): gap 18.5%, near misses 44/81.
Train ≈ held-out on nearly all top tasks — **what the memory learns DOES generalize; it simply
cannot express the full rule.** The binding constraint is expressiveness of local/content rules,
exactly roadmap #2 (compose content + geometry) — confirmed by measurement, not guessed.
