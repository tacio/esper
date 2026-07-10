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

**07:35 — CONTENT×GEOMETRY COMPOSED (roadmap #1, the v2-confirmed binding constraint):
`GeomCountComposedMemory` solves `out = geom(M(count_P(in)))` — a class NO single memory
expresses — held-out 1.0 cold on all four task types, first full run.** The block-5 recipe lifted
one level, and it lifted CLEANLY.

The structural insight that made it a one-probe block: the count rule is local and
translation-covariant on the toroidal Moore-8 lattice, and flips/transpose are SYMMETRIES of that
lattice (each cell's neighbour-set maps exactly) — so **the content rule and the permutation
geometry COMMUTE**, exactly like block 5's cellwise colour map but one level up. That gives both
halves of the invariant-decoupling for free:
1. **Content write, geometry-invariant:** block 4's correspondence salience (per-cell (count,out)
   pairs) is scrambled by an unknown geometry — but HISTOGRAMS are position-free. For candidate
   colour p, the per-demo count-bin histograms n_j^d must reproduce the output-colour histograms
   m_c^d (m_c = n_{M⁻¹(c)} for every demo, whatever the permutation), so each bin is matched to the
   colour whose across-demo signature it explains, and P is selected by the residual of the best
   assignment. Closed-form, one pass. Probe: exact 12/12; test Ckpt A: exact under
   identity/flip/transpose.
2. **Geometry, the proven pinned fit:** premap demos through the inferred content rule →
   fit_operator[AttnGatherMemory] on the exact B3 landscape (fit_geomcount mirrors fit_geomcolor
   verbatim).

Results (test_composed_content, full tier, 34s; budget N=64/2000 probe-validated — full FIT_* at
6×6 would bloat the suite; the cold bar is about no per-task staging, and the budget is uniform):
Ckpt B = identity/flip_h/flip_v/transpose ∘ countmap ALL **held-out 1.0, per-task cold**. Honesty
controls: (i) the block-4 correspondence statistic collapses under flip (variance-reduction 1.0
identity → **0.12** flipped) while the histogram route stays exact — the invariance is
load-bearing, not decoration; (ii) content-ablated (perfect geometry, seed content) → **0.18**.
Scope honest: injective M (a permutation of bins), A=5/B=5, toroidal Moore-8; non-injective M and
shift geometry noted as extensions. Additive: new struct in memory_composed.mojo + fit_geomcount +
the test; zero core change. One recurring Mojo gotcha rediscovered twice: `var g =
demos[d].input_grid` implicitly copies an ArcGrid (not ImplicitlyCopyable) — borrow through the
subscript expression instead.

The deeper pattern is now twice-proven and worth naming for the roadmap: **find the factor pair's
commuting representation, fit each factor on the signal the other cannot touch, compose forward.**
Colour maps commute with permutations via cellwise-ness; count rules commute via
translation-covariance on the lattice the permutations preserve. The candidates for "what commutes"
are becoming the real design questions for shape change and the CMS chain.

**08:31 — FEW-DEMO ROBUSTNESS DONE (roadmap #1): measured the degradation curves, hardened by
evidence, recovered d511f180 to 1.0.** The corpus median is 3 demos; every synth proof used 8.
Measurement first (200 fits, 5 families × n∈{2,3,5,8} × 10 per-task-seeded tasks, diagnostics read
off the written state so the identical probe re-measures after hardening).

**The measurement split the problem in two, and the split was NOT the one predicted.**
(1) `write_color` failed exactly as expected: at n=3, 3–5/10 tasks had a wrong or COLLIDING V (two
colours assigned the same target — the independent per-colour softmax has no injectivity), and
unseen colours got nonsense. (2) `write_content` was **never wrong — 0 errors in all 80 tasks even
at n=2** — no hardening allowed (it must earn its way in; it didn't). Its flip-family degradation
(6/10 at n=3) was the GEOMETRY ES: refitting the same tasks at full budget solved 9/10 — budget
starvation, not ambiguity (fewer demos ⇒ weaker fitness contrast on the demo average, while each
iteration is CHEAPER ∝ n_demos).

**Two task-independent mechanisms, each matched to its measured failure:**
- `write_color` → **global-min greedy INJECTIVE assignment** (the map family is injective — use the
  class structure, not a threshold), **identity defaults for unseen colours**, and **identity
  preference on exact ties** (count vectors are integers; ties are exact — the same maximum-prior
  convention as the ES identity anchor). The old soft softmax (and GEOMCOLOR_TAU) is gone.
- both composed fit drivers → **constant-compute budgeting**: iters × FIT_DEMO_REF(=8) / n_demos —
  every task gets the same total demo-evaluations regardless of demo count. At n=8 the factor is 1,
  so every existing proof is bit-unchanged.

**Before → after at the corpus median n=3** (solved/10, mean held-out):
recolor 7→**9** (0.91→0.98, collisions 3→0); flip_h 4→**8** (0.69→0.83, V-wrong 4→0);
flip_h∘recolor 4→**7** (0.67→0.88, V-wrong 5→0); flip_h∘countmap 6→**9** (0.71→0.93).
n=8 all 1.0 (no regression). n=2 stays honestly rough (recolor V-wrong 7/10: at 32 cells, exact
signature ties are GENUINELY unknowable — identity-on-tie is the wrong guess for a +1 permutation
and the right one for real-ARC unchanged colours; the underdetermination ceiling is documented, not
fought). **d511f180 — the block's real-corpus exhibit — recovered: held-out 1.0, train 1.0** at the
v2 corpus budget (was 0.75/0.80; M8's full-budget ES-fit LUT had solved it, the un-hardened write
had lost it). `test_few_demo` (full tier, ~63s) locks the bars: n=3 aggregate ≥0.85 (measured 0.87;
pre-hardening 0.74), n=8 per-family ≥0.95.

---

## 2026-07-03 13:12 — Shape change: the output-size seam + first shape-change family (Vision A, Next #1)

The roadmap's Next #1, and the binding constraint the 07-03 re-measure named: 32% of both corpus
splits change shape (out dims ≠ in dims), and the whole engine was hard-wired same-shape — every
memory's `apply` wrote exactly `inp.rows × inp.cols` cells, `fitness[M]` slammed `-1e9` on any demo
with `in_n ≠ out_n`, and `arc_solve` scored shape-changing pairs 0. Output shape was never even
*represented* as a quantity distinct from the input. This block builds the reusable seam and proves
one shape-change family cold.

**Why a new trait, not an extension of `Memory`.** A same-shape `Memory.apply(weights, inp, dst)`
has no place to learn a different output size, and giving all 10+ existing memories an output-shape
method (defaulting to input shape) would churn every one and reintroduce the OOB the `-1e9` guard
exists to prevent. So `ShapeMemory` is a distinct trait (parallel to `SelfModMemory`, the B4
precedent): `write(state, demos)` infers the shape rule closed-form, `out_rows`/`out_cols` predict
the output dims, and `apply(state, inp, out_rows, out_cols, dst)` produces that many cells. The
same-shape core and every existing memory are untouched; the shape-aware fitness/fit driver is a
generic `[M: ShapeMemory]` sibling. Additive, zero regressions.

**Why the composition pattern again.** A shape-change task factors — like block 5 and the
content×geometry block — into two factors fit on signals invariant to each other:
1. a **shape rule** `out = round(k·in + b)` per axis, WRITTEN closed-form by least-squares over the
   demo dim-pairs (position-free shape arithmetic, no geometry knowledge, one pass, never
   ES-searched); and
2. the **content** — the *proven* AttnGather gather, generalized so its query grid is the OUTPUT grid
   and it reads the INPUT grid. `M=I` reads a centred crop, `M=sI` a subsample, `M=±perm` a
   flip/transpose within the resize. ES-fit over the same 7 attention params, on the same B3
   landscape, only reading a differently-sized output.

**Why an output-shaped gather is a one-variable change.** `AttnGatherMemory.apply` already loops the
query over the grid centred on `inp` and gathers from `inp`. Decoupling the *query* extent/centre
(now the output's) from the *source* extent/centre (still the input's) — plus the output stride on
`dst` — is the whole mechanism. For out==in it reduces to the old code bit-identically, so the new
`apply_shaped` is what `apply` now delegates to; `test_attn_memory`/the fast gate stay 1.0 unchanged.

**Why the demos must vary input size.** With one input size the shape rule's slope/intercept are
underdetermined (only their combination at that size is pinned) — the honest analogue of the n=2
signature ties. So the proof draws each demo (and the held-out test) at a RANDOM size in [4,8]:
`≥2` distinct sizes identify `(k,b)`, and the fresh-size test is an uncheatable probe that the *rule*
generalizes, not a memorized size. The least-squares fallback (mean ratio, b=0) keeps the write exact
at a fixed size for the underdetermined case; documented, not fought.

**Where the ES search lands.** `fill_scale` zeros the shape-rule slots (the GeomColor freeze trick),
so the ES moves only the 7 attention params; the written shape rule rides along frozen. `fitness_shape`
scores at the OUTPUT area and heavy-penalizes a predicted-shape/true-shape area mismatch (the honest
successor to the same-shape guard — a wrong shape rule can't be rescued by content). The L2 anchor's
frozen-slot terms cancel in the antithetic F+−F−, so only content feels it. `fit_shape` is a
self-contained annealed ES (its own scratch, sized to output capacity, like `meta_fit_selfmod`);
`fit_shape_geom` is the per-task driver (write → fit, constant-compute budgeted).

**Result — `test_shape_change` (full tier), cold, held-out at a fresh size:**
- Ckpt A: the least-squares write recovers crop1's `(k=1, b=−2)` per axis exactly.
- Ckpt B: `{crop1, flip_h_crop1, subsample2}` each **held-out 1.0**, per-task cold — including
  subsample2, where the ES had to find `M=2I, t=(−0.5,−0.5)` (a 2× scale-up) from the identity seed.
- Control: the SAME content fit WITHOUT the shape write (identity shape rule) predicts the wrong
  output size on every pair → **held-out 0.0** — the inferred shape rule is load-bearing, not
  scaffolding.
- Fast gate + `mojo format` clean; same-shape numbers unchanged (bit-identical gather).

Synth side: `SHAPE_TRANSFORMS` (crop1, flip_h_crop1, subsample2) + `generate_shape_task_groups`
(varies input size across a task's demos); `corpus_stats` reports the emitted bundles as 0% same-shape
(the `.task` format already self-describes per-grid dims — the seam was always consumer-side).

**Deferred (documented):** upscale/tiling (need a floor/modular gather — a follow-on family on this
same seam; the affine gather provably can't express blocky replication); wiring `arc_solve --report`
to score the real 32% (follow-on — keeps this a clean synth proof); colour composition on top of shape
(a `write_color` pre-map, which commutes cellwise). Latent same-shape assumptions in
`_selfmod_meta_fitness` and the selfmod-grid `adapt` output reads are left as-is (those stay same-shape
families). Base for the next entry: 0e138cf.

## 2026-07-03 17:12 — Re-ran the real ARC-AGI-2 public-eval split (120 tasks), 6 workers, budget 64/1500

Re-measured `arc_solve --report` against `data_bin/arc2_eval` (`eval_parallel.sh data_bin/arc2_eval
scratch/arc2_eval_results.txt 6 64 1500`) at the documented corpus budget, same as every prior
real-corpus number. `arc_solve.mojo` still fits `GeomColorComposedMemory` only (block 5) — the
shape-change seam (`ShapeGeomComposedMemory`/`fit_shape_geom`, landed this session) is **not yet
wired into `arc_solve`** (a documented deferral), so this run is a direct like-for-like comparison to
the prior eval-split baseline, not a measurement of the shape work.

**Result:** `Solved 0 / 120 (solve rate: 0.000%, mean held-out: 0.387962)`, scored in 8204s (~137 min
wall on 6 workers — noticeably slower than the ~56 min a prior run logged at the same budget; the
machine had only ~1-3GB free RAM for most of the run, so this looks like memory-pressure/swap
slowdown, not a change in per-task cost).

Mean held-out **0.388** vs the last-logged eval-split number of **0.319** (both 0/120 exact-solved —
the corpus's genuine shape-changing/multi-rule tasks are still out of reach without the shape seam
wired in). The improvement is consistent with noise across the run-to-run seeded ES stochasticity
already documented for these composed memories) rather than a code change on the eval path — no
`arc_solve`/`memory_composed` edits landed between the two measurements. Exact-solve stays 0 because
the composed memory is still same-shape-only on this driver; the shape-change seam is the natural next
rung to wire in before the number can move on the ~32% shape-changing slice this run still can't touch.

Raw dump: `scratch/arc2_eval_results.txt` (gitignored, reproducible per-task via the per-task RNG
seed).

## 2026-07-04 07:40 — Upscale/tiling: the output-GROWING families land (Next #1 rung a) — via a measured grid of 12 configurations

The roadmap's shape rung (a): outputs LARGER than the input — blocky upscale (each cell → an s×s
block) and tiling (the grid replicated k×k). The shape rules (k=s, b=0) were already covered by the
least-squares shape write; the whole battle was the CONTENT gather. This block took ~12 measured
fit-configurations to land honestly; the failures fixed the design, so they're recorded.

**Correction to the last entry's claim.** "The affine gather provably can't express blocky
replication" is WRONG for upscale: `floor(r/s) = round((r−(s−1)/2)/s)` is an exact identity with no
ties, so nearest-cell reading of an affine map expresses blocky upscale exactly (probe-verified:
hand-set `M = I/s` scores 1.0 at sharp temperature, both parities, s∈{2,3}). What is genuinely
outside any affine `(M, t)` is TILING — `out[r] = in[r mod n]` is a sawtooth. And even for upscale
the EXPRESSIBILITY was never the issue — the FIT was.

**The mechanism (memory side).** `attn_gather_toroidal` (memory_es) — the output-shaped AttnGather
read with three additions, used by `ShapeGeomComposedMemory.apply`:
1. **Toroidal source**: per-axis displacements wrap into (−extent/2, extent/2], so a query past the
   edge reads the input's periodic image — tiling's sawtooth is the nearest WRAPPED cell of an
   affine map. A substrate choice (precedent: the selfmod-grid memories' toroidal neighbourhoods),
   not a task primitive. Window span capped at the torus period (else a cell is scanned twice).
2. **Extent-relative translation** `trel` (2 new slots, SHAPEGEOM_DIM 11→13): the centred frames
   leave tiling with a size-dependent phase (n/2 for k=2) that no constant t can cancel across
   varying demo sizes; `q += trel·extent` absorbs it with one size-free parameter.
3. **Query normalization by the WRITTEN shape slope** (`v_out/k`): resize-as-identity. Without it a
   resize family's `M = 1/k` and its exactness tolerance shrinks with the output extent (upscale-2's
   m11 needed ±0.045 — under the ES's settling noise at any workable sigma; measured: fits parked at
   0.554, the plateau EDGE, four runs in a row). Normalized, `M = I` IS every pure resize and all
   tolerances are size-free. The shape factor informing the content factor's coordinate frame is the
   composition pattern once more.

**The fit (driver side) — what the 12 configs taught.** The permutation families read at integer
positions (temperature-insensitive); upscale reads at ±1/4 offsets and NEEDS a sharp read. The
failure grid, each cell measured (fit_shape_geom):
- Temperature SEARCHED (any seed, any preconditioner): the ES drives it SOFT — it optimizes the
  Gaussian-smoothed objective, where a soft read is robust to the sampler's own jitter. Geometry
  converges exactly; the read blurs; held-out ~0.12. Seeded sharp (raw 3.0) it re-softens to 1.4.
- Temperature ANNEALED up mid-fit (soft→sharp alongside sigma/alpha, wide or narrow): the staircase
  landscape + shrinking sigma = a noise walk; the geometry itself diverges (t drifted to ±1.5).
- Temperature FROZEN sharp, sigma annealed to 0.01/0.05: below the staircase's ~0.25 step scale the
  antithetic differences are almost always zero — divergence (M diag hit 2.24).
- Temperature FROZEN sharp, sigma floored at 0.15 (the step scale): tile2 1.0/1.0 — the DISCRETE
  regime works when the ES is run as the stochastic hill-climber it then is. But upscale parked at
  the plateau edge (its pre-normalization tolerance was under the floor), and one seed frame cannot
  serve both families (below).
- An L2-anchor-parks-at-plateau-edge hypothesis was falsified cleanly (reg=0 reproduced the same
  trajectory to 5 decimals — the 1e-4 anchor is ~2e-6 in fitness, negligible).
**Two identity frames.** With the normalized query, upscale's solution is the SEED (`M = I`) — and
tiling's moved to `M = kI, trel = (k−1)/2` (corner-aligned periodic read), one unit of travel away;
fits from the wrong frame reliably fall into a degenerate constant-read basin (measured both
directions: whichever family's solution is at the seed solves, the other collapses). A k-fold size
change simply HAS two canonical identity continuations — the rescaled plane and the periodic plane —
both derivable from the WRITTEN slope. So `fit_shape_geom` runs the SAME TWO cold starts for every
task, each DISCOVER (wide soft anneal, temperature searched — the proven-smooth landscape) then
SETTLE (temperature hard-frozen at `SHAPE_BETA_READ` via `ShapeGeomSettleMemory` — fill_scale is
static per type, so the phase difference is a thin delegating type — sigma HELD at the 0.15 step
floor where the plateau-edge gradient stays alive, alpha decayed: measured to centre t/trel to
~0.01), inside the same total budget (half each), winner by demo fitness at the hard read.
An honest multi-start — selection by the task's own train signal; no task-specific staging anywhere.
`trel` gets a small ES scale (0.2): it multiplies the extent, and the frame seeds already place it
at its solution — refinement only.

**Result — `test_shape_change` (full tier, 4m48s):** all five families cold, held-out at fresh
sizes: crop1 **1.0**, flip_h_crop1 **1.0** (the real-travel case, still discovered at the halved
per-start budget), subsample2 **1.0**, upscale2 **0.98**, tile2 **1.0**. Controls: tile2 through the
PLAIN gather **0.14** (the toroidal wrap is load-bearing); no-shape-write **0.0** (unchanged). The
experiment battery's final config: upscale2 1.0/1.0 (M≈I, |t|,|trel| ≤ 0.02), tile2 1.0/1.0
(M≈2I, trel = 0.4998 — the settle phase centres to ~3 decimal places).

Synth: `upscale2`/`tile2` added to `SHAPE_TRANSFORMS`. The doubling families' test dims are [3,6]
(outputs ≤ 12×12) to keep the full-budget fit cheap; identifiability (≥2 distinct sizes) unchanged.

**Deferred (documented):** non-uniform factors (upscale3, tile3 — expected free: seed B's
trel = (k−1)/2 is exact for every k, `1 ≡ 0 (mod 1)` for k=3); mirror-tilings (sign flips near seed
B); wiring `arc_solve --report` (rung b — the user wants BOTH corpus splits re-measured when it
lands); colour on top of shape (rung c).

## 2026-07-04 11:54 — Rung (b): the shape seam wired into arc_solve; real ARC-AGI-2 v3 measure

`arc_solve.mojo` now DISPATCHES per task on a closed-form observable of the demos: any train pair
whose dims differ → `ShapeGeomComposedMemory` via `fit_shape_geom` (the rung-(a) two-frame
multi-start); all-same-dims → the unchanged `GeomColorComposedMemory` path (byte-identical for the
same-shape 68%). This is driver-level routing, not a runtime memory-selector — the same-shape
memory PROVABLY scores 0 on the dispatched class, so nothing is being "chosen" that the data
doesn't force. Shape-path scoring: the memory predicts its own output dims from the written rule;
a predicted/true dims mismatch scores that pair 0 (never applied — no OOB). The old
skip-if-every-test-shape-changes shortcut survives only for same-shape-dispatched tasks (where it
remains exact). Each per-task line now carries a trailing `mem: same|shape` marker (appended after
the existing fields — `eval_parallel.sh` reads held-out positionally), giving corpus breakdowns for
free. CI: the full tier's arc_solve leg adds one crop1 shape bundle (fast gate unchanged, ~2 min).
Smoke proof: mixed synth dir {flip_h, crop1, upscale2} → 3/3 solved, markers correct.

**Eval split v3** (`eval_parallel.sh data_bin/arc2_eval scratch/arc2_eval_v3.txt 10 64 1500`,
10 workers, 11197s): **0 / 120 solved, mean held-out 0.404** (v2: 0.388). Breakdown from the
markers: 39/120 tasks dispatched shape (32.5% — matches corpus_stats exactly); their mean held-out
is **0.054, 0 solved** — the honest first number on the previously-untouchable slice. The
like-for-like control holds: the 81 same-shape tasks mean 0.572 vs v2's implied 0.575 (seeded-ES
noise) — the wiring changed nothing on the same-shape path. Top shape near-misses: 136b0064 (0.66,
gap 0.03) and eee78d87 (0.61) — right predicted dims, partial content. The verdict matches the
rung's documented limits: real eval-split shape tasks overwhelmingly have CONTENT-dependent output
sizes (outside the affine-in-dims rule) or need colour on top of shape (rung c) — the seam is
measured, the expressiveness gaps are now named and quantified.

**Train split v3** (same harness/budget, 10 workers, 38808s ≈ 10.8h): **22 / 1000 solved (2.2%),
mean held-out 0.501** — more than DOUBLE v2's 10/1000. The marker breakdown decomposes the gain
exactly:
- **shape-dispatched: 320/1000 (32%), 9 solved, mean 0.239** — the first real-ARC shape-changing
  solves ever (v2 scored this whole slice 0 by construction): 2dee498d, 60c09cac, 68b67ca3,
  8597cfd7, 8d5021e8, 963e52fc, a416b8f3, be03b35f, c59eb873. Near-misses at 0.89 (53b68214,
  2dc579da). Train's shape slice scores far above eval's (0.239 vs 0.054) — smaller grids, more
  affine-in-dims size rules.
- **same-shape: 680/1000, 13 solved, mean 0.625** — net +3 vs v2's 10. Not from this rung: the
  same-shape path is byte-identical here; the delta is the **few-demo hardening** (0e138cf), which
  landed AFTER the v2 measure and is corpus-measured for the first time now: +5 new solves
  (3c9b0459, 5582e5ca, 74dd1130, 9dfd6313, and its designed exhibit **d511f180 — solved at corpus
  budget, as the block predicted**), −2 lost (b1948b0a 0.94, ed36ccf7 0.78 — the identity-on-tie
  convention's documented trade).

Raw dumps: `scratch/arc2_eval_v3.txt` / `scratch/arc2_train_v3.txt` (gitignored; per-task
reproducible via SOLVE_SEED). The v3 verdict for the roadmap: the shape seam is now measurable and
productive on the real corpus (train), and the eval split's shape slice names rung (c) — colour on
top of shape — plus content-dependent output sizes as the binding constraints there.

## 2026-07-04 23:40 — v3 diagnostic breakdown → the expressiveness rung plan (ROADMAP "Next" rewritten)

Mined the v3 dumps (`scratch/arc2_{train,eval}_v3.txt`, per-task lines with `mem:` markers; all
numbers below reproducible with awk over held-out = field 4, train-fit = field 6) to rank the next
expressiveness rungs on evidence rather than intuition:

**Train same-shape (680):** 13 solved | **88 near-misses at held-out 0.90–0.99, mean train-fit
0.93** (fail on a FEW cells — the biggest shallow-headroom pool) | 238 at 0.7–0.9 | 146 at <0.4
with mean train-fit **0.34** — the deep floor: these can't even fit their demos (multi-step /
object-level rules; CMS-chain territory).
**Train shape (320):** 9 solved | quadrants: only 3 tasks are train-high/held-low (what the memory
expresses, it generalizes — again) | **107 with train-fit ≥0.5** (dims + most content fit —
convertible by better content, i.e. colour-on-shape) | **63 with train-fit exactly 0.0** — the
affine-in-dims rule fits NO demo: content-dependent output sizes.
**Eval shape (39):** 19/39 in that dims-never-fit class, 14 partial, 6 with train-fit ≥0.5 —
content-dependent sizes DOMINATE eval's shape slice.

The ROADMAP "Next" section is rewritten as the evidence-ranked rung ladder, each with its
research/implementation split and named blocker: **C** colour-on-shape (kernel: count signatures
aren't conserved under shape change; validate area-ratio normalization, measure crop's border-loss
robustness at n=3; fallback = correspondence write after a geometry prefit), **S** shape-from-
content (shape WRITE over a small content-statistic basis, residual-selected — bbox-crop the
headline class; audit the 63 ids first), **A** the near-miss audit (measure-first; leading
candidate a self-written mask/gate), **D** k=3/mirror-tiling cheap extensions, **CMS** the depth
chain (expect a wall past depth 2 — literature pass planned at it), with the **GPU gate** as an
explicit zero-capability infrastructure block scheduled immediately before CMS: CPU suffices
through C/S/A/D; the blocker is the `mojo==1.0.0b2` pin (no `gpu` package — MAX migration, all
proof numbers re-proven), not kernel design (the ES is embarrassingly parallel; ~10–30×/fit
realistic).

Overall split: roughly half research, but each rung's research kernel is ONE identifiable question
— the pattern of the last three landed blocks. No engine code in this entry; next session starts
Rung C.

## 2026-07-05 — Rung C: colour on top of shape (ShapeGeomColorComposedMemory)

The top-ranked expressiveness rung. The shape path (`ShapeGeomComposedMemory`) expressed
shape+geometry but had NO colour remapping, so the ~107 convertible train-shape tasks (train-fit
≥0.5, "dims + most content fit") plus eval analogues lost the recolored cells. Rung C composes a
written colour table V on top — the block-5 recipe's THIRD application (after GeomColor, GeomCount):
`out = shape_geom_gather(V(in))`. Colour is cellwise, so it commutes with the copy gather
(colour-then-gather); V is written closed-form, then the unchanged two-frame `fit_shape_geom` runs
on V-pre-mapped demos (the geometry search never sees V — exactly how `fit_geomcolor` reuses
`fit_operator[AttnGatherMemory]`).

**New surface (all additive; the same-shape core and every existing memory untouched):**
`ShapeGeomColorComposedMemory` (a thin ShapeMemory wrapper: prefix `[0:SHAPEGEOM_DIM]` the
shape+geometry state, suffix `[+COLOR_DIM]` the written V; `apply` = the toroidal shape gather then a
hard V-lookup; `fill_scale` freezes V), `write_color_shaped`, and the `fit_shape_color` driver.
`arc_solve` routes the shape-dispatch branch through it; V=identity makes it byte-identical to the
old shape path. Synth `recolor_{crop1,subsample2,upscale2,tile2}` families + `tests/test_shape_color`.

**Research kernel #1 — the count-signature write breaks under shape change.** `write_color`
matches per-colour COUNT vectors, assuming count CONSERVATION; shape change breaks it (upscale/tile
multiply every count by the area ratio kr·kc EXACTLY; crop/subsample scale it approximately —
crop drops a colour-dependent border). Fix: normalize each demo's histogram to FRACTIONS (÷ cell
total) before the mismatch — scale-invariant, so `frac_in[c]` matches `frac_out[V(c)]` under any
proportion-preserving resize. Exact for upscale/tile; robust for crop/subsample.

**Research kernel #2 — the write needs colour-count CONTRAST (measured).** The fraction match
identifies a permutation only when colours have DISTINGUISHABLE frequencies. Under EXACT same-shape
conservation (GeomColor) even a uniform-random grid's tiny fluctuations match exactly; shape change
is LOSSY, so the signal must be REAL contrast — which real ARC grids have (background + sparse
objects, very different counts) and uniform-random grids lack. First measured exhibit:
`recolor_crop1` Ckpt A missed 5/10 colours on uniform grids, recovered fully once grids were drawn
from a per-task palette of distinct frequencies (real-ARC-like). The uniform-crop ceiling is kept
as a control (6/10 missed — the precondition is load-bearing, documented not fought).

**The strict-superset trap (found via the arc_solve smoke, fixed).** A pure-shape crop task on
UNIFORM grids has no recolor, but the greedy-injective assignment overfits the border-loss noise and
writes a SCRAMBLED V — regressing pure-shape crop from held-out 1.0 to 0.17 (diagnostic:
`V=[6 1 2 4 4 5 6 7 8 3]` instead of identity). Since real corpus shape tasks are overwhelmingly
pure-shape, this would have silently regressed the v3 shape slice. Fix, MEASURED not guessed:
keep `write_color`'s complete greedy-injective matching (full recolor recovery) but add a GLOBAL
ACCEPTANCE GATE — accept the whole map only if its total fraction-mismatch `R_assign < 0.4·R_id`
(the identity assignment's total). A forced injective permutation on a task with NO recolor has a
HIGH residual (measured greedy R_assign/R_id **0.82–1.35** on uniform pure crop across 7 seeds —
forcing a permutation *costs*); a genuine recolor's true permutation has a LOW one (**0.15**
palette crop, **~0** exact-conservation upscale/tile). The 0.4 gate sits wide in that gap
(insensitive across [0.3, 0.55]; real-ARC contrast pushes recolor toward 0). A GLOBAL ratio, not
per-colour — noise averages out (a per-colour 0.5 gate failed: min-over-targets is systematically
below identity on pure noise; and a MUTUAL-best assignment protected pure-shape but under-recovered
recolor, 0.875 — the greedy complete matching is what recovers all colours). `test_shape_color`
guards this: a pure UNIFORM crop1 must keep V=identity AND held-out ≥0.95.

**Also measured:** subsample-recolor needs adequate output cells (a 2×2 subsampled output starves
the low-frequency colours) — synth subsample dims lifted to {8,10,12}. Real-ARC subsample tasks
aren't 4×4; the identifiability precondition, not a fudge.

**Result — `test_shape_color` (full tier, cold, held-out at FRESH sizes):**
`recolor_{crop1,subsample2,upscale2,tile2}` each **1.0**; colour-ablation (V=identity) **0.0** (the
colour module is load-bearing); pure tile2 through the colour memory **1.0** and pure uniform crop1
**1.0 with V=identity** (the strict superset); few-demo n=3 (corpus median) **0.998**; ceiling
control (uniform crop, no contrast) 6/10 colours missed. arc_solve smoke {flip_h, crop1,
recolor_crop1}: flip_h 1.0 (same-shape, byte-identical), crop1 1.0 (pure shape preserved),
recolor_crop1 0.28 (synth uses uniform grids = the documented no-contrast ceiling). CI's full-tier
arc_solve leg adds a `recolor_crop1` bundle so the `fit_shape_color` path runs end-to-end.

The composition pattern lands a third time; the fraction-signature write + the measured contrast
precondition + the global recolor gate are the new, reusable pieces. Corpus v4 re-measure
(both splits, documented budget) is the separate overnight trigger, deferred by scope.

## 2026-07-05 21:20 — Rung S audit: the shape-write gate returns STOP (measure-first)

Rung S (shape-from-content) is a measure-first rung — the ROADMAP mandate was "audit the 63 ids
before coding," with an explicit GATE to stop if the class fragments with no dominant basis. Built
`tools/shape_from_content.py` (reads the dims-never-fit ids straight from the v3 dumps — `mem: shape`
+ train-fit == 0.0, exactly the 63 train / 19 eval the breakdown named — and reads their grids from
the raw ARC JSON via the `arg-agi-2-data` symlink). For each task it tests whether the output dims
follow a small basis of INPUT content statistics (non-bg bbox dims via border-majority background,
distinct-colour count, input dims) under both an EXACT rule (out == feature, k=1 b=0 — the
non-overfit signal) and a free affine `round(k*f+b)`.

**The verdict — no dominant basis; the class is content-model-bound, not shape-write-bound:**
- **Exact bbox-crop (the headline hypothesis) fits 2/63 train, 0/19 eval.** The strong, unambiguous
  signal is ~3% of the class. My plan's headline sub-class does not exist at scale here.
- Free-affine "coverage" (nonbg_bbox ~24%, distinct_colours ~27%) is **spurious**: a 2-param affine
  through the corpus-median 3 demos fits by chance, and `distinct_colours → output_size` is not a
  real rule. It is not a basis, it is overfitting the identifiability floor.
- **54/63 train (86%) and 18/19 eval (95%) are output < input** — reductions / selections /
  extractions (e.g. 27a28665 3×3→1×1 = which-colour-dominates; 1b2d62fb 5×7→5×3; 1a6449f1 shrinks as
  its object count grows). 9/63 are CONSTANT output size — which the *existing* affine already
  predicts (k=0) — yet they scored train-fit 0.0 anyway, because the CONTENT rule (a reduction) is
  outside the AttnGather copy-gather family. So even a correct output-size write yields no solve.

**Gate decision: STOP — do not build the content-statistic shape write.** Rung S as scoped (a shape
WRITE over content statistics + the unchanged gather) would convert ~2 tasks; it is a documented
NEGATIVE / measurement result, not a milestone. The measure-first discipline paid off exactly as the
ROADMAP's long-tail hazard predicted — a rung's worth of machinery (state-layout ripple, synth
family, proof) was about to be spent on 2 tasks. The real binding constraint for the dims-never-fit
class is a CONTENT model that can EXTRACT / REDUCE / SELECT (object-level: largest object, dominant
colour, a coloured region, symmetry completion) — the output is not a positional copy of the input,
so no shape+gather memory expresses it regardless of how the output size is written. That is Rung CMS
/ object-level territory (multi-stage, content-generating), reachable only by the depth-chained
composition pattern, not a closed-form shape write. `tools/shape_from_content.py` is kept as the
reusable evidence (reproducible: `python tools/shape_from_content.py`). Next direction is the user's
call — the evidence re-points "the path after Rung C" from the shape write toward the content family.

## 2026-07-06 11:49 — Rung A audit: the near-miss failures are UNDER-APPLICATION (content-conditioned local writes the copy-gather can't express)

Rung A (the same-shape near-miss audit) is measure-first: characterize WHERE the 88 train + 12 eval
same-shape tasks at held-out 0.90-0.99 go wrong before designing anything. Added a `--diff` mode to
`arc_solve` (guarded, report-implied, emits EXTRA `DIFF in/pred/true` lines only for the same-shape
path — the positional per-task contract eval_parallel.sh reads is untouched) that dumps the fitted
`GeomColorComposedMemory`'s test prediction vs the truth. Ran all 100 near-miss ids at the v3 corpus
budget (64/1500, 8 shards) so the diffs are the REAL few-wrong-cells regime, then clustered them with
`tools/near_miss_audit.py` (per-task fingerprint: border fraction, colour-swap kinds, under/over-
application vs the INPUT, contiguity of the wrong-mask).

**The verdict — one dominant, coherent cluster (the gate PASSES, not fragmentation):**
- **under-applied 55 + border 4 = 59/100.** At the wrong cells `pred == INPUT` but `true != input`:
  the memory OUTPUT IDENTITY where the true rule makes a local change. It nails the global transform
  (hence held-out 0.90-0.99) and misses a localized/content-selected subset. This is exactly the
  ROADMAP's leading candidate (a self-written content MASK/GATE), now MEASURED not guessed.
- scattered 30 (mixed under/over, many colours — the genuine long tail), over-applied 6, colour-swap
  4 (colour table a hair off — few-demo tie territory), region 1.

**Sharpening the mechanism (what KIND of under-application):** of the 59, only **11 are one
contiguous spatial blob** (a spatial mask); **48 are DISTRIBUTED**. By colour of the missed cells:
**44/59 share a single input colour** — but honestly split, **29 miss BACKGROUND cells** (the rule
WRITES NEW CONTENT into empty space — fill / draw / extend / complete — which a copy-gather can never
generate, since every output cell is a recoloured copy of some EXISTING input cell) and **15 miss a
specific NON-background object colour** (a local transform of one coloured object the global map
can't isolate). So the gate is content-keyed and often generative, not a clean spatial mask.

**Root cause named (the mechanism the audit licenses):** the composed memory is a COPY-gather + a
global colour map; it cannot express a LOCAL, CONTENT-CONDITIONED WRITE — whether filling background,
transforming one object, or a small region. The natural in-family lead: this is precisely what the
existing self-mod GRID memories already do (`memory_selfmod_grid.mojo`: GridContext / GridNbhd /
GridCountMap write per-cell outputs conditioned on the cell's colour + Moore-8 neighbourhood), but
they were only ever proven on synth in ISOLATION — `arc_solve` never dispatches them; the same-shape
path is copy-gather-only. The Rung A mechanism is therefore composing a content-conditioned local
write (the self-mod grid family, or a new gate keyed on colour/neighbourhood) WITH / on top of the
gather — the first real content-gated composition on the corpus, exactly the roadmap's "first step
toward content-gated composition." Whether that lands as a driver-level dispatch, an additive energy,
or a genuine composed memory is the DESIGN question for the next rung (not decided here — measure-
first stops at naming the mechanism). Evidence kept: `tools/near_miss_audit.py`, `scratch/rungA/`
(reproducible: `--diff --fit 64 1500` over the near-miss ids).

## 2026-07-06 14:55 — Rung A build (Approach 1b): a local content-conditioned write, composed on the gather

The Rung A audit (11:49 entry) named the mechanism: 59/100 same-shape near-misses are
UNDER-APPLICATION (the copy-gather+colour memory outputs identity where the true rule makes a LOCAL
change), and 80/100 are (near-)identity geometry + a local rule. Approach 1b (the plan's cold,
values-clean choice) adds the composition pattern's FOURTH closed-form factor: a per-cell local
content override WRITTEN from the demos, composed on top of the proven GeomColor gather.

**Mechanism.** `LocalWriteComposedMemory` (memory_composed.mojo): layout [0:GEOMCOLOR_DIM] the
gather+V state | [+90] a written table keyed on a bounded, position-free local SIGNATURE — (centre
colour, #Moore-8 neighbours DIFFERING from centre), 10x9. `write_local` majority-votes each
signature's output over the demo cells (support + dominance thresholds; sentinel -1 = "keep the
gather output" for unseen signatures — an n-gram prior). `apply` = GeomColor.apply then override by
signature. `fit_local` (esper_evolution.mojo) = the unchanged fit_geomcolor, then write_local on the
gather's RESIDUAL. A GLOBAL ACCEPTANCE GATE keeps the table only on a strict demo-residual
improvement over a non-exact gather — so a pure geometry/colour task writes NO table and the memory
is BYTE-IDENTICAL to GeomColor (strict superset; the override is input-position-keyed, correct only
when the gather is ~identity, so a real-geometry task fails the gate — the 20% geometry-and-miss
slice honestly deferred). arc_solve's same-shape branch routes through it.

**Proof (test_local_write, full tier, cold, held-out at a FRESH blocky grid):** `outline` (edge /
object-colour class) **1.0**, `fill_enclosed` (background-fill class) **1.0**, few-demo n=3 **0.975**;
ablation (GeomColor only) **0.62** (the local write is load-bearing); strict superset (a pure recolor
writes an EMPTY table AND stays **1.0**). The synth families need STRUCTURED (blocky) grids — a
background with solid regions — so the local signature is identifiable (uniform-random grids make
every cell a border); a 2-colour palette + the fully-enclosed (diff-count==8) fill signature keep the
n-gram key well-covered (the honest identifiability precondition, like the shape tasks' varying
sizes).

**Corpus re-measure (the 100 near-miss ids through the new fit_local path, budget 64/1500, vs v3):**
**+2 new solves** (97c75046, b1948b0a — the latter RECOVERS a documented v3 identity-on-tie loss to
1.0), **54/100 improved, 0 REGRESSED** (mean held-out 0.9403 -> 0.9516). The strict superset held
exactly: zero regressions. `./esper fast` stays green (the fast tier's memories are unchanged; the
arc_solve same-shape leg now runs fit_local and still solves flip_h/recolor).

**The honest verdict + Approach 2 scope.** 1b is a real but BOUNDED gain, exactly as the plan
predicted: the (centre, differing-count) n-gram key + corpus budget capture the clean local-signature
classes (outline/fill cold-proven) and broadly help (54 improved) but only push 2 near-misses over the
solve bar — the remaining 52 improved-but-unsolved need a richer / meta-trained local read (the
disjunctive/count classes GridNbhd/GridCountMap express). That 52-task slice is now the
evidence-backed motivation for **Approach 2 (a meta-trained self-mod grid factor) folded into roadmap
#6 (persistent slow weights)** — designed with the M9 wash-out mitigation, not bolted on. The
composition pattern's fourth application; the reusable new pieces are the local-signature write + the
residual acceptance gate. Corpus v4 re-measure (both full splits) stays the deferred overnight
trigger.

## 2026-07-06 16:01 — Rung D measure-first: k=3 mostly free, subsample/3 deferred, mirror-tiling is the prize

Started Rung D ("k=3 factors and mirror-tilings", the roadmap's small paid-for extensions) with a
measure-first pass — probe the existing `fit_shape_geom` at k=3 BEFORE writing any machinery, and
count the corpus prevalence of each candidate so effort tracks solves (the Rung S discipline).

**k=3 probe (scratch, full FIT_* budget, held-out at fresh sizes):**
- `tile3` **1.0**, `upscale3` **0.966** — the output-GROWING k=3 families ALREADY work with ZERO code
  change. The shape rule is least-squares (writes kr=3), `seed_periodic` sets M=kI algebraically for
  any k, and the toroidal read is `in[r mod R]` for any period — so k=3 grow is already dispatched
  through fit_shape_geom in arc_solve TODAY and already counted in v3. Rung D's job here is a
  REGRESSION TEST, not new solves.
- `subsample3` **0.31** (0.28 even at realistic {6,9,12} sizes). Diagnosed: the fitted M collapses /
  t misses (m00≈0.81, t_r≈-0.2 where the exact solution is M=I, t=-(K-1)/2=-1.0). ROOT CAUSE: the
  query normalization divides by the shape slope kr — for GROW (kr=k>1) this SHRINKS the query range
  and makes tolerances size-free (its design intent), but for SHRINK (kr=1/k, inv_kr=k>1) it
  AMPLIFIES the query by k, so a δM error moves the source read by ~k·(extent/2)·δM — the
  downscale-by-3 staircase step falls BELOW the settle sigma floor (0.15, the noise floor). subsample2
  (inv=2, t=-0.5) survives; subsample3 (inv=3, t=-1.0) doesn't. Fixing it means reworking the core
  normalization the proven GROW families depend on (regression risk) for only ~6 lower-value corpus
  tasks → **documented NEGATIVE / deferred**, exactly the Rung S "don't build core-touching machinery
  for a handful of tasks" call.

**Corpus prevalence (tools scan over data_bin, train/eval):**
- Consistent integer shape ratios (train): x2/x2 **22**, x3/x3 **15**, /3 **6**, /2 **3**, x4 **2**;
  eval has **0** clean k=3 tasks. So k=3 grow is a train-only ~15-task band already covered.
- **Tiling by EXACT pixel content (the gather's actual reach): plain (periodic) tiling 3 train tasks;
  MIRROR (reflect-alternate-tiles) tiling 8 train tasks** (eval 0 of either). Mirror-tiling — the
  kaleidoscope reflect where odd tiles are flipped, `out[R+i]=in[R-1-i]` — is MORE than double the
  plain count and is provably OUTSIDE the current periodic (wrapping) torus. **This is the real Rung D
  prize: up to +8 train solves, genuinely new capability.** The 8 ids are pure geometric mirror-tiles
  (my detector required exact pixel equality incl. colour, so no recolor is entangled).

**Rung D plan (revised by the evidence):** (1) build the mirror-tiling capability as a THIRD identity
frame — a reflect/fold gather + a third cold start in fit_shape_geom (the established "honest
multi-start, not a selector" pattern, alongside the rescaled and periodic frames); (2) lock tile3 /
upscale3 with a regression test (already-working, guard against future breakage); (3) subsample/3 and
k>=4 documented as deferred negatives. Scratch evidence in scratchpad/probe_k3.mojo.

## 2026-07-06 16:42 — Rung D build: mirror-tiling as a THIRD identity frame (reflect gather)

Built the Rung D capability the measure-first pass named as the real prize: MIRROR (kaleidoscope)
tiling — the 8-train-task family (vs 3 plain) that the periodic torus provably cannot express.

**Mechanism (values-clean, minimal-surface).** A k-fold size change has a THIRD canonical identity
frame beyond the resized (M=I) and periodic (M=kI) planes: the MIRROR plane, where odd tiles are
flipped (`out[R+i]=in[R-1-i]`). The key derivation: the mirror read is the SAME centered query the
toroidal gather already computes, with the periodic WRAP (`d -= extent·round(d/extent)`, a sawtooth)
replaced by a symmetric triangle FOLD about the grid edges (`_reflect_fold`, period 2·extent). Two
consequences fall out for free: (1) the SAME `seed_periodic` (M=kI, trel=(k-1)/2) is the exact mirror
solution — verified by hand: that seed makes q the centered source index of the output cell, which the
fold then reflects into the base tile; (2) the triangle is CONTINUOUS (slope ±1, no jumps), so the ES
gradient is actually CLEANER than the sawtooth's. So the whole capability is: `attn_gather_reflect`
(the fold twin of `attn_gather_toroidal`, a plain bounded window over the base cells — no period
doubling, same cost as apply_shaped) + a written `SHAPEGEOM_MODE_OFF` slot (0=torus, 1=reflect;
frozen in fill_scale, a written observable like trel — NOT a runtime selector) that `apply`
dispatches on + a THIRD cold start in `fit_shape_geom` that sets the mode slot and reuses the existing
memory types (apply reads the slot, so no new struct). Winner by demo fitness — the established
"honest multi-start, not a selector" pattern.

**Constant-compute + strict superset preserved.** The reflect frame is a candidate ONLY when the
written shape rule GROWS an axis (a shrink/same-size task can't be a reflection) — so non-growing
tasks keep the byte-identical two-start regime, and the budget is split by the actual start count
(2 or 3), keeping total demo-evaluations constant. Rung C (`ShapeGeomColorComposedMemory`) and
`arc_solve` inherit mirror with ZERO change: Rung C's apply delegates the gather to
`ShapeGeomComposedMemory.apply` (now dispatching), and its driver calls `fit_shape_geom` (now 3-start).

**Proof (`test_mirror_tiling`, full tier, cold, held-out at a FRESH size):** mirror_tile2 **1.0**,
mirror_tile3 **1.0**, each with the reflect frame winning (mode=1). k=3 grow regression: tile3 **1.0**,
upscale3 **0.990** (mode=0, untouched). Controls: mode ablation — a mirror-fitted state read through
the FORCED torus collapses to **0.44** (reflect is load-bearing); strict superset — a plain tile2
still solves **1.0** AND lands mode=0 (the reflect start never hijacks a non-mirror task).
Regression: `test_shape_change` unchanged (crop1/flip_h_crop1/subsample2 **1.0**, upscale2 **0.987**,
tile2 **1.0** — the grow families now run 3 starts at iters/3 and still clear the bar). Also proved
out-of-box: tile3 **1.0** / upscale3 **0.966** already worked at k=3 (measure-first) — locked as
regression families. subsample/3 and k≥4-shrink are documented negatives (the downscale
query-normalization amplifies the staircase past the sigma floor — 16:01 entry).

**Files:** `attn_gather_reflect`/`_reflect_fold` (memory_es.mojo); `SHAPEGEOM_MODE_OFF` + apply
dispatch (memory_composed.mojo); the 3rd start + grow-gate (esper_evolution.mojo `fit_shape_geom`);
`tile3`/`upscale3`/`mirror_tile{2,3}` synth families (synth_tasks.py); `test_mirror_tiling.mojo`
(the full-tier end-to-end proof; NO mirror bundle added to the CI arc_solve leg — a grow bundle at
full budget × 3 starts is ~4 min, a poor CI cost/value trade since the dedicated test already
exercises the fit + reflect apply; the Rung C-wrapper dispatch was smoke-checked manually instead).
**Corpus value:** up to +8
train solves (the pure mirror-tilings; eval has 0), to be booked in the deferred v4 re-measure — and
by prior agreement the shape-slice-only before/after check isolates Rung D's contribution from Rung C.

**Regression cross-check (honest, done not assumed).** End-to-end through the real corpus driver:
arc_solve solves mirror_tile2 ×2 + tile3 **3/3 = 1.0** (mem: shape — the Rung C
`ShapeGeomColorComposedMemory` wrapper inherits the reflect dispatch), and recolor_tile2 (grow +
colour, the 3-start + colour-write path) **1.0**. `./esper fast` green. One shrink+colour probe
(recolor_crop1) scored 0.1 — I `git stash`ed the change and re-ran it on MASTER: **identical 0.1 /
train 0.194**, so it's a pre-existing hard case (crop border-loss corrupts the colour histogram
signature at that seed), NOT a Rung D regression — the non-grow path is byte-identical by construction
(mode slot frozen 0, two-start budget unchanged) and the matching numbers confirm it.

---

## 2026-07-07 — GPU environment: already there (pixi evaluated and rejected)

**09:40–10:05.** Goal: prepare the environment for GPU processing, starting from Modular's
GPU-puzzles howto (puzzles.modular.com/howto.html), which recommends pixi. Measured first instead
of installing: probed the existing uv venv's Mojo 1.0.0b2 directly.

**Finding: no environment work was needed.** The PyPI `mojo` wheel already carries full GPU
support — `has_accelerator()` → True, `DeviceContext()` enumerates the RTX 2060, and a real
`vec_add` kernel (`enqueue_create_buffer` → `enqueue_function` → `synchronize` → `map_to_host`)
compiled, launched, and read back correct results end-to-end on the first working attempt. The
driver (580.159.03) exactly meets Modular's ≥580 requirement (below that you'd need
`MODULAR_NVPTX_COMPILER_PATH` → system `ptxas`); Turing is Modular's "known compatible for
development" tier — fine for our purposes, just not production-serving validated.

**Why pixi was rejected**, not just skipped: the puzzles repo's pixi env pins `mojo <1.0.0`
(pre-1.0 nightlies — `fn`, `alias`, unqualified stdlib imports). Adopting it would have *broken*
our hard 1.0.0b2 pin and its idioms. Pixi's actual value there is conda-side CUDA
profilers/sanitizers, which we don't need yet; if we ever do, it can be added alongside uv without
touching the Mojo toolchain.

**Landed** (branch `gpu-env`): `tests/test_gpu_env.mojo` — the smoke test proving the environment
claim (device name + 1024-element vec_add, per-element check, `raise Error` on mismatch). Gated
with `comptime if not has_accelerator()` so device code isn't even compiled on GPU-less hosts —
CI passes as an explicit SKIP. Untagged → runs in the fast tier (sub-second on GPU). One 1.0
gotcha for future kernels: `@parameter if` is deprecated in b2 — use `comptime if`; kernel pointer
params need the usual explicit `MutAnyOrigin`. Idioms recorded in CLAUDE.md (Toolchain section).

---

## 2026-07-07 — GPU rungs G0–G3: the ES fitness boundary batched on the GPU

**10:20–12:30, branch `gpu-env`.** With the environment proved GPU-capable this morning, the
refactor landed in four measured rungs. The compute map (explored first) was stark: ~all engine
FLOPs are the windowed attention-gather forward inside `fitness`/`fitness_shape` — 2·N·n_demos
forwards per ES iteration × thousands of iterations — while the searched vector is 7–14 wide. So
the GPU seam is exactly the **fitness boundary** and nothing else: the RNG draw (same serial
stream as the CPU path), antithetic coefficients, gradient reduction and update stay on CPU; one
kernel launch per iteration scores all (candidate × demo) pairs.

**G0 (CPU-only, behavior-preserving).** The window-scan bodies of `apply_shaped` /
`attn_gather_toroidal` / `attn_gather_reflect` extracted into `attn_pixel_{plain,toroidal,reflect}`
free functions over raw pointers — the SINGLE source of truth both the CPU loops and the GPU
kernels call (the design rule that makes CPU/GPU divergence structurally impossible). Verified:
test_attn_memory and test_shape_change scores identical to the recorded baselines.

**G1 (same-shape path).** `src/gpu_es.mojo`: `_fitness_kernel_plain` — one thread block per
(candidate, demo), threads striding output pixels, fixed-order shared-memory tree reduction (no
atomics ⇒ a GPU run is bit-identical to ITSELF; per-task seeding keeps corpus numbers
reproducible). `fit_operator_gpu` mirrors `fit_operator[AttnGatherMemory]`'s schedule exactly;
`fit_geomcolor`/`fit_geomcount` route through it under `comptime if has_accelerator()` +
`use_gpu` (CPU-only hosts, e.g. CI, compile zero device code). Buffers alloc-once per fit;
per-iteration traffic is KBs. **The determinism contract (user decision): GPU ≠ CPU bitwise**
(MSE reduction order differs ⇒ ES trajectories diverge); parity is pinned where it is meaningful —
`test_gpu_parity` checks device fitness == CPU fitness (observed ≤ 2e-6 relative on grids up to
30×30, penalties exact), quality at the usual held-out bars. Measured: same-shape real-ARC task
135a2760 at 64/1500: **73 s (12 cores saturated) → 4.4 s (~17×, one core + GPU)**.

**G2 (shape path).** `_fitness_kernel_shape`: same block layout; per-demo PREDICTED output dims
computed once on the host (the written shape slots are frozen during the ES) and ZEROED for
mismatched demos (kernel no-ops — no OOB; `_assemble_fitness` applies the heavy penalty, shared
with G1). Mode slot dispatches toroidal/reflect in-kernel. `fit_shape_gpu` is deliberately
non-generic: the Composed/Settle pair share the SHAPEGEOM layout and differ only in `fill_scale`,
carried by a `settle` flag. Both DISCOVER and SETTLE phases of `fit_shape_geom`'s multi-start now
run batched. Parity: toroidal/reflect/soft/mismatch cases all EXACT (|diff| = 0). The full-tier
shape proofs re-verified on the GPU path at their bars — test_shape_change 36 s, test_mirror_tiling
26 s, test_shape_color 53 s (each formerly minutes).

**G3 (drivers + headline).** `arc_solve`: GPU default on accelerator hosts, `--cpu` forces the
reference path, and the report header prints the backend (`fitness backend: gpu (NVIDIA GeForce
RTX 2060)`) so runs are self-documenting. `eval_parallel.sh`: on a GPU host the worker default
drops to 2 (one process nearly saturates the device; `ESPER_CPU=1` restores the nproc CPU A/B
mode). Gotcha for future me: the script expects the venv on PATH — a worker dying instantly with
"scored 0 tasks in 0s" means `mojo` wasn't found, not a GPU failure.

**Headline (120-task public-eval split, budget 64/1500):** `Solved 0/120, mean held-out 0.486434,
scored in 409 s (~6.8 min)` on 2 workers. Wall-time vs the recorded CPU baselines at the same
budget: 8204 s (memory-pressured) / ~3360 s (clean) → **~8–20× harness-level**, and the GPU run
did MORE work per shape task (the multi-start shape path wasn't wired into arc_solve at those
baselines). Honesty note: mean held-out 0.486 vs the 2026-07-03 CPU run's 0.388 is NOT a GPU
effect and NOT directly comparable — that baseline predates the shape seam + Rungs C/D on this
driver; this is the FIRST corpus number with the shape path wired in (81 same / 39 shape
dispatches). Exact-solve stays 0/120, as expected pre-CMS. The clean same-engine A/B is the
3-task probe: **5m12s CPU → 17 s GPU (~18×)**, identical held-out scores.

**What stays CPU (measured as not worth it):** closed-form writes (once per task), the self-mod
families (off the corpus path), noise/gradient/update (7–14 dims). The CPU `Memory` path remains
the generic reference for every family; `gpu_es.mojo` is deliberately specialized to the corpus
forwards. The ROADMAP's pre-CMS "GPU gate" is hereby done early — its blocker premise (slim wheel
lacks `gpu`) was false for 1.0.0b2, and its prediction (bit-identity will not survive; re-prove at
bars) held exactly. Full-budget (128/4000) corpus runs are now affordable — the budget-raise
experiment is the natural follow-up.

---

## 2026-07-08 — Budget-raise experiment: the corpus is expressiveness-limited, not budget-limited

The follow-up the GPU speedup unlocked, run by hand on the 120-task public-eval split (2 GPU
workers): full proof budget 128/4000 → `0/120, mean held-out 0.483976, 1809 s (~30 min)`. Versus
yesterday's 64/1500 run (0.486434, 409 s): **5.3× more ES compute moved the mean −0.0025** —
seed-trajectory noise, not signal. Conclusion: the 7-param gather + written factors saturate this
corpus at the corpus budget already; the remaining gap is EXPRESSIVENESS (multi-step / object-level
rules), which is precisely the CMS-chain rung. The budget knob is dead as a score lever; the GPU
win is wall time (the full-budget corpus number went from a ~10 h overnight to 30 min).

Two operational notes from the run: (1) a second invocation WITHOUT the trailing budget pair
produced bit-identical numbers — omitting `fit_N fit_iters` selects arc_solve's DEFAULT (the full
128/4000), not the corpus budget, so it was an exact repeat; a clean live demonstration of the
per-task-seed reproducibility contract. (2) Both runs land within ~5% wall time of each other —
the harness is GPU-bound and stable at 2 workers.

---

## 2026-07-08 — In-process INFO progress for the corpus eval

Monitoring an eval no longer needs an outside `watch` on temp files: `arc_solve` now prints
`INFO [+<elapsed>s] task k/n start|done (<s>/task, avg, eta)` around every task (perf_counter_ns;
the eta is per-process = per-shard under the harness), and `eval_parallel.sh` streams each
worker's output live to the terminal with a `[wN]` prefix via `stdbuf -oL … | tee | sed`. The
`tee` keeps the per-worker capture files prefix-FREE — the `^  task:` positional aggregation is
untouched (verified: 6/6 result lines on a 2-worker mini run; footer format identical). Verified
liveness through a pipe (INFO lines visible mid-run, not at exit). INFO is additive monitoring
output only; consumers keep reading the unchanged `  task:` lines. `./esper fast` green.

---

## 2026-07-08 — Rung CMS-0: deep-floor audit — GATE RETURNED STOP (documented negative)

**09:00 — Plan approved for the CMS rung.** Structure: CMS-0 (this measure-first audit),
CMS-1 (chain mechanism + synth proof, only on GO), CMS-2 (corpus wiring), CMS-3 (the wall as a
trigger). Pre-registered gate: GO iff ≥25 of the 146 deep-floor ids (same-shape train, held-out
<0.4 at v3, mean train-fit 0.337 — the bucketing reproduced exactly) land in coherent
chain-of-proven-factors-shaped clusters.

**09:20 — `tools/deep_floor_audit.py`.** Per-task decomposition testing on the RAW demo pairs
(no fitted memory in the loop): depth-2 baseline `out[i] = g(π(in)[i])` over the 8 lattice
symmetries, vs the two proven depth-3 chain keys on the premapped grid — LocalWrite-class
`(colour, toroidal ndiff8)` and count-class `(colour, toroidal count_P)`. All scores
leave-one-demo-out so a rich key must GENERALIZE across demos (the Schug over-capacity hazard
controlled inside the audit).

**09:35 — The audit itself had to be calibrated before its verdict could be trusted.** Synth
ground truth (known depth-3 chains vs pure depth-2 controls at corpus-like n=3) exposed two
defects in the first cut: (1) clamped-border keys vs the engine's TOROIDAL convention broke
`outline` chains at every border cell (score 0.83 where ~1.0 was true); (2) a GLOBAL per-cell
threshold cannot separate "depth-2 plus a small load-bearing residual" (fill: d2 0.989, genuinely
depth-3) from "pure depth-2 with LOO noise" (0.994) — the discriminator must be the PAIRED
residual fix: of the cells the depth-2 table misses, the net fraction the chain key fixes, with
the chain predictor mirroring the actual mechanism (a gated override that FALLS BACK to the base
map on unseen keys, not a replacement — without the fallback, fill chains scored net_fix −11 from
fragmentation misses). After both fixes: chain families label chain-local at net_fix +0.5…+0.99,
depth-2 controls produce ZERO false chain labels (fx +0.00), and the only misses are chains whose
factor has ≤2 cells of total demo support — genuine underdetermination, unlearnable in-context by
any mechanism. Conservative in exactly the honest direction.

**09:45 — Verdict: STOP (3/146 chain-shaped; gate ≥25).** Clusters: **81 unexplained** (neither
proven key family nor any tested symmetry explains the residual — best net_fix < 0.25),
**45 object-level** (object-count deltas ±3…±48: per-object selection / movement / counting),
**17 chain-partial** (net_fix 0.25–0.5 — the proven-factor chain fixes only a quarter to half of
the residual), 2 chain-count + 1 chain-local at the bar. Even chain + partial = 20 < 25. The Rung
S finding repeats one level up: the deep floor is **not chain-of-proven-factors shaped** — the
missing capability is the FACTORS (object-level / content-extraction reads), not the depth of
their composition. Chaining what we have would express ~3 more tasks, not the floor.

**Consequence for the roadmap.** CMS-1/2 as planned (a chain of the existing four factors) is
NOT built — the audit saved the rung's machinery, as CMS-0 was designed to. The evidence points
where Rung A already pointed: the 81+45 need a richer, meta-trained content read (rung #6's
self-mod grid factor, designed with the wash-out mitigation), and the 17 chain-partials become
its first corpus exhibit list. The three chain-shaped ids (543a7ed5, cc9053aa, e4888269) are
real but too few to justify a rung. Audit + calibration protocol journaled here; the tool stays
(`tools/deep_floor_audit.py`) as the reusable gate for any future factor proposal — a new factor
family can be dropped into its key set and re-measured against the same 146 ids in ~20 s.

---

## 2026-07-08 — Factor-coverage scan: the deep floor is not PER-CELL expressible at all

**13:49 — The follow-up the CMS-0 STOP demanded.** The audit said the missing capability is the
FACTORS, not composition depth — so before any literature pass, measure WHICH factor would cover
the floor. `tools/factor_scan.py` drops candidate object-level per-cell key families into the same
calibrated harness (LOO paired residual-fix, gated-override fallback, same bars as the audit's
chain label): component size / size-rank(largest,smallest) / bbox dims, position-in-bbox class,
mirror context (h/v/180), colour-frequency rank, distance-to-the-other-set (BFS, capped). The
substrates (connected components, mirrors, histograms) are representations a learned read could
operate over — the scan asks what a factor must EXPRESS, not how it is learned. False-positive
control: on 20 pure depth-2 synth tasks every family shows net_fix +0.00 and zero false covers,
except sym-h "covering" flip tasks at π=identity — the mirror key legitimately CONTAINS the flip
read, which cannot confound the floor (a flip-expressible task would have been labeled
depth2-fit-failure by the audit; only 1/146 was).

**Result: union of ALL new families = 4/146** (comp-rank 1, comp-size 1, sym-h 1, sym-180 1,
dist 1, comp-dims 1 — overlapping; proven baselines 3), plus 18 near-misses at net_fix 0.25–0.5
(sym-* and dist dominate that band). Even "recolor the largest object"-class reads — the exact
family the object-level cluster's size deltas suggested — cover ONE task.

**The finding, sharper than CMS-0's:** the deep floor lies outside the entire class of per-cell
functions over position-aligned context — however rich the key. What these tasks share (and the
45-task object-level cluster's ±3…±48 object deltas corroborate) is that output content is
constructed at positions OTHER than where the input evidence sits: move/copy/draw/extend rules
are CONTENT-ADDRESSED — the writer must select input structure by content and place it by rule,
which no keyed table over the aligned cell can express by definition. (The 17+18 partial-fix ids
are the boundary band: part per-cell, part constructive.) This is Rung S's "content is not a
positional copy" finding, now measured as the dominant property of the whole floor.

**Where this aims the literature pass** (next step): not richer per-cell keys but mechanisms of
content-addressed retrieval/construction compatible with the spine — content-attention (the
AttnGather generalized from position-queries to learned content-keys, i.e. proper kv-attention
over the input grid), object-slot / grouped representations feeding the self-mod write rule
(rung #6), and constructive/generative test-time-training approaches from the ARC Prize crop.
The scan harness stays the gate: any proposed mechanism's expressible class can be dropped in as
a key/predictor and re-measured against the same 146 ids before it is built.

## 2026-07-08 — Literature pass: content-addressed construction (RESEARCH-NOTES updated)

**14:00** — Ran the literature pass the factor scan commissioned. Full findings + sources in
`RESEARCH-NOTES.md` (2026-07-08 section); the shape of the answer:

The field's 2025 ARC crop converged on exactly the property the floor demands. Nobody in the
prize's top tier computes the output in one per-cell pass — the unifying theme is *generate →
verify → refine*, and the two most Esper-relevant results are tiny-model constructive loops:
**TRM** (7M params, 2 layers, 45% ARC-AGI-1) maintains a materialized answer grid `y` + latent
scratchpad `z` and repeatedly applies one small net (`z ← net(x,y,z)`; `y ← net(y,z)`) — each
pass can write where it didn't read, which is precisely the measured missing capability; and
**CompressARC** (76K params, *no pretraining, no dataset*, 20% ARC-AGI-1) — the closest published
relative of our cold-fit bar, proof that per-task-only fitting has real headroom. TRM's ablations
double as a Schug confirmation: 2 layers beats 4, one net beats two — recursion depth substitutes
for parameters, which favors a derivative-free ES.

**Distilled next steps** (booked in RESEARCH-NOTES + to flow into ROADMAP when acted on):
1. *Nearest rung* — content-keyed gather: AttnGather's query generalized from affine-position
   (7 params) to position+content, making copy/move/draw expressible with a handful of extra
   params, same single ES search, same GPU kernel shape. Gated the same way as everything since
   CMS-0: prototype content-addressed read families in `factor_scan` first, build only on
   coverage evidence over the 146.
2. *Rung #6 shaping* — the meta-trained self-mod write rule becomes an iterated editor over a
   materialized answer grid (TRM's loop), reading object-slot *relations* (Slot Abstractors'
   relational bottleneck: relations-only reads cost −52% when removed — i.e. they're load-bearing
   for generalization) to decide where to write; ES meta-learns only the small rule.
3. *MDL acceptance* — description-length-vs-residual as the uniform capacity guardrail on written
   stages (CompressARC's objective, decoupled from its backprop).

## 2026-07-08 — Content-addressed read scan: GATE RETURNED GO (22/146)

**15:16** — Extended `tools/factor_scan.py` with the content-addressed read families the
literature pass proposed, and the pre-registered gate (content-family union >= 20/146, user
decision this morning) returned **GO: 22/146** — the first positive gate since the deep-floor
work began. Full output preserved at `scratch/content_scan_v1.txt`.

**The design discovery (calibration-driven, before touching the corpus):** the first cut — per-cell
keys that *include* fetched values (`fetch-ray4`, `fetch-registers`, ...) — mostly failed its own
positive controls (0–6/20 per class). Diagnosis: a keyed TABLE can only emit colours it has seen
for a key, so copy-through rules (out = the fetched cell's value, colours varying across demos)
are invisible to it — but copy-through is exactly what a sharp content-keyed gather does. The fix
is the `copy-*` family group: colour-ABSTRACT relational keys (centre==bg, fetched==centre, ...)
whose LOO tables vote abstract ACTIONS — KEEP (out = centre) / COPY (out = fetched) / constant —
implemented in `loo_paired_fetch` (same pairing, gating, and fallback as the audit's `loo_paired`).
After one key refinement (anchor needed fetched==centre in the key to separate KEEP cells from
COPY cells), all five classes calibrate 20/20 at n_train=3, with zero false covers on 20 pure
depth-2 control tasks, and the committed families' corpus numbers stay bit-identical. Two lessons
land at once: (1) the mechanism's essential power IS copy-through, not richer keys — the Mojo
gather must emit the attended cell's VALUE; (2) abstract relational keys are what generalize
across demos with varying colours — the Slot-Abstractors relational-bottleneck lesson, reproduced
independently by our own calibration.

**Corpus result:** copy-registers 12, copy-nearest 10, copy-ray 9, copy-objlocal 8, copy-anchor 7,
fetch-ray4 6 (keyed variants otherwise ~0); content union **22/146** vs the per-cell union's 4.
Greedy cover: copy-registers + fetch-ray4 + copy-nearest = 20. Equally important: the near-miss
band (partial fix 0.25–0.5, or strong fix at loo just under the bar) grew from 18 to **~72 ids**
— the content class doesn't just cover 22, it GRAZES half the floor; a soft ES-fit gather with
content terms plus the usual residual factor writes plausibly converts a slice of that band.

**Next (the gated build):** the content-keyed gather rung — `AttnGatherMemory`'s score generalized
from affine-position queries to position + content-match terms, output = the attended cell's
value; one ES search, `fill_scale` freeze, pre-map recipe and GPU kernel shape carried over. The
22 covered ids are the corpus target list; the scan stays the acceptance harness.

## 2026-07-08 — Rung CF begins: the content-fetch layer (gated build)

**16:20 — CF-0: grid substrate module landed.** Plan approved (mechanism = written fetch layer on
the LocalWrite prefix, per the user's decision; corpus gate pre-registered at net>0 new solves on
the 22 scan-covered ids + 0 regressions vs v3). `src/grid_substrate.mojo`: a per-grid
`GridSubstrate` (plurality bg, 4-connected components with size/bbox/colour, largest/smallest/
unique/majority registers + bbox anchors, nearest-nonbg multi-source BFS capped at 5, four
first-nonbg ray sweeps) exposing `fetch(view, r, c) -> [rel_bucket, fetched_value]` over the 15
views the Python scan measured — border/wrap conventions copied exactly (rays clamped, anchors
toroidal) because the 22/146 evidence was measured under them. Substrate in the factor_scan sense:
representations a learned read operates over, never a transform. First Mojo-1.0 friction: tuple
returns of `(Int, Int)` don't satisfy Movable in this build — `InlineArray[Int, 2]` instead.
`tests/test_substrate.mojo` (fast tier): exact assertions on hand-built grids — green.

**19:07 — CF-1: the content-fetch memory PROVEN cold, all five classes exact.**
`ContentFetchComposedMemory` (memory_composed.mojo): view slot + 16-entry action table
(key = is_bg × rel_bucket; actions −1/const/KEEP/COPY) written closed-form on the LocalWrite
prefix; `fit_content` (esper_evolution.mojo) = fit_local → write_content; synth ground truth
`CONTENT_TRANSFORMS` (5 families) + `generate_content_task_groups`; proof
`tests/test_content_fetch.mojo` (full tier): **ray_down / recolor_largest / halo_nearest /
anchor_shift / objlocal_mirror each held-out 1.0, each selecting its intended view** (0/5/4/9/13),
ablation control degrades (0.85), strict superset writes no view at 1.0, few-demo n=3 at 1.0.
`./esper fast` green.

Two build discoveries, both measured mid-block (the smoke/proof runs caught them):
1. **The fitted prefix can be WORSE than identity on content demos — and comparing prefixes
   in-sample picks the wrong one.** `write_color`'s count matching writes a garbage V on content
   rules (their colour counts shift incoherently), the gather then drifts on the garbage-premapped
   demos (halo: prediction ≠ input on 117/144 cells; M ≈ random contraction), and a drifted prefix
   still in-sample-beats identity by a few memorized cells (anchor: 196 vs 210). The honest form —
   the plan's own fit_shape_geom precedent — is TWO FULL COLD BRANCHES compared on the FINAL demo
   residual: {fitted prefix + content write} vs {EXACT identity + content write}, keep the lower.
   Constant compute, never a runtime selector; tasks the fitted prefix serves keep it (0-regressions
   preserved by construction).
2. **Two subtle substrate-integrity rules.** The seed gather is only approximately identity (soft
   beta blurs every colour boundary — the fallback must sharpen beta to be a true identity); and the
   fallback branch must stay PURE (no in-sample-gated local table): a polluted prediction snapshot
   corrupts the registers/anchors the content views read on held-out grids (measured: view 9 exact
   in-sample, 0.82 held-out with a local table in the branch; 1.0 without).

**19:33 — CF-2: corpus wiring + slice re-measure — GATE PASSED, the deep floor MOVED.**
Same-shape path swapped to `ContentFetchComposedMemory`/`fit_content` (the exact Rung-A swap shape:
two imports, state_dim, seed+fit, two apply sites; `mem: same` unchanged); full suite green with a
`ray_down` CI bundle solving 1.0 end-to-end. Slice re-measure at the corpus budget (64/1500, GPU),
subset dirs `data_bin/floor146` + `data_bin/solved_same`, before = the v3 lines (same budget):

- **Pre-registered gate: PASS.** Net new solves on the scan-covered ids = **2** (9caf5b84
  0.04→1.00 copy-registers; d037b0a7 0.11→1.00 copy-ray) with **0 regressions** on the 13
  previously-solved same-shape tasks (the composed strict-superset gates held by construction).
- **The context number is the real event: mean held-out on the full 146 deep-floor ids went
  0.188 → 0.625** (133 up / 3 down / 10 flat), mean train-fit 0.337 → 0.764, and 19/146 now sit
  at held-out ≥ 0.9 (0 before). The floor that three audits measured as content-addressed — and
  that the budget-raise experiment proved compute-insensitive — responds to exactly the mechanism
  the audits prescribed. Most of the movement is partial (the 0.99 exact-solve cliff converts only
  2), which matches the scan's prediction: the covered ids' sharp-table LOO ranged 0.92–1.0 and
  the ~72-id near-miss band was always going to be partial-fix territory.
- The 3 down ids (bd4472b8, bd5af378, f76d97a5, all unsolved before and after) are
  identity-fallback branch flips on in-sample ties — noise at the bottom of the band, not
  regressions.

Rung CF is done: built exactly on the audit trail (STOP → scan → literature → gated scan GO →
gated build → measured corpus win), with every mechanism decision forced by a measured failure.

## 2026-07-09

**P1A — soft content-keyed gather, scan pre-gate: STOP (documented negative).**
Added a `softscore-*` family group to `factor_scan.py` (`soft-larger`/`soft-smaller`: nearest
cell whose component is strictly larger/smaller than the centre's, emit its colour) — the faithful
scan proxy for the Mojo gather's `argmax_j[−β·|q_i−x_j|² + w·feat]` read (I rejected an earlier
"reflection-through-a-content-landmark" family: additive-score argmax can't express reflection
about a data point, so it would have been an unfaithful proxy). Substrate is a new
`_bfs_nearest_col` multi-source BFS over centre-relative size classes, added to `precompute`;
existing families don't read the new fields, so committed coverage stays bit-identical (verified:
every prior family line diffs clean against `content_scan_v1.txt`).

Calibration (`scratch/calib_softscore.py`): **20/20 positive controls covered, 0/20 negative
false-covers** — the family expresses its own rule and generalizes at n=3, and depth-2 recolor
doesn't trip it.

Corpus scan (`scratch/soft_scan_v1.txt`): soft-larger covers 11, soft-smaller 9 — but they overlap
almost entirely with `copy-registers`/`copy-ray`/`fetch-ray4`. **Incremental over the hard-content
union = 3** (`25094a63 52364a65 7d1f7ee8`), against the pre-registered bar of **15 ⇒ STOP**. The
soft, ES-moveable score doesn't open new band territory beyond what the sharp CF table already
grazes; the ~72-id near-miss band is not a "score just under the bar" problem, it's a
different-mechanism problem. No Mojo built for Phase 1 (measure-first discipline: a failing scan is
a documented negative, not a reason to build). Phase 2 (rung #6 constructive editor) proceeds
independently — the plan anticipated exactly this fork.

**P2A — rung #6 constructive editor, scan pre-gate: STOP (documented negative).**
Added `EDITOR_FAMILIES` + `scan_editor` to `factor_scan.py`: a faithful offline simulation of the
TRM-style iterated-edit loop — materialize an answer grid `y` (init = input), apply ONE
colour-abstract local relational rule read over the *evolving* grid (`ed-flood` = majority non-bg
4-neighbour; `ed-dir` = directional neighbour; `ed-ray` = first non-bg along a ray), write where
it fires, up to 16 passes to a fixed point. Fit = the same `loo_paired_fetch` KEEP/COPY voting but
**pure-constant votes dropped**, so palette memorization can't fake coverage — the editor earns
coverage only through *relational propagation* (writes become evidence: positions-written !=
positions-read, the one thing a single per-cell pass structurally cannot express). Cheap `_light`
substrate (bg + 4 rays, O(RC), no components/BFS) keeps 16 passes tractable (~10s for all 146).

Calibration (`scratch/calib_editor.py`): **20/20 extend-right, 20/20 flood-down positives covered,
0/20 recolor false-covers** — the proxy genuinely fires on iterative propagation and stays silent
on non-propagation. So the corpus result is a real signal, not an under-powered proxy.

Corpus scan (`scratch/editor_scan_v1.txt`): **EDITOR union = 1, incremental over the entire
per-cell/content new-family union = 0**, against the pre-registered bar of **15 ⇒ STOP**. The
deep floor does **not** contain a colour-abstract local-propagation class of any size — it is a
content-addressed **SELECTION** problem (copy-registers / copy-ray / copy-nearest, which CF already
expresses), not an iterative **CONSTRUCTION** problem. Committed families stayed bit-identical
(only the appended editor block differs).

**Strategic finding (both branches off Rung CF STOP).** The plan's two proposed forward mechanisms
— soft ES-moveable selection (Phase 1) and iterative construction (Phase 2) — *both* fail their
pre-registered gates (incremental 3 and 0). The evidence relocates the problem: the ~72-id
partial-fix band is not "the score is just under the bar" (soft gather) nor "the output is
constructed elsewhere" (editor) — it is CF's **selection consistency**. The near-miss block is
dominated by `copy-*` families sitting at LOO 0.70–0.90 (just under the 0.90 bar) with net_fix
0.25–0.5: the right *source* is being grazed but the keyed table isn't consistent enough to cross.
The next rung, on this evidence, is **sharpening CF's existing content read** (key granularity /
tie-breaking / LOO consistency on the band), not a new memory family. No Mojo built for either
phase — measure-first held: two rungs' worth of build averted by two ~10-minute scans.

## 2026-07-09 — Direction discussion: rung #6 sharpened; current course runs to the end first

**11:15 — Strategy discussion (user + Claude), recorded for the circle-back.** The user proposed
pivoting hard to rung #6 (persistent slow weights + task-stream), on the argument that missing
problem types should be emergently discovered (meta-learned) rather than chased one expressiveness
rung at a time. The discussion's conclusions, to steer by when we return:

1. **The recent record is not "expressiveness failed"** — Rung CF moved the deep floor 0.188 →
   0.625 (the largest single movement to date); the two same-day STOPs (soft gather +3, editor +0)
   were ~10-minute scans that *averted* builds. What the evidence actually isolates is CF's
   **selection consistency**: the ~72-id partial-fix band is dominated by `copy-*` families at LOO
   0.70–0.90 — right source, inconsistent closed-form table.
2. **The reframe that matters for rung #6: persistence alone won't add solves.** The budget-raise
   experiment proved the corpus compute-insensitive, so "fit speed improves with stream position"
   (the rung's current headline metric) measures the wrong thing; and the M9 lesson says a flat
   Reptile prior across a heterogeneous stream washes out. The value of rung #6 is
   **expressiveness-through-consolidation**: replace the closed-form WRITTEN reads (CF's voted
   action table, LocalWrite's n-gram table, the salience scorers) with small META-LEARNED read/write
   rules whose slow weights persist and improve across the task stream — the principled fix for the
   3-demo regime where closed-form LOO voting starves. This also convergently matches Rung A-build's
   verdict ("52 improved-yet-unsolved need a richer/meta-trained local read → rung #6") and the B4
   discipline (fast adaptation = the memory's own write rule; ES fits only the slow vector) — CF's
   written table is currently the one factor family NOT built that way.
3. **Shaped rung #6, when we take it:** the task stream with persistent slow weights; the CF read
   (later the local-write read) as a meta-learned rule, Reptile-nudged per in-context fit, with the
   Schug mitigation (per-task code × shared templates) designed in from day one. Metrics reordered
   by strength: (a) solves on the exhibit band (~72 ids + 17 chain-partials) improve with stream
   position — the headline; (b) no forgetting on early families; (c) frozen-prior + shuffled-stream
   controls. Framed this way it is not a pivot but the roadmap's own arrow — and it doubles as
   Vision B's first rung (same persistence machinery, later driven by self-generated novelty).
4. **One cheap measure-first probe belongs before the rung:** a day-scale Python check of whether
   the band's LOO inconsistency yields to DETERMINISTIC fixes (tie-breaking, key granularity) in
   the scan harness. Either outcome pays: a cheap win, or the documented negative that becomes
   rung #6's opening evidence.

**User decision: run the current course to the end first** (there are things to understand better
before steering consciously) — starting with the deferred **corpus v4 re-measure**, which books
Rungs C + D + CF into one honest baseline and is the "before" number any stream experiment needs
anyway. Wiring verified before launch: same-shape → ContentFetchComposedMemory/fit_content (CF-2
swap), shape → ShapeGeomColorComposedMemory/fit_shape_color (Rung C), reflect frame as the
growth-gated third cold start in fit_shape_geom (Rung D). Budget 64/1500 (the documented corpus
budget, comparable to v2/v3), GPU backend, both splits.

## 2026-07-09 — Corpus v4 re-measure: train 41/1000 (v3: 22), the same-shape corpus mean at 0.81

**11:19–12:01 — the deferred v4 run** (both splits, budget 64/1500, GPU, 2 workers; eval 344 s,
train 2116 s — the full-corpus measure is now a lunch break, not an overnight). Books Rungs A-build
+ C + D + CF into one honest baseline. One harness stumble first: the initial launch died with the
documented "scored 0 tasks in 0s" signature (eval_parallel.sh needs the venv on PATH — the runner
script now activates it). Wiring verified before launch: same-shape → ContentFetchComposedMemory/
fit_content, shape → ShapeGeomColorComposedMemory/fit_shape_color, reflect frame grow-gated in
fit_shape_geom.

**Headline (solve bar 0.99, per SOLVE_THRESHOLD):**

- **Train: 41/1000 solved (v3: 22)** — mean held-out 0.628. Same-shape (680): mean 0.625 → 0.806,
  solved 13 → 23 (Rung CF's corpus effect at full width — incl. its gate exhibits 9caf5b84,
  d037b0a7, plus b1948b0a and 7 more). Shape (320): solved 9 → 18 (Rungs C + D — the mirror-tiling
  and colour-on-shape families landing as booked), mean 0.239 → 0.249.
- **Eval: 0/120 still** — but mean held-out 0.404 → 0.543; the same-shape slice (81) moved
  0.572 → 0.778 (CF transfers to eval's distribution), while the shape slice (39) sat at
  0.054 → 0.055, re-confirming Rung S: eval's shape slice is dims-never-fit content-extraction,
  untouchable by any shape-write + gather.
- Train movement: 621 up / 48 down; the 0.90–0.99 near-miss band is now **230 train ids** (was ~100
  at v3) + 28 eval — the partial-fix frontier keeps widening faster than the solve cliff converts.

**The two "lost" solves, bisected to their rungs (worktree A/B at ccdd8af → b2780af → 32f8f41;
CPU == GPU on both, so not backend noise):**

- **8597cfd7 (1.00 → 0.00, flipped at Rung C).** The v3 line was **train-fit 0.25, held-out 1.0**
  (gap −0.75): a constant-2×2-output extraction task (content-model-bound class, per Rung S) whose
  4-cell test output the v3 fit hit by luck while explaining the demos at chance. Rung C's fitted-V
  branch explains the demos strictly better (train 0.75) and predicts differently. Losing a
  coin-flip solve to a better in-sample explanation is the honest direction; the task's class was
  never expressible.
- **963e52fc (1.00 → 0.89, flipped at Rung D).** A width-doubling tile (5×7→5×14). The reflect
  frame wins the multi-start in-sample (0.949 vs the toroidal frame's 0.930) but generalizes worse
  (0.89 vs 1.0) — the known multi-start hazard: demo-fitness winner selection on a near-tie picks
  the wrong identity frame. Real, small, and structural (not noise); the honest fix, if the class
  ever matters at scale, is a tie-margin rule measured across the corpus, not per-task staging.

Net verdict: **+21 new solves, −2 diagnosed flips (one of which was luck to begin with)** — the v4
baseline is cleaner than the raw ±: every rung's booked corpus value landed where its synth proof
predicted. v4 result files: `scratch/arc2_{train,eval}_v4.txt`; comparison script
`scratch/v4_compare.py`. This is the "before" number for whatever comes next (the rung #6
circle-back or the CF-read probe).

## 2026-07-09 — CF-read probe: deterministic sharpening STOPs (+1 ≪ 15) → rung #6's opening evidence

**19:45 — the pre-rung-#6 measure-first probe (ROADMAP rung #6, plan-approved).** With the v4
baseline booked, the roadmap gates rung #6 behind one cheap Python check: *does the ~72-id
partial-fix band's LOO inconsistency yield to DETERMINISTIC fixes (tie-breaking / key granularity)
in the scan harness?* A cheap win → sharpen CF's written read in Mojo; a negative → the documented
opening evidence that the closed-form read needs a **meta-learned** replacement (the actual value
of rung #6). Built additively in `tools/factor_scan.py` behind a `--probe` flag (the default board
stays **bit-identical** — verified by diff), so nothing committed moves:

- **(a) Deterministic tie-break** (`loo_paired_fetch_det` + `_det_argmax`): replaces
  `Counter.most_common(1)` (insertion-order ties) at both the base and abstract-action decision
  sites with an explicit precedence (COPY/KEEP beat a memorized constant; lower colour wins among
  constants). Scores the EXACT committed copy-* sources, isolating how much of the band is pure tie
  nondeterminism.
- **(b) Finer keys** (`PROBE_COPY_FAMILIES`): the same copy-* sources with the
  KEEP/COPY-disambiguating `f == centre` bit (+ small portable extras: register-vs-bg, nearest
  distance-band), each kept < `SUB_REL_K = 8` buckets so a winner would port into Mojo's `fetch`
  rel without widening the 16-entry `CONTENTFETCH_KEYS` table.

**Result (`scratch/cfprobe_scan_v1.txt`, over the 146 deep-floor ids):**

- **tie-break-only incremental over the committed union: +0.** The band's inconsistency is *not*
  arbitrary tie resolution — a deterministic vote breaks nothing loose.
- **finer-key incremental: +1** (only `9ddd00f0`, via `copy-nearest`).
- **PROBE union incremental: 1 — GATE CFPROBE (pre-registered ≥15 ⇒ GO): STOP.**

**The calibration surfaced *why* — a clean theoretical finding, not a weak proxy**
(`scratch/calib_cfprobe.py`, GUARD PASS: **0/20 false-covers** on both a pure-recolor and a
random-output/support-starvation negative, for coarse and finer alike). On the KEEP/COPY-split
positive **finer AND coarse both cover 20/20** — because an **`f == centre` key bit is provably
VACUOUS for KEEP vs COPY**: exactly when it fires (fetched == centre) the two actions emit the same
colour, so it can never separate them; when it does not fire, the key is the coarse one. The only
KEEP/COPY-discriminating bits are colour/object-**IDENTITY** keys — and the content scan already
found those "stay near 0" (JOURNAL 2026-07-08: coverage is copy-through semantics, *not* richer
keys). So the STOP is doubly grounded: empirically (+1) and structurally (the abstract-relational
key space has no deterministic granularity left to spend on the band).

**What the probe relocates.** The band is not tie nondeterminism (a-fix +0) and not a missing
colour-abstract key bit (b-fix +1, vacuity proof). It is CF's **selection consistency at 2–3
demos**: `copy-*` families graze the right source at LOO 0.70–0.90 but the closed-form voted table
can't cross 0.90 without a *content-dependent* read that a fixed abstract key can't encode. That is
precisely the case for a **meta-learned read/write rule whose slow weights consolidate across the
task stream** — rung #6's headline. Two rungs' worth of Mojo build again averted by a ~day-scale
Python scan; measure-first held. Files: `tools/factor_scan.py` (`--probe`, additive),
`scratch/calib_cfprobe.py`, `scratch/cfprobe_scan_v1.txt`, baseline `scratch/cfprobe_baseline.txt`.
No `src/` (Mojo) touched — Phase 2 was gated on GO.

### 2026-07-09 20:52 — Rung #6, increment 1: the cross-task meta-read probe — GATE CF6 STOP (0/band, robust)

**Started rung #6** (persistent slow weights + task-stream; the real payload is replacing CF's
closed-form voted read with a *meta-learned* read whose slow weights consolidate across the stream).
Per the measure-first discipline — and the user's decision (probe first; build the stream driver
only after a read GOes) — increment 1 is a **cheap Python probe of the bet rung #6 makes**, not a
Mojo build. The CF-read probe (19:45 above) proved the band doesn't yield to *deterministic*
sharpening and that the only KEEP/COPY-discriminating bits are colour/object-**identity** keys —
which the **per-task** 3-demo vote finds near-0. The untested question: do those identity features
become informative when their weighting is **consolidated ACROSS the band** (one shared "slow" read
vector)? This probe measures exactly that, before any Mojo.

**Mechanism** (`tools/factor_scan.py --meta-probe`, additive; default board **bit-identical** —
`main()` untouched, verified by `git diff`): (1) select the copy-* partial-fix band (best copy-*
at LOO 0.70–0.90, net_fix 0.25–0.50, uncovered); (2) per cell build an **identity feature vector**
the colour-abstract vote discards — rank/symmetry-normalized, *never raw colour* (the #1 false-GO
leak): is_bg, f-present, f==centre, f==bg, centre/fetched freq-rank one-hots, size-rank / largest /
smallest, bbox-pos, nearest-distance band, ray/register agreement — plus the KEEP/COPY/CONST action
label `loo_paired_fetch` votes over; (3) fit **ONE shared multinomial-logistic selection read**
over cells pooled across a *train split* of band tasks (mirrors the Mojo SelfModMemory
`a=softmax(τ·(W·feat+b))` head; W is pure selection, the CONST value stays the per-task fast write);
(4) freeze W, score each held-out band task by per-task LOO (only CONST written from that task's own
demos — Mojo slow-frozen / fast-from-demos), count newly `covered` ids; (5) 4-fold. Pre-registered
**GATE CF6: held-out covered ≥ 15 ⇒ GO** (port to a Mojo `ContentFetchSelfMod` + stream driver);
< 15 ⇒ STOP.

**Result (`scratch/cf6probe_scan_v1.txt`): band = 18, held-out covered = 0/18 — GATE CF6 STOP.**
All four variants agree: PRIMARY (all features) 0, portable-features-only 0, CONST-disabled 0, and
the ES-over-the-same-linear-read guard 0. And it is **robust to the band definition**
(`scratch/cf6probe_sensitivity.py`, one extra scan): widening the band to chain[0.60,0.90)/fix≥0.20
(45 ids), chain[0.50,0.90)/fix≥0.10 (58), and MAX = *any* uncovered copy-* chain<0.90/fix>0 (84 ids,
≈ the ~72 near-miss population) **all still give held-out covered = 0** under both logistic and ES.

**Why this is a true negative, not a broken harness.** The calibration
(`scratch/calib_cf6probe.py`, GUARD PASS) runs the *same* shared-fit / held-out pipeline and
**covers 24/24 held-out** on two synth positives whose generating rule IS a shared cross-task
content read (COPY-iff-largest-component; COPY-iff-most-common-nonbg-colour) — rules the
colour-abstract vote provably can't express — while yielding **0 false-covers** on all three
negatives, including the critical **NO-SHARED-STRUCTURE** control (each task an independent random
rule): there, train action-accuracy is 0.75 (high) yet held-out coverage is 0, proving the k-fold
split *detects* overfitting and the probe cannot hallucinate coverage by memorizing the train split.
So the pipeline demonstrably lights up when shared structure exists — and stays dark on the real
band.

**What it means for rung #6 (the re-scope).** The copy-* band's selection inconsistency is genuinely
**per-task** — there is no flat shared selection rule over identity/content features that separates
KEEP/COPY and generalizes across held-out band tasks. This is the *measured* form of the hazard the
ROADMAP flagged from day one: **a single flat prior across a heterogeneous stream washes out.** So
rung #6's naive framing (persist slow weights + one meta-learned read) will not move the band; the
justified path is **per-family / emergent structure** — the Schug hypernetwork (per-task code ×
shared templates, RESEARCH-NOTES #2) and/or the CMS frequency hierarchy (#5) — not a flat prior.
A multi-day Mojo build (new SelfModMemory + stream driver) averted by a ~day-scale Python probe;
measure-first held again. No `src/` (Mojo) touched — increment 2 was gated on GO. Files:
`tools/factor_scan.py` (`--meta-probe`, additive), `scratch/calib_cf6probe.py`,
`scratch/cf6probe_scan_v1.txt`, `scratch/cf6probe_sensitivity.py` + `.txt`.

### 2026-07-10 09:09 — Rung #6, increment 2: the per-family (mixture) read probe — GATE FAM STOP, and the *reason* is EXPRESSIVITY (ceiling 2/84)

CF6 (increment 1) showed a *flat* shared read covers 0 band ids → "a single prior across the
heterogeneous band washes out." The ROADMAP's prescribed fix for that hazard is **per-family
structure** (Schug: per-task code × shared templates). Before building that hypernetwork (a large
Mojo lift), the measure-first discipline says probe the bet: **does the band cluster into families
each with an internally-consistent read?** Increment 2 is that probe (`tools/factor_scan.py
--family-probe`, additive; default board bit-identical, `main()` untouched), and crucially it also
measures a **per-task expressivity ceiling** so a STOP names the right next lever.

**Mechanism** (reuses the CF6 machinery): (1) take the MAX partial-fix band — any uncovered copy-*
with chain<0.90, fix>0 (**84 ids**, ≈ the near-miss population); (2) **flat baseline** — one shared
read, leave-one-task-out; (3) **per-task ceiling** — fit each task's read on its OWN cells and score
*in-sample* (can this read class even represent the task's KEEP/COPY/CONST, no transfer demanded?);
(4) **clustered** — group tasks by winning source family AND by k-means (k=2..5) on a per-task
class-conditional feature signature, then within-cluster leave-one-task-out fit + held-out score.

**Result (`scratch/famprobe_scan_v1.txt`, band = 84):**
- flat baseline: **0/84** (reproduces CF6 on this population).
- clustered by source family: **0/84**; clustered by k-means (all k): **0/84**.
- **per-task expressivity ceiling: 2/84.** — the load-bearing number.
- **GATE FAM (best clustered ≥ 15 ⇒ GO): STOP** (best = 0).

**The ceiling relocates the whole problem.** Clustering/consolidation/persistence are all about
*sharing and transfer across tasks* — but the ceiling says the per-cell linear read over these ~30
identity/relational features (is_bg, f==centre, colour freq-ranks, size-rank/largest/smallest,
bbox-pos, nearest-distance, ray/register agreement) can express only **2 of 84** band tasks *even
fitting each on its own cells with no generalization asked*. You cannot cluster or hypernetwork your
way to expressing a decision that isn't expressible per-task in the first place — the ceiling caps
every downstream consolidation variant at ~2. So the band's wall is **expressivity, not
persistence/structure**: rung #6's entire family of levers (flat prior, meta-learned read,
per-family hypernetwork) is the wrong tool for this population.

**Guarded, not a broken metric.** The ceiling was validated on synth: **16/20** on a known-expressible
rule (COPY-iff-largest-component) vs **0/20** on pure noise — the real band's 2/84 sits with noise,
not with an expressible rule. The clustering path was validated by `scratch/calib_famprobe.py`
(GUARD PASS): on a synth MIXTURE of two directly-opposing sub-rules (COPY-iff-largest vs
COPY-iff-NOT-largest) k-means clustering **recovers 21/24 held-out and beats the flat read**, while
per-task-random / recolor / support-starvation bands all give **0** clustered — clustering covers a
real mixture yet cannot fabricate coverage.

**Where this leaves rung #6.** Three increments of ~day-scale probes (CF-read deterministic,
CF6 cross-task, FAM per-family) have now converged on one verdict for the copy-* band: it is a
**content-addressed CONSTRUCTION** problem (output written where the per-cell evidence isn't — the
deep-floor's dominant property all along; the earlier editor probe's "output constructed elsewhere"),
and **no read/consolidation mechanism over per-cell identity features can express it** (ceiling 2/84).
Rung #6 (persistence + meta-learned read + per-family structure) does **not** address this wall and
should not be built *for the band*; its consolidation value, if any, belongs to a *different*
population that is per-task expressible. The band's justified next lever is **expressivity** — a
richer, non-per-cell read (the constructive/iterated-edit direction, or a content substrate that
carries the construction), which is a separate rung. A large hypernetwork Mojo build averted by the
probe that measured the ceiling. No `src/` touched (increment 3 was gated on GO). Files:
`tools/factor_scan.py` (`--family-probe`, additive), `scratch/calib_famprobe.py`,
`scratch/famprobe_scan_v1.txt`.

### 2026-07-10 10:35 — Vision B study round: the literature says the open-ended path is ES-native

Deliberate pause from the Vision A rung ladder (user-directed) to give **Vision B** its first real
shape. A literature pass over the three seeded areas — unsupervised skill discovery / empowerment
(DADS), UED / open-endedness (POET), world models (Dreamer) — plus the adjacent ES/QD territory,
written up as the new top section of `RESEARCH-NOTES.md` (2026-07-10), with per-finding Esper
mappings and a five-rung POC ladder.

The headline discoveries, in order of how much they de-risk Vision B for *this* codebase:

1. **Skill discovery does not need backprop.** Chalumeau et al. (ICLR 2023 spotlight) show
   quality-diversity neuroevolution equals or beats DIAYN/DADS-style RL at skill discovery across
   8 algorithms — so the MI-discriminator apparatus is optional, and the QD route is literally our
   ES with a different scalar (NS-ES: novelty from an archive-kNN replaces reward in the ES update;
   workspaces/SIMD untouched). POET itself ran on OpenAI-ES — the founding open-endedness loop was
   already derivative-free.
2. **In a tiny discrete world, the deepest intrinsic signals are closed-form or free.** Empowerment
   is exactly computable by Blahut–Arimoto (no learned estimator), and Oudeyer/Schmidhuber
   learning-progress — the noise-proof intrinsic reward — is the *slope of our ES fitness curve*,
   which the engine already computes every iteration.
3. **The definition of the metric exists.** Hughes et al. (ICML 2024): open-ended = novel +
   learnable w.r.t. an observer — the same pair as compression progress. And the Darwin Gödel
   Machine's metric-gaming failure (it falsified its own test results) confirms our value #2: the
   outer metric must be uncheatable, i.e. Vision B is scored in Vision A's currency — held-out
   few-shot transfer using the discovered repertoire (the convergence hypothesis made measurable).
4. **The POC world can be ours.** Craftax/XLand-MiniGrid's lesson is "symbolic-tiny + composable
   depth"; a pure-Mojo avatar-on-`ArcGrid` micro-world (≤16×16, ~6 actions, 2–3 parameterizable
   CA-flavoured rules, no reward channel) reuses the substrate, memory families, and GPU-batched ES
   unchanged, and its rule parameters are the later UED (ACCEL-style curation) mutation surface.

Also satisfying: rung #6's persistence/consolidation machinery — which three probes proved was the
wrong lever for the copy-* band — has a natural home here as the MAP-Elites repertoire archive
(B-POC-2), exactly as ROADMAP's "tabled first rung of Vision B" note anticipated. ROADMAP's Vision B
bullet updated from bare placeholder to point at the study + POC ladder. No `src/` touched — this
was a research round; POC design/implementation is a separate decision.

### 2026-07-10 12:35 — B-POC-1 lands: reward-free novelty search works on the substrate (with two honest findings)

Vision B's first implemented rung, built to the approved plan: `src/sandbox.mojo` (a deterministic
16×16 gridworld with NO reward channel — avatar, move×4/paint/cycle-brush, one parameterizable
gravity rule whose `grav_dir`/`grav_rate` sit in `SandboxTask` as the future UED surface; the
294-param patch+compass→tanh(8)→6-logit argmax policy; the 18-dim block-occupancy BC; the
Go-Explore cell key + `CellSet`; `SandboxDomain`/`SandboxPolicyMemory` conformances) and
`src/novelty_es.mojo` (`NoveltyArchive` with re-entrant SIMD kNN novelty, and `ns_es_run`, the
Conti-style NS-ES meta-population driver copied from the `meta_fit_selfmod` skeleton).
`tests/test_novelty_coverage.mojo` is the proof; `./esper fast` runs it (untagged, ~8s).

**The calibration story (the interesting part).** The first run FAILED its own gate: NS-ES
2,519 cells vs. random-policy 2,177 at an equal 13k-rollout budget — 1.16×, nowhere near the
planned 2×. Diagnosis via an instrumented scratch driver: the ES was barely moving — the selected
center drifted |Δw| ≈ 1.5 over a whole run against an init norm of ~8.6, because raw antithetic
novelty differences live at the BC-distance scale (~0.1) and shrink as the archive densifies, so
the plan's "no annealing" call was necessary but not sufficient. Fix: **unit-std fitness shaping**
of the antithetic coefficients (OpenAI-ES practice; step = α/N·Σ(coeff/sd)·ε), which makes the
step size scale-free. A second finding from the same sweep: at equal total rollouts, **more
iterations beat more samples** (400×16 ≫ 200×32) — every iteration adds an archive entry and
re-aims the search; novelty needs direction more than gradient precision. Final config
K=5, iters=400, N=16, α=0.2, σ=0.4.

**Locked numbers** (seed 0, bit-reproducible across runs — verified by diffing two full runs):
NS-ES **10,578** distinct cells / **1,372** distinct end-states vs. random-policy **2,166 / 461**
at the same 13,205-rollout budget → **4.88× / 2.98×**. Gates locked at 3×/2× with an absolute
floor of 1,000 cells (headroom ~60/50%). Runtime ~8s total.

**The honest caveat, printed by the test and booked here:** a uniform random-ACTION reference —
a stochastic controller, not expressible by our deterministic policy class — covers MORE raw
per-tick cells than anything else (15,068) and 3,392 end-states. In an open world, entropy is a
formidable visitation baseline; Go-Explore's superiority story lives in hard-exploration worlds,
not this one. The gated claim is therefore deliberately within-class: directed search vs.
undirected search over the SAME deterministic policy space, on visitation AND on distinct
end-states — end-states being what the novelty objective actually optimizes and the repertoire
currency B-POC-2 (MAP-Elites) will consume. Also recorded in the RESEARCH-NOTES addendum: the
"intrinsic fitness through the Domain seam" phrasing resolved into trajectory-Example-through-
`Memory.apply` (proven, B-POC-4's path) + driver-hosted novelty (like `meta_fit_selfmod`'s
meta-fitness); and `ns_es_run` allocates its stripes inline once per run, the `meta_fit_selfmod`
precedent, rather than a separate workspace struct.

**2026-07-10 14:02 — B-POC-2: the search becomes a skill library (`src/map_elites.mojo`, `test_repertoire`).**
Vision B rung 2, ~2 hours after rung 1, reusing everything it built. The design question was what
"MAP-Elites" means with no reward: **bin = the end-state Go-Explore cell** (so the repertoire is
directly denominated in B-POC-1's end-state currency), and **within-bin quality = directness** —
the earliest tick from which the trajectory sits in its final cell and never leaves (`settle_tick`,
computed from the per-tick cells log; goal-free, and a directer skill is better reuse currency for
B-POC-4). User locked three choices up front: compare BOTH variation arms against the NS-ES
baseline at equal rollout budget; directness as quality; exact empowerment deferred (booked as a
possible B-POC-2.5).

Results (seed 0, all three arms at exactly 13,205 rollouts, ~4 s total, bit-identical across
runs): **the ES-emitter arm stores 4,317 distinct replayable elites vs. the 1,372 end-states NS-ES
ever touched (3.15×; gate locked at 2× + a 2,000 floor); the pure-mutation arm stores 1,716.**
100 % of stored elites re-reach exactly their bin on replay (the map is real, not an accounting
artifact — this gate would catch any stripe-aliasing bug where stored weights ≠ the weights that
earned the bin). Refinement is real too: 1,135 quality-improving replacements, mean settle
57.0 → 55.3 (mutation arm: 817, 49.1 → 44.2).

Two findings worth the ink:
- **The emitter wants a step size 4× larger than NS-ES's.** The calibration sweep peaked at
  α = 0.8 for the emitter (3,367→4,317 depending on RNG stream position) vs. the α = 0.2 that
  B-POC-1 calibrated for NS-ES, and the asymmetry makes sense: NS-ES's product is its *centers*,
  so it wants measured steps up the novelty landscape; the emitter's product is the *map*, so a
  step that overshoots the novelty peak still lands somewhere — and landing somewhere NEW is the
  whole game. Conversely the mutation arm wants σ_mut = 0.2, HALF of the ES probe σ = 0.4: a
  mutation is a whole move (the child IS the perturbation), while an ES probe only measures a
  direction and must flip argmax actions to register at all.
- **Harvesting is most of the emitter's win.** The emitter arm's coverage (32,665 cells) triples
  the NS-ES arm's despite running the identical ES skeleton — because every antithetic probe, not
  just the center, deposits into the map. The probes were always reaching those states; B-POC-1
  simply threw them away. This required widening the perturbation stripe to N×2×`POLICY_DIM`
  (ns_es_run reuses one stripe for both antithetic signs; harvesting needs the exact weights that
  earned each BC still live at insert time).

Deliberately not built: map serialization (B-POC-4's seam — noted in ARCHITECTURE), curiosity-
weighted parent selection (uniform is canonical v1; the known upgrade if arm A saturates), and
colour in the BC/cell key (unchanged from B-POC-1 so the arms stay comparable across rungs).

**2026-07-10 14:28 — B-POC-2.5: exact empowerment, the archive-free second signal (`src/empowerment.mojo`, `test_empowerment`).**
The deferred half of rung 2. The pleasant design collapse: the sandbox is fully deterministic, so
the Blahut–Arimoto iteration the notes anticipated is unnecessary — a deterministic channel's
capacity is exactly log₂ of its distinct-output count, so **exact n-step empowerment = log₂(#states
reachable in n steps)**, computed by an iterative DFS over all 6ⁿ action sequences with a
full-state FNV hash into a per-sample seen-set. No learned parts, no approximation, and — in
deliberate contrast to novelty — no archive: the signal is stateless and stationary. Horizon
locked at n = 4 (0.7 ms/eval; 6⁵ leaves would overflow the seen-set's 50 % load). One additive
sandbox change: `sandbox_rollout_state` exposes the final avatar r/c/brush (the final grid was
already in the caller's scratch); `sandbox_rollout` now delegates to it, behaviour-identical.

User-locked design: budget stays denominated in ROLLOUTS with the enumeration ticks printed as an
uncharged caveat (19.9 M hidden ticks vs 845 k charged — ~24×); gate = beat the equal-budget
random-policy repertoire; the novelty head-to-head is reported ungated whichever way it goes.

Results (seed 0, 13,205 rollouts/arm, ~6 s, bit-identical double-run): **empowerment emitter
1,513 elites vs random 447 = 3.38× (gate 2×), 100 % replay.** The findings the rung existed to
surface, all booked ungated:
- **Empowerment concentrates; restarts make it explore.** At B-POC-2's reseed interval (25) the
  emitter parks in one high-optionality region and stores 382 elites — BELOW random's 448. The
  calibration lever was reseed = 5 + σ = 0.8: frequent uniform-elite restarts with wide probes
  turn a concentrating signal into an exploration engine (382 → 1,513). Novelty never needed
  this because its objective moves away from wherever it has been by construction.
- **At this RNG stream the two signals tie** (1,513 vs 1,607) — and that exposed real variance in
  the novelty emitter itself: the same constants drew 4,317 in `test_repertoire`'s stream
  position. The emitter family's repertoire is high-variance across streams; per-stream gates
  hold because each test is deterministic, but cross-rung comparisons should quote the spread,
  not a single draw.
- **The paint action flattens the optionality landscape.** Mean elite empowerment is ≈ equal
  across the empowerment, novelty, and random maps (7.94 / 7.98 / 7.94 bits at n = 4): painting
  gives every state combinatorial options, so seeking optionality buys almost nothing here. In a
  world with genuinely constraining states (traps, walls that close) the signal should
  differentiate — worth revisiting when B-POC-5 mutates the world rules.

Signal sanity is gated directly with no search: corner empowerment (7.62 bits) < open-field
(8.27) < n·log₂6 (10.34). Wall-clock note: n = 4 enumeration is 0.7 ms/eval and the whole
three-arm test runs in ~6 s — exhaustive exact empowerment is entirely affordable at this world
size, vindicating the "small symbolic world on commodity hardware" bet (value #5).

**2026-07-10 20:16 — B-POC-3: a world model + learning progress, and three wrong LP allocators (`src/world_model.mojo`, `test_world_model`).**
Vision B rung 3, the densest calibration story so far. The seam worked exactly as designed: a
transition is an `ExamplePair[SandboxState]` of a new `TransitionDomain`, so the UNCHANGED generic
ES core (`fitness`/`fit_operator`/`ESWorkspace`) fits the world model with zero new learning
machinery — the strongest reuse proof yet for the Domain/Memory abstraction.

**The architecture had to be found by measurement.** Three heads, each diagnosed by a per-event
breakdown (departure / arrival / paint) of held-out changed-cell accuracy:
- Squash head (MLPMemory-style): plateaus at 0.46 changed — and identically from a zero seed or a
  colour-LUT basis seed, so colour separation was NOT the wall.
- Residual head (out = centre + bounded delta): identity exact at zero weights; learns DEPARTURES
  perfectly (113/113 — a saturating gate) but arrivals 0/115: a tanh value head cannot emit an
  exact graded COPY of a neighbour's colour. This is the Rung CF expressivity lesson reproduced in
  a 400-parameter miniature.
- **Selector head (landed): softmax over value sources** {9 patch cells, brush, empty} — the
  dynamics are selections ("take the colour from above"), and selection is ES-learnable where
  graded synthesis is not: 0.625 held-out changed-cell (identity = 0, chance ≈ 0.1), overall 0.991.
  Two landscape traps en route: the all-zero seed is a TOTAL saddle for a two-layer selector (hid
  = 0 kills W2's first-order signal AND W1 routes through W2 = 0 — the ES measurably never moved;
  fixed by a deterministic pseudo-random W1/b1 seed), and WM_HID = 16 froze outright where 12
  climbs (bigger ≠ better for ES landscapes). Honest residual: the PAINT event (rare agent-writes,
  ~17 instances/batch) is never learned; paint-biased collection made everything WORSE (busier
  grids harden the gravity patches faster than they soften paint) — booked as a limitation, the
  fix is curriculum not data flooding. Also booked: the fit is schedule-sensitive (a narrow
  0.1-anneal never learns events; wide 0.3 → fine staging is load-bearing), and avatar-position
  INDICATOR features (at-cell / directly-above) were needed — exact-position conditions cannot be
  carved from graded offset ramps.

**Gate 2 (LP separation) needed two redesigns to be honest.** A cyclic-shift target scramble turned
out LEARNABLE (a shift-by-one target is the state two gravity ticks ahead — the model made real
progress on it). The landed construction is symmetric and airtight: mastered = a copy of the batch
the model was just trained on; unlearnable = another copy of the SAME batch made CONTRADICTORY
(distant pairs share an identical input with two different targets — no function fits both, at any
model state, immune to memorization). Measured: LP(novel gravity) = 22× the max of the other two,
unlearnable raw error 8.7× mastered with LP ≈ 0 — the noisy-TV immunity, gated at 10×/1.5×.

**Gate 3 (LP-guided > uniform) burned through two allocators before the honest one:**
1. Clone-probe LP (fit a clone on a 16-transition probe batch) measures batch MEMORIZABILITY —
   rel-LP ≈ 1.0 forever on any real region.
2. MSE-slope LP on validation batches chased the NOISE: the TV-static region's error floor (~−18)
   is 100× the real regions', so learning the noise's mean is a huge one-off absolute delta — the
   allocator handed every round to static (and, absurdly, still "won" once — a pass for the wrong
   mechanism is a fail).
3. **Windowed CHANGED-CELL SCORE slope (landed):** the discrete exact-match currency is scale-free
   and mean-learning never moves it — static reads ≈ 0 from round one (one rounding blip in round
   2, self-extinguished by round 3: 0.101 → 0.007 → 0.002, visible in the test's printed trace).
   Result: **LP-guided 0.362 vs uniform 0.153 (2.4×)** at an equal 600-transition budget, gate
   locked at +0.08 (measured delta 0.209). Also load-bearing: per-round training samples a bounded
   slice of the CUMULATIVE pool (per-round-only pools gave catastrophic churn — the last round's
   focus overwrote everything).

Test is `# suite-tier: full` (4m04s, bit-deterministic across a double-run). The rung's one-line
moral for RESEARCH-NOTES: *LP is the right signal, but only in the right currency — the ES fitness
slope diagnoses learnability; allocation must use the uncheatable discrete score's slope.*
