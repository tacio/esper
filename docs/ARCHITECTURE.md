# Esper Architecture

The detailed, per-module map of the engine. This is a **companion to `CLAUDE.md`**, which keeps
only a high-level module index — read `CLAUDE.md` first for the spine, the hard constraints, and the
toolchain; this doc is the depth (data structures, the ES core, the memory-trait seams, and every
memory family / driver). Sibling docs: **`ROADMAP.md`** (direction & status), **`JOURNAL.md`** (the
timestamped *why* behind each step), **`NL-summary.md`** (the Nested Learning / HOPE theory), and
**`RESEARCH-NOTES.md`** (external literature mapped onto Esper at decision points).

All of `src/` is flat `-I src` Mojo modules with direct imports — no re-export façade.

## Core execution — `src/hope.mojo`

Core data structures + operator *execution* only (no learning, so no `arc_io`/ES deps):

- `ArcGrid` — owned row-major Float32 grid.
- `HopeArena` — move-only bump allocator (`bump[T](count)` / `alloc_node[T]`).
- `HopeNode` — a **POD** node (raw `slow`/`fast` weight slices into an arena + inline child
  indices), built by `build_node`. Kept POD so it can be `init_pointee_move`'d into the arena.
- The **structured learned operator** (Phase A, now dormant): `OP_DIM` layout = 6-param centered
  affine + 10-entry **normalized** colour LUT (stored `/9` to keep colour at the affine's ~unit
  scale); `seed_identity_operator`, `apply_operator`. The operator is a **bilinear geometry gather**
  where each of the four corner input cells is mapped through the colour LUT (`_color_of`) *before*
  the blend — **colour-then-gather** decouples the colour fit from the geometry's precision. Smooth
  (so the ES has a gradient everywhere) yet exact at integer params.
- The generic demo containers `ExamplePair[E]` / `Task[E]`, with `ArcTaskPair` / `ArcTask` as grid
  `comptime` aliases (the ES core is generic over the Example type, so the containers must carry it —
  Mojo checks generic bodies eagerly).

The old `prim_*` / `apply_primitive` and `update_fast_weights` remain but are **dormant** (off the
operator path). `forward_with_learning` lives in `esper_evolution.mojo` for import-DAG reasons.

## Learning core — `src/esper_evolution.mojo`

All learning, **generic over `[M: Memory]`** (Phase B): `fitness[M]`, `evolve_fast_weights[M]`,
`fit_operator[M]`, `reptile_meta_train[M]`, and `ESWorkspace[M]` are parametric; the SIMD/FMA bodies
are unchanged across memories.

- `fitness[M]` — runs a candidate memory (`M.apply`) on each demo input, scores the Domain metric
  (`M.Dom.distance`) vs the demo output (heavy penalty when output area ≠ input area — same-shape),
  minus an L2 anchor toward the slow prior. Replaced the deleted `evaluate_primitives` memorization
  surrogate.
- `ESWorkspace` — **param-sized** ES vectors + a **grid-sized** `op_output` scratch + a
  per-parameter `scale` (a diagonal **preconditioner**; e.g. the colour group gets `COLOR_SCALE < 1`
  so one step size fits both geometry and colour). Alloc-once / reuse — no hot-loop allocation.
- `evolve_fast_weights` — one antithetic-sampling ES step over the demos: **real Gaussian noise**
  (`randn_float64`), mirrored `grad += (F(w+σ·scale·ε) − F(w−σ·scale·ε))·ε`, then
  `W_fast += alpha/(2Nσ)·scale·grad`.
- `fit_operator` — the **annealed** fit loop (shared `FIT_*` schedule: sigma 0.5→0.01, alpha
  0.1→0.003 over ~4000 iters — explore wide, then settle onto exact integers; the wide `FIT_SIGMA0`
  is what makes transpose's four-param move robust).
- `forward_with_learning` — ties it together: fit `node.fast` to the demos (anchored to
  `node.slow`), then `apply_operator` on the test input.
- **M9 — the second timescale:** `reptile_meta_train` is the outer, low-frequency loop that turns
  `slow` from a fixed identity anchor into a *meta-learned prior* — per task it fits `fast`
  in-context FROM the current prior (reusing `fit_operator`), then Reptile-nudges
  `slow += META_LR·(fast − slow)` (`reptile_update` / `copy_weights` are the weight helpers).
  Crucially the **eval** schedule is a *narrow exploit* fit (`EVAL_SIGMA0=0.12`, `EVAL_ITERS=300`),
  not the wide `FIT_SIGMA0`: a warm start only pays off when you can then afford a cheap local fit
  (a wide sigma washes out the init). Meta-train keeps the wide sigma (it must *discover* from cold).
- **Composed fit drivers:** `meta_fit_selfmod[M: SelfModMemory]` (the self-mod family's meta-fit);
  `fit_geomcolor` (block 5: colour write → pre-map the demos through V once per task → the standard
  `fit_operator[AttnGatherMemory]` on the composed state's 7 attention slots — zero core change, hot
  loop alloc-free); `fit_geomcount` (same shape for the count-content module). Both apply
  **constant-compute budgeting**: iters × `FIT_DEMO_REF`/n_demos, so every task gets the same total
  demo-evaluations (the n=8 proofs are unchanged).
- **Shape seam (Next #1):** `fitness_shape[M: ShapeMemory]` / `fit_shape[M]` are the shape-aware
  siblings (the memory predicts its own output dims; a predicted/true area mismatch is
  heavy-penalized). `fit_shape_geom` is the per-task driver — a **multi-start** fit over the
  canonical identity frames (winner by demo fitness; an honest multi-start, no staging). Each start
  runs DISCOVER (soft, temperature searched) then SETTLE (temperature hard-frozen at
  `SHAPE_BETA_READ` via the settle-variant type, sigma held at `SHAPE_SIGMA_FLOOR` — the sharp
  landscape's staircase step scale, below which ES updates are pure noise — alpha decayed),
  constant-compute budgeted. The starts: **rescaled** (`M=I`) and **periodic** (`M=kI`) always, plus
  (**Rung D**) a **mirror** start added only when the written shape rule GROWS an axis (a
  shrink/same-size task can't be a reflection tiling, so non-growing tasks stay byte-identical
  two-start; budget split by the actual start count). `fit_shape_color` is the Rung-C analogue
  (colour pre-map → `fit_shape_geom` on the prefix, V frozen in the suffix).

## GPU-batched fitness — `src/gpu_es.mojo`

The GPU rungs (G1–G2, journal 2026-07-07). The compute map is stark: ~all engine FLOPs are the
windowed attention-gather forward inside `fitness`/`fitness_shape`, evaluated 2·N·n_demos times per
ES iteration over thousands of iterations, while the searched vector is 7–14 wide. So the GPU seam
is exactly the **fitness boundary**: the noise draw (same serial RNG stream as the CPU path),
antithetic coefficients, gradient reduction and parameter update stay on the CPU; one kernel launch
per iteration scores every (candidate × demo) pair — one thread block per pair, threads striding
output pixels, a fixed-order shared-memory tree reduction for the SSE (no atomics ⇒ a GPU run is
bit-identical to itself; it is NOT bit-identical to a CPU run — reduction order differs, so ES
trajectories diverge; parity is pinned at the fitness level by `test_gpu_parity`, quality at the
usual held-out bars).

- The kernels call the **same `attn_pixel_{plain,toroidal,reflect}` free functions** the CPU
  `Memory.apply` loops use (extracted in rung G0, memory_es.mojo) — the gather math lives once.
- `_fitness_kernel_plain` / `fit_operator_gpu` — the same-shape path (`fit_geomcolor`/`fit_local`/
  `fit_geomcount` route through it). `_fitness_kernel_shape` / `fit_shape_gpu` — the shape path
  (predicted per-demo output dims computed once on the host — the written shape slots are frozen —
  and ZEROED for mismatched demos so the kernel no-ops instead of reading OOB; mode slot dispatches
  toroidal vs reflect in-kernel; the DISCOVER/SETTLE difference is exactly `fill_scale`, carried by
  a `settle` flag). `_assemble_fitness` reproduces the CPU fitness semantics (per-demo negative MSE
  mean / heavy penalty, L2 anchor) once for both.
- **Routing**: the composed fit drivers in esper_evolution.mojo gate on `comptime if
  has_accelerator()` (CPU-only hosts compile zero device code — CI unaffected) with a runtime
  `use_gpu: Bool = True` opt-out plumbed from `arc_solve --cpu` for A/B runs. The CPU path remains
  the generic reference for every memory family; gpu_es is deliberately specialized to the corpus
  path's forwards.
- Buffers are created once per fit and reused (the alloc-once discipline); per-iteration traffic is
  ~KBs (candidates up, partials down). Measured at the corpus budget: a same-shape real-ARC task
  73 s → 4.4 s (~17×); a 3-task probe 5m12s → 17 s; the full-tier shape proofs dropped from minutes
  to ≤ 1 min each.

## I/O + the Domain seam — `src/arc_io.mojo`

- `_read_grid_block` — reads one validated grid-block (16-byte header: two little-endian Int64
  `rows`/`cols`, then the float32 payload) at an offset, advancing it. Shared by `load_arc_grid`
  (single `.bin`) and `load_arc_task` (a `.task` **bundle**: `[n_train][n_test]` then the train/test
  grid-blocks → an `ArcTask`). Header lengths are validated so truncated/malformed input raises.
  (Mojo's `open` accepts only `r/w/rw/a`; `read_bytes()` returns raw bytes.)
- `calculate_fitness` = SIMD negative MSE (the continuous ES signal); `exact_match` = discrete reward
  (fraction equal after `round`).
- **The Domain seam (Phase B):** the `Domain` trait + `GridDomain` live here — a Domain is
  `(associated Example type, distance, score, capacity)`. `GridDomain.Example = ArcGrid` and its
  metrics wrap `calculate_fitness` / `exact_match`. The ES core reaches metrics only through a
  Domain, never ARC directly.

## Memory trait seams — `src/memory.mojo`

Traits only; the families live in the `memory_*` modules below.

- `Memory` — what the ES fits in-context: `param_dim` (replaces `OP_DIM`), `seed`, `fill_scale` (the
  per-memory ES preconditioner), `apply(weights, inp: Self.Dom.Example, dst)`, with an associated
  `comptime Dom: Domain` reached as `M.Dom` (Mojo traits can't take parameters, so the domain is an
  associated type, not `Memory[D]`).
- `SelfModMemory` — the self-write counterpart: `slow_dim` / `state_dim` / `seed_slow` /
  `fill_scale` / `adapt` (writes the fast state from the demos via the memory's own rule) / `apply`.
- `ShapeMemory` — the output-size seam (Next #1): `write` (infers the shape rule closed-form from
  the demo dim-pairs), `out_rows` / `out_cols` (predict output dims), `apply(state, inp, out_rows,
  out_cols, dst)`. A distinct trait so the same-shape core and memories stay untouched.

There is **no runtime memory-selector** — the memory is a compile-time choice, each measured on the
subset it expresses.

## ES-fit forward family — `src/memory_es.mojo`

- `OperatorMemory` — the structured Phase-A affine+LUT operator, **dormant** (subsumed emergently by
  `GeomColorComposedMemory`; kept only as the arc_solve/M8 baseline; owns `COLOR_SCALE`).
- `MLPMemory` (**B1**, the first emergent memory) — per-cell `1→H→1` tanh MLP; learns recolor with
  no hand-coded LUT, output squashed to `[0,9]`.
- `SeqOperatorMemory` / `SeqMLPMemory` (**B2**) — the sequence-domain pair proving the seam is
  domain-generic.
- `AttnGatherMemory` (**B3**, emergent geometry) — a 7-param learned position-attention gather
  (`M`, `t`, temperature `beta=raw²`; integer projections ⇒ exact flip/transpose). The softmax is
  **windowed** — a `(2·ATTN_WINDOW+1)²` window centred on `q`, bit-identical on synth-scale grids and
  ~3–5× cheaper at real-ARC 30×30. `apply_shaped` decouples the query grid (output dims) from the
  source grid.
- `attn_gather_toroidal` (the shape-seam gather) — the output-shaped read with a **toroidal source**
  (wrapped images make tiling's sawtooth expressible), an **extent-relative translation** `trel`
  (absorbs tiling's size-dependent phase), and the query **normalized by the written shape slope**
  (resize-as-identity — content never re-learns the scale the shape rule knows).
- `attn_gather_reflect` (**Rung D**) — the mirror-tiling twin: the SAME centered query read through a
  symmetric triangle **fold** (`_reflect_fold`, period 2·extent) instead of the periodic wrap, so odd
  tiles mirror (`out[R+i]=in[R-1-i]`) — the kaleidoscope tiling the torus can't express. The fold is
  continuous (cleaner ES gradient than the sawtooth) and the same `seed_periodic` is its exact
  solution.

`AttnGatherMemory` itself is untouched by the shape work (same-shape proofs bit-identical).

## Composed memories — `src/memory_composed.mojo`

The emergent composition-pattern memories (a commuting representation + invariant per-factor fits +
forward composition):

- `GeomColorComposedMemory` (**block 5** — the emergent retirement of `OperatorMemory`): a
  count-signature colour table **written closed-form** from the demos (geometry-invariant — counts
  are position-free under permutation geometry) composed with the AttnGather ES run on V-pre-mapped
  demos (**colour-then-gather**; driver `fit_geomcolor`). `fill_scale` zeroes the V group so no ES
  path moves the written table. Few-demo hardened: global-min greedy **injective** assignment with
  identity defaults for unseen colours and identity preference on exact ties (corpus median is 3
  demos).
- `GeomCountComposedMemory` (content×geometry) — the same recipe one level up: a neighbourhood-count
  content rule `out = geom(M(count_P(in)))`, its `(P, M)` written **geometry-invariantly from
  histogram signatures** (count rules commute with the lattice-symmetry geometry class), geometry via
  `fit_geomcount` on content-premapped demos.
- **The shape-seam pair (Next #1):** `ShapeGeomComposedMemory` — layout `[0:7]` attention |
  `[7:9]` trel | `[9:13]` shape rule (`out = round(k·in + b)` per axis, least-squares-written,
  frozen) | `[13]` **frame-mode slot** (`SHAPEGEOM_MODE_OFF`: 0=toroidal, 1=reflect — written by the
  fit driver, frozen in `fill_scale`, dispatched in `apply`). Its gather is `attn_gather_toroidal`
  or (mode 1) `attn_gather_reflect`. `ShapeGeomSettleMemory` is a thin delegating variant whose only
  difference is a frozen temperature slot (`fill_scale` is static per type, so the settle phase is a
  type). A k-fold size change has **three canonical identity frames** — rescaled (`M=I`), periodic
  (`M=kI, trel=(k−1)/2`, via `seed_periodic`), and (**Rung D**) mirror (the periodic seed read
  through the reflect fold, mode 1) — which is why `fit_shape_geom` is a grow-gated multi-start.
- `ShapeGeomColorComposedMemory` (**Rung C** — colour on shape): the third composition application,
  wrapping the shape+geometry state with the fraction-normalized colour write; inherits the reflect
  dispatch and the mirror frame for free via its delegation to `ShapeGeomComposedMemory.apply` /
  `fit_shape_geom`.
- `LocalWriteComposedMemory` (**Rung A**): a per-cell local content override written from the demos
  (table keyed on `(centre colour, #Moore-8 neighbours differing)`), composed on the GeomColor
  gather via `fit_local` (fit the gather, then write on the RESIDUAL), gated so a pure
  geometry/colour task writes no table and stays byte-identical (strict superset).
- `ContentFetchComposedMemory` (**Rung CF** — the written content-keyed gather): the fifth
  closed-form factor, reaching the deep floor's CONTENT-ADDRESSED class (copy/move/draw/extend —
  output written where the input evidence is not). A written **fetch view** (one of the 15
  `GridSubstrate` content-selected reads) + a 16-entry **relational action table**
  (key = `is_bg × rel_bucket`; actions KEEP / COPY-the-fetched-value / constant colour), applied on
  a SNAPSHOT of the LocalWrite prefix's prediction. `fit_content` compares **two full cold
  branches on the final demo residual** — {fitted prefix + content write} vs {EXACT-identity
  prefix (sharp beta) + content write} — because a drifted prefix can in-sample-beat identity while
  starving the content substrate (JOURNAL 2026-07-08). Same strict-superset gate family: no view
  written ⇒ byte-identical to LocalWrite.

## Grid substrate — `src/grid_substrate.mojo`

`GridSubstrate` (**Rung CF**): per-grid content representations the fetch views read — plurality
background, 4-connected components (size/bbox/colour), largest/smallest/unique/majority registers +
bbox anchors, nearest-nonbg BFS (colour + capped distance), four first-nonbg ray sweeps;
`fetch(view, r, c) → [rel_bucket, fetched_value]` over 15 views. Substrate in the factor-scan
sense (representations a learned read operates over, never a transform); border/wrap conventions
mirror `tools/factor_scan.py` exactly (the 22/146 coverage evidence was measured under them).
Computed once per grid at write/apply time — never inside an ES hot loop.

## Self-write families — `src/memory_selfmod.mojo`, `src/memory_selfmod_grid.mojo`

Fast adaptation is the memory's own write rule over the demos (a forward pass, never ES-searched);
the ES meta-learns only the small slow vector.

- `memory_selfmod.mojo` (**B4**, core mechanisms): `RecolorSelfWrite` (fixed-projection checkpoint),
  `RecolorSelfModMemory` (meta-learned associative read; a fresh recolor adapts in ONE pass),
  `DeltaSelfModMemory` (gated delta-rule self-write `S ← (1−α)S + η(v−S·k)k` with self-generated
  key/η/α, sequence domain).
- `memory_selfmod_grid.mojo` (the 2-D grid self-mod memories, ARC-AGI-2 blocks 1–4):
  `GridContextSelfModMemory` (additive centre/neighbour rules via outer-product keys),
  `GridNbhdSelfModMemory` (disjunctive/count class: centre-free Moore-8 histogram key +
  sigmoid-threshold read, whole 2-level rule inferred in-context), `GridCountMapSelfModMemory`
  (arbitrary count→colour maps: meta-learned scoring salience + soft count-bin value table).

## Vision B sandbox — `src/sandbox.mojo`, `src/novelty_es.mojo`, `src/map_elites.mojo`, `src/empowerment.mojo`, `src/world_model.mojo`

The B-POC-1 pair (open-ended rung 1): a reward-free world + the NS-ES intrinsic-fitness driver;
plus the B-POC-2 repertoire (rung 2): the persistent elite-per-cell skill library; plus the
B-POC-2.5 second intrinsic signal (exact empowerment); plus the B-POC-3 world model + learning
progress (rung 3).

- `sandbox.mojo` — a deterministic 16×16 gridworld with **no reward channel**: an avatar with 6
  actions (move×4 / paint / cycle-brush) under ONE parameterizable dynamics rule (gravity;
  `grav_dir`/`grav_rate` live in `SandboxTask` — the future UED mutation surface). Holds the policy
  (egocentric 5×5 patch + 4 compass scalars → tanh(8) → 6 logits, argmax, `POLICY_DIM = 294`), the
  behaviour characterization (16 block-occupancy fractions + avatar pos, `BC_DIM = 18`), the
  Go-Explore-style cell key + `CellSet` (the coverage metric), and the seam conformances:
  `SandboxDomain(Domain)` (Example = `SandboxTask`) and `SandboxPolicyMemory(Memory)` whose
  `apply` = a full rollout writing the trajectory's BC — so the unchanged generic
  `fit_operator` can already fit a policy toward a target end-state (B-POC-4's scoring path).
- `novelty_es.mojo` — `NoveltyArchive` (flat BC store; re-entrant kNN novelty, k=10 mean) +
  `ns_es_run`, the NS-ES meta-population driver (Conti et al. 2018): K agents, shared archive,
  novelty-proportional agent selection, one antithetic ES step per iteration whose scalar fitness is
  the candidate's archive novelty. A bespoke copy of the `meta_fit_selfmod` skeleton (novelty cannot
  flow through the static, target-based `Domain.distance`); deliberate deviations, both calibrated:
  fixed alpha/sigma (the objective is non-stationary) and unit-std **fitness shaping** of the
  antithetic coefficients (raw novelty differences are tiny and shrink as the archive densifies —
  un-shaped steps barely moved the centers).
- `map_elites.mojo` — the **persistent repertoire** (B-POC-2): `EliteMap`, an open-addressing
  Int64→slot map (the `CellSet` pattern) keyed by the END-STATE cell, whose payload is the elite
  policy's weights + BC + settle tick; within-bin quality = **directness** (`settle_tick` — the
  earliest tick from which the trajectory sits in its final cell; strictly-smaller replaces). Two
  variation drivers share the map: `me_mutation_run` (canonical MAP-Elites — uniform elite parent,
  Gaussian child, one rollout, fill-or-improve) and `me_emitter_run` (CMA-ME-flavoured — a single
  novelty-ES emitter on the `ns_es_run` skeleton, re-seeded from a uniform elite every E
  iterations, EVERY rollout harvested into the map; its pert stripe is N×2×`POLICY_DIM` because
  harvesting needs both antithetic weight vectors live at insert time). Budgets are compared in
  rollouts — one insert attempt per rollout in every arm. The map is in-memory by design;
  serialization is B-POC-4's seam.
- `empowerment.mojo` — **exact empowerment** (B-POC-2.5): the sandbox is deterministic, so
  Blahut–Arimoto collapses and n-step empowerment = log₂(#distinct states reachable in n steps) —
  `empowerment` is an iterative DFS over all 6ⁿ action sequences (per-depth grid copy +
  `sandbox_step`, full-state FNV hash into a re-entrant per-sample seen-set), no learned parts, no
  archive, stationary. `emp_es_run` is `me_emitter_run` with the scalar swapped (empowerment of
  the candidate rollout's final state via `sandbox_rollout_state`, which also hands back final
  avatar r/c/brush); it counts its enumeration ticks for the uncharged-cost caveat (budgets stay
  denominated in rollouts).
- `world_model.mojo` — the **world model + learning progress** (B-POC-3). `SandboxState` /
  `TransitionDomain`: a transition is an `ExamplePair[SandboxState]`, so the UNCHANGED generic ES
  core fits the model — no new learning machinery. `WorldModelMemory` is a per-cell local
  predictor with a learned **softmax-selector head** over value sources {9 patch cells, brush,
  empty} (dynamics are selections; a tanh value head provably cannot emit graded colour copies —
  see the module comments for the measured head comparison and the seed-saddle fix).
  `lp_probe` = instantaneous LP as the ES fitness slope (clone, fixed constant-α/σ stage,
  delta). `train_uniform` / `train_lp_guided` = the equal-budget collection arms over regions =
  dynamics contexts (gravity directions + a TV-static region); the guided allocator uses the
  windowed **changed-cell score slope** on held-out per-region validation batches (MSE-slope and
  clone-probe LP both fail as allocators — documented in-code), with per-round training on a
  bounded subsample of the cumulative pool.

## Drivers — `src/main.mojo`, `src/arc_solve.mojo`

- `main.mojo` — end-to-end driver: builds an `OP_DIM` node, seeds slow (prior) and fast (init) to
  identity, learns `flip_h` in-context via `forward_with_learning`, prints result + held-out exact
  match. `mojo run -I src src/main.mojo`.
- `arc_solve.mojo` — the **held-out generalization driver** (replaced `benchmark.mojo`). Takes
  `.task` bundle paths via argv, fits each task's **emergent composed memory** on its train pairs,
  scores the **unseen** test pair(s), and reports per-task held-out + train-fit + train/test gap +
  a trailing `mem: same|shape` marker (appended AFTER the existing fields — `eval_parallel.sh` reads
  held-out positionally), then the aggregate solve rate. **Dispatch:** any train pair whose DIMS
  differ routes to the shape memory / `fit_shape_geom` (predicts output dims; a predicted/true dims
  mismatch scores that pair 0, never applied — no OOB); all-same-dims keeps the byte-identical
  same-shape path, where a test pair whose output area ≠ input area honestly scores 0.
  Driver-level routing on a closed-form observable of the demos, **not** a memory-selector. Raises on
  0 solved (a CI regression signal for the synth bundles) **unless** the first arg is `--report`
  (honest real-ARC eval mode, where 0% is legitimate). `--fit N ITERS` overrides the ES budget
  (default = the full proven `FIT_*`); real-corpus runs use a smaller **documented** budget (quoted
  with the number) because full-budget 30×30 fits are compute-prohibitive. Seeds the RNG **per task**
  (`SOLVE_SEED`, inside `solve_task`) so the benchmark number is reproducible and invariant to task
  ordering / sharding. Uncheatable by memorization.
- `eval_parallel.sh` (repo root) — shards a `.task` directory round-robin across `nproc` worker
  processes (optional trailing `fit_N fit_iters` forwards the corpus budget), one driver invocation
  per shard, then re-aggregates the per-task lines — ~`nproc`× faster, identical numbers
  (process-level sharding, since the `ESWorkspace` / global RNG aren't thread-safe for in-process
  parallelism).

## Offline tools — `tools/`

Python is isolated strictly to this offline data-compilation toolchain — never on the runtime path.

- `arc_compiler.py` — the one sanctioned Python component. `_write_grid`/`_save_grid` (single grid
  `.bin`) and `_save_task` (a `.task` bundle) are the single source of the on-disk formats;
  `compile_arc_json` converts ARC JSON to per-grid `.bin`s. `compile_task_to_bundle` /
  `compile_arc_dir` + a `__main__` CLI (`python tools/arc_compiler.py <json_dir> <out_dir>`)
  batch-ingest a real **ARC-AGI 2** corpus into `{task_id}.task` bundles (the M8 path). The corpus
  is **not** vendored (`data_bin/` and the `arg-agi-2-data` symlink are gitignored).
- `synth_tasks.py` — offline deterministic generator: `generate_tasks` (single-grid pairs),
  `generate_task_groups` (ARC-shaped same-shape `.task` bundles), and `generate_shape_task_groups`
  (shape-changing bundles, input size varied across a task's demos so the shape rule is
  identifiable). It is the *ground-truth generator* the engine must rediscover — the symbolic
  transforms (flip/transpose/recolor/shift, the shape families, mirror tilings) live here, **never**
  in the engine.

## Test roster — `tests/`

Each test is a held-out generalization proof (or a mechanism/round-trip check); tests `raise
Error(...)` on assertion failure rather than using a framework — follow that pattern for new tests.
All import the real `src` modules via `-I src`. The ES-based tests anneal a few thousand iters; the
`# suite-tier: full`-tagged ones are the heavy milestone proofs (see `CLAUDE.md` → Testing for the
fast/full tiers).

- **Foundations:** `test_arena`, `test_operator` (hand-set weights reproduce the transforms
  exactly), `test_fitness`, `test_demo_fitness` (keystone: ES fits `flip_h` and generalizes),
  `test_forward_learning` (end-to-end node path, fit-once/generalize-many), `test_task_loader`
  (bundle round-trip), `test_io`, `test_shape` (same-shape fits; shape-change penalized, no crash),
  `test_generalization` (the whole expressible subset ≥0.95 held-out).
- **Meta / emergent:** `test_meta_prior` (**M9**: a Reptile-meta-learned `slow` prior fits a fresh
  flip_h task ≥0.95 at a narrow eval budget where a cold prior scores ~0), `test_mlp_memory`
  (**B1**: emergent `MLPMemory` learns recolor ≥0.95, no LUT).
- **Composition:** `test_composed_generalization` (**block 5**, the OperatorMemory retirement proof:
  matches the operator's whole subset ≥0.95 cold AND solves a composed flip∘recolor no single memory
  expresses; geometry-only ablation must fail), `test_composed_content` (**content×geometry**:
  `geom∘countmap` 1.0 cold; correspondence-statistic and content-ablation controls must fail),
  `test_few_demo` (**few-demo robustness** at the corpus-median 3 demos: aggregate ≥0.85 + an n=8
  regression guard), `test_local_write` (**Rung A**: the local content-override `LocalWriteComposedMemory`
  solves {outline, fill_enclosed} ≥0.95 cold; controls — GeomColor-only ablation fails outline (the
  local write is load-bearing), a pure recolor writes no table (strict superset); few-demo n=3),
  `test_substrate` (fast tier: exact assertions on `GridSubstrate` components/registers/BFS/rays/
  fetch over hand-built grids), `test_content_fetch` (**Rung CF**: the five content-addressed
  classes {ray_down, recolor_largest, halo_nearest, anchor_shift, objlocal_mirror} each ≥0.95
  held-out cold with fresh colours per demo; controls — LocalWrite-prefix-only ablation fails
  ray_down, a pure recolor writes no view (strict superset); few-demo n=3).
- **Shape seam:** `test_shape_change` (**the output-size seam**: {crop1, flip_h_crop1, subsample2,
  upscale2, tile2} each ≥0.95 held-out at a FRESH size, per-task cold; controls: no-shape-write → 0,
  plain non-toroidal gather fails tile2), `test_mirror_tiling` (**Rung D**: mirror_tile{2,3}
  ≥0.95 cold via the reflect gather as a third `fit_shape_geom` frame, reflect frame winning; k=3
  grow regression tile3/upscale3; controls: mode ablation forcing the torus collapses the mirror
  read, strict superset plain tile2 stays mode 0), `test_shape_color` (**Rung C**: the
  `ShapeGeomColorComposedMemory` fraction colour write + shape gather solves the recolored shape
  families {recolor_crop1, recolor_subsample2, recolor_upscale2, recolor_tile2}; controls — V-forced-identity
  ablation fails the recolored families, and a pure-shape task reproduces `test_shape_change`'s bar
  (strict superset); few-demo n=3).
- **Self-mod family (B2–B4):** `test_seq_domain`, `test_attn_memory`, `test_selfmod_memory`,
  `test_delta_selfmod`, `test_grid_context_selfmod`, `test_grid_nbhd_selfmod`,
  `test_grid_countmap_selfmod`.
- **Vision B:** `test_novelty_coverage` (**B-POC-1**: with zero hand-coded goals, the NS-ES
  meta-population covers ≥3× the distinct cells and ≥2× the distinct end-states of an equal-budget
  random-policy baseline in the sandbox; inline gravity unit check; fully deterministic under
  `seed(0)`); `test_repertoire` (**B-POC-2**: the best MAP-Elites arm stores ≥2× as many distinct
  replayable skills as the NS-ES baseline ever touched end-states at the same budget, with a 100 %
  elite replay gate and a refinement gate — replacements happened and lowered the mean settle
  tick; fully deterministic under `seed(0)`); `test_empowerment` (**B-POC-2.5**: exact-empowerment
  sanity — corner < open field, within n·log₂6 — and an empowerment-only emitter builds ≥2× the
  equal-budget random-policy repertoire, 100 % replay; the empowerment-vs-novelty head-to-head is
  printed ungated; fully deterministic under `seed(0)`); `test_world_model` (**B-POC-3**,
  suite-tier full ~4 min: the world model reaches ≥0.4 held-out changed-cell accuracy where
  identity scores 0; LP separates novel ≥10× from mastered AND contradiction-scrambled while the
  scrambled region keeps ≥1.5× the mastered raw error; LP-guided collection beats uniform by
  ≥0.08 held-out changed-cell at equal transition budget; fully deterministic under `seed(0)`).

Phase-A expressible subset = {identity, flip_h, flip_v, transpose, recolor}; `shift` deferred (the
affine zero-fills, synth `_shift` wraps).
