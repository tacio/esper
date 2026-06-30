# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine for logical deduction and pattern recognition (e.g. ARC-AGI). It implements a "Nested Learning" architecture (HOPE) with three layers:

1. **Slow Weights** — base routing knowledge, updated rarely (the L2 anchor for the fast weights).
2. **Fast Weights (Continuum Memory System)** — memory dynamically updated during the forward pass via a derivative-free Evolution Strategy (ES), not backprop.
3. **HopeArena** — a strictly move-only, zero-overhead contiguous bump-pointer allocator that hosts POD `HopeNode`s and their weight slices.

## Direction (active roadmap)

The foundational learning path now works (milestones M1–M4 done): the memorization surrogate is gone and the engine genuinely learns a transformation from demonstrations and generalizes to held-out inputs (e.g. it learns `flip_h` to ~0.99 held-out exact match). The spine decision — which all future work must respect:

- **Fast weights parameterize a *learned* grid→grid operator**, fit in-context by the ES on each task's demonstration pairs; the slow weights are the meta-learned prior (the existing L2 anchor). This unifies the two conflicting "fast weights" notions (op-params vs. one-weight-per-pixel) into a single, grid-size-independent operator parameter vector (`OP_DIM`).
- **Never hand the engine a symbolic DSL.** Geometric/color transforms (flip/transpose/recolor) must *emerge* as fitted operator parameters; they live only in the offline `synth_tasks.py` generator as ground truth to rediscover. The north star is fully emergent (a general parametric memory); structured priors are removable "training wheels".
- **Metric = held-out generalization** (fit on a task's train pairs, score the unseen test pair — uncheatable by memorization), not fitting a known target. The old `benchmark.mojo` memorization harness has been removed; a held-out generalization driver (`src/arc_solve.mojo`) lands at M7.

Phase 2 then strips the structural priors toward an emergent self-modifying memory and lifts the grid/task abstraction to a domain interface so the same ES / two-timescale core serves other reasoning problems. (Full milestone plan: `~/.claude/plans/let-s-create-a-roadmap-ticklish-adleman.md`, user-local.)

The theory behind the architecture (the "Nested Learning"/HOPE paper, <https://abehrouz.github.io/files/NL.pdf>) is distilled in **`docs/NL-summary.md`** — read that instead of fetching the 52-page PDF. It ends with a table mapping each paper concept onto Esper's `slow`/`fast` weights, ES optimizer, and CMS direction.

## Journal (keep it updated)

Maintain a running, timestamped development narrative in **`docs/JOURNAL.md`**. Append an entry (with a `HH:MM` local timestamp) at every step of real progress, every unexpected discovery, and every blocker — including the diagnosis and the fix. Newest entries at the bottom. The goal is a complete story of *why* the code looks the way it does (e.g. why the operator uses bilinear sampling, why the ES anneals sigma), not just *what* changed — git already records the what. Update it as you work, not only at the end.

## Toolchain (important)

Core/runtime code targets **Mojo 1.0.0b2** (1.0 beta). This is a hard pin: the 1.0 line removed `fn` (use `def`), renamed `alias`→`comptime`, requires `std.`-qualified stdlib imports, removed the stdlib `Tensor` type, and changed the `UnsafePointer`/argument conventions. Code written for older nightlies will not compile. Mojo is installed from **PyPI** (`uv pip install "mojo==1.0.0b2" --prerelease allow`), not the Modular wheel index.

Key 1.0 idioms used throughout (match these when extending):
- `def` everywhere; add `raises` to any `def` that can raise. `comptime X = ...` for compile-time constants.
- Imports are `std.`-qualified: `std.memory` (`alloc`, `memset_zero`, `memcpy(dest=, src=, count=)`, `UnsafePointer`), `std.sys` (`size_of`, `simd_width_of`, `argv`), `std.math` (`fma`, `round`), `std.random` (`randn_float64`, `seed`), `std.collections` (`List`, `InlineArray`).
- SIMD comparison/rounding gotchas: `a == b` on two SIMD vectors returns a **scalar `Bool`** (whole-vector equality) — for an elementwise mask use `a.eq(b)`, then `.cast[DType.float32]().reduce_add()` to count hits. There is **no SIMD `.round()` method**; use the free `round()` from `std.math` (works on both SIMD vectors and scalars). See `exact_match` in `arc_io.mojo`.
- Raw memory: `var p = alloc[T](count)` / `p.free()`. Pointer fields need an explicit origin: `UnsafePointer[Float32, MutAnyOrigin]`. Arithmetic is `(p + n).bitcast[T]()`; SIMD is `p.load[width=nelts](i)` / `p.store[width=nelts](i, v)`.
- Lifecycle: `def __init__(out self, ...)`, `def __del__(deinit self)`. Do **not** write `__moveinit__` — derive `(Movable)` / `(Copyable, Movable)`; moves consume the source so a moved-from value's `__del__` never runs.
- `UnsafePointer` is non-null by design — do **not** guard `__del__` with `if self.data:`; free unconditionally.

## Hard constraints (do not violate)

- **Language: Pure Mojo** for all core/runtime code. Python is isolated strictly to the offline data-compilation toolchain (`src/arc_compiler.py`) — never introduce Python fallbacks or dynamic bindings into the execution/inference path.
- **Zero external ML libraries.** No PyTorch, no TensorFlow, no autodiff frameworks. Learning is done via Evolution Strategies, not gradient descent/backprop.
- **No dynamic allocation in hot loops.** Fast-weight updates must reuse pre-allocated workspaces (see `ESWorkspace` in `src/esper_evolution.mojo`) — allocate once, reuse via `UnsafePointer`, never `alloc`/`free` per-iteration.
- **SIMD + FMA required.** Any vector math over weights/gradients must be vectorized using `simd_width_of[DType.float32]()` and `fma()`, with an explicit scalar remainder loop for sizes not divisible by the SIMD width. Follow the existing pattern in `update_fast_weights` / `evolve_fast_weights` / `calculate_fitness` exactly when adding similar loops.
- **POD over the arena.** Structs placed into the `HopeArena` (e.g. `HopeNode`) must stay POD (raw pointers + integers + `InlineArray`, no owning members), so they can be `init_pointee_move`'d into bump-allocated memory without per-node heap allocation or leaks.
- **Learn the transformation; never hand-code a DSL.** The engine must *learn* the grid→grid mapping in-context (fast weights = a learned operator fit by ES on demonstrations), not select among hand-written symbolic primitives. The existing `prim_*`/`apply_primitive` are being demoted; symbolic transforms belong only in the offline `synth_tasks.py` generator. See "Direction (active roadmap)".

## Development environment

Infrastructure is hermetic via Nix + `uv`.

```bash
nix develop                 # enters the dev shell; shellHook bootstraps .venv and installs mojo if missing
uv venv --python 3.12       # pin 3.12 — the Mojo 1.0.0b2 wheels have no 3.14 build, and bare `uv venv` may pick 3.14
source .venv/bin/activate
uv pip install "mojo==1.0.0b2" --prerelease allow   # also done by the flake shellHook
uv pip install numpy        # needed by src/arc_compiler.py
```

The repo's `uv.toml` exists solely to make that install work: the user's global uv config sets `exclude-newer = "30 days"`, which filters out the Modular toolchain (mojo, mojo-compiler, mblack, …), all published 2026-06-18. `uv.toml` overrides the cutoff per-package so the install resolves with no extra flags. Bump those dates if the version pin moves to a newer build.

## Testing

```bash
./run_tests.sh
```

`run_tests.sh` generates sample fixtures (a `.bin` grid and a `.task` bundle), runs every `tests/test_*.mojo` with `mojo run -I src <file>` (the `-I src` puts the source modules on the import path), runs the `src/main.mojo` driver, and finally generates a few `.task` bundles via `synth_tasks.generate_task_groups` and runs the `src/arc_solve.mojo` held-out generalization driver over them. To run a single test directly: `mojo run -I src tests/test_demo_fitness.mojo`. The ES-based tests anneal over a few thousand iterations (a few seconds each), so the full suite is ~60s. Tests `raise Error(...)` on assertion failure rather than using a test framework — follow that pattern for new tests. CI (`.github/workflows/ci.yml`) also enforces `mojo format` (note: this build's `mojo format` has **no `--check` flag** — CI runs the formatter then checks for a git diff).

## Architecture map

- `src/hope.mojo` — Core data structures + operator *execution* (no learning, so it has no `arc_io`/ES deps): `ArcGrid` (owned row-major Float32 grid), `HopeArena` (move-only bump allocator with `bump[T](count)` / `alloc_node[T]`), the **POD** `HopeNode` (raw `slow`/`fast` weight slices into an arena + inline child indices) and its `build_node` factory, and the **structured learned operator** — `OP_DIM` layout (6-param centered affine + 10-entry **normalized** color LUT, stored /9 to keep colour at the affine's ~unit scale), `seed_identity_operator`, and `apply_operator`. The operator is a **bilinear geometry gather** where each of the four corner input cells is mapped through the colour LUT (`_color_of`) *before* the blend — **colour-then-gather** decouples the colour fit from the geometry's precision. Smooth so the ES has a real gradient everywhere, yet exact at integer params. `ArcTaskPair` (demo input→output pair) and `ArcTask` (`train`/`test` lists) live here too. The old `prim_*`/`apply_primitive` and `update_fast_weights` remain but are **dormant** (off the operator path); `forward_with_learning` moved to `esper_evolution.mojo` (import-DAG reasons).
- `src/esper_evolution.mojo` — All learning. `operator_fitness` runs a candidate operator on each demo input, scores negative MSE vs the demo output (penalizing a heavy constant if the demo's output area differs from its input area — the operator is same-shape), minus an L2 anchor toward the slow prior. It replaced the deleted `evaluate_primitives` memorization surrogate. `ESWorkspace` holds **param-sized** ES vectors plus a **grid-sized** `op_output` scratch and a per-parameter `scale` (the colour group gets `COLOR_SCALE` < 1 — a diagonal **preconditioner** so one step size fits both geometry and colour). `evolve_fast_weights` is one antithetic-sampling ES step over the demos — **real Gaussian noise** (`randn_float64`), mirrored `grad += (F(w+σ·scale·ε) − F(w−σ·scale·ε))·ε`, then `W_fast += alpha/(2Nσ)·scale·grad`. `fit_operator` is the **annealed** fit loop (the shared `FIT_*` schedule: sigma 0.5→0.01, alpha 0.1→0.003 over ~4000 iters — explore wide, then settle onto exact integers; the wide `FIT_SIGMA0` is what makes transpose's four-param move robust). `forward_with_learning` ties it together: fit `node.fast` to the demos (anchored to `node.slow`), then `apply_operator` on the test input.
- `src/arc_io.mojo` — `_read_grid_block` reads one validated grid-block (16-byte header: two little-endian Int64 `rows`/`cols`, then the float32 payload) at an offset, advancing it; shared by `load_arc_grid` (single `.bin`) and `load_arc_task` (a `.task` **bundle**: `[n_train][n_test]` then the train/test grid-blocks → an `ArcTask`). Header lengths are validated against the file so truncated/malformed input raises. `calculate_fitness` = SIMD negative MSE (continuous ES signal); `exact_match` = discrete reward (fraction equal after `round`). (Note: Mojo's `open` only accepts `r/w/rw/a`; `read_bytes()` returns raw bytes.)
- `src/main.mojo` — End-to-end driver: builds an `OP_DIM` node, seeds slow (prior) and fast (init) to identity, learns `flip_h` in-context via `forward_with_learning`, prints result + held-out exact match. `mojo run -I src src/main.mojo`.
- `src/arc_solve.mojo` — **Held-out generalization driver** (replaced `benchmark.mojo`). Takes `.task` bundle paths via argv (shell-globbed), fits each task's operator on its train pairs via `fit_operator`, scores the **unseen** test pair(s), and reports per-task held-out + train-fit + the train/test gap, then the aggregate solve rate. A pair whose output area ≠ input area honestly scores 0 (the operator is same-shape — this guards against an OOB compare on real shape-changing ARC tasks). Raises on 0 solved (a CI regression signal for the synth bundles) **unless** the first arg is `--report` (honest real-ARC eval mode, where 0% is a legitimate number). Uncheatable by memorization.
- `src/arc_compiler.py` — The one sanctioned Python component: offline compiler. `_write_grid`/`_save_grid` (single grid `.bin`) and `_save_task` (a `.task` bundle) are the single source of the on-disk formats; `compile_arc_json` converts ARC JSON to per-grid `.bin`s. `compile_task_to_bundle`/`compile_arc_dir` + a `__main__` CLI (`python src/arc_compiler.py <json_dir> <out_dir>`) batch-ingest a real **ARC-AGI 2** corpus directory into `{task_id}.task` bundles for `arc_solve.mojo` (the M8 path). The corpus is **not** vendored (`data_bin/` and the `arg-agi-2-data` symlink are gitignored). Not part of the runtime path.
- `src/synth_tasks.py` — Offline deterministic generator: `generate_tasks` (single-grid pairs) and `generate_task_groups` (ARC-shaped `.task` bundles: N train demos + a held-out test, per transform), reusing the `arc_compiler` writers. It is the *ground-truth generator* the engine must rediscover — the symbolic transforms (flip/transpose/recolor/shift) live here, never in the engine.
- `tests/` — `test_arena`, `test_operator` (hand-set weights reproduce the transforms exactly), `test_fitness`, `test_demo_fitness` (keystone: ES fits `flip_h` and generalizes), `test_forward_learning` (end-to-end node path, fit-once/generalize-many), `test_task_loader` (bundle round-trip), `test_shape` (same-shape fits; shape-change penalized, no crash), `test_generalization` (**whole expressible subset** learned to ≥0.95 held-out), `test_io`. All import the real `src` modules via `-I src`. Phase-A expressible subset = {identity, flip_h, flip_v, transpose, recolor}; `shift` deferred (the affine zero-fills, synth `_shift` wraps). The ES-based tests anneal a few thousand iters, so the full suite is ~60s.

## Conventions to preserve when extending

- Pointer-owning structs pair `alloc` in `__init__` with `free` in `__del__` (unconditional — see the toolchain note on non-null pointers); rely on derived `Movable` for transfer, never hand-write `__moveinit__`.
- SIMD loops follow the same three-part shape every time: vectorized main loop over `range(0, size - nelts + 1, nelts)`, then a scalar remainder loop over `range(size - remainder, size)`.
- Anything placed into the arena must remain POD; reference children/other nodes by arena index, not by owning pointer.
