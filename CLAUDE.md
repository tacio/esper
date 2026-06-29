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

`run_tests.sh` generates a sample `.bin` (via `arc_compiler.py`), then runs every `tests/test_*.mojo` with `mojo run -I src <file>` (the `-I src` puts the source modules on the import path), and runs the `src/main.mojo` driver end-to-end (which learns `flip_h` in-context and reports held-out generalization). To run a single test directly: `mojo run -I src tests/test_demo_fitness.mojo`. The ES-based tests anneal over a few thousand iterations, so each takes a few seconds. Tests `raise Error(...)` on assertion failure rather than using a test framework — follow that pattern for new tests. CI (`.github/workflows/ci.yml`) also enforces `mojo format` (note: this build's `mojo format` has **no `--check` flag** — CI runs the formatter then checks for a git diff).

## Architecture map

- `src/hope.mojo` — Core data structures + operator *execution* (no learning, so it has no `arc_io`/ES deps): `ArcGrid` (owned row-major Float32 grid), `HopeArena` (move-only bump allocator with `bump[T](count)` / `alloc_node[T]`), the **POD** `HopeNode` (raw `slow`/`fast` weight slices into an arena + inline child indices) and its `build_node` factory, and the **structured learned operator** — `OP_DIM` layout (6-param centered affine + 10-entry color LUT), `seed_identity_operator`, and `apply_operator` (a **bilinear** geometry gather + **linearly-interpolated** color LUT; smooth so the ES has a real gradient everywhere, yet exact at integer params). `ArcTaskPair` (a demo input→output pair) lives here too. The old `prim_*`/`apply_primitive` primitives and `update_fast_weights` remain but are **dormant** (off the operator path); `forward_with_learning` moved to `esper_evolution.mojo` (import-DAG reasons).
- `src/esper_evolution.mojo` — All learning. `operator_fitness` runs a candidate operator on each demo input and scores negative MSE vs the demo output, minus an L2 anchor toward the slow prior — the real, demonstration-driven signal (it replaced the deleted `evaluate_primitives` memorization surrogate). `ESWorkspace` holds **param-sized** ES vectors (grad estimate / epsilon / perturbed weights, length `OP_DIM`) plus a **grid-sized** `op_output` scratch. `evolve_fast_weights` is one antithetic-sampling ES step over the demos — **real Gaussian noise** (`randn_float64`), mirrored sampling `grad += (F(w+σε) − F(w−σε))·ε`, then `W_fast += alpha/(2Nσ)·grad`; degenerate `N<=0`/`σ==0` guarded. `fit_operator` is the **annealed** fit loop (sigma/alpha decay geometrically over `iters` — explore wide, then settle onto exact integer params; this annealing is what makes the piecewise structure learnable). `forward_with_learning` ties it together: fit `node.fast` to the demos via `fit_operator` (anchored to `node.slow`), then `apply_operator` on the test input.
- `src/arc_io.mojo` — `load_arc_grid` reads the custom `.bin` format produced by `arc_compiler.py` (16-byte header: two little-endian Int64 `rows`/`cols`, then the flattened float32 payload) into an `ArcGrid` via `memcpy`, **validating the header against the file length** so a truncated/malformed file raises instead of reading out of bounds. `calculate_fitness` computes SIMD-accelerated negative MSE — the continuous ES fitness signal; `exact_match` is the **discrete** objective reward (fraction of cells equal after `round`, mirroring ARC's per-cell scoring). (Note: Mojo's `open` only accepts `r/w/rw/a`, not `rb`; `read_bytes()` already returns raw bytes.)
- `src/main.mojo` — End-to-end driver: builds an `OP_DIM` arena-hosted node, seeds slow (prior) and fast (init) to identity, learns `flip_h` in-context from random demonstration pairs via `forward_with_learning`, and prints the result + held-out exact match. Run with `mojo run -I src src/main.mojo`.
- `src/arc_compiler.py` — The one sanctioned Python component: offline compiler converting ARC-AGI JSON task files into the raw `.bin` format. Not part of the runtime/inference path.
- `src/synth_tasks.py` — Offline (Python) deterministic task generator: emits single-transform (flip/transpose/recolor/shift) input/target grid pairs as `.bin`, reusing `arc_compiler._save_grid` so the on-disk format stays single-sourced. Offline-only, like `arc_compiler.py`; it is the *ground-truth data generator* the engine must rediscover — the symbolic transforms live here, never in the engine.
- `tests/` — `test_arena` (arena + POD node), `test_operator` (hand-set operator weights reproduce identity/flip_h/transpose/recolor exactly), `test_fitness` (known-value MSE), `test_demo_fitness` (the keystone: annealed ES fits `flip_h` from demos and generalizes to held-out inputs), `test_forward_learning` (end-to-end node path: fit once, generalize to many held-out inputs), `test_io` (round-trip + malformed-file rejection). All import the real `src` modules via `-I src` (never duplicate definitions). Phase-A expressible subset = {identity, flip_h, flip_v, transpose, recolor}; `shift` is deferred (the affine zero-fills, synth `_shift` wraps).

## Conventions to preserve when extending

- Pointer-owning structs pair `alloc` in `__init__` with `free` in `__del__` (unconditional — see the toolchain note on non-null pointers); rely on derived `Movable` for transfer, never hand-write `__moveinit__`.
- SIMD loops follow the same three-part shape every time: vectorized main loop over `range(0, size - nelts + 1, nelts)`, then a scalar remainder loop over `range(size - remainder, size)`.
- Anything placed into the arena must remain POD; reference children/other nodes by arena index, not by owning pointer.
