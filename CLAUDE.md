# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine for logical deduction and pattern recognition (e.g. ARC-AGI). It implements a "Nested Learning" architecture (HOPE) with three layers:

1. **Slow Weights** — base routing knowledge, updated rarely (the L2 anchor for the fast weights).
2. **Fast Weights (Continuum Memory System)** — memory dynamically updated during the forward pass via a derivative-free Evolution Strategy (ES), not backprop.
3. **HopeArena** — a strictly move-only, zero-overhead contiguous bump-pointer allocator that hosts POD `HopeNode`s and their weight slices.

## Direction (active roadmap)

The current learning path is deliberately a **placeholder being replaced**: `evaluate_primitives` is an identity surrogate (it memorizes the target grid) and `forward_with_learning` uses a constant gradient. The active roadmap makes the engine *genuinely learn*. The spine decision — which all future work must respect:

- **Fast weights parameterize a *learned* grid→grid operator**, fit in-context by the ES on each task's demonstration pairs; the slow weights are the meta-learned prior (the existing L2 anchor). This unifies the two conflicting "fast weights" notions (op-params vs. one-weight-per-pixel) into a single, grid-size-independent operator parameter vector (`OP_DIM`).
- **Never hand the engine a symbolic DSL.** Geometric/color transforms (flip/transpose/recolor) must *emerge* as fitted operator parameters; they live only in the offline `synth_tasks.py` generator as ground truth to rediscover. The north star is fully emergent (a general parametric memory); structured priors are removable "training wheels".
- **Metric = held-out generalization** (fit on a task's train pairs, score the unseen test pair — uncheatable by memorization), not fitting a known target. `benchmark.mojo`'s memorization is being replaced by a generalization driver (`src/arc_solve.mojo`).

Phase 2 then strips the structural priors toward an emergent self-modifying memory and lifts the grid/task abstraction to a domain interface so the same ES / two-timescale core serves other reasoning problems. (Full milestone plan: `~/.claude/plans/let-s-create-a-roadmap-ticklish-adleman.md`, user-local.)

The theory behind the architecture (the "Nested Learning"/HOPE paper, <https://abehrouz.github.io/files/NL.pdf>) is distilled in **`docs/NL-summary.md`** — read that instead of fetching the 52-page PDF. It ends with a table mapping each paper concept onto Esper's `slow`/`fast` weights, ES optimizer, and CMS direction.

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

`run_tests.sh` generates a sample `.bin` (via `arc_compiler.py`), then runs every `tests/test_*.mojo` with `mojo run -I src <file>` (the `-I src` puts the source modules on the import path), runs the `src/main.mojo` driver end-to-end, and finally generates a small task set via `synth_tasks.py` and runs the `src/benchmark.mojo` objective-reward harness. To run a single test directly: `mojo run -I src tests/test_es.mojo`. Tests `raise Error(...)` on assertion failure rather than using a test framework — follow that pattern for new tests. CI (`.github/workflows/ci.yml`) also enforces `mojo format` (note: this build's `mojo format` has **no `--check` flag** — CI runs the formatter then checks for a git diff).

## Architecture map

- `src/hope.mojo` — Core data structures: `ArcGrid` (owned row-major Float32 grid), `HopeArena` (move-only bump allocator with `bump[T](count)` / `alloc_node[T]`), the **POD** `HopeNode` (raw `slow`/`fast` weight slices into an arena + inline child indices) and its `build_node` factory, the logic primitives (`prim_identity`, `prim_shift`, dispatched by op-code via `apply_primitive`), the vectorized `update_fast_weights` (`W_fast -= alpha*(grad + lambda*W_slow)`, real L2 anchor on the slow weights), and `forward_with_learning` (learning phase reuses one gradient buffer — no per-iteration alloc — then an evaluation phase runs a real primitive forward pass on the test input).
- `src/esper_evolution.mojo` — The Evolution Strategy optimizer for fast weights: `ESWorkspace` (persistent, reusable scratch — grad estimate, epsilon, perturbed weights) and `evolve_fast_weights`, which uses **real Gaussian noise** (`randn_float64`) with **antithetic (mirrored) sampling** — `grad += (F(w+σε) − F(w−σε))·ε`, then `W_fast += alpha/(2Nσ)·grad`. Mirroring centres the fitness, so no separate normalisation is needed; degenerate `N<=0`/`σ==0` are guarded. `evaluate_primitives` is the seam to the model forward pass — it currently scores perturbed weights directly against the target (an identity surrogate); a full implementation would load them into a `HopeNode` and run `apply_primitive` first.
- `src/arc_io.mojo` — `load_arc_grid` reads the custom `.bin` format produced by `arc_compiler.py` (16-byte header: two little-endian Int64 `rows`/`cols`, then the flattened float32 payload) into an `ArcGrid` via `memcpy`, **validating the header against the file length** so a truncated/malformed file raises instead of reading out of bounds. `calculate_fitness` computes SIMD-accelerated negative MSE — the continuous ES fitness signal; `exact_match` is the **discrete** objective reward (fraction of cells equal after `round`, mirroring ARC's per-cell scoring). (Note: Mojo's `open` only accepts `r/w/rw/a`, not `rb`; `read_bytes()` already returns raw bytes.)
- `src/main.mojo` — End-to-end driver: builds an arena-hosted node, runs `forward_with_learning` over a demonstration with `PRIM_SHIFT`, and prints the result + fitness. Run with `mojo run -I src src/main.mojo`.
- `src/benchmark.mojo` — Objective-reward harness: takes `*_out.bin` target paths as argv, evolves a fresh fast-weight buffer to fit each via `evolve_fast_weights`, and reports negative-MSE reward, exact-match %, and an aggregate solve rate (raises if it solves 0). `evolve_fast_weights` takes the target as an `UnsafePointer` (pass `grid.data`), not a grid object.
- `src/arc_compiler.py` — The one sanctioned Python component: offline compiler converting ARC-AGI JSON task files into the raw `.bin` format. Not part of the runtime/inference path.
- `src/synth_tasks.py` — Offline (Python) deterministic task generator: emits single-transform (flip/transpose/recolor/shift) input/target grid pairs as `.bin`, reusing `arc_compiler._save_grid` so the on-disk format stays single-sourced. Feeds the benchmark and is offline-only, like `arc_compiler.py`.
- `tests/` — `test_arena` (arena + POD node), `test_fitness` (known-value MSE), `test_es` (ES must measurably reduce MSE — guards against regressing the noise to a constant), `test_io` (round-trip + malformed-file rejection), `test_es_convergence` (end-to-end objective-reward check: reward must climb toward a known target and reach an exact match). All import the real `src` modules via `-I src` (never duplicate definitions).

## Conventions to preserve when extending

- Pointer-owning structs pair `alloc` in `__init__` with `free` in `__del__` (unconditional — see the toolchain note on non-null pointers); rely on derived `Movable` for transfer, never hand-write `__moveinit__`.
- SIMD loops follow the same three-part shape every time: vectorized main loop over `range(0, size - nelts + 1, nelts)`, then a scalar remainder loop over `range(size - remainder, size)`.
- Anything placed into the arena must remain POD; reference children/other nodes by arena index, not by owning pointer.
