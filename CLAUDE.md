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

Phase 2 then strips the structural priors toward an emergent self-modifying memory and lifts the grid/task abstraction to a domain interface so the same ES / two-timescale core serves other reasoning problems. (Full roadmap — status, next steps, and the working discipline — lives in **`docs/ROADMAP.md`**, the canonical project direction. Per-session Claude Code plan files are ephemeral and not authoritative.)

The theory behind the architecture (the "Nested Learning"/HOPE paper, <https://abehrouz.github.io/files/NL.pdf>) is distilled in **`docs/NL-summary.md`** — read that instead of fetching the 52-page PDF. It ends with a table mapping each paper concept onto Esper's `slow`/`fast` weights, ES optimizer, and CMS direction.

## Journal (keep it updated)

Maintain a running, timestamped development narrative in **`docs/JOURNAL.md`**. Append an entry (with a `HH:MM` local timestamp) at every step of real progress, every unexpected discovery, and every blocker — including the diagnosis and the fix. Newest entries at the bottom. The goal is a complete story of *why* the code looks the way it does (e.g. why the operator uses bilinear sampling, why the ES anneals sigma), not just *what* changed — git already records the what. Update it as you work, not only at the end.

## Version control

This is a local-first repo. When asked to commit, commit **directly to `master`** — no feature branch, no PR — and **never push to the remote**. (The standard `Co-Authored-By` commit-message trailer still applies.)

## Toolchain (important)

Core/runtime code targets **Mojo 1.0.0b2** (1.0 beta). This is a hard pin: the 1.0 line removed `fn` (use `def`), renamed `alias`→`comptime`, requires `std.`-qualified stdlib imports, removed the stdlib `Tensor` type, and changed the `UnsafePointer`/argument conventions. Code written for older nightlies will not compile. Mojo is installed from **PyPI** (`uv pip install "mojo==1.0.0b2" --prerelease allow`), not the Modular wheel index.

Key 1.0 idioms used throughout (match these when extending):
- `def` everywhere; add `raises` to any `def` that can raise. `comptime X = ...` for compile-time constants.
- Imports are `std.`-qualified: `std.memory` (`alloc`, `memset_zero`, `memcpy(dest=, src=, count=)`, `UnsafePointer`), `std.sys` (`size_of`, `simd_width_of`, `argv`), `std.math` (`fma`, `round`), `std.random` (`randn_float64`, `seed`), `std.collections` (`List`, `InlineArray`).
- SIMD comparison/rounding gotchas: `a == b` on two SIMD vectors returns a **scalar `Bool`** (whole-vector equality) — for an elementwise mask use `a.eq(b)`, then `.cast[DType.float32]().reduce_add()` to count hits. There is **no SIMD `.round()` method**; use the free `round()` from `std.math` (works on both SIMD vectors and scalars). See `exact_match` in `arc_io.mojo`.
- Raw memory: `var p = alloc[T](count)` / `p.free()`. Pointer fields need an explicit origin: `UnsafePointer[Float32, MutAnyOrigin]`. Arithmetic is `(p + n).bitcast[T]()`; SIMD is `p.load[width=nelts](i)` / `p.store[width=nelts](i, v)`.
- Lifecycle: `def __init__(out self, ...)`, `def __del__(deinit self)`. Do **not** write `__moveinit__` — derive `(Movable)` / `(Copyable, Movable)`; moves consume the source so a moved-from value's `__del__` never runs.
- `UnsafePointer` is non-null by design — do **not** guard `__del__` with `if self.data:`; free unconditionally.
- **GPU support ships in the same PyPI wheel** — no extra install (pixi/conda not needed; the GPU-puzzles pixi env pins `mojo <1.0.0` and would conflict with this pin). Requires NVIDIA driver ≥ 580 (older needs `MODULAR_NVPTX_COMPILER_PATH` pointing at a system `ptxas`); Turing-class cards (e.g. the dev box's RTX 2060) are Modular's "known compatible for development" tier. Idioms: `from std.gpu.host import DeviceContext`, `from std.gpu import global_idx`; kernels are plain `def`s whose pointer params are `UnsafePointer[T, MutAnyOrigin]`; host side is `ctx.enqueue_create_buffer[dtype](n)`, `with buf.map_to_host() as h:` for host access, `ctx.enqueue_function[kernel](..., grid_dim=, block_dim=)`, `ctx.synchronize()`. Gate GPU code with `comptime if not has_accelerator()` (comptime — device code isn't compiled on CPU-only hosts, e.g. CI). See `tests/test_gpu_env.mojo` for the canonical smoke test.

## Hard constraints (do not violate)

- **Language: Pure Mojo** for all core/runtime code. Python is isolated strictly to the offline data-compilation toolchain (`tools/arc_compiler.py`, `tools/synth_tasks.py`) — `src/` is pure Mojo by construction — never introduce Python fallbacks or dynamic bindings into the execution/inference path.
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
uv pip install numpy        # needed by tools/arc_compiler.py
```

The repo's `uv.toml` exists solely to make that install work: the user's global uv config sets `exclude-newer = "30 days"`, which filters out the Modular toolchain (mojo, mojo-compiler, mblack, …), all published 2026-06-18. `uv.toml` overrides the cutoff per-package so the install resolves with no extra flags. Bump those dates if the version pin moves to a newer build.

## Testing

```bash
./esper suite        # = ./run_tests.sh (full suite: everything, ~10 min)
./esper fast         # = ./run_tests.sh fast (quick local gate, ~2 min)
./esper test <name>  # run tests/test_<name>.mojo  (e.g. ./esper test mlp_memory)
./esper run <file>   # mojo run -I src <file>
./esper fmt          # mojo format src tests
```

**`./esper`** is the dev harness: one entrypoint that activates the project venv and runs from the repo root, so commands don't need `cd` + `source .venv/bin/activate` + `mojo run -I src` spelled out each time (`run`/`test`/`main`/`solve`/`fmt`/`suite`/`fast`/`mojo` subcommands; see the script header). Use it for ad-hoc runs.

**Suite tiers.** The full suite is ~10 min, dominated by the large-scale milestone proofs tagged `# suite-tier: full` (`test_grid_context_selfmod`, `test_grid_nbhd_selfmod`, `test_grid_countmap_selfmod`, `test_delta_selfmod`, `test_composed_generalization`). `./esper fast` (= `run_tests.sh fast`) skips those for a ~2 min local gate; `./esper suite` (default `full`) runs everything and is what CI runs. **The fast tier is a strict subset at FULL budget** — no reduced iterations, no relaxed thresholds — so it never makes a weakened claim; it still exercises every code path (structural, the ES operator fit, and the self-mod `meta_fit_selfmod` core via `test_selfmod_memory`) and only defers the large-scale milestone proofs. A test opts *out* of `fast` by tagging a `# suite-tier: full` comment line at its top (grepped by `run_tests.sh`); untagged tests default to `fast`, so new tests run in both unless they declare themselves heavy.

`run_tests.sh [full|fast]` generates sample fixtures (a `.bin` grid and a `.task` bundle), runs the tier's `tests/test_*.mojo` with `mojo run -I src <file>` (the `-I src` puts the source modules on the import path), runs the `src/main.mojo` driver, and finally generates a few `.task` bundles via `synth_tasks.generate_task_groups` (plus, full tier only, one shape-changing bundle via `generate_shape_task_groups` so the driver's shape dispatch runs in CI) and runs the `src/arc_solve.mojo` held-out generalization driver over them. To run a single test directly: `mojo run -I src tests/test_demo_fitness.mojo`. Tests `raise Error(...)` on assertion failure rather than using a test framework — follow that pattern for new tests. CI (`.github/workflows/ci.yml`) also enforces `mojo format` (note: this build's `mojo format` has **no `--check` flag** — CI runs the formatter then checks for a git diff).

## Architecture map

The engine is a small set of flat `-I src` Mojo modules. **The detailed per-module map — data structures, the ES core, the memory-trait seams, every memory family and driver, and the full test roster — lives in `docs/ARCHITECTURE.md`** (read on demand, like the other companion docs). High-level layout:

- `src/hope.mojo` — core POD data structures (`ArcGrid`, `HopeArena`, the POD `HopeNode`) + the structured operator's *execution*; the generic `ExamplePair`/`Task` demo containers.
- `src/esper_evolution.mojo` — **all learning**, generic over `[M: Memory]`: the derivative-free ES core (`fitness` / `evolve_fast_weights` / `fit_operator`), the two-timescale Reptile meta-loop, and the per-task composed / shape fit drivers.
- `src/gpu_es.mojo` — the **GPU-batched fitness backend** (default on accelerator hosts; `arc_solve --cpu` forces the CPU reference): one kernel launch per ES iteration scores all (candidate × demo) pairs via the same `attn_pixel_*` per-pixel functions the CPU path uses; everything else (RNG, gradient, update) stays CPU. CPU-only hosts compile zero device code (`comptime if has_accelerator()`).
- `src/arc_io.mojo` — the on-disk `.bin` / `.task` readers + the `Domain` trait / `GridDomain` (the metric seam the ES reaches metrics through — never ARC directly).
- `src/memory.mojo` — **the memory-trait seams**: `Memory`, `SelfModMemory`, `ShapeMemory`. No runtime selector — each memory is a compile-time choice, measured on the subset it expresses.
- `src/memory_es.mojo` — the **ES-fit forward family** (the dormant structured operator, MLP, the sequence-domain pair, the `AttnGather` geometry gather + the toroidal / reflect shape gathers).
- `src/memory_composed.mojo` — the **emergent composed memories** (geometry × colour × count × shape; the shape-seam pair plus its Rung C colour and Rung A local-write layers).
- `src/memory_selfmod.mojo`, `src/memory_selfmod_grid.mojo` — the **self-write memory families** (fast adaptation is the memory's own write rule over the demos; the ES meta-learns only the small slow vector).
- `src/main.mojo`, `src/arc_solve.mojo` — the end-to-end driver + the **held-out generalization driver** (`eval_parallel.sh` shards it across processes).
- `tools/arc_compiler.py`, `tools/synth_tasks.py` — the offline corpus compiler + the ground-truth synth generator (the only sanctioned Python; never on the runtime path).
- `tests/` — one `test_*.mojo` per milestone / memory, each a held-out generalization proof; `raise Error` on failure.

## Conventions to preserve when extending

- Pointer-owning structs pair `alloc` in `__init__` with `free` in `__del__` (unconditional — see the toolchain note on non-null pointers); rely on derived `Movable` for transfer, never hand-write `__moveinit__`.
- SIMD loops follow the same three-part shape every time: vectorized main loop over `range(0, size - nelts + 1, nelts)`, then a scalar remainder loop over `range(size - remainder, size)`.
- Anything placed into the arena must remain POD; reference children/other nodes by arena index, not by owning pointer.
