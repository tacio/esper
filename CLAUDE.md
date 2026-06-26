# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine for logical deduction and pattern recognition (e.g. ARC-AGI). It implements a "Nested Learning" architecture (HOPE) with three layers:

1. **Slow Weights** — base routing knowledge, updated rarely.
2. **Fast Weights (Continuum Memory System)** — memory dynamically updated during the forward pass via a derivative-free Evolution Strategy (ES), not backprop.
3. **HopeArena** — a strictly move-only, zero-overhead contiguous bump-pointer allocator driving hierarchical `HopeNode` generation.

## Hard constraints (do not violate)

- **Language: Pure Mojo** for all core/runtime code. Python is isolated strictly to the offline data-compilation toolchain (`src/arc_compiler.py`) — never introduce Python fallbacks or dynamic bindings into the execution/inference path.
- **Zero external ML libraries.** No PyTorch, no TensorFlow, no autodiff frameworks. Learning is done via Evolution Strategies, not gradient descent/backprop.
- **No dynamic allocation in hot loops.** Fast-weight updates must reuse pre-allocated workspaces (see `ESWorkspace` in `src/esper_evolution.mojo`) — allocate once, reuse via `UnsafePointer`, never `alloc`/`free` per-iteration.
- **SIMD + FMA required.** Any vector math over weights/gradients must be vectorized using `simdwidthof[DType.float32]()` and `fma()`, with an explicit scalar remainder loop for sizes not divisible by the SIMD width. Follow the existing pattern in `update_fast_weights` / `evolve_fast_weights` / `calculate_fitness` exactly when adding similar loops.
- **Manual memory ownership.** Structs owning raw pointers (`HopeArena`, `ESWorkspace`) must implement `__moveinit__` (nullify the source pointer/fields after transfer) and `__del__` (free only if the pointer is non-null) to avoid double-frees — mirror the existing implementations.

## Development environment

Infrastructure is hermetic via Nix + `uv`; Mojo itself is installed from the Modular nightly wheel index, not via system packages.

```bash
nix develop                 # enters the dev shell; shellHook bootstraps .venv and installs mojo if missing
uv venv
source .venv/bin/activate
uv pip install numpy        # needed by src/arc_compiler.py
```

The `flake.nix` shellHook auto-detects `.venv/bin/mojo` and bootstraps it from `https://whl.modular.com/nightly/simple/` (`--prerelease allow`) on first run.

## Testing

```bash
./run_tests.sh
```

This runs `mojo run tests/test_arena.mojo` (and any further `mojo run tests/test_*.mojo` added to the script). To run a single test file directly: `mojo run tests/test_arena.mojo`. Tests use `raise Error(...)` on assertion failure rather than a test framework — follow that pattern for new tests (no pytest/testing library is used for Mojo code).

## Architecture map

- `src/hope.mojo` — Core data structures: `ArcGrid` (int8 grid wrapper over `Tensor`), `FastWeightBuffer`, `HopeNode` (slow + fast weights, child pointers), `HopeArena` (bump allocator, see Phase 2.1), primitive function signatures (`PrimitiveFunc`, `prim_shift`, `prim_flood_fill` — currently stubs), and the vectorized `update_fast_weights` FMA loop. `forward_with_learning` is the Phase 2 execution loop: it runs a "learning phase" over demonstration pairs (updating fast weights) followed by an "evaluation phase" on the test input. Several pieces here (gradients, routing, output prediction) are still mocked/simplified — check before assuming production behavior.
- `src/esper_evolution.mojo` — The real Evolution Strategy optimizer for fast weights: `ESWorkspace` (persistent, reusable ES scratch buffers — grad estimate, perturbed weights, epsilon) and `evolve_fast_weights`, which perturbs weights with noise, evaluates fitness via `calculate_fitness` (imported from `arc_io`), accumulates a gradient estimate weighted by fitness, and applies the final FMA update `W_fast += alpha/(N*sigma) * grad_estimate`. Noise generation is currently a fixed mock value (`0.1`), not a real RNG — a likely area of future work.
- `src/arc_io.mojo` — `load_arc_grid` reads the custom `.bin` format produced by `arc_compiler.py` (16-byte header: two Int64 `rows`/`cols`, followed by flattened float32 payload) directly into a `Tensor` via `memcpy`. `calculate_fitness` computes SIMD-accelerated negative MSE between a prediction and target buffer — this is the ES fitness signal.
- `src/arc_compiler.py` — The one sanctioned Python component: offline compiler that converts ARC-AGI JSON task files (`train`/`test`, `input`/`output` grids) into the raw `.bin` format consumed by `arc_io.mojo`. Not part of the runtime/inference path.
- `tests/test_arena.mojo` — Standalone smoke test for `HopeArena` (duplicates the struct rather than importing it — keep in sync with `src/hope.mojo` if `HopeArena` changes, or refactor to share the definition).

## Conventions to preserve when extending

- Pointer-owning structs always pair `alloc` in `__init__` with `free` in `__del__`, guarded by an "is non-null" check.
- SIMD loops follow the same three-part shape every time: vectorized main loop over `range(0, size - nelts + 1, nelts)`, then a scalar remainder loop over `range(size - remainder, size)`.
- `evaluate_primitives` in `esper_evolution.mojo` is the seam between the ES optimizer and the actual model forward pass — currently it treats raw perturbed weights as the output state directly; a full implementation would load them into an `Expert`/`HopeNode` and run a real forward pass.
