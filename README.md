# Esper: Neuro-Symbolic Reasoning Engine

## Architecture
Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine designed specifically for logical deduction and pattern recognition (e.g., ARC-AGI).

The system relies on **Nested Learning**:
1. **Slow Weights:** Base routing knowledge.
2. **Fast Weights (Continuum Memory System):** Memory dynamically updated during the forward pass utilizing a derivative-free Evolution Strategy.
3. **HopeArena:** A strictly move-only, zero-overhead contiguous memory allocator driving hierarchical node generation.

## Status & Roadmap
The core infrastructure — the move-only `HopeArena`, POD `HopeNode`s, the SIMD/FMA antithetic-sampling Evolution Strategy, and validated `.bin` IO — is in place, and the **foundational learning loop now works**: the fast weights are a *learned* grid→grid operator (a smooth, bilinear, structured operator of fixed size) fit in-context by an annealed Evolution Strategy on demonstration pairs, anchored to the slow weights as a meta-learned prior. The engine learns a transformation purely from examples and **generalizes to held-out inputs** (e.g. it learns `flip_h` to ~0.99 held-out exact match) — no hand-coded DSL, no target memorization.

**Phase 1 — Learn ARC-AGI 2 (in progress).** Done (M1–M8): the learned-operator forward pass, the demonstration-driven fitness, the task loader and held-out generalization driver, the **whole expressible transform subset** ({flip_h, flip_v, transpose, recolor}) learned from demonstrations to **held-out 1.0 with train/test gap 0.0**, and the **real ARC-AGI 2 ingest + honest eval**: on the 120-task public-eval split, fit on each task's train pairs and scored on the unseen test, the engine solves **0/120 (mean held-out exact-match 0.033)** — the expected, truthful number, since real ARC-AGI 2 is almost entirely outside the geometry+color subset (objects, counting, symmetry, shape change). Remaining: meta-learn the slow prior (M9). Geometric/color transforms **emerge** as fitted parameters ("training wheels"), to be progressively stripped toward a fully general parametric memory. Success is measured by **held-out generalization** — fit on the train pairs, score the unseen test pair (uncheatable by memorization).

To reproduce the ARC-AGI 2 number: clone the [ARC-AGI-2 corpus](https://github.com/arcprize/ARC-AGI-2) (not vendored), then `python src/arc_compiler.py <corpus>/data/evaluation data_bin/arc2_eval` and `mojo run -I src src/arc_solve.mojo --report data_bin/arc2_eval/*.task`.

**Phase 2 — Generalize.** Remove the structural priors toward an emergent self-modifying memory, and lift the grid/task abstraction to a domain interface so the same ES / two-timescale core serves other reasoning problems.

See `docs/JOURNAL.md` for the development narrative and `docs/NL-summary.md` for the architecture's theoretical basis.

## Constraints & Requirements
This project enforces a strict, bare-metal systems engineering ethos:
* **Language:** Pure Mojo, targeting **Mojo 1.0.0b2** (1.0 beta).
* **Dependencies:** Zero external ML libraries (No PyTorch, No TensorFlow).
* **Execution Loop:** No Python fallbacks or dynamic bindings in the core execution or inference passes. Python is isolated strictly to the offline data compilation toolchain.
* **Hardware Utilization:** Fast weight updates mandate `simd_width_of` vectorization and Fused Multiply-Add (FMA) instructions.

## Development Environment
Infrastructure is strictly defined via Nix and `uv`. Mojo is installed from PyPI (the 1.0 beta is a prerelease).

### Initialization
```bash
nix develop                                          # shellHook bootstraps the steps below
uv venv
source .venv/bin/activate
uv pip install "mojo==1.0.0b2" --prerelease allow
uv pip install numpy
```

### Testing
```bash
./run_tests.sh
```
This generates a sample `.bin`, runs every `tests/test_*.mojo` (with `-I src`), and runs the `src/main.mojo` driver end-to-end.
