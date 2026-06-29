# Esper: Neuro-Symbolic Reasoning Engine

## Architecture
Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine designed specifically for logical deduction and pattern recognition (e.g., ARC-AGI).

The system relies on **Nested Learning**:
1. **Slow Weights:** Base routing knowledge.
2. **Fast Weights (Continuum Memory System):** Memory dynamically updated during the forward pass utilizing a derivative-free Evolution Strategy.
3. **HopeArena:** A strictly move-only, zero-overhead contiguous memory allocator driving hierarchical node generation.

## Status & Roadmap
The core infrastructure — the move-only `HopeArena`, POD `HopeNode`s, the SIMD/FMA antithetic-sampling Evolution Strategy, and validated `.bin` IO — is in place, and the **foundational learning loop now works**: the fast weights are a *learned* grid→grid operator (a smooth, bilinear, structured operator of fixed size) fit in-context by an annealed Evolution Strategy on demonstration pairs, anchored to the slow weights as a meta-learned prior. The engine learns a transformation purely from examples and **generalizes to held-out inputs** (e.g. it learns `flip_h` to ~0.99 held-out exact match) — no hand-coded DSL, no target memorization.

**Phase 1 — Learn ARC-AGI 2 (in progress).** Done: the learned-operator forward pass, the demonstration-driven fitness, and end-to-end held-out generalization on the expressible transform subset. Remaining: a task loader, variable-shape handling, a held-out generalization benchmark over the full subset, real ARC-AGI 2 ingest, and meta-learning the slow prior. Geometric/color transforms must **emerge** as fitted parameters ("training wheels"), progressively stripped toward a fully general parametric memory. Success is measured by **held-out generalization** — fit on the train pairs, score the unseen test pair (uncheatable by memorization).

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
