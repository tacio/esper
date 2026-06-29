# Esper: Neuro-Symbolic Reasoning Engine

## Architecture
Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine designed specifically for logical deduction and pattern recognition (e.g., ARC-AGI).

The system relies on **Nested Learning**:
1. **Slow Weights:** Base routing knowledge.
2. **Fast Weights (Continuum Memory System):** Memory dynamically updated during the forward pass utilizing a derivative-free Evolution Strategy.
3. **HopeArena:** A strictly move-only, zero-overhead contiguous memory allocator driving hierarchical node generation.

## Status & Roadmap
The core infrastructure — the move-only `HopeArena`, POD `HopeNode`s, the SIMD/FMA antithetic-sampling Evolution Strategy, the L2-anchored fast-weight update, and validated `.bin` IO — is in place. The learning path is currently a **placeholder/surrogate** (it memorizes a target grid rather than learning a transformation); the active roadmap replaces it with genuine in-context learning.

**Phase 1 — Learn ARC-AGI 2.** The fast weights are redefined as a *learned* grid→grid operator, fit in-context by the Evolution Strategy on each task's demonstration pairs; the slow weights are the meta-learned prior. The engine is **never handed a symbolic DSL** — geometric/color transforms must emerge as fitted parameters of a structured operator ("training wheels"), progressively stripped toward a fully general parametric memory. Success is measured by **held-out generalization**: fit on the train pairs, score the unseen test pair (uncheatable by memorization).

**Phase 2 — Generalize.** Remove the structural priors toward an emergent self-modifying memory, and lift the grid/task abstraction to a domain interface so the same ES / two-timescale core serves other reasoning problems.

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
