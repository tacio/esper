# Esper: Neuro-Symbolic Reasoning Engine

## Architecture
Esper is a bare-metal, high-performance Neuro-Symbolic Reasoning Engine designed specifically for logical deduction and pattern recognition (e.g., ARC-AGI).

The system relies on **Nested Learning**:
1. **Slow Weights:** Base routing knowledge.
2. **Fast Weights (Continuum Memory System):** Memory dynamically updated during the forward pass utilizing a derivative-free Evolution Strategy.
3. **HopeArena:** A strictly move-only, zero-overhead contiguous memory allocator driving hierarchical node generation.

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
