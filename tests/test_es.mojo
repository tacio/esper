# Proves the Evolution Strategy actually learns: with real Gaussian noise and
# antithetic sampling, evolve_fast_weights must measurably reduce MSE toward a
# synthetic target. (With the old constant-noise mock this loop was a no-op.)
from arc_io import calculate_fitness
from esper_evolution import ESWorkspace, evolve_fast_weights
from std.memory import alloc
from std.random import seed


def main() raises:
    seed(7)
    var size = 16
    var weights = alloc[Float32](size)
    var target = alloc[Float32](size)
    for i in range(size):
        weights[i] = 0.0
        target[i] = Float32(i) * 0.5 - 3.0

    var workspace = ESWorkspace(size)
    var mse_before = -calculate_fitness(weights, target, size)
    for _ in range(200):
        evolve_fast_weights(weights, workspace, target, 24, 0.20, 0.10)
    var mse_after = -calculate_fitness(weights, target, size)

    if mse_after >= mse_before:
        raise Error("ERROR: ES failed to reduce MSE (no learning).")
    if mse_after > 0.01:
        raise Error("ERROR: ES did not converge close enough to the target.")

    # Degenerate guards must be no-ops, not crashes / NaNs.
    evolve_fast_weights(weights, workspace, target, 0, 0.20, 0.10)
    evolve_fast_weights(weights, workspace, target, 24, 0.20, 0.0)

    weights.free()
    target.free()
    print(
        "Evolution Strategy tests passed (MSE",
        mse_before,
        "->",
        mse_after,
        ").",
    )
