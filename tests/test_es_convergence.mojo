from std.memory import alloc, UnsafePointer
from std.random import seed

# Run from the project root with `mojo run -I src tests/test_es_convergence.mojo`
# so these cross-directory imports resolve.
from esper_evolution import ESWorkspace, evolve_fast_weights
from arc_io import calculate_fitness, exact_match

# ==========================================================================
# Objective-reward convergence test.
#
# Master's ES uses real Gaussian noise with antithetic sampling, so the fast
# weights should march toward a known target grid: the reward (negative MSE)
# must climb toward 0 and the rounded prediction must reach an exact match.
# This is the smallest end-to-end check that the arena/IO/ES/SIMD/reward
# plumbing actually learns (it would fail for the old constant-epsilon RNG,
# whose gradient estimate was degenerate).
# ==========================================================================


def main() raises:
    seed(0)

    comptime size = 64

    # Build an integer-valued target grid, like an ARC grid (colors 0-9).
    var target = alloc[Float32](size)
    for i in range(size):
        target[i] = Float32(i % 10)

    # Fast weights start at zero, i.e. far from the target.
    var fast = alloc[Float32](size)
    for i in range(size):
        fast[i] = 0.0

    var workspace = ESWorkspace(size)

    var N = 96
    var alpha = Float32(6.0)
    var sigma = Float32(0.1)
    var iters = 60

    var initial_reward = calculate_fitness(fast, target, size)
    var prev = initial_reward
    var improvements = 0

    for _ in range(iters):
        evolve_fast_weights(fast, workspace, target, N, alpha, sigma)
        var r = calculate_fitness(fast, target, size)
        if r > prev:
            improvements += 1
        prev = r

    var final_reward = calculate_fitness(fast, target, size)
    var match_frac = exact_match(fast, target, size)

    print("Initial reward (-MSE):", initial_reward)
    print("Final   reward (-MSE):", final_reward)
    print("Exact-match fraction :", match_frac)
    print("Improving iterations :", improvements, "/", iters)

    # 1. Reward must improve overall.
    if final_reward <= initial_reward:
        fast.free()
        target.free()
        raise Error("ERROR: ES did not improve reward toward the target.")

    # 2. Improvement must be sustained, not a single lucky step. We require at
    #    least a third of iterations to improve: a degenerate loop plateaus
    #    almost immediately, while a working optimizer climbs steadily before
    #    jittering near the optimum.
    if improvements * 3 < iters:
        fast.free()
        target.free()
        raise Error(
            "ERROR: ES reward did not improve steadily across iterations."
        )

    # 3. The objective (discrete) reward must be (near) solved.
    if match_frac < 0.95:
        fast.free()
        target.free()
        raise Error("ERROR: ES did not converge to an exact grid match.")

    fast.free()
    target.free()
    print("ES convergence test passed.")
