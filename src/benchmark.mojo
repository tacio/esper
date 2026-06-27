from std.sys import argv
from std.memory import alloc, UnsafePointer
from std.random import seed

from arc_io import load_arc_grid, calculate_fitness, exact_match
from esper_evolution import ESWorkspace, evolve_fast_weights

# ==========================================================================
# Esper objective-reward benchmark harness.
#
# Each argument is a path to a compiled target grid (`*_out.bin`, produced by
# `src/synth_tasks.py`). For every task we evolve a fresh fast-weight buffer to
# fit that target, then report two objective rewards:
#   * reward  = negative MSE   (continuous; closer to 0 is better)
#   * match%  = exact_match    (discrete; fraction of cells correct after round)
# A task counts as solved when match% >= SOLVE_THRESHOLD. The aggregate solve
# rate is the headline objective metric for the engine.
#
# Run from the project root, e.g.:
#   python src/synth_tasks.py --transform flip_h --out data_bin --count 8
#   mojo run -I src src/benchmark.mojo data_bin/flip_h_*_out.bin
# ==========================================================================

comptime SOLVE_THRESHOLD = Float32(0.99)
comptime ES_SAMPLES = 96
comptime ES_SIGMA = Float32(0.1)
comptime ES_ITERS = 150


def solve_task(target_path: String) raises -> Float32:
    """Evolve fast weights to fit one target grid; return its exact-match fraction.
    """
    var target = load_arc_grid(target_path)
    var size = target.size()
    var tptr = target.data

    # Fresh fast weights start at zero (no memory carried between tasks).
    var fast = alloc[Float32](size)
    for i in range(size):
        fast[i] = 0.0

    var workspace = ESWorkspace(size)

    # Scale the learning rate with grid size so the effective step
    # (~2 * alpha / size) stays in a stable range across grid dimensions.
    var alpha = Float32(0.1) * Float32(size)

    for _ in range(ES_ITERS):
        evolve_fast_weights(fast, workspace, tptr, ES_SAMPLES, alpha, ES_SIGMA)

    var reward = calculate_fitness(fast, tptr, size)
    var match_frac = exact_match(fast, tptr, size)

    print("  task:", target_path)
    print("    reward (-MSE):", reward, " match%:", match_frac * 100.0)

    fast.free()
    return match_frac


def main() raises:
    seed(0)

    var args = argv()
    if len(args) < 2:
        print(
            "Usage: mojo run -I src src/benchmark.mojo <target_grid.bin>"
            " [more.bin ...]"
        )
        print("Generate tasks first, e.g.:")
        print(
            "  python src/synth_tasks.py --transform flip_h --out data_bin"
            " --count 8"
        )
        return

    var total = len(args) - 1
    var solved = 0

    print("Esper objective-reward benchmark over", total, "task(s)")
    for idx in range(1, len(args)):
        var match_frac = solve_task(String(args[idx]))
        if match_frac >= SOLVE_THRESHOLD:
            solved += 1

    var solve_rate = Float32(solved) / Float32(total) * 100.0
    print("--------------------------------------------------")
    print("Solved", solved, "/", total, " (solve rate:", solve_rate, "% )")

    # Non-zero exit (via raised error) if nothing solved, so the harness/CI
    # treats a fully-failing benchmark as a failure.
    if solved == 0:
        raise Error(
            "ERROR: benchmark solved 0 tasks; the ES loop is not learning."
        )
