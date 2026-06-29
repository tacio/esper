from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_forward_learning.mojo`.
from hope import (
    HopeArena,
    build_node,
    ArcGrid,
    ArcTaskPair,
    OP_DIM,
    seed_identity_operator,
    apply_operator,
)
from esper_evolution import forward_with_learning
from arc_io import exact_match

# ==========================================================================
# End-to-end nested-learning test (first true generalization check).
#
# Build an arena-hosted node whose fast weights are an OP_DIM operator vector,
# seed it (and the slow prior) to identity, then `forward_with_learning` fits the
# operator to flip_h demonstrations and applies it to a HELD-OUT test input. The
# result must match the true flip of that unseen input — proving the engine
# learned a transferable transformation, not a memorized grid.
# ==========================================================================


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


def flip_h_grid(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            out.set(r, c, g.get(r, g.cols - 1 - c))
    return out^


def main() raises:
    seed(0)

    var rows = 4
    var cols = 4

    var arena = HopeArena(1 << 16)
    var node = build_node(arena, OP_DIM, OP_DIM)
    seed_identity_operator(node[].slow)
    seed_identity_operator(node[].fast)

    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rows, cols)
        var gout = flip_h_grid(gin)
        demos.append(ArcTaskPair(gin^, gout^))

    # Fit ONCE on the demos (this adapts node.fast), then generalize to several
    # held-out inputs by applying the now-learned operator — fit-once / reuse.
    var n = rows * cols
    var first_in = rand_grid(rows, cols)
    var first_truth = flip_h_grid(first_in)
    var first_result = forward_with_learning(node, demos, first_in)
    var match_sum = exact_match(first_result.data, first_truth.data, n)

    var trials = 6
    for _ in range(trials - 1):
        var test_in = rand_grid(rows, cols)
        var truth = flip_h_grid(test_in)
        var pred = alloc[Float32](n)
        apply_operator(node[].fast, test_in.data, pred, rows, cols)
        match_sum += exact_match(pred, truth.data, n)
        pred.free()
    var held_out = match_sum / Float32(trials)

    print("Held-out exact match:", held_out)

    if held_out < 0.95:
        raise Error(
            "ERROR: forward_with_learning did not generalize the learned"
            " transform to held-out inputs."
        )

    print("Forward-learning generalization test passed.")
