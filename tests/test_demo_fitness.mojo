from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_demo_fitness.mojo`.
from hope import (
    ArcGrid,
    ArcTaskPair,
    OP_DIM,
    apply_operator,
    seed_identity_operator,
)
from esper_evolution import ESWorkspace, fit_operator, operator_fitness
from arc_io import exact_match

# ==========================================================================
# Demonstration-driven learning test (the keystone: real fitness, not the old
# memorization surrogate).
#
# Starting from an identity operator, the annealed ES must fit `flip_h` purely
# from input->output demonstration pairs, then GENERALIZE: applying the fitted
# operator to held-out inputs it never saw must reproduce the flip. This is
# uncheatable by memorization (the held-out grids are not in the demos). The
# operator's bilinear gather makes the fitness smooth so the ES has a real
# gradient; the annealed sigma is what pins the parameters onto the exact flip.
# ==========================================================================


def flip_h_grid(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            out.set(r, c, g.get(r, g.cols - 1 - c))
    return out^


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


def main() raises:
    seed(0)

    var rows = 4
    var cols = 4
    var n = rows * cols

    # Random demonstration grids (random so flip_h is uniquely identifiable —
    # structured ramps have affine symmetries that make it ambiguous).
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rows, cols)
        var gout = flip_h_grid(gin)
        demos.append(ArcTaskPair(gin^, gout^))

    # Fast weights start at identity; slow weights are the identity prior/anchor.
    var fast = alloc[Float32](OP_DIM)
    var slow = alloc[Float32](OP_DIM)
    seed_identity_operator(fast)
    seed_identity_operator(slow)

    var workspace = ESWorkspace(OP_DIM, n)

    var init_fit = operator_fitness(
        fast, slow, demos, workspace.op_output, Float32(0.0001)
    )

    # Annealed in-context fit (the shared recipe).
    fit_operator(
        fast,
        workspace,
        slow,
        demos,
        128,  # N samples
        Float32(0.1),  # alpha0
        Float32(0.003),  # alpha1
        Float32(0.3),  # sigma0
        Float32(0.01),  # sigma1
        4000,  # iters
        Float32(0.0001),  # reg_lambda
    )

    var final_fit = operator_fitness(
        fast, slow, demos, workspace.op_output, Float32(0.0001)
    )

    # Held-out generalization: grids the operator never trained on.
    var match_sum = Float32(0.0)
    var trials = 10
    for _ in range(trials):
        var test_in = rand_grid(rows, cols)
        var test_out = flip_h_grid(test_in)
        var pred = alloc[Float32](n)
        apply_operator(fast, test_in.data, pred, rows, cols)
        match_sum += exact_match(pred, test_out.data, n)
        pred.free()
    var held_out = match_sum / Float32(trials)

    print("Initial fitness     :", init_fit)
    print("Final   fitness     :", final_fit)
    print("Held-out exact match:", held_out)

    if final_fit <= init_fit:
        raise Error("ERROR: ES did not improve operator fitness on the demos.")
    if held_out < 0.95:
        raise Error(
            "ERROR: fitted operator did not generalize flip_h to held-out"
            " inputs (it memorized or failed to learn the transform)."
        )

    fast.free()
    slow.free()
    print("Demo-fitness learning test passed.")
