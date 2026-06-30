from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_generalization.mojo`.
from hope import (
    ArcGrid,
    ArcTaskPair,
    OP_DIM,
    apply_operator,
    seed_identity_operator,
)
from memory import OperatorMemory
from esper_evolution import (
    ESWorkspace,
    fit_operator,
    FIT_N,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_ITERS,
    FIT_REG,
)
from arc_io import exact_match

# ==========================================================================
# Whole-expressible-subset generalization test (the M7 milestone proof).
#
# For each transform in {flip_h, flip_v, transpose (square), recolor}, fit the
# operator on random demonstration pairs, then score it on HELD-OUT inputs it
# never saw. Every transform must reach >= 0.95 held-out exact match, and the
# aggregate solve rate must be 100%. This proves the engine learns the *whole*
# expressible subset from demonstrations, not just flip_h. Self-contained (no
# generated files) — `src/arc_solve.mojo` separately exercises the bundle/loader
# path on disk.
# ==========================================================================


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


def apply_transform(name: String, g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            if name == "flip_h":
                out.set(r, c, g.get(r, g.cols - 1 - c))
            elif name == "flip_v":
                out.set(r, c, g.get(g.rows - 1 - r, c))
            elif name == "transpose":
                out.set(r, c, g.get(c, r))
            else:  # recolor
                out.set(r, c, Float32((Int(g.get(r, c)) + 1) % 10))
    return out^


# Fit `name` on random demos, return the average held-out exact match.
def learn_and_eval(name: String, rows: Int, cols: Int) raises -> Float32:
    var n = rows * cols
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rows, cols)
        var gout = apply_transform(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))

    var fast = alloc[Float32](OP_DIM)
    var slow = alloc[Float32](OP_DIM)
    seed_identity_operator(fast)
    seed_identity_operator(slow)
    var ws = ESWorkspace[OperatorMemory](n)
    fit_operator[OperatorMemory](
        fast,
        ws,
        slow,
        demos,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )

    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var test_in = rand_grid(rows, cols)
        var truth = apply_transform(name, test_in)
        var pred = alloc[Float32](n)
        apply_operator(fast, test_in.data, pred, rows, cols)
        match_sum += exact_match(pred, truth.data, n)
        pred.free()

    fast.free()
    slow.free()
    return match_sum / Float32(trials)


def main() raises:
    seed(0)

    # transpose uses a square grid so it stays same-shape.
    var names = List[String]()
    names.append("flip_h")
    names.append("flip_v")
    names.append("transpose")
    names.append("recolor")

    var solved = 0
    for i in range(len(names)):
        var held_out = learn_and_eval(names[i], 4, 4)
        print("  ", names[i], " held-out:", held_out)
        if held_out >= 0.95:
            solved += 1

    if solved != len(names):
        raise Error(
            "ERROR: the engine did not learn the whole expressible subset to"
            " >= 0.95 held-out ("
            + String(solved)
            + "/"
            + String(len(names))
            + " transforms)."
        )

    print("Generalization test passed: full expressible subset learned.")
