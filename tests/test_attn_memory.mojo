from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_attn_memory.mojo`.
from hope import ArcGrid, ArcTaskPair
from memory import Memory
from memory_es import AttnGatherMemory, ATTN_DIM
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
# B3 proof: emergent GLOBAL ADDRESSING. Geometry (flip_h/flip_v/transpose) is a
# global coordinate permutation that the per-cell MLP (B1) cannot express. Here a
# learned position-attention gather (AttnGatherMemory — a 2x2 coord projection + a
# learned temperature, NO hand-coded affine apply) re-earns it: each output cell
# reads from all input cells weighted by coordinate similarity, the softmax
# sharpening toward a one-hot gather as the ES grows beta. Fit on random demos,
# score on HELD-OUT grids. Each transform must reach >= 0.95 held-out. recolor is
# deliberately out of scope (geometry-only; the value "local map" is a follow-up).
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
            else:  # transpose (square)
                out.set(r, c, g.get(c, r))
    return out^


# Fit `name` on random demos via the standard annealed ES, return mean held-out
# exact match on unseen grids (the fit never sees the test grids).
def learn_and_eval(name: String, rows: Int, cols: Int) raises -> Float32:
    var n = rows * cols
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rows, cols)
        var gout = apply_transform(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))

    var fast = alloc[Float32](ATTN_DIM)
    var slow = alloc[Float32](ATTN_DIM)
    AttnGatherMemory.seed(fast)
    AttnGatherMemory.seed(slow)
    var ws = ESWorkspace[AttnGatherMemory](n)
    fit_operator[AttnGatherMemory](
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
        AttnGatherMemory.apply(fast, test_in, pred)
        match_sum += exact_match(pred, truth.data, n)
        pred.free()

    fast.free()
    slow.free()
    return match_sum / Float32(trials)


def main() raises:
    seed(0)

    var names = List[String]()
    names.append("flip_h")
    names.append("flip_v")
    names.append("transpose")

    var solved = 0
    for i in range(len(names)):
        var held_out = learn_and_eval(names[i], 4, 4)
        print("  ", names[i], " held-out:", held_out)
        if held_out >= 0.95:
            solved += 1

    if solved != len(names):
        raise Error(
            "ERROR: emergent attention did not re-earn the geometry subset to"
            " >= 0.95 held-out ("
            + String(solved)
            + "/"
            + String(len(names))
            + " transforms)."
        )

    print(
        "Attention-memory test passed: geometry re-earned emergently (global"
        " addressing, no affine)."
    )
