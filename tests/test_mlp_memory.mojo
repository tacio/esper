from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_mlp_memory.mojo`.
from hope import ArcGrid, ArcTaskPair
from memory_es import MLPMemory, MLP_DIM
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
# First emergent memory test (B1 capstone): a per-cell MLP learns RECOLOR with
# NO hand-coded colour LUT. The memory is a generic 1->H->1 function approximator
# (see MLPMemory); the colour permutation is fit by the same annealed ES that
# fits the structured operator, through the same generic Memory/Domain seam. Fit
# on random demonstration pairs, score on HELD-OUT grids it never saw. This is
# the first real training-wheel removal — the colour structure emerges from data.
# ==========================================================================

comptime ROWS = 4
comptime COLS = 4


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


# recolor: out[r, c] = (in[r, c] + 1) % 10 (the synth recolor; includes the 9->0
# wrap, the hardest point for a smooth approximator).
def recolor(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for k in range(g.rows * g.cols):
        out.data[k] = Float32((Int(g.data[k]) + 1) % 10)
    return out^


def main() raises:
    seed(0)
    var n = ROWS * COLS

    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(ROWS, COLS)
        var gout = recolor(gin)
        demos.append(ArcTaskPair(gin^, gout^))

    var fast = alloc[Float32](MLP_DIM)
    var slow = alloc[Float32](MLP_DIM)
    MLPMemory.seed(fast)
    MLPMemory.seed(slow)

    var ws = ESWorkspace[MLPMemory](n)
    fit_operator[MLPMemory](
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
        var test_in = rand_grid(ROWS, COLS)
        var truth = recolor(test_in)
        var pred = alloc[Float32](n)
        MLPMemory.apply(fast, test_in, pred)
        match_sum += exact_match(pred, truth.data, n)
        pred.free()

    var held_out = match_sum / Float32(trials)
    print("  MLP recolor held-out:", held_out)

    fast.free()
    slow.free()

    if held_out < 0.95:
        raise Error(
            "ERROR: the emergent MLP memory did not learn recolor to >= 0.95"
            " held-out (got "
            + String(held_out)
            + ")."
        )

    print("MLP-memory test passed: recolor learned emergently (no LUT).")
