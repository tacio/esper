# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.math import round
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_local_write.mojo`.
from hope import ArcGrid, ArcTaskPair, COLOR_DIM
from memory_composed import (
    LocalWriteComposedMemory,
    GeomColorComposedMemory,
    LOCALWRITE_DIM,
    LOCALWRITE_OFF,
    LOCALWRITE_TABLE,
)
from esper_evolution import (
    fit_local,
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
# RUNG A BUILD (Approach 1b) — a LOCAL content-conditioned write composed on the
# gather. The Rung A audit found the same-shape near-misses are dominated by
# UNDER-APPLICATION: the copy-gather+colour memory outputs identity where the
# true rule makes a LOCAL change (fill background, recolor an edge). This proves
# LocalWriteComposedMemory expresses that class through a written signature table
# (centre colour, #Moore-8 neighbours differing from centre) composed on top of
# the proven GeomColor gather, cold and held-out.
#
#   Ckpt A — {outline, fill_enclosed} each >= 0.95 held-out at a FRESH grid,
#            per-task cold. `outline` recolors every cell that borders a
#            different colour (the edge / object-colour class); `fill_enclosed`
#            fills enclosed background cells (the background-fill class).
#   Control (ablation) — the SAME fitted state through GeomColor ONLY (no local
#            write) fails outline: the local module is load-bearing.
#   Control (strict superset) — a PURE recolor task writes NO local table (all
#            sentinel) AND stays >= 0.95: byte-identical to the GeomColor path.
#   Few-demo — outline at n=3 (the corpus median) clears the bar.
# ==========================================================================


# Random integer in [lo, hi].
def ri(lo: Int, hi: Int) -> Int:
    return lo + Int(random_float64(0.0, Float64(hi - lo + 1)))


# A structured "blocky" grid: a background of 0 with a few solid rectangles from
# a small palette, and a few punched single-cell holes (enclosed background).
# Local-neighbourhood rules are IDENTIFIABLE on such grids (interiors, edges,
# enclosed cells all occur) — unlike uniform-random grids where every cell
# borders a different colour.
def blocky_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = 0.0
    for _ in range(ri(2, 4)):
        var col = ri(1, 2)
        var h = ri(2, rows - 2)
        var w = ri(2, cols - 2)
        var r0 = ri(0, rows - h)
        var c0 = ri(0, cols - w)
        for r in range(r0, r0 + h):
            for c in range(c0, c0 + w):
                g.data[r * cols + c] = Float32(col)
    # Punch enclosed holes: a cell whose 8 neighbours are all non-background.
    for _ in range(ri(1, 3)):
        var r = ri(1, rows - 2)
        var c = ri(1, cols - 2)
        var ok = True
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if g.data[(r + dr) * cols + (c + dc)] == 0.0:
                    ok = False
        if ok:
            g.data[r * cols + c] = 0.0
    return g^


# Toroidal count of Moore-8 neighbours differing from the centre (the same
# signature key LocalWriteComposedMemory uses — the ground truth to rediscover).
def diff_count(g: ArcGrid, r: Int, c: Int) -> Int:
    var rows = g.rows
    var cols = g.cols
    var ctr = Int(round(g.data[r * cols + c]))
    var k = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            var rr = (r + dr + rows) % rows
            var cc = (c + dc + cols) % cols
            if Int(round(g.data[rr * cols + cc])) != ctr:
                k += 1
    return k


# The ground-truth LOCAL transforms (all same-shape, signature-expressible).
def apply_local(name: String, g: ArcGrid) -> ArcGrid:
    var rows = g.rows
    var cols = g.cols
    var out = ArcGrid(rows, cols)
    for r in range(rows):
        for c in range(cols):
            var k = r * cols + c
            var d = diff_count(g, r, c)
            var v = g.data[k]
            if name == "outline":
                if d >= 1:
                    v = 9.0
            elif name == "fill_enclosed":
                # Fully-enclosed background (all 8 neighbours non-background) —
                # a single well-covered signature, (0, 8).
                if Int(round(g.data[k])) == 0 and d == 8:
                    v = 8.0
            elif name == "recolor":
                v = Float32((Int(round(g.data[k])) + 1) % COLOR_DIM)
            out.data[k] = v
    return out^


def make_demos(name: String, n: Int, rows: Int, cols: Int) -> List[ArcTaskPair]:
    var demos = List[ArcTaskPair]()
    for _ in range(n):
        var gin = blocky_grid(rows, cols)
        var gout = apply_local(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))
    return demos^


def eval_held_out(
    name: String,
    state: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    ablate: Bool,
) raises -> Float32:
    var gin = blocky_grid(rows, cols)
    var truth = apply_local(name, gin)
    var pred = alloc[Float32](rows * cols)
    if ablate:
        # GeomColor prefix only (no local override) — the ablation control.
        GeomColorComposedMemory.apply(state, gin, pred)
    else:
        LocalWriteComposedMemory.apply(state, gin, pred)
    var m = exact_match(pred, truth.data, rows * cols)
    pred.free()
    return m


# Count written (non-sentinel) local-table entries — 0 ⇒ byte-identical to the
# GeomColor path (the strict-superset check).
def n_written(state: UnsafePointer[Float32, MutAnyOrigin]) -> Int:
    var c = 0
    for s in range(LOCALWRITE_TABLE):
        if state[LOCALWRITE_OFF + s] >= Float32(0.0):
            c += 1
    return c


def fit_task(
    name: String, task_seed: Int, n: Int, rows: Int, cols: Int
) raises -> UnsafePointer[Float32, MutAnyOrigin]:
    seed(task_seed)
    var demos = make_demos(name, n, rows, cols)
    var state = alloc[Float32](LOCALWRITE_DIM)
    LocalWriteComposedMemory.seed(state)
    fit_local(
        state,
        demos,
        rows * cols,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )
    return state


def main() raises:
    comptime R = 11
    comptime C = 11
    print("== Rung A build (1b): local content-conditioned write ==")

    # Ckpt A — the local classes, cold, held-out at a fresh grid.
    var names = List[String]()
    names.append("outline")
    names.append("fill_enclosed")
    for i in range(len(names)):
        var st = fit_task(names[i], i + 1, 8, R, C)
        var ho = eval_held_out(names[i], st, R, C, False)
        print("  ", names[i], " held-out:", ho, " written:", n_written(st))
        if ho < 0.95:
            st.free()
            raise Error("FAIL: " + names[i] + " held-out < 0.95")
        # Ablation control on outline: GeomColor only must fail.
        if names[i] == "outline":
            var abl = eval_held_out(names[i], st, R, C, True)
            print(
                "   control (ablation, GeomColor only) outline held-out:", abl
            )
            if abl >= 0.95:
                st.free()
                raise Error(
                    "FAIL: ablation did not degrade — local not load-bearing"
                )
        st.free()

    # Strict superset — a pure recolor writes NO local table and stays high.
    var sr = fit_task("recolor", 7, 8, R, C)
    var rho = eval_held_out("recolor", sr, R, C, False)
    var nw = n_written(sr)
    print("  recolor (strict superset) held-out:", rho, " written:", nw)
    if rho < 0.95 or nw != 0:
        sr.free()
        raise Error(
            "FAIL: recolor superset broken (held-out or non-empty table)"
        )
    sr.free()

    # Few-demo — outline at the corpus median n=3.
    var sf = fit_task("outline", 11, 3, R, C)
    var fho = eval_held_out("outline", sf, R, C, False)
    print("  outline n=3 (few-demo) held-out:", fho)
    if fho < 0.9:
        sf.free()
        raise Error("FAIL: outline n=3 held-out < 0.9")
    sf.free()

    print("PASS: local-write composition proven cold + controls")
