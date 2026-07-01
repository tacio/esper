from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_grid_context_selfmod.mojo`.
from hope import ArcGrid, ArcTaskPair, Task
from memory import (
    GridContextSelfWrite,
    GridContextSelfModMemory,
    GRIDCTX_A,
    GRIDCTX_ONEHOT_STATE,
    GRIDCTX_SLOW_DIM,
    GRIDCTX_DK,
)
from esper_evolution import meta_fit_selfmod
from arc_io import exact_match

# ==========================================================================
# 2-D context keys (first ARC-AGI-2 block) — a self-modifying memory that keys on a
# grid NEIGHBOURHOOD and learns a local pattern->colour rule in-context. Proving
# task: out[r,c] = h1(center, up) + h2(center, left) (toroidal), the ADDITIVE
# center<->neighbour class. Genuinely 2-D — a per-cell MLP, a colour-LUT, and the
# 1-D sequence memory all provably fail it (it needs BOTH the up and left neighbours).
#
# Checkpoint A: validate the grid gated-delta self-write with FIXED one-hot keys
# (no ES). The delta write solves the additive decomposition from the demo cells over
# a few epochs; read on HELD-OUT grids. Fail-fast gate.
# ==========================================================================

comptime A = GRIDCTX_A
comptime R = 4
comptime C = 4
comptime GridTask = Task[ArcGrid]


def rand_grid() -> ArcGrid:
    var g = ArcGrid(R, C)
    for k in range(R * C):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(A))))
    return g^


# A random additive local rule: h1,h2: (center, neighbour) -> {0,1,2}, so
# out = h1 + h2 in [0,4] (no mod-wrap — matches the additive linear read exactly).
def rand_rule(
    h1: UnsafePointer[Int, MutAnyOrigin], h2: UnsafePointer[Int, MutAnyOrigin]
):
    for i in range(A * A):
        h1[i] = Int(random_float64(0.0, 3.0))
        h2[i] = Int(random_float64(0.0, 3.0))


def apply_rule(
    g: ArcGrid,
    h1: UnsafePointer[Int, MutAnyOrigin],
    h2: UnsafePointer[Int, MutAnyOrigin],
) -> ArcGrid:
    var o = ArcGrid(g.rows, g.cols)
    var rows = g.rows
    var cols = g.cols
    for r in range(rows):
        for c in range(cols):
            var center = Int(g.data[r * cols + c])
            var up = Int(g.data[((r - 1 + rows) % rows) * cols + c])
            var left = Int(g.data[r * cols + ((c - 1 + cols) % cols)])
            o.data[r * cols + c] = Float32(
                h1[center * A + up] + h2[center * A + left]
            )
    return o^


def make_task(
    h1: UnsafePointer[Int, MutAnyOrigin],
    h2: UnsafePointer[Int, MutAnyOrigin],
    nt: Int,
    nv: Int,
) -> GridTask:
    var train = List[ArcTaskPair]()
    for _ in range(nt):
        var g = rand_grid()
        train.append(ArcTaskPair(g^, apply_rule(g, h1, h2)))
    var test = List[ArcTaskPair]()
    for _ in range(nv):
        var g = rand_grid()
        test.append(ArcTaskPair(g^, apply_rule(g, h1, h2)))
    return GridTask(train^, test^)


# A FRESH random rule, unseen grids: one (multi-epoch) adapt pass, then held-out.
def fresh_heldout(slow: UnsafePointer[Float32, MutAnyOrigin]) raises -> Float32:
    var h1 = alloc[Int](A * A)
    var h2 = alloc[Int](A * A)
    rand_rule(h1, h2)
    var task = make_task(h1, h2, 8, 8)
    var state = alloc[Float32](GRIDCTX_DK)
    GridContextSelfModMemory.adapt(slow, task.train, state)
    var ms = Float32(0.0)
    var n = R * C
    for j in range(len(task.test)):
        var pred = alloc[Float32](n)
        GridContextSelfModMemory.apply(
            slow, state, task.test[j].input_grid, pred
        )
        ms += exact_match(pred, task.test[j].output_grid.data, n)
        pred.free()
    state.free()
    h1.free()
    h2.free()
    return ms / Float32(len(task.test))


def main() raises:
    seed(0)
    var n = R * C

    var h1 = alloc[Int](A * A)
    var h2 = alloc[Int](A * A)
    rand_rule(h1, h2)

    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid()
        demos.append(ArcTaskPair(gin^, apply_rule(gin, h1, h2)))

    var state = alloc[Float32](GRIDCTX_ONEHOT_STATE)
    GridContextSelfWrite.adapt(demos, state)

    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var t = rand_grid()
        var truth = apply_rule(t, h1, h2)
        var pred = alloc[Float32](n)
        GridContextSelfWrite.apply(state, t, pred)
        match_sum += exact_match(pred, truth.data, n)
        pred.free()
    var held_out = match_sum / Float32(trials)
    print(
        "  ckptA grid 2-D self-write held-out (one-hot key, epochs):", held_out
    )

    state.free()
    h1.free()
    h2.free()

    if held_out < 0.95:
        raise Error(
            "ERROR: the grid 2-D gated-delta self-write did not learn the"
            " additive local-context rule to >= 0.95 held-out (got "
            + String(held_out)
            + ")."
        )

    print(
        "Grid-context checkpoint A passed: the gated delta-rule learned a 2-D"
        " local pattern rule (center + up + left) from the demos and"
        " generalised."
    )

    # ---------- Checkpoint B: meta-learn the projections (emergent) ----------
    # Re-seed so ckpt B's RNG is independent of ckpt A (reproducible in isolation).
    seed(0)
    var slow = alloc[Float32](GRIDCTX_SLOW_DIM)
    GridContextSelfModMemory.seed_slow(slow)

    var before = Float32(0.0)
    for _ in range(5):
        before += fresh_heldout(slow)
    before = before / 5.0
    print("  ckptB fresh held-out BEFORE meta-fit (generic seed):", before)

    var meta = List[GridTask]()
    for _ in range(8):
        var mh1 = alloc[Int](A * A)
        var mh2 = alloc[Int](A * A)
        rand_rule(mh1, mh2)
        meta.append(make_task(mh1, mh2, 6, 4))
        mh1.free()
        mh2.free()
    meta_fit_selfmod[GridContextSelfModMemory](
        slow, meta, n, 96, 0.1, 0.003, 0.5, 0.01, 2500
    )

    var after = Float32(0.0)
    var n_fresh = 8
    for _ in range(n_fresh):
        after += fresh_heldout(slow)
    after = after / Float32(n_fresh)
    print(
        "  ckptB fresh held-out AFTER meta-fit (one-pass, 8 unseen rules):",
        after,
    )
    slow.free()

    if after < 0.95:
        raise Error(
            "ERROR: the meta-learned grid-context memory did not solve a fresh"
            " 2-D local rule to >= 0.95 held-out in one adapt pass (got "
            + String(after)
            + ")."
        )
    if before >= 0.5:
        raise Error(
            "ERROR: the generic seed already generalised (before="
            + String(before)
            + ") — the meta-fit is not doing the work; emergence claim vacuous."
        )

    print(
        "Grid-context checkpoint B passed: the self-modifying memory"
        " meta-learned its 2-D neighbourhood projections cold; a fresh local"
        " rule adapts in ONE pass (a per-cell/1-D memory cannot express it)."
    )
