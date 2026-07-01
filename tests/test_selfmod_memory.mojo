from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_selfmod_memory.mojo`.
from hope import ArcGrid, ArcTaskPair, ArcTask
from memory import (
    RecolorSelfWrite,
    RecolorSelfModMemory,
    SELFMOD_STATE_DIM,
    SELFMOD_SLOW_DIM,
)
from esper_evolution import meta_fit_selfmod
from arc_io import exact_match

# ==========================================================================
# B4 — self-modifying memory: a learned in-context self-WRITE rule.
#
# Checkpoint 1 (mechanism, fixed projections, no ES): a self-modifying associative
# memory builds the recolor map from the demo (in,out) cells in a SINGLE forward
# pass and reads it on HELD-OUT grids. Validates the self-write generalises.
#
# Checkpoint 2 (emergent, meta-learned read): the read projections (colour
# embeddings + temperature) are META-LEARNED ONCE across a family of random recolor
# permutations by the antithetic parallel ES (the ES fits only the small SLOW
# vector; the fast state is WRITTEN by adapt, never searched). A FRESH unseen
# permutation is then solved to held-out >= 0.95 by a SINGLE adapt pass — vs the
# ES-fit MLPMemory (B1) which needs a ~4000-iter fit per task. Cold meta-fit is the
# bar; a generic seed must FAIL (proving the meta-learning did the work).
# ==========================================================================

comptime R = 4
comptime C = 4


def rand_grid() -> ArcGrid:
    var g = ArcGrid(R, C)
    for k in range(R * C):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


# ---- checkpoint 1 helpers (the fixed +1 recolor) ----
def recolor_inc(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for k in range(g.rows * g.cols):
        out.data[k] = Float32((Int(g.data[k]) + 1) % 10)
    return out^


# ---- checkpoint 2 helpers (arbitrary recolor permutations) ----
def rand_perm(p: UnsafePointer[Int, MutAnyOrigin]):
    for i in range(10):
        p[i] = i
    for i in range(9, 0, -1):
        var j = Int(random_float64(0.0, Float64(i + 1)))
        var tmp = p[i]
        p[i] = p[j]
        p[j] = tmp


def recolor_perm(g: ArcGrid, p: UnsafePointer[Int, MutAnyOrigin]) -> ArcGrid:
    var o = ArcGrid(g.rows, g.cols)
    for k in range(g.rows * g.cols):
        o.data[k] = Float32(p[Int(g.data[k])])
    return o^


def make_task(
    p: UnsafePointer[Int, MutAnyOrigin], n_train: Int, n_test: Int
) -> ArcTask:
    var train = List[ArcTaskPair]()
    for _ in range(n_train):
        var gin = rand_grid()
        train.append(ArcTaskPair(gin^, recolor_perm(gin, p)))
    var test = List[ArcTaskPair]()
    for _ in range(n_test):
        var gin = rand_grid()
        test.append(ArcTaskPair(gin^, recolor_perm(gin, p)))
    return ArcTask(train^, test^)


def fresh_heldout(slow: UnsafePointer[Float32, MutAnyOrigin]) raises -> Float32:
    # A FRESH recolor permutation, unseen grids: one-pass adapt then score held-out.
    var p = alloc[Int](10)
    rand_perm(p)
    var task = make_task(p, 8, 8)
    var state = alloc[Float32](SELFMOD_STATE_DIM)
    RecolorSelfModMemory.adapt(slow, task.train, state)
    var ms = Float32(0.0)
    var n = R * C
    for j in range(len(task.test)):
        var pred = alloc[Float32](n)
        RecolorSelfModMemory.apply(slow, state, task.test[j].input_grid, pred)
        ms += exact_match(pred, task.test[j].output_grid.data, n)
        pred.free()
    state.free()
    p.free()
    return ms / Float32(len(task.test))


def main() raises:
    seed(0)
    var n = R * C

    # ---------- Checkpoint 1: validate the self-write mechanism ----------
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid()
        demos.append(ArcTaskPair(gin^, recolor_inc(gin)))
    var state = alloc[Float32](SELFMOD_STATE_DIM)
    RecolorSelfWrite.adapt(demos, state)
    var ck1 = Float32(0.0)
    for _ in range(8):
        var test_in = rand_grid()
        var truth = recolor_inc(test_in)
        var pred = alloc[Float32](n)
        RecolorSelfWrite.apply(state, test_in, pred)
        ck1 += exact_match(pred, truth.data, n)
        pred.free()
    ck1 = ck1 / 8.0
    state.free()
    print("  ckpt1 self-write held-out (one pass, fixed proj):", ck1)
    if ck1 < 0.95:
        raise Error(
            "ERROR: self-write mechanism failed to generalise recolor (got "
            + String(ck1)
            + ")."
        )

    # ---------- Checkpoint 2: meta-learn the read, prove on a fresh task ----------
    var slow = alloc[Float32](SELFMOD_SLOW_DIM)
    RecolorSelfModMemory.seed_slow(slow)
    var before = fresh_heldout(slow)
    print("  ckpt2 fresh held-out BEFORE meta-fit (generic seed):", before)

    var meta = List[ArcTask]()
    for _ in range(6):
        var p = alloc[Int](10)
        rand_perm(p)
        meta.append(make_task(p, 6, 4))
        p.free()
    meta_fit_selfmod[RecolorSelfModMemory](
        slow, meta, n, 128, 0.1, 0.003, 0.5, 0.01, 2000
    )

    var after = Float32(0.0)
    for _ in range(5):
        after += fresh_heldout(slow)
    after = after / 5.0
    print(
        "  ckpt2 fresh held-out AFTER meta-fit (one-pass, 5 unseen perms):",
        after,
    )
    slow.free()

    if after < 0.95:
        raise Error(
            "ERROR: meta-learned self-modifying read did not solve a fresh"
            " recolor task to >= 0.95 held-out in one pass (got "
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
        "Self-mod memory test passed: the read projections were meta-learned"
        " (cold), and a fresh recolor task adapts in ONE forward pass (no"
        " per-task ES fit)."
    )
