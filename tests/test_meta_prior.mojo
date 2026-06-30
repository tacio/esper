from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_meta_prior.mojo`.
from hope import (
    ArcGrid,
    ArcTaskPair,
    ArcTask,
    OP_DIM,
    apply_operator,
    seed_identity_operator,
)
from esper_evolution import (
    ESWorkspace,
    fit_operator,
    reptile_meta_train,
    copy_weights,
    FIT_N,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA1,
    FIT_REG,
    META_LR,
    META_ITERS,
    META_FIT_ITERS,
    EVAL_SIGMA0,
    EVAL_ITERS,
)
from arc_io import exact_match

# ==========================================================================
# Meta-prior test (the M9 milestone proof): the slow weights are a LEARNED
# prior, not a fixed identity anchor.
#
# Reptile-meta-train `slow` on many flip_h tasks (random grids, shared
# transform), then fit a FRESH held-out flip_h task at a small fixed budget
# (EVAL_ITERS, well below the ~4000 a cold identity prior needs) from two
# starting priors:
#   - identity prior (cold)         -> identity_ho
#   - the meta-learned prior        -> meta_ho
# The meta prior must reach >= 0.95 held-out and beat the cold prior by a clear
# margin at equal cost — the slow timescale genuinely accelerates the inner
# in-context fit. The fit only ever sees train pairs; held-out is scored after,
# so there is no leak. Self-contained (no generated files).
# ==========================================================================

comptime ROWS = 4
comptime COLS = 4
comptime N_TRAIN = 6


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


# flip_h: out[r, c] = in[r, cols - 1 - c].
def flip_h(g: ArcGrid) -> ArcGrid:
    var out = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            out.set(r, c, g.get(r, g.cols - 1 - c))
    return out^


def flip_h_pair(rows: Int, cols: Int) -> ArcTaskPair:
    var gin = rand_grid(rows, cols)
    var gout = flip_h(gin)
    return ArcTaskPair(gin^, gout^)


def make_flip_h_task(n_train: Int, rows: Int, cols: Int) -> ArcTask:
    var train = List[ArcTaskPair]()
    for _ in range(n_train):
        train.append(flip_h_pair(rows, cols))
    var test = List[ArcTaskPair]()
    test.append(flip_h_pair(rows, cols))
    return ArcTask(train^, test^)


# Fit a fresh operator on `demos` at EVAL_ITERS, anchored to and initialized from
# `slow`, then return the held-out exact match on (test_in -> test_out).
def fit_and_eval(
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    test_in: ArcGrid,
    test_out: ArcGrid,
    mut ws: ESWorkspace,
) raises -> Float32:
    var fast = alloc[Float32](OP_DIM)
    copy_weights(fast, slow, OP_DIM)
    fit_operator(
        fast,
        ws,
        slow,
        demos,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        EVAL_SIGMA0,
        FIT_SIGMA1,
        EVAL_ITERS,
        FIT_REG,
    )
    var n = test_in.rows * test_in.cols
    var pred = alloc[Float32](n)
    apply_operator(fast, test_in.data, pred, test_in.rows, test_in.cols)
    var ho = exact_match(pred, test_out.data, n)
    pred.free()
    fast.free()
    return ho


def main() raises:
    seed(0)

    var capacity = ROWS * COLS
    var ws = ESWorkspace(OP_DIM, capacity)

    # --- Meta-train the slow prior on a family of flip_h tasks. ---
    var meta_tasks = List[ArcTask]()
    for _ in range(8):
        meta_tasks.append(make_flip_h_task(N_TRAIN, ROWS, COLS))

    var slow_meta = alloc[Float32](OP_DIM)
    seed_identity_operator(slow_meta)
    reptile_meta_train(
        slow_meta, meta_tasks, ws, META_ITERS, META_FIT_ITERS, META_LR
    )

    # --- Held-out (unseen-grid) flip_h task, fit at the small fixed budget. ---
    var held = make_flip_h_task(N_TRAIN, ROWS, COLS)

    var slow_identity = alloc[Float32](OP_DIM)
    seed_identity_operator(slow_identity)

    var identity_ho = fit_and_eval(
        slow_identity,
        held.train,
        held.test[0].input_grid,
        held.test[0].output_grid,
        ws,
    )
    var meta_ho = fit_and_eval(
        slow_meta,
        held.train,
        held.test[0].input_grid,
        held.test[0].output_grid,
        ws,
    )

    print("  identity-prior held-out:", identity_ho)
    print("  meta-prior held-out:    ", meta_ho)
    print("  speedup gap:            ", meta_ho - identity_ho)

    slow_meta.free()
    slow_identity.free()

    if meta_ho < 0.95:
        raise Error(
            "ERROR: meta-prior did not fit the held-out task at the reduced"
            " budget (held-out "
            + String(meta_ho)
            + " < 0.95)."
        )
    if meta_ho - identity_ho < 0.3:
        raise Error(
            "ERROR: the meta-learned prior did not beat the identity prior by a"
            " clear margin (gap "
            + String(meta_ho - identity_ho)
            + " < 0.3); the slow timescale is not accelerating the fit."
        )

    print("Meta-prior test passed: the learned slow prior fits faster.")
