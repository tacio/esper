# suite-tier: full
#   Heavy cold meta-fit milestone proof. Deferred from the `fast` tier — the
#   meta_fit_selfmod core is already covered there by test_selfmod_memory.
#   See run_tests.sh / CLAUDE.md "Testing".
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List, InlineArray

# Run from the project root: `mojo run -I src tests/test_grid_nbhd_selfmod.mojo`.
from hope import ArcGrid, ArcTaskPair, Task
from memory import (
    GridNbhdSelfModMemory,
    GRIDNBHD_A,
    GRIDNBHD_DE,
    GRIDNBHD_DK,
    GRIDNBHD_NBRS,
    GRIDNBHD_SLOW_DIM,
    GRIDNBHD_E_OFF,
    GRIDNBHD_G_OFF,
    GRIDNBHD_C_OFF,
    GRIDNBHD_LO_OFF,
    GRIDNBHD_HI_OFF,
    GRIDNBHD_BETA_OFF,
    GRIDNBHD_BALPHA_OFF,
    _gridctx_sym,
)
from esper_evolution import meta_fit_selfmod
from arc_io import exact_match

# ==========================================================================
# Richer neighbourhoods + NONLINEAR read (ARC-AGI-2 block 2). GridContext reads
# pred = S·k (LINEAR) and expresses only additive positional rules. This memory
# keys on the Moore-8 neighbour-count histogram and reads through a sigmoid
# THRESHOLD, so it expresses the DISJUNCTIVE / COUNT class:
#   out[r,c] = C1 if (# Moore-8 neighbours == P) >= t else C2      (toroidal)
# provably outside GridContext's additive class (a linear read of a count cannot
# be a sharp step) and outside any per-cell / 1-D memory (it needs all 8
# neighbours). At t in {2,3} with A=5 colours the two classes are ~balanced.
#
# Ckpt A: FIXED (scaled one-hot) key + hand-set nonlinear read; the gated delta
# self-write learns one such rule to >= 0.95 held-out. CONTROL: the same write
# with a LINEAR (identity, clamped) read fails (< 0.6) — proof the nonlinearity
# does the work, not scaffolding. Fail-fast gate before the meta-fit.
# ==========================================================================

comptime A = GRIDNBHD_A
comptime R = 6
comptime C = 6
comptime T = 2  # threshold; t=2 gives ~balanced classes (majority baseline ~0.52)
comptime GridTask = Task[ArcGrid]


def rand_grid() -> ArcGrid:
    var g = ArcGrid(R, C)
    for k in range(R * C):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(A))))
    return g^


def count_match(g: ArcGrid, r: Int, c: Int, p: Int) -> Int:
    var rows = g.rows
    var cols = g.cols
    var cnt = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            var rr = (r + dr + rows) % rows
            var cc = (c + dc + cols) % cols
            if _gridctx_sym(g.data[rr * cols + cc]) == p:
                cnt += 1
    return cnt


# out = C1 if (#Moore-8 neighbours == p) >= t else C2.
def apply_rule(g: ArcGrid, p: Int, t: Int, c1: Float32, c2: Float32) -> ArcGrid:
    var o = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            o.data[r * g.cols + c] = c1 if count_match(g, r, c, p) >= t else c2
    return o^


def make_task(
    p: Int, t: Int, c1: Float32, c2: Float32, nt: Int, nv: Int
) -> GridTask:
    var train = List[ArcTaskPair]()
    for _ in range(nt):
        var g = rand_grid()
        train.append(ArcTaskPair(g^, apply_rule(g, p, t, c1, c2)))
    var test = List[ArcTaskPair]()
    for _ in range(nv):
        var g = rand_grid()
        test.append(ArcTaskPair(g^, apply_rule(g, p, t, c1, c2)))
    return GridTask(train^, test^)


# The fixed-parameter Ckpt-A config validated by the offline probe: scaled one-hot
# embeddings (De == A), moderate sharpness, low decay, output levels 0/4.
def set_fixed(slow: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(GRIDNBHD_SLOW_DIM):
        slow[i] = 0.0
    for cc in range(A):
        slow[GRIDNBHD_E_OFF + cc * GRIDNBHD_DE + cc] = 3.0
    slow[GRIDNBHD_G_OFF] = 2.0
    slow[GRIDNBHD_C_OFF] = 0.0
    slow[GRIDNBHD_LO_OFF] = 0.0
    slow[GRIDNBHD_HI_OFF] = 4.0
    slow[GRIDNBHD_BETA_OFF] = 0.0  # eta ~ 0.5
    slow[GRIDNBHD_BALPHA_OFF] = -6.0  # alpha ~ 0.002


# A LINEAR-read control sharing the exact centre-free key + gated delta write, but
# R = identity (clamped to [0,A-1]). A linear read of a count cannot be a sharp
# step, so a threshold rule must fail — isolating the nonlinearity as the cause.
def heldout_linear_control(p: Int, t: Int) raises -> Float32:
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid()
        demos.append(ArcTaskPair(gin^, apply_rule(gin, p, t, 4.0, 0.0)))
    var slow = alloc[Float32](GRIDNBHD_SLOW_DIM)
    for i in range(GRIDNBHD_SLOW_DIM):
        slow[i] = 0.0
    for cc in range(A):
        slow[GRIDNBHD_E_OFF + cc * GRIDNBHD_DE + cc] = 3.0
    var eta = Float32(0.3)
    var state = alloc[Float32](GRIDNBHD_DK)
    for j in range(GRIDNBHD_DK):
        state[j] = 0.0
    for _epoch in range(6):
        for d in range(len(demos)):
            ref pair = demos[d]
            for r in range(R):
                for c in range(C):
                    var nbrs = InlineArray[Int, GRIDNBHD_NBRS](fill=0)
                    GridNbhdSelfModMemory._gather8(
                        pair.input_grid.data, R, C, r, c, nbrs
                    )
                    var k = InlineArray[Float32, GRIDNBHD_DK](fill=0.0)
                    GridNbhdSelfModMemory._key(slow, nbrs, k)
                    var z = Float32(0.0)
                    for j in range(GRIDNBHD_DK):
                        z += state[j] * k[j]
                    var e = pair.output_grid.data[r * C + c] - z
                    for j in range(GRIDNBHD_DK):
                        state[j] += eta * e * k[j]
    var ms = Float32(0.0)
    var trials = 8
    var n = R * C
    for _ in range(trials):
        var tg = rand_grid()
        var truth = apply_rule(tg, p, t, 4.0, 0.0)
        var pred = alloc[Float32](n)
        for r in range(R):
            for c in range(C):
                var nbrs = InlineArray[Int, GRIDNBHD_NBRS](fill=0)
                GridNbhdSelfModMemory._gather8(tg.data, R, C, r, c, nbrs)
                var k = InlineArray[Float32, GRIDNBHD_DK](fill=0.0)
                GridNbhdSelfModMemory._key(slow, nbrs, k)
                var z = Float32(0.0)
                for j in range(GRIDNBHD_DK):
                    z += state[j] * k[j]
                if z < 0.0:
                    z = 0.0
                elif z > Float32(A - 1):
                    z = Float32(A - 1)
                pred[r * C + c] = z
        ms += exact_match(pred, truth.data, n)
        pred.free()
    state.free()
    slow.free()
    return ms / Float32(trials)


# One (multi-epoch) adapt pass on a fresh rule, then held-out on unseen grids.
def fresh_heldout(
    slow: UnsafePointer[Float32, MutAnyOrigin], p: Int, t: Int
) raises -> Float32:
    var task = make_task(p, t, 4.0, 0.0, 8, 8)
    var state = alloc[Float32](GRIDNBHD_DK)
    GridNbhdSelfModMemory.adapt(slow, task.train, state)
    var ms = Float32(0.0)
    var n = R * C
    for j in range(len(task.test)):
        var pred = alloc[Float32](n)
        GridNbhdSelfModMemory.apply(slow, state, task.test[j].input_grid, pred)
        ms += exact_match(pred, task.test[j].output_grid.data, n)
        pred.free()
    state.free()
    return ms / Float32(len(task.test))


def main() raises:
    seed(0)
    var n = R * C

    # ---------- Checkpoint A: fixed-key nonlinear write + linear control ----------
    var slow = alloc[Float32](GRIDNBHD_SLOW_DIM)
    set_fixed(slow)
    var demos = List[ArcTaskPair]()
    for _ in range(12):
        var gin = rand_grid()
        demos.append(ArcTaskPair(gin^, apply_rule(gin, 2, T, 4.0, 0.0)))
    var state = alloc[Float32](GRIDNBHD_DK)
    GridNbhdSelfModMemory.adapt(slow, demos, state)
    var match_sum = Float32(0.0)
    var trials = 12
    for _ in range(trials):
        var tg = rand_grid()
        var truth = apply_rule(tg, 2, T, 4.0, 0.0)
        var pred = alloc[Float32](n)
        GridNbhdSelfModMemory.apply(slow, state, tg, pred)
        match_sum += exact_match(pred, truth.data, n)
        pred.free()
    var nonlin = match_sum / Float32(trials)
    state.free()
    slow.free()
    print(
        "  ckptA nonlinear self-write held-out (count-threshold rule):", nonlin
    )

    seed(0)
    var lin = heldout_linear_control(2, T)
    print("  ckptA LINEAR-read control (same write, identity+clamp):", lin)

    if nonlin < 0.95:
        raise Error(
            "ERROR: the nonlinear Moore-8 gated-delta self-write did not learn"
            " the count-threshold rule to >= 0.95 held-out (got "
            + String(nonlin)
            + ")."
        )
    # The control is the STRONGEST linear read (a clamped ramp): on a balanced
    # threshold it approximates the step and reaches ~0.6, but cannot round the
    # boundary counts correctly — far short of the bar and of the nonlinear 0.995.
    # The decisive signal is the gap; require the linear read to stay clearly
    # short of the 0.95 bar.
    if lin >= 0.75:
        raise Error(
            "ERROR: the LINEAR-read control unexpectedly approached the bar"
            " (got "
            + String(lin)
            + ") — the nonlinearity is not the load-bearing part; claim"
            " vacuous."
        )
    print(
        (
            "Grid-neighbourhood checkpoint A passed: the nonlinear read + gated"
            " delta self-write learned a Moore-8 count-threshold rule to"
        ),
        nonlin,
        "held-out, where the strongest linear read reaches only",
        lin,
        "(cannot sharply threshold a count).",
    )

    # ---------- Checkpoint B: meta-learn the projections (emergent) ----------
    seed(0)
    var slow_b = alloc[Float32](GRIDNBHD_SLOW_DIM)
    GridNbhdSelfModMemory.seed_slow(slow_b)

    var before = Float32(0.0)
    for i in range(5):
        var p = Int(random_float64(0.0, Float64(A)))
        before += fresh_heldout(slow_b, p, T)
    before = before / 5.0
    print("  ckptB fresh held-out BEFORE meta-fit (generic seed):", before)

    var meta = List[GridTask]()
    for _ in range(8):
        var p = Int(random_float64(0.0, Float64(A)))
        meta.append(make_task(p, T, 4.0, 0.0, 8, 4))
    # Budget trimmed (N, iters) to keep the full suite under the 10-min line: the
    # before/after gap is large (0.0 -> 1.0 at N=96/1500 iters), leaving ample head-
    # room. See docs/JOURNAL.md and CLAUDE.md "Testing" (this test is full-tier).
    meta_fit_selfmod[GridNbhdSelfModMemory](
        slow_b, meta, n, 64, 0.1, 0.003, 0.5, 0.01, 1000
    )

    var after = Float32(0.0)
    var n_fresh = 8
    for i in range(n_fresh):
        var p = Int(random_float64(0.0, Float64(A)))
        after += fresh_heldout(slow_b, p, T)
    after = after / Float32(n_fresh)
    print(
        "  ckptB fresh held-out AFTER meta-fit (one-pass, 8 unseen rules):",
        after,
    )
    slow_b.free()

    if after < 0.95:
        raise Error(
            "ERROR: the meta-learned grid-neighbourhood memory did not solve a"
            " fresh count-threshold rule to >= 0.95 held-out in one adapt pass"
            " (got "
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
        "Grid-neighbourhood checkpoint B passed: the self-modifying memory"
        " meta-learned its neighbourhood embeddings + nonlinear read cold; a"
        " fresh disjunctive/count rule adapts in ONE pass (a linear / per-cell"
        " / 1-D memory cannot express it)."
    )
