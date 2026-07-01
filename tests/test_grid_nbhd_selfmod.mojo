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
    GRIDNBHD_STATE_DIM,
    GRIDNBHD_SLOW_DIM,
    GRIDNBHD_E_OFF,
    GRIDNBHD_G_OFF,
    GRIDNBHD_C_OFF,
    GRIDNBHD_BETA_OFF,
    GRIDNBHD_BALPHA_OFF,
    _gridctx_sym,
)
from esper_evolution import meta_fit_selfmod
from arc_io import exact_match

# ==========================================================================
# Broadening the nonlinear class (ARC-AGI-2 block 3). Block 2 proved the
# disjunctive/count class but with the output colours (fixed {0,4}) and threshold
# (t=2) baked into meta params. This generalised memory infers the WHOLE 2-level
# rule from the demos:
#   out[r,c] = C1 if (# Moore-8 neighbours == P) >= t else C2   (toroidal)
# with P, t, AND both output colours C1/C2 all varying per task. The read is still
# a single sigmoid THRESHOLD on the neighbour-count histogram; the trick is
# DECOUPLING — the two colours are read off the demos (v0/v1 = min/max output,
# WRITTEN), and the salience is trained as a binary classifier of which colour a
# cell outputs, so its learned sign handles inverted rules (fire -> smaller colour)
# and the bias slot self-calibrates any threshold t.
#
# Ckpt A: FIXED (scaled one-hot) key + hand-set read; the self-write learns
# ARBITRARY rules (arbitrary colours incl. inverted, t in {2,3}) each to >= 0.95
# held-out. CONTROL: a LINEAR read (identity, clamped) fails (< 0.75). Fail-fast.
# ==========================================================================

comptime A = GRIDNBHD_A
comptime R = 6
comptime C = 6
comptime GridTask = Task[ArcGrid]


def rand_grid() -> ArcGrid:
    var g = ArcGrid(R, C)
    for k in range(R * C):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(A))))
    return g^


def count_match(g: ArcGrid, r: Int, c: Int, p: Int) -> Int:
    var cnt = 0
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            if dr == 0 and dc == 0:
                continue
            var rr = (r + dr + g.rows) % g.rows
            var cc = (c + dc + g.cols) % g.cols
            if _gridctx_sym(g.data[rr * g.cols + cc]) == p:
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


# A random 2-level count rule: predicate P, threshold t in {2,3}, distinct C1/C2.
def rand_task(nt: Int, nv: Int) -> GridTask:
    var p = Int(random_float64(0.0, Float64(A)))
    var t = 2 + Int(random_float64(0.0, 2.0))
    var a = Int(random_float64(0.0, Float64(A)))
    var b = Int(random_float64(0.0, Float64(A)))
    while b == a:
        b = Int(random_float64(0.0, Float64(A)))
    return make_task(p, t, Float32(a), Float32(b), nt, nv)


# The fixed-parameter Ckpt-A config validated by the offline probe: scaled one-hot
# embeddings (De == A), moderate sharpness, low decay.
def set_fixed(slow: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(GRIDNBHD_SLOW_DIM):
        slow[i] = 0.0
    for cc in range(A):
        slow[GRIDNBHD_E_OFF + cc * GRIDNBHD_DE + cc] = 3.0
    slow[GRIDNBHD_G_OFF] = 2.0
    slow[GRIDNBHD_C_OFF] = 0.0
    slow[GRIDNBHD_BETA_OFF] = 0.0  # eta ~ 0.5
    slow[GRIDNBHD_BALPHA_OFF] = -6.0  # alpha ~ 0.002


# Fixed-config held-out for one specific rule (Ckpt A).
def fixed_heldout(p: Int, t: Int, c1: Float32, c2: Float32) raises -> Float32:
    var slow = alloc[Float32](GRIDNBHD_SLOW_DIM)
    set_fixed(slow)
    var demos = List[ArcTaskPair]()
    for _ in range(12):
        var g = rand_grid()
        demos.append(ArcTaskPair(g^, apply_rule(g, p, t, c1, c2)))
    var state = alloc[Float32](GRIDNBHD_STATE_DIM)
    GridNbhdSelfModMemory.adapt(slow, demos, state)
    var ms = Float32(0.0)
    var n = R * C
    for _ in range(12):
        var tg = rand_grid()
        var truth = apply_rule(tg, p, t, c1, c2)
        var pred = alloc[Float32](n)
        GridNbhdSelfModMemory.apply(slow, state, tg, pred)
        ms += exact_match(pred, truth.data, n)
        pred.free()
    state.free()
    slow.free()
    return ms / 12.0


# LINEAR-read control: same centre-free key + gated delta write, but R = identity
# (clamped to [0,A-1]). A linear read of a count cannot be a sharp step, so a
# threshold rule must fail — isolating the nonlinearity as the cause.
def heldout_linear_control(
    p: Int, t: Int, c1: Float32, c2: Float32
) raises -> Float32:
    var demos = List[ArcTaskPair]()
    for _ in range(12):
        var gin = rand_grid()
        demos.append(ArcTaskPair(gin^, apply_rule(gin, p, t, c1, c2)))
    var slow = alloc[Float32](GRIDNBHD_SLOW_DIM)
    for i in range(GRIDNBHD_SLOW_DIM):
        slow[i] = 0.0
    for cc in range(A):
        slow[GRIDNBHD_E_OFF + cc * GRIDNBHD_DE + cc] = 3.0
    var eta = Float32(0.3)
    var state = alloc[Float32](GRIDNBHD_STATE_DIM)
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
    var n = R * C
    for _ in range(12):
        var tg = rand_grid()
        var truth = apply_rule(tg, p, t, c1, c2)
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
    return ms / 12.0


# One (multi-epoch) adapt pass on a fresh random rule, then held-out.
def fresh_heldout(slow: UnsafePointer[Float32, MutAnyOrigin]) raises -> Float32:
    var task = rand_task(8, 8)
    var state = alloc[Float32](GRIDNBHD_STATE_DIM)
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

    # ---------- Checkpoint A: fixed-key, arbitrary rules + linear control ----------
    # Three rules covering: normal (fire->larger), INVERTED (fire->smaller), t=3.
    var r1 = fixed_heldout(1, 2, 4.0, 0.0)  # fire -> 4  (normal)
    var r2 = fixed_heldout(3, 2, 2.0, 3.0)  # fire -> 2  (inverted: 2 < 3)
    var r3 = fixed_heldout(4, 3, 3.0, 0.0)  # t = 3
    print(
        "  ckptA fixed-config held-out: normal", r1, " inverted", r2, " t=3", r3
    )
    seed(0)
    var lin = heldout_linear_control(1, 2, 4.0, 0.0)
    print("  ckptA LINEAR-read control (identity+clamp):", lin)

    if r1 < 0.95 or r2 < 0.95 or r3 < 0.95:
        raise Error(
            "ERROR: the generalised self-write did not learn an arbitrary"
            " 2-level count rule (arbitrary colours / inverted / t=3) to >="
            " 0.95 held-out (got "
            + String(r1)
            + ", "
            + String(r2)
            + ", "
            + String(r3)
            + ")."
        )
    if lin >= 0.75:
        raise Error(
            "ERROR: the LINEAR-read control unexpectedly approached the bar"
            " (got "
            + String(lin)
            + ") — the nonlinearity is not load-bearing; claim vacuous."
        )
    print(
        "Grid-neighbourhood checkpoint A passed: the memory infers arbitrary"
        " 2-level count rules (colours + threshold, incl. inverted) in-context,"
        " where the linear read cannot."
    )

    # ---------- Checkpoint B: meta-learn the projections cold (emergent) ----------
    seed(0)
    var slow = alloc[Float32](GRIDNBHD_SLOW_DIM)
    GridNbhdSelfModMemory.seed_slow(slow)  # E = 0: no learned representation

    var before = Float32(0.0)
    for _ in range(8):
        before += fresh_heldout(slow)
    before = before / 8.0
    print(
        "  ckptB fresh held-out BEFORE meta-fit (E=0 seed = best-constant):",
        before,
    )

    var meta = List[GridTask]()
    for _ in range(8):
        meta.append(rand_task(8, 4))
    # Budget matched to block 2 (~150s) so the full suite stays under the 10-min
    # line; before/after gap is large, leaving headroom.
    meta_fit_selfmod[GridNbhdSelfModMemory](
        slow, meta, R * C, 64, 0.1, 0.003, 0.5, 0.01, 1000
    )

    var after = Float32(0.0)
    var n_fresh = 10
    for _ in range(n_fresh):
        after += fresh_heldout(slow)
    after = after / Float32(n_fresh)
    print(
        "  ckptB fresh held-out AFTER meta-fit (one-pass, 10 unseen rules):",
        after,
    )
    slow.free()

    if after < 0.95:
        raise Error(
            "ERROR: the meta-learned memory did not solve a fresh arbitrary"
            " 2-level count rule to >= 0.95 held-out in one adapt pass (got "
            + String(after)
            + ")."
        )
    # Non-vacuous: from a zero-embedding prior the write can only predict the
    # majority colour (~best constant); meta-learning must DISCOVER separable
    # colour embeddings to classify the count. Require a clear gap below the bar.
    if before >= 0.85:
        raise Error(
            "ERROR: the E=0 generic seed already generalised (before="
            + String(before)
            + ") — the meta-fit is not doing the work; emergence claim vacuous."
        )

    print(
        "Grid-neighbourhood checkpoint B passed: from a zero-embedding prior"
        " the self-modifying memory meta-learned colour embeddings + the"
        " nonlinear read cold; a fresh arbitrary 2-level count rule (predicate,"
        " threshold, and both output colours) adapts in ONE pass."
    )
