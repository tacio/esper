# suite-tier: full
#   Heavy cold meta-fit milestone proof. Deferred from the `fast` tier — the
#   meta_fit_selfmod core is already covered there by test_selfmod_memory.
#   See run_tests.sh / CLAUDE.md "Testing".
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_grid_countmap_selfmod.mojo`.
from hope import ArcGrid, ArcTaskPair, Task
from memory import (
    GridCountMapSelfModMemory,
    GRIDCMAP_A,
    GRIDCMAP_B,
    GRIDCMAP_STATE_DIM,
    GRIDCMAP_SLOW_DIM,
    GRIDCMAP_W_OFF,
    GRIDCMAP_TAU_OFF,
    GRIDCMAP_TEMP_OFF,
    GRIDCMAP_MU_OFF,
    GRIDCMAP_EV_OFF,
    _gridctx_sym,
    GridNbhdSelfModMemory,
    GRIDNBHD_A,
    GRIDNBHD_DE,
    GRIDNBHD_STATE_DIM,
    GRIDNBHD_SLOW_DIM,
    GRIDNBHD_E_OFF,
    GRIDNBHD_G_OFF,
    GRIDNBHD_BETA_OFF,
    GRIDNBHD_BALPHA_OFF,
)
from esper_evolution import meta_fit_selfmod
from arc_io import exact_match

# ==========================================================================
# Multi-bin count -> colour MAP (ARC-AGI-2 block 4). Blocks 2-3 read a colour's
# Moore-8 neighbour count through a single sigmoid -> a 2-LEVEL output. This
# memory reads an ARBITRARY map out = M(count_P): count -> colour, non-monotone,
# >= 3 output colours; both P and M inferred per task. The salience is a
# META-LEARNED SCORING over per-colour demo statistics (it learns to weight
# variance-reduction, which captures non-monotone dependence, over linear
# correlation, which does not); the read is a soft count-bin value table.
#
# Ckpt A: FIXED good scoring; the write solves arbitrary NON-MONOTONE maps to
# >= 0.95 held-out. CONTROL: the block-3 2-LEVEL memory (GridNbhdSelfModMemory),
# which can emit only two output colours, FAILS the same >=3-colour map (< 0.85)
# — proving the multi-bin read is load-bearing. Fail-fast gate before the meta-fit.
# ==========================================================================

comptime A = GRIDCMAP_A
comptime R = 6
comptime C = 6
comptime GridTask = Task[ArcGrid]


def rand_grid() -> ArcGrid:
    var g = ArcGrid(R, C)
    for k in range(R * C):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(A))))
    return g^


# out = M[min(count_P, B-1)] with M given as B colour values.
def apply_rule(
    g: ArcGrid, p: Int, m0: Int, m1: Int, m2: Int, m3: Int, m4: Int
) -> ArcGrid:
    var o = ArcGrid(R, C)
    for r in range(R):
        for c in range(C):
            var k = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    if (
                        _gridctx_sym(
                            g.data[((r + dr + R) % R) * C + ((c + dc + C) % C)]
                        )
                        == p
                    ):
                        k += 1
            var col = m0
            if k == 1:
                col = m1
            elif k == 2:
                col = m2
            elif k == 3:
                col = m3
            elif k >= 4:
                col = m4
            o.data[r * C + c] = Float32(col)
    return o^


def make_task(
    p: Int, m0: Int, m1: Int, m2: Int, m3: Int, m4: Int, nt: Int, nv: Int
) -> GridTask:
    var train = List[ArcTaskPair]()
    for _ in range(nt):
        var g = rand_grid()
        train.append(ArcTaskPair(g^, apply_rule(g, p, m0, m1, m2, m3, m4)))
    var test = List[ArcTaskPair]()
    for _ in range(nv):
        var g = rand_grid()
        test.append(ArcTaskPair(g^, apply_rule(g, p, m0, m1, m2, m3, m4)))
    return GridTask(train^, test^)


# A random arbitrary count map (predicate P; >=3 distinct colours across the bins).
def rand_task(nt: Int, nv: Int) -> GridTask:
    var p = Int(random_float64(0.0, Float64(A)))
    var m0 = 0
    var m1 = 0
    var m2 = 0
    var m3 = 0
    var m4 = 0
    while True:
        m0 = Int(random_float64(0.0, Float64(A)))
        m1 = Int(random_float64(0.0, Float64(A)))
        m2 = Int(random_float64(0.0, Float64(A)))
        m3 = Int(random_float64(0.0, Float64(A)))
        m4 = Int(random_float64(0.0, Float64(A)))
        var distinct = 0
        var seen = List[Int]()
        for v in [m0, m1, m2, m3, m4]:
            var isnew = True
            for u in seen:
                if u == v:
                    isnew = False
            if isnew:
                seen.append(v)
                distinct += 1
        if distinct >= 3:
            break
    return make_task(p, m0, m1, m2, m3, m4, nt, nv)


# Fixed good scoring config (validated by the offline probe): weight the
# variance-reduction feature, sharp selection + bins, integer bin centres.
def set_fixed(slow: UnsafePointer[Float32, MutAnyOrigin]):
    for i in range(GRIDCMAP_SLOW_DIM):
        slow[i] = 0.0
    slow[GRIDCMAP_W_OFF + 0] = 6.0  # weight variance-reduction
    slow[GRIDCMAP_TAU_OFF] = 3.0
    slow[GRIDCMAP_TEMP_OFF] = 4.0
    for j in range(GRIDCMAP_B):
        slow[GRIDCMAP_MU_OFF + j] = Float32(j)
    slow[GRIDCMAP_EV_OFF] = 0.0


def fixed_heldout(
    p: Int, m0: Int, m1: Int, m2: Int, m3: Int, m4: Int
) raises -> Float32:
    var slow = alloc[Float32](GRIDCMAP_SLOW_DIM)
    set_fixed(slow)
    var demos = List[ArcTaskPair]()
    for _ in range(12):
        var g = rand_grid()
        demos.append(ArcTaskPair(g^, apply_rule(g, p, m0, m1, m2, m3, m4)))
    var state = alloc[Float32](GRIDCMAP_STATE_DIM)
    GridCountMapSelfModMemory.adapt(slow, demos, state)
    var ms = Float32(0.0)
    var n = R * C
    for _ in range(12):
        var tg = rand_grid()
        var truth = apply_rule(tg, p, m0, m1, m2, m3, m4)
        var pred = alloc[Float32](n)
        GridCountMapSelfModMemory.apply(slow, state, tg, pred)
        ms += exact_match(pred, truth.data, n)
        pred.free()
    state.free()
    slow.free()
    return ms / 12.0


# Honesty control: the block-3 2-LEVEL memory (only two output colours) on the
# same >=3-colour map. Fixed block-3 config; adapt + apply.
def control_2level(
    p: Int, m0: Int, m1: Int, m2: Int, m3: Int, m4: Int
) raises -> Float32:
    var slow = alloc[Float32](GRIDNBHD_SLOW_DIM)
    for i in range(GRIDNBHD_SLOW_DIM):
        slow[i] = 0.0
    for cc in range(GRIDNBHD_A):
        slow[GRIDNBHD_E_OFF + cc * GRIDNBHD_DE + cc] = 3.0
    slow[GRIDNBHD_G_OFF] = 2.0
    slow[GRIDNBHD_BETA_OFF] = 0.0
    slow[GRIDNBHD_BALPHA_OFF] = -6.0
    var demos = List[ArcTaskPair]()
    for _ in range(12):
        var g = rand_grid()
        demos.append(ArcTaskPair(g^, apply_rule(g, p, m0, m1, m2, m3, m4)))
    var state = alloc[Float32](GRIDNBHD_STATE_DIM)
    GridNbhdSelfModMemory.adapt(slow, demos, state)
    var ms = Float32(0.0)
    var n = R * C
    for _ in range(12):
        var tg = rand_grid()
        var truth = apply_rule(tg, p, m0, m1, m2, m3, m4)
        var pred = alloc[Float32](n)
        GridNbhdSelfModMemory.apply(slow, state, tg, pred)
        ms += exact_match(pred, truth.data, n)
        pred.free()
    state.free()
    slow.free()
    return ms / 12.0


def fresh_heldout(slow: UnsafePointer[Float32, MutAnyOrigin]) raises -> Float32:
    var task = rand_task(10, 8)
    var state = alloc[Float32](GRIDCMAP_STATE_DIM)
    GridCountMapSelfModMemory.adapt(slow, task.train, state)
    var ms = Float32(0.0)
    var n = R * C
    for j in range(len(task.test)):
        var pred = alloc[Float32](n)
        GridCountMapSelfModMemory.apply(
            slow, state, task.test[j].input_grid, pred
        )
        ms += exact_match(pred, task.test[j].output_grid.data, n)
        pred.free()
    state.free()
    return ms / Float32(len(task.test))


def main() raises:
    seed(0)

    # ---------- Checkpoint A: fixed scoring, arbitrary non-monotone maps ----------
    # Three arbitrary NON-MONOTONE >=3-colour maps.
    var r1 = fixed_heldout(2, 3, 1, 4, 1, 2)
    var r2 = fixed_heldout(1, 0, 2, 4, 2, 0)
    var r3 = fixed_heldout(3, 4, 2, 0, 2, 4)
    print("  ckptA fixed-scoring held-out (non-monotone maps):", r1, r2, r3)
    seed(0)
    var ctrl = control_2level(2, 3, 1, 4, 1, 2)
    print("  ckptA 2-level control (block-3 memory, same map):", ctrl)

    if r1 < 0.95 or r2 < 0.95 or r3 < 0.95:
        raise Error(
            "ERROR: the scoring salience + soft-bin value table did not solve"
            " an arbitrary non-monotone count map to >= 0.95 held-out (got "
            + String(r1)
            + ", "
            + String(r2)
            + ", "
            + String(r3)
            + ")."
        )
    if ctrl >= 0.85:
        raise Error(
            "ERROR: the 2-level control unexpectedly solved a >=3-colour map"
            " (got "
            + String(ctrl)
            + ") — multi-bin is not load-bearing; claim vacuous."
        )
    print(
        "Multi-bin checkpoint A passed: the soft-bin value table reads an"
        " arbitrary non-monotone count->colour map, where the 2-level read"
        " (block 3) cannot."
    )

    # ---------- Checkpoint B: meta-learn the scoring cold (emergent) ----------
    seed(0)
    var slow = alloc[Float32](GRIDCMAP_SLOW_DIM)
    GridCountMapSelfModMemory.seed_slow(slow)  # w=0: uniform selection

    var before = Float32(0.0)
    for _ in range(8):
        before += fresh_heldout(slow)
    before = before / 8.0
    print("  ckptB fresh held-out BEFORE meta-fit (w=0 seed):", before)

    var meta = List[GridTask]()
    for _ in range(8):
        meta.append(rand_task(10, 4))
    meta_fit_selfmod[GridCountMapSelfModMemory](
        slow, meta, R * C, 48, 0.1, 0.003, 0.5, 0.01, 500
    )

    var after = Float32(0.0)
    var n_fresh = 10
    for _ in range(n_fresh):
        after += fresh_heldout(slow)
    after = after / Float32(n_fresh)
    print(
        "  ckptB fresh held-out AFTER meta-fit (one-pass, 10 unseen maps):",
        after,
    )
    slow.free()

    if after < 0.95:
        raise Error(
            "ERROR: the meta-learned scoring did not solve a fresh arbitrary"
            " count map to >= 0.95 held-out in one adapt pass (got "
            + String(after)
            + ")."
        )
    if before >= 0.65:
        raise Error(
            "ERROR: the w=0 seed already generalised (before="
            + String(before)
            + ") — the meta-fit is not doing the work; emergence claim vacuous."
        )

    print(
        "Multi-bin checkpoint B passed: from an uninformative (w=0) scoring"
        " prior the memory meta-learned to identify the predicate colour by"
        " variance-reduction; a fresh arbitrary count->colour map adapts in ONE"
        " pass (a 2-level read cannot express it)."
    )
