# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.math import round
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_composed_content.mojo`.
from hope import ArcGrid, ArcTaskPair
from memory_composed import (
    GeomCountComposedMemory,
    GEOMCOUNT_A,
    GEOMCOUNT_B,
    GEOMCOUNT_DIM,
    GEOMCOUNT_P_OFF,
    GEOMCOUNT_V_OFF,
)
from esper_evolution import (
    fit_geomcount,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_REG,
)
from arc_io import exact_match

# ==========================================================================
# The content×geometry composition proof (roadmap #1, the v2-confirmed binding
# constraint): GeomCountComposedMemory solves out = geom(M(count_P(in))) —
# a task class NO existing single memory expresses (AttnGather has no content
# rule; GridCountMap has no geometry; GeomColorComposed composes only a
# cellwise colour map).
#
#   Ckpt A — the content write is exact: (P, M) recovered from histogram
#            signatures alone, under flip/transpose/identity geometry (the
#            correspondence-free route; positions never used).
#   Ckpt B — the milestone: {countmap, flip_h∘countmap, flip_v∘countmap,
#            transpose∘countmap} each >= 0.95 held-out, per-task cold
#            (seed → one fit_geomcount call).
#   Controls — (i) the block-4 CORRESPONDENCE statistic (per-cell variance
#            reduction, paired by position) collapses under a flip while the
#            histogram route stays exact — the invariance is load-bearing;
#            (ii) content-write ablation on a composed task fails.
#
# Budget note: fits run at N=64 / iters=2000 (probe-validated to 1.0; the
# full FIT_N/FIT_ITERS at 6x6 would push the suite past its time budget).
# The cold-fit bar is about NO per-task staging — the budget is uniform.
# ==========================================================================

comptime R = 6
comptime C = 6
comptime FIT_N_CC = 64
comptime FIT_ITERS_CC = 2000


def rand_grid() -> ArcGrid:
    var g = ArcGrid(R, C)
    for k in range(R * C):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(GEOMCOUNT_A))))
    return g^


# Ground truth content rule: out = M[min(count_P, B-1)], toroidal Moore-8.
def count_rule(g: ArcGrid, p: Int, m: List[Int]) -> ArcGrid:
    var o = ArcGrid(R, C)
    for r in range(R):
        for c in range(C):
            var k = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    if (
                        Int(g.data[((r + dr + R) % R) * C + ((c + dc + C) % C)])
                        == p
                    ):
                        k += 1
            if k > GEOMCOUNT_B - 1:
                k = GEOMCOUNT_B - 1
            o.data[r * C + c] = Float32(m[k])
    return o^


def geom(name: String, g: ArcGrid) -> ArcGrid:
    var o = ArcGrid(R, C)
    for r in range(R):
        for c in range(C):
            if name == "flip_h":
                o.set(r, c, g.get(r, C - 1 - c))
            elif name == "flip_v":
                o.set(r, c, g.get(R - 1 - r, c))
            elif name == "transpose":
                o.set(r, c, g.get(c, r))
            else:  # identity
                o.set(r, c, g.get(r, c))
    return o^


def rand_perm_m() -> List[Int]:
    # Random permutation of 0..B-1 (injective map — this block's scope).
    var m = List[Int]()
    for j in range(GEOMCOUNT_B):
        m.append(j)
    for j in range(GEOMCOUNT_B - 1, 0, -1):
        var i = Int(random_float64(0.0, Float64(j + 1)))
        var t = m[j]
        m[j] = m[i]
        m[i] = t
    return m^


def make_demos(gname: String, p: Int, m: List[Int]) -> List[ArcTaskPair]:
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid()
        var mid = count_rule(gin, p, m)
        var gout = geom(gname, mid)
        demos.append(ArcTaskPair(gin^, gout^))
    return demos^


# Cold per-task protocol (per-task RNG seed — the arc_solve protocol).
def learn_and_eval(gname: String, task_seed: Int) raises -> Float32:
    seed(task_seed)
    var p = Int(random_float64(0.0, Float64(GEOMCOUNT_A)))
    var m = rand_perm_m()
    var demos = make_demos(gname, p, m)
    var state = alloc[Float32](GEOMCOUNT_DIM)
    GeomCountComposedMemory.seed(state)
    fit_geomcount(
        state,
        demos,
        R * C,
        FIT_N_CC,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS_CC,
        FIT_REG,
    )
    var pred = alloc[Float32](R * C)
    var ms = Float32(0.0)
    for _ in range(8):
        var tin = rand_grid()
        var mid = count_rule(tin, p, m)
        var truth = geom(gname, mid)
        GeomCountComposedMemory.apply(state, tin, pred)
        ms += exact_match(pred, truth.data, R * C)
    pred.free()
    state.free()
    return ms / 8.0


# Block-4-style CORRESPONDENCE statistic: variance reduction of out|count_p
# with (count, out) paired BY POSITION. Under an unknown geometry the pairing
# is scrambled — this is exactly what the histogram route avoids.
def corr_reduction(demos: List[ArcTaskPair], p: Int) -> Float32:
    var num = Float32(0.0)
    var osum = Float32(0.0)
    var osq = Float32(0.0)
    var bsum = alloc[Float32](GEOMCOUNT_B)
    var bsq = alloc[Float32](GEOMCOUNT_B)
    var bn = alloc[Float32](GEOMCOUNT_B)
    for j in range(GEOMCOUNT_B):
        bsum[j] = 0.0
        bsq[j] = 0.0
        bn[j] = 0.0
    for d in range(len(demos)):
        for r in range(R):
            for c in range(C):
                var k = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        if dr == 0 and dc == 0:
                            continue
                        if (
                            Int(
                                demos[d].input_grid.data[
                                    ((r + dr + R) % R) * C + ((c + dc + C) % C)
                                ]
                            )
                            == p
                        ):
                            k += 1
                if k > GEOMCOUNT_B - 1:
                    k = GEOMCOUNT_B - 1
                var o = demos[d].output_grid.data[r * C + c]
                num += 1.0
                osum += o
                osq += o * o
                bsum[k] += o
                bsq[k] += o * o
                bn[k] += 1.0
    var mean = osum / num
    var ovar = osq / num - mean * mean
    var within = Float32(0.0)
    for j in range(GEOMCOUNT_B):
        if bn[j] > 0:
            var bm = bsum[j] / bn[j]
            within += (bsq[j] / bn[j] - bm * bm) * (bn[j] / num)
    bsum.free()
    bsq.free()
    bn.free()
    if ovar <= 0:
        return 0.0
    return (ovar - within) / ovar


def main() raises:
    # ---- Ckpt A: (P, M) recovered exactly from histograms, per geometry.
    var gnames = List[String]()
    gnames.append("identity")
    gnames.append("flip_h")
    gnames.append("transpose")
    for i in range(len(gnames)):
        seed(10 + i)
        var p = Int(random_float64(0.0, Float64(GEOMCOUNT_A)))
        var m = rand_perm_m()
        var demos = make_demos(gnames[i], p, m)
        var state = alloc[Float32](GEOMCOUNT_DIM)
        GeomCountComposedMemory.seed(state)
        GeomCountComposedMemory.write_content(state, demos)
        var got_p = GeomCountComposedMemory._selected_p(state)
        if got_p != p:
            raise Error(
                "ERROR (Ckpt A): wrong predicate colour under "
                + gnames[i]
                + ": got "
                + String(got_p)
                + ", want "
                + String(p)
            )
        for j in range(GEOMCOUNT_B):
            if Int(round(state[GEOMCOUNT_V_OFF + j])) != m[j]:
                raise Error(
                    "ERROR (Ckpt A): wrong map entry under "
                    + gnames[i]
                    + " at bin "
                    + String(j)
                )
        state.free()
    print("Ckpt A passed: (P, M) recovered from histograms under geometry.")

    # ---- Control (i): the correspondence statistic collapses under a flip.
    seed(20)
    var p_c = Int(random_float64(0.0, Float64(GEOMCOUNT_A)))
    var m_c = rand_perm_m()
    var demos_id = make_demos("identity", p_c, m_c)
    seed(20)
    var _p2 = Int(random_float64(0.0, Float64(GEOMCOUNT_A)))
    var m_c2 = rand_perm_m()
    var demos_fl = make_demos("flip_h", p_c, m_c2)
    var red_id = corr_reduction(demos_id, p_c)
    var red_fl = corr_reduction(demos_fl, p_c)
    print(
        "   correspondence variance-reduction: identity",
        red_id,
        " flip_h",
        red_fl,
    )
    if not (red_id > 0.9 and red_fl < 0.5):
        raise Error(
            "ERROR (control i): expected the correspondence statistic to be"
            " ~1 under identity and to collapse under flip_h."
        )
    print(
        "Control (i) passed: correspondence pairing collapses under geometry;"
        " the histogram route is load-bearing."
    )

    # ---- Ckpt B: the milestone — the composed class, per-task cold.
    var tasks = List[String]()
    tasks.append("identity")
    tasks.append("flip_h")
    tasks.append("flip_v")
    tasks.append("transpose")
    var solved = 0
    for i in range(len(tasks)):
        var held_out = learn_and_eval(tasks[i], i + 1)
        print("  ", tasks[i], "∘countmap held-out:", held_out)
        if held_out >= 0.95:
            solved += 1
    if solved != len(tasks):
        raise Error(
            "ERROR (Ckpt B): composed content×geometry reached only "
            + String(solved)
            + "/"
            + String(len(tasks))
            + " tasks at >= 0.95 held-out."
        )
    print("Ckpt B passed: content×geometry class solved cold.")

    # ---- Control (ii): content-write ablation — geometry fit alone fails.
    seed(30)
    var p_a = Int(random_float64(0.0, Float64(GEOMCOUNT_A)))
    var m_a = rand_perm_m()
    var demos_a = make_demos("flip_h", p_a, m_a)
    var state_a = alloc[Float32](GEOMCOUNT_DIM)
    GeomCountComposedMemory.seed(state_a)
    # Full fit first (fit_geomcount always writes the content)...
    fit_geomcount(
        state_a,
        demos_a,
        R * C,
        FIT_N_CC,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS_CC,
        FIT_REG,
    )
    # ...then ABLATE the written content back to the uninformative seed: even
    # with a perfectly fitted geometry, the composition must fail without the
    # content module — proving it is load-bearing, not decoration.
    for i in range(GEOMCOUNT_A):
        state_a[GEOMCOUNT_P_OFF + i] = 1.0 / Float32(GEOMCOUNT_A)
    for j in range(GEOMCOUNT_B):
        state_a[GEOMCOUNT_V_OFF + j] = Float32(j)
    var pred_a = alloc[Float32](R * C)
    var ms_a = Float32(0.0)
    for _ in range(8):
        var tin = rand_grid()
        var mid = count_rule(tin, p_a, m_a)
        var truth = geom("flip_h", mid)
        GeomCountComposedMemory.apply(state_a, tin, pred_a)
        ms_a += exact_match(pred_a, truth.data, R * C)
    ms_a = ms_a / 8.0
    print("   control (content ablated) held-out:", ms_a)
    pred_a.free()
    state_a.free()
    if ms_a >= 0.5:
        raise Error(
            "ERROR (control ii): the content-ablated memory reached "
            + String(ms_a)
            + " — the content module may not be load-bearing."
        )
    print("Control (ii) passed: geometry alone cannot express the rule.")

    print(
        "Composed content×geometry test passed: the class no single memory"
        " expresses is solved cold."
    )
