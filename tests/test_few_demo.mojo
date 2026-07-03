# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_few_demo.mojo`.
from hope import ArcGrid, ArcTaskPair, COLOR_DIM
from memory_composed import (
    GeomColorComposedMemory,
    GEOMCOLOR_DIM,
    GeomCountComposedMemory,
    GEOMCOUNT_A,
    GEOMCOUNT_B,
    GEOMCOUNT_DIM,
)
from esper_evolution import (
    fit_geomcolor,
    fit_geomcount,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_REG,
)
from arc_io import exact_match

# ==========================================================================
# Few-demo robustness regression (the few-demo block). The real ARC corpus
# has MEDIAN 3 demos per task (min 2); the synth proofs fit from 8. This test
# locks in the block's two hardenings at the corpus-median demo count:
#   - write_color: global-min greedy INJECTIVE assignment, identity defaults
#     for unseen colours, identity preference on exact ties (before: at n=3,
#     3-5/10 tasks had a wrong/colliding V; after: 0-1/10).
#   - constant-compute budgeting in the composed fit drivers (iters scaled by
#     FIT_DEMO_REF / n_demos — measured: flip∘countmap n=3 solved 6/10 at the
#     nominal budget, 9/10 at constant compute).
#
# Bars (from the measured degradation tables, with margin — JOURNAL
# 2026-07-03): at n=3 the AGGREGATE mean held-out over all four families must
# be >= 0.85 (measured after: 0.91; before the hardening: 0.74). At n=8 each
# family stays >= 0.95 (no regression — the n=8 path must be untouched).
# Per-task fixed seeds (the arc_solve protocol): deterministic, no
# seed-shopping — the seeds are the first tasks of the measurement cells.
# ==========================================================================

comptime GC_R = 4
comptime GC_C = 4
comptime CT_R = 6
comptime CT_C = 6
comptime NFIT = 64
comptime ITERS = 2000


def rand_grid(rows: Int, cols: Int, a: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(a))))
    return g^


def gc_xform(name: String, g: ArcGrid) -> ArcGrid:
    var o = ArcGrid(g.rows, g.cols)
    for r in range(g.rows):
        for c in range(g.cols):
            if name == "flip_h":
                o.set(r, c, g.get(r, g.cols - 1 - c))
            elif name == "recolor":
                o.set(r, c, Float32((Int(g.get(r, c)) + 1) % 10))
            else:  # flip_h_recolor
                o.set(r, c, Float32((Int(g.get(r, g.cols - 1 - c)) + 1) % 10))
    return o^


def gc_task(name: String, n_train: Int, task_seed: Int) raises -> Float32:
    seed(task_seed)
    var demos = List[ArcTaskPair]()
    for _ in range(n_train):
        var gin = rand_grid(GC_R, GC_C, 10)
        var gout = gc_xform(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))
    var state = alloc[Float32](GEOMCOLOR_DIM)
    GeomColorComposedMemory.seed(state)
    fit_geomcolor(
        state,
        demos,
        GC_R * GC_C,
        NFIT,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        ITERS,
        FIT_REG,
    )
    var pred = alloc[Float32](GC_R * GC_C)
    var ms = Float32(0.0)
    for _ in range(8):
        var tin = rand_grid(GC_R, GC_C, 10)
        var truth = gc_xform(name, tin)
        GeomColorComposedMemory.apply(state, tin, pred)
        ms += exact_match(pred, truth.data, GC_R * GC_C)
    pred.free()
    state.free()
    return ms / 8.0


def ct_count_rule(g: ArcGrid, p: Int, m: List[Int]) -> ArcGrid:
    var o = ArcGrid(CT_R, CT_C)
    for r in range(CT_R):
        for c in range(CT_C):
            var k = 0
            for dr in range(-1, 2):
                for dc in range(-1, 2):
                    if dr == 0 and dc == 0:
                        continue
                    if (
                        Int(
                            g.data[
                                ((r + dr + CT_R) % CT_R) * CT_C
                                + ((c + dc + CT_C) % CT_C)
                            ]
                        )
                        == p
                    ):
                        k += 1
            if k > GEOMCOUNT_B - 1:
                k = GEOMCOUNT_B - 1
            o.data[r * CT_C + c] = Float32(m[k])
    return o^


def ct_task(n_train: Int, task_seed: Int) raises -> Float32:
    # flip_h ∘ countmap — the family the constant-compute budgeting fixed.
    seed(task_seed)
    var p = Int(random_float64(0.0, Float64(GEOMCOUNT_A)))
    var m = List[Int]()
    for j in range(GEOMCOUNT_B):
        m.append(j)
    for j in range(GEOMCOUNT_B - 1, 0, -1):
        var i = Int(random_float64(0.0, Float64(j + 1)))
        var t = m[j]
        m[j] = m[i]
        m[i] = t
    var demos = List[ArcTaskPair]()
    for _ in range(n_train):
        var gin = rand_grid(CT_R, CT_C, GEOMCOUNT_A)
        var mid = ct_count_rule(gin, p, m)
        var gout = ArcGrid(CT_R, CT_C)
        for r in range(CT_R):
            for c in range(CT_C):
                gout.set(r, c, mid.get(r, CT_C - 1 - c))
        demos.append(ArcTaskPair(gin^, gout^))
    var state = alloc[Float32](GEOMCOUNT_DIM)
    GeomCountComposedMemory.seed(state)
    fit_geomcount(
        state,
        demos,
        CT_R * CT_C,
        NFIT,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        ITERS,
        FIT_REG,
    )
    var pred = alloc[Float32](CT_R * CT_C)
    var ms = Float32(0.0)
    for _ in range(8):
        var tin = rand_grid(CT_R, CT_C, GEOMCOUNT_A)
        var mid = ct_count_rule(tin, p, m)
        var truth = ArcGrid(CT_R, CT_C)
        for r in range(CT_R):
            for c in range(CT_C):
                truth.set(r, c, mid.get(r, CT_C - 1 - c))
        GeomCountComposedMemory.apply(state, tin, pred)
        ms += exact_match(pred, truth.data, CT_R * CT_C)
    pred.free()
    state.free()
    return ms / 8.0


def main() raises:
    var gc_names = List[String]()
    gc_names.append("recolor")
    gc_names.append("flip_h")
    gc_names.append("flip_h_recolor")

    # ---- n=3 (the corpus median): aggregate bar over all four families.
    var total = Float32(0.0)
    var count = 0
    for f in range(len(gc_names)):
        var fam_sum = Float32(0.0)
        for t in range(4):
            # Same seed lines as the measurement cells (no seed-shopping).
            var ho = gc_task(gc_names[f], 3, 1000 * (f + 1) + 300 + t)
            fam_sum += ho
            total += ho
            count += 1
        print("  n=3", gc_names[f], " mean held-out:", fam_sum / 4.0)
    var ct_sum = Float32(0.0)
    for t in range(4):
        var ho = ct_task(3, 10300 + t)
        ct_sum += ho
        total += ho
        count += 1
    print("  n=3 flip_h∘countmap mean held-out:", ct_sum / 4.0)
    var aggregate = total / Float32(count)
    print("  n=3 AGGREGATE mean held-out:", aggregate)
    if aggregate < 0.85:
        raise Error(
            "ERROR: few-demo aggregate held-out at n=3 fell to "
            + String(aggregate)
            + " (bar 0.85; pre-hardening was ~0.74 — regression in the"
            " write hardening or the constant-compute budgeting)."
        )
    print("n=3 bar passed (>= 0.85 aggregate; pre-hardening ~0.74).")

    # ---- n=8 regression guard: the proof-scale path must stay perfect.
    for f in range(len(gc_names)):
        var ho = gc_task(gc_names[f], 8, 1000 * (f + 1) + 800)
        print("  n=8", gc_names[f], " held-out:", ho)
        if ho < 0.95:
            raise Error(
                "ERROR: n=8 regression in "
                + gc_names[f]
                + " (held-out "
                + String(ho)
                + ")."
            )
    var ho8 = ct_task(8, 10800)
    print("  n=8 flip_h∘countmap held-out:", ho8)
    if ho8 < 0.95:
        raise Error("ERROR: n=8 regression in flip_h∘countmap.")
    print("n=8 regression guard passed.")

    print("Few-demo robustness test passed.")
