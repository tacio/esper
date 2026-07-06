# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_mirror_tiling.mojo`.
from hope import ArcGrid, ArcTaskPair
from memory_composed import (
    ShapeGeomComposedMemory,
    SHAPEGEOM_DIM,
    SHAPEGEOM_MODE_OFF,
)
from esper_evolution import (
    fit_shape_geom,
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
# The MIRROR-TILING seam proof (Vision A / Next #1, Rung D): a k-fold size
# change has a THIRD canonical identity frame beyond the resized and periodic
# planes — the MIRROR (kaleidoscope) tiling where odd tiles are flipped,
# out[R+i] = in[R-1-i]. Provably OUTSIDE the periodic torus (which repeats,
# sawtooth); expressed by the REFLECT (triangle-fold) gather, discovered as a
# third cold start in fit_shape_geom (winner by demo fitness — an honest
# multi-start, not a selector). Corpus measure-first: mirror-tiling is the
# DOMINANT tiling family (8 train tasks vs 3 plain).
#
#   Ckpt A — mirror bar: {mirror_tile2, mirror_tile3} each >= 0.95 held-out at a
#            FRESH input size, per-task cold, and the winning frame is the
#            reflect mode (state[MODE] == 1).
#   Ckpt B — k=3 regression: {tile3, upscale3} (the output-growing k=3 families,
#            already expressed by the toroidal gather) stay >= 0.95 held-out —
#            the third start must not disturb them.
#   Control — mode ablation: a fitted mirror task read through the FORCED
#            toroidal gather (mode := 0) collapses to ~0 — reflect addressing is
#            load-bearing, not scaffolding.
#   Control — strict superset: a plain tile2 still solves AND lands mode == 0 —
#            the reflect start never hijacks a non-mirror task.
# ==========================================================================


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


# Inputs kept in [3, 5] so k=3 outputs (up to 15) stay cheap under the ES fit.
def rand_dim() -> Int:
    return 3 + Int(random_float64(0.0, 3.0))  # [3, 5]


def apply_transform(name: String, g: ArcGrid) -> ArcGrid:
    var k = 2
    if name == "mirror_tile3" or name == "tile3" or name == "upscale3":
        k = 3
    var out = ArcGrid(g.rows * k, g.cols * k)
    for R in range(g.rows * k):
        for C in range(g.cols * k):
            var ir = R % g.rows
            var ic = C % g.cols
            if name == "upscale3":
                ir = R // 3
                ic = C // 3
            elif name == "mirror_tile2" or name == "mirror_tile3":
                if (R // g.rows) % 2 == 1:
                    ir = g.rows - 1 - ir
                if (C // g.cols) % 2 == 1:
                    ic = g.cols - 1 - ic
            out.set(R, C, g.get(ir, ic))
    return out^


def make_demos(name: String) -> List[ArcTaskPair]:
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rand_dim(), rand_dim())
        var gout = apply_transform(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))
    return demos^


# Held-out eval at a fresh input size. `force_torus` re-reads the fitted state
# through the toroidal gather (mode ablation) by zeroing the mode slot.
def eval_held_out(
    name: String,
    state: UnsafePointer[Float32, MutAnyOrigin],
    force_torus: Bool,
) raises -> Float32:
    var saved = state[SHAPEGEOM_MODE_OFF]
    if force_torus:
        state[SHAPEGEOM_MODE_OFF] = 0.0
    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var test_in = rand_grid(rand_dim(), rand_dim())
        var truth = apply_transform(name, test_in)
        var pr = ShapeGeomComposedMemory.out_rows(state, test_in)
        var pc = ShapeGeomComposedMemory.out_cols(state, test_in)
        if pr != truth.rows or pc != truth.cols:
            continue
        var pred = alloc[Float32](pr * pc)
        ShapeGeomComposedMemory.apply(state, test_in, pr, pc, pred)
        match_sum += exact_match(pred, truth.data, pr * pc)
        pred.free()
    state[SHAPEGEOM_MODE_OFF] = saved
    return match_sum / Float32(trials)


def demos_capacity(demos: List[ArcTaskPair]) -> Int:
    var cap = 1
    for d in range(len(demos)):
        if demos[d].input_grid.size() > cap:
            cap = demos[d].input_grid.size()
        if demos[d].output_grid.size() > cap:
            cap = demos[d].output_grid.size()
    return cap


def learn_and_eval(
    name: String,
    task_seed: Int,
    state: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Float32:
    seed(task_seed)
    var demos = make_demos(name)
    var cap = demos_capacity(demos)
    ShapeGeomComposedMemory.seed(state)
    fit_shape_geom(
        state,
        demos,
        cap,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )
    return eval_held_out(name, state, False)


def main() raises:
    seed(0)

    # ---- Ckpt A: the mirror bar — each family cold, reflect frame wins.
    var mirror_names = List[String]()
    mirror_names.append("mirror_tile2")
    mirror_names.append("mirror_tile3")
    var mode_ablation = Float32(-1.0)
    for i in range(len(mirror_names)):
        var st = alloc[Float32](SHAPEGEOM_DIM)
        var ho = learn_and_eval(mirror_names[i], i + 1, st)
        var mode = st[SHAPEGEOM_MODE_OFF]
        print("  ", mirror_names[i], " held-out:", ho, " mode:", mode)
        if ho < 0.95:
            st.free()
            raise Error(
                "ERROR (Ckpt A): "
                + mirror_names[i]
                + " held-out "
                + String(ho)
                + " < 0.95 — the mirror frame did not solve it cold."
            )
        if mode < 0.5:
            st.free()
            raise Error(
                "ERROR (Ckpt A): "
                + mirror_names[i]
                + " solved but the reflect frame did not win (mode "
                + String(mode)
                + ")."
            )
        if mirror_names[i] == "mirror_tile2":
            mode_ablation = eval_held_out("mirror_tile2", st, True)
        st.free()
    print("Ckpt A passed: mirror_tile{2,3} solved cold, reflect frame wins.")

    # ---- Ckpt B: k=3 grow regression — the toroidal families untouched.
    var k3_names = List[String]()
    k3_names.append("tile3")
    k3_names.append("upscale3")
    for i in range(len(k3_names)):
        var st = alloc[Float32](SHAPEGEOM_DIM)
        var ho = learn_and_eval(k3_names[i], i + 10, st)
        print(
            "  ",
            k3_names[i],
            " held-out:",
            ho,
            " mode:",
            st[SHAPEGEOM_MODE_OFF],
        )
        if ho < 0.95:
            st.free()
            raise Error(
                "ERROR (Ckpt B): "
                + k3_names[i]
                + " held-out "
                + String(ho)
                + " < 0.95 — the third start disturbed a k=3 grow family."
            )
        st.free()
    print("Ckpt B passed: tile3 / upscale3 stay solved (k=3 grow regression).")

    # ---- Control: mode ablation (reflect addressing is load-bearing).
    print("  mirror_tile2 forced-toroidal held-out:", mode_ablation)
    if mode_ablation >= 0.5:
        raise Error(
            "ERROR (mode ablation): mirror_tile2 read through the toroidal"
            " gather scored "
            + String(mode_ablation)
            + " >= 0.5 — the reflect frame should be load-bearing."
        )
    print("Control passed: forcing the torus collapses the mirror read.")

    # ---- Control: strict superset — a plain tile2 solves AND lands mode 0.
    var st2 = alloc[Float32](SHAPEGEOM_DIM)
    var ho2 = learn_and_eval("tile2", 42, st2)
    var mode2 = st2[SHAPEGEOM_MODE_OFF]
    print("  tile2 held-out:", ho2, " mode:", mode2)
    st2.free()
    if ho2 < 0.95:
        raise Error(
            "ERROR (superset): plain tile2 held-out "
            + String(ho2)
            + " < 0.95 — the reflect start disturbed the periodic family."
        )
    if mode2 >= 0.5:
        raise Error(
            "ERROR (superset): plain tile2 landed the reflect frame (mode "
            + String(mode2)
            + ") — the mirror start hijacked a non-mirror task."
        )
    print("Control passed: plain tile2 solved in the toroidal frame (mode 0).")
    print("test_mirror_tiling PASSED.")
