# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.math import round
from std.collections import List, InlineArray

# Run from the project root: `mojo run -I src tests/test_shape_color.mojo`.
from hope import ArcGrid, ArcTaskPair, COLOR_DIM
from memory_composed import (
    ShapeGeomColorComposedMemory,
    SHAPEGEOMCOLOR_DIM,
    SHAPEGEOMCOLOR_V_OFF,
)
from esper_evolution import (
    fit_shape_color,
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
# RUNG C — COLOUR ON TOP OF SHAPE. ShapeGeomColorComposedMemory composes a
# written colour table V on top of the proven shape+geometry gather:
#
#     out = shape_geom_gather( V(in) )
#
# Colour is cellwise (commutes with the copy gather — colour-then-gather), so V
# is written closed-form from the demos' FRACTION signatures (scale-invariant —
# the fix for count-conservation breaking under shape change) and the geometry
# is the unchanged two-frame multi-start fit on V-pre-mapped demos.
#
# THE COUNT-SIGNATURE WRITE'S HONEST PRECONDITION — colour-count CONTRAST. It
# matches colours by their across-demo count signatures, which identifies a
# permutation only when colours have DISTINGUISHABLE frequencies. Under EXACT
# same-shape conservation (GeomColor) even the tiny fluctuations of a uniform-
# random grid match exactly. Shape change is LOSSY (crop drops a border,
# subsample thins), so the signal must be REAL contrast, not fluctuation — which
# is exactly what real ARC grids have (a background + a few sparse colours, very
# different counts) and uniform-random grids lack. So these tasks draw grids
# from a per-task palette of DISTINCT per-colour frequencies (real-ARC-like);
# the uniform-random adversarial ceiling for crop is measured separately below.
#
#   Ckpt A — the fraction colour write recovers the task's permutation on
#            recolor_upscale2 (exact area ratio) AND recolor_crop1 (lossy border).
#   Ckpt B — {recolor_crop1, recolor_subsample2, recolor_upscale2, recolor_tile2}
#            each >= 0.95 held-out at a FRESH input size, per-task cold.
#   Control (colour ablation) — the SAME fitted state with V forced to identity
#            fails the recolored families: the colour module is load-bearing.
#   Control (count-contrast ceiling) — the write on UNIFORM-random crop demos
#            (no contrast) fails to recover V: the precondition is real, the
#            contrast load-bearing (documented, not fought).
#   Regression — a PURE-shape family (tile2, V = identity) through the new
#            memory reproduces test_shape_change's bar (the strict superset).
#   Few-demo — recolor_upscale2 at n=3 (the corpus median) clears the bar.
#
# Each task uses a RANDOM per-task palette (used colours + distinct frequencies)
# and a RANDOM permutation over the used colours (identity on the rest), held
# CONSTANT across the task's demos and re-derivable by re-seeding (so controls
# need no plumbing). Demos are drawn at VARYING sizes (shape rule identifiable);
# the held-out test is a fresh, unseen input size.
# ==========================================================================

comptime N_USED = 5  # colours actually used per task (real ARC uses few)


# A per-task palette: `perm` (the colour permutation ground truth, identity on
# unused colours) and `w` (per-colour sampling weight, 0 for unused, DISTINCT
# for the used colours so their count signatures are separable).
struct Palette(Copyable, Movable):
    var perm: InlineArray[Int, COLOR_DIM]
    var w: InlineArray[Float32, COLOR_DIM]

    def __init__(
        out self,
        perm: InlineArray[Int, COLOR_DIM],
        w: InlineArray[Float32, COLOR_DIM],
    ):
        self.perm = perm
        self.w = w


# Draw a task palette off the current RNG: shuffle the colours, take the first
# N_USED as the used set, permute them cyclically (a derangement — every used
# colour actually changes), and give them distinct descending weights.
def rand_palette() -> Palette:
    var idx = InlineArray[Int, COLOR_DIM](fill=0)
    for i in range(COLOR_DIM):
        idx[i] = i
    for i in range(COLOR_DIM - 1, 0, -1):
        var j = Int(random_float64(0.0, Float64(i + 1)))
        var tmp = idx[i]
        idx[i] = idx[j]
        idx[j] = tmp

    var perm = InlineArray[Int, COLOR_DIM](fill=0)
    for i in range(COLOR_DIM):
        perm[i] = i  # identity on unused colours
    var w = InlineArray[Float32, COLOR_DIM](fill=0.0)
    for i in range(N_USED):
        perm[idx[i]] = idx[
            (i + 1) % N_USED
        ]  # cyclic permutation of the used set
        w[idx[i]] = Float32(N_USED - i + 1)  # distinct weights: 6, 5, 4, 3, 2
    return Palette(perm, w)


# A grid whose cells are drawn from the palette's weighted colour distribution
# (categorical sampling) — distinct per-colour frequencies, real-ARC-like.
def rand_grid(
    rows: Int, cols: Int, w: InlineArray[Float32, COLOR_DIM]
) -> ArcGrid:
    var total = Float32(0.0)
    for c in range(COLOR_DIM):
        total += w[c]
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        var u = Float32(random_float64(0.0, Float64(total)))
        var acc = Float32(0.0)
        var col = 0
        for c in range(COLOR_DIM):
            acc += w[c]
            if u < acc:
                col = c
                break
        g.data[k] = Float32(col)
    return g^


# A uniform-random grid (the adversarial no-contrast case for the ceiling
# control): every colour equiprobable, so count signatures barely separate.
def rand_grid_uniform(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, Float64(COLOR_DIM))))
    return g^


def rand_dim(name: String) -> Int:
    if name == "recolor_subsample2":
        # Even dims {8, 10, 12}: subsample HALVES each axis, so the output must
        # be large enough for a reliable colour signature (a 2x2 output starves
        # the low-frequency colours — real-ARC subsample tasks aren't 4x4).
        return 8 + 2 * Int(random_float64(0.0, 3.0))  # {8, 10, 12}
    if name == "recolor_upscale2" or name == "recolor_tile2" or name == "tile2":
        return 3 + Int(
            random_float64(0.0, 4.0)
        )  # [3, 6] (doubling stays cheap)
    return 4 + Int(random_float64(0.0, 5.0))  # [4, 8] (crop1, recolor_crop1)


# A pure-shape stress family draws UNIFORM-random grids (no colour contrast) so
# the count-signature write has NO recolor signal — the adversarial case where a
# naive write scrambles V. The global recolor gate must keep V = identity here
# (strict superset). "tile2" keeps palette grids (exact-conservation regression).
def is_uniform_pure(name: String) -> Bool:
    return name == "crop1"


# Ground-truth colour-on-shape transform: recolor each cell through `perm`, then
# apply the shape change. (Recolor commutes with the copy gather.) The pure
# "tile2" regression control skips the recolor so its written V is identity.
def apply_transform(
    name: String, g: ArcGrid, perm: InlineArray[Int, COLOR_DIM]
) -> ArcGrid:
    var rc = ArcGrid(g.rows, g.cols)
    for k in range(g.rows * g.cols):
        if name == "tile2" or name == "crop1":
            rc.data[k] = g.data[k]  # pure shape: no recolor
        else:
            rc.data[k] = Float32(perm[Int(round(g.data[k]))])

    if name == "recolor_crop1" or name == "crop1":
        var out = ArcGrid(rc.rows - 2, rc.cols - 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, rc.get(r + 1, c + 1))
        return out^
    elif name == "recolor_subsample2":
        var out = ArcGrid(rc.rows // 2, rc.cols // 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, rc.get(2 * r, 2 * c))
        return out^
    elif name == "recolor_upscale2":
        var out = ArcGrid(rc.rows * 2, rc.cols * 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, rc.get(r // 2, c // 2))
        return out^
    else:  # recolor_tile2 or tile2 (both tile; tile2's rc is un-recolored)
        var out = ArcGrid(rc.rows * 2, rc.cols * 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, rc.get(r % rc.rows, c % rc.cols))
        return out^


def make_demos(name: String, n: Int, pal: Palette) -> List[ArcTaskPair]:
    var demos = List[ArcTaskPair]()
    for _ in range(n):
        var r = rand_dim(name)
        var c = rand_dim(name)
        var gin = rand_grid_uniform(r, c) if is_uniform_pure(
            name
        ) else rand_grid(r, c, pal.w)
        var gout = apply_transform(name, gin, pal.perm)
        demos.append(ArcTaskPair(gin^, gout^))
    return demos^


# Held-out eval at a FRESH input size (same palette). A wrong predicted shape
# scores 0 for that trial. `ablate_v` forces V = identity onto a COPY of the
# fitted state (the colour-ablation control).
def eval_held_out(
    name: String,
    state: UnsafePointer[Float32, MutAnyOrigin],
    pal: Palette,
    ablate_v: Bool,
) raises -> Float32:
    var st = alloc[Float32](SHAPEGEOMCOLOR_DIM)
    for i in range(SHAPEGEOMCOLOR_DIM):
        st[i] = state[i]
    if ablate_v:
        for s in range(COLOR_DIM):
            st[SHAPEGEOMCOLOR_V_OFF + s] = Float32(s)

    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var tr = rand_dim(name)
        var tc = rand_dim(name)
        var test_in = rand_grid_uniform(tr, tc) if is_uniform_pure(
            name
        ) else rand_grid(tr, tc, pal.w)
        var truth = apply_transform(name, test_in, pal.perm)
        var pr = ShapeGeomColorComposedMemory.out_rows(st, test_in)
        var pc = ShapeGeomColorComposedMemory.out_cols(st, test_in)
        if pr != truth.rows or pc != truth.cols:
            continue  # predicted shape wrong -> 0 for this trial
        var pred = alloc[Float32](pr * pc)
        ShapeGeomColorComposedMemory.apply(st, test_in, pr, pc, pred)
        match_sum += exact_match(pred, truth.data, pr * pc)
        pred.free()
    st.free()
    return match_sum / Float32(trials)


def demos_capacity(demos: List[ArcTaskPair]) -> Int:
    var cap = 1
    for d in range(len(demos)):
        if demos[d].input_grid.size() > cap:
            cap = demos[d].input_grid.size()
        if demos[d].output_grid.size() > cap:
            cap = demos[d].output_grid.size()
    return cap


# Cold per-task protocol: seed -> ONE fit_shape_color -> held-out eval. RNG
# re-seeded PER TASK (the arc_solve protocol) so the fit depends only on the
# task; the palette is re-derivable by re-seeding task_seed. Caller owns `state`.
def learn_and_eval(
    name: String,
    task_seed: Int,
    n_demos: Int,
    state: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Float32:
    seed(task_seed)
    var pal = rand_palette()
    var demos = make_demos(name, n_demos, pal)
    var cap = demos_capacity(demos)
    ShapeGeomColorComposedMemory.seed(state)
    fit_shape_color(
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
    return eval_held_out(name, state, pal, False)


# Count how many of the palette's colours the written V got wrong (write-only —
# no ES fit, so cheap; the geometry-invariant colour write is all that runs).
def write_miss_count(name: String, task_seed: Int, uniform: Bool) raises -> Int:
    seed(task_seed)
    var pal = rand_palette()
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var r = rand_dim(name)
        var c = rand_dim(name)
        var gin = rand_grid_uniform(r, c) if uniform else rand_grid(r, c, pal.w)
        var gout = apply_transform(name, gin, pal.perm)
        demos.append(ArcTaskPair(gin^, gout^))
    var st = alloc[Float32](SHAPEGEOMCOLOR_DIM)
    ShapeGeomColorComposedMemory.seed(st)
    ShapeGeomColorComposedMemory.write(st, demos)
    var wrong = 0
    # A uniform grid uses all colours; a palette grid only its used set. Compare
    # only colours the demos actually contain (unseen default to identity).
    for c in range(COLOR_DIM):
        if uniform or pal.w[c] > 0.0:
            if Int(round(st[SHAPEGEOMCOLOR_V_OFF + c])) != pal.perm[c]:
                wrong += 1
    st.free()
    return wrong


def main() raises:
    seed(0)

    # ---- Ckpt A: the fraction colour write recovers the task permutation on
    # both the exact-ratio (upscale) and lossy (crop) families, given contrast.
    var a_names = List[String]()
    a_names.append("recolor_upscale2")
    a_names.append("recolor_crop1")
    a_names.append("recolor_subsample2")
    for ai in range(len(a_names)):
        var wrong = write_miss_count(a_names[ai], 100 + ai, False)
        if wrong != 0:
            raise Error(
                "ERROR (Ckpt A): "
                + a_names[ai]
                + " colour write missed "
                + String(wrong)
                + " used colours of the task permutation."
            )
    print("Ckpt A passed: the fraction colour write recovers the permutation.")

    # ---- Control (count-contrast ceiling): the SAME write on UNIFORM-random
    # crop demos (no contrast) should FAIL to recover V — documenting that the
    # count-signature precondition is real, not scaffolding.
    var uni_wrong = write_miss_count("recolor_crop1", 100, True)
    print(
        "   control (uniform-random crop, no contrast) colours missed:",
        uni_wrong,
        "/ 10",
    )
    if uni_wrong == 0:
        raise Error(
            "ERROR (control): the colour write recovered V from UNIFORM-random"
            " crop demos — the count-contrast precondition should be load-"
            "bearing under lossy shape change."
        )
    print(
        "Control passed: no contrast -> crop colour write fails (as it must)."
    )

    # ---- Ckpt B: each colour-on-shape family cold at a fresh size.
    var names = List[String]()
    names.append("recolor_crop1")
    names.append("recolor_subsample2")
    names.append("recolor_upscale2")
    names.append("recolor_tile2")

    var ablate_ctl = Float32(-1.0)
    var solved = 0
    for i in range(len(names)):
        var state_b = alloc[Float32](SHAPEGEOMCOLOR_DIM)
        var held_out = learn_and_eval(names[i], i + 1, 8, state_b)
        print("  ", names[i], " held-out:", held_out)
        if held_out >= 0.95:
            solved += 1
        if names[i] == "recolor_upscale2":
            # Colour ablation: the SAME fitted state, V forced to identity.
            # Re-derive the task palette by re-seeding (deterministic).
            seed(i + 1)
            var pal_b = rand_palette()
            ablate_ctl = eval_held_out(names[i], state_b, pal_b, True)
        state_b.free()
    if solved != len(names):
        raise Error(
            "ERROR (Ckpt B): the colour-on-shape family was not solved to"
            " >= 0.95 held-out ("
            + String(solved)
            + "/"
            + String(len(names))
            + " families)."
        )
    print("Ckpt B passed: the whole colour-on-shape family solved cold.")

    # ---- Control (colour ablation): fitted recolor_upscale2 with V = identity.
    print("   control (V = identity) recolor_upscale2 held-out:", ablate_ctl)
    if ablate_ctl >= 0.5:
        raise Error(
            "ERROR (control): recolor_upscale2 reached "
            + String(ablate_ctl)
            + " with V = identity — the colour module should be load-bearing."
        )
    print("Control passed: no colour table -> recolor collapses (as it must).")

    # ---- Regression: a PURE-shape family (tile2) through the new memory. With
    # no recolor the written V is identity, so this must reproduce
    # test_shape_change's tile2 bar (the strict-superset guard).
    var state_r = alloc[Float32](SHAPEGEOMCOLOR_DIM)
    var held_r = learn_and_eval("tile2", 5, 8, state_r)
    state_r.free()
    print("   regression (pure tile2 through colour memory) held-out:", held_r)
    if held_r < 0.95:
        raise Error(
            "ERROR (regression): pure tile2 through the colour memory scored "
            + String(held_r)
            + " (< 0.95) — the identity-V path should match the shape memory."
        )
    print("Regression passed: V = identity is the pure shape path.")

    # ---- Regression (the strict-superset guard): a PURE-shape CROP on
    # UNIFORM-random grids (no colour contrast) — the adversarial case where a
    # naive count write scrambles V (measured 1.0 -> 0.17). The GLOBAL recolor
    # gate must keep V = identity, so held-out stays at the pure-shape bar.
    var state_c = alloc[Float32](SHAPEGEOMCOLOR_DIM)
    var held_c = learn_and_eval("crop1", 11, 8, state_c)
    var v_wrong = 0
    for c in range(COLOR_DIM):
        if Int(round(state_c[SHAPEGEOMCOLOR_V_OFF + c])) != c:
            v_wrong += 1
    state_c.free()
    print(
        "   regression (pure UNIFORM crop1) held-out:",
        held_c,
        " V non-identity entries:",
        v_wrong,
    )
    if held_c < 0.95 or v_wrong != 0:
        raise Error(
            "ERROR (strict superset): pure uniform crop1 scored "
            + String(held_c)
            + " with "
            + String(v_wrong)
            + " non-identity V entries — the recolor gate must keep V ="
            " identity"
            " on a no-contrast pure-shape task (else the colour path regresses"
            " the pure-shape path)."
        )
    print("Regression passed: no-contrast pure crop keeps V = identity.")

    # ---- Few-demo (corpus median n=3): the fraction-write must still recover V
    # and clear the bar at 3 demos.
    var state_f = alloc[Float32](SHAPEGEOMCOLOR_DIM)
    var held_f = learn_and_eval("recolor_upscale2", 9, 3, state_f)
    state_f.free()
    print("   few-demo (recolor_upscale2, n=3) held-out:", held_f)
    if held_f < 0.95:
        raise Error(
            "ERROR (few-demo): recolor_upscale2 at n=3 scored "
            + String(held_f)
            + " (< 0.95) — the fraction colour write should hold at the corpus"
            " median 3 demos."
        )
    print("Few-demo passed: the fraction colour write holds at n=3.")

    print(
        "Colour-on-shape test passed (Rung C): the shape path now composes a"
        " written colour table — held-out, cold, at fresh sizes."
    )
