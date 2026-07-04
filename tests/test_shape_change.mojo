# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed, random_float64
from std.math import round
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_shape_change.mojo`.
from hope import ArcGrid, ArcTaskPair
from memory_composed import (
    ShapeGeomComposedMemory,
    SHAPEGEOM_DIM,
    SHAPEGEOM_SHAPE_OFF,
)
from memory_es import AttnGatherMemory, ATTN_BETA_OFF
from esper_evolution import (
    fit_shape_geom,
    fit_shape,
    FIT_N,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_ITERS,
    FIT_REG,
    SHAPE_BETA_READ,
)
from arc_io import exact_match

# ==========================================================================
# The SHAPE-CHANGE seam proof (Vision A / Next #1): ShapeGeomComposedMemory
# produces outputs whose dims DIFFER from the input, with the output shape
# INFERRED IN-CONTEXT from the demos (a closed-form per-axis affine shape rule),
# composed with the proven AttnGather content gather run on the OUTPUT grid.
#
# Each task's demos are drawn at VARYING input sizes, so the shape rule
# out = k*in + b is genuinely identifiable (not memorized), and the held-out
# test — a FRESH, unseen input size — is an uncheatable generalization probe.
#
#   Ckpt A — the shape write is exact: on crop1 the least-squares rule recovers
#            (k=1, b=-2) per axis from the varying-size demos.
#   Ckpt B — the seam + geometry bar: {crop1, flip_h_crop1, subsample2,
#            upscale2, tile2} each >= 0.95 held-out at a fresh size, per-task
#            cold (seed → one fit_shape_geom call). upscale2 (blocky
#            replication, M = I/2 at sharp temperature) and tile2 (the modular
#            sawtooth `in[r mod n]` — nearest WRAPPED cell of the affine map at
#            trel = 1/2) are the output-GROWING families the toroidal gather
#            exists for.
#   Control — shape ablation: the SAME content fit WITHOUT the shape write
#            (identity shape rule) predicts the wrong output size on every
#            demo/test, so held-out collapses to ~0 — the inferred shape rule is
#            load-bearing, not scaffolding.
#   Control — wrap ablation: the fitted tile2 state read through the PLAIN
#            (non-toroidal) gather collapses — the modular source addressing is
#            load-bearing for tiling, exactly as the sawtooth argument says.
# ==========================================================================


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


# A random axis length per family (rows and cols drawn independently; the same
# RNG stream as the grids, for determinism). subsample2 needs even dims (out =
# in/2 must be integer-exact); the doubling families (upscale2/tile2) keep
# inputs in [3, 6] so their 2x outputs stay cheap under the full-budget ES fit
# (gather cost ~ out_cells x window area); everything else draws [4, 8].
def rand_dim(name: String) -> Int:
    if name == "subsample2":
        return 4 + 2 * Int(random_float64(0.0, 3.0))  # {4, 6, 8}
    if name == "upscale2" or name == "tile2":
        return 3 + Int(random_float64(0.0, 4.0))  # [3, 6]
    return 4 + Int(random_float64(0.0, 5.0))  # [4, 8]


# Ground-truth shape-changing transforms (what the engine must rediscover).
def apply_transform(name: String, g: ArcGrid) -> ArcGrid:
    if name == "crop1":
        var out = ArcGrid(g.rows - 2, g.cols - 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, g.get(r + 1, c + 1))
        return out^
    elif name == "flip_h_crop1":
        var out = ArcGrid(g.rows - 2, g.cols - 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, g.get(r + 1, g.cols - 2 - c))
        return out^
    elif name == "subsample2":
        # Every 2nd cell (even dims -> out = in/2 exactly).
        var out = ArcGrid(g.rows // 2, g.cols // 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, g.get(2 * r, 2 * c))
        return out^
    elif name == "upscale2":
        # Blocky replication: each cell -> a 2x2 block.
        var out = ArcGrid(g.rows * 2, g.cols * 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, g.get(r // 2, c // 2))
        return out^
    else:  # tile2: the grid replicated 2x2 (the modular sawtooth)
        var out = ArcGrid(g.rows * 2, g.cols * 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, g.get(r % g.rows, c % g.cols))
        return out^


def make_demos(name: String) -> List[ArcTaskPair]:
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rand_dim(name), rand_dim(name))
        var gout = apply_transform(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))
    return demos^


# Held-out eval at a FRESH input size: the predicted output shape must match the
# truth (a wrong shape rule scores 0 for that trial — no OOB compare). With
# `plain` the content is read through the NON-toroidal gather instead (the wrap
# ablation: same fitted state, modular addressing removed).
def eval_held_out(
    name: String,
    state: UnsafePointer[Float32, MutAnyOrigin],
    plain: Bool,
) raises -> Float32:
    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var test_in = rand_grid(rand_dim(name), rand_dim(name))
        var truth = apply_transform(name, test_in)
        var pr = ShapeGeomComposedMemory.out_rows(state, test_in)
        var pc = ShapeGeomComposedMemory.out_cols(state, test_in)
        if pr != truth.rows or pc != truth.cols:
            continue  # predicted shape wrong -> 0 for this trial
        var pred = alloc[Float32](pr * pc)
        if plain:
            # The attention slots are state[0:7]; the plain gather ignores
            # trel and reads a bounded (non-wrapped) source.
            AttnGatherMemory.apply_shaped(state, test_in, pr, pc, pred)
        else:
            ShapeGeomComposedMemory.apply(state, test_in, pr, pc, pred)
        match_sum += exact_match(pred, truth.data, pr * pc)
        pred.free()
    return match_sum / Float32(trials)


# Worst-case flat capacity over a demo list (the forward scratch size).
def demos_capacity(demos: List[ArcTaskPair]) -> Int:
    var cap = 1
    for d in range(len(demos)):
        if demos[d].input_grid.size() > cap:
            cap = demos[d].input_grid.size()
        if demos[d].output_grid.size() > cap:
            cap = demos[d].output_grid.size()
    return cap


# Cold per-task protocol: seed -> ONE fit_shape_geom call -> held-out eval.
# The RNG is re-seeded PER TASK (the arc_solve protocol) so a task's stochastic
# ES fit depends only on the task, not its position in the test. The caller
# owns `state` (seeded + fit here) so controls can re-read the fitted params.
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

    # ---- Ckpt A: the shape write recovers (k=1, b=-2) per axis on crop1.
    var demos_a = make_demos("crop1")
    var state_a = alloc[Float32](SHAPEGEOM_DIM)
    ShapeGeomComposedMemory.seed(state_a)
    ShapeGeomComposedMemory.write(state_a, demos_a)
    var kr = state_a[SHAPEGEOM_SHAPE_OFF + 0]
    var br = state_a[SHAPEGEOM_SHAPE_OFF + 1]
    var kc = state_a[SHAPEGEOM_SHAPE_OFF + 2]
    var bc = state_a[SHAPEGEOM_SHAPE_OFF + 3]
    state_a.free()
    var tol = Float32(0.05)
    if (
        abs(kr - 1.0) > tol
        or abs(br + 2.0) > tol
        or abs(kc - 1.0) > tol
        or abs(bc + 2.0) > tol
    ):
        raise Error(
            "ERROR (Ckpt A): crop1 shape rule = ("
            + String(kr)
            + "*r+"
            + String(br)
            + ", "
            + String(kc)
            + "*c+"
            + String(bc)
            + "), expected (1*r-2, 1*c-2)."
        )
    print("Ckpt A passed: shape write recovers crop1's (k=1, b=-2) per axis.")

    # ---- Ckpt B: the seam + geometry bar — each family cold at a fresh size.
    # The wrap-ablation control below re-reads tile2's fitted params.
    var names = List[String]()
    names.append("crop1")
    names.append("flip_h_crop1")
    names.append("subsample2")
    names.append("upscale2")
    names.append("tile2")

    var wrap_ctl = Float32(-1.0)
    var solved = 0
    for i in range(len(names)):
        var state_b = alloc[Float32](SHAPEGEOM_DIM)
        var held_out = learn_and_eval(names[i], i + 1, state_b)
        print("  ", names[i], " held-out:", held_out)
        if held_out >= 0.95:
            solved += 1
        if names[i] == "tile2":
            # Wrap ablation: the SAME fitted state through the plain gather.
            wrap_ctl = eval_held_out("tile2", state_b, True)
        state_b.free()
    if solved != len(names):
        raise Error(
            "ERROR (Ckpt B): the shape memory did not solve the whole shape"
            " family {crop, flip-crop, subsample, upscale, tile} to >= 0.95"
            " held-out ("
            + String(solved)
            + "/"
            + String(len(names))
            + " families)."
        )
    print("Ckpt B passed: the whole shape family solved cold, held-out.")

    # ---- Control (wrap ablation): tile2's fitted params, but read through the
    # NON-toroidal gather — the modular sawtooth `in[r mod n]` is provably
    # outside any single affine (M, t), so removing the wrap must collapse it.
    print("   control (plain gather) tile2 held-out:", wrap_ctl)
    if wrap_ctl >= 0.5:
        raise Error(
            "ERROR (control): tile2 reached "
            + String(wrap_ctl)
            + " through the plain (non-toroidal) gather — the modular source"
            " addressing should be load-bearing."
        )
    print("Control passed: no wrap -> tiling collapses (as it must).")

    # ---- Control (shape ablation): the SAME content ES fit but with NO shape
    # write (identity shape rule from the seed). Every demo's predicted output
    # area then mismatches its true output area (fitness_shape penalizes them
    # all -> no ES signal), and at eval the predicted shape is wrong -> ~0.
    seed(7)
    var demos_ctl = make_demos("crop1")
    var cap_ctl = demos_capacity(demos_ctl)
    var state_ctl = alloc[Float32](SHAPEGEOM_DIM)
    ShapeGeomComposedMemory.seed(state_ctl)  # identity shape rule, NOT written
    # Mirror fit_shape_geom's discrete regime (hard frozen read) — the control
    # ablates ONLY the shape write.
    state_ctl[ATTN_BETA_OFF] = SHAPE_BETA_READ
    var slow_ctl = alloc[Float32](SHAPEGEOM_DIM)
    ShapeGeomComposedMemory.seed(slow_ctl)
    fit_shape[ShapeGeomComposedMemory](
        state_ctl,
        slow_ctl,
        demos_ctl,
        cap_ctl,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )
    var held_out_ctl = eval_held_out("crop1", state_ctl, False)
    slow_ctl.free()
    state_ctl.free()
    print("   control (no shape write) crop1 held-out:", held_out_ctl)
    if held_out_ctl >= 0.5:
        raise Error(
            "ERROR (control): crop1 reached "
            + String(held_out_ctl)
            + " without the shape write — the ablation should fail, the shape"
            " rule may not be load-bearing."
        )
    print("Control passed: no shape rule -> wrong output size (as it must).")

    print(
        "Shape-change test passed: the output-size seam works — output shape"
        " inferred in-context, geometry fit on the output grid."
    )
