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
#   Ckpt B — the seam + geometry bar: {crop1, flip_h_crop1, subsample2} each
#            >= 0.95 held-out at a fresh size, per-task cold (seed → one
#            fit_shape_geom call).
#   Control — shape ablation: the SAME content fit WITHOUT the shape write
#            (identity shape rule) predicts the wrong output size on every
#            demo/test, so held-out collapses to ~0 — the inferred shape rule is
#            load-bearing, not scaffolding.
# ==========================================================================


def rand_grid(rows: Int, cols: Int) -> ArcGrid:
    var g = ArcGrid(rows, cols)
    for k in range(rows * cols):
        g.data[k] = Float32(Int(random_float64(0.0, 10.0)))
    return g^


# A random axis length in [4, 8] (even-only for subsample, where out = in/2 must
# be integer-exact). Rows and cols are drawn independently from this. Uses the
# same RNG stream as the grids for determinism.
def rand_dim(even: Bool) -> Int:
    if even:
        return 4 + 2 * Int(random_float64(0.0, 3.0))  # {4, 6, 8}
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
    else:  # subsample2: every 2nd cell (even dims -> out = in/2 exactly)
        var out = ArcGrid(g.rows // 2, g.cols // 2)
        for r in range(out.rows):
            for c in range(out.cols):
                out.set(r, c, g.get(2 * r, 2 * c))
        return out^


def make_demos(name: String) -> List[ArcTaskPair]:
    var even = name == "subsample2"
    var demos = List[ArcTaskPair]()
    for _ in range(8):
        var gin = rand_grid(rand_dim(even), rand_dim(even))
        var gout = apply_transform(name, gin)
        demos.append(ArcTaskPair(gin^, gout^))
    return demos^


# Held-out eval at a FRESH input size: the predicted output shape must match the
# truth (a wrong shape rule scores 0 for that trial — no OOB compare).
def eval_held_out(
    name: String,
    state: UnsafePointer[Float32, MutAnyOrigin],
) raises -> Float32:
    var even = name == "subsample2"
    var match_sum = Float32(0.0)
    var trials = 8
    for _ in range(trials):
        var test_in = rand_grid(rand_dim(even), rand_dim(even))
        var truth = apply_transform(name, test_in)
        var pr = ShapeGeomComposedMemory.out_rows(state, test_in)
        var pc = ShapeGeomComposedMemory.out_cols(state, test_in)
        if pr != truth.rows or pc != truth.cols:
            continue  # predicted shape wrong -> 0 for this trial
        var pred = alloc[Float32](pr * pc)
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
# ES fit depends only on the task, not its position in the test.
def learn_and_eval(name: String, task_seed: Int) raises -> Float32:
    seed(task_seed)
    var demos = make_demos(name)
    var cap = demos_capacity(demos)
    var state = alloc[Float32](SHAPEGEOM_DIM)
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
    var held_out = eval_held_out(name, state)
    state.free()
    return held_out


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
    var names = List[String]()
    names.append("crop1")
    names.append("flip_h_crop1")
    names.append("subsample2")

    var solved = 0
    for i in range(len(names)):
        var held_out = learn_and_eval(names[i], i + 1)
        print("  ", names[i], " held-out:", held_out)
        if held_out >= 0.95:
            solved += 1
    if solved != len(names):
        raise Error(
            "ERROR (Ckpt B): the shape memory did not solve the whole"
            " crop/subsample family to >= 0.95 held-out ("
            + String(solved)
            + "/"
            + String(len(names))
            + " families)."
        )
    print("Ckpt B passed: crop/subsample family solved cold, held-out.")

    # ---- Control (shape ablation): the SAME content ES fit but with NO shape
    # write (identity shape rule from the seed). Every demo's predicted output
    # area then mismatches its true output area (fitness_shape penalizes them
    # all -> no ES signal), and at eval the predicted shape is wrong -> ~0.
    seed(7)
    var demos_ctl = make_demos("crop1")
    var cap_ctl = demos_capacity(demos_ctl)
    var state_ctl = alloc[Float32](SHAPEGEOM_DIM)
    ShapeGeomComposedMemory.seed(state_ctl)  # identity shape rule, NOT written
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
    var held_out_ctl = eval_held_out("crop1", state_ctl)
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
