from std.sys import argv, has_accelerator
from std.gpu.host import DeviceContext
from std.memory import alloc, UnsafePointer
from std.random import seed
from std.math import round
from std.time import perf_counter_ns

from memory_composed import (
    LocalWriteComposedMemory,
    LOCALWRITE_DIM,
    ShapeGeomColorComposedMemory,
    SHAPEGEOMCOLOR_DIM,
)
from esper_evolution import (
    fit_local,
    fit_shape_color,
    FIT_N,
    FIT_ALPHA0,
    FIT_ALPHA1,
    FIT_SIGMA0,
    FIT_SIGMA1,
    FIT_ITERS,
    FIT_REG,
)
from arc_io import load_arc_task, exact_match

# ==========================================================================
# Esper held-out generalization driver.
#
# Each argument is a path to a `.task` bundle (produced by
# `synth_tasks.generate_task_groups` / `generate_shape_task_groups`, or
# compiled from real ARC-AGI tasks). For every task we fit an EMERGENT
# composed memory ON THE TRAIN PAIRS ONLY, then score it on the held-out TEST
# pair(s) it never saw. Which memory is a DRIVER-LEVEL dispatch on a
# closed-form observable of the demos (NOT a runtime memory-selector — the
# other memory provably scores 0 on the dispatched class):
# - all demo dims equal   → GeomColorComposedMemory (block 5: count-signature
#   colour self-write + windowed attention-gather geometry ES),
# - any demo changes dims → ShapeGeomColorComposedMemory (the shape seam: per-axis
#   affine shape rule written closed-form from the demo dims + the toroidal
#   normalized-query gather, two-frame multi-start fit, composed with a written
#   colour table V — Rung C; V = identity makes it byte-identical to the pure
#   shape+geometry path). Known, documented limits on the real corpus:
#   the shape rule is affine in the INPUT DIMS — tasks whose output size
#   depends on grid CONTENT mispredict and honestly score 0.
# A task is solved iff every test pair matches above SOLVE_THRESHOLD. Held-out
# generalization is uncheatable by memorization. We also report the train-fit
# vs held-out gap: `train ~ 1, held-out ~ 0` is memorize-not-generalize;
# `train ~ 0` is an expressiveness gap — the breakdown that prioritizes the
# roadmap. Each per-task line carries a trailing `mem:` marker (same | shape)
# so corpus runs break down by dispatch for free (appended AFTER the existing
# fields — eval_parallel.sh reads held-out positionally as field 4).
#
# Flags (must precede the task paths):
#   --report        honest-eval mode: 0 solved is a legitimate number, no raise.
#   --fit N ITERS   override the ES budget (default: the full FIT_N/FIT_ITERS
#                   the synth proofs use). The real-corpus runs use a smaller,
#                   DOCUMENTED budget — uniform across tasks, reported with the
#                   number — because full-budget fits at 30x30 ARC scale are
#                   compute-prohibitive on CPU. CI's synth path passes no flag
#                   and keeps the full proven budget.
#   --cpu           force the CPU reference ES path on an accelerator host
#                   (A/B benchmarking; the default on such hosts is the GPU-
#                   batched fitness, rungs G1-G2). No-op on CPU-only hosts.
#                   The report header prints which path ran (self-documenting).
#
# Progress: every task emits `INFO [+<elapsed>s] task k/n start|done ...`
# lines (the done line carries per-task seconds, the running average and a
# per-PROCESS eta — per-shard under eval_parallel.sh). INFO lines are additive
# monitoring output: the positional `  task:` result lines are unchanged and
# remain the only lines consumers (eval_parallel's grep) read.
#
# Run from the project root, e.g.:
#   mojo run -I src src/arc_solve.mojo data_bin/flip_h_*.task
#   mojo run -I src src/arc_solve.mojo --report --fit 64 1500 data_bin/arc2_train/*.task
# ==========================================================================

comptime SOLVE_THRESHOLD = Float32(0.99)

# Per-task RNG seed. Seeding before each task's fit makes a task's result depend
# only on the task itself, never on its position in the argv list — so the
# benchmark number is reproducible and invariant under how the corpus is split
# across workers (see eval_parallel.sh). Without this, one shared stream seeded
# once in main() makes every task's stochastic ES fit depend on ordering.
comptime SOLVE_SEED = 0


# Fit one task on its train pairs and return its held-out exact-match (the
# minimum over all test pairs — solved iff every test pair matches). Prints the
# per-task held-out, the train fit, and their gap.
# Rung A near-miss audit (measure-first): print a same-shape test pair's input,
# the fitted memory's PREDICTION, and the truth as flat integer grids, so the
# offline clustering tool (tools/near_miss_audit.py) can characterize WHERE the
# few wrong cells fall (border / localized region / colour swap / scattered).
# Only emitted in `--diff` mode and only for the same-shape path (Rung A scope);
# adds EXTRA lines, never touching the positional per-task line eval_parallel.sh
# reads.
def _dump_grid(label: String, p: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var s = String("  DIFF ") + label + String(":")
    for k in range(n):
        s += String(" ") + String(Int(round(p[k])))
    print(s)


def solve_task(
    task_path: String,
    n_fit: Int,
    iters: Int,
    dump_diff: Bool = False,
    use_gpu: Bool = True,
) raises -> Float32:
    var task = load_arc_task(task_path)

    # Deterministic per-task RNG (order/shard invariant) — see SOLVE_SEED.
    seed(SOLVE_SEED)

    # Dispatch on the demos (see the header): any train pair whose DIMS differ
    # routes to the shape memory.
    var shape_task = False
    for i in range(len(task.train)):
        if (
            task.train[i].input_grid.rows != task.train[i].output_grid.rows
            or task.train[i].input_grid.cols != task.train[i].output_grid.cols
        ):
            shape_task = True

    # Same-shape dispatch + EVERY test pair shape-changing ⇒ held-out is 0 by
    # construction (the same-shape memory cannot express any test pair), so
    # the fit cannot change the result — skip it. Exact w.r.t. the solve
    # metric. (Shape-dispatched tasks are always fit.)
    if not shape_task:
        var any_test_same_shape = False
        for i in range(len(task.test)):
            if (
                task.test[i].output_grid.size()
                == task.test[i].input_grid.size()
            ):
                any_test_same_shape = True
        if not any_test_same_shape:
            print(
                "  task:",
                task_path,
                " held-out: 0.0  train: 0.0  gap: 0.0  mem: same",
            )
            return 0.0

    # Forward scratch must hold the largest grid the memory touches.
    var capacity = 1
    for i in range(len(task.train)):
        if task.train[i].input_grid.size() > capacity:
            capacity = task.train[i].input_grid.size()
        if task.train[i].output_grid.size() > capacity:
            capacity = task.train[i].output_grid.size()
    for i in range(len(task.test)):
        if task.test[i].input_grid.size() > capacity:
            capacity = task.test[i].input_grid.size()
        if task.test[i].output_grid.size() > capacity:
            capacity = task.test[i].output_grid.size()

    # Cold per-task fit of the dispatched composed memory.
    var state_dim = LOCALWRITE_DIM
    if shape_task:
        state_dim = SHAPEGEOMCOLOR_DIM
    var state = alloc[Float32](state_dim)
    if shape_task:
        # Shape rule + colour table written closed-form from the demos, then the
        # two-frame multi-start geometry fit on the V-pre-mapped output-shaped
        # gather (Rung C).
        ShapeGeomColorComposedMemory.seed(state)
        fit_shape_color(
            state,
            task.train,
            capacity,
            n_fit,
            FIT_ALPHA0,
            FIT_ALPHA1,
            FIT_SIGMA0,
            FIT_SIGMA1,
            iters,
            FIT_REG,
            use_gpu,
        )
    else:
        # Colour table written from the demos, the annealed geometry ES on the
        # V-pre-mapped demos, then the closed-form local-content write on the
        # residual (Rung A). Byte-identical to the old GeomColor path when the
        # local table stays empty (the strict-superset gate).
        LocalWriteComposedMemory.seed(state)
        fit_local(
            state,
            task.train,
            capacity,
            n_fit,
            FIT_ALPHA0,
            FIT_ALPHA1,
            FIT_SIGMA0,
            FIT_SIGMA1,
            iters,
            FIT_REG,
            use_gpu,
        )

    var pred = alloc[Float32](capacity)

    # Held-out: minimum exact-match over all test pairs.
    # - shape path: the memory PREDICTS its output dims from the written rule;
    #   a predicted/true dims mismatch scores 0 for that pair (never applied —
    #   no OOB). This is where content-dependent output sizes honestly fail.
    # - same-shape path: a test pair whose output dims differ from its input
    #   dims is inexpressible — it honestly scores 0 (and skipping the compare
    #   avoids an out-of-bounds read across the mismatched buffers).
    var held_out = Float32(1.0)
    for i in range(len(task.test)):
        var m = Float32(0.0)
        if shape_task:
            var pr = ShapeGeomColorComposedMemory.out_rows(
                state, task.test[i].input_grid
            )
            var pc = ShapeGeomColorComposedMemory.out_cols(
                state, task.test[i].input_grid
            )
            if (
                pr == task.test[i].output_grid.rows
                and pc == task.test[i].output_grid.cols
            ):
                ShapeGeomColorComposedMemory.apply(
                    state, task.test[i].input_grid, pr, pc, pred
                )
                m = exact_match(pred, task.test[i].output_grid.data, pr * pc)
        else:
            var rows = task.test[i].input_grid.rows
            var cols = task.test[i].input_grid.cols
            if task.test[i].output_grid.size() == rows * cols:
                LocalWriteComposedMemory.apply(
                    state, task.test[i].input_grid, pred
                )
                m = exact_match(
                    pred, task.test[i].output_grid.data, rows * cols
                )
                if dump_diff and i == 0:
                    print(
                        "  DIFF task:", task_path, "rows:", rows, "cols:", cols
                    )
                    _dump_grid("in", task.test[i].input_grid.data, rows * cols)
                    _dump_grid("pred", pred, rows * cols)
                    _dump_grid(
                        "true", task.test[i].output_grid.data, rows * cols
                    )
        if m < held_out:
            held_out = m

    # Train fit (how well the fitted memory reproduces the demos it saw). Same
    # dims guards as the held-out scoring above.
    var train_sum = Float32(0.0)
    for i in range(len(task.train)):
        if shape_task:
            var pr = ShapeGeomColorComposedMemory.out_rows(
                state, task.train[i].input_grid
            )
            var pc = ShapeGeomColorComposedMemory.out_cols(
                state, task.train[i].input_grid
            )
            if (
                pr != task.train[i].output_grid.rows
                or pc != task.train[i].output_grid.cols
            ):
                continue
            ShapeGeomColorComposedMemory.apply(
                state, task.train[i].input_grid, pr, pc, pred
            )
            train_sum += exact_match(
                pred, task.train[i].output_grid.data, pr * pc
            )
        else:
            var rows = task.train[i].input_grid.rows
            var cols = task.train[i].input_grid.cols
            if task.train[i].output_grid.size() != rows * cols:
                continue
            LocalWriteComposedMemory.apply(
                state, task.train[i].input_grid, pred
            )
            train_sum += exact_match(
                pred, task.train[i].output_grid.data, rows * cols
            )
    var train_fit = train_sum / Float32(len(task.train))

    var mem_name = String("same")
    if shape_task:
        mem_name = String("shape")
    print(
        "  task:",
        task_path,
        " held-out:",
        held_out,
        " train:",
        train_fit,
        " gap:",
        train_fit - held_out,
        " mem:",
        mem_name,
    )

    pred.free()
    state.free()
    return held_out


def main() raises:
    seed(0)

    var args = argv()

    # Flag parsing: flags must precede the `.task` paths (see the header).
    var report_only = False
    var dump_diff = False
    var use_gpu = True
    var n_fit = FIT_N
    var iters = FIT_ITERS
    var first = 1
    while first < len(args):
        if String(args[first]) == "--report":
            report_only = True
            first += 1
        elif String(args[first]) == "--cpu":
            use_gpu = False
            first += 1
        elif String(args[first]) == "--diff":
            # Near-miss audit: emit input/pred/truth grids (report-implied, so a
            # 0-solved diagnostic run never raises).
            dump_diff = True
            report_only = True
            first += 1
        elif String(args[first]) == "--fit" and first + 2 < len(args):
            n_fit = Int(String(args[first + 1]))
            iters = Int(String(args[first + 2]))
            first += 3
        else:
            break

    if len(args) <= first:
        print(
            "Usage: mojo run -I src src/arc_solve.mojo [--report]"
            " [--fit N ITERS] <task.task> [more ...]"
        )
        print(
            "Generate task bundles first via synth_tasks.generate_task_groups,"
            " or arc_compiler.py for real ARC."
        )
        return

    var total = len(args) - first
    var solved = 0
    var held_sum = Float32(0.0)

    # Self-documenting compute path: which fitness backend this run used.
    var backend = String("cpu")
    comptime if has_accelerator():
        if use_gpu:
            var ctx = DeviceContext()
            backend = String("gpu (") + ctx.name() + String(")")
    print("Esper held-out generalization over", total, "task(s)")
    print("  fitness backend:", backend)
    var t_start = perf_counter_ns()
    for idx in range(first, len(args)):
        var k = idx - first + 1
        var t_task = perf_counter_ns()
        print(
            "INFO [+"
            + String(Int((t_task - t_start) // 1_000_000_000))
            + "s] task "
            + String(k)
            + "/"
            + String(total)
            + " start: "
            + String(args[idx])
        )
        var held_out = solve_task(
            String(args[idx]), n_fit, iters, dump_diff, use_gpu
        )
        var t_now = perf_counter_ns()
        var task_s = Float64(t_now - t_task) / 1.0e9
        var avg_s = Float64(t_now - t_start) / 1.0e9 / Float64(k)
        var eta_s = avg_s * Float64(total - k)
        print(
            "INFO [+"
            + String(Int((t_now - t_start) // 1_000_000_000))
            + "s] task "
            + String(k)
            + "/"
            + String(total)
            + " done ("
            + String(Int(round(task_s)))
            + "s, avg "
            + String(Int(round(avg_s)))
            + "s/task, eta ~"
            + String(Int(round(eta_s)))
            + "s)"
        )
        held_sum += held_out
        if held_out >= SOLVE_THRESHOLD:
            solved += 1

    var solve_rate = Float32(solved) / Float32(total) * 100.0
    print("--------------------------------------------------")
    print(
        "Solved",
        solved,
        "/",
        total,
        " (solve rate:",
        solve_rate,
        "%, mean held-out:",
        held_sum / Float32(total),
        ")",
    )

    # Non-zero exit (raised error) if nothing generalized, so CI fails loudly —
    # unless `--report` (honest real-ARC eval, where 0% is an honest number).
    if solved == 0 and not report_only:
        raise Error(
            "ERROR: 0 tasks solved on held-out inputs; the engine is not"
            " generalizing."
        )
