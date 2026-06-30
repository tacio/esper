from std.sys import argv
from std.memory import alloc, UnsafePointer
from std.random import seed

from hope import OP_DIM, apply_operator, seed_identity_operator
from memory import OperatorMemory
from esper_evolution import (
    ESWorkspace,
    fit_operator,
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
# `synth_tasks.generate_task_groups`, or compiled from real ARC-AGI tasks). For
# every task we fit the learned operator ON THE TRAIN PAIRS ONLY, then score it
# on the held-out TEST pair(s) it never saw. A task is solved iff every test pair
# matches above SOLVE_THRESHOLD. This replaces the old benchmark.mojo, which
# memorized a known target grid — held-out generalization is uncheatable by
# memorization. We also report the train-fit vs held-out gap: a small gap with
# high held-out is genuine learning; a large gap is overfitting.
#
# Run from the project root, e.g.:
#   python - <<'PY'
#   import sys; sys.path.insert(0,"src")
#   from synth_tasks import generate_task_groups
#   generate_task_groups("flip_h", "data_bin", num_tasks=4, n_train=6,
#                        rows=4, cols=4, seed=0)
#   PY
#   mojo run -I src src/arc_solve.mojo data_bin/flip_h_*.task
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
def solve_task(task_path: String) raises -> Float32:
    var task = load_arc_task(task_path)

    # Deterministic per-task RNG (order/shard invariant) — see SOLVE_SEED.
    seed(SOLVE_SEED)

    # Operator-output scratch must hold the largest grid the operator touches.
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

    var fast = alloc[Float32](OP_DIM)
    var slow = alloc[Float32](OP_DIM)
    seed_identity_operator(fast)
    seed_identity_operator(slow)

    var workspace = ESWorkspace[OperatorMemory](capacity)
    fit_operator[OperatorMemory](
        fast,
        workspace,
        slow,
        task.train,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )

    var pred = alloc[Float32](capacity)

    # Held-out: minimum exact-match over all test pairs. The operator is
    # same-shape, so a test pair whose output dims differ from its input dims is
    # inexpressible — it honestly scores 0 (and skipping the compare avoids an
    # out-of-bounds read across the mismatched buffers). Real ARC-AGI tasks are
    # often shape-changing; this is where the honest number stays low.
    var held_out = Float32(1.0)
    for i in range(len(task.test)):
        var rows = task.test[i].input_grid.rows
        var cols = task.test[i].input_grid.cols
        if task.test[i].output_grid.size() != rows * cols:
            held_out = 0.0
            continue
        apply_operator(fast, task.test[i].input_grid.data, pred, rows, cols)
        var m = exact_match(pred, task.test[i].output_grid.data, rows * cols)
        if m < held_out:
            held_out = m

    # Train fit (how well the fitted operator reproduces the demos it saw). Same
    # same-shape guard as the held-out scoring above.
    var train_sum = Float32(0.0)
    for i in range(len(task.train)):
        var rows = task.train[i].input_grid.rows
        var cols = task.train[i].input_grid.cols
        if task.train[i].output_grid.size() != rows * cols:
            continue
        apply_operator(fast, task.train[i].input_grid.data, pred, rows, cols)
        train_sum += exact_match(
            pred, task.train[i].output_grid.data, rows * cols
        )
    var train_fit = train_sum / Float32(len(task.train))

    print(
        "  task:",
        task_path,
        " held-out:",
        held_out,
        " train:",
        train_fit,
        " gap:",
        train_fit - held_out,
    )

    pred.free()
    fast.free()
    slow.free()
    return held_out


def main() raises:
    seed(0)

    var args = argv()

    # `--report` (honest-eval mode) suppresses the raise-on-zero: a 0% solve
    # rate on the real ARC-AGI corpus is a legitimate honest result, not a CI
    # regression. Without the flag (the synth-bundle path in run_tests.sh, where
    # some tasks MUST solve) a 0 solve rate raises loudly. The flag must be the
    # first argument; the rest are `.task` paths.
    var report_only = len(args) > 1 and String(args[1]) == "--report"
    var first = 2 if report_only else 1

    if len(args) <= first:
        print(
            "Usage: mojo run -I src src/arc_solve.mojo [--report] <task.task>"
            " [more ...]"
        )
        print(
            "Generate task bundles first via synth_tasks.generate_task_groups,"
            " or arc_compiler.py for real ARC."
        )
        return

    var total = len(args) - first
    var solved = 0
    var held_sum = Float32(0.0)

    print("Esper held-out generalization over", total, "task(s)")
    for idx in range(first, len(args)):
        var held_out = solve_task(String(args[idx]))
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
