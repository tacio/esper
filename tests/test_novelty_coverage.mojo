from std.memory import alloc
from std.random import seed, randn_float64, random_float64

# Run from the project root: `mojo run -I src tests/test_novelty_coverage.mojo`.
from sandbox import (
    SB_CELLS,
    SB_COLS,
    SB_ROWS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    CellSet,
    sandbox_step,
    sandbox_rollout,
    sandbox_obs,
    sandbox_cell_key,
)
from novelty_es import NoveltyArchive, ns_es_run

# ==========================================================================
# B-POC-1 proof (Vision B rung 1): with ZERO hand-coded goals — no reward
# channel exists in the sandbox — an NS-ES meta-population whose only fitness
# is self-generated novelty covers strictly more of the world's state space
# than a random-policy baseline granted the EXACT same rollout budget. The
# metric is an uncheatable count of distinct Go-Explore-style cells touched
# across every rollout of each arm. Same policy class, same per-rollout cost,
# same deterministic world — the only difference is the search.
# ==========================================================================

comptime NS_K = 5
# Budget split calibrated 2026-07-10: at equal total rollouts, MORE iterations
# with FEWER samples per step (400x16 over 200x32) covers substantially more —
# each iteration adds an archive entry and re-aims the search, and novelty
# needs direction more than gradient precision.
comptime NS_ITERS = 400
comptime NS_N = 16
comptime NS_ALPHA = Float32(0.2)
# Sigma must be large relative to the 0.5 init scale: argmax is a step
# function, so perturbations must actually flip actions somewhere along the
# rollout or the antithetic difference is zero (0.4 measured best in the
# 2026-07-10 sweep).
comptime NS_SIGMA = Float32(0.4)
comptime INIT_SCALE = Float32(0.5)

# Pass thresholds (calibrated 2026-07-10; measured 4.88x coverage / 2.98x
# end-states / 10578 cells at seed 0 — see JOURNAL for the margins). The
# ratios carry the comparative claim (raw visitation AND distinct reachable
# end-states — the latter is what novelty over an end-state BC actually
# optimizes, and the repertoire currency of B-POC-2/4); the absolute floor
# guards the degenerate pass where both arms are near zero (e.g. a frozen
# world).
comptime MIN_RATIO = Float32(3.0)
comptime MIN_END_RATIO = Float32(2.0)
comptime MIN_CELLS = 1000


# The world's one dynamics rule, checked inline (a free unit test of
# sandbox_step): a painted block falls one cell per tick and rests on the
# floor.
def check_world() raises:
    var task = SandboxTask()
    var grid = alloc[Float32](SB_CELLS)
    for i in range(SB_CELLS):
        grid[i] = task.grid[i]
    var r = task.start_r
    var c = task.start_c
    var brush = task.start_brush
    # Paint at (8,8); the same tick's gravity phase drops the block to (9,8).
    sandbox_step(grid, r, c, brush, task.grav_dir, task.grav_rate, 4)
    if grid[9 * SB_COLS + 8] != 1.0 or grid[8 * SB_COLS + 8] != 0.0:
        raise Error("ERROR: painted block did not fall one cell on tick 1")
    # Six more ticks (brush cycles — no grid writes) land it on the floor.
    for _ in range(6):
        sandbox_step(grid, r, c, brush, task.grav_dir, task.grav_rate, 5)
    if grid[15 * SB_COLS + 8] != 1.0:
        raise Error("ERROR: painted block did not settle on the floor row")
    for rr in range(SB_ROWS - 1):
        if grid[rr * SB_COLS + 8] != 0.0:
            raise Error("ERROR: falling block left a trail")
    grid.free()


def fill_randn(w: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    for j in range(n):
        w[j] = Float32(randn_float64(0.0, 1.0)) * INIT_SCALE


def main() raises:
    seed(0)
    check_world()

    var task = SandboxTask()

    # --- NS-ES arm: K novelty-seeking agents, shared archive, no reward.
    var pop = alloc[Float32](NS_K * POLICY_DIM)
    fill_randn(pop, NS_K * POLICY_DIM)
    var archive = NoveltyArchive()
    var ns_cov = CellSet()
    var ns_end = CellSet()
    var budget = ns_es_run(
        pop,
        task,
        archive,
        ns_cov,
        ns_end,
        NS_K,
        NS_ITERS,
        NS_N,
        NS_ALPHA,
        NS_SIGMA,
    )
    var cov_ns = ns_cov.count

    # --- Random-POLICY baseline at the exact same budget: fresh weight vector
    # per rollout, same distribution as the NS-ES inits — same policy class,
    # same per-rollout cost; the only difference is the search.
    var w = alloc[Float32](POLICY_DIM)
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logits = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var rand_cov = CellSet()
    var rand_end = CellSet()
    for _ in range(budget):
        fill_randn(w, POLICY_DIM)
        sandbox_rollout(w, task, grid, obs, logits, bc, cells, True)
        for t in range(SB_T):
            _ = rand_cov.insert(cells[t])
        _ = rand_end.insert(cells[SB_T - 1])
    var cov_rand = rand_cov.count

    # --- Random-ACTION reference (printed, not gated): uniform actions each
    # tick, same budget — the weaker of the two natural baselines.
    var act_cov = CellSet()
    var act_end = CellSet()
    for _ in range(budget):
        var r = task.start_r
        var c = task.start_c
        var brush = task.start_brush
        for i in range(SB_CELLS):
            grid[i] = task.grid[i]
        var last_key = Int64(0)
        for t in range(SB_T):
            var a = Int(random_float64(0.0, Float64(SB_ACTIONS)))
            if a >= SB_ACTIONS:
                a = SB_ACTIONS - 1
            sandbox_step(grid, r, c, brush, task.grav_dir, task.grav_rate, a)
            last_key = sandbox_cell_key(grid, r, c)
            _ = act_cov.insert(last_key)
        _ = act_end.insert(last_key)

    var ratio = Float32(cov_ns) / Float32(cov_rand if cov_rand > 0 else 1)
    var end_ratio = Float32(ns_end.count) / Float32(
        rand_end.count if rand_end.count > 0 else 1
    )
    print("  budget (rollouts, both arms):", budget)
    print(
        "  NS-ES coverage:              ", cov_ns, " end-states:", ns_end.count
    )
    print(
        "  random-policy coverage:      ",
        cov_rand,
        " end-states:",
        rand_end.count,
    )
    print(
        "  random-action coverage (ref):",
        act_cov.count,
        " end-states:",
        act_end.count,
    )
    print("  ratio (NS-ES / random-policy):", ratio, " end-ratio:", end_ratio)
    print("  archive size:                ", archive.count)

    if cov_ns < MIN_CELLS:
        raise Error(
            "ERROR: NS-ES coverage below the absolute floor ("
            + String(cov_ns)
            + " < "
            + String(MIN_CELLS)
            + ")."
        )
    if ratio < MIN_RATIO:
        raise Error(
            "ERROR: NS-ES did not beat the equal-budget random-policy"
            " baseline by the required coverage ratio (got "
            + String(ratio)
            + "x, need >= "
            + String(MIN_RATIO)
            + "x)."
        )
    if end_ratio < MIN_END_RATIO:
        raise Error(
            "ERROR: NS-ES did not beat the equal-budget random-policy"
            " baseline by the required END-STATE ratio (got "
            + String(end_ratio)
            + "x, need >= "
            + String(MIN_END_RATIO)
            + "x)."
        )

    pop.free()
    w.free()
    grid.free()
    obs.free()
    logits.free()
    bc.free()
    cells.free()

    print(
        "Novelty-coverage test passed: self-generated novelty alone covers"
        " the sandbox far beyond an equal-budget random baseline (B-POC-1)."
    )
