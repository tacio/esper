from std.memory import alloc, memset_zero
from std.random import seed, randn_float64

# Run from the project root: `mojo run -I src tests/test_empowerment.mojo`.
from sandbox import (
    SB_CELLS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    CellSet,
    sandbox_rollout_state,
)
from novelty_es import NoveltyArchive
from map_elites import EliteMap, me_emitter_run, settle_tick
from empowerment import empowerment, emp_es_run, EMP_SET_CAP

# ==========================================================================
# B-POC-2.5 proof (Vision B rung 2's deferred half): EXACT empowerment as a
# second, learned-part-free intrinsic signal. The sandbox is deterministic,
# so Blahut–Arimoto collapses: n-step empowerment = log2(#distinct states
# reachable in n steps), computed by exhaustive enumeration — no learned
# parts, no archive, stationary. Three checks:
#   1. Signal sanity, computed directly: a corner state has strictly lower
#      empowerment than an open-field state; 0 < E <= n*log2(6).
#   2. The GATED claim: an ES emitter whose only fitness is empowerment
#      builds a repertoire >= 2x an equal-rollout-budget random-policy map's
#      (directed search from a goal-free stationary signal), 100% replayable.
#   3. The REPORTED head-to-head (ungated, honest-measurement style): the
#      novelty emitter (B-POC-2's winner, its locked constants) at the same
#      budget — whichever way it goes, plus the concentration probe (mean
#      elite empowerment per arm) and the uncharged enumeration-tick caveat:
#      budgets are denominated in ROLLOUTS, and one empowerment evaluation
#      spends ~6+36+...+6^n hidden world ticks (printed, not charged).
# ==========================================================================

# Shared budget: B-POC-2's 13,205 rollouts (= 5 + 400 x (2*16+1), the arm-C
# NS-ES consumption; kept as a constant so this test does not re-run NS-ES).
comptime BUDGET = 13205
comptime INIT_SCALE = Float32(0.5)

# Empowerment arm (calibrated 2026-07-10): horizon 4 (1,554 enumeration ticks
# per evaluation; 6^5 leaves would overflow EMP_SET_CAP's 50% load), reseed
# every 5 iterations, alpha = 0.2, sigma = 0.8. Empowerment is a CONCENTRATING
# signal (it scores optionality, not newness) — left to run long it parks the
# emitter in one high-E region (reseed 25: 382 elites, BELOW random's 448);
# frequent uniform-elite restarts plus wide probes are what turn it into an
# exploration signal (1,513 elites at these values).
comptime EMP_H = 4
comptime EMP_RESEED = 5
comptime EMP_ALPHA = Float32(0.2)
comptime EMP_SIGMA = Float32(0.8)

# Novelty-emitter reference: B-POC-2's locked constants (test_repertoire).
comptime NOV_RESEED = 25
comptime NOV_ALPHA = Float32(0.8)
comptime NOV_SIGMA = Float32(0.4)

# Pass thresholds (calibrated 2026-07-10; measured at seed 0: empowerment map
# 1,513 vs random map 448 -> 3.38x, replay 0 failures). Locked with >= 40%
# headroom below the measurement.
comptime MIN_RATIO = Float32(2.0)
comptime MIN_CELLS = 750


# Mean exact empowerment over a map's stored elites (first `cap` of them):
# re-roll each elite to its final state, evaluate. The concentration probe —
# does an arm's repertoire sit in higher-optionality states?
def mean_elite_emp(
    emap: EliteMap, task: SandboxTask, horizon: Int, cap: Int
) -> Float32:
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logits = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var lvl = alloc[Float32]((horizon + 1) * SB_CELLS)
    var seen = alloc[Int64](EMP_SET_CAP)
    var ticks = alloc[Int](1)
    ticks[0] = 0
    var n = emap.count
    if n > cap:
        n = cap
    var total = Float32(0.0)
    for i in range(n):
        var slot = emap.filled[i]
        var r = 0
        var c = 0
        var b = 0
        sandbox_rollout_state(
            emap.weights + slot * POLICY_DIM,
            task,
            grid,
            obs,
            logits,
            bc,
            cells,
            False,
            r,
            c,
            b,
        )
        total += empowerment(grid, r, c, b, task, horizon, lvl, seen, ticks)
    grid.free()
    obs.free()
    logits.free()
    bc.free()
    cells.free()
    lvl.free()
    seen.free()
    ticks.free()
    if n == 0:
        return 0.0
    return total / Float32(n)


def main() raises:
    seed(0)
    var task = SandboxTask()

    # --- 1. Signal sanity, no search: corner vs open field on an empty grid.
    var grid0 = alloc[Float32](SB_CELLS)
    memset_zero(grid0, SB_CELLS)
    var lvl = alloc[Float32]((EMP_H + 1) * SB_CELLS)
    var seen = alloc[Int64](EMP_SET_CAP)
    var ticks0 = alloc[Int](1)
    ticks0[0] = 0
    var e_corner = empowerment(grid0, 0, 0, 1, task, EMP_H, lvl, seen, ticks0)
    var e_center = empowerment(grid0, 8, 8, 1, task, EMP_H, lvl, seen, ticks0)
    var e_max = Float32(EMP_H) * Float32(2.5849626)  # n * log2(6)
    print("  empowerment sanity: corner", e_corner, " center", e_center)
    if not (e_corner < e_center):
        raise Error("ERROR: corner state not less empowered than open field.")
    if e_corner <= 0.0 or e_center > e_max:
        raise Error("ERROR: empowerment out of (0, n*log2(6)] bounds.")
    grid0.free()
    lvl.free()
    seen.free()
    ticks0.free()

    # --- 2. Empowerment-ES arm.
    var e_map = EliteMap()
    var e_cov = CellSet()
    var e_end = CellSet()
    var enum_ticks = 0
    var e_rollouts = emp_es_run(
        e_map,
        task,
        e_cov,
        e_end,
        BUDGET,
        EMP_RESEED,
        16,
        EMP_ALPHA,
        EMP_SIGMA,
        INIT_SCALE,
        EMP_H,
        enum_ticks,
    )

    # --- 3. Novelty-emitter reference (B-POC-2's winner), same budget.
    var n_map = EliteMap()
    var n_archive = NoveltyArchive()
    var n_cov = CellSet()
    var n_end = CellSet()
    var n_rollouts = me_emitter_run(
        n_map,
        task,
        n_archive,
        n_cov,
        n_end,
        BUDGET,
        NOV_RESEED,
        16,
        NOV_ALPHA,
        NOV_SIGMA,
        INIT_SCALE,
    )

    # --- 4. Random-policy baseline map (the gated comparison), same budget.
    var r_map = EliteMap()
    var r_cov = CellSet()
    var w = alloc[Float32](POLICY_DIM)
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logits = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    for _ in range(BUDGET):
        for j in range(POLICY_DIM):
            w[j] = Float32(randn_float64(0.0, 1.0)) * INIT_SCALE
        var fr = 0
        var fc = 0
        var fb = 0
        sandbox_rollout_state(
            w, task, grid, obs, logits, bc, cells, True, fr, fc, fb
        )
        for t in range(SB_T):
            _ = r_cov.insert(cells[t])
        _ = r_map.insert(cells[SB_T - 1], settle_tick(cells), w, bc)
    w.free()

    if e_rollouts != BUDGET or n_rollouts != BUDGET:
        raise Error("ERROR: unequal rollout budgets across arms.")

    # --- 5. Replay check on the empowerment map: every stored elite must
    # re-reach exactly its bin.
    var replay_fail = 0
    for i in range(e_map.count):
        var slot = e_map.filled[i]
        var fr = 0
        var fc = 0
        var fb = 0
        sandbox_rollout_state(
            e_map.weights + slot * POLICY_DIM,
            task,
            grid,
            obs,
            logits,
            bc,
            cells,
            True,
            fr,
            fc,
            fb,
        )
        if cells[SB_T - 1] != e_map.keys[slot]:
            replay_fail += 1
    grid.free()
    obs.free()
    logits.free()
    bc.free()
    cells.free()

    # --- 6. The concentration probe (reported): mean elite empowerment.
    var me_e = mean_elite_emp(e_map, task, EMP_H, 400)
    var me_n = mean_elite_emp(n_map, task, EMP_H, 400)
    var me_r = mean_elite_emp(r_map, task, EMP_H, 400)

    var ratio = Float32(e_map.count) / Float32(
        r_map.count if r_map.count > 0 else 1
    )
    print("  budget (rollouts, all arms):", BUDGET)
    print(
        "  empowerment-ES repertoire:",
        e_map.count,
        " coverage:",
        e_cov.count,
        " meanE:",
        me_e,
    )
    print(
        "  novelty-ES     repertoire:",
        n_map.count,
        " coverage:",
        n_cov.count,
        " meanE:",
        me_n,
    )
    print(
        "  random-policy  repertoire:",
        r_map.count,
        " coverage:",
        r_cov.count,
        " meanE:",
        me_r,
    )
    print("  ratio (empowerment / random):", ratio)
    print(
        "  distinctness emp/nov:",
        e_map.mean_pairwise_bc(4096),
        "/",
        n_map.mean_pairwise_bc(4096),
    )
    print(
        "  uncharged enumeration ticks:",
        enum_ticks,
        " (vs",
        BUDGET * SB_T,
        "charged rollout ticks)",
    )
    print("  replay failures:", replay_fail)

    if replay_fail != 0:
        raise Error(
            "ERROR: "
            + String(replay_fail)
            + " empowerment elites failed to re-reach their bin on replay."
        )
    if e_map.count < MIN_CELLS:
        raise Error(
            "ERROR: empowerment repertoire below the absolute floor ("
            + String(e_map.count)
            + " < "
            + String(MIN_CELLS)
            + ")."
        )
    if ratio < MIN_RATIO:
        raise Error(
            "ERROR: empowerment-driven search did not beat the equal-budget"
            " random-policy repertoire by the required ratio (got "
            + String(ratio)
            + "x, need >= "
            + String(MIN_RATIO)
            + "x)."
        )

    print(
        "Empowerment test passed: exact empowerment — a learned-part-free,"
        " archive-free intrinsic signal — drives repertoire growth well"
        " beyond an equal-budget random baseline (B-POC-2.5)."
    )
