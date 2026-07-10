from std.memory import alloc
from std.random import seed, randn_float64

# Run from the project root: `mojo run -I src tests/test_repertoire.mojo`.
from sandbox import (
    SB_CELLS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    CellSet,
    sandbox_rollout,
)
from novelty_es import NoveltyArchive, ns_es_run
from map_elites import EliteMap, me_mutation_run, me_emitter_run

# ==========================================================================
# B-POC-2 proof (Vision B rung 2): a PERSISTENT elite-per-cell repertoire
# (MAP-Elites over Go-Explore end-state cells) accumulates strictly more
# distinct, stored, REPLAYABLE skills than B-POC-1's transient NS-ES
# population at the exact same rollout budget. Three arms, equal budget in
# rollouts (the honest cost unit — every rollout in each arm gets exactly one
# archive/insert attempt):
#   A = canonical mutation MAP-Elites
#   B = ES-emitter MAP-Elites (novelty-ES bursts re-seeded from elites)
#   C = NS-ES (B-POC-1's exact configuration, rerun) — the baseline; its
#       product is transient, so its "repertoire" is the count of distinct
#       end-state cells it ever touched (an upper bound on what it could have
#       stored).
# Gates: max(A, B) repertoire vs C's end-state count; an absolute floor; a
# refinement proof (replacements happened AND made elites more direct); and a
# 100% replay check — every stored elite, re-executed, must re-reach exactly
# its bin (the repertoire is real, not an accounting artifact).
# ==========================================================================

# Arm C: B-POC-1's calibrated constants (test_novelty_coverage.mojo).
comptime NS_K = 5
comptime NS_ITERS = 400
comptime NS_N = 16
comptime NS_ALPHA = Float32(0.2)
comptime NS_SIGMA = Float32(0.4)
comptime INIT_SCALE = Float32(0.5)

# Arm A: batch size + mutation scale. The 2026-07-10 sweep peaked at 0.2 —
# half of NS_SIGMA: a mutation is a whole move in policy space (the child IS
# the perturbation), so it wants to be smaller than an ES probe whose job is
# only to measure a direction.
comptime ME_N = 16
comptime ME_SIGMA = Float32(0.2)
comptime ME_SEEDS = 5

# Arm B: emitter burst length + step size. The sweep peaked at alpha = 0.8 —
# 4x arm C's. NS-ES climbs one novelty landscape and wants a measured step;
# an emitter's product is the map, so big jumps that overshoot the novelty
# peak still land somewhere, and landing somewhere NEW is the whole game.
comptime EM_RESEED = 25
comptime EM_ALPHA = Float32(0.8)

# Pass thresholds (calibrated 2026-07-10; measured at seed 0: A = 1716,
# B = 4317 vs C = 1372 end-states -> ratio 3.15x, replay 0 failures — see
# JOURNAL for the sweep). Locked with >= 50% headroom below the measurement.
comptime MIN_RATIO = Float32(2.0)
comptime MIN_CELLS = 2000


def main() raises:
    seed(0)
    var task = SandboxTask()

    # --- Arm C: NS-ES (B-POC-1 rerun). Sets the shared rollout budget.
    var pop = alloc[Float32](NS_K * POLICY_DIM)
    for j in range(NS_K * POLICY_DIM):
        pop[j] = Float32(randn_float64(0.0, 1.0)) * INIT_SCALE
    var c_archive = NoveltyArchive()
    var c_cov = CellSet()
    var c_end = CellSet()
    var budget = ns_es_run(
        pop,
        task,
        c_archive,
        c_cov,
        c_end,
        NS_K,
        NS_ITERS,
        NS_N,
        NS_ALPHA,
        NS_SIGMA,
    )

    # --- Arm A: mutation MAP-Elites at the same budget.
    var a_map = EliteMap()
    var a_cov = CellSet()
    var a_end = CellSet()
    var a_rollouts = me_mutation_run(
        a_map, task, a_cov, a_end, budget, ME_N, ME_SIGMA, INIT_SCALE, ME_SEEDS
    )

    # --- Arm B: ES-emitter MAP-Elites at the same budget.
    var b_map = EliteMap()
    var b_archive = NoveltyArchive()
    var b_cov = CellSet()
    var b_end = CellSet()
    var b_rollouts = me_emitter_run(
        b_map,
        task,
        b_archive,
        b_cov,
        b_end,
        budget,
        EM_RESEED,
        NS_N,
        EM_ALPHA,
        NS_SIGMA,
        INIT_SCALE,
    )

    if a_rollouts != budget or b_rollouts != budget:
        raise Error(
            "ERROR: unequal budgets: C="
            + String(budget)
            + " A="
            + String(a_rollouts)
            + " B="
            + String(b_rollouts)
        )

    # --- Replay check (derailment/honesty): every stored elite, re-executed
    # from the task's start state, must re-reach exactly its bin key. Catches
    # any stripe-aliasing bug where the stored weights are not the weights
    # that earned the bin.
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logits = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var replay_fail = 0
    for i in range(a_map.count):
        var slot = a_map.filled[i]
        sandbox_rollout(
            a_map.weights + slot * POLICY_DIM,
            task,
            grid,
            obs,
            logits,
            bc,
            cells,
            True,
        )
        if cells[SB_T - 1] != a_map.keys[slot]:
            replay_fail += 1
    for i in range(b_map.count):
        var slot = b_map.filled[i]
        sandbox_rollout(
            b_map.weights + slot * POLICY_DIM,
            task,
            grid,
            obs,
            logits,
            bc,
            cells,
            True,
        )
        if cells[SB_T - 1] != b_map.keys[slot]:
            replay_fail += 1
    grid.free()
    obs.free()
    logits.free()
    bc.free()
    cells.free()

    var best = a_map.count
    if b_map.count > best:
        best = b_map.count
    var ratio = Float32(best) / Float32(c_end.count if c_end.count > 0 else 1)

    print("  budget (rollouts, all arms): ", budget)
    print(
        "  C  NS-ES        end-states:",
        c_end.count,
        " coverage:",
        c_cov.count,
    )
    print(
        "  A  mutation-ME  repertoire:",
        a_map.count,
        " coverage:",
        a_cov.count,
        " end-states:",
        a_end.count,
    )
    print(
        "     replaced:",
        a_map.replaced,
        " settle first->final:",
        a_map.mean_first_settle(),
        "->",
        a_map.mean_settle(),
        " distinctness:",
        a_map.mean_pairwise_bc(4096),
    )
    print(
        "  B  emitter-ME   repertoire:",
        b_map.count,
        " coverage:",
        b_cov.count,
        " end-states:",
        b_end.count,
    )
    print(
        "     replaced:",
        b_map.replaced,
        " settle first->final:",
        b_map.mean_first_settle(),
        "->",
        b_map.mean_settle(),
        " distinctness:",
        b_map.mean_pairwise_bc(4096),
    )
    print("  ratio (best map / C end-states):", ratio)
    print("  replay failures:", replay_fail)

    if replay_fail != 0:
        raise Error(
            "ERROR: "
            + String(replay_fail)
            + " stored elites failed to re-reach their bin on replay."
        )
    if best < MIN_CELLS:
        raise Error(
            "ERROR: best repertoire below the absolute floor ("
            + String(best)
            + " < "
            + String(MIN_CELLS)
            + ")."
        )
    if ratio < MIN_RATIO:
        raise Error(
            "ERROR: the persistent map did not beat the transient NS-ES"
            " baseline's end-state count by the required ratio (got "
            + String(ratio)
            + "x, need >= "
            + String(MIN_RATIO)
            + "x)."
        )
    # Refinement: the winning arm must have actually improved incumbents, and
    # the final repertoire must be more direct than its first fills were.
    var win_replaced = a_map.replaced
    var win_first = a_map.mean_first_settle()
    var win_final = a_map.mean_settle()
    if b_map.count > a_map.count:
        win_replaced = b_map.replaced
        win_first = b_map.mean_first_settle()
        win_final = b_map.mean_settle()
    if win_replaced <= 0:
        raise Error("ERROR: winning arm never improved an incumbent elite.")
    if win_final >= win_first:
        raise Error(
            "ERROR: replacements did not make the repertoire more direct"
            " (mean settle "
            + String(win_first)
            + " -> "
            + String(win_final)
            + ")."
        )

    pop.free()
    print(
        "Repertoire test passed: the persistent elite-per-cell map stores"
        " strictly more replayable skills than the transient NS-ES baseline"
        " at equal budget (B-POC-2)."
    )
