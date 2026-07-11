# suite-tier: full
from std.random import seed
from std.memory import alloc

# Run from the project root: `mojo run -I src tests/test_transfer.mojo`.
from sandbox import (
    SB_CELLS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    SandboxPolicyMemory,
    CellSet,
    sandbox_rollout,
)
from novelty_es import NoveltyArchive
from map_elites import EliteMap, me_emitter_run, load_elite_map
from esper_evolution import ESWorkspace
from transfer import (
    ComposeMemory,
    COMPOSE_DIM,
    run_family,
    gen_family_s,
    gen_family_c,
    BUILD_BUDGET,
    BUILD_RESEED,
    BUILD_N,
    BUILD_ALPHA,
    BUILD_SIGMA,
    INIT_SCALE,
    FEW_N,
)

# ==========================================================================
# B-POC-4 proof (Vision B rung 4): the convergence test. A skill repertoire
# discovered by UNSUPERVISED exploration (B-POC-2's MAP-Elites emitter) is
# saved to disk, reloaded, and shown to transfer to HELD-OUT goals under a tiny
# few-shot budget — scored in Vision A's uncheatable currency, through the
# UNCHANGED generic ES core (fit_operator over SandboxPolicyMemory/ComposeMemory).
#
# A goal is a target end-state BC whose Go-Explore cell key is NOT a repertoire
# bin (so retrieval can never return the exact answer). Four arms at equal
# few-shot budget (same N and iters => same rollouts; only the seed differs):
#   cold     — zero seed (baseline)
#   random   — a uniformly drawn elite's weights (generic warm-init control)
#   nearest  — the BC-nearest elite's weights (the retrieval-transfer arm)
#   compose  — the K nearest primitives frozen in a schedule (composition)
# Two goal families: S (single random policies) and C (two-phase A->B rollouts
# that reward sequencing).
#
# Gates:
#   1. RETRIEVAL (Family S): nearest is >= 3x closer (MSE) than cold AND than
#      random — indexed retrieval, not generic warm-init, is the lever.
#   2. COMPOSITION (Family C): compose is >= 5% closer (MSE) than nearest on the
#      compositional goals, and beats cold — sequencing frozen primitives
#      extends reach past single retrieval where a single primitive falls short.
#   3. SERIALIZATION fidelity: the reloaded map equals the saved one elite-for-
#      elite, and every reloaded elite re-reaches its bin key on replay.
#   4. HELD-OUT discipline: no goal's cell key is a repertoire bin.
# (Thresholds locked below the seed-0 measurement — see JOURNAL for the sweep.)
# ==========================================================================

comptime NUM_GOALS = 24

# Gate 1 — retrieval must be at least this many times closer (MSE) than the
# cold / random baselines. Measured at seed 0: 7.3x (cold), 30x (random).
comptime MIN_NEAR_COLD = Float32(3.0)
comptime MIN_NEAR_RAND = Float32(3.0)
# Gate 2 — composition must be at least this factor closer (MSE) than nearest on
# the compositional family. Measured at seed 0: 1.12x.
comptime MIN_COMP_NEAR = Float32(1.05)


def main() raises:
    seed(0)
    var task = SandboxTask()

    # --- Phase 1: unsupervised pretraining — build the repertoire, then make
    # the build/few-shot split PHYSICAL by saving and reloading it.
    var built = EliteMap()
    var arch = NoveltyArchive()
    var b_cov = CellSet()
    var b_end = CellSet()
    var build_rollouts = me_emitter_run(
        built,
        task,
        arch,
        b_cov,
        b_end,
        BUILD_BUDGET,
        BUILD_RESEED,
        BUILD_N,
        BUILD_ALPHA,
        BUILD_SIGMA,
        INIT_SCALE,
    )
    var rep_path = String("build/b_poc_4.rep")
    built.save(rep_path)
    var emap = load_elite_map(rep_path)

    # --- Serialization fidelity: reloaded map equals the saved one elite-for-
    # elite (key present, settle + weights + bc bit-identical), and every
    # reloaded elite re-reaches its bin key on replay (the test_repertoire
    # honesty check, now across a save/load boundary).
    var mismatch = 0
    if emap.count != built.count:
        mismatch += 1
    for i in range(emap.count):
        var slot = emap.filled[i]
        var key = emap.keys[slot]
        var o = built.find(key)
        if o < 0:
            mismatch += 1
            continue
        if emap.settle[slot] != built.settle[o]:
            mismatch += 1
        for j in range(POLICY_DIM):
            if (
                emap.weights[slot * POLICY_DIM + j]
                != built.weights[o * POLICY_DIM + j]
            ):
                mismatch += 1
                break
        for j in range(BC_DIM):
            if emap.bc[slot * BC_DIM + j] != built.bc[o * BC_DIM + j]:
                mismatch += 1
                break

    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logits = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var replay_fail = 0
    for i in range(emap.count):
        var slot = emap.filled[i]
        sandbox_rollout(
            emap.weights + slot * POLICY_DIM,
            task,
            grid,
            obs,
            logits,
            bc,
            cells,
            True,
        )
        if cells[SB_T - 1] != emap.keys[slot]:
            replay_fail += 1

    # --- Phase 2: few-shot transfer over held-out goals, against the RELOADED
    # repertoire. Goals are filtered so no cell key is a repertoire bin.
    var gs_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var gs_key = alloc[Int64](NUM_GOALS)
    var ns = gen_family_s(emap, task, gs_bc, gs_key, NUM_GOALS, 8000)
    var gc_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var gc_key = alloc[Int64](NUM_GOALS)
    var nc = gen_family_c(emap, task, gc_bc, gc_key, NUM_GOALS, 40000)

    # Held-out discipline: assert no goal is a repertoire bin.
    var leak = 0
    for g in range(ns):
        if emap.contains(gs_key[g]):
            leak += 1
    for g in range(nc):
        if emap.contains(gc_key[g]):
            leak += 1

    var pw = ESWorkspace[SandboxPolicyMemory](BC_DIM, FEW_N)
    var cw = ESWorkspace[ComposeMemory](BC_DIM, FEW_N)
    var pslow = alloc[Float32](POLICY_DIM)
    for i in range(POLICY_DIM):
        pslow[i] = 0.0
    var cslow = alloc[Float32](COMPOSE_DIM)
    for i in range(COMPOSE_DIM):
        cslow[i] = 0.0

    var s = run_family(emap, task, gs_bc, gs_key, ns, pw, cw, pslow, cslow)
    var c = run_family(emap, task, gc_bc, gc_key, nc, pw, cw, pslow, cslow)

    # MSE magnitudes (fitness is negative MSE; smaller magnitude = closer).
    var s_cold = -s.cold
    var s_rand = -s.randel
    var s_near = -s.nearest
    var c_near = -c.nearest
    var c_comp = -c.compose
    var c_cold = -c.cold

    var near_cold = s_cold / s_near if s_near > 0.0 else Float32(0.0)
    var near_rand = s_rand / s_near if s_near > 0.0 else Float32(0.0)
    var comp_near = c_near / c_comp if c_comp > 0.0 else Float32(0.0)

    print("  build rollouts:", build_rollouts, " repertoire:", built.count)
    print(
        "  reloaded:",
        emap.count,
        " mismatch:",
        mismatch,
        " replay_fail:",
        replay_fail,
    )
    print("  goals: S", ns, " C", nc, " held-out leaks:", leak)
    print(
        "  --- Family S (retrieval): mean end-state MSE to goal (lower=closer)"
    )
    print("      cold   ", s_cold, " hit", s.cold_hit)
    print("      random ", s_rand, " hit", s.randel_hit)
    print("      nearest", s_near, " hit", s.nearest_hit)
    print("      compose", -s.compose, " hit", s.compose_hit)
    print(
        "      nearest is",
        near_cold,
        "x closer than cold,",
        near_rand,
        "x than random",
    )
    print("  --- Family C (composition): compositional goals")
    print("      cold   ", c_cold, " hit", c.cold_hit)
    print("      nearest", c_near, " hit", c.nearest_hit)
    print("      compose", c_comp, " hit", c.compose_hit)
    print("      compose is", comp_near, "x closer than nearest")

    # --- Gate 3: serialization.
    if mismatch != 0:
        raise Error(
            "ERROR: reloaded repertoire differs from the saved one ("
            + String(mismatch)
            + " mismatches)."
        )
    if replay_fail != 0:
        raise Error(
            "ERROR: "
            + String(replay_fail)
            + " reloaded elites failed to re-reach their bin on replay."
        )
    # --- Gate 4: held-out discipline.
    if leak != 0:
        raise Error(
            "ERROR: "
            + String(leak)
            + " goals had a cell key already in the repertoire (retrieval could"
            " lookup the exact answer)."
        )
    if ns < NUM_GOALS or nc < NUM_GOALS:
        raise Error(
            "ERROR: could not synthesize enough held-out goals (S="
            + String(ns)
            + " C="
            + String(nc)
            + ")."
        )
    # --- Gate 1: retrieval transfer.
    if near_cold < MIN_NEAR_COLD:
        raise Error(
            "ERROR: nearest-elite warm-start did not beat cold-start by the"
            " required margin (got "
            + String(near_cold)
            + "x, need >= "
            + String(MIN_NEAR_COLD)
            + "x)."
        )
    if near_rand < MIN_NEAR_RAND:
        raise Error(
            "ERROR: nearest-elite did not beat the random-elite control by the"
            " required margin (got "
            + String(near_rand)
            + "x, need >= "
            + String(MIN_NEAR_RAND)
            + "x) — the win is not from indexed retrieval."
        )
    # --- Gate 2: composition extends reach.
    if comp_near < MIN_COMP_NEAR:
        raise Error(
            "ERROR: composition did not beat single-elite retrieval on the"
            " compositional goals (got "
            + String(comp_near)
            + "x, need >= "
            + String(MIN_COMP_NEAR)
            + "x)."
        )
    if c_comp >= c_cold:
        raise Error(
            "ERROR: composition did not beat cold-start on the compositional"
            " goals."
        )

    grid.free()
    obs.free()
    logits.free()
    bc.free()
    cells.free()
    print(
        "Transfer test passed: an unsupervised repertoire, saved and reloaded,"
        " transfers to held-out goals few-shot — retrieval beats cold/random"
        " and composition extends reach past single retrieval (B-POC-4)."
    )
