# suite-tier: full
from std.random import seed
from std.memory import alloc, memcpy, UnsafePointer
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_cross_world.mojo`.
from sandbox import (
    SB_CELLS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SB_WALLS_COLUMNS,
    SB_WALLS_ROOM,
    SandboxTask,
    SandboxPolicyMemory,
    CellSet,
    gen_walls_layout,
    sandbox_rollout,
)
from novelty_es import NoveltyArchive
from map_elites import EliteMap, me_emitter_run, load_elite_map
from esper_evolution import ESWorkspace
from transfer import (
    ComposeMemory,
    COMPOSE_DIM,
    ArmStats,
    run_family,
    gen_family_s,
    BUILD_BUDGET,
    BUILD_RESEED,
    BUILD_N,
    BUILD_ALPHA,
    BUILD_SIGMA,
    INIT_SCALE,
    FEW_N,
)

# ==========================================================================
# T-POC-1 proof — CROSS-WORLD transfer: knowledge earned unsupervised in one
# world carries to a topologically different world.
#
# This is the first rung of the transfer ladder (ROADMAP 2026-07-14: the
# mission's measured currency is what carries from world N to world N+1).
# World 1 is the open gravity sandbox; world 2 is the SAME substrate with
# WALLS (columns / room layouts — dynamics the policies never saw: blocked
# moves, mid-air settling). The two worlds share the grid, action space,
# POLICY_DIM, and BC definition by construction, so B-POC-4's
# repertoire->retrieval->warm-start machinery applies with zero core changes.
#
# Phases:
#   1. Unsupervised pretraining IN WORLD 1 (B-POC-2 emitter), saved to .rep
#      and reloaded — the carried knowledge is a durable artifact.
#   2. Ceiling reference: an equal-budget repertoire built NATIVELY in each
#      world-2 task (what a no-world-gap library would offer).
#   3. Held-out goals generated IN world 2 (reference rollouts there), keys
#      disjoint from the W1 repertoire — the arm under test can never look
#      up the answer. (The first measurement showed why the CEILING needs
#      its own, stricter subset: in a confined world the native build's bins
#      cover nearly the whole reachable end-state space — the room world had
#      ZERO goals outside the native map. So the claim is gated on the
#      W1-held-out families, and the native ceiling is computed on the
#      doubly-held-out SUBSET, whose size is itself a reported finding:
#      native-coverage saturation of a confined world.)
#   4. Four arms per family at equal few-shot budget, all fits rolling out
#      IN WORLD 2: cold / random-elite(W1) / nearest(W1) / compose(W1);
#      then the same driver over the native map on the doubly-held-out
#      subset — its nearest arm is the ceiling. The world-gap cost =
#      d_nearest(W1) / d_nearest(native), same-subset comparison.
#
# Gates (form pre-registered; numbers pinned with headroom below the seed-0
# measurement, per B-POC convention):
#   1. HELD-OUT: zero goal-key leaks vs the W1 map on the gated families
#      (and vs the native map on the ceiling subset); a full NUM_GOALS per
#      gated family.
#   2. SERIALIZATION: the reloaded W1 map is elite-for-elite identical and
#      100% replayable in world 1.
#   3. TRANSFER (the rung's gated claim, per family): what carries across
#      the world change is the INDEX. Measured (seed 0; directions confirmed
#      at seed 1): a MISMATCHED world-1 skill is 4-8x WORSE than cold under
#      the new topology (random-elite arm), and BC-nearest retrieval
#      reliably rescues that to cold-parity — nearest(W1) 5.78x/5.19x
#      closer than random (columns/room). The pre-registered warm-basin
#      hope — nearest(W1) beating COLD — did NOT materialize robustly
#      (0.75x/1.71x at seed 0, 1.12x/0.96x at seed 1, ~1x noisy): the
#      same-world 7.3x warm-start advantage (B-POC-4) is eaten by the world
#      gap. BOOKED as the rung's honest finding; the cold ratio is gated
#      only as a benign-retrieval floor (never catastrophically below cold).
#   4. CEILING (reported, not gated): the world-gap cost per family on the
#      doubly-held-out subset, and that subset's size (native saturation:
#      the room world's equal-budget native build covers the whole reachable
#      goal space — subset 0). Subset ceilings are tiny-n and seed-noisy
#      (5.2x at seed 0, ~1x at seed 1 on columns) — context, not claims.
# ==========================================================================

comptime NUM_GOALS = 24

# Gate 3 thresholds, pinned with headroom below the seed-0 measurement
# (columns 5.78x / room 5.19x vs random; cold-floor measured 0.75/1.71 at
# seed 0 and 1.12/0.96 at seed 1).
comptime MIN_NEAR_RAND = Float32(2.0)
comptime MIN_COLD_FLOOR = Float32(0.5)


# Build one repertoire in `task` with the locked B-POC-2/4 constants; returns
# the rollouts consumed (printed for the equal-budget audit).
def build_repertoire(mut emap: EliteMap, task: SandboxTask) raises -> Int:
    var arch = NoveltyArchive()
    var cov = CellSet()
    var end = CellSet()
    return me_emitter_run(
        emap,
        task,
        arch,
        cov,
        end,
        BUILD_BUDGET,
        BUILD_RESEED,
        BUILD_N,
        BUILD_ALPHA,
        BUILD_SIGMA,
        INIT_SCALE,
    )


# Copy the goals whose cell key is NOT a bin of `other` into the dst buffers
# (the doubly-held-out ceiling subset); returns its size. The src family is
# left intact — it stays the gated claim's goal set.
def subset_outside(
    mut other: EliteMap,
    src_bc: UnsafePointer[Float32, MutAnyOrigin],
    src_key: UnsafePointer[Int64, MutAnyOrigin],
    n: Int,
    dst_bc: UnsafePointer[Float32, MutAnyOrigin],
    dst_key: UnsafePointer[Int64, MutAnyOrigin],
) -> Int:
    var kept = 0
    for g in range(n):
        if other.contains(src_key[g]):
            continue
        memcpy(
            dest=dst_bc + kept * BC_DIM,
            src=src_bc + g * BC_DIM,
            count=BC_DIM,
        )
        dst_key[kept] = src_key[g]
        kept += 1
    return kept


def leaks(
    mut emap: EliteMap, goal_key: UnsafePointer[Int64, MutAnyOrigin], n: Int
) -> Int:
    var bad = 0
    for g in range(n):
        if emap.contains(goal_key[g]):
            bad += 1
    return bad


def report_family(name: String, w1: ArmStats) raises -> Tuple[Float32, Float32]:
    # MSE magnitudes (fitness is negative MSE; smaller = closer).
    var cold = -w1.cold
    var rand = -w1.randel
    var near = -w1.nearest
    var comp = -w1.compose
    var near_cold = cold / near if near > 0.0 else Float32(0.0)
    var near_rand = rand / near if near > 0.0 else Float32(0.0)
    print("  --- Family", name, "(world-2 goals; mean MSE, lower=closer)")
    print("      cold        ", cold, " hit", w1.cold_hit)
    print("      random (W1) ", rand, " hit", w1.randel_hit)
    print("      nearest(W1) ", near, " hit", w1.nearest_hit)
    print("      compose(W1) ", comp, " hit", w1.compose_hit)
    print(
        "      nearest(W1) is",
        near_cold,
        "x closer than cold,",
        near_rand,
        "x than random",
    )
    return (near_cold, near_rand)


# The world-gap reference: nearest(W1) vs nearest(native) on the SAME
# doubly-held-out subset. Reported, never gated; a subset of size 0 is
# itself the saturation finding.
def report_ceiling(name: String, w1_sub: ArmStats, nat_sub: ArmStats) raises:
    var near = -w1_sub.nearest
    var ceil = -nat_sub.nearest
    var gap = near / ceil if ceil > 0.0 else Float32(0.0)
    print(
        "      ceiling (",
        w1_sub.n,
        "doubly-held-out goals): nearest(W1)",
        near,
        " nearest(native)",
        ceil,
        " world-gap cost",
        gap,
        "x",
    )


def main() raises:
    seed(0)
    var w1_task = SandboxTask()
    var w2a_task = SandboxTask()
    gen_walls_layout(w2a_task, SB_WALLS_COLUMNS, 1)
    var w2b_task = SandboxTask()
    gen_walls_layout(w2b_task, SB_WALLS_ROOM, 1)

    # --- Phase 1: world-1 pretraining, made physical via .rep round-trip.
    var built = EliteMap()
    var w1_rollouts = build_repertoire(built, w1_task)
    var rep_path = String("build/t_poc_1.rep")
    built.save(rep_path)
    var emap = load_elite_map(rep_path)

    var mismatch = 0
    if emap.count != built.count:
        mismatch += 1
    for i in range(emap.count):
        var slot = emap.filled[i]
        var o = built.find(emap.keys[slot])
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
            w1_task,
            grid,
            obs,
            logits,
            bc,
            cells,
            True,
        )
        if cells[SB_T - 1] != emap.keys[slot]:
            replay_fail += 1
    grid.free()
    obs.free()
    logits.free()
    bc.free()
    cells.free()

    # --- Phase 2: the native world-2 ceiling repertoires, equal budget.
    var native_a = EliteMap()
    var na_rollouts = build_repertoire(native_a, w2a_task)
    var native_b = EliteMap()
    var nb_rollouts = build_repertoire(native_b, w2b_task)
    print(
        "  build rollouts: W1",
        w1_rollouts,
        " native-A",
        na_rollouts,
        " native-B",
        nb_rollouts,
    )
    print(
        "  repertoires: W1",
        emap.count,
        "(reloaded)  native-A",
        native_a.count,
        " native-B",
        native_b.count,
    )

    # --- Phase 3: held-out goals IN each world 2, disjoint from the W1 map
    # (the arm under test); the ceiling's doubly-held-out subsets extracted
    # separately (their size is the native-saturation finding).
    var ga_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var ga_key = alloc[Int64](NUM_GOALS)
    var na = gen_family_s(emap, w2a_task, ga_bc, ga_key, NUM_GOALS, 60000)
    var gb_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var gb_key = alloc[Int64](NUM_GOALS)
    var nb = gen_family_s(emap, w2b_task, gb_bc, gb_key, NUM_GOALS, 60000)

    var sa_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var sa_key = alloc[Int64](NUM_GOALS)
    var nsa = subset_outside(native_a, ga_bc, ga_key, na, sa_bc, sa_key)
    var sb_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var sb_key = alloc[Int64](NUM_GOALS)
    var nsb = subset_outside(native_b, gb_bc, gb_key, nb, sb_bc, sb_key)

    var leak = leaks(emap, ga_key, na) + leaks(emap, gb_key, nb)
    leak += leaks(native_a, sa_key, nsa) + leaks(native_b, sb_key, nsb)
    print(
        "  goals: columns",
        na,
        "(doubly-held-out",
        nsa,
        ")  room",
        nb,
        "(doubly-held-out",
        nsb,
        ")  held-out leaks:",
        leak,
    )

    # --- Phase 4: the arms, all few-shot fits rolling out in world 2.
    var pw = ESWorkspace[SandboxPolicyMemory](BC_DIM, FEW_N)
    var cw = ESWorkspace[ComposeMemory](BC_DIM, FEW_N)
    var pslow = alloc[Float32](POLICY_DIM)
    for i in range(POLICY_DIM):
        pslow[i] = 0.0
    var cslow = alloc[Float32](COMPOSE_DIM)
    for i in range(COMPOSE_DIM):
        cslow[i] = 0.0

    var a_w1 = run_family(
        emap, w2a_task, ga_bc, ga_key, na, pw, cw, pslow, cslow
    )
    var b_w1 = run_family(
        emap, w2b_task, gb_bc, gb_key, nb, pw, cw, pslow, cslow
    )
    var a_ratios = report_family("columns", a_w1)
    if nsa > 0:
        var a_w1_sub = run_family(
            emap, w2a_task, sa_bc, sa_key, nsa, pw, cw, pslow, cslow
        )
        var a_nat_sub = run_family(
            native_a, w2a_task, sa_bc, sa_key, nsa, pw, cw, pslow, cslow
        )
        report_ceiling("columns", a_w1_sub, a_nat_sub)
    else:
        print("      ceiling: native map saturates this world (subset 0)")
    var b_ratios = report_family("room   ", b_w1)
    if nsb > 0:
        var b_w1_sub = run_family(
            emap, w2b_task, sb_bc, sb_key, nsb, pw, cw, pslow, cslow
        )
        var b_nat_sub = run_family(
            native_b, w2b_task, sb_bc, sb_key, nsb, pw, cw, pslow, cslow
        )
        report_ceiling("room   ", b_w1_sub, b_nat_sub)
    else:
        print("      ceiling: native map saturates this world (subset 0)")

    # --- Gate 2: serialization.
    if mismatch != 0 or replay_fail != 0:
        raise Error(
            "ERROR: reloaded W1 repertoire not faithful (mismatch "
            + String(mismatch)
            + ", replay failures "
            + String(replay_fail)
            + ")."
        )
    # --- Gate 1: held-out discipline.
    if leak != 0:
        raise Error(
            "ERROR: "
            + String(leak)
            + " goal keys found in a repertoire (retrieval could look up the"
            " answer)."
        )
    if na < NUM_GOALS or nb < NUM_GOALS:
        raise Error(
            "ERROR: could not synthesize enough held-out goals (columns "
            + String(na)
            + ", room "
            + String(nb)
            + ", need "
            + String(NUM_GOALS)
            + ")."
        )
    # --- Gate 3: the index carries — retrieval >> random warm-init on
    # world-2 goals, both families.
    if a_ratios[1] < MIN_NEAR_RAND or b_ratios[1] < MIN_NEAR_RAND:
        raise Error(
            "ERROR: world-1 nearest-elite did not beat the random-elite"
            " control on world-2 goals (columns "
            + String(a_ratios[1])
            + "x, room "
            + String(b_ratios[1])
            + "x, need >= "
            + String(MIN_NEAR_RAND)
            + "x) — the carry-over is not from indexed retrieval."
        )
    # --- Gate 3 floor: retrieval stays benign vs cold (the booked finding is
    # ~1x cold-parity; a collapse below the floor would be a regression into
    # the negative transfer the random arm exhibits).
    if a_ratios[0] < MIN_COLD_FLOOR or b_ratios[0] < MIN_COLD_FLOOR:
        raise Error(
            "ERROR: retrieval fell below the benign floor vs cold-start"
            " (columns "
            + String(a_ratios[0])
            + "x, room "
            + String(b_ratios[0])
            + "x, need >= "
            + String(MIN_COLD_FLOOR)
            + "x)."
        )

    ga_bc.free()
    ga_key.free()
    gb_bc.free()
    gb_key.free()
    sa_bc.free()
    sa_key.free()
    sb_bc.free()
    sb_key.free()
    pslow.free()
    cslow.free()
    print(
        "Cross-world transfer test passed: a repertoire earned unsupervised"
        " in the open world, saved and reloaded, carries to walled worlds"
        " through its INDEX — BC-nearest retrieval rescues the negative"
        " transfer a mismatched skill exhibits (4-8x worse than cold) back to"
        " cold-parity, and lands exact goals in the confined world where cold"
        " rarely does. The warm-basin advantage itself does not survive the"
        " world gap — the booked T-POC-1 finding."
    )
