# suite-tier: full
from std.random import seed, random_float64
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_trial_select.mojo`.
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
)
from novelty_es import NoveltyArchive
from map_elites import EliteMap, me_emitter_run
from esper_evolution import ESWorkspace
from transfer import (
    SelectStats,
    run_family_select,
    pool_trials,
    gen_family_s,
    SEL_K,
    SEL_POOL,
    SEL_FIT_ITERS,
    BUILD_BUDGET,
    BUILD_RESEED,
    BUILD_N,
    BUILD_ALPHA,
    BUILD_SIGMA,
    INIT_SCALE,
    FEW_N,
    FEW_ITERS,
)

# ==========================================================================
# Route C proof — ITE-style REAL-TRIAL selection over nearest_k
# (the T-POC-2 fallback rung, scheduled 2026-07-16).
#
# Causal history, in one breath: T-POC-1 (test_cross_world) proved the INDEX
# carries across a world change — BC-nearest retrieval rescues 4-8x negative
# transfer back to cold-parity in the walls worlds — but the same-world 7.3x
# warm-basin advantage is eaten by the world gap (nearest-vs-cold ~1x
# post-fit). T-POC-2 tried to reclaim it by PICKING among k retrieved elites
# inside a learned dream and STOPped at its ranking-fidelity gate (the dream
# orders, tau 0.4-0.7, but cannot pick: top-1 regret 3.9x vs the 2.0x bar in
# columns — test_dream_rank, the booked negative). Route A (K-step rollout-
# level WM fidelity) fixed columns robustly (regret 3.93x -> 1.44x/1.05x at
# seeds 0/1) but room missed at both seeds (4.64x, 2.39x) — GO required both
# worlds, gate held at STOP. Route C drops the dream: rollouts here are
# DETERMINISTIC (argmax policy, fixed world), so ONE real 64-tick rollout per
# candidate scores it EXACTLY — selection by real trial has regret 1.0 over
# its pool by construction, at 576 real ticks per goal. Cully et al. 2015's
# Intelligent Trial and Error, with the BO surrogate degenerate at pool size
# 9. HONESTY CAVEAT (carried from ROADMAP): this is SELECTION, not
# re-grounding — it sidesteps the mission's re-grounding question.
#
# Phases:
#   1. Unsupervised pretraining IN WORLD 1 (the locked B-POC-2/4 emitter
#      build; serialization already gated by T-POC-1, so no .rep round-trip
#      here — this test is standalone).
#   2. Held-out goals per world (keys disjoint from the W1 map): world 1
#      (probe calibration), columns, room.
#   3. INCREMENT 0 — the oracle-headroom probe (cheap gate, licenses the
#      arms): per goal, one real trial per pool member (SEL_K nearest elites
#      + cold-zero); oracle gain = d(nearest-1) / d(true best of pool),
#      PRE-fit. The few-shot fit historically washes seed advantage OUT
#      (T-POC-1's ~1x nearest-vs-cold), never amplifies it — so if even the
#      oracle seed advantage is below the post-fit bar, the mechanism cannot
#      clear it and the expensive arms never run.
#   4. INCREMENT 1 — four arms per walls world at equal-or-less budget, all
#      ending in one real policy_score on the held-out goal:
#        cold+fit / nearest-1+fit (the T-POC-1 arm) / ITE best-of-pool+fit
#        (THE arm; its 576 trial ticks overcharged by dropping one full ES
#        iteration = 4096 ticks from its fit) / uniform-pool-pick+fit (the
#        control that could refute: if a RANDOM pool member fits as well as
#        the trial-picked one, selection added nothing over the pool).
#
# RNG draw order (fixed, load-bearing — the B-POC-3 stream-position lesson):
# seed(0) -> W1 build -> W1 goals -> columns goals -> room goals -> probes
# (ZERO draws) -> columns arms -> room arms. Per goal inside the arms:
# cold fit -> nearest fit -> ITE trials (zero draws) + fit -> uniform pick
# (one draw) + fit.
#
# Gates (pre-registered 2026-07-16, BEFORE any measurement; bars inherited
# from T-POC-2 increment 1's committed values — no bar shopping; judged at
# these ORIGINAL values, median per-goal ratios, both walls worlds; a GO
# additionally requires the seed-1 manual re-run to agree, AND rule):
#   G0. HELD-OUT: zero goal-key leaks; full NUM_GOALS per family; the
#       repertoire holds >= SEL_K elites.
#   G1. INCREMENT-0 LICENSE: median oracle gain >= 1.3 in BOTH walls worlds,
#       else STOP (arms unlicensed, booked as the negative).
#   G2. GO-primary: median(d_nearest / d_ITE) >= 1.3 in both walls worlds.
#   G3. GO-floor: median(d_cold / d_ITE) >= 1.2 in both walls worlds.
#   G4. Selection-vs-pool control: median(d_uniform / d_ITE) > 1.0 in both
#       walls worlds — the trial-picked seed must beat a random pool member.
#   G5. Non-regression: room exact-hit fraction for ITE not below the
#       nearest-1 arm's (T-POC-2's own committed condition).
#
# MEASURED (2026-07-16, seed 0 committed / seed 1 manual re-run) — VERDICT:
# PARTIAL, booked. The rung is judged by the ORIGINAL bars above under the
# AND rule, and G2 fails at ONE seed in ONE world:
#
#                nearest/ITE  cold/ITE  uniform/ITE  ITE hits vs nearest
#   s0 columns      4.19x       1.89x      4.08x        1/24 vs 0/24
#   s0 room         1.21x       3.11x      5.23x        6/24 vs 4/24   <- G2 fail
#   s1 columns      1.55x       2.01x      2.65x        2/24 vs 1/24
#   s1 room         2.07x       3.49x      2.70x        8/24 vs 4/24
#   probe (s0): W1 1.00 (clean control: at home nearest-1 IS the pool's
#   best, zero picks changed) columns 6.19 room 1.68; (s1) 2.34 / 3.57.
#
# Every OTHER bar passes at BOTH seeds in BOTH worlds with real margins:
# the license, the cold floor, the uniform-pool control, and exact hits
# (ITE improves them everywhere). Room's primary margin STRADDLES its bar
# across seeds (1.21 / 2.07 vs 1.3) — the same straddle pattern Route A's
# room regret showed — so under the AND rule this is NOT a GO, and per the
# discipline the bar is not flipped. The committed gates below therefore
# assert exactly what reproduces at seed 0: the license, the columns
# primary, the room DIRECTIONAL floor (> 1.0), G3/G4/G5 in full, and the
# two-world GO condition asserted FALSE — if selection starts clearing the
# full GO here, this test fails LOUDLY so the rung is re-opened and re-run
# under the seed AND rule; do not relax that assert.
# ==========================================================================

comptime NUM_GOALS = 24

# G1 bar: the increment-0 license (see header for the wash-out rationale).
comptime MIN_ORACLE_GAIN = Float32(1.3)
# G2/G3 bars: inherited verbatim from T-POC-2 increment 1 (MIN_DREAM_NEAR /
# MIN_DREAM_COLD in the unrun test_adapt scaffolding).
comptime MIN_ITE_NEAR = Float32(1.3)
comptime MIN_ITE_COLD = Float32(1.2)
# Ratio denominators are clamped here (an exact reach has d = 0); clamp
# events are counted and printed — silence means the guard never fired.
comptime EPS = Float32(1e-9)


# Build one repertoire in `task` with the locked B-POC-2/4 constants; returns
# the rollouts consumed (printed for the budget audit).
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


def leaks(
    mut emap: EliteMap, goal_key: UnsafePointer[Int64, MutAnyOrigin], n: Int
) -> Int:
    var bad = 0
    for g in range(n):
        if emap.contains(goal_key[g]):
            bad += 1
    return bad


# Median via insertion sort on a copy (n is tiny; even n = the midpair mean).
def median_of(vals: UnsafePointer[Float32, MutAnyOrigin], n: Int) -> Float32:
    var tmp = List[Float32]()
    for i in range(n):
        tmp.append(vals[i])
    for i in range(1, n):
        var v = tmp[i]
        var j = i - 1
        while j >= 0 and tmp[j] > v:
            tmp[j + 1] = tmp[j]
            j -= 1
        tmp[j + 1] = v
    if n % 2 == 1:
        return tmp[n // 2]
    return Float32(0.5) * (tmp[n // 2 - 1] + tmp[n // 2])


# Median of per-goal ratios num/den with the EPS clamp; returns (median,
# clamp count) so the caller can print when the guard fired.
def ratio_median(
    num_d: UnsafePointer[Float32, MutAnyOrigin],
    den_d: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Tuple[Float32, Int]:
    var r = alloc[Float32](n)
    var clamped = 0
    for g in range(n):
        var d = den_d[g]
        if d < EPS:
            d = EPS
            clamped += 1
        r[g] = num_d[g] / d
    var m = median_of(r, n)
    r.free()
    return (m, clamped)


# Increment 0: the oracle-headroom probe over one goal family. Runs
# pool_trials (the EXACT mechanism the ITE arm uses) per goal, zero RNG.
# Returns (median oracle gain, mean gain, pick-changed fraction).
def probe_family(
    name: String,
    mut emap: EliteMap,
    task: SandboxTask,
    goal_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: UnsafePointer[Int64, MutAnyOrigin],
    num: Int,
) raises -> Tuple[Float32, Float32, Float32]:
    var slots = alloc[Int](SEL_K)
    var pool_d = alloc[Float32](SEL_POOL)
    var cand = alloc[Float32](POLICY_DIM)
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logit = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var gains = alloc[Float32](num)

    var mean_gain = Float32(0.0)
    var changed = 0
    var clamped = 0
    for g in range(num):
        var tgt = goal_bc + g * BC_DIM
        var best = pool_trials(
            emap,
            task,
            tgt,
            goal_key[g],
            slots,
            pool_d,
            cand,
            grid,
            obs,
            logit,
            bc,
            cells,
        )
        var d_best = pool_d[best]
        if d_best < EPS:
            d_best = EPS
            clamped += 1
        gains[g] = pool_d[0] / d_best
        mean_gain += gains[g]
        if best != 0:
            changed += 1
    mean_gain /= Float32(num if num > 0 else 1)
    var med = median_of(gains, num)
    var pick_rate = Float32(changed) / Float32(num if num > 0 else 1)
    print(
        "  probe",
        name,
        ": median oracle gain",
        med,
        " mean",
        mean_gain,
        " pick-changed",
        pick_rate,
    )
    if clamped > 0:
        print("      (probe ratio clamps fired:", clamped, ")")

    slots.free()
    pool_d.free()
    cand.free()
    grid.free()
    obs.free()
    logit.free()
    bc.free()
    cells.free()
    gains.free()
    return (med, mean_gain, pick_rate)


def report_arms(name: String, s: SelectStats) raises:
    print("  --- Arms", name, "(world-2 goals; mean MSE, lower=closer)")
    print("      cold        ", -s.cold, " hit", s.cold_hit)
    print("      nearest(W1) ", -s.nearest, " hit", s.nearest_hit)
    print("      ITE (trials)", -s.ite, " hit", s.ite_hit)
    print("      uniform-pool", -s.unif, " hit", s.unif_hit)
    print(
        "      pick-changed",
        s.pick_changed,
        "/",
        s.n,
        "  trial ticks",
        s.trial_ticks,
        " (overcharged by",
        2 * FEW_N * SB_T * s.n - s.trial_ticks,
        "ticks: ITE fit runs",
        SEL_FIT_ITERS,
        "of",
        FEW_ITERS,
        "iters)",
    )


def main() raises:
    seed(0)
    var w1_task = SandboxTask()
    var w2a_task = SandboxTask()
    gen_walls_layout(w2a_task, SB_WALLS_COLUMNS, 1)
    var w2b_task = SandboxTask()
    gen_walls_layout(w2b_task, SB_WALLS_ROOM, 1)

    # --- Phase 1: world-1 pretraining (standalone build, locked constants).
    var emap = EliteMap()
    var w1_rollouts = build_repertoire(emap, w1_task)
    print("  build rollouts: W1", w1_rollouts, " elites", emap.count)
    if emap.count < SEL_K:
        raise Error(
            "ERROR: repertoire too small for the pool ("
            + String(emap.count)
            + " elites, need >= "
            + String(SEL_K)
            + ")."
        )

    # --- Phase 2: held-out goals per world (W1 first: probe calibration).
    var gw_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var gw_key = alloc[Int64](NUM_GOALS)
    var nw = gen_family_s(emap, w1_task, gw_bc, gw_key, NUM_GOALS, 60000)
    var ga_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var ga_key = alloc[Int64](NUM_GOALS)
    var na = gen_family_s(emap, w2a_task, ga_bc, ga_key, NUM_GOALS, 60000)
    var gb_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var gb_key = alloc[Int64](NUM_GOALS)
    var nb = gen_family_s(emap, w2b_task, gb_bc, gb_key, NUM_GOALS, 60000)
    var leak = (
        leaks(emap, gw_key, nw)
        + leaks(emap, ga_key, na)
        + leaks(emap, gb_key, nb)
    )
    print(
        "  goals: W1",
        nw,
        " columns",
        na,
        " room",
        nb,
        "  held-out leaks:",
        leak,
    )
    # G0: held-out discipline.
    if leak != 0:
        raise Error(
            "ERROR: "
            + String(leak)
            + " goal keys found in the repertoire (retrieval could look up"
            " the answer)."
        )
    if nw < NUM_GOALS or na < NUM_GOALS or nb < NUM_GOALS:
        raise Error(
            "ERROR: could not synthesize enough held-out goals (W1 "
            + String(nw)
            + ", columns "
            + String(na)
            + ", room "
            + String(nb)
            + ", need "
            + String(NUM_GOALS)
            + ")."
        )

    # --- Phase 3 (increment 0): the oracle-headroom probe. Zero RNG draws,
    # so the arms below land at the same stream position whether or not this
    # phase's numbers change.
    print("  --- Increment 0: oracle-headroom probe (pre-fit, real trials)")
    var pw1 = probe_family("W1 (calib)", emap, w1_task, gw_bc, gw_key, nw)
    var pa = probe_family("columns   ", emap, w2a_task, ga_bc, ga_key, na)
    var pb = probe_family("room      ", emap, w2b_task, gb_bc, gb_key, nb)
    _ = pw1  # calibration is reported, not gated (same-world sanity read).
    # G1: the license.
    if pa[0] < MIN_ORACLE_GAIN or pb[0] < MIN_ORACLE_GAIN:
        raise Error(
            "STOP (increment 0): median oracle gain below the license bar"
            " (columns "
            + String(pa[0])
            + "x, room "
            + String(pb[0])
            + "x, need >= "
            + String(MIN_ORACLE_GAIN)
            + "x in both) — even ORACLE selection over the pool cannot clear"
            " the post-fit bar, so the arms stay unlicensed. Book the"
            " negative; do not relax this bar."
        )

    # --- Phase 4 (increment 1): the arms.
    var pw = ESWorkspace[SandboxPolicyMemory](BC_DIM, FEW_N)
    var pslow = alloc[Float32](POLICY_DIM)
    for i in range(POLICY_DIM):
        pslow[i] = 0.0
    var da_cold = alloc[Float32](NUM_GOALS)
    var da_near = alloc[Float32](NUM_GOALS)
    var da_ite = alloc[Float32](NUM_GOALS)
    var da_unif = alloc[Float32](NUM_GOALS)
    var db_cold = alloc[Float32](NUM_GOALS)
    var db_near = alloc[Float32](NUM_GOALS)
    var db_ite = alloc[Float32](NUM_GOALS)
    var db_unif = alloc[Float32](NUM_GOALS)

    var a_stats = run_family_select(
        emap,
        w2a_task,
        ga_bc,
        ga_key,
        na,
        pw,
        pslow,
        da_cold,
        da_near,
        da_ite,
        da_unif,
    )
    var b_stats = run_family_select(
        emap,
        w2b_task,
        gb_bc,
        gb_key,
        nb,
        pw,
        pslow,
        db_cold,
        db_near,
        db_ite,
        db_unif,
    )
    print("  --- Increment 1: the arms (median per-goal ratios gate)")
    report_arms("columns", a_stats)
    report_arms("room   ", b_stats)

    var a_nr = ratio_median(da_near, da_ite, na)
    var a_cr = ratio_median(da_cold, da_ite, na)
    var a_ur = ratio_median(da_unif, da_ite, na)
    var b_nr = ratio_median(db_near, db_ite, nb)
    var b_cr = ratio_median(db_cold, db_ite, nb)
    var b_ur = ratio_median(db_unif, db_ite, nb)
    var clamps = a_nr[1] + a_cr[1] + a_ur[1] + b_nr[1] + b_cr[1] + b_ur[1]
    print(
        "      columns medians: nearest/ITE",
        a_nr[0],
        " cold/ITE",
        a_cr[0],
        " uniform/ITE",
        a_ur[0],
    )
    print(
        "      room    medians: nearest/ITE",
        b_nr[0],
        " cold/ITE",
        b_cr[0],
        " uniform/ITE",
        b_ur[0],
    )
    if clamps > 0:
        print("      (ratio clamps fired:", clamps, ")")
    var base_ticks = 2 * FEW_N * FEW_ITERS * SB_T * (na + nb)
    print(
        "      budget: baseline arm",
        base_ticks,
        "fit ticks; ITE arm",
        2 * FEW_N * SEL_FIT_ITERS * SB_T * (na + nb)
        + a_stats.trial_ticks
        + b_stats.trial_ticks,
        "(trials + reduced fit — strictly under)",
    )

    # G2 (as booked, 2026-07-16 partial): columns clears the primary bar at
    # both seeds (4.19x / 1.55x) — gated; room only STRADDLES it across seeds
    # (1.21x / 2.07x) — gated as a DIRECTIONAL floor, and the full two-world
    # GO asserted FALSE below so a change that fixes room re-opens the rung.
    if a_nr[0] < MIN_ITE_NEAR:
        raise Error(
            "REGRESSION: columns ITE lost its primary margin over nearest-1"
            " (median nearest/ITE "
            + String(a_nr[0])
            + "x, booked 4.19x at seed 0, bar >= "
            + String(MIN_ITE_NEAR)
            + "x)."
        )
    if b_nr[0] <= Float32(1.0):
        raise Error(
            "REGRESSION: room ITE fell below the directional floor vs"
            " nearest-1 (median nearest/ITE "
            + String(b_nr[0])
            + "x, booked 1.21x at seed 0, need > 1.0x)."
        )
    if a_nr[0] >= MIN_ITE_NEAR and b_nr[0] >= MIN_ITE_NEAR:
        raise Error(
            "GOOD failure — the booked PARTIAL stopped reproducing: the full"
            " two-world GO condition now PASSES at seed 0 (median"
            " nearest/ITE columns "
            + String(a_nr[0])
            + "x, room "
            + String(b_nr[0])
            + "x, both >= "
            + String(MIN_ITE_NEAR)
            + "x). Do not relax this assert — re-open the rung and re-judge"
            " under the seed AND rule (seed-1 manual re-run)."
        )
    # G3: GO-floor.
    if a_cr[0] < MIN_ITE_COLD or b_cr[0] < MIN_ITE_COLD:
        raise Error(
            "STOP (increment 1): ITE did not beat cold at the pre-registered"
            " floor (median cold/ITE columns "
            + String(a_cr[0])
            + "x, room "
            + String(b_cr[0])
            + "x, need >= "
            + String(MIN_ITE_COLD)
            + "x in both)."
        )
    # G4: the control that could refute — trials must beat a random pool
    # member, else the win is the POOL's, not the selection's.
    if a_ur[0] <= Float32(1.0) or b_ur[0] <= Float32(1.0):
        raise Error(
            "STOP (increment 1): the uniform-pool control matched ITE"
            " (median uniform/ITE columns "
            + String(a_ur[0])
            + "x, room "
            + String(b_ur[0])
            + "x, need > 1.0 in both) — selection added nothing over the pool."
        )
    # G5: room exact-hit non-regression (T-POC-2's committed condition).
    if b_stats.ite_hit < b_stats.nearest_hit:
        raise Error(
            "STOP (increment 1): ITE regressed room exact-hits ("
            + String(b_stats.ite_hit)
            + " vs nearest "
            + String(b_stats.nearest_hit)
            + ")."
        )

    gw_bc.free()
    gw_key.free()
    ga_bc.free()
    ga_key.free()
    gb_bc.free()
    gb_key.free()
    pslow.free()
    da_cold.free()
    da_near.free()
    da_ite.free()
    da_unif.free()
    db_cold.free()
    db_near.free()
    db_ite.free()
    db_unif.free()
    print(
        "Trial-select test passed (the 2026-07-16 PARTIAL, booked): the"
        " license, the columns primary, the room directional floor, the cold"
        " floor, the uniform-pool control and the exact-hit non-regression"
        " all reproduce at seed 0 — and the full two-world GO stays"
        " unearned."
    )
