# suite-tier: full
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.random import seed, randn_float64, random_float64
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_dream_rank.mojo`.
from hope import ExamplePair
from esper_evolution import ESWorkspace
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
from world_model import (
    SandboxState,
    WorldModelMemory,
    WMRolloutMemory,
    WM_ROLLOUT_K,
    WM_DIM,
    collect_transitions,
    held_out_score,
    rollout_accuracy_curve,
)
from transfer import (
    gen_family_s,
    policy_score,
    BUILD_BUDGET,
    BUILD_RESEED,
    BUILD_N,
    BUILD_ALPHA,
    BUILD_SIGMA,
    INIT_SCALE,
)
from adapt import (
    PoseStepMemory,
    WriteStepMemory,
    POSE_DIM,
    POSE_OUT,
    agent_cases,
    pose_held_out,
    dream_score,
    fit_wm_restarts,
    fit_wm_rollout_restarts,
    fit_pose_restarts,
    fit_write_restarts,
    WeightedWMMemory,
)

# ==========================================================================
# T-POC-2 increment 0 — the BOOKED NEGATIVE (regression).
#
# This test began as the rung's pre-registered GO/STOP gate: can a fully
# LEARNED dream of a world (grid model + agent models, all ES-fit from that
# world's own transitions) rank candidate policies by real goal distance
# well enough to ADAPT inside? It STOPped on 2026-07-15, and it is kept
# here inverted — it now asserts the negative it measured, so that the
# finding cannot silently rot and so that a future fix announces itself by
# FAILING this test (see "if this test fails" below).
#
# THE FINDING: the dream half-ORDERS but cannot PICK.
#
#   * Kendall tau (does the dream order candidates sensibly?) reads
#     ~0.4-0.7 — real signal, well clear of the scrambled control's ~0.1.
#   * Top-1 regret (is the dream's BEST pick any good?) fails DECISIVELY in
#     columns: 3.9x against a pre-registered MAX_REGRET of 2.0 (and 5.6x even
#     in world 1's calibration, where the models are strongest).
#
# Be precise about what the STOP rests on: the GO condition required BOTH
# walls worlds, and ROOM MEETS IT — tau 0.55 (bar 0.5), regret 1.97x (bar
# 2.0x). Columns fails both. So the negative is carried by columns, plus
# room's total absence of margin: room straddles the regret bar across runs
# (1.97x / 2.06x / 2.17x), i.e. it sits ON the bar rather than clearing it.
# "The dream cannot pick" is a claim about columns and about room having no
# headroom — NOT a universal failure in every world.
#
# That split is the whole result, because nothing downstream consumes the
# ordering: fit_dream_policy drives a policy toward the dream's ARGMAX. A
# dream with decent tau and broken regret walks a policy, confidently, to a
# point that is multiples worse in reality — which is exactly the arm the
# rung existed to test. So increment 1 is NOT licensed and its arms do not
# run (adapt.run_family_adapt is unrun scaffolding).
#
# THE CAUSE (and why it is not a capacity problem): one-step accuracy is
# the wrong objective for a 64-tick autoregressive dream. At overall ~0.99
# the grid model misplaces ~1% of cells per tick and compounds it 64 times.
# The trend is monotonic in the wrong direction — as the grid model got
# BETTER at one-step events (changed 0.0 -> 0.53 -> 0.93) the dream's top-1
# regret got WORSE (1.0x -> 1.6x -> 5.6x). The identity collapse that the
# weighted objective removed was, for RANKING, partly PROTECTIVE: "nothing
# moves" is safe to iterate 64 times. Fixing the fit did not create this
# problem, it removed the mask hiding it.
#
# Measurements, same harness (all printed; the gated ones marked):
#   * CALIBRATION in world 1 — the false-positive guard. If the harness
#     cannot read rank fidelity where the models are strongest, it cannot
#     indict anything. [gated: tau >= MIN_CALIB_TAU]
#   * SCRAMBLED-MODEL CONTROL — the ENTIRE learned tail (grid + pose +
#     write) weight-permuted must read ~nothing, or the measurement has a
#     free-ranking backdoor. Scrambling the grid model ALONE is reported as
#     context only: the intact agent models still rank through the BC's
#     avatar dims (~0.5-0.6), so it is not a zero-reference. [gated:
#     tau <= MAX_SCRAM_TAU]
#   * MODEL VALIDITY — the negative is only worth booking if the models
#     WORK. [gated: changed >= MIN_WM_CHANGED, blocked >= MIN_POSE_BLOCKED]
#   * THE NEGATIVE — the GO condition (tau >= MIN_W2_TAU AND regret <=
#     MAX_REGRET, in BOTH walls worlds) is NOT met. [gated]
#   * IDENTITY-GRID ARM — the same dream with the grid half replaced by the
#     unfitted near-identity seed. Reported, NOT gated: it is inconsistent
#     (learned beats identity on tau in world 1, loses in columns), which is
#     itself the point — the grid model's ranking contribution is small and
#     unreliable rather than uniformly harmful.
#
# IF THIS TEST FAILS: do not relax it. A failure means the negative stopped
# reproducing — most likely someone fixed the dream's rollout fidelity
# (ROADMAP Route A: fit the model over K ticks of its OWN rollout rather
# than one ground-truth step). That is a GOOD failure: re-open the rung,
# re-read the gate as a GO, and license increment 1.
#
# Read numbers here for DIRECTION, not precision. Both metrics are noisy:
# tau swung 0.31 -> 0.57 -> 0.47 -> 0.36 for columns across runs at fixed
# bars, and regret is a mean of RATIOS (heavy-tailed — a goal with a tiny
# best-real denominator dominates it; a 68x reading is "unbounded", not a
# calibrated 68). The bars below are therefore deliberately loose: they
# assert the SHAPE of the finding, not its digits.
#
# Budget honesty: model-fit transitions are real ticks (printed); dream
# rollouts are uncharged and counted; the real rollouts here are
# MEASUREMENT, not an arm.
# ==========================================================================

comptime NUM_GOALS = 8
comptime NUM_CAND = 6
comptime WM_TRAIN = 128
comptime WM_HOLD = 64
comptime POSE_TRAIN = 512
comptime POSE_HOLD = 128
comptime EP_LEN = 64
comptime WM_N = 16
comptime POSE_N = 64

# The rung's ORIGINAL pre-registered GO bars. They are kept at their
# pre-registration values — unmoved, so the negative is judged by the bar
# the rung committed to before it measured anything.
comptime MIN_W2_TAU = Float32(0.5)
comptime MAX_REGRET = Float32(2.0)

# Harness-integrity bars (loose, below the worst reading across four runs:
# calibration tau 0.583-0.733, scrambled-tail tau 0.100-0.133).
comptime MIN_CALIB_TAU = Float32(0.45)
comptime MAX_SCRAM_TAU = Float32(0.35)

# Model-validity bars — the negative is only meaningful with WORKING models
# (measured: changed 0.93 W1 / 0.87 columns / 0.51 room, all above B-POC-3's
# own 0.4 gate; pose blocked-move 1.0 / 0.8 / 0.6-1.0).
comptime MIN_WM_CHANGED = Float32(0.4)
comptime MIN_POSE_BLOCKED = Float32(0.55)


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


# Fit both dream halves from one world's transitions; returns the real
# transitions consumed (the charged unit). Both fits are the best-of-K
# restart recipes from adapt.mojo (the single-shot B-POC-3 schedule is a
# measured ~50% lottery between the events basin and the identity basin —
# see fit_wm_restarts).
def fit_dream_models(
    task: SandboxTask,
    wm_w: UnsafePointer[Float32, MutAnyOrigin],
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    mut wm_ws: ESWorkspace[WeightedWMMemory],
    mut wm_rollout_ws: ESWorkspace[WMRolloutMemory],
    mut pose_ws: ESWorkspace[PoseStepMemory],
    mut write_ws: ESWorkspace[WriteStepMemory],
    mut wm_overall: Float32,
    mut wm_changed: Float32,
    mut move_acc: Float32,
    mut blocked_acc: Float32,
    mut write_acc: Float32,
    mut rollout_acc64: Float32,
) -> Int:
    var wm_train = List[ExamplePair[SandboxState]]()
    collect_transitions(task, WM_TRAIN, EP_LEN, wm_train)
    var wm_hold = List[ExamplePair[SandboxState]]()
    collect_transitions(task, WM_HOLD, EP_LEN, wm_hold)
    var pose_train = List[ExamplePair[SandboxState]]()
    collect_transitions(task, POSE_TRAIN, EP_LEN, pose_train)
    var pose_hold = List[ExamplePair[SandboxState]]()
    collect_transitions(task, POSE_HOLD, EP_LEN, pose_hold)

    # Route A: the world model is fit over K ticks of its OWN rollout (a
    # warm-started fine-tune on top of the proven one-step fit), not one
    # ground-truth step — see adapt.fit_wm_rollout_restarts and JOURNAL
    # 2026-07-16. held_out_score stays the UNCHANGED one-step ruler, so
    # wm_overall/wm_changed stay directly comparable to every prior number.
    fit_wm_rollout_restarts(
        wm_w, wm_ws, wm_rollout_ws, wm_train, EP_LEN, WM_N, WM_N
    )
    held_out_score(wm_w, wm_hold, wm_overall, wm_changed)

    # NEW measurement (not gated): 64-tick held-out rollout accuracy — shows
    # directly whether the fit actually reduced autoregressive drift,
    # independent of whether tau/regret clear their bars below.
    var curve = alloc[Float32](EP_LEN)
    rollout_accuracy_curve(wm_w, wm_hold, EP_LEN, curve)
    var racc = Float32(0.0)
    for t in range(EP_LEN):
        racc += curve[t]
    rollout_acc64 = racc / Float32(EP_LEN)
    curve.free()

    var pose_tc = agent_cases(pose_train)
    var pose_hc = agent_cases(pose_hold)
    fit_pose_restarts(pose_w, pose_ws, pose_tc, POSE_N)
    fit_write_restarts(write_w, write_ws, pose_tc, POSE_N)
    pose_held_out(pose_w, write_w, pose_hc, move_acc, blocked_acc, write_acc)

    return WM_TRAIN + WM_HOLD + POSE_TRAIN + POSE_HOLD


# Kendall tau between two distance arrays (all-pairs concordance; ties
# count half — with float MSEs exact ties are measure-zero anyway).
def kendall_tau(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
) -> Float32:
    var conc = Float32(0.0)
    var pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairs += 1
            var da = a[i] - a[j]
            var db = b[i] - b[j]
            if (da > 0.0 and db > 0.0) or (da < 0.0 and db < 0.0):
                conc += 1.0
            elif da == 0.0 or db == 0.0:
                conc += 0.5
    if pairs == 0:
        return 0.0
    return 2.0 * conc / Float32(pairs) - 1.0


# One world's rank-fidelity measurement: per goal, a spread of NUM_CAND
# candidates (nearest elite, two sigma-perturbations of it, two random
# elites, cold zero), each scored by a REAL rollout and by a DREAM rollout;
# returns (mean Kendall tau, mean top-1 real regret). `dream_ticks` counts
# the uncharged imagined ticks.
def rank_fidelity(
    mut emap: EliteMap,
    task: SandboxTask,
    goal_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: UnsafePointer[Int64, MutAnyOrigin],
    num: Int,
    wm_w: UnsafePointer[Float32, MutAnyOrigin],
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    mut dream_ticks: Int,
) -> Tuple[Float32, Float32]:
    var cand = alloc[Float32](NUM_CAND * POLICY_DIM)
    var real_d = alloc[Float32](NUM_CAND)
    var dream_d = alloc[Float32](NUM_CAND)
    var grid = alloc[Float32](SB_CELLS)
    var pred = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logit = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)

    var tau_sum = Float32(0.0)
    var regret_sum = Float32(0.0)
    for g in range(num):
        var tgt = goal_bc + g * BC_DIM
        # Candidate 0: the BC-nearest elite (the retrieval seed).
        var n_slot = emap.nearest(tgt)
        memcpy(
            dest=cand, src=emap.weights + n_slot * POLICY_DIM, count=POLICY_DIM
        )
        # 1..2: sigma-perturbations of it (mid-quality neighbours).
        for k in range(2):
            var sg = Float32(0.1) if k == 0 else Float32(0.3)
            var dst = cand + (1 + k) * POLICY_DIM
            for j in range(POLICY_DIM):
                dst[j] = cand[j] + Float32(randn_float64(0.0, 1.0)) * sg
        # 3..4: random stored elites (unmatched skills).
        for k in range(2):
            var slot = emap.select_uniform(Float32(random_float64(0.0, 1.0)))
            memcpy(
                dest=cand + (3 + k) * POLICY_DIM,
                src=emap.weights + slot * POLICY_DIM,
                count=POLICY_DIM,
            )
        # 5: the cold zero policy.
        memset_zero(cand + 5 * POLICY_DIM, POLICY_DIM)

        for k in range(NUM_CAND):
            var w = cand + k * POLICY_DIM
            var r = policy_score(
                w, task, tgt, goal_key[g], grid, obs, logit, bc, cells
            )
            real_d[k] = -r[0]
            dream_d[k] = -dream_score(
                w, wm_w, pose_w, write_w, task, tgt, grid, pred, obs, logit, bc
            )
            dream_ticks += SB_T
        tau_sum += kendall_tau(real_d, dream_d, NUM_CAND)

        var best_real = real_d[0]
        var best_dream_k = 0
        for k in range(1, NUM_CAND):
            if real_d[k] < best_real:
                best_real = real_d[k]
            if dream_d[k] < dream_d[best_dream_k]:
                best_dream_k = k
        if best_real > 0.0:
            regret_sum += real_d[best_dream_k] / best_real
        else:
            regret_sum += 1.0

    cand.free()
    real_d.free()
    dream_d.free()
    grid.free()
    pred.free()
    obs.free()
    logit.free()
    bc.free()
    cells.free()
    var inv = Float32(1.0) / Float32(num if num > 0 else 1)
    return (tau_sum * inv, regret_sum * inv)


# Deterministic Fisher-Yates permutation of a weight vector (own LCG): the
# scrambled model keeps the exact weight DISTRIBUTION but no function.
def scramble_weights(w: UnsafePointer[Float32, MutAnyOrigin], n: Int):
    var state = UInt64(0xB5297A4D3F84C2E1)
    for i in range(n - 1, 0, -1):
        state = state * 6364136223846793005 + 1442695040888963407
        var j = Int((state >> 33) % UInt64(i + 1))
        var tmp = w[i]
        w[i] = w[j]
        w[j] = tmp


def main() raises:
    seed(0)
    var w1_task = SandboxTask()
    var w2a_task = SandboxTask()
    gen_walls_layout(w2a_task, SB_WALLS_COLUMNS, 1)
    var w2b_task = SandboxTask()
    gen_walls_layout(w2b_task, SB_WALLS_ROOM, 1)

    # --- The carried repertoire (world 1, B-POC-2 emitter, locked constants).
    var emap = EliteMap()
    var build_rollouts = build_repertoire(emap, w1_task)
    print("  repertoire:", emap.count, "elites (", build_rollouts, "rollouts)")

    var wm_ws = ESWorkspace[WeightedWMMemory](SB_CELLS, WM_N)
    var wm_rollout_ws = ESWorkspace[WMRolloutMemory](
        WM_ROLLOUT_K * SB_CELLS, WM_N
    )
    var pose_ws = ESWorkspace[PoseStepMemory](POSE_OUT, POSE_N)
    var write_ws = ESWorkspace[WriteStepMemory](POSE_OUT, POSE_N)
    var wm_w = alloc[Float32](WM_DIM)
    var pose_w = alloc[Float32](POSE_DIM)
    var write_w = alloc[Float32](POSE_DIM)
    var wm_o = Float32(0.0)
    var wm_c = Float32(0.0)
    var mv = Float32(0.0)
    var bl = Float32(0.0)
    var wr = Float32(0.0)
    var racc = Float32(0.0)
    var dream_ticks = 0

    # ---------- Calibration: world 1, where the WM is known-fittable ----------
    var t_w1 = fit_dream_models(
        w1_task,
        wm_w,
        pose_w,
        write_w,
        wm_ws,
        wm_rollout_ws,
        pose_ws,
        write_ws,
        wm_o,
        wm_c,
        mv,
        bl,
        wr,
        racc,
    )
    print("  W1 rollout-drift (64-tick held-out mean changed-cell acc):", racc)
    print(
        "  W1 models: wm overall",
        wm_o,
        " changed",
        wm_c,
        " | pose move",
        mv,
        " blocked",
        bl,
        " write",
        wr,
        " (",
        t_w1,
        "transitions )",
    )
    var w1_ch = wm_c
    var w1_bl = bl
    var g_bc = alloc[Float32](NUM_GOALS * BC_DIM)
    var g_key = alloc[Int64](NUM_GOALS)
    var n1 = gen_family_s(emap, w1_task, g_bc, g_key, NUM_GOALS, 60000)
    var calib = rank_fidelity(
        emap, w1_task, g_bc, g_key, n1, wm_w, pose_w, write_w, dream_ticks
    )
    print("  calibration (W1): tau", calib[0], " top-1 regret", calib[1], "x")

    # ---------- Scrambled-model control: the ENTIRE learned tail (grid,
    # pose and write models) weight-permuted — a useless model as a whole
    # must read ~nothing. Scrambling only the WM is reported as context:
    # the intact agent models still rank partly through the BC's avatar
    # dims — real signal, not a backdoor, but it means the WM-only scramble
    # is not a zero-reference. ----
    var wm_scram = alloc[Float32](WM_DIM)
    memcpy(dest=wm_scram, src=wm_w, count=WM_DIM)
    scramble_weights(wm_scram, WM_DIM)
    var scram_wm = rank_fidelity(
        emap, w1_task, g_bc, g_key, n1, wm_scram, pose_w, write_w, dream_ticks
    )
    print(
        "  scrambled-WM-only (context): tau",
        scram_wm[0],
        " regret",
        scram_wm[1],
        "x",
    )
    var pose_scram = alloc[Float32](POSE_DIM)
    memcpy(dest=pose_scram, src=pose_w, count=POSE_DIM)
    scramble_weights(pose_scram, POSE_DIM)
    var write_scram = alloc[Float32](POSE_DIM)
    memcpy(dest=write_scram, src=write_w, count=POSE_DIM)
    scramble_weights(write_scram, POSE_DIM)
    var scram = rank_fidelity(
        emap,
        w1_task,
        g_bc,
        g_key,
        n1,
        wm_scram,
        pose_scram,
        write_scram,
        dream_ticks,
    )
    print("  scrambled-model control: tau", scram[0], " regret", scram[1], "x")

    # ---------- The gate: the two walls worlds, W2-fit models ----------
    var t_w2a = fit_dream_models(
        w2a_task,
        wm_w,
        pose_w,
        write_w,
        wm_ws,
        wm_rollout_ws,
        pose_ws,
        write_ws,
        wm_o,
        wm_c,
        mv,
        bl,
        wr,
        racc,
    )
    print(
        "  columns rollout-drift (64-tick held-out mean changed-cell acc):",
        racc,
    )
    print(
        "  columns models: wm overall",
        wm_o,
        " changed",
        wm_c,
        " | pose move",
        mv,
        " blocked",
        bl,
        " write",
        wr,
        " (",
        t_w2a,
        "transitions )",
    )
    var a_ch = wm_c
    var a_bl = bl
    var na = gen_family_s(emap, w2a_task, g_bc, g_key, NUM_GOALS, 60000)
    var gate_a = rank_fidelity(
        emap, w2a_task, g_bc, g_key, na, wm_w, pose_w, write_w, dream_ticks
    )
    print("  gate (columns): tau", gate_a[0], " top-1 regret", gate_a[1], "x")

    var wm_b = alloc[Float32](WM_DIM)
    var pose_b = alloc[Float32](POSE_DIM)
    var write_b = alloc[Float32](POSE_DIM)
    var t_w2b = fit_dream_models(
        w2b_task,
        wm_b,
        pose_b,
        write_b,
        wm_ws,
        wm_rollout_ws,
        pose_ws,
        write_ws,
        wm_o,
        wm_c,
        mv,
        bl,
        wr,
        racc,
    )
    print(
        "  room rollout-drift (64-tick held-out mean changed-cell acc):", racc
    )
    print(
        "  room models: wm overall",
        wm_o,
        " changed",
        wm_c,
        " | pose move",
        mv,
        " blocked",
        bl,
        " write",
        wr,
        " (",
        t_w2b,
        "transitions )",
    )
    var b_ch = wm_c
    var b_bl = bl
    var nb = gen_family_s(emap, w2b_task, g_bc, g_key, NUM_GOALS, 60000)
    var gate_b = rank_fidelity(
        emap, w2b_task, g_bc, g_key, nb, wm_b, pose_b, write_b, dream_ticks
    )
    print("  gate (room):    tau", gate_b[0], " top-1 regret", gate_b[1], "x")

    # ---------- Identity-grid arm (reported, NOT gated) ----------
    # The same dream with the grid half replaced by the unfitted
    # near-identity seed ("keep this cell's colour") — i.e. what the
    # collapsed fits used to produce. Reported because it is INCONSISTENT:
    # learned beats identity on tau in one world and loses in the other,
    # which is the honest shape of the grid model's contribution (small and
    # unreliable, not uniformly harmful). Gating on it would overclaim.
    var wm_ident = alloc[Float32](WM_DIM)
    WorldModelMemory.seed(wm_ident)
    var ident_a = rank_fidelity(
        emap, w2a_task, g_bc, g_key, na, wm_ident, pose_w, write_w, dream_ticks
    )
    print(
        "  identity-grid (columns, context): tau",
        ident_a[0],
        " regret",
        ident_a[1],
        "x",
    )
    print("  uncharged dream ticks:", dream_ticks)

    if n1 < NUM_GOALS or na < NUM_GOALS or nb < NUM_GOALS:
        raise Error(
            "ERROR: could not synthesize enough held-out goals ("
            + String(n1)
            + "/"
            + String(na)
            + "/"
            + String(nb)
            + ", need "
            + String(NUM_GOALS)
            + ")."
        )
    # ---- Harness integrity: the measurement can read, and cannot cheat ----
    if calib[0] < MIN_CALIB_TAU:
        raise Error(
            "ERROR: calibration tau "
            + String(calib[0])
            + " below "
            + String(MIN_CALIB_TAU)
            + " — the harness cannot read rank fidelity even in W1, so it"
            " cannot indict anything. This is a broken measurement, not a"
            " finding."
        )
    if scram[0] > MAX_SCRAM_TAU:
        raise Error(
            "ERROR: scrambled-model control tau "
            + String(scram[0])
            + " above "
            + String(MAX_SCRAM_TAU)
            + " — the measurement has a free-ranking backdoor."
        )

    # ---- The models WORK: without this the negative would be worthless ----
    if w1_ch < MIN_WM_CHANGED or a_ch < MIN_WM_CHANGED or b_ch < MIN_WM_CHANGED:
        raise Error(
            "ERROR: a grid model failed validity (changed W1 "
            + String(w1_ch)
            + ", columns "
            + String(a_ch)
            + ", room "
            + String(b_ch)
            + ", need "
            + String(MIN_WM_CHANGED)
            + ") — the dream's models must WORK for its ranking failure to"
            " mean anything. Suspect the fit (WeightedWMMemory), not the rung."
        )
    if (
        w1_bl < MIN_POSE_BLOCKED
        or a_bl < MIN_POSE_BLOCKED
        or b_bl < MIN_POSE_BLOCKED
    ):
        raise Error(
            "ERROR: an agent model failed validity (blocked-move W1 "
            + String(w1_bl)
            + ", columns "
            + String(a_bl)
            + ", room "
            + String(b_bl)
            + ", need "
            + String(MIN_POSE_BLOCKED)
            + ")."
        )

    # ---- THE BOOKED NEGATIVE ----
    # The rung's own pre-registered GO condition, evaluated and asserted
    # FALSE. Regret is the operative half: fit_dream_policy follows the
    # dream's argmax, so a dream that orders well but picks badly cannot
    # carry the adaptation arm.
    var go_a = gate_a[0] >= MIN_W2_TAU and gate_a[1] <= MAX_REGRET
    var go_b = gate_b[0] >= MIN_W2_TAU and gate_b[1] <= MAX_REGRET
    if go_a and go_b:
        raise Error(
            "The booked NEGATIVE no longer reproduces: the dream now MEETS"
            " the rung's pre-registered GO condition in both walls worlds"
            " (columns tau "
            + String(gate_a[0])
            + " regret "
            + String(gate_a[1])
            + "x; room tau "
            + String(gate_b[0])
            + " regret "
            + String(gate_b[1])
            + "x; bars "
            + String(MIN_W2_TAU)
            + " / "
            + String(MAX_REGRET)
            + "x). This is a GOOD failure — do NOT relax this test. Someone"
            " has plausibly fixed the dream's rollout fidelity (ROADMAP"
            " Route A). Re-open T-POC-2, re-read this gate as a GO, and"
            " license increment 1 (adapt.run_family_adapt)."
        )
    # The finding's shape: ordering survives, picking does not.
    if gate_a[1] <= MAX_REGRET and gate_b[1] <= MAX_REGRET:
        raise Error(
            "The booked negative's SHAPE changed: top-1 regret now clears"
            " MAX_REGRET in both walls worlds (columns "
            + String(gate_a[1])
            + "x, room "
            + String(gate_b[1])
            + "x) even though the GO condition did not. The rung was booked"
            " on regret, not tau — re-measure before trusting either."
        )
    print("  BOOKED NEGATIVE reproduces: the dream orders (tau", gate_a[0])
    print("  columns /", gate_b[0], "room) but cannot pick (regret", gate_a[1])
    print(
        "  x /", gate_b[1], "x, bar", MAX_REGRET, "x). Increment 1 unlicensed."
    )
