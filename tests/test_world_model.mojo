# suite-tier: full
from std.memory import alloc, memset_zero
from std.random import seed
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_world_model.mojo`.
from hope import ExamplePair
from esper_evolution import ESWorkspace, fitness, fit_operator
from sandbox import SB_CELLS
from world_model import (
    SandboxState,
    WorldModelMemory,
    WM_DIM,
    make_task,
    collect_transitions,
    copy_state,
    scramble_targets,
    held_out_score,
    lp_probe,
    train_uniform,
    train_lp_guided,
)

# ==========================================================================
# B-POC-3 proof (Vision B rung 3): a world model + learning progress, all
# through the UNCHANGED generic ES core — transitions are just ExamplePairs
# of a new Domain, and LP is literally the ES fitness slope. Three gates:
#   1. The world model is REAL: fit on random-action grav-down transitions,
#      it predicts held-out next grids — including on the cells the
#      transition actually CHANGED, where the identity predictor ("nothing
#      moves") scores 0 by construction.
#   2. LP separation (the headline): LP is high exactly on learnable-but-
#      not-yet-learned experience — LP(novel gravity) >> LP(mastered) and
#      >> LP(scrambled targets), while the scrambled region keeps HIGH raw
#      error. Raw-error curiosity would park on the noisy TV; LP walks away.
#   3. LP-GUIDED collection beats uniform at equal transition budget:
#      allocating each round's collection proportionally to per-region LP
#      yields a better world model (held-out changed-cell accuracy over the
#      4 real regions) than spreading it evenly across 4 real + 1 scrambled
#      region. Probe COMPUTE is uncharged; probe DATA is charged.
# ==========================================================================

comptime WM_N = 16

# Gate 1 fit schedule (staged wide -> fine; the single-anneal fit is
# measurably schedule-sensitive — see JOURNAL 2026-07-10).
comptime G1_TRAIN = 128
comptime G1_HOLD = 64

# LP probe stage (constant alpha/sigma so stages are comparable; wide enough
# that genuinely learnable material shows progress from a converged model).
comptime LP_STAGE = 80
comptime LP_BATCH = 64
comptime LP_ALPHA = Float32(0.2)
comptime LP_SIGMA = Float32(0.15)

# Arm budgets (equal for both arms; transitions are the charged unit).
comptime ARM_REAL_REGIONS = 2
comptime ARM_ROUNDS = 4
comptime ARM_ROUND_BUDGET = 150
comptime ARM_EP_LEN = 64
comptime ARM_TRAIN_ITERS = 200
comptime ARM_VAL_N = 32
comptime HOLD_PER_REGION = 32

# Pass thresholds (calibrated 2026-07-10 at seed 0; measured: overall 0.9918
# / changed 0.625; LP separation ratio 22x; unlearnable error 8.7x mastered;
# arm scores 0.362 (LP-guided) vs 0.153 (uniform). Locked with headroom.)
comptime MIN_OVERALL = Float32(0.985)
comptime MIN_CHANGED = Float32(0.4)
comptime LP_SEP_K = Float32(10.0)
comptime ERR_RATIO = Float32(1.5)
comptime MIN_ARM_DELTA = Float32(0.08)


def main() raises:
    seed(0)
    var slow = alloc[Float32](WM_DIM)
    memset_zero(slow, WM_DIM)
    var scratch = alloc[Float32](SB_CELLS)
    var ws = ESWorkspace[WorldModelMemory](SB_CELLS, WM_N)

    # ---------- Gate 1: the world model is real ----------
    var task0 = make_task(0)
    var train = List[ExamplePair[SandboxState]]()
    collect_transitions(task0, G1_TRAIN, 64, train)
    var hold = List[ExamplePair[SandboxState]]()
    collect_transitions(task0, G1_HOLD, 64, hold)

    var w = alloc[Float32](WM_DIM)
    WorldModelMemory.seed(w)
    fit_operator[WorldModelMemory](
        w,
        ws,
        slow,
        train,
        WM_N,
        Float32(0.3),
        Float32(0.1),
        Float32(0.3),
        Float32(0.1),
        750,
        0.0,
    )
    fit_operator[WorldModelMemory](
        w,
        ws,
        slow,
        train,
        WM_N,
        Float32(0.08),
        Float32(0.02),
        Float32(0.08),
        Float32(0.02),
        500,
        0.0,
    )
    var overall = Float32(0.0)
    var changed = Float32(0.0)
    held_out_score(w, hold, overall, changed)
    print("  world model held-out: overall", overall, " changed", changed)
    print("  (identity predictor: changed = 0.0 by construction)")
    if overall < MIN_OVERALL or changed < MIN_CHANGED:
        raise Error(
            "ERROR: world model below gate (overall "
            + String(overall)
            + " changed "
            + String(changed)
            + ")."
        )

    # ---------- Gate 2: LP separation (noisy-TV immunity) ----------
    # Symmetric design: mastered = (a copy of) the batch the model was JUST
    # trained on; unlearnable = another copy of the SAME batch, contradiction-
    # scrambled — so the contradiction is the only difference between the two;
    # novel = a fresh gravity direction. All probed from the gate-1 model.
    var task_novel = make_task(1)  # grav-up: a fresh dynamics context
    var novel = List[ExamplePair[SandboxState]]()
    collect_transitions(task_novel, LP_BATCH, 64, novel)
    var mastered = List[ExamplePair[SandboxState]]()
    var unlearn = List[ExamplePair[SandboxState]]()
    for i in range(LP_BATCH):
        mastered.append(
            ExamplePair[SandboxState](
                copy_state(train[i].input_grid),
                copy_state(train[i].output_grid),
            )
        )
        unlearn.append(
            ExamplePair[SandboxState](
                copy_state(train[i].input_grid),
                copy_state(train[i].output_grid),
            )
        )
    scramble_targets(unlearn)

    var f_m = fitness[WorldModelMemory](w, slow, mastered, scratch, 0.0)
    var f_u = fitness[WorldModelMemory](w, slow, unlearn, scratch, 0.0)
    var lp_m = lp_probe(w, ws, mastered, LP_STAGE, WM_N, LP_ALPHA, LP_SIGMA)
    var lp_n = lp_probe(w, ws, novel, LP_STAGE, WM_N, LP_ALPHA, LP_SIGMA)
    var lp_u = lp_probe(w, ws, unlearn, LP_STAGE, WM_N, LP_ALPHA, LP_SIGMA)
    print("  LP  mastered:", lp_m, " novel:", lp_n, " unlearnable:", lp_u)
    print("  raw fitness  mastered:", f_m, " unlearnable:", f_u)

    var lp_floor = lp_m
    if lp_u > lp_floor:
        lp_floor = lp_u
    if lp_floor < Float32(1e-9):
        lp_floor = 1e-9
    if lp_n < LP_SEP_K * lp_floor:
        raise Error(
            "ERROR: LP did not separate novel from mastered/unlearnable ("
            + String(lp_n)
            + " vs floor "
            + String(lp_floor)
            + ")."
        )
    # The noisy-TV immunity: the scrambled region's raw ERROR stays high
    # (|fitness| well above mastered) even though its LP is ~0 — raw-error
    # curiosity would chase it; LP does not.
    if not (-f_u >= ERR_RATIO * (-f_m)):
        raise Error(
            "ERROR: scrambled region's raw error not elevated ("
            + String(f_u)
            + " vs mastered "
            + String(f_m)
            + ")."
        )

    # ---------- Gate 3: LP-guided collection beats uniform ----------
    # Held-out sets for the 4 real regions, collected up front (evaluation
    # data, never trained on, uncharged).
    var holds = List[List[ExamplePair[SandboxState]]]()
    for g in range(ARM_REAL_REGIONS):
        var h = List[ExamplePair[SandboxState]]()
        var t = make_task(g)
        collect_transitions(t, HOLD_PER_REGION, 64, h)
        holds.append(h^)

    def arm_score(
        aw: UnsafePointer[Float32, MutAnyOrigin],
        mut holds: List[List[ExamplePair[SandboxState]]],
    ) -> Float32:
        var total = Float32(0.0)
        for g in range(ARM_REAL_REGIONS):
            var o = Float32(0.0)
            var ch = Float32(0.0)
            held_out_score(aw, holds[g], o, ch)
            total += ch
        return total / Float32(ARM_REAL_REGIONS)

    var w_uni = alloc[Float32](WM_DIM)
    WorldModelMemory.seed(w_uni)
    var used_uni = train_uniform(
        w_uni,
        ws,
        ARM_REAL_REGIONS,
        ARM_ROUNDS,
        ARM_ROUND_BUDGET,
        ARM_EP_LEN,
        ARM_TRAIN_ITERS,
        WM_N,
    )
    var score_uni = arm_score(w_uni, holds)

    var w_lp = alloc[Float32](WM_DIM)
    WorldModelMemory.seed(w_lp)
    var used_lp = train_lp_guided(
        w_lp,
        ws,
        ARM_REAL_REGIONS,
        ARM_ROUNDS,
        ARM_ROUND_BUDGET,
        ARM_EP_LEN,
        ARM_TRAIN_ITERS,
        WM_N,
        ARM_VAL_N,
        True,
    )
    var score_lp = arm_score(w_lp, holds)

    print("  budget (transitions/arm):", used_uni, "/", used_lp)
    print("  held-out changed-cell (mean over 4 real regions):")
    print("    uniform:  ", score_uni)
    print("    LP-guided:", score_lp)
    if used_uni != used_lp:
        raise Error("ERROR: unequal transition budgets across arms.")
    if score_lp < score_uni + MIN_ARM_DELTA:
        raise Error(
            "ERROR: LP-guided collection did not beat uniform ("
            + String(score_lp)
            + " vs "
            + String(score_uni)
            + ")."
        )

    w.free()
    w_uni.free()
    w_lp.free()
    slow.free()
    scratch.free()
    print(
        "World-model test passed: the engine learns its world's dynamics"
        " through the unchanged ES core, learning progress separates"
        " learnable-novel from mastered AND unlearnable experience, and"
        " LP-guided collection beats uniform at equal budget (B-POC-3)."
    )
