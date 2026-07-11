# suite-tier: full
from std.memory import alloc, memset_zero
from std.random import seed
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_ued.mojo`.
from hope import ExamplePair
from esper_evolution import ESWorkspace
from sandbox import SB_CELLS
from world_model import (
    SandboxState,
    WorldModelMemory,
    WM_DIM,
    held_out_score,
    collect_transitions,
)
from ued import (
    ArmOut,
    UED_N,
    EP_LEN,
    N_HELD,
    HOLD_PER_LEVEL,
    held_level,
    level_sig,
    paint_score,
    accel_run,
    dr_run,
)

# ==========================================================================
# B-POC-5 proof (Vision B rung 5): ACCEL-style UED — the emergent curriculum.
# The last hand-coded thing (the fixed set of training worlds) is replaced by
# a growing, mutation-fed, learnability-curated replay buffer, all through the
# UNCHANGED ES core / world model. Two arms at BYTE-IDENTICAL transition
# budget, differing ONLY in level selection:
#   ACCEL  — mutate a learnability-weighted parent, curate the buffer by the
#            round-over-round held-out changed-cell-accuracy delta (B-POC-3's
#            direct, mean-learning-immune learnability score).
#   DR     — uniform-random levels, no buffer/curation.
# The uncheatable metric is held-out changed-cell accuracy on a DISJOINT,
# moderate-density level distribution (built up front, never trained on). Gates:
#   1. Equal budget (consumed_accel == consumed_dr).
#   2. Held-out leak = 0 (no held-out level in the trained buffer).
#   3. Curriculum churns (buffer evictions > 0 — not static).
#   4. Concentration (ACCEL's mutated levels are at least as learnable, on
#      average, as DR's uniform draws).
#   5. Headline: ACCEL's held-out changed-cell accuracy clears an absolute
#      floor AND beats DR's by an additive margin MIN_DELTA (a ratio is ill-
#      defined here — changed-cell accuracies sit near zero, and DR ~ 0).
# The paint sub-metric (changed-cell accuracy on agent-write transitions —
# B-POC-3's honest residual) is REPORTED, not gated.
# ==========================================================================

# Pass thresholds — calibrated 2026-07-11 at seed 0; measured: ACCEL held-out
# changed-cell 0.154 vs DR 0.0 (DR's per-round random difficulty churns under
# the anneal restart and never accumulates — all 6 DR rounds wasted), delta
# 0.154; ACCEL mean per-round learnability 0.072 vs DR -0.008; 3 evictions.
# Locked with headroom. The headline is an ADDITIVE margin (B-POC-3's arm-delta
# precedent): changed-cell accuracies sit near zero, so a ratio is ill-defined
# when DR ~ 0.
comptime MIN_CHANGED = Float32(0.09)
comptime MIN_DELTA = Float32(0.08)


def main() raises:
    seed(0)
    var ws = ESWorkspace[WorldModelMemory](SB_CELLS, UED_N)

    # ---------- Held-out level set (disjoint, dense-random, never trained) ----------
    var held = List[ExamplePair[SandboxState]]()
    var held_sigs = List[UInt64]()
    for _ in range(N_HELD):
        var lvl = held_level()
        held_sigs.append(level_sig(lvl))
        collect_transitions(lvl, HOLD_PER_LEVEL, EP_LEN, held)

    # ---------- Arm DR (domain randomization) ----------
    var w_dr = alloc[Float32](WM_DIM)
    WorldModelMemory.seed(w_dr)
    var dr = dr_run(w_dr, ws)
    var o = Float32(0.0)
    var ch_dr = Float32(0.0)
    held_out_score(w_dr, held, o, ch_dr)
    var overall_dr = o
    var paint_dr = paint_score(w_dr, held)

    # ---------- Arm ACCEL (curated mutation) ----------
    var w_ac = alloc[Float32](WM_DIM)
    WorldModelMemory.seed(w_ac)
    var leak = 0
    var ac = accel_run(w_ac, ws, held_sigs, leak)
    var ch_ac = Float32(0.0)
    held_out_score(w_ac, held, o, ch_ac)
    var overall_ac = o
    var paint_ac = paint_score(w_ac, held)

    var delta = ch_ac - ch_dr

    # ---------- Traces ----------
    print("  budget (transitions/arm):", dr.consumed, "/", ac.consumed)
    print("  held-out changed-cell accuracy:")
    print("    DR    :", ch_dr, " (overall", overall_dr, ")")
    print("    ACCEL :", ch_ac, " (overall", overall_ac, ")")
    print("    delta (ACCEL - DR):", delta)
    print("  paint sub-metric (changed-cell on agent-writes):")
    print("    DR", paint_dr, " ACCEL", paint_ac)
    print(
        "  ACCEL buffer: evictions",
        ac.evictions,
        " final activity",
        ac.buf_activity,
        " wasted rounds",
        ac.wasted,
    )
    print(
        "  mean per-round learnability  ACCEL",
        ac.mean_L,
        " DR",
        dr.mean_L,
        " (DR wasted rounds",
        dr.wasted,
        ")",
    )
    print("  held-out leak:", leak)

    # ---------- Gates ----------
    if dr.consumed != ac.consumed:
        raise Error(
            "ERROR: unequal transition budgets across arms ("
            + String(dr.consumed)
            + " vs "
            + String(ac.consumed)
            + ")."
        )
    if leak != 0:
        raise Error(
            "ERROR: "
            + String(leak)
            + " held-out levels appeared in the trained buffer."
        )
    if ac.evictions <= 0:
        raise Error(
            "ERROR: the replay buffer never churned (no curation happened)."
        )
    if ac.mean_L < dr.mean_L:
        raise Error(
            "ERROR: curated mutation did not concentrate on more-learnable"
            " levels than uniform DR (ACCEL mean-L "
            + String(ac.mean_L)
            + " < DR mean-L "
            + String(dr.mean_L)
            + ")."
        )
    if ch_ac < MIN_CHANGED:
        raise Error(
            "ERROR: ACCEL held-out changed-cell below the absolute floor ("
            + String(ch_ac)
            + " < "
            + String(MIN_CHANGED)
            + ")."
        )
    if delta < MIN_DELTA:
        raise Error(
            "ERROR: the emergent curriculum did not beat domain randomization"
            " on held-out generalization at equal budget (delta "
            + String(delta)
            + ", need >= "
            + String(MIN_DELTA)
            + ")."
        )

    w_dr.free()
    w_ac.free()
    print(
        "UED test passed: an emergent ACCEL curriculum — mutate levels, curate"
        " by learnability — trains a world model that generalizes to held-out"
        " worlds better than domain randomization at equal budget, through the"
        " unchanged ES core (B-POC-5)."
    )
