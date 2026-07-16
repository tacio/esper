# suite-tier: full
from std.memory import alloc, UnsafePointer
from std.random import seed
from std.collections import List

# Run from the project root: `mojo run -I src tests/test_wm_rollout.mojo`.
from hope import ExamplePair
from esper_evolution import ESWorkspace
from sandbox import SB_CELLS, SandboxTask
from world_model import (
    SandboxState,
    WeightedWMMemory,
    WMRolloutMemory,
    WM_ROLLOUT_K,
    WM_DIM,
    collect_transitions,
    rollout_accuracy_curve,
    held_out_score,
)
from adapt import fit_wm_restarts, fit_wm_rollout_restarts

# ==========================================================================
# Route A mechanism probe — CHEAP, before test_dream_rank.mojo's full budget.
#
# T-POC-2 STOPped (JOURNAL 2026-07-15) because the world model is fit on
# ONE-STEP prediction but consumed AUTOREGRESSIVELY for SB_T=64 ticks: at
# ~99% one-step accuracy it still misplaces ~1% of cells/tick and that
# compounds, measured monotonic the WRONG way (better one-step accuracy ->
# worse 64-tick top-1 regret). Route A's fix (fit_wm_rollout_restarts,
# adapt.mojo) adds a K=8-tick rollout fine-tune stage on top of the proven
# one-step warm start. Before spending test_dream_rank.mojo's full
# MAP-Elites + policy-scoring budget on the real gate, this test asks the
# MECHANISM question cheaply: does the rollout fine-tune actually reduce
# autoregressive drift, measured at the FULL 64-tick horizon (deliberately
# beyond the K=8 fit horizon — the honest generalization question)?
#
# Mirrors test_world_model.mojo's gate-4 A/B: same train/hold batch, same
# one-step judge (held_out_score, kept as the ruler for comparability) — the
# NEW measurement is rollout_accuracy_curve's mean over the full episode.
# No MAP-Elites, no policy/pose machinery — just collect_transitions + two
# fits + the drift curve, which is what keeps this cheap relative to the
# real gate.
# ==========================================================================

comptime WM_TRAIN = 128
comptime WM_HOLD = 64
comptime EP_LEN = 64
comptime WM_N = 16


def mean_curve(
    w: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[SandboxState]],
    ep_len: Int,
) -> Float32:
    var curve = alloc[Float32](ep_len)
    rollout_accuracy_curve(w, demos, ep_len, curve)
    var m = Float32(0.0)
    for t in range(ep_len):
        m += curve[t]
    curve.free()
    return m / Float32(ep_len)


def main() raises:
    seed(0)
    var task = SandboxTask()
    var train = List[ExamplePair[SandboxState]]()
    collect_transitions(task, WM_TRAIN, EP_LEN, train)
    var hold = List[ExamplePair[SandboxState]]()
    collect_transitions(task, WM_HOLD, EP_LEN, hold)

    # ---- Arm A: one-step-only fit (today's fit_wm_restarts). ----
    var w1 = alloc[Float32](WM_DIM)
    var ws1 = ESWorkspace[WeightedWMMemory](SB_CELLS, WM_N)
    fit_wm_restarts(w1, ws1, train, WM_N)
    var o1 = Float32(0.0)
    var c1 = Float32(0.0)
    held_out_score(w1, hold, o1, c1)
    var m1 = mean_curve(w1, hold, EP_LEN)
    print(
        "  one-step fit:  held-out overall",
        o1,
        " changed",
        c1,
        " mean rollout-acc(64)",
        m1,
    )

    # ---- Arm B: rollout fine-tuned fit (fit_wm_rollout_restarts). ----
    var w2 = alloc[Float32](WM_DIM)
    var ws1b = ESWorkspace[WeightedWMMemory](SB_CELLS, WM_N)
    var ws2 = ESWorkspace[WMRolloutMemory](WM_ROLLOUT_K * SB_CELLS, WM_N)
    fit_wm_rollout_restarts(w2, ws1b, ws2, train, EP_LEN, WM_N, WM_N)
    var o2 = Float32(0.0)
    var c2 = Float32(0.0)
    held_out_score(w2, hold, o2, c2)
    var m2 = mean_curve(w2, hold, EP_LEN)
    print(
        "  rollout fit:   held-out overall",
        o2,
        " changed",
        c2,
        " mean rollout-acc(64)",
        m2,
    )
    print(
        "  rollout-fit / one-step-fit mean rollout accuracy:",
        m2 / m1 if m1 > 0.0 else Float32(0.0),
        "x",
    )

    w1.free()
    w2.free()

    # Directional gate ONLY: the rollout fine-tune must not make 64-tick
    # drift WORSE than the one-step-only fit it warm-starts from (the
    # fine-tune's own safety net already guarantees this on the TRAIN batch;
    # this checks it holds on HELD-OUT episodes too). A real magnitude bar
    # is pinned once test_dream_rank.mojo's own numbers are in — this probe
    # exists to catch a mechanism failure cheaply, not to replace the gate.
    if m2 < m1:
        raise Error(
            "ERROR: the rollout fine-tune did not improve (or held-out"
            " generalize) 64-tick rollout accuracy over the one-step-only"
            " fit (rollout "
            + String(m2)
            + " vs one-step "
            + String(m1)
            + ") — the Route A mechanism did not move in the right"
            " direction on held-out episodes."
        )
    print(
        "test_wm_rollout mechanism probe passed: the rollout fine-tune does"
        " not regress 64-tick held-out rollout accuracy vs the one-step-only"
        " fit."
    )
