# ==========================================================================
# ACCEL-style UED on the sandbox (Vision B / B-POC-5) — the emergent
# curriculum. This replaces the LAST hand-coded thing in the pipeline: the
# fixed, human-chosen set of training worlds. B-POC-3's `train_lp_guided`
# allocated a transition budget across an ENUMERATED set of dynamics contexts;
# B-POC-5 generalizes that fixed set into a GROWING, mutation-fed, curated
# replay buffer — genuine ACCEL (Parker-Holder et al. 2022): random edits to
# existing levels + curation by learnability, no learned generator. Evolution
# proposes; the curation is the intelligence.
#
# Curation signal = the No-Regrets/TRACED-approved DIRECT learnability score
# (never a regret proxy): the round-over-round HELD-OUT CHANGED-CELL-ACCURACY
# delta of the ACTUAL learner on a level's validation batch — the exact
# scale-free, mean-learning-immune windowed slope B-POC-3 proved as an
# allocator (fitness-scale LP chases the noisy TV; clone-probe LP measures
# memorizability; the discrete score is immune to both).
#
# Solver curated toward = the world model (WorldModelMemory), fit through the
# UNCHANGED generic ES core; the outer metric stays uncheatable held-out
# transfer (held-out changed-cell accuracy on a disjoint level distribution),
# never a self-scored quantity (the DGM cautionary tale).
#
# Mutation surface = the initial-grid CONFIG (which cells start painted, and the
# avatar start). Gravity (direction AND rate) is held FIXED — a constant of the
# world — because learning even ONE gravity function to a measurable changed-
# cell level costs hundreds of ES iters (B-POC-3), so a bounded per-arm budget
# spread across direction-functions leaves none learned; and grav_rate has no
# WorldModelMemory input feature at all (mutating it would manufacture
# contradictory training pairs). The operative UED axis is the config's
# gravity-EVENT DENSITY: too sparse teaches almost nothing in-budget, moderate
# is richly learnable — the curriculum's job is to find and hold that band.
#
# Baseline = domain randomization (DR): each round a fresh level whose density is
# uniform-random over [0, DR_HI], no buffer/curation/mutation. Both arms run
# identical shared-model training loops at byte-identical transition budgets; the
# ONLY difference is level selection.
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.math import round
from std.random import random_float64
from std.collections import List

from hope import ExamplePair
from esper_evolution import ESWorkspace
from sandbox import SB_ROWS, SB_COLS, SB_CELLS, SandboxTask
from world_model import (
    SandboxState,
    WorldModelMemory,
    WM_DIM,
    collect_transitions,
    held_out_score,
    train_round,
    sample_pool,
)

# The paint action (sandbox: 0 up, 1 down, 2 left, 3 right, 4 paint, 5 cycle).
comptime SB_PAINT = 4

# --- Loop budget (transitions are the charged unit; both arms consume exactly
# UED_ROUNDS * (VAL_N + TRAIN_COLLECT), and both train on the same
# UED_ROUNDS * TRAIN_COLLECT transitions). Few rounds (like B-POC-3's 4) so the
# per-round wide->fine anneal restart refines rather than churns the model. ---
comptime UED_ROUNDS = 6
comptime VAL_N = 32  # per-level held-out validation batch (charged, never trained)
comptime TRAIN_COLLECT = 100  # per-round training transitions (charged, trained)
comptime EP_LEN = 64
comptime TRAIN_ITERS = 220
comptime UED_N = 16  # ES population (matches B-POC-3)

# --- Buffer / curation ---
comptime BUF_CAP = 3  # replay-buffer capacity (the learnable frontier) — small so it churns
comptime N_SEED = 3  # first N_SEED children are moderate-density seeds
comptime PARENT_FLOOR = Float32(
    0.02
)  # so every buffer level stays a reachable parent

# --- Mutation surface = the initial-grid CONFIG (density/pattern). gravity is
# held FIXED (down) alongside grav_rate: learning even ONE gravity function to a
# measurable changed-cell level costs ~hundreds of ES iters (B-POC-3), so
# spreading a bounded per-arm budget across 4 direction-functions leaves none
# learned — the config axis is the operative UED surface, gravity is a constant
# of the world here. (Empirically: a world model reaches ~0.44 held-out
# changed-cell on moderate-density grav-down in 400 focused iters; sparse
# configs teach almost nothing at that budget — see JOURNAL calibration.)
comptime EDIT_K = 10  # cells flipped per mutation (a small edit to an existing level)
comptime P_START = Float32(0.5)  # chance a mutation nudges the avatar start
# The difficulty GRADIENT the curriculum must exploit is DENSITY: a config's
# gravity-event rate. Too SPARSE => almost no dynamics to learn from in-budget;
# MODERATE => rich, learnable. DR draws density uniformly from [0, DR_HI], so it
# wastes budget on the sparse tail; held-out and the seed/target band sit at the
# learnable moderate density. ACCEL's learnability curation should concentrate
# there; uniform DR cannot.
comptime DR_HI = Float32(
    0.16
)  # DR upper density (spans the wasteful sparse tail)
comptime SEED_RHO = Float32(0.08)  # moderate-density seeds / mutation center
comptime HELD_RHO = Float32(0.08)  # held-out = the learnable target band

# --- Held-out metric ---
comptime N_HELD = 6  # held-out levels (disjoint, fixed moderate density)
comptime HOLD_PER_LEVEL = 48


# One arm's outcome (Copyable/Movable so it returns cleanly, POD).
struct ArmOut(Copyable, Movable):
    var consumed: Int
    var mean_L: Float32  # mean per-round learnability of the levels this arm trained on
    var evictions: Int  # buffer turnover (ACCEL only)
    var buf_activity: Float32  # final buffer mean dynamics activity (ACCEL only)
    var wasted: Int  # rounds whose level was mastered/unlearnable (L <= 0)

    def __init__(out self):
        self.consumed = 0
        self.mean_L = 0.0
        self.evictions = 0
        self.buf_activity = 0.0
        self.wasted = 0


# ==========================================
# Level construction + mutation
# ==========================================
def _clampi(v: Int, lo: Int, hi: Int) -> Int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def _rand_color() -> Int:
    var col = Int(random_float64(1.0, 10.0))
    return _clampi(col, 1, 9)


# A level at a given initial-grid density: gravity DOWN (fixed), each cell
# painted with probability `rho`, random start pose/brush.
def level_at_density(rho: Float64) -> SandboxTask:
    var t = SandboxTask()  # grav_dir defaults to SB_GRAV_DOWN, grav_rate to 1
    for i in range(SB_CELLS):
        if random_float64(0.0, 1.0) < rho:
            t.grid[i] = Float32(_rand_color())
    t.start_r = _clampi(
        Int(random_float64(0.0, Float64(SB_ROWS))), 0, SB_ROWS - 1
    )
    t.start_c = _clampi(
        Int(random_float64(0.0, Float64(SB_COLS))), 0, SB_COLS - 1
    )
    t.start_brush = _rand_color()
    return t^


# A DR draw: density uniform over [0, DR_HI] — spread across the difficulty
# gradient, so DR spends part of its budget on the wasteful sparse tail.
def random_level() -> SandboxTask:
    return level_at_density(random_float64(0.0, Float64(DR_HI)))


# A seed level: moderate density (the learnable band) — the frontier's start.
def seed_level() -> SandboxTask:
    return level_at_density(Float64(SEED_RHO))


# A held-out draw: fixed moderate density (the learnable target band).
def held_level() -> SandboxTask:
    return level_at_density(Float64(HELD_RHO))


# A small random edit to an existing level (ACCEL's "random edits"): flip
# EDIT_K grid cells and maybe nudge the avatar start — the child stays NEAR the
# parent so the frontier advances gradually. Gravity is untouched (a constant).
def mutate_level(parent: SandboxTask) -> SandboxTask:
    var t = SandboxTask()
    memcpy(dest=t.grid, src=parent.grid, count=SB_CELLS)
    t.start_r = parent.start_r
    t.start_c = parent.start_c
    t.start_brush = parent.start_brush
    t.grav_dir = parent.grav_dir  # gravity is a constant of the world (down)
    t.grav_rate = 1
    for _ in range(EDIT_K):
        var idx = _clampi(
            Int(random_float64(0.0, Float64(SB_CELLS))), 0, SB_CELLS - 1
        )
        if random_float64(0.0, 1.0) < 0.5:
            t.grid[idx] = 0.0
        else:
            t.grid[idx] = Float32(_rand_color())
    if random_float64(0.0, 1.0) < Float64(P_START):
        t.start_r = _clampi(
            Int(random_float64(0.0, Float64(SB_ROWS))), 0, SB_ROWS - 1
        )
        t.start_c = _clampi(
            Int(random_float64(0.0, Float64(SB_COLS))), 0, SB_COLS - 1
        )
    return t^


# FNV-1a fold of a level's full signature (grid + params) — for the held-out
# leak check (no held-out level may appear among the trained buffer).
def level_sig(t: SandboxTask) -> UInt64:
    var h = UInt64(0xCBF29CE484222325)
    for i in range(SB_CELLS):
        h = (h ^ UInt64(Int(round(t.grid[i])))) * UInt64(0x100000001B3)
    h = (h ^ UInt64(t.grav_dir)) * UInt64(0x100000001B3)
    h = (h ^ UInt64(t.start_r)) * UInt64(0x100000001B3)
    h = (h ^ UInt64(t.start_c)) * UInt64(0x100000001B3)
    h = (h ^ UInt64(t.start_brush)) * UInt64(0x100000001B3)
    return h


# ==========================================
# Diagnostics over a batch / model
# ==========================================
# Mean changed-cell count per transition — a level's "dynamics activity"
# (trivial settled/empty configs ~0; rich falling/painting configs high).
def batch_activity(demos: List[ExamplePair[SandboxState]]) -> Float32:
    if len(demos) == 0:
        return 0.0
    var tot = 0
    for d in range(len(demos)):
        var pre = demos[d].input_grid.grid
        var post = demos[d].output_grid.grid
        for i in range(SB_CELLS):
            if round(pre[i]) != round(post[i]):
                tot += 1
    return Float32(tot) / Float32(len(demos))


# Held-out changed-cell accuracy restricted to PAINT transitions — isolates
# B-POC-3's honest unlearned-paint residual (agent-write events).
def paint_score(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[SandboxState]],
) -> Float32:
    var pred = alloc[Float32](SB_CELLS)
    var ch_cells = 0
    var ch_hits = 0
    for d in range(len(demos)):
        if demos[d].input_grid.action != SB_PAINT:
            continue
        WorldModelMemory.apply(weights, demos[d].input_grid, pred)
        var pre = demos[d].input_grid.grid
        var post = demos[d].output_grid.grid
        for i in range(SB_CELLS):
            var t = round(post[i])
            if round(pre[i]) != t:
                ch_cells += 1
                if round(pred[i]) == t:
                    ch_hits += 1
    pred.free()
    return Float32(ch_hits) / Float32(ch_cells if ch_cells > 0 else 1)


# ==========================================
# Parent selection (learnability-weighted)
# ==========================================
def _pick_parent(Ls: List[Float32]) -> Int:
    var total = Float32(0.0)
    for i in range(len(Ls)):
        var v = Ls[i]
        if v < 0.0:
            v = 0.0
        total += v + PARENT_FLOOR
    var x = Float32(random_float64(0.0, Float64(total)))
    var acc = Float32(0.0)
    for i in range(len(Ls)):
        var v = Ls[i]
        if v < 0.0:
            v = 0.0
        acc += v + PARENT_FLOOR
        if x <= acc:
            return i
    return len(Ls) - 1


# ==========================================
# The two arms
# ==========================================
# ACCEL: each round proposes a child (an easy seed while the buffer is filling,
# else a mutation of a learnability-weighted parent), collects its charged
# validation + training batches, measures every buffer level's fresh
# learnability as the changed-cell-accuracy delta across THIS round's training
# (the B-POC-3 windowed discrete slope), then curates the buffer to the BUF_CAP
# highest-learnability levels. All collected data trains; the buffer only steers
# which levels are mutated next — the frontier advances as levels are mastered.
def accel_run(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WorldModelMemory],
    mut held_sigs: List[UInt64],
    mut leak: Int,
) -> ArmOut:
    var res = ArmOut()
    var all_data = List[ExamplePair[SandboxState]]()
    var levels = List[SandboxTask]()
    var vals = List[List[ExamplePair[SandboxState]]]()
    var Ls = List[Float32]()
    var L_sum = Float32(0.0)

    for rnd in range(UED_ROUNDS):
        # --- propose a child ---
        var child: SandboxTask
        if len(levels) < N_SEED:
            child = seed_level()  # moderate-density seed (the learnable band)
        else:
            child = mutate_level(levels[_pick_parent(Ls)])
        var cval = List[ExamplePair[SandboxState]]()
        collect_transitions(child, VAL_N, EP_LEN, cval)
        res.consumed += VAL_N
        collect_transitions(child, TRAIN_COLLECT, EP_LEN, all_data)
        res.consumed += TRAIN_COLLECT

        # --- add child to the working set ---
        levels.append(child^)
        vals.append(cval^)
        Ls.append(0.0)
        var m = len(levels)

        # --- windowed learnability: score all, train, score all, delta ---
        var f_before = alloc[Float32](m)
        var o = Float32(0.0)
        var ch = Float32(0.0)
        for i in range(m):
            held_out_score(w, vals[i], o, ch)
            f_before[i] = ch
        var pool = sample_pool(all_data, rnd)
        train_round(w, ws, pool, TRAIN_ITERS, UED_N)
        for i in range(m):
            held_out_score(w, vals[i], o, ch)
            Ls[i] = ch - f_before[i]
        f_before.free()

        var child_L = Ls[m - 1]
        L_sum += child_L
        if child_L <= 0.0:
            res.wasted += 1

        # --- curate: keep the BUF_CAP highest-learnability levels ---
        while len(levels) > BUF_CAP:
            var mj = 0
            for i in range(1, len(levels)):
                if Ls[i] < Ls[mj]:
                    mj = i
            _ = levels.pop(mj)
            _ = vals.pop(mj)
            _ = Ls.pop(mj)
            res.evictions += 1

    res.mean_L = L_sum / Float32(UED_ROUNDS)

    # Final buffer diagnostics + the held-out leak check.
    var act_sum = Float32(0.0)
    for i in range(len(levels)):
        act_sum += batch_activity(vals[i])
        var sig = level_sig(levels[i])
        for h in range(len(held_sigs)):
            if held_sigs[h] == sig:
                leak += 1
    res.buf_activity = act_sum / Float32(len(levels) if len(levels) > 0 else 1)
    return res^


# DR: the same shared-model training loop and byte-identical budget, but each
# round draws a uniform-random level — no buffer, no curation, no mutation.
def dr_run(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WorldModelMemory],
) -> ArmOut:
    var res = ArmOut()
    var all_data = List[ExamplePair[SandboxState]]()
    var L_sum = Float32(0.0)
    for rnd in range(UED_ROUNDS):
        var lvl = random_level()
        var val = List[ExamplePair[SandboxState]]()
        collect_transitions(lvl, VAL_N, EP_LEN, val)
        res.consumed += VAL_N
        collect_transitions(lvl, TRAIN_COLLECT, EP_LEN, all_data)
        res.consumed += TRAIN_COLLECT
        var o = Float32(0.0)
        var ch = Float32(0.0)
        held_out_score(w, val, o, ch)
        var before = ch
        var pool = sample_pool(all_data, rnd)
        train_round(w, ws, pool, TRAIN_ITERS, UED_N)
        held_out_score(w, val, o, ch)
        var L = ch - before
        L_sum += L
        if L <= 0.0:
            res.wasted += 1
    res.mean_L = L_sum / Float32(UED_ROUNDS)
    return res^
