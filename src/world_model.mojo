# ==========================================================================
# World model + learning progress (Vision B / B-POC-3).
#
# The world model is a LEARNED grid->grid operator over transitions — exactly
# the spine's fast-weights notion pointed at dynamics instead of task
# transforms: Example = a full sandbox state (grid + avatar + the action taken
# + the gravity parameterization), a transition is an ExamplePair[pre, post],
# and the UNCHANGED generic ES core (fitness / fit_operator / ESWorkspace,
# esper_evolution.mojo) fits WorldModelMemory on transition batches. No new
# learning machinery: predicting next_grid is "just another Domain".
#
# Learning progress (Oudeyer/Schmidhuber; RESEARCH-NOTES §4) falls out for
# free: our ES fitness trajectory IS a prediction-error curve, so LP = the
# fitness delta across a fixed fit stage (lp_probe: clone weights, fit a fixed
# stage at constant alpha/sigma, subtract). Chasing LP instead of raw error is
# what avoids the noisy-TV trap: an unlearnable region keeps HIGH error but
# yields ZERO progress, so an LP-guided collector walks away from it — the
# separation the test gates directly (mastered / novel / scrambled regions).
#
# "Explore where LP is highest" is realized over REGIONS = dynamics contexts
# (gravity directions — SandboxTask's UED surface — plus a TV-static region
# with pseudo-random targets): the guided collector keeps a held-out
# validation batch per region and allocates each round's collection
# proportionally to the round-over-round CHANGED-CELL SCORE delta — the
# windowed LP of the actual learner in the discrete uncheatable currency
# (see train_lp_guided for why fitness-scale LP and clone-probe LP both fail
# as allocators). Spatial return-then-explore over stored elites folds into
# B-POC-4/5.
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.math import fma, tanh, exp, round
from std.random import random_float64
from std.collections import List, InlineArray

from arc_io import Domain, calculate_fitness, exact_match
from memory import Memory
from hope import ExamplePair
from esper_evolution import ESWorkspace, fitness, fit_operator
from sandbox import (
    SB_ROWS,
    SB_COLS,
    SB_CELLS,
    SB_ACTIONS,
    SandboxTask,
    sandbox_step,
)


# ==========================================
# SandboxState — the TransitionDomain Example
# ==========================================
# One full world state plus the action taken FROM it (unused on post-states)
# and the dynamics parameterization the model must condition on. Lifecycle
# mirrors ArcGrid (alloc in __init__, unconditional free in __del__, derived
# Movable) so it slots into the generic ExamplePair/Task containers.
struct SandboxState(Movable):
    var grid: UnsafePointer[Float32, MutAnyOrigin]
    var r: Int
    var c: Int
    var brush: Int
    var action: Int
    var grav_dir: Int
    var grav_rate: Int

    def __init__(out self):
        self.grid = alloc[Float32](SB_CELLS)
        memset_zero(self.grid, SB_CELLS)
        self.r = 0
        self.c = 0
        self.brush = 1
        self.action = 0
        self.grav_dir = 0
        self.grav_rate = 1

    def __del__(deinit self):
        self.grid.free()


# The transition domain: prediction = a flat next-grid buffer, target = the
# post-state; the metrics are the same shape-agnostic SIMD kernels every other
# domain scores through (the honest metric transfer, again).
struct TransitionDomain(Domain):
    comptime Example = SandboxState

    @staticmethod
    def distance(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: SandboxState,
        n: Int,
    ) -> Float32:
        return calculate_fitness(pred, target.grid, n)

    @staticmethod
    def score(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: SandboxState,
        n: Int,
    ) -> Float32:
        return exact_match(pred, target.grid, n)

    @staticmethod
    def capacity(ex: SandboxState) -> Int:
        return SB_CELLS


# ==========================================
# WorldModelMemory — a per-cell local predictor
# ==========================================
# CA-shaped and grid-size independent (the spine: a learned operator, never a
# DSL): each cell's next colour is predicted from its 3x3 neighbourhood, the
# action one-hot, the gravity-direction one-hot, the avatar's clamped offset
# from the cell, and the brush. The true dynamics (gravity is a local
# neighbour rule; paint is an avatar-local write) are expressible in exactly
# this receptive field — but the mapping itself is fit by the ES from
# transitions, never written down.
comptime WM_IN = 24  # 9 patch + 6 action + 4 grav + 2 offset + 2 indicators + 1 brush
comptime WM_HID = 12
# The head is a learned SOFTMAX SELECTOR over value sources — the 9 patch
# cells, the brush, and the constant empty colour — not a free-form value:
# the sandbox's dynamics are selections ("take the colour from above", "take
# the brush", "keep the centre", "become empty"), and a tanh value-head
# provably learns only the saturating gates (measured 2026-07-10: departures
# 113/113, arrivals 0/115 — it cannot emit an exact graded COPY of a
# neighbour's colour). Selection is the same expressivity lesson the
# AttnGather / Rung CF content-fetch families already taught: WHICH source to
# select under WHICH conditions stays fully learned.
comptime WM_SRC = 11  # 9 patch cells + brush + empty
comptime WM_W1_OFF = 0
comptime WM_B1_OFF = WM_IN * WM_HID
comptime WM_W2_OFF = WM_B1_OFF + WM_HID
comptime WM_B2_OFF = WM_W2_OFF + WM_HID * WM_SRC
comptime WM_DIM = WM_B2_OFF + WM_SRC  # 24*12 + 12 + 12*11 + 11 = 443
comptime WM_OOB = Float32(-0.5)
# Selection sharpness: logits are scaled by WM_GAIN inside the softmax so a
# sigma-sized weight perturbation can actually flip a selection (an unscaled
# softmax next to a saturated centre bias left the ES on a flat plateau).
comptime WM_GAIN = Float32(3.0)
comptime WM_CENTER_BIAS = Float32(1.0)


# Feature layout: [0..8] the 3x3 patch, [9..14] action one-hot, [15..18]
# grav one-hot, [19..20] avatar offset, [21] at-avatar indicator, [22]
# avatar-directly-above indicator, [23] brush. The two indicators are honest
# positional features (exact avatar-relative position, no rule injected) —
# without them the paint event (a condition on Δr==0 AND Δc==0) was never
# learned from the graded offset ramps alone (0/17 in every 2026-07-10
# probe). Indices 9..18 and 23 are CONSTANT across a transition's 256 cells,
# so apply hoists their W1 contribution into a per-hidden-unit base
# activation; the per-cell work is the 13 varying inputs.


def _wm_cell(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    base: UnsafePointer[Float32, MutAnyOrigin],
    patch: UnsafePointer[Float32, MutAnyOrigin],
    brush_x: Float32,
    dr_av: Float32,
    dc_av: Float32,
    at_av: Float32,
    above_av: Float32,
) -> Float32:
    # Hidden layer over the varying inputs (base carries the hoisted
    # transition-constant contribution).
    var hid = InlineArray[Float32, WM_HID](fill=0.0)
    for h in range(WM_HID):
        var a = base[h]
        var w1 = weights + WM_W1_OFF + h * WM_IN
        for j in range(9):
            a = fma(w1[j], patch[j], a)
        a = fma(w1[19], dr_av, a)
        a = fma(w1[20], dc_av, a)
        a = fma(w1[21], at_av, a)
        a = fma(w1[22], above_av, a)
        hid[h] = tanh(a)
    # Softmax selection over the value sources.
    var logits = InlineArray[Float32, WM_SRC](fill=0.0)
    var zmax = Float32(-1e30)
    for s in range(WM_SRC):
        var z = weights[WM_B2_OFF + s]
        for h in range(WM_HID):
            z = fma(weights[WM_W2_OFF + h * WM_SRC + s], hid[h], z)
        z = z * WM_GAIN
        logits[s] = z
        if z > zmax:
            zmax = z
    var denom = Float32(0.0)
    var out = Float32(0.0)
    for s in range(WM_SRC):
        var p = exp(logits[s] - zmax)
        denom += p
        var val: Float32
        if s < 9:
            val = patch[s] * 9.0  # a patch cell's colour (OOB -> -4.5)
        elif s == 9:
            val = brush_x * 9.0  # the brush colour
        else:
            val = 0.0  # the empty colour
        out = fma(p, val, out)
    return out / denom


struct WorldModelMemory(Memory):
    comptime Dom = TransitionDomain

    @staticmethod
    def param_dim() -> Int:
        return WM_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Near-identity prior with a broken saddle. Zero W1/b1 makes every
        # hidden unit exactly 0, which zeroes the first-order effect of BOTH
        # layers' perturbations (W2 sees hid = 0; W1 routes through W2 = 0) —
        # the ES measurably never left that saddle. So: small deterministic
        # pseudo-random W1/b1 (own LCG, fixed constant — seed() must stay
        # argument-free and reproducible), W2 = 0, and a positive bias on the
        # CENTRE source's logit (p_centre ≈ 0.98 under WM_GAIN — "keep this
        # cell's colour"). No condition→selection rule is injected; the
        # dynamics are grown entirely by the ES.
        memset_zero(weights, WM_DIM)
        var state = UInt64(0x9E3779B97F4A7C15)
        for i in range(WM_B1_OFF + WM_HID):
            state = state * 6364136223846793005 + 1442695040888963407
            var u = Float32(Int((state >> 33) & 0xFFFF)) / Float32(65536.0)
            weights[i] = (u - 0.5) * 0.6
        weights[WM_B2_OFF + 4] = WM_CENTER_BIAS

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: SandboxState,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # Stack locals only — apply runs inside the ES hot loop, so it must
        # not heap-allocate. `base` holds each hidden unit's bias plus the
        # W1 contribution of the transition-constant features (action, grav,
        # brush); the per-cell loop adds only the 11 varying inputs.
        var base_buf = InlineArray[Float32, WM_HID](fill=0.0)
        var base = base_buf.unsafe_ptr()
        var patch_buf = InlineArray[Float32, 9](fill=0.0)
        var patch = patch_buf.unsafe_ptr()
        var brush_x = Float32(inp.brush) / 9.0
        for h in range(WM_HID):
            var w1 = weights + WM_W1_OFF + h * WM_IN
            var a = weights[WM_B1_OFF + h]
            a = fma(w1[9 + inp.action], 1.0, a)
            a = fma(w1[15 + inp.grav_dir], 1.0, a)
            a = fma(w1[23], brush_x, a)
            base[h] = a
        for r in range(SB_ROWS):
            for c in range(SB_COLS):
                var k = 0
                for dr in range(-1, 2):
                    for dc in range(-1, 2):
                        var rr = r + dr
                        var cc = c + dc
                        if rr < 0 or rr >= SB_ROWS or cc < 0 or cc >= SB_COLS:
                            patch[k] = WM_OOB
                        else:
                            patch[k] = inp.grid[rr * SB_COLS + cc] / 9.0
                        k += 1
                var dr_av = inp.r - r
                if dr_av > 2:
                    dr_av = 2
                if dr_av < -2:
                    dr_av = -2
                var dc_av = inp.c - c
                if dc_av > 2:
                    dc_av = 2
                if dc_av < -2:
                    dc_av = -2
                var at_av = Float32(1.0) if (
                    inp.r == r and inp.c == c
                ) else Float32(0.0)
                var above_av = Float32(1.0) if (
                    inp.r == r - 1 and inp.c == c
                ) else Float32(0.0)
                dst[r * SB_COLS + c] = _wm_cell(
                    weights,
                    base,
                    patch,
                    brush_x,
                    Float32(dr_av) / 2.0,
                    Float32(dc_av) / 2.0,
                    at_av,
                    above_av,
                )


# ==========================================
# Transition collection (the charged budget unit)
# ==========================================
def make_task(grav_dir: Int) -> SandboxTask:
    var task = SandboxTask()
    task.grav_dir = grav_dir
    return task^


def _snapshot(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    r: Int,
    c: Int,
    brush: Int,
    action: Int,
    task: SandboxTask,
) -> SandboxState:
    var s = SandboxState()
    memcpy(dest=s.grid, src=grid, count=SB_CELLS)
    s.r = r
    s.c = c
    s.brush = brush
    s.action = action
    s.grav_dir = task.grav_dir
    s.grav_rate = task.grav_rate
    return s^


# Collect `count` transitions of uniform-random-action experience in `task`'s
# world, in episodes of `ep_len` ticks from the task's start state (fresh
# episode whenever the current one ends). Uniform actions measured best
# (2026-07-10): paint-biased mixes (0.1/0.25) fill the board and make the
# gravity patches HARDER faster than they make the paint rule learnable —
# the rare agent-write event stays an honest unlearned residual at this data
# scale. Serial; draws from the caller's seeded global RNG stream.
def collect_transitions(
    task: SandboxTask,
    count: Int,
    ep_len: Int,
    mut demos: List[ExamplePair[SandboxState]],
):
    var grid = alloc[Float32](SB_CELLS)
    var t_in_ep = ep_len  # force a reset on the first transition
    var r = task.start_r
    var c = task.start_c
    var brush = task.start_brush
    for _ in range(count):
        if t_in_ep >= ep_len:
            memcpy(dest=grid, src=task.grid, count=SB_CELLS)
            r = task.start_r
            c = task.start_c
            brush = task.start_brush
            t_in_ep = 0
        var a = Int(random_float64(0.0, Float64(SB_ACTIONS)))
        if a >= SB_ACTIONS:
            a = SB_ACTIONS - 1
        var pre = _snapshot(grid, r, c, brush, a, task)
        sandbox_step(grid, r, c, brush, task.grav_dir, task.grav_rate, a)
        var post = _snapshot(grid, r, c, brush, 0, task)
        demos.append(ExamplePair[SandboxState](pre^, post^))
        t_in_ep += 1
    grid.free()


# Deep copy of one state (test/driver helper — e.g. building a scrambled
# twin of a batch without touching the original).
def copy_state(src: SandboxState) -> SandboxState:
    var s = SandboxState()
    memcpy(dest=s.grid, src=src.grid, count=SB_CELLS)
    s.r = src.r
    s.c = src.c
    s.brush = src.brush
    s.action = src.action
    s.grav_dir = src.grav_dir
    s.grav_rate = src.grav_rate
    return s^


# The deterministic noisy-TV: make the batch CONTRADICTORY — each odd
# transition's full INPUT state is overwritten with its even neighbour's, so
# pairs share an identical input but keep their two DIFFERENT targets. No
# function can fit both (the optimum is the pair mean, then a hard floor), so
# prediction error stays high while learning progress stays ~0 — at ANY model
# state, and immune to memorization. (A first cut that merely cyclic-shifted
# the targets turned out LEARNABLE: a shift-by-one target is the state two
# gravity ticks ahead, and the model made real LP on it — measured
# 2026-07-10.)
def scramble_targets(mut demos: List[ExamplePair[SandboxState]]):
    var n = len(demos)
    var half = n // 2
    # Pair each transition with its DISTANT partner (i, i+half) — distant
    # states differ in many cells, so the contradiction's error floor is
    # high; adjacent pairs would contradict by only a couple of cells.
    for i in range(half):
        memcpy(
            dest=demos[i + half].input_grid.grid,
            src=demos[i].input_grid.grid,
            count=SB_CELLS,
        )
        demos[i + half].input_grid.r = demos[i].input_grid.r
        demos[i + half].input_grid.c = demos[i].input_grid.c
        demos[i + half].input_grid.brush = demos[i].input_grid.brush
        demos[i + half].input_grid.action = demos[i].input_grid.action


# ==========================================
# Held-out scoring (the uncheatable metric)
# ==========================================
# overall = per-cell exact match on the predicted next grid; changed = exact
# match restricted to cells the transition actually CHANGED — the identity
# predictor ("nothing moves") scores 0 there by construction, so changed-cell
# accuracy is the world-model claim the model cannot inherit from copying.
def held_out_score(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[SandboxState]],
    mut overall: Float32,
    mut changed: Float32,
):
    var pred = alloc[Float32](SB_CELLS)
    var cells = 0
    var hits = 0
    var ch_cells = 0
    var ch_hits = 0
    for d in range(len(demos)):
        WorldModelMemory.apply(weights, demos[d].input_grid, pred)
        var pre = demos[d].input_grid.grid
        var post = demos[d].output_grid.grid
        for i in range(SB_CELLS):
            var p = round(pred[i])
            var t = round(post[i])
            cells += 1
            if p == t:
                hits += 1
            if round(pre[i]) != t:
                ch_cells += 1
                if p == t:
                    ch_hits += 1
    pred.free()
    overall = Float32(hits) / Float32(cells if cells > 0 else 1)
    changed = Float32(ch_hits) / Float32(ch_cells if ch_cells > 0 else 1)


# ==========================================
# Learning progress — the ES fitness slope
# ==========================================
# Clone the current model, fit one fixed stage (CONSTANT alpha/sigma so stages
# are comparable) on the region's probe batch, return the batch-fitness delta.
# reg_lambda = 0 and slow = zeros: the pure prediction signal, no prior.
def lp_probe(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WorldModelMemory],
    demos: List[ExamplePair[SandboxState]],
    stage_iters: Int,
    N: Int,
    alpha: Float32,
    sigma: Float32,
) -> Float32:
    var clone = alloc[Float32](WM_DIM)
    memcpy(dest=clone, src=weights, count=WM_DIM)
    var slow = alloc[Float32](WM_DIM)
    memset_zero(slow, WM_DIM)
    var scratch = alloc[Float32](SB_CELLS)
    var f0 = fitness[WorldModelMemory](clone, slow, demos, scratch, 0.0)
    fit_operator[WorldModelMemory](
        clone, ws, slow, demos, N, alpha, alpha, sigma, sigma, stage_iters, 0.0
    )
    var f1 = fitness[WorldModelMemory](clone, slow, demos, scratch, 0.0)
    clone.free()
    slow.free()
    scratch.free()
    return f1 - f0


# ==========================================
# The two collection arms (equal transition budgets)
# ==========================================
# Both arms run `rounds` rounds; each round collects `round_budget` new
# transitions split across `n_real + 1` regions (gravity directions
# 0..n_real-1 + the TV-static region), then trains the shared model on a
# bounded subsample of everything collected so far for `train_iters` annealed
# ES iterations. The ONLY difference is
# the split: uniform divides evenly; LP-guided first spends `probe_n` charged
# transitions per region, computes each region's LP on that probe batch, and
# splits the remaining budget proportionally to max(LP, 0) (+ a small floor so
# no region is starved forever). Probe COMPUTE is uncharged (stated in-test);
# probe DATA is charged and also trained on — data is data. Returns
# transitions consumed.


def train_round(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WorldModelMemory],
    demos: List[ExamplePair[SandboxState]],
    train_iters: Int,
    N: Int,
):
    var slow = alloc[Float32](WM_DIM)
    memset_zero(slow, WM_DIM)
    # Wide -> mid anneal: the fit is measurably schedule-sensitive (a narrow
    # 0.1-anneal never learns the events; see JOURNAL 2026-07-10).
    fit_operator[WorldModelMemory](
        w,
        ws,
        slow,
        demos,
        N,
        Float32(0.3),
        Float32(0.05),
        Float32(0.3),
        Float32(0.05),
        train_iters,
        0.0,
    )
    slow.free()


# The loop's unlearnable region is classic TV STATIC — every collected
# transition's target grid is replaced with deterministic pseudo-random
# colours (own LCG). Unlike the contradiction twin (gate 2's device, whose
# untouched real half kept yielding honest LP on every fresh draw), static
# targets admit exactly one improvement — predict the mean — after which LP
# collapses while raw error stays high, which is the trap the LP allocator
# must walk away from.
def randomize_targets(mut demos: List[ExamplePair[SandboxState]]):
    var state = UInt64(0x853C49E6748FEA9B)
    for d in range(len(demos)):
        for i in range(SB_CELLS):
            state = state * 6364136223846793005 + 1442695040888963407
            demos[d].output_grid.grid[i] = Float32(Int((state >> 33) % 10))


def _collect_region(
    region: Int,
    n_real: Int,
    count: Int,
    ep_len: Int,
    mut demos: List[ExamplePair[SandboxState]],
):
    var task = make_task(region if region < n_real else 0)
    var fresh = List[ExamplePair[SandboxState]]()
    collect_transitions(task, count, ep_len, fresh)
    if region >= n_real:
        randomize_targets(fresh)
    for _ in range(len(fresh)):
        demos.append(fresh.pop(0))


# Deterministic fixed-size subsample of the cumulative data pool (own LCG):
# per-round training always sees a bounded, well-mixed slice of EVERYTHING
# collected so far — constant compute per round and no catastrophic
# forgetting (training on each round's fresh pool alone let the final
# round's narrow focus overwrite the rest; measured 2026-07-10).
comptime WM_TRAIN_SLICE = 200


def sample_pool(
    all_data: List[ExamplePair[SandboxState]],
    round_idx: Int,
) -> List[ExamplePair[SandboxState]]:
    var out = List[ExamplePair[SandboxState]]()
    var n = len(all_data)
    if n <= WM_TRAIN_SLICE:
        for i in range(n):
            out.append(
                ExamplePair[SandboxState](
                    copy_state(all_data[i].input_grid),
                    copy_state(all_data[i].output_grid),
                )
            )
        return out^
    var state = UInt64(0xD1B54A32D192ED03) + UInt64(round_idx)
    for _ in range(WM_TRAIN_SLICE):
        state = state * 6364136223846793005 + 1442695040888963407
        var i = Int((state >> 33) % UInt64(n))
        out.append(
            ExamplePair[SandboxState](
                copy_state(all_data[i].input_grid),
                copy_state(all_data[i].output_grid),
            )
        )
    return out^


def train_uniform(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WorldModelMemory],
    n_real: Int,
    rounds: Int,
    round_budget: Int,
    ep_len: Int,
    train_iters: Int,
    N: Int,
) -> Int:
    var regions = n_real + 1
    var consumed = 0
    var all_data = List[ExamplePair[SandboxState]]()
    for rnd in range(rounds):
        var per = round_budget // regions
        for region in range(regions):
            var n = per
            if region == regions - 1:
                n = round_budget - per * (regions - 1)
            _collect_region(region, n_real, n, ep_len, all_data)
            consumed += n
        var pool = sample_pool(all_data, rnd)
        train_round(w, ws, pool, train_iters, N)
    return consumed


def train_lp_guided(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WorldModelMemory],
    n_real: Int,
    rounds: Int,
    round_budget: Int,
    ep_len: Int,
    train_iters: Int,
    N: Int,
    val_n: Int,
    verbose: Bool,
) -> Int:
    var regions = n_real + 1
    var consumed = 0
    var scratch = alloc[Float32](SB_CELLS)
    var slow0 = alloc[Float32](WM_DIM)
    memset_zero(slow0, WM_DIM)
    var all_data = List[ExamplePair[SandboxState]]()

    # Persistent per-region validation batches (charged; held out from
    # training so the LP signal measures generalizing improvement, not
    # memorization). LP here is the WINDOWED SCORE slope of the ACTUAL
    # learner: after each round's training, the shared model's held-out
    # CHANGED-CELL accuracy is measured on each region's validation batch,
    # and the next round's collection is allocated proportionally to the
    # round-over-round DELTA. The score (discrete exact-match) is the load-
    # bearing choice: it is scale-free and mean-learning never moves it — on
    # the TV-static region, fitting the noise's mean is a HUGE one-off MSE
    # delta (error floor ~-18 vs real regions' ~-0.1; an MSE-slope allocator
    # measurably handed every round to the noise, 2026-07-10) but never
    # matches a random target exactly, so the static region reads ~0 from
    # round one. (A small clone-probe LP failed differently: on a
    # 16-transition batch it measures batch MEMORIZABILITY, rel-LP ~1.0
    # forever on any real region.)
    var vals = List[List[ExamplePair[SandboxState]]]()
    for region in range(regions):
        var v = List[ExamplePair[SandboxState]]()
        _collect_region(region, n_real, val_n, ep_len, v)
        consumed += val_n
        vals.append(v^)
    var f_prev = alloc[Float32](regions)
    var have_prev = False

    for rnd in range(rounds):
        # --- Allocation shares: even in round 0 (no slope yet), then
        # proportional to max(fitness delta, 0) + floor.
        var alloc_budget = round_budget
        if rnd == 0:
            alloc_budget = round_budget - val_n * regions
            if alloc_budget < 0:
                alloc_budget = 0
        var spent = 0
        for region in range(regions):
            var n: Int
            if not have_prev:
                n = alloc_budget // regions
            else:
                var total = Float32(0.0)
                for r2 in range(regions):
                    var v2 = f_prev[r2]
                    if v2 < 0.0:
                        v2 = 0.0
                    total += v2 + Float32(1e-6)
                var v = f_prev[region]
                if v < 0.0:
                    v = 0.0
                n = Int((v + Float32(1e-6)) / total * Float32(alloc_budget))
            if region == regions - 1:
                n = alloc_budget - spent
            if n < 0:
                n = 0
            if verbose and have_prev:
                print(
                    "    region", region, " LP:", f_prev[region], " alloc:", n
                )
            _collect_region(region, n_real, n, ep_len, all_data)
            consumed += n
            spent += n

        # --- Snapshot validation fitness before training, train, then store
        # the per-region DELTA as the next round's LP.
        var f_before = alloc[Float32](regions)
        var o = Float32(0.0)
        var ch = Float32(0.0)
        for region in range(regions):
            held_out_score(w, vals[region], o, ch)
            f_before[region] = ch
        var pool = sample_pool(all_data, rnd)
        train_round(w, ws, pool, train_iters, N)
        for region in range(regions):
            held_out_score(w, vals[region], o, ch)
            f_prev[region] = ch - f_before[region]
        have_prev = True
        f_before.free()

    f_prev.free()
    scratch.free()
    slow0.free()
    return consumed
