# ==========================================================================
# T-POC-2 — adaptation: re-ground a retrieved skill in a new world's
# dynamics by fitting it INSIDE a learned model of that world (the
# Ha-Schmidhuber split with ES on both sides; RESEARCH-NOTES §4).
#
# T-POC-1 measured that the repertoire's INDEX carries across a world change
# but the warm basin does not: a BC-nearest world-1 skill lands at cold-
# parity under walls. This module supplies the missing piece — a fully
# LEARNED dream of world 2 in which the retrieved skill is adapted before
# the standard real few-shot fit ever runs:
#
#   * The GRID half of the dream is the B-POC-3 world model, fit on charged
#     world-2 transitions through WeightedWMMemory (world_model.mojo) — the
#     same forward and the same weights, fit against the changed-cell
#     objective that removes the identity basin. Its raw-argument forward
#     (dream_wm_step) lives there too, so the imagined state can sit in
#     stack scratch rather than a heap-owning SandboxState.
#   * The AGENT half (pose + brush under blocked moves, plus the agent's
#     two WRITE cells — paint and the gravity-settle cell) is
#     PoseStepMemory, selector heads in the WM's own philosophy: candidate
#     values are computed FROM the state, and WHICH candidate applies under
#     WHICH neighbourhood/action is fully learned by the ES from the same
#     transitions. Hand-coding the walls or paint rule into the dream would
#     hand the engine the very dynamics the rung claims to transfer (stone
#     soup), so nothing here steps the world.
#   * DreamPolicyMemory is the twice-proven frozen-tail composition
#     (ComposeMemory / ShapeMemory fill_scale=0): param = [POLICY_DIM fitted
#     policy ; WM_DIM + POSE_DIM frozen fitted models]. Its apply is a full
#     imagined rollout whose prediction is the trajectory BC, so the
#     UNCHANGED fit_operator[DreamPolicyMemory] adapts a policy toward a
#     goal BC entirely in imagination — dream ticks cost zero real ticks.
#
# STATUS (2026-07-15): the increment-0 gate STOPped — see test_dream_rank,
# the booked regression. The dream half-ORDERS candidates (tau ~0.4-0.7) but
# cannot PICK one: top-1 regret fails decisively in columns (3.9x vs a
# pre-registered 2.0x bar; room only straddles it at 1.97x), and
# fit_dream_policy follows the dream's ARGMAX, not its ordering. The cause is
# an objective mismatch rather than model capacity: one-step accuracy is the
# wrong target for a 64-tick autoregressive rollout. So run_family_adapt /
# fit_dream_policy below are the increment-1 driver UNRUN — they compile and
# are exercised by nothing. Treat them as scaffolding awaiting a dream fit at
# rollout level (ROADMAP Route A), not as a measured result.
#
# Budget honesty: real transitions for the model fits are charged to the
# arm that uses them (amortized across a goal family); dream ticks are
# uncharged but printed (the B-POC-2.5 enumeration-ticks precedent).
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.math import fma, tanh, exp, round
from std.collections import List, InlineArray

from arc_io import Domain, calculate_fitness
from memory import Memory
from hope import ExamplePair
from esper_evolution import ESWorkspace, fit_operator
from sandbox import (
    SB_ROWS,
    SB_COLS,
    SB_CELLS,
    SB_ACTIONS,
    SB_T,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    SandboxDomain,
    sandbox_obs,
    policy_forward,
    policy_argmax,
    sandbox_bc,
)
from world_model import (
    SandboxState,
    WorldModelMemory,
    WeightedWMMemory,
    WMCase,
    wm_cases,
    dream_wm_step,
    dream_commit,
    WM_DIM,
    held_out_score,
    WMRolloutMemory,
    wm_rollout_cases,
    rollout_accuracy_curve,
    WM_ROLLOUT_K,
)
from sandbox import SandboxPolicyMemory
from map_elites import EliteMap
from transfer import (
    make_demos,
    policy_score,
    FEW_N,
    FEW_ITERS,
    FEW_ALPHA0,
    FEW_ALPHA1,
    FEW_SIGMA0,
    FEW_SIGMA1,
)


# ==========================================
# PoseStepMemory — the learned agent phase
# ==========================================
# The agent phase of one tick, fully learned: pose + brush + the two
# agent-affected grid cells (the avatar's own cell, where paint lands, and
# its gravity-side neighbour, where a fresh block settles one pass later).
# The write heads exist because B-POC-3 booked the PAINT event as unlearned
# by the grid model at this data scale — and in these worlds ALL content is
# painted (start grids are empty of blocks), so a dream without a learned
# write channel has frozen occupancy dims and nothing for the WM's gravity
# to move. Paint is an agent WRITE, so it belongs to the agent model.
#
# Everything is the WM-selector philosophy: each head is a learned softmax
# over CANDIDATE values computed from the state — pose over the 5
# board-clamped one-step poses, brush over {keep, cycle}, each write cell
# over {keep, brush, empty, gravity-source} — the receptive field is chosen
# (like the WM's 3x3 patch and at-avatar indicators), the RULE is grown
# entirely by the ES from transitions. Direct input->logit skip weights
# join the tanh pathway: the blocked-move and paint rules are LINEAR in the
# skips (action one-hot up, a −1 wall neighbour down), and without them the
# conjunction measurably never emerged (blocked-acc 0 at every schedule
# tried, 2026-07-15). Nothing is pre-wired — skips seed at zero.
comptime POSE_IN = 12  # 6 action one-hot + 4 neighbours + own cell + brush
comptime POSE_HID = 6
comptime POSE_MOVES = 5  # stay, up, down, left, right
comptime POSE_BRUSH = 2  # keep, cycle (brush % 9 + 1)
comptime POSE_CELL = 4  # keep, brush, empty, gravity-source
comptime POSE_HEADS = POSE_MOVES + POSE_BRUSH + 2 * POSE_CELL  # 15
comptime POSE_OUT = 5  # r, c, brush, at-cell value, below-cell value
comptime POSE_W1_OFF = 0
comptime POSE_B1_OFF = POSE_IN * POSE_HID
comptime POSE_WH_OFF = POSE_B1_OFF + POSE_HID
comptime POSE_BH_OFF = POSE_WH_OFF + POSE_HID * POSE_HEADS
comptime POSE_D_OFF = POSE_BH_OFF + POSE_HEADS
comptime POSE_DIM = POSE_D_OFF + POSE_IN * POSE_HEADS  # 72+6+90+15+180 = 363
# Head slot offsets inside the 15 logits.
comptime POSE_H_POSE = 0
comptime POSE_H_BRUSH = POSE_MOVES
comptime POSE_H_AT = POSE_MOVES + POSE_BRUSH
comptime POSE_H_BELOW = POSE_H_AT + POSE_CELL
# Selection-sharpness (the WM_GAIN device) and the near-identity prior
# biases ("stay"/"keep" heads). GAIN 3 + bias 1 saturated into a hard-stay
# saddle the ES measurably could not leave; 2/0.5 stays escapable.
comptime POSE_GAIN = Float32(2.0)
comptime POSE_STAY_BIAS = Float32(0.5)


# One agent-phase training/inference case, self-contained: the NN input
# features, every head's candidate VALUES, and (on target cases) the five
# supervision values read from the post-state at the PRE avatar position.
# Precomputing candidates at case-build time keeps the Memory's apply pure
# math (no grid indexing), gives the write heads honest supervision (the
# Domain never has to reconstruct pre-positions from a post-state), and
# stays heap-free (InlineArrays only) so dream ticks can build one on the
# stack inside the ES hot loop.
struct AgentCase(Copyable, Movable):
    var x: InlineArray[Float32, POSE_IN]
    var cand_r: InlineArray[Float32, POSE_MOVES]
    var cand_c: InlineArray[Float32, POSE_MOVES]
    var cand_brush: InlineArray[Float32, POSE_BRUSH]
    var cand_at: InlineArray[Float32, POSE_CELL]
    var cand_below: InlineArray[Float32, POSE_CELL]
    var below_valid: Float32  # 0 when the gravity neighbour is off-board
    var action: Int
    var target: InlineArray[Float32, POSE_OUT]

    def __init__(out self):
        self.x = InlineArray[Float32, POSE_IN](fill=0.0)
        self.cand_r = InlineArray[Float32, POSE_MOVES](fill=0.0)
        self.cand_c = InlineArray[Float32, POSE_MOVES](fill=0.0)
        self.cand_brush = InlineArray[Float32, POSE_BRUSH](fill=0.0)
        self.cand_at = InlineArray[Float32, POSE_CELL](fill=0.0)
        self.cand_below = InlineArray[Float32, POSE_CELL](fill=0.0)
        self.below_valid = 0.0
        self.action = 0
        self.target = InlineArray[Float32, POSE_OUT](fill=0.0)


# Cell feature encoding: colours/9 with the NEGATIVE range rescaled to −1
# (walls/OOB at full contrast — at the obs encoding's −0.111 the wall sat
# an order below the action one-hots and the block rule was never learned;
# raw colour units instead saturated the tanh layer).
def _cell_feat(v: Float32) -> Float32:
    return v / 9.0 if v >= 0.0 else Float32(-1.0)


def _cell_at(
    grid: UnsafePointer[Float32, MutAnyOrigin], r: Int, c: Int
) -> Float32:
    if r < 0 or r >= SB_ROWS or c < 0 or c >= SB_COLS:
        return -1.0
    return grid[r * SB_COLS + c]


# Build one case's inputs + candidates from a pre-state. The gravity
# displacement (dr_g, dc_g) picks which neighbour is the settle cell and
# which is each write cell's arrival source — candidate plumbing, chosen by
# the task's declared grav_dir exactly as the WM conditions on it.
def make_agent_case(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    r: Int,
    c: Int,
    brush: Int,
    action: Int,
    grav_dir: Int,
) -> AgentCase:
    var ac = AgentCase()
    ac.action = action
    ac.x[action] = 1.0
    # Neighbours in candidate order (up, down, left, right), then own cell.
    var v_up = _cell_at(grid, r - 1, c)
    var v_dn = _cell_at(grid, r + 1, c)
    var v_lf = _cell_at(grid, r, c - 1)
    var v_rt = _cell_at(grid, r, c + 1)
    var v_own = grid[r * SB_COLS + c]
    ac.x[6] = _cell_feat(v_up)
    ac.x[7] = _cell_feat(v_dn)
    ac.x[8] = _cell_feat(v_lf)
    ac.x[9] = _cell_feat(v_rt)
    ac.x[10] = _cell_feat(v_own)
    ac.x[11] = Float32(brush) / 9.0
    # Pose candidates (board-clamped — edges are given arena geometry; wall
    # blocking is NOT encoded, it must be selected against).
    ac.cand_r[0] = Float32(r)
    ac.cand_c[0] = Float32(c)
    ac.cand_r[1] = Float32(r - 1 if r > 0 else r)
    ac.cand_c[1] = Float32(c)
    ac.cand_r[2] = Float32(r + 1 if r < SB_ROWS - 1 else r)
    ac.cand_c[2] = Float32(c)
    ac.cand_r[3] = Float32(r)
    ac.cand_c[3] = Float32(c - 1 if c > 0 else c)
    ac.cand_r[4] = Float32(r)
    ac.cand_c[4] = Float32(c + 1 if c < SB_COLS - 1 else c)
    ac.cand_brush[0] = Float32(brush)
    ac.cand_brush[1] = Float32(brush % 9 + 1)
    # Gravity displacement.
    var dr_g = 0
    var dc_g = 0
    if grav_dir == 0:
        dr_g = 1
    elif grav_dir == 1:
        dr_g = -1
    elif grav_dir == 2:
        dc_g = -1
    else:
        dc_g = 1
    # Own-cell write candidates: keep, brush, empty, arrival from the
    # gravity source (the cell a falling block reaches this cell FROM).
    ac.cand_at[0] = v_own
    ac.cand_at[1] = Float32(brush)
    ac.cand_at[2] = 0.0
    ac.cand_at[3] = _cell_at(grid, r - dr_g, c - dc_g)
    # Settle-cell (gravity neighbour) write candidates: its own keep value,
    # brush, empty, and ITS gravity source — the avatar cell.
    var rb = r + dr_g
    var cb = c + dc_g
    if rb < 0 or rb >= SB_ROWS or cb < 0 or cb >= SB_COLS:
        ac.below_valid = 0.0
    else:
        ac.below_valid = 1.0
        ac.cand_below[0] = grid[rb * SB_COLS + cb]
        ac.cand_below[1] = Float32(brush)
        ac.cand_below[2] = 0.0
        ac.cand_below[3] = v_own
    return ac^


# Fill a target ac's supervision from the post-state, read at the PRE
# avatar position (where the agent's writes land).
def set_agent_targets(
    mut ac: AgentCase,
    post_grid: UnsafePointer[Float32, MutAnyOrigin],
    post_r: Int,
    post_c: Int,
    post_brush: Int,
    pre_r: Int,
    pre_c: Int,
    grav_dir: Int,
):
    ac.target[0] = Float32(post_r)
    ac.target[1] = Float32(post_c)
    ac.target[2] = Float32(post_brush)
    ac.target[3] = post_grid[pre_r * SB_COLS + pre_c]
    var dr_g = 0
    var dc_g = 0
    if grav_dir == 0:
        dr_g = 1
    elif grav_dir == 1:
        dr_g = -1
    elif grav_dir == 2:
        dc_g = -1
    else:
        dc_g = 1
    var rb = pre_r + dr_g
    var cb = pre_c + dc_g
    if rb >= 0 and rb < SB_ROWS and cb >= 0 and cb < SB_COLS:
        ac.target[4] = post_grid[rb * SB_COLS + cb]
    else:
        ac.target[4] = 0.0


# Convert collected transitions into agent-phase training pairs.
def agent_cases(
    demos: List[ExamplePair[SandboxState]],
) -> List[ExamplePair[AgentCase]]:
    var out = List[ExamplePair[AgentCase]]()
    for d in range(len(demos)):
        var pre = make_agent_case(
            demos[d].input_grid.grid,
            demos[d].input_grid.r,
            demos[d].input_grid.c,
            demos[d].input_grid.brush,
            demos[d].input_grid.action,
            demos[d].input_grid.grav_dir,
        )
        var tgt = AgentCase()
        set_agent_targets(
            tgt,
            demos[d].output_grid.grid,
            demos[d].output_grid.r,
            demos[d].output_grid.c,
            demos[d].output_grid.brush,
            demos[d].input_grid.r,
            demos[d].input_grid.c,
            demos[d].input_grid.grav_dir,
        )
        out.append(ExamplePair[AgentCase](pre^, tgt^))
    return out^


# All 15 head distributions for one ac. Zero-alloc (stack scratch only).
def agent_probs(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    ac: AgentCase,
    p: UnsafePointer[Float32, MutAnyOrigin],  # POSE_HEADS probabilities
):
    var hid = InlineArray[Float32, POSE_HID](fill=0.0)
    for h in range(POSE_HID):
        var a = weights[POSE_B1_OFF + h]
        for j in range(POSE_IN):
            a = fma(weights[POSE_W1_OFF + h * POSE_IN + j], ac.x[j], a)
        hid[h] = tanh(a)
    var z = InlineArray[Float32, POSE_HEADS](fill=0.0)
    for t in range(POSE_HEADS):
        var v = weights[POSE_BH_OFF + t]
        for h in range(POSE_HID):
            v = fma(weights[POSE_WH_OFF + h * POSE_HEADS + t], hid[h], v)
        for j in range(POSE_IN):
            v = fma(weights[POSE_D_OFF + j * POSE_HEADS + t], ac.x[j], v)
        z[t] = v * POSE_GAIN

    # Per-head stable softmax.
    var starts = InlineArray[Int, 5](fill=0)
    starts[0] = POSE_H_POSE
    starts[1] = POSE_H_BRUSH
    starts[2] = POSE_H_AT
    starts[3] = POSE_H_BELOW
    starts[4] = POSE_HEADS
    for head in range(4):
        var a0 = starts[head]
        var a1 = starts[head + 1]
        var zmax = z[a0]
        for t in range(a0 + 1, a1):
            if z[t] > zmax:
                zmax = z[t]
        var denom = Float32(0.0)
        for t in range(a0, a1):
            p[t] = exp(z[t] - zmax)
            denom += p[t]
        for t in range(a0, a1):
            p[t] = p[t] / denom


# Argmax slot of one head — the discrete selection dream ticks commit.
def _head_argmax(
    p: UnsafePointer[Float32, MutAnyOrigin], off: Int, k: Int
) -> Int:
    var best = 0
    for s in range(1, k):
        if p[off + s] > p[off + best]:
            best = s
    return best


# The two agent-phase Domains. The pose/brush factor and the write-cell
# factor are fit as SEPARATE models over the same AgentCase data and the
# same weight layout — the commuting-composition recipe (invariant
# per-factor fits + forward composition): in a single joint fit the write
# objectives measurably starved the pose heads at every weighting tried
# (blocked-acc 1.0 alone -> 0.14-0.53 jointly, 2026-07-15), while each
# factor is reliably learnable in isolation. All values RAW units (a
# [0,1]-normalised MSE left the antithetic update microscopic).
struct PoseDomain(Domain):
    comptime Example = AgentCase

    @staticmethod
    def distance(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: AgentCase,
        n: Int,
    ) -> Float32:
        var total = Float32(0.0)
        for i in range(3):
            var d = pred[i] - target.target[i]
            total += d * d
        return -total / 3.0

    @staticmethod
    def score(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: AgentCase,
        n: Int,
    ) -> Float32:
        var hits = 0
        for i in range(3):
            if round(pred[i]) == round(target.target[i]):
                hits += 1
        return Float32(hits) / 3.0

    @staticmethod
    def capacity(ex: AgentCase) -> Int:
        return 3


struct WriteDomain(Domain):
    comptime Example = AgentCase

    @staticmethod
    def distance(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: AgentCase,
        n: Int,
    ) -> Float32:
        var d0 = pred[0] - target.target[3]
        var d1 = pred[1] - target.target[4]
        return -(d0 * d0 + d1 * d1) / 2.0

    @staticmethod
    def score(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: AgentCase,
        n: Int,
    ) -> Float32:
        var hits = 0
        if round(pred[0]) == round(target.target[3]):
            hits += 1
        if round(pred[1]) == round(target.target[4]):
            hits += 1
        return Float32(hits) / 2.0

    @staticmethod
    def capacity(ex: AgentCase) -> Int:
        return 2


struct PoseStepMemory(Memory):
    comptime Dom = PoseDomain

    @staticmethod
    def param_dim() -> Int:
        return POSE_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # The WM's broken-saddle prior, same rationale: small deterministic
        # pseudo-random W1/b1 (own LCG) so perturbations have first-order
        # effect, zero heads/skips, and a positive bias on every head's
        # KEEP/STAY slot — a near-identity agent the ES teaches to act.
        memset_zero(weights, POSE_DIM)
        var state = UInt64(0xA24BAED4963EE407)
        for i in range(POSE_B1_OFF + POSE_HID):
            state = state * 6364136223846793005 + 1442695040888963407
            var u = Float32(Int((state >> 33) & 0xFFFF)) / Float32(65536.0)
            weights[i] = (u - 0.5) * 0.6
        weights[POSE_BH_OFF + POSE_H_POSE] = POSE_STAY_BIAS
        weights[POSE_BH_OFF + POSE_H_BRUSH] = POSE_STAY_BIAS
        weights[POSE_BH_OFF + POSE_H_AT] = POSE_STAY_BIAS
        weights[POSE_BH_OFF + POSE_H_BELOW] = POSE_STAY_BIAS

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: AgentCase,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # Soft expectation over each head — smooth in the weights for the
        # ES; the dream rollout commits the argmax selections instead.
        var p = InlineArray[Float32, POSE_HEADS](fill=0.0)
        agent_probs(weights, inp, p.unsafe_ptr())
        var er = Float32(0.0)
        var ec = Float32(0.0)
        for s in range(POSE_MOVES):
            er = fma(p[POSE_H_POSE + s], inp.cand_r[s], er)
            ec = fma(p[POSE_H_POSE + s], inp.cand_c[s], ec)
        var eb = Float32(0.0)
        for s in range(POSE_BRUSH):
            eb = fma(p[POSE_H_BRUSH + s], inp.cand_brush[s], eb)
        dst[0] = er
        dst[1] = ec
        dst[2] = eb


# The write factor: same weight layout and forward (its pose/brush heads
# are simply never read), fit against the two write-cell targets alone.
struct WriteStepMemory(Memory):
    comptime Dom = WriteDomain

    @staticmethod
    def param_dim() -> Int:
        return POSE_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        PoseStepMemory.seed(weights)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: AgentCase,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var p = InlineArray[Float32, POSE_HEADS](fill=0.0)
        agent_probs(weights, inp, p.unsafe_ptr())
        var ea = Float32(0.0)
        for s in range(POSE_CELL):
            ea = fma(p[POSE_H_AT + s], inp.cand_at[s], ea)
        var el = Float32(0.0)
        for s in range(POSE_CELL):
            el = fma(p[POSE_H_BELOW + s], inp.cand_below[s], el)
        dst[0] = ea
        dst[1] = inp.below_valid * el


# Held-out agent-model accuracy, discrete argmax picks vs targets:
#   move/blocked — pose on move actions (blocked = the world refused it,
#   the walls-specific slice);
#   write — the two write cells restricted to cases where the target
#   DIFFERS from the keep candidate (the identity predictor scores 0 there
#   by construction, exactly like the WM's changed-cell metric).
def pose_held_out(
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    cases: List[ExamplePair[AgentCase]],
    mut move_acc: Float32,
    mut blocked_acc: Float32,
    mut write_acc: Float32,
):
    var moves = 0
    var move_hits = 0
    var blocked = 0
    var blocked_hits = 0
    var writes = 0
    var write_hits = 0
    var p = InlineArray[Float32, POSE_HEADS](fill=0.0)
    var pw2 = InlineArray[Float32, POSE_HEADS](fill=0.0)
    for d in range(len(cases)):
        agent_probs(pose_w, cases[d].input_grid, p.unsafe_ptr())
        agent_probs(write_w, cases[d].input_grid, pw2.unsafe_ptr())
        var pi = _head_argmax(p.unsafe_ptr(), POSE_H_POSE, POSE_MOVES)
        var ai = _head_argmax(pw2.unsafe_ptr(), POSE_H_AT, POSE_CELL)
        var li = _head_argmax(pw2.unsafe_ptr(), POSE_H_BELOW, POSE_CELL)
        var pr = cases[d].input_grid.cand_r[pi]
        var pc = cases[d].input_grid.cand_c[pi]
        var tr = cases[d].output_grid.target[0]
        var tc = cases[d].output_grid.target[1]
        if cases[d].input_grid.action < 4:
            moves += 1
            var hit = 1 if (pr == tr and pc == tc) else 0
            move_hits += hit
            if (
                tr == cases[d].input_grid.cand_r[0]
                and tc == cases[d].input_grid.cand_c[0]
            ):
                blocked += 1
                blocked_hits += hit
        var t_at = round(cases[d].output_grid.target[3])
        if t_at != round(cases[d].input_grid.cand_at[0]):
            writes += 1
            if round(cases[d].input_grid.cand_at[ai]) == t_at:
                write_hits += 1
        if cases[d].input_grid.below_valid > 0.0:
            var t_bl = round(cases[d].output_grid.target[4])
            if t_bl != round(cases[d].input_grid.cand_below[0]):
                writes += 1
                if round(cases[d].input_grid.cand_below[li]) == t_bl:
                    write_hits += 1
    move_acc = Float32(move_hits) / Float32(moves if moves > 0 else 1)
    blocked_acc = Float32(blocked_hits) / Float32(blocked if blocked > 0 else 1)
    write_acc = Float32(write_hits) / Float32(writes if writes > 0 else 1)


# ==========================================
# Robust dream-model fits (best-of-K restarts)
# ==========================================
# The WM fit is run against the CHANGED-CELL-WEIGHTED objective
# (WeightedWMMemory) rather than TransitionDomain's unweighted MSE. The
# history is worth keeping, because two plausible diagnoses were wrong
# before the third was right (JOURNAL 2026-07-15):
#
#   * Not the sigma schedule alone. Widening the wide stage to sigma 0.5 /
#     1500 iters CAUSES an identity collapse rather than escaping it (3/3
#     in world 1 AND columns, at a bit-identical overall across seeds — a
#     deterministic attractor, not a noise draw). The stage stays B-POC-3's
#     0.3/750.
#   * Not the data scale. 4x the transitions rescued nothing (0/3 either
#     way) and lowered event density, since longer collection samples more
#     already-settled states.
#   * It was the OBJECTIVE: unweighted MSE over ~200:1 static-to-changed
#     cells makes "predict keep everywhere" near-optimal, and best-of-K was
#     buying lottery tickets against it. Under the weighted objective every
#     restart escapes — 9/9 across all three worlds, held-out changed 0.93
#     world 1 / 0.88 columns / 0.63 room, versus 2/3, 0/3, 0/3 unweighted.
#
# So restarts are no longer load-bearing against collapse; they now buy the
# best changed x overall PRODUCT. That product matters because the weighted
# objective trades a little static fidelity for event fidelity (overall
# ~0.955-0.99 vs ~0.99+ unweighted), and a sloppy event-learner makes walls
# flicker in an autoregressive 64-tick dream. Selection is on the TRAIN
# batch only (no held-out peeking; an identity collapse would score exactly
# 0 there). WM_ACCEPT is deliberately ABOVE the typical weighted `changed`
# floor so early-accept cannot settle for a low-changed/high-overall draw
# that a full sweep would have beaten on the product.
comptime WM_RESTARTS = 4
comptime WM_ACCEPT = Float32(0.6)
comptime WM_ACCEPT_OVERALL = Float32(0.97)
comptime POSE_RESTARTS = 3


def fit_wm_restarts(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WeightedWMMemory],
    demos: List[ExamplePair[SandboxState]],
    N: Int,
):
    # The weighted objective's cases (mask precomputed once, offline).
    var cases = wm_cases(demos)
    var slow = alloc[Float32](WM_DIM)
    memset_zero(slow, WM_DIM)
    var best = alloc[Float32](WM_DIM)
    var best_q = Float32(-1.0)
    var o = Float32(0.0)
    var ch = Float32(0.0)
    for _ in range(WM_RESTARTS):
        WeightedWMMemory.seed(w)
        fit_operator[WeightedWMMemory](
            w,
            ws,
            slow,
            cases,
            N,
            Float32(0.3),
            Float32(0.1),
            Float32(0.3),
            Float32(0.1),
            750,
            0.0,
        )
        fit_operator[WeightedWMMemory](
            w,
            ws,
            slow,
            cases,
            N,
            Float32(0.08),
            Float32(0.02),
            Float32(0.08),
            Float32(0.02),
            500,
            0.0,
        )
        # Judged on the TRAIN batch only, and through the UNWEIGHTED
        # held_out_score — the fit's objective is weighted but its judge is
        # B-POC-3's, so selection cannot reward the weighting itself. The
        # product demands BOTH event learning (changed — identity scores
        # exactly 0) and static fidelity (overall — a sloppy event-learner
        # makes walls flicker in the dream).
        held_out_score(w, demos, o, ch)
        var q = ch * o
        if q > best_q:
            best_q = q
            memcpy(dest=best, src=w, count=WM_DIM)
        if ch >= WM_ACCEPT and o >= WM_ACCEPT_OVERALL:
            break
    memcpy(dest=w, src=best, count=WM_DIM)
    slow.free()
    best.free()


# ==========================================
# Route A — fit the SAME world model over K ticks of its OWN rollout
# ==========================================
# fit_wm_restarts above is a one-step fit; the STOP that ended T-POC-2
# increment 0 (test_dream_rank, JOURNAL 2026-07-15) diagnosed that as the
# wrong objective for a 64-tick autoregressive dream. This adds a rollout
# fine-tune stage on top of it via WMRolloutMemory (world_model.mojo).
#
# Curriculum, not a cold restart: fit_wm_restarts already proved it escapes
# the identity-collapse basin 9/9 restarts (its own header documents two
# wrong diagnoses before that fix); the K-step objective is plausibly a
# harder landscape (errors compound across ticks before the ES sees
# gradient), so starting the rollout stage cold would repeat a risk this
# codebase already paid to avoid. Warm-starting from the proven one-step fit
# means the fine-tune only has to correct compounding, not rediscover the
# dynamics.
#
# SCHEDULE, empirically pinned (2026-07-16 probes, not guessed): a single
# NARROW stage (sigma 0.05->0.01) produced literally bit-identical weights
# before/after — not just unmeasurable, an exact float32 match across 400+
# iterations. Root-caused with a raw per-iteration fitness trace: the
# continuous K-step objective DOES have a real gradient near the one-step
# optimum, but it is tiny there (~1e-6/iter at sigma=0.3) — the one-step fit
# already saturates most of what a small perturbation can find. An
# unannealed WIDE sigma (1.0) made things measurably WORSE before partially
# recovering (classic overshoot, not a clean ascent) rather than helping.
# The schedule below (0.15 initial, annealed down, TWO stages like every
# other fit_operator recipe in this codebase) is the smallest step that
# reliably produced real, verified movement in a direct apples-to-apples
# check (delta +0.0012 in continuous fitness over 100 iters, weight L2 up to
# 0.14 in a raw evolve_fast_weights trace) — narrower than fit_wm_restarts's
# own cold-start 0.3 (this is a FINE-TUNE off an already-good point, not a
# cold search) but well clear of the 0.05 dead zone.
comptime WM_ROLLOUT_ITERS_1 = 250
comptime WM_ROLLOUT_ALPHA0_1 = Float32(0.12)
comptime WM_ROLLOUT_ALPHA1_1 = Float32(0.05)
comptime WM_ROLLOUT_SIGMA0_1 = Float32(0.15)
comptime WM_ROLLOUT_SIGMA1_1 = Float32(0.06)
comptime WM_ROLLOUT_ITERS_2 = 150
comptime WM_ROLLOUT_ALPHA0_2 = Float32(0.04)
comptime WM_ROLLOUT_ALPHA1_2 = Float32(0.01)
comptime WM_ROLLOUT_SIGMA0_2 = Float32(0.05)
comptime WM_ROLLOUT_SIGMA1_2 = Float32(0.02)
comptime WM_ROLLOUT_N = 16
# Overlapping windows (stride < K): free — no new real ticks — and the first
# lever the plan named for a thin batch (WM_TRAIN=128/EP_LEN=64/K=8 gives
# only 16 NON-overlapping windows). stride=4 roughly quadruples the window
# count at zero data cost.
comptime WM_ROLLOUT_STRIDE = 4


def fit_wm_rollout_restarts(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws_1step: ESWorkspace[WeightedWMMemory],
    mut ws_rollout: ESWorkspace[WMRolloutMemory],
    demos: List[ExamplePair[SandboxState]],
    ep_len: Int,
    N1: Int,
    N2: Int,
):
    # Stage 0: the proven one-step warm start, unchanged.
    fit_wm_restarts(w, ws_1step, demos, N1)

    var pre = alloc[Float32](WM_DIM)
    memcpy(dest=pre, src=w, count=WM_DIM)

    # Stage 1+2: the rollout fine-tune, from the warm start, wide-then-narrow.
    var cases = wm_rollout_cases(demos, ep_len, WM_ROLLOUT_STRIDE)
    var slow = alloc[Float32](WM_DIM)
    memset_zero(slow, WM_DIM)
    fit_operator[WMRolloutMemory](
        w,
        ws_rollout,
        slow,
        cases,
        N2,
        WM_ROLLOUT_ALPHA0_1,
        WM_ROLLOUT_ALPHA1_1,
        WM_ROLLOUT_SIGMA0_1,
        WM_ROLLOUT_SIGMA1_1,
        WM_ROLLOUT_ITERS_1,
        0.0,
    )
    fit_operator[WMRolloutMemory](
        w,
        ws_rollout,
        slow,
        cases,
        N2,
        WM_ROLLOUT_ALPHA0_2,
        WM_ROLLOUT_ALPHA1_2,
        WM_ROLLOUT_SIGMA0_2,
        WM_ROLLOUT_SIGMA1_2,
        WM_ROLLOUT_ITERS_2,
        0.0,
    )

    # Safety net: the fine-tune stage is new and uncalibrated — keep
    # whichever of {pre, post-finetune} has the better TRAIN-batch rollout
    # accuracy (mean changed-cell accuracy across the full-episode curve),
    # so a bad draw of the new stage cannot net-regress the proven warm
    # start. One extra rollout_accuracy_curve call, not a second ES fit —
    # the appropriately-sized version of fit_wm_restarts's own
    # best-by-product selection, scaled to this stage's actual risk.
    var curve = alloc[Float32](ep_len)
    rollout_accuracy_curve(pre, demos, ep_len, curve)
    var pre_score = Float32(0.0)
    for t in range(ep_len):
        pre_score += curve[t]
    pre_score = pre_score / Float32(ep_len)

    rollout_accuracy_curve(w, demos, ep_len, curve)
    var post_score = Float32(0.0)
    for t in range(ep_len):
        post_score += curve[t]
    post_score = post_score / Float32(ep_len)

    if pre_score > post_score:
        memcpy(dest=w, src=pre, count=WM_DIM)

    curve.free()
    pre.free()
    slow.free()


def fit_pose_restarts(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[PoseStepMemory],
    demos: List[ExamplePair[AgentCase]],
    N: Int,
):
    var slow = alloc[Float32](POSE_DIM)
    memset_zero(slow, POSE_DIM)
    var best = alloc[Float32](POSE_DIM)
    var best_q = Float32(-1.0e30)
    var mv = Float32(0.0)
    var bl = Float32(0.0)
    var wr = Float32(0.0)
    for _ in range(POSE_RESTARTS):
        PoseStepMemory.seed(w)
        fit_operator[PoseStepMemory](
            w,
            ws,
            slow,
            demos,
            N,
            Float32(0.5),
            Float32(0.05),
            Float32(0.5),
            Float32(0.05),
            1500,
            0.0,
        )
        fit_operator[PoseStepMemory](
            w,
            ws,
            slow,
            demos,
            N,
            Float32(0.1),
            Float32(0.01),
            Float32(0.1),
            Float32(0.01),
            750,
            0.0,
        )
        # Judged on the TRAIN cases in the discrete currencies the dream
        # actually commits (argmax picks): the blocked-move rule is the
        # fragile basin (measured 0.14-1.0 across identical schedules), and
        # continuous train fitness does not separate its restarts reliably
        # the way it does for the WM.
        pose_held_out(w, w, demos, mv, bl, wr)
        var q = mv + bl
        if q > best_q:
            best_q = q
            memcpy(dest=best, src=w, count=POSE_DIM)
        if mv >= 0.95 and bl >= 0.9:
            break
    memcpy(dest=w, src=best, count=POSE_DIM)
    slow.free()
    best.free()


def fit_write_restarts(
    w: UnsafePointer[Float32, MutAnyOrigin],
    mut ws: ESWorkspace[WriteStepMemory],
    demos: List[ExamplePair[AgentCase]],
    N: Int,
):
    var slow = alloc[Float32](POSE_DIM)
    memset_zero(slow, POSE_DIM)
    var best = alloc[Float32](POSE_DIM)
    var best_q = Float32(-1.0e30)
    var mv = Float32(0.0)
    var bl = Float32(0.0)
    var wr = Float32(0.0)
    for _ in range(POSE_RESTARTS):
        WriteStepMemory.seed(w)
        fit_operator[WriteStepMemory](
            w,
            ws,
            slow,
            demos,
            N,
            Float32(0.5),
            Float32(0.05),
            Float32(0.5),
            Float32(0.05),
            1500,
            0.0,
        )
        fit_operator[WriteStepMemory](
            w,
            ws,
            slow,
            demos,
            N,
            Float32(0.1),
            Float32(0.01),
            Float32(0.1),
            Float32(0.01),
            750,
            0.0,
        )
        # Judged on the TRAIN cases by changed-target write accuracy (the
        # identity-resistant currency for this factor).
        pose_held_out(w, w, demos, mv, bl, wr)
        if wr > best_q:
            best_q = wr
            memcpy(dest=best, src=w, count=POSE_DIM)
        if wr >= 0.7:
            break
    memcpy(dest=w, src=best, count=POSE_DIM)
    slow.free()
    best.free()


# ==========================================
# The imagined rollout
# ==========================================
# SB_T ticks of observe -> policy -> (learned agent step, learned grid
# step), no call into sandbox_step anywhere: every dynamic consequence is
# the models'. Both model halves are conditioned on the PRE state, exactly
# as they were trained (collect_transitions snapshots pre with the action).
# The grid model predicts the whole next grid; the agent model's discrete
# selections then overwrite the two agent-affected cells (its receptive
# field) — the learned write channel the WM measurably lacks (paint). All
# buffers caller-provided; zero allocation inside.
def dream_rollout(
    policy_w: UnsafePointer[Float32, MutAnyOrigin],
    wm_w: UnsafePointer[Float32, MutAnyOrigin],
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    grid: UnsafePointer[Float32, MutAnyOrigin],
    pred: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    bc_out: UnsafePointer[Float32, MutAnyOrigin],
):
    memcpy(dest=grid, src=task.grid, count=SB_CELLS)
    var r = task.start_r
    var c = task.start_c
    var brush = task.start_brush
    var dr_g = 0
    var dc_g = 0
    if task.grav_dir == 0:
        dr_g = 1
    elif task.grav_dir == 1:
        dr_g = -1
    elif task.grav_dir == 2:
        dc_g = -1
    else:
        dc_g = 1
    var p = InlineArray[Float32, POSE_HEADS](fill=0.0)
    var pw = InlineArray[Float32, POSE_HEADS](fill=0.0)
    for t in range(SB_T):
        sandbox_obs(grid, r, c, brush, t, obs)
        policy_forward(policy_w, obs, logits)
        var action = policy_argmax(logits)
        var ac = make_agent_case(grid, r, c, brush, action, task.grav_dir)
        agent_probs(pose_w, ac, p.unsafe_ptr())
        agent_probs(write_w, ac, pw.unsafe_ptr())
        var pi = _head_argmax(p.unsafe_ptr(), POSE_H_POSE, POSE_MOVES)
        var bi = _head_argmax(p.unsafe_ptr(), POSE_H_BRUSH, POSE_BRUSH)
        var ai = _head_argmax(pw.unsafe_ptr(), POSE_H_AT, POSE_CELL)
        var li = _head_argmax(pw.unsafe_ptr(), POSE_H_BELOW, POSE_CELL)
        # Grid: WM everywhere, then the write model's two cells.
        dream_wm_step(wm_w, grid, r, c, brush, action, task.grav_dir, pred)
        dream_commit(pred, grid)
        grid[r * SB_COLS + c] = round(ac.cand_at[ai])
        if ac.below_valid > 0.0:
            grid[(r + dr_g) * SB_COLS + (c + dc_g)] = round(ac.cand_below[li])
        # Pose + brush.
        r = Int(ac.cand_r[pi])
        c = Int(ac.cand_c[pi])
        brush = Int(ac.cand_brush[bi])
    sandbox_bc(grid, r, c, bc_out)


# Dream goal-distance for one policy (negative MSE of the imagined BC
# against the goal BC — the dream twin of policy_score's fitness half).
def dream_score(
    policy_w: UnsafePointer[Float32, MutAnyOrigin],
    wm_w: UnsafePointer[Float32, MutAnyOrigin],
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    target_bc: UnsafePointer[Float32, MutAnyOrigin],
    grid: UnsafePointer[Float32, MutAnyOrigin],
    pred: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    bc: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    dream_rollout(
        policy_w, wm_w, pose_w, write_w, task, grid, pred, obs, logits, bc
    )
    return calculate_fitness(bc, target_bc, BC_DIM)


# ==========================================
# DreamPolicyMemory — adaptation through the unchanged core
# ==========================================
# param = [POLICY_DIM policy (scale 1, fit) ; WM + pose + write models,
# frozen (scale 0)]. The caller prefills the policy block with the
# retrieved seed and the tail with the world-2-fit models;
# fit_operator[DreamPolicyMemory] then searches only the policy, scored by
# imagined rollouts through SandboxDomain — adaptation at zero real ticks.
comptime DREAM_WM_OFF = POLICY_DIM
comptime DREAM_POSE_OFF = POLICY_DIM + WM_DIM
comptime DREAM_WRITE_OFF = DREAM_POSE_OFF + POSE_DIM
comptime DREAM_DIM = DREAM_WRITE_OFF + POSE_DIM


struct DreamPolicyMemory(Memory):
    comptime Dom = SandboxDomain

    @staticmethod
    def param_dim() -> Int:
        return DREAM_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        memset_zero(weights, DREAM_DIM)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        # 1.0 on the policy block, 0.0 on the frozen model tail — scale
        # gates both perturbation and update, so the dream never edits
        # itself to please the fit.
        for i in range(n):
            scale[i] = 1.0 if i < POLICY_DIM else 0.0

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: SandboxTask,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var grid = InlineArray[Float32, SB_CELLS](fill=0.0)
        var pred = InlineArray[Float32, SB_CELLS](fill=0.0)
        var obs = InlineArray[Float32, OBS_DIM](fill=0.0)
        var logits = InlineArray[Float32, SB_ACTIONS](fill=0.0)
        dream_rollout(
            weights,
            weights + DREAM_WM_OFF,
            weights + DREAM_POSE_OFF,
            weights + DREAM_WRITE_OFF,
            inp,
            grid.unsafe_ptr(),
            pred.unsafe_ptr(),
            obs.unsafe_ptr(),
            logits.unsafe_ptr(),
            dst,
        )


# Fit a policy toward a goal ENTIRELY in imagination: prefill the frozen
# tail with the fitted models, seed the policy block with `policy` (the
# retrieved elite, or zeros for the cold control), run the unchanged
# fit_operator, and hand the adapted policy block back. Every rollout here
# is imagined (uncharged; the caller counts the dream ticks).
def fit_dream_policy(
    policy: UnsafePointer[Float32, MutAnyOrigin],
    wm_w: UnsafePointer[Float32, MutAnyOrigin],
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[SandboxTask]],
    mut ws: ESWorkspace[DreamPolicyMemory],
    N: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
):
    var w = alloc[Float32](DREAM_DIM)
    var slow = alloc[Float32](DREAM_DIM)
    memset_zero(slow, DREAM_DIM)
    memcpy(dest=w, src=policy, count=POLICY_DIM)
    memcpy(dest=w + DREAM_WM_OFF, src=wm_w, count=WM_DIM)
    memcpy(dest=w + DREAM_POSE_OFF, src=pose_w, count=POSE_DIM)
    memcpy(dest=w + DREAM_WRITE_OFF, src=write_w, count=POSE_DIM)
    fit_operator[DreamPolicyMemory](
        w, ws, slow, demos, N, alpha0, alpha1, sigma0, sigma1, iters, 0.0
    )
    memcpy(dest=policy, src=w, count=POLICY_DIM)
    w.free()
    slow.free()


# ==========================================
# The T-POC-2 adaptation arms (increment 1)
# ==========================================
# Mean goal-distance MSE magnitudes and exact-hit fractions for one goal
# family, five arms. Plain POD, mirroring transfer.ArmStats.
struct AdaptStats(Copyable, Movable):
    var cold: Float32
    var nearest: Float32
    var dream: Float32
    var colddream: Float32
    var puredream: Float32
    var cold_hit: Float32
    var nearest_hit: Float32
    var dream_hit: Float32
    var colddream_hit: Float32
    var puredream_hit: Float32
    var n: Int

    def __init__(out self):
        self.cold = 0.0
        self.nearest = 0.0
        self.dream = 0.0
        self.colddream = 0.0
        self.puredream = 0.0
        self.cold_hit = 0.0
        self.nearest_hit = 0.0
        self.dream_hit = 0.0
        self.colddream_hit = 0.0
        self.puredream_hit = 0.0
        self.n = 0


# One family of world-2 goals through the five arms:
#   cold+real / nearest+real     — T-POC-1's baselines, unchanged budgets.
#   nearest+dream->real (THE arm) — retrieve, adapt in imagination, then the
#                                   standard real few-shot fit.
#   cold+dream->real (control)   — separates the dream's contribution from
#                                   the index's.
#   nearest+pure-dream (reported) — dream fit only; the sample-efficiency
#                                   view (its only real ticks are the shared
#                                   transitions and the final scoring
#                                   rollout).
# Every arm's final score is a REAL rollout (policy_score — uncheatable).
# The dream fit reuses the real few-shot schedule (FEW_*) so the imagined
# stage is budget-shaped exactly like the real one; its ticks are imagined
# (uncharged), tallied into dream_ticks for the honesty print.
def run_family_adapt(
    mut emap: EliteMap,
    task: SandboxTask,
    goal_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: UnsafePointer[Int64, MutAnyOrigin],
    num: Int,
    wm_w: UnsafePointer[Float32, MutAnyOrigin],
    pose_w: UnsafePointer[Float32, MutAnyOrigin],
    write_w: UnsafePointer[Float32, MutAnyOrigin],
    mut pol_ws: ESWorkspace[SandboxPolicyMemory],
    mut dream_ws: ESWorkspace[DreamPolicyMemory],
    pol_slow: UnsafePointer[Float32, MutAnyOrigin],
    mut dream_ticks: Int,
) -> AdaptStats:
    var pol_fast = alloc[Float32](POLICY_DIM)
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logit = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)

    var stats = AdaptStats()
    stats.n = num

    for g in range(num):
        var tgt = goal_bc + g * BC_DIM
        var key = goal_key[g]
        var demos = make_demos(task, tgt)

        # --- cold+real.
        memset_zero(pol_fast, POLICY_DIM)
        fit_operator[SandboxPolicyMemory](
            pol_fast,
            pol_ws,
            pol_slow,
            demos,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
            Float32(0.0),
        )
        var r = policy_score(
            pol_fast, task, tgt, key, grid, obs, logit, bc, cells
        )
        stats.cold += -r[0]
        stats.cold_hit += Float32(r[1])

        # --- nearest+real.
        var n_slot = emap.nearest(tgt)
        memcpy(
            dest=pol_fast,
            src=emap.weights + n_slot * POLICY_DIM,
            count=POLICY_DIM,
        )
        fit_operator[SandboxPolicyMemory](
            pol_fast,
            pol_ws,
            pol_slow,
            demos,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
            Float32(0.0),
        )
        r = policy_score(pol_fast, task, tgt, key, grid, obs, logit, bc, cells)
        stats.nearest += -r[0]
        stats.nearest_hit += Float32(r[1])

        # --- nearest+dream->real (THE arm).
        memcpy(
            dest=pol_fast,
            src=emap.weights + n_slot * POLICY_DIM,
            count=POLICY_DIM,
        )
        fit_dream_policy(
            pol_fast,
            wm_w,
            pose_w,
            write_w,
            demos,
            dream_ws,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
        )
        dream_ticks += 2 * FEW_N * FEW_ITERS * SB_T
        fit_operator[SandboxPolicyMemory](
            pol_fast,
            pol_ws,
            pol_slow,
            demos,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
            Float32(0.0),
        )
        r = policy_score(pol_fast, task, tgt, key, grid, obs, logit, bc, cells)
        stats.dream += -r[0]
        stats.dream_hit += Float32(r[1])

        # --- cold+dream->real (control).
        memset_zero(pol_fast, POLICY_DIM)
        fit_dream_policy(
            pol_fast,
            wm_w,
            pose_w,
            write_w,
            demos,
            dream_ws,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
        )
        dream_ticks += 2 * FEW_N * FEW_ITERS * SB_T
        fit_operator[SandboxPolicyMemory](
            pol_fast,
            pol_ws,
            pol_slow,
            demos,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
            Float32(0.0),
        )
        r = policy_score(pol_fast, task, tgt, key, grid, obs, logit, bc, cells)
        stats.colddream += -r[0]
        stats.colddream_hit += Float32(r[1])

        # --- nearest+pure-dream (reported, not gated).
        memcpy(
            dest=pol_fast,
            src=emap.weights + n_slot * POLICY_DIM,
            count=POLICY_DIM,
        )
        fit_dream_policy(
            pol_fast,
            wm_w,
            pose_w,
            write_w,
            demos,
            dream_ws,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
        )
        dream_ticks += 2 * FEW_N * FEW_ITERS * SB_T
        r = policy_score(pol_fast, task, tgt, key, grid, obs, logit, bc, cells)
        stats.puredream += -r[0]
        stats.puredream_hit += Float32(r[1])

    var inv = Float32(1.0) / Float32(num if num > 0 else 1)
    stats.cold *= inv
    stats.nearest *= inv
    stats.dream *= inv
    stats.colddream *= inv
    stats.puredream *= inv
    stats.cold_hit *= inv
    stats.nearest_hit *= inv
    stats.dream_hit *= inv
    stats.colddream_hit *= inv
    stats.puredream_hit *= inv

    pol_fast.free()
    grid.free()
    obs.free()
    logit.free()
    bc.free()
    cells.free()
    return stats^
