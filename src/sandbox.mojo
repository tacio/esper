# ==========================================================================
# The Vision B sandbox (B-POC-1): a tiny deterministic gridworld with NO
# reward channel. An avatar moves and paints on a 16x16 board under one
# parameterizable dynamics rule (gravity); the only learning signal anywhere
# is self-generated (novelty over behaviour descriptors — see novelty_es.mojo).
# The rule parameters (grav_dir/grav_rate) live in SandboxTask so a later UED
# rung (B-POC-5) can mutate the world itself without touching this module.
#
# Seam design (recorded in RESEARCH-NOTES 2026-07-10 addendum): the Domain
# Example is a TASK (start state + rule params) and `SandboxPolicyMemory.apply`
# is a full deterministic rollout whose flat prediction is the trajectory's
# behaviour characterization — so the UNCHANGED generic ES core can already fit
# a policy toward a target end-state through Domain.distance (B-POC-4's seam).
# The intrinsic novelty fitness itself cannot flow through Domain.distance
# (static and target-based; it cannot see a runtime archive), so it is hosted
# by the bespoke NS-ES driver, exactly as meta_fit_selfmod hosts its
# meta-fitness.
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.math import fma, tanh
from std.collections import InlineArray

from arc_io import Domain, calculate_fitness, exact_match
from memory import Memory

comptime SB_ROWS = 16
comptime SB_COLS = 16
comptime SB_CELLS = SB_ROWS * SB_COLS
# Colour 0 = empty; 1..9 are paintable block colours (paint never writes empty,
# so painting is not trivially reversible — part of the world's composable depth).
comptime SB_COLORS = 10
# Walls (world 2 — T-POC-1): the ONLY negative cell value. Static topology —
# never falls, blocks the avatar, refuses paint, and blocks falling blocks
# (any non-empty destination already does). Cell semantics: < 0 wall, == 0
# empty, > 0 movable block. Wall-free worlds are a strict superset case: with
# no negative cells present every wall guard below reduces to the old
# behaviour (gated byte-identical on the six Vision-B proofs).
comptime SB_WALL = Float32(-1.0)
# Actions: 0 up, 1 down, 2 left, 3 right, 4 paint, 5 cycle-brush.
comptime SB_ACTIONS = 6
# Rollout length: 16 ticks cross the board, so 64 allows ~4 traverse-and-paint
# phases — enough to build multi-block structures under gravity, while a full
# rollout stays microseconds.
comptime SB_T = 64

# Gravity directions (SandboxTask.grav_dir).
comptime SB_GRAV_DOWN = 0
comptime SB_GRAV_UP = 1
comptime SB_GRAV_LEFT = 2
comptime SB_GRAV_RIGHT = 3

# --- Policy (observation -> action logits) ---
# Egocentric 5x5 colour patch + 4 compass scalars. The patch is colours/9 with
# an out-of-bounds sentinel OUTSIDE the valid [0,1] span so board edges are
# perceivable under clamped movement. Wall cells need no special casing: they
# render through the same formula as SB_WALL/9 ≈ −0.111 — negative like the
# OOB sentinel (impassable reads as "outside the block span") yet distinct
# from it, so a policy can tell interior topology from the board edge. The compass carries normalized avatar
# row/col, brush, and t/SB_T — the time feature lets a FIXED weight vector
# express time-extended programs ("walk right until t~0.5, then paint") and
# breaks the short limit cycles a deterministic argmax policy otherwise enters
# in a near-static world.
comptime SB_PATCH = 5
comptime SB_PATCH_HALF = SB_PATCH // 2
comptime OBS_DIM = SB_PATCH * SB_PATCH + 4
comptime SB_OOB = Float32(-0.5)
comptime SB_HID = 8
comptime P_W1_OFF = 0
comptime P_B1_OFF = OBS_DIM * SB_HID
comptime P_W2_OFF = P_B1_OFF + SB_HID
comptime P_B2_OFF = P_W2_OFF + SB_HID * SB_ACTIONS
comptime POLICY_DIM = P_B2_OFF + SB_ACTIONS  # 29*8 + 8 + 8*6 + 6 = 294

# --- Behaviour characterization (continuous, for kNN novelty) ---
# 16 x occupancy fraction of each 4x4 block of the FINAL grid + final avatar
# row/col normalized. Occupancy (not mean colour) varies smoothly with each
# paint action — a denser ES signal — and every dim lands in [0,1], so no
# per-dim scale mismatch inside the kNN distance.
comptime BC_BLOCK = 4
comptime BC_BLOCKS = (SB_ROWS // BC_BLOCK) * (SB_COLS // BC_BLOCK)
comptime BC_DIM = BC_BLOCKS + 2

# --- Coverage cells (discrete, for the metric — deliberately distinct from
# the BC: the BC must be continuous/low-dim for kNN, the metric must be an
# uncheatable count). Per-tick Int64 key: 16 blocks x 2-bit clamped occupancy
# level (0, 1, 2, >=3 painted cells) | avatar quadrant in the high bits.
comptime CELLSET_CAP = 1 << 16
comptime CELLSET_MASK = CELLSET_CAP - 1
comptime CELLSET_EMPTY = Int64(-1)


# ==========================================
# SandboxTask — the Domain Example
# ==========================================
# A task = an initial world (grid + avatar pose + brush) + the dynamics
# parameterization + an optional target BC (zeros in B-POC-1; only read by the
# Domain metrics, i.e. by a B-POC-4-style "reach this end-state" fit).
# Lifecycle mirrors ArcGrid exactly (alloc + zero in __init__, unconditional
# free in __del__), Movable so it slots into ExamplePair/Task containers.
struct SandboxTask(Movable):
    var grid: UnsafePointer[Float32, MutAnyOrigin]
    var target_bc: UnsafePointer[Float32, MutAnyOrigin]
    var start_r: Int
    var start_c: Int
    var start_brush: Int
    var grav_dir: Int
    var grav_rate: Int

    def __init__(out self):
        self.grid = alloc[Float32](SB_CELLS)
        memset_zero(self.grid, SB_CELLS)
        self.target_bc = alloc[Float32](BC_DIM)
        memset_zero(self.target_bc, BC_DIM)
        self.start_r = SB_ROWS // 2
        self.start_c = SB_COLS // 2
        self.start_brush = 1
        self.grav_dir = SB_GRAV_DOWN
        self.grav_rate = 1

    def __del__(deinit self):
        # A moved-from task is consumed, so its destructor never runs and the
        # live buffers are always valid here — free unconditionally.
        self.grid.free()
        self.target_bc.free()


# ==========================================
# Wall layouts (world 2 — T-POC-1)
# ==========================================
# Deterministic topology builders: write SB_WALL cells into a task's START
# grid (walls live in the grid itself, so SandboxTask/Domain/policy and the
# .rep serialization are all untouched). The avatar's start cell is always
# skipped, so a task can never begin inside a wall — set start_r/start_c
# BEFORE laying walls. Parametric on (kind, variant) so a later UED rung
# inherits a genuine topology axis; this rung uses them as a fixed family.


# Fill the inclusive rect [r0..r1] x [c0..c1] with walls, clamped to the
# board, skipping the task's avatar start cell.
def add_wall_rect(task: SandboxTask, r0: Int, c0: Int, r1: Int, c1: Int):
    var ra = r0
    var rb = r1
    var ca = c0
    var cb = c1
    if ra < 0:
        ra = 0
    if ca < 0:
        ca = 0
    if rb > SB_ROWS - 1:
        rb = SB_ROWS - 1
    if cb > SB_COLS - 1:
        cb = SB_COLS - 1
    for r in range(ra, rb + 1):
        for c in range(ca, cb + 1):
            if r == task.start_r and c == task.start_c:
                continue
            task.grid[r * SB_COLS + c] = SB_WALL


# Wall-layout kinds (gen_walls_layout).
comptime SB_WALLS_SHELVES = 0
comptime SB_WALLS_COLUMNS = 1
comptime SB_WALLS_ROOM = 2
comptime SB_WALLS_SCATTER = 3
comptime SB_WALLS_KINDS = 4


# Lay one of the named topologies into the task's start grid. `variant`
# deterministically shifts gap/wall positions so each (kind, variant) pair is
# a distinct world of the same family. Layouts deliberately create what the
# open world lacks: mid-air support (shelves), corridors (columns), a
# closable region (room), and irregular obstacles (scatter).
def gen_walls_layout(task: SandboxTask, kind: Int, variant: Int):
    if kind == SB_WALLS_SHELVES:
        # Two horizontal shelves with gaps on alternating sides: blocks
        # settle mid-air, the avatar zigzags between levels.
        var gap_a = 2 + (variant % 5)
        var gap_b = 9 + (variant * 3 % 5)
        add_wall_rect(task, 5, 0, 5, gap_a - 1)
        add_wall_rect(task, 5, gap_a + 2, 5, SB_COLS - 1)
        add_wall_rect(task, 10, 0, 10, gap_b - 1)
        add_wall_rect(task, 10, gap_b + 2, 10, SB_COLS - 1)
    elif kind == SB_WALLS_COLUMNS:
        # Three vertical dividers with door gaps at variant-shifted rows:
        # corridors that force detours.
        for k in range(3):
            var col = 3 + 5 * k + (variant % 2)
            var door = 2 + (variant + 4 * k) % 11
            add_wall_rect(task, 0, col, door - 1, col)
            add_wall_rect(task, door + 2, col, SB_ROWS - 1, col)
    elif kind == SB_WALLS_ROOM:
        # A closed rectangle around the avatar start with ONE door: a
        # closable region — the empowerment differentiator.
        var door = 4 + (variant % 8)
        add_wall_rect(task, 4, 4, 4, 11)
        add_wall_rect(task, 12, 4, 12, 11)
        add_wall_rect(task, 4, 4, 12, 4)
        add_wall_rect(task, 4, 11, door - 1, 11)
        add_wall_rect(task, door + 2, 11, 12, 11)
    else:  # SB_WALLS_SCATTER
        # ~14 single wall cells from a tiny LCG stream seeded by variant:
        # irregular moderate-density obstacles.
        var s = Int64(variant * 2 + 1)
        for _ in range(14):
            s = s * 6364136223846793005 + 1442695040888963407
            var h = Int((s >> 33) & 255)
            var r = h % SB_ROWS
            var c = (h // SB_ROWS) % SB_COLS
            add_wall_rect(task, r, c, r, c)


# ==========================================
# World dynamics
# ==========================================
# One gravity pass: every movable block (cell > 0) whose neighbour in the
# gravity direction is empty AND in-bounds moves one cell that way. The scan
# runs in dependency order (destination side first), so a whole unsupported
# column shifts exactly one cell per pass — blocks visibly "fall" one cell per
# tick at grav_rate=1. The avatar is not a block: gravity never moves it.
# Walls (< 0) are excluded as sources so they never fall; as destinations they
# are non-empty, so blocks settle ON them — mid-air shelves hold.
def _gravity_pass(grid: UnsafePointer[Float32, MutAnyOrigin], grav_dir: Int):
    if grav_dir == SB_GRAV_DOWN:
        for r in range(SB_ROWS - 2, -1, -1):
            for c in range(SB_COLS):
                var i = r * SB_COLS + c
                if grid[i] > 0.0 and grid[i + SB_COLS] == 0.0:
                    grid[i + SB_COLS] = grid[i]
                    grid[i] = 0.0
    elif grav_dir == SB_GRAV_UP:
        for r in range(1, SB_ROWS):
            for c in range(SB_COLS):
                var i = r * SB_COLS + c
                if grid[i] > 0.0 and grid[i - SB_COLS] == 0.0:
                    grid[i - SB_COLS] = grid[i]
                    grid[i] = 0.0
    elif grav_dir == SB_GRAV_LEFT:
        for c in range(1, SB_COLS):
            for r in range(SB_ROWS):
                var i = r * SB_COLS + c
                if grid[i] > 0.0 and grid[i - 1] == 0.0:
                    grid[i - 1] = grid[i]
                    grid[i] = 0.0
    else:  # SB_GRAV_RIGHT
        for c in range(SB_COLS - 2, -1, -1):
            for r in range(SB_ROWS):
                var i = r * SB_COLS + c
                if grid[i] > 0.0 and grid[i + 1] == 0.0:
                    grid[i + 1] = grid[i]
                    grid[i] = 0.0


# One world tick: agent phase (move clamped at board edges and blocked by
# wall cells / paint, refused on wall cells / cycle brush), then the dynamics
# phase (grav_rate gravity passes). Fully deterministic. The avatar coexists
# with blocks (it can stand on and paint over them); only walls are solid.
def sandbox_step(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    mut r: Int,
    mut c: Int,
    mut brush: Int,
    grav_dir: Int,
    grav_rate: Int,
    action: Int,
):
    if action == 0:
        if r > 0 and grid[(r - 1) * SB_COLS + c] >= 0.0:
            r -= 1
    elif action == 1:
        if r < SB_ROWS - 1 and grid[(r + 1) * SB_COLS + c] >= 0.0:
            r += 1
    elif action == 2:
        if c > 0 and grid[r * SB_COLS + (c - 1)] >= 0.0:
            c -= 1
    elif action == 3:
        if c < SB_COLS - 1 and grid[r * SB_COLS + (c + 1)] >= 0.0:
            c += 1
    elif action == 4:
        if grid[r * SB_COLS + c] >= 0.0:
            grid[r * SB_COLS + c] = Float32(brush)
    else:  # cycle brush over 1..9 (never paints "empty")
        brush = brush % 9 + 1

    for _ in range(grav_rate):
        _gravity_pass(grid, grav_dir)


# ==========================================
# Observation + policy forward
# ==========================================
def sandbox_obs(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    r: Int,
    c: Int,
    brush: Int,
    t: Int,
    obs: UnsafePointer[Float32, MutAnyOrigin],
):
    var idx = 0
    for dr in range(-SB_PATCH_HALF, SB_PATCH_HALF + 1):
        for dc in range(-SB_PATCH_HALF, SB_PATCH_HALF + 1):
            var rr = r + dr
            var cc = c + dc
            if rr < 0 or rr >= SB_ROWS or cc < 0 or cc >= SB_COLS:
                obs[idx] = SB_OOB
            else:
                obs[idx] = grid[rr * SB_COLS + cc] / 9.0
            idx += 1
    obs[idx] = Float32(r) / Float32(SB_ROWS - 1)
    obs[idx + 1] = Float32(c) / Float32(SB_COLS - 1)
    obs[idx + 2] = Float32(brush) / 9.0
    obs[idx + 3] = Float32(t) / Float32(SB_T)


# One hidden tanh layer, linear logits. Scalar loops with fma — 294 params is
# far below the SIMD payoff threshold; the vectorized mandate is honoured where
# it matters (the ES perturbation/gradient loops and the archive distances).
def policy_forward(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
):
    var hidden = InlineArray[Float32, SB_HID](fill=0.0)
    for h in range(SB_HID):
        var z = weights[P_B1_OFF + h]
        for i in range(OBS_DIM):
            z = fma(weights[P_W1_OFF + h * OBS_DIM + i], obs[i], z)
        hidden[h] = tanh(z)
    for a in range(SB_ACTIONS):
        var z = weights[P_B2_OFF + a]
        for h in range(SB_HID):
            z = fma(weights[P_W2_OFF + a * SB_HID + h], hidden[h], z)
        logits[a] = z


# Argmax with first-max tie-break: fully deterministic action selection, so a
# weight vector IS a reproducible trajectory (no softmax sampling anywhere).
def policy_argmax(logits: UnsafePointer[Float32, MutAnyOrigin]) -> Int:
    var best = 0
    var best_v = logits[0]
    for a in range(1, SB_ACTIONS):
        if logits[a] > best_v:
            best_v = logits[a]
            best = a
    return best


# ==========================================
# Behaviour characterization + coverage cells
# ==========================================
# Occupancy counts only movable/painted content (> 0), never walls: the BC
# measures what the agent's world DID, not its static topology — which keeps
# goal BCs comparable across worlds (the T-POC-1 cross-world retrieval seam).
def sandbox_bc(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    r: Int,
    c: Int,
    bc: UnsafePointer[Float32, MutAnyOrigin],
):
    for br in range(SB_ROWS // BC_BLOCK):
        for bcol in range(SB_COLS // BC_BLOCK):
            var count = 0
            for rr in range(br * BC_BLOCK, (br + 1) * BC_BLOCK):
                for cc in range(bcol * BC_BLOCK, (bcol + 1) * BC_BLOCK):
                    if grid[rr * SB_COLS + cc] > 0.0:
                        count += 1
            bc[br * (SB_COLS // BC_BLOCK) + bcol] = Float32(count) / Float32(
                BC_BLOCK * BC_BLOCK
            )
    bc[BC_BLOCKS] = Float32(r) / Float32(SB_ROWS - 1)
    bc[BC_BLOCKS + 1] = Float32(c) / Float32(SB_COLS - 1)


# Go-Explore-style cell key for one world state: 16 blocks x 2-bit clamped
# occupancy level in the low 32 bits, avatar quadrant in bits 32..35.
# Counts movable content only (> 0), like sandbox_bc — same rationale.
def sandbox_cell_key(
    grid: UnsafePointer[Float32, MutAnyOrigin], r: Int, c: Int
) -> Int64:
    var key = Int64(0)
    var b = 0
    for br in range(SB_ROWS // BC_BLOCK):
        for bcol in range(SB_COLS // BC_BLOCK):
            var count = 0
            for rr in range(br * BC_BLOCK, (br + 1) * BC_BLOCK):
                for cc in range(bcol * BC_BLOCK, (bcol + 1) * BC_BLOCK):
                    if grid[rr * SB_COLS + cc] > 0.0:
                        count += 1
            var level = count
            if level > 3:
                level = 3
            key |= Int64(level) << Int64(2 * b)
            b += 1
    var quadrant = (r // BC_BLOCK) * (SB_COLS // BC_BLOCK) + (c // BC_BLOCK)
    key |= Int64(quadrant) << 32
    return key


# ==========================================
# Rollout
# ==========================================
# Run one deterministic episode: memcpy the task's start grid into the caller's
# scratch, then SB_T ticks of observe -> forward -> argmax -> step. Writes the
# final behaviour characterization into bc_out and (when record_cells) each
# tick's cell key into cells (SB_T entries). ALL buffers are caller-provided —
# zero allocation inside, so the ES hot loop can drive this from pre-allocated
# per-sample stripes.
def sandbox_rollout(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    scratch_grid: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    bc_out: UnsafePointer[Float32, MutAnyOrigin],
    cells: UnsafePointer[Int64, MutAnyOrigin],
    record_cells: Bool,
):
    var r = 0
    var c = 0
    var brush = 0
    sandbox_rollout_state(
        weights,
        task,
        scratch_grid,
        obs,
        logits,
        bc_out,
        cells,
        record_cells,
        r,
        c,
        brush,
    )


# Same rollout, additionally handing back the final avatar state (the final
# grid is already in the caller's scratch): consumers that score the terminal
# STATE rather than the BC — e.g. the empowerment signal — need r/c/brush too.
def sandbox_rollout_state(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    scratch_grid: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logits: UnsafePointer[Float32, MutAnyOrigin],
    bc_out: UnsafePointer[Float32, MutAnyOrigin],
    cells: UnsafePointer[Int64, MutAnyOrigin],
    record_cells: Bool,
    mut r: Int,
    mut c: Int,
    mut brush: Int,
):
    memcpy(dest=scratch_grid, src=task.grid, count=SB_CELLS)
    r = task.start_r
    c = task.start_c
    brush = task.start_brush
    for t in range(SB_T):
        sandbox_obs(scratch_grid, r, c, brush, t, obs)
        policy_forward(weights, obs, logits)
        var action = policy_argmax(logits)
        sandbox_step(
            scratch_grid, r, c, brush, task.grav_dir, task.grav_rate, action
        )
        if record_cells:
            cells[t] = sandbox_cell_key(scratch_grid, r, c)
    sandbox_bc(scratch_grid, r, c, bc_out)


# ==========================================
# CellSet — the coverage metric's container
# ==========================================
# Pre-allocated open-addressing Int64 hash set (linear probing, sentinel -1 —
# valid keys are 36-bit non-negative). Allocated once per run, never in a hot
# loop; inserts are strictly serial (the drivers merge per-sample cell logs
# after the parallel section). Drops inserts at 50% load so probing stays
# bounded — CELLSET_CAP is sized far above any reachable distinct-cell count.
struct CellSet(Movable):
    var data: UnsafePointer[Int64, MutAnyOrigin]
    var count: Int

    def __init__(out self):
        self.data = alloc[Int64](CELLSET_CAP)
        for i in range(CELLSET_CAP):
            self.data[i] = CELLSET_EMPTY
        self.count = 0

    def __del__(deinit self):
        self.data.free()

    # Insert a key; returns True when the key is new. Fibonacci-hash the key
    # into the table, linear-probe from there.
    def insert(mut self, key: Int64) -> Bool:
        if self.count * 2 >= CELLSET_CAP:
            return False
        var h = Int((key * 0x9E3779B97F4A7C15) & Int64(CELLSET_MASK))
        while True:
            if self.data[h] == key:
                return False
            if self.data[h] == CELLSET_EMPTY:
                self.data[h] = key
                self.count += 1
                return True
            h = (h + 1) & CELLSET_MASK


# ==========================================
# Domain + Memory conformances
# ==========================================
# The sandbox Domain: Example = SandboxTask, prediction = a BC_DIM behaviour
# characterization, target = the task's target_bc. Reuses the same SIMD metric
# kernels every other domain scores through.
struct SandboxDomain(Domain):
    comptime Example = SandboxTask

    @staticmethod
    def distance(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: SandboxTask,
        n: Int,
    ) -> Float32:
        return calculate_fitness(pred, target.target_bc, n)

    @staticmethod
    def score(
        pred: UnsafePointer[Float32, MutAnyOrigin],
        target: SandboxTask,
        n: Int,
    ) -> Float32:
        return exact_match(pred, target.target_bc, n)

    @staticmethod
    def capacity(ex: SandboxTask) -> Int:
        return BC_DIM


# The policy as a Memory: `apply` = one full deterministic rollout from the
# task's start state, prediction = the trajectory's behaviour characterization.
# This is the honest "Example = trajectory" seam — the unchanged generic
# fit_operator[SandboxPolicyMemory] can already fit a policy toward a target
# end-state (B-POC-4's scoring path). Rollout scratch is InlineArray stack
# locals (~1.5 KB): no heap allocation even when `apply` sits in an ES hot loop.
struct SandboxPolicyMemory(Memory):
    comptime Dom = SandboxDomain

    @staticmethod
    def param_dim() -> Int:
        return POLICY_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # The zero policy (argmax -> action 0) is a legitimate degenerate
        # start; DIVERSITY across a population is the caller's job (per-agent
        # random inits from the caller's seeded RNG stream), so the trait's
        # seed stays deterministic and argument-free.
        memset_zero(weights, POLICY_DIM)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: SandboxTask,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var grid_buf = InlineArray[Float32, SB_CELLS](fill=0.0)
        var obs_buf = InlineArray[Float32, OBS_DIM](fill=0.0)
        var logit_buf = InlineArray[Float32, SB_ACTIONS](fill=0.0)
        var cells_buf = InlineArray[Int64, 1](fill=0)
        sandbox_rollout(
            weights,
            inp,
            grid_buf.unsafe_ptr(),
            obs_buf.unsafe_ptr(),
            logit_buf.unsafe_ptr(),
            dst,
            cells_buf.unsafe_ptr(),
            False,
        )
