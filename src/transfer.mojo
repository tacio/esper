# ==========================================================================
# B-POC-4 — the convergence test: repertoire -> held-out few-shot transfer.
#
# The fourth Vision-B rung makes the project's convergence hypothesis
# measurable: primitives discovered by open-ended exploration (the B-POC-2
# MAP-Elites repertoire) become the reusable vocabulary that few-shot
# composition draws on to reach a goal fast. It is scored in Vision A's
# uncheatable currency — held-out few-shot transfer using the discovered
# repertoire — through the UNCHANGED generic ES core (fit_operator/ESWorkspace).
#
# A "goal" is a target end-state behaviour characterization (BC). Fitting a
# policy toward it is just fit_operator[SandboxPolicyMemory] over a single demo
# whose Domain target is that BC (SandboxPolicyMemory.apply is a rollout; its
# prediction is the trajectory BC). Two mechanisms, both through the core:
#
#   * WARM-START (single-elite): retrieve the BC-nearest repertoire skill and
#     memcpy its weights as the ES seed (fit_operator takes a caller-prefilled
#     fast_weights buffer — it does not call M.seed). Controls: `cold` (zero
#     seed) and `random-elite` (a uniformly drawn elite) isolate that it is
#     indexed RETRIEVAL, not generic warm-init, that transfers.
#   * COMPOSE (multi-elite): ComposeMemory fits a tiny SCHEDULE head over the K
#     nearest primitives, which ride along FROZEN in the fitted vector's tail
#     via fill_scale=0 (the GeomColor/ShapeMemory freeze trick — scale gates
#     both the perturbation and the update in evolve_fast_weights). Composition
#     is therefore fit by the same fit_operator, no bespoke loop.
#
# Held-out discipline: goals whose Go-Explore cell key is a repertoire bin are
# discarded, so retrieval can never return the exact answer — the few-shot fit
# must close a real gap. The pretraining/few-shot split is made physical: the
# repertoire is built once, saved to a `.rep` file, and reloaded before the
# few-shot phase (EliteMap.save/load_elite_map).
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.math import fma, exp
from std.random import randn_float64, random_float64
from std.collections import List, InlineArray

from sandbox import (
    SB_CELLS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    SandboxDomain,
    SandboxPolicyMemory,
    CellSet,
    sandbox_obs,
    policy_forward,
    policy_argmax,
    sandbox_step,
    sandbox_bc,
    sandbox_cell_key,
    sandbox_rollout,
    sandbox_rollout_state,
)
from map_elites import EliteMap, me_emitter_run, load_elite_map
from novelty_es import NoveltyArchive
from esper_evolution import ESWorkspace, fit_operator
from memory import Memory
from arc_io import calculate_fitness
from hope import ExamplePair

# --- Repertoire build (the unsupervised "pretraining" — NOT charged to the
# few-shot budget, exactly as Vision A's meta-learned prior is not charged
# per-task). Arm-B constants from tests/test_repertoire.mojo. -----------------
comptime BUILD_BUDGET = 13205
comptime BUILD_RESEED = 25
comptime BUILD_N = 16
comptime BUILD_ALPHA = Float32(0.8)
comptime BUILD_SIGMA = Float32(0.4)
comptime INIT_SCALE = Float32(0.5)

# --- Few-shot fit (equalized across ALL arms — same N and iters => same
# rollout count per fit; the only variable is the seed). reg_lambda = 0: pure
# few-shot, no prior anchor (Vision B has no meta-learned slow prior for the
# policy yet), so the seed is isolated as the sole difference. -----------------
comptime FEW_N = 32
comptime FEW_ITERS = 30
comptime FEW_ALPHA0 = Float32(0.1)
comptime FEW_ALPHA1 = Float32(0.02)
comptime FEW_SIGMA0 = Float32(0.3)
comptime FEW_SIGMA1 = Float32(0.05)

# --- Composition fan-in. slots are ordered by ascending BC distance, so slot 0
# is the nearest primitive. The schedule head is seeded biased toward slot 0
# (SCHED_BIAS): the composite STARTS as ~the nearest-elite trajectory (a proper
# floor — composition is a superset of single retrieval), and the few-shot fit
# explores mixing in the other primitives from there. --------------------------
comptime COMPOSE_K = 4
comptime COMPOSE_DIM = COMPOSE_K + COMPOSE_K * POLICY_DIM
comptime SCHED_BIAS = Float32(4.0)


# ==========================================
# Compose rollout (frozen primitives + fitted schedule)
# ==========================================
# Run a temporally-scheduled composite of COMPOSE_K frozen primitives. The
# COMPOSE_K schedule logits (softmax -> cumulative fractions of SB_T) carve the
# horizon into K contiguous segments; segment i is driven by primitive i, whose
# 294 weights live in the frozen tail. All scratch is caller-provided so
# ComposeMemory.apply can drive this zero-alloc from an ES hot loop.
def compose_rollout(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    grid: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logit: UnsafePointer[Float32, MutAnyOrigin],
    bc_out: UnsafePointer[Float32, MutAnyOrigin],
    cells: UnsafePointer[Int64, MutAnyOrigin],
    record_cells: Bool,
):
    # Stable softmax over the schedule head -> per-segment upper boundary (in
    # ticks). bound[i] is the exclusive tick at which segment i ends.
    var mx = weights[0]
    for i in range(1, COMPOSE_K):
        if weights[i] > mx:
            mx = weights[i]
    var s = Float32(0.0)
    var frac = InlineArray[Float32, COMPOSE_K](fill=0.0)
    for i in range(COMPOSE_K):
        var e = exp(weights[i] - mx)
        frac[i] = e
        s += e
    var bound = InlineArray[Float32, COMPOSE_K](fill=0.0)
    var cum = Float32(0.0)
    for i in range(COMPOSE_K):
        cum += frac[i] / s
        bound[i] = cum * Float32(SB_T)

    var tail = weights + COMPOSE_K
    memcpy(dest=grid, src=task.grid, count=SB_CELLS)
    var r = task.start_r
    var c = task.start_c
    var brush = task.start_brush
    for t in range(SB_T):
        var active = COMPOSE_K - 1
        for i in range(COMPOSE_K):
            if Float32(t) < bound[i]:
                active = i
                break
        sandbox_obs(grid, r, c, brush, t, obs)
        policy_forward(tail + active * POLICY_DIM, obs, logit)
        var action = policy_argmax(logit)
        sandbox_step(grid, r, c, brush, task.grav_dir, task.grav_rate, action)
        if record_cells:
            cells[t] = sandbox_cell_key(grid, r, c)
    sandbox_bc(grid, r, c, bc_out)


# ==========================================
# ComposeMemory — composition through the unchanged ES core
# ==========================================
# param = [COMPOSE_K schedule logits (scale 1, fit) ; COMPOSE_K x POLICY_DIM
# frozen primitive weights (scale 0)]. The caller prefills the tail with the K
# retrieved elites and zeros the head; fit_operator then searches only the
# schedule. Same-shape prediction (a BC_DIM behaviour characterization), scored
# through SandboxDomain like the policy memory.
struct ComposeMemory(Memory):
    comptime Dom = SandboxDomain

    @staticmethod
    def param_dim() -> Int:
        return COMPOSE_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        memset_zero(weights, COMPOSE_DIM)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        # 1.0 on the schedule head, 0.0 on the frozen primitive tail — scale
        # gates both perturbation and update, so the primitives never move.
        for i in range(n):
            scale[i] = 1.0 if i < COMPOSE_K else 0.0

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: SandboxTask,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var grid = InlineArray[Float32, SB_CELLS](fill=0.0)
        var obs = InlineArray[Float32, OBS_DIM](fill=0.0)
        var logit = InlineArray[Float32, SB_ACTIONS](fill=0.0)
        var cells = InlineArray[Int64, 1](fill=0)
        compose_rollout(
            weights,
            inp,
            grid.unsafe_ptr(),
            obs.unsafe_ptr(),
            logit.unsafe_ptr(),
            dst,
            cells.unsafe_ptr(),
            False,
        )


# ==========================================
# Two-phase rollout (compositional goal generation)
# ==========================================
# Policy A for the first `switch_t` ticks, then policy B — produces end-states
# that reward SEQUENCING (a single primitive rarely reaches them). Records the
# final cell key and BC. Used only to synthesize Family-C goals; not on any fit
# path.
def two_phase_rollout(
    wA: UnsafePointer[Float32, MutAnyOrigin],
    wB: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    grid: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logit: UnsafePointer[Float32, MutAnyOrigin],
    bc_out: UnsafePointer[Float32, MutAnyOrigin],
    cells: UnsafePointer[Int64, MutAnyOrigin],
    switch_t: Int,
):
    memcpy(dest=grid, src=task.grid, count=SB_CELLS)
    var r = task.start_r
    var c = task.start_c
    var brush = task.start_brush
    for t in range(SB_T):
        sandbox_obs(grid, r, c, brush, t, obs)
        var w = wA if t < switch_t else wB
        policy_forward(w, obs, logit)
        var action = policy_argmax(logit)
        sandbox_step(grid, r, c, brush, task.grav_dir, task.grav_rate, action)
        cells[t] = sandbox_cell_key(grid, r, c)
    sandbox_bc(grid, r, c, bc_out)


# ==========================================
# Goal container + demo builder
# ==========================================
# Goals are stored flat: NUM x BC_DIM target BCs + NUM cell keys. A demo for the
# ES is one ExamplePair whose input is the GIVEN TASK'S WORLD (grid + pose +
# dynamics — so the few-shot fit rolls out in the world the goal lives in;
# T-POC-1 fits in a walls world this way) and whose output carries the goal BC
# (Domain.distance scores a rollout BC against it).
def make_demos(
    task: SandboxTask,
    target: UnsafePointer[Float32, MutAnyOrigin],
) -> List[ExamplePair[SandboxTask]]:
    var a = SandboxTask()
    memcpy(dest=a.grid, src=task.grid, count=SB_CELLS)
    a.start_r = task.start_r
    a.start_c = task.start_c
    a.start_brush = task.start_brush
    a.grav_dir = task.grav_dir
    a.grav_rate = task.grav_rate
    memcpy(dest=a.target_bc, src=target, count=BC_DIM)
    var b = SandboxTask()
    memcpy(dest=b.target_bc, src=target, count=BC_DIM)
    var demos = List[ExamplePair[SandboxTask]]()
    demos.append(ExamplePair[SandboxTask](a^, b^))
    return demos^


# ==========================================
# Per-goal scoring (higher = closer to goal)
# ==========================================
# Score a FITTED policy: its rollout-BC fitness against the goal BC (calculate_
# fitness = negative MSE, so ~0 is a perfect reach) plus the uncheatable exact
# cell-key hit. Shared scratch, caller-owned.
def policy_score(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    target_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: Int64,
    grid: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logit: UnsafePointer[Float32, MutAnyOrigin],
    bc: UnsafePointer[Float32, MutAnyOrigin],
    cells: UnsafePointer[Int64, MutAnyOrigin],
) -> Tuple[Float32, Int]:
    sandbox_rollout(weights, task, grid, obs, logit, bc, cells, True)
    var hit = 1 if cells[SB_T - 1] == goal_key else 0
    return (calculate_fitness(bc, target_bc, BC_DIM), hit)


def compose_score(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    target_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: Int64,
    grid: UnsafePointer[Float32, MutAnyOrigin],
    obs: UnsafePointer[Float32, MutAnyOrigin],
    logit: UnsafePointer[Float32, MutAnyOrigin],
    bc: UnsafePointer[Float32, MutAnyOrigin],
    cells: UnsafePointer[Int64, MutAnyOrigin],
) -> Tuple[Float32, Int]:
    compose_rollout(weights, task, grid, obs, logit, bc, cells, True)
    var hit = 1 if cells[SB_T - 1] == goal_key else 0
    return (calculate_fitness(bc, target_bc, BC_DIM), hit)


# ==========================================
# Transfer result
# ==========================================
# Aggregate mean fitness (higher = closer) and exact-hit fraction per arm, for
# one goal family. Kept as a plain POD so run_transfer can hand both families
# back to the test for gating.
struct ArmStats(Copyable, Movable):
    var cold: Float32
    var randel: Float32
    var nearest: Float32
    var compose: Float32
    var cold_hit: Float32
    var randel_hit: Float32
    var nearest_hit: Float32
    var compose_hit: Float32
    var n: Int

    def __init__(out self):
        self.cold = 0.0
        self.randel = 0.0
        self.nearest = 0.0
        self.compose = 0.0
        self.cold_hit = 0.0
        self.randel_hit = 0.0
        self.nearest_hit = 0.0
        self.compose_hit = 0.0
        self.n = 0


# ==========================================
# The transfer driver
# ==========================================
# One family of goals through all four arms at equal few-shot budget. Reuses the
# two pre-built workspaces and slow/fast buffers; builds fresh demos per goal in
# the given task's world (fit and score in the same world the goals live in).
def run_family(
    mut emap: EliteMap,
    task: SandboxTask,
    goal_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: UnsafePointer[Int64, MutAnyOrigin],
    num: Int,
    mut pol_ws: ESWorkspace[SandboxPolicyMemory],
    mut cmp_ws: ESWorkspace[ComposeMemory],
    pol_slow: UnsafePointer[Float32, MutAnyOrigin],
    cmp_slow: UnsafePointer[Float32, MutAnyOrigin],
) -> ArmStats:
    var pol_fast = alloc[Float32](POLICY_DIM)
    var cmp_fast = alloc[Float32](COMPOSE_DIM)
    var slots = alloc[Int](COMPOSE_K)
    # Scoring scratch.
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logit = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)

    var stats = ArmStats()
    stats.n = num

    for g in range(num):
        var tgt = goal_bc + g * BC_DIM
        var key = goal_key[g]
        var demos = make_demos(task, tgt)

        # --- cold: zero seed.
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
        var r_cold = policy_score(
            pol_fast, task, tgt, key, grid, obs, logit, bc, cells
        )
        stats.cold += r_cold[0]
        stats.cold_hit += Float32(r_cold[1])

        # --- random-elite: a uniformly drawn stored elite as seed (the honest
        # control — generic warm-init without matching).
        var r_slot = emap.select_uniform(Float32(random_float64(0.0, 1.0)))
        memcpy(
            dest=pol_fast,
            src=emap.weights + r_slot * POLICY_DIM,
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
        var r_rand = policy_score(
            pol_fast, task, tgt, key, grid, obs, logit, bc, cells
        )
        stats.randel += r_rand[0]
        stats.randel_hit += Float32(r_rand[1])

        # --- nearest-elite: BC-nearest stored skill as seed (the transfer arm).
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
        var r_near = policy_score(
            pol_fast, task, tgt, key, grid, obs, logit, bc, cells
        )
        stats.nearest += r_near[0]
        stats.nearest_hit += Float32(r_near[1])

        # --- compose: the K nearest primitives frozen in the tail, schedule
        # head fit through the same core.
        emap.nearest_k(tgt, slots, COMPOSE_K)
        memset_zero(cmp_fast, COMPOSE_K)
        cmp_fast[0] = SCHED_BIAS
        for i in range(COMPOSE_K):
            memcpy(
                dest=cmp_fast + COMPOSE_K + i * POLICY_DIM,
                src=emap.weights + slots[i] * POLICY_DIM,
                count=POLICY_DIM,
            )
        fit_operator[ComposeMemory](
            cmp_fast,
            cmp_ws,
            cmp_slow,
            demos,
            FEW_N,
            FEW_ALPHA0,
            FEW_ALPHA1,
            FEW_SIGMA0,
            FEW_SIGMA1,
            FEW_ITERS,
            Float32(0.0),
        )
        var r_cmp = compose_score(
            cmp_fast, task, tgt, key, grid, obs, logit, bc, cells
        )
        stats.compose += r_cmp[0]
        stats.compose_hit += Float32(r_cmp[1])

    var inv = Float32(1.0) / Float32(num if num > 0 else 1)
    stats.cold *= inv
    stats.randel *= inv
    stats.nearest *= inv
    stats.compose *= inv
    stats.cold_hit *= inv
    stats.randel_hit *= inv
    stats.nearest_hit *= inv
    stats.compose_hit *= inv

    pol_fast.free()
    cmp_fast.free()
    slots.free()
    grid.free()
    obs.free()
    logit.free()
    bc.free()
    cells.free()
    return stats^


# ==========================================
# Goal generators
# ==========================================
# Family S: end-states of fresh random single policies (RNG stream disjoint from
# the build), held out (cell key not a repertoire bin), de-duplicated by key.
def gen_family_s(
    mut emap: EliteMap,
    task: SandboxTask,
    goal_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: UnsafePointer[Int64, MutAnyOrigin],
    num: Int,
    max_tries: Int,
) raises -> Int:
    var w = alloc[Float32](POLICY_DIM)
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logit = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var seen = CellSet()
    var found = 0
    var tries = 0
    var r = 0
    var c = 0
    var brush = 0
    while found < num and tries < max_tries:
        tries += 1
        for j in range(POLICY_DIM):
            w[j] = Float32(randn_float64(0.0, 1.0)) * INIT_SCALE
        sandbox_rollout_state(
            w, task, grid, obs, logit, bc, cells, True, r, c, brush
        )
        var k = cells[SB_T - 1]
        if emap.contains(k):
            continue
        if not seen.insert(k):
            continue
        memcpy(dest=goal_bc + found * BC_DIM, src=bc, count=BC_DIM)
        goal_key[found] = k
        found += 1
    w.free()
    grid.free()
    obs.free()
    logit.free()
    bc.free()
    cells.free()
    return found


# Family C: end-states of two-phase (A-then-B) rollouts, held out and
# de-duplicated. These reward sequencing.
def gen_family_c(
    mut emap: EliteMap,
    task: SandboxTask,
    goal_bc: UnsafePointer[Float32, MutAnyOrigin],
    goal_key: UnsafePointer[Int64, MutAnyOrigin],
    num: Int,
    max_tries: Int,
) raises -> Int:
    var wA = alloc[Float32](POLICY_DIM)
    var wB = alloc[Float32](POLICY_DIM)
    var grid = alloc[Float32](SB_CELLS)
    var obs = alloc[Float32](OBS_DIM)
    var logit = alloc[Float32](SB_ACTIONS)
    var bc = alloc[Float32](BC_DIM)
    var cells = alloc[Int64](SB_T)
    var seen = CellSet()
    var found = 0
    var tries = 0
    while found < num and tries < max_tries:
        tries += 1
        for j in range(POLICY_DIM):
            wA[j] = Float32(randn_float64(0.0, 1.0)) * INIT_SCALE
            wB[j] = Float32(randn_float64(0.0, 1.0)) * INIT_SCALE
        two_phase_rollout(wA, wB, task, grid, obs, logit, bc, cells, SB_T // 2)
        var k = cells[SB_T - 1]
        if emap.contains(k):
            continue
        if not seen.insert(k):
            continue
        memcpy(dest=goal_bc + found * BC_DIM, src=bc, count=BC_DIM)
        goal_key[found] = k
        found += 1
    wA.free()
    wB.free()
    grid.free()
    obs.free()
    logit.free()
    bc.free()
    cells.free()
    return found
