# ==========================================================================
# MAP-Elites repertoire (Vision B / B-POC-2): a PERSISTENT elite-per-cell
# archive of policies over the sandbox (Mouret & Clune 2015 recast onto
# Go-Explore end-state cells). B-POC-1 proved directed novelty search covers
# the world; its product was transient — population and BC archive discarded,
# only counts survived. This module makes the product persistent: every
# rollout's END-STATE cell (sandbox_cell_key of the final state) is a bin, and
# the bin keeps the WEIGHTS of the best policy that ended there. The map is
# the skill library B-POC-4's few-shot transfer will consume, and rung #6's
# persistence machinery re-hosted where it belongs.
#
# With no reward channel, "best" within a bin = DIRECTNESS: the earliest tick
# from which the trajectory sits in its final cell and never leaves
# (settle_tick, computed from the per-tick cells log). Goal-free, and directer
# skills are better reuse currency. A child replaces the incumbent only on a
# strictly smaller settle tick.
#
# Two variation arms share the map type (compared head-to-head at equal
# rollout budget in tests/test_repertoire.mojo):
# - me_mutation_run (arm A): canonical MAP-Elites — uniform parent from the
#   filled bins, Gaussian perturbation, one rollout, fill-or-improve.
# - me_emitter_run (arm B): CMA-ME-flavoured — a single novelty-ES emitter
#   (the ns_es_run skeleton, same two documented deviations: fixed
#   alpha/sigma, unit-std fitness shaping) re-seeded from a uniform elite
#   every E iterations, with EVERY rollout harvested into the map.
# Budgets are equal in ROLLOUTS — the honest cost unit: every rollout in
# either arm gets exactly one map-insert attempt.
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.sys import simd_width_of
from std.math import fma, sqrt
from std.random import randn_float64, random_float64
from std.algorithm import parallelize

from sandbox import (
    SB_CELLS,
    SB_T,
    SB_ACTIONS,
    OBS_DIM,
    BC_DIM,
    POLICY_DIM,
    SandboxTask,
    SandboxPolicyMemory,
    CellSet,
    sandbox_rollout,
)
from novelty_es import NoveltyArchive

comptime nelts = simd_width_of[DType.float32]()

# Elite-map capacity: open-addressing slots; inserts drop at 50% load so
# probing stays bounded. B-POC-1 reached ~1.4k distinct end-state cells at
# this budget, so the 8k usable bins never bind. ~21 MB of parallel arrays —
# allocated once, never in a hot loop.
comptime ELITE_CAP = 1 << 14
comptime ELITE_MASK = ELITE_CAP - 1
comptime ELITE_EMPTY = Int64(-1)


# The directness quality: the earliest tick t such that every cell from t to
# the end equals the final cell (the trajectory has settled). Lower is better.
# A trajectory still moving at the end gets SB_T - 1 — a valid worst rank, no
# special-casing.
def settle_tick(cells: UnsafePointer[Int64, MutAnyOrigin]) -> Int:
    var last = cells[SB_T - 1]
    var t = SB_T - 1
    while t > 0 and cells[t - 1] == last:
        t -= 1
    return t


# ==========================================
# EliteMap — the persistent repertoire
# ==========================================
# Open-addressing Int64 -> slot hash map (CellSet's Fibonacci-hash/linear-
# probe/sentinel pattern) with per-slot payload in parallel arrays: the elite
# policy's weights, its behaviour characterization, and its settle tick. A
# dense `filled` slot-index list gives O(1) uniform parent selection. Inserts
# are strictly serial (the drivers merge after their parallel sections);
# reads of stored weights during parent selection happen only in the same
# serial phases.
struct EliteMap(Movable):
    var keys: UnsafePointer[Int64, MutAnyOrigin]
    var settle: UnsafePointer[Int, MutAnyOrigin]
    var weights: UnsafePointer[Float32, MutAnyOrigin]
    var bc: UnsafePointer[Float32, MutAnyOrigin]
    var filled: UnsafePointer[Int, MutAnyOrigin]
    var count: Int
    # Refinement metrics: number of quality-improving replacements, and the
    # sum of each bin's FIRST-fill settle tick (so the test can compare the
    # final mean settle against the unrefined mean at identical bins).
    var replaced: Int
    var first_settle_sum: Int

    def __init__(out self):
        self.keys = alloc[Int64](ELITE_CAP)
        for i in range(ELITE_CAP):
            self.keys[i] = ELITE_EMPTY
        self.settle = alloc[Int](ELITE_CAP)
        self.weights = alloc[Float32](ELITE_CAP * POLICY_DIM)
        self.bc = alloc[Float32](ELITE_CAP * BC_DIM)
        self.filled = alloc[Int](ELITE_CAP)
        self.count = 0
        self.replaced = 0
        self.first_settle_sum = 0

    def __del__(deinit self):
        self.keys.free()
        self.settle.free()
        self.weights.free()
        self.bc.free()
        self.filled.free()

    # Fill-or-improve: a new key claims an empty slot (below 50% load); an
    # existing key is replaced only by a strictly smaller settle tick.
    # Returns True when the map changed.
    def insert(
        mut self,
        key: Int64,
        settle_t: Int,
        w: UnsafePointer[Float32, MutAnyOrigin],
        bc_v: UnsafePointer[Float32, MutAnyOrigin],
    ) -> Bool:
        var h = Int((key * 0x9E3779B97F4A7C15) & Int64(ELITE_MASK))
        while True:
            if self.keys[h] == key:
                if settle_t < self.settle[h]:
                    self.settle[h] = settle_t
                    memcpy(
                        dest=self.weights + h * POLICY_DIM,
                        src=w,
                        count=POLICY_DIM,
                    )
                    memcpy(dest=self.bc + h * BC_DIM, src=bc_v, count=BC_DIM)
                    self.replaced += 1
                    return True
                return False
            if self.keys[h] == ELITE_EMPTY:
                if self.count * 2 >= ELITE_CAP:
                    return False
                self.keys[h] = key
                self.settle[h] = settle_t
                memcpy(
                    dest=self.weights + h * POLICY_DIM,
                    src=w,
                    count=POLICY_DIM,
                )
                memcpy(dest=self.bc + h * BC_DIM, src=bc_v, count=BC_DIM)
                self.filled[self.count] = h
                self.count += 1
                self.first_settle_sum += settle_t
                return True
            h = (h + 1) & ELITE_MASK

    # Uniform parent selection: u in [0,1) from the caller's RNG stream maps
    # to a filled slot index.
    def select_uniform(self, u: Float32) -> Int:
        var i = Int(u * Float32(self.count))
        if i >= self.count:
            i = self.count - 1
        return self.filled[i]

    def mean_settle(self) -> Float32:
        if self.count == 0:
            return 0.0
        var total = 0
        for i in range(self.count):
            total += self.settle[self.filled[i]]
        return Float32(total) / Float32(self.count)

    def mean_first_settle(self) -> Float32:
        if self.count == 0:
            return 0.0
        return Float32(self.first_settle_sum) / Float32(self.count)

    # Distinctness: mean Euclidean BC distance over a deterministic pair
    # subsample (own LCG, fixed seed — independent of the global RNG stream so
    # calling this never perturbs a run's reproducibility).
    def mean_pairwise_bc(self, max_pairs: Int) -> Float32:
        if self.count < 2:
            return 0.0
        comptime remainder = BC_DIM % nelts
        comptime rem_start = BC_DIM - remainder
        var state = UInt64(0x243F6A8885A308D3)
        var total = Float32(0.0)
        var pairs = 0
        for _ in range(max_pairs):
            state = state * 6364136223846793005 + 1442695040888963407
            var i = Int((state >> 33) % UInt64(self.count))
            state = state * 6364136223846793005 + 1442695040888963407
            var j = Int((state >> 33) % UInt64(self.count))
            if i == j:
                continue
            var a = self.bc + self.filled[i] * BC_DIM
            var b = self.bc + self.filled[j] * BC_DIM
            var acc = SIMD[DType.float32, nelts](0.0)
            for k in range(0, BC_DIM - nelts + 1, nelts):
                var diff = a.load[width=nelts](k) - b.load[width=nelts](k)
                acc = fma(diff, diff, acc)
            var d2 = acc.reduce_add()
            if remainder > 0:
                for k in range(rem_start, BC_DIM):
                    var diff = a[k] - b[k]
                    d2 = fma(diff, diff, d2)
            total += sqrt(d2)
            pairs += 1
        if pairs == 0:
            return 0.0
        return total / Float32(pairs)

    # --- B-POC-4 seams: membership, retrieval, serialization ---------------
    # These are read-only over the stored elites (retrieval/membership) or a
    # single serial dump/reload; none run in a hot loop, so they favour clarity.

    # Is `key` a filled bin? (Same Fibonacci-hash/linear-probe walk as insert.)
    # The held-out-goal filter uses this so retrieval can never return the exact
    # answer.
    def contains(self, key: Int64) -> Bool:
        var h = Int((key * 0x9E3779B97F4A7C15) & Int64(ELITE_MASK))
        while True:
            if self.keys[h] == key:
                return True
            if self.keys[h] == ELITE_EMPTY:
                return False
            h = (h + 1) & ELITE_MASK

    # Hash slot holding `key`, or -1 if absent (for the serialization-fidelity
    # check: look an original elite up by its reloaded key).
    def find(self, key: Int64) -> Int:
        var h = Int((key * 0x9E3779B97F4A7C15) & Int64(ELITE_MASK))
        while True:
            if self.keys[h] == key:
                return h
            if self.keys[h] == ELITE_EMPTY:
                return -1
            h = (h + 1) & ELITE_MASK

    # Squared Euclidean BC distance from a stored elite (hash slot `slot`) to a
    # target BC — the mean_pairwise_bc SIMD/FMA kernel, one operand external.
    def bc_dist2(
        self, slot: Int, target: UnsafePointer[Float32, MutAnyOrigin]
    ) -> Float32:
        comptime remainder = BC_DIM % nelts
        comptime rem_start = BC_DIM - remainder
        var a = self.bc + slot * BC_DIM
        var acc = SIMD[DType.float32, nelts](0.0)
        for k in range(0, BC_DIM - nelts + 1, nelts):
            var diff = a.load[width=nelts](k) - target.load[width=nelts](k)
            acc = fma(diff, diff, acc)
        var d2 = acc.reduce_add()
        if remainder > 0:
            for k in range(rem_start, BC_DIM):
                var diff = a[k] - target[k]
                d2 = fma(diff, diff, d2)
        return d2

    # Nearest elite to a target BC. Returns the hash SLOT index (as `filled`
    # holds), so the caller reads `self.weights + slot * POLICY_DIM` exactly like
    # the replay path; -1 when the map is empty.
    def nearest(self, target: UnsafePointer[Float32, MutAnyOrigin]) -> Int:
        if self.count == 0:
            return -1
        var best_slot = self.filled[0]
        var best_d2 = self.bc_dist2(best_slot, target)
        for i in range(1, self.count):
            var slot = self.filled[i]
            var d2 = self.bc_dist2(slot, target)
            if d2 < best_d2:
                best_d2 = d2
                best_slot = slot
        return best_slot

    # The k nearest elites (ascending distance) into out_slots[0..k). k is tiny
    # (the compose fan-in), so k successive min-scans excluding already-picked is
    # cheap. If the map holds fewer than k DISTINCT elites, the tail repeats the
    # last pick so the compose primitive slots are always fully populated.
    def nearest_k(
        self,
        target: UnsafePointer[Float32, MutAnyOrigin],
        out_slots: UnsafePointer[Int, MutAnyOrigin],
        k: Int,
    ):
        var chosen = 0
        while chosen < k:
            var best_slot = -1
            var best_d2 = Float32(1.0e30)
            for i in range(self.count):
                var slot = self.filled[i]
                var skip = False
                for p in range(chosen):
                    if out_slots[p] == slot:
                        skip = True
                        break
                if skip:
                    continue
                var d2 = self.bc_dist2(slot, target)
                if d2 < best_d2:
                    best_d2 = d2
                    best_slot = slot
            if best_slot < 0:
                out_slots[chosen] = out_slots[chosen - 1] if chosen > 0 else (
                    self.filled[0] if self.count > 0 else 0
                )
            else:
                out_slots[chosen] = best_slot
            chosen += 1

    # Serialize the filled elites to a raw `.rep` binary (the deferred B-POC-2
    # seam that physically decouples the unsupervised build phase from the
    # few-shot phase). Layout: header [count, POLICY_DIM, BC_DIM] as int64, then
    # per elite [key:int64, settle:int64, weights:POLICY_DIM f32, bc:BC_DIM f32].
    # One buffer, one write — mirrors arc_io's single-read readers in reverse.
    def save(self, path: String) raises:
        var n = self.count
        var rec = 16 + (POLICY_DIM + BC_DIM) * 4
        var total = 24 + n * rec
        var buf = alloc[UInt8](total)
        var hp = buf.bitcast[Int64]()
        hp[0] = Int64(n)
        hp[1] = Int64(POLICY_DIM)
        hp[2] = Int64(BC_DIM)
        var off = 24
        for i in range(n):
            var slot = self.filled[i]
            var ip = (buf + off).bitcast[Int64]()
            ip[0] = self.keys[slot]
            ip[1] = Int64(self.settle[slot])
            off += 16
            memcpy(
                dest=(buf + off).bitcast[Float32](),
                src=self.weights + slot * POLICY_DIM,
                count=POLICY_DIM,
            )
            off += POLICY_DIM * 4
            memcpy(
                dest=(buf + off).bitcast[Float32](),
                src=self.bc + slot * BC_DIM,
                count=BC_DIM,
            )
            off += BC_DIM * 4
        var f = open(path, "w")
        f.write_bytes(Span[UInt8, MutAnyOrigin](ptr=buf, length=total))
        f.close()
        buf.free()


# ==========================================
# Arm A — canonical mutation MAP-Elites
# ==========================================
# Seed the map with n_seed fresh randn*init_scale policies (the same init
# distribution as the NS-ES arm's population — fairness), then batches of N:
# serial parent picks + serial Gaussian child draws (fixed RNG order),
# parallel rollouts in disjoint stripes, serial merge (coverage sets + one
# fill-or-improve insert per child). The final batch shrinks to land on the
# rollout budget EXACTLY. Returns rollouts consumed.
def me_mutation_run(
    mut emap: EliteMap,
    task: SandboxTask,
    mut coverage: CellSet,
    mut end_cov: CellSet,
    budget: Int,
    N: Int,
    sigma_mut: Float32,
    init_scale: Float32,
    n_seed: Int,
) -> Int:
    if budget <= 0 or N <= 0 or n_seed <= 0:
        return 0
    comptime pdim = POLICY_DIM

    var child_all = alloc[Float32](N * pdim)
    var grid_all = alloc[Float32](N * SB_CELLS)
    var obs_all = alloc[Float32](N * OBS_DIM)
    var logit_all = alloc[Float32](N * SB_ACTIONS)
    var bc_all = alloc[Float32](N * BC_DIM)
    var cells_all = alloc[Int64](N * SB_T)

    var rollouts = 0

    # --- Seeds: fresh random policies, serial (n_seed is tiny; reuses the
    # first stripe of every buffer).
    for _ in range(n_seed):
        if rollouts >= budget:
            break
        for j in range(pdim):
            child_all[j] = Float32(randn_float64(0.0, 1.0)) * init_scale
        sandbox_rollout(
            child_all,
            task,
            grid_all,
            obs_all,
            logit_all,
            bc_all,
            cells_all,
            True,
        )
        rollouts += 1
        for t in range(SB_T):
            _ = coverage.insert(cells_all[t])
        _ = end_cov.insert(cells_all[SB_T - 1])
        _ = emap.insert(
            cells_all[SB_T - 1], settle_tick(cells_all), child_all, bc_all
        )

    # --- Batched mutation loop.
    while rollouts < budget:
        var n_batch = N
        if budget - rollouts < n_batch:
            n_batch = budget - rollouts

        # Serial: pick parents + draw children in fixed RNG order.
        for s in range(n_batch):
            var slot = emap.select_uniform(Float32(random_float64(0.0, 1.0)))
            var parent = emap.weights + slot * pdim
            var child = child_all + s * pdim
            for j in range(pdim):
                child[j] = (
                    parent[j] + Float32(randn_float64(0.0, 1.0)) * sigma_mut
                )

        @parameter
        def sample(s: Int):
            sandbox_rollout(
                child_all + s * pdim,
                task,
                grid_all + s * SB_CELLS,
                obs_all + s * OBS_DIM,
                logit_all + s * SB_ACTIONS,
                bc_all + s * BC_DIM,
                cells_all + s * SB_T,
                True,
            )

        parallelize[sample](n_batch)
        rollouts += n_batch

        # Serial merge: coverage + one insert attempt per child.
        for s in range(n_batch):
            var cells_s = cells_all + s * SB_T
            for t in range(SB_T):
                _ = coverage.insert(cells_s[t])
            _ = end_cov.insert(cells_s[SB_T - 1])
            _ = emap.insert(
                cells_s[SB_T - 1],
                settle_tick(cells_s),
                child_all + s * pdim,
                bc_all + s * BC_DIM,
            )

    child_all.free()
    grid_all.free()
    obs_all.free()
    logit_all.free()
    bc_all.free()
    cells_all.free()
    return rollouts


# ==========================================
# Arm B — ES-emitter MAP-Elites (CMA-ME-flavoured)
# ==========================================
# One novelty-ES emitter (the ns_es_run skeleton: serial epsilon draws,
# parallel antithetic rollouts, unit-std fitness shaping — see novelty_es.mojo
# for why shaping and the fixed alpha/sigma deviate from the fit_operator
# schedule) whose weight vector is re-seeded from a uniformly-selected elite
# every reseed_every iterations. EVERY rollout — both antithetic sides and the
# per-iteration center eval — is harvested into the map, which is why the
# perturbation stripe here is N x 2 x pdim (ns_es_run reuses one stripe for
# both signs; harvesting needs the exact weights that earned each BC still
# live at insert time). Novelty against `archive` stays the ES scalar; the map
# is the product, not the fitness. Leftover budget below one full iteration's
# cost (2N+1) is spent on fresh random-policy rollouts so both arms consume
# the budget EXACTLY. Returns rollouts consumed.
def me_emitter_run(
    mut emap: EliteMap,
    task: SandboxTask,
    mut archive: NoveltyArchive,
    mut coverage: CellSet,
    mut end_cov: CellSet,
    budget: Int,
    reseed_every: Int,
    N: Int,
    alpha: Float32,
    sigma: Float32,
    init_scale: Float32,
) -> Int:
    if budget <= 0 or N <= 0 or reseed_every <= 0:
        return 0
    comptime pdim = POLICY_DIM
    comptime remainder = pdim % nelts
    comptime rem_start = pdim - remainder

    var w = alloc[Float32](pdim)
    var eps_all = alloc[Float32](N * pdim)
    var pert_all = alloc[Float32](N * 2 * pdim)
    var grad = alloc[Float32](pdim)
    var scale = alloc[Float32](pdim)
    var coeff = alloc[Float32](N)
    var grid_all = alloc[Float32](N * SB_CELLS)
    var obs_all = alloc[Float32](N * OBS_DIM)
    var logit_all = alloc[Float32](N * SB_ACTIONS)
    var bc_all = alloc[Float32](N * 2 * BC_DIM)
    var cells_all = alloc[Int64](N * 2 * SB_T)
    SandboxPolicyMemory.fill_scale(scale, pdim)

    var rollouts = 0

    # --- First seed: a fresh random policy; its rollout both starts the
    # novelty archive and opens the map.
    for j in range(pdim):
        w[j] = Float32(randn_float64(0.0, 1.0)) * init_scale
    sandbox_rollout(
        w, task, grid_all, obs_all, logit_all, bc_all, cells_all, True
    )
    rollouts += 1
    archive.add(bc_all)
    for t in range(SB_T):
        _ = coverage.insert(cells_all[t])
    _ = end_cov.insert(cells_all[SB_T - 1])
    _ = emap.insert(cells_all[SB_T - 1], settle_tick(cells_all), w, bc_all)

    var iter = 0
    while rollouts + 2 * N + 1 <= budget:
        # --- Re-seed the emitter from a uniform elite every reseed_every
        # iterations (not at iter 0 — the fresh seed just went in).
        if iter > 0 and iter % reseed_every == 0 and emap.count > 0:
            var slot = emap.select_uniform(Float32(random_float64(0.0, 1.0)))
            memcpy(dest=w, src=emap.weights + slot * pdim, count=pdim)
        iter += 1

        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                eps_s[j] = Float32(randn_float64(0.0, 1.0))

        @parameter
        def sample(s: Int):
            var eps_s = eps_all + s * pdim
            var pert_plus = pert_all + s * 2 * pdim
            var pert_minus = pert_plus + pdim
            var grid_s = grid_all + s * SB_CELLS
            var obs_s = obs_all + s * OBS_DIM
            var logit_s = logit_all + s * SB_ACTIONS
            var bc_plus = bc_all + s * 2 * BC_DIM
            var bc_minus = bc_plus + BC_DIM
            var cells_plus = cells_all + s * 2 * SB_T
            var cells_minus = cells_plus + SB_T

            var pos = SIMD[DType.float32, nelts](sigma)
            for j in range(0, pdim - nelts + 1, nelts):
                var w_vec = w.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert_plus.store[width=nelts](j, fma(seps, pos, w_vec))
            if remainder > 0:
                for j in range(rem_start, pdim):
                    pert_plus[j] = fma(eps_s[j] * scale[j], sigma, w[j])
            sandbox_rollout(
                pert_plus,
                task,
                grid_s,
                obs_s,
                logit_s,
                bc_plus,
                cells_plus,
                True,
            )
            var f_plus = archive.novelty(bc_plus)

            var neg = SIMD[DType.float32, nelts](-sigma)
            for j in range(0, pdim - nelts + 1, nelts):
                var w_vec = w.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert_minus.store[width=nelts](j, fma(seps, neg, w_vec))
            if remainder > 0:
                for j in range(rem_start, pdim):
                    pert_minus[j] = fma(eps_s[j] * scale[j], -sigma, w[j])
            sandbox_rollout(
                pert_minus,
                task,
                grid_s,
                obs_s,
                logit_s,
                bc_minus,
                cells_minus,
                True,
            )
            var f_minus = archive.novelty(bc_minus)

            coeff[s] = f_plus - f_minus

        parallelize[sample](N)
        rollouts += 2 * N

        # --- Serial: coverage merge + harvest all 2N rollouts into the map.
        for i in range(N * 2 * SB_T):
            _ = coverage.insert(cells_all[i])
        for s in range(2 * N):
            var cells_s = cells_all + s * SB_T
            _ = end_cov.insert(cells_s[SB_T - 1])
            _ = emap.insert(
                cells_s[SB_T - 1],
                settle_tick(cells_s),
                pert_all + s * pdim,
                bc_all + s * BC_DIM,
            )

        # --- Unit-std fitness shaping + reduce + preconditioned step (the
        # ns_es_run skeleton verbatim).
        var cmean = Float32(0.0)
        for s in range(N):
            cmean += coeff[s]
        cmean /= Float32(N)
        var cvar = Float32(0.0)
        for s in range(N):
            var d = coeff[s] - cmean
            cvar += d * d
        var csd = sqrt(cvar / Float32(N)) + Float32(1e-9)
        for s in range(N):
            coeff[s] = coeff[s] / csd

        memset_zero(grad, pdim)
        for s in range(N):
            var eps_s = eps_all + s * pdim
            var c_vec = SIMD[DType.float32, nelts](coeff[s])
            for j in range(0, pdim - nelts + 1, nelts):
                var g = grad.load[width=nelts](j)
                grad.store[width=nelts](
                    j, fma(eps_s.load[width=nelts](j), c_vec, g)
                )
            if remainder > 0:
                for j in range(rem_start, pdim):
                    grad[j] = fma(eps_s[j], coeff[s], grad[j])

        var fac = alpha / Float32(N)
        var fac_vec = SIMD[DType.float32, nelts](fac)
        for j in range(0, pdim - nelts + 1, nelts):
            var w_vec = w.load[width=nelts](j)
            var sg = grad.load[width=nelts](j) * scale.load[width=nelts](j)
            w.store[width=nelts](j, fma(sg, fac_vec, w_vec))
        if remainder > 0:
            for j in range(rem_start, pdim):
                w[j] = fma(grad[j] * scale[j], fac, w[j])

        # --- Center eval: archive its BC (the per-generation add), harvest.
        sandbox_rollout(
            w, task, grid_all, obs_all, logit_all, bc_all, cells_all, True
        )
        rollouts += 1
        archive.add(bc_all)
        for t in range(SB_T):
            _ = coverage.insert(cells_all[t])
        _ = end_cov.insert(cells_all[SB_T - 1])
        _ = emap.insert(cells_all[SB_T - 1], settle_tick(cells_all), w, bc_all)

    # --- Leftover budget (< 2N+1): fresh random-policy rollouts so both arms
    # consume the budget exactly.
    while rollouts < budget:
        for j in range(pdim):
            w[j] = Float32(randn_float64(0.0, 1.0)) * init_scale
        sandbox_rollout(
            w, task, grid_all, obs_all, logit_all, bc_all, cells_all, True
        )
        rollouts += 1
        for t in range(SB_T):
            _ = coverage.insert(cells_all[t])
        _ = end_cov.insert(cells_all[SB_T - 1])
        _ = emap.insert(cells_all[SB_T - 1], settle_tick(cells_all), w, bc_all)

    w.free()
    eps_all.free()
    pert_all.free()
    grad.free()
    scale.free()
    coeff.free()
    grid_all.free()
    obs_all.free()
    logit_all.free()
    bc_all.free()
    cells_all.free()
    return rollouts


# ==========================================
# Repertoire reload (B-POC-4 seam)
# ==========================================
# Read a `.rep` binary written by EliteMap.save back into a fresh EliteMap by
# re-inserting each stored elite through the normal hash/fill path (so the
# reloaded map is bit-identical to the saved one, and the replay-fidelity check
# still holds). Single read + parse, mirroring arc_io.load_arc_task.
def load_elite_map(path: String) raises -> EliteMap:
    var f = open(path, "r")
    var data = f.read_bytes()
    f.close()
    if len(data) < 24:
        raise Error("elite map file too small for its header")
    var ptr = data.unsafe_ptr()
    var hp = ptr.bitcast[Int64]()
    var n = Int(hp[0])
    var pdim = Int(hp[1])
    var bdim = Int(hp[2])
    if pdim != POLICY_DIM or bdim != BC_DIM:
        raise Error("elite map serialization dim mismatch")
    var emap = EliteMap()
    var off = 24
    for _ in range(n):
        var ip = (ptr + off).bitcast[Int64]()
        var key = ip[0]
        var settle_t = Int(ip[1])
        off += 16
        var wf = (ptr + off).bitcast[Float32]()
        off += POLICY_DIM * 4
        var bf = (ptr + off).bitcast[Float32]()
        off += BC_DIM * 4
        _ = emap.insert(key, settle_t, wf, bf)
    return emap^
