# ==========================================================================
# NS-ES (Vision B / B-POC-1): novelty search inside the antithetic ES.
# Conti et al. 2018 ("Improving Exploration in Evolution Strategies ... via a
# Population of Novelty-Seeking Agents"): K agent weight vectors share a
# behaviour archive; each iteration one agent (chosen with probability
# proportional to its novelty) takes one ES step whose scalar fitness is the
# NOVELTY of the perturbed candidate's behaviour characterization — the mean
# kNN distance to the archive. There is NO reward anywhere: the fitness is
# self-generated, which is the entire point of the rung.
#
# The driver is a self-contained copy of the meta_fit_selfmod ES skeleton
# (esper_evolution.mojo): serial epsilon draws in fixed RNG order, parallel
# antithetic evaluations in disjoint per-sample stripes, serial SIMD/FMA
# reduce + preconditioned step — so the reproducibility argument carries over
# verbatim. It does NOT reuse fitness[M]/fit_operator[M]: novelty cannot flow
# through Domain.distance (static and target-based; it cannot see a runtime
# archive), exactly as meta_fit_selfmod's meta-fitness could not.
#
# Deliberate deviations from the fit_operator schedule, both diagnosed during
# calibration (JOURNAL 2026-07-10):
# - alpha/sigma are FIXED, no annealing — the novelty objective is
#   non-stationary by construction (the archive grows every iteration), so an
#   anneal-to-exploit schedule is conceptually wrong here.
# - The antithetic coefficients are NORMALIZED to unit std before the update
#   (OpenAI-ES-style fitness shaping; step = alpha/N * sum(coeff/sd * eps)).
#   Raw novelty differences are tiny (BC distances ~0.1) and shrink further as
#   the archive densifies, so the un-shaped step barely moved the centers
#   (|dw| ~ 1 over a whole run vs. an init norm of ~8.6) and coverage stayed
#   at the random-baseline level; shaping makes the step size scale-free and
#   quadrupled measured coverage at the same budget.
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.sys import simd_width_of
from std.math import fma, sqrt
from std.random import randn_float64, random_float64
from std.algorithm import parallelize
from std.collections import InlineArray

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

comptime nelts = simd_width_of[DType.float32]()

# kNN size for the novelty estimate (k=10 per Conti et al.; the k-mean — not
# 1-NN — smooths the estimator against archive density noise).
comptime NOVELTY_K = 10
# Archive capacity: the B-POC-1 loop adds K + iters entries (~205), so the cap
# never binds — it exists so the buffer is fixed-size, allocated once.
comptime ARCHIVE_CAP = 2048


# ==========================================
# NoveltyArchive
# ==========================================
# Flat append-only store of behaviour characterizations. `novelty` is
# RE-ENTRANT (no shared scratch in the struct): it is called concurrently from
# the N parallel ES samples while the archive is read-only; `add` runs only in
# the serial phase between parallel sections.
struct NoveltyArchive(Movable):
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var count: Int

    def __init__(out self):
        self.data = alloc[Float32](ARCHIVE_CAP * BC_DIM)
        memset_zero(self.data, ARCHIVE_CAP * BC_DIM)
        self.count = 0

    def __del__(deinit self):
        self.data.free()

    def add(mut self, bc: UnsafePointer[Float32, MutAnyOrigin]):
        if self.count >= ARCHIVE_CAP:
            return
        memcpy(dest=self.data + self.count * BC_DIM, src=bc, count=BC_DIM)
        self.count += 1

    # Mean Euclidean distance to the min(NOVELTY_K, count) nearest archive
    # entries. The squared-distance scan is the driver's hottest loop
    # (2N x count evaluations per iteration) — three-part SIMD+FMA over BC_DIM;
    # the k-smallest pass is a local insertion into a NOVELTY_K-slot array
    # (k is tiny; no heap, no sort, no shared state).
    def novelty(self, bc: UnsafePointer[Float32, MutAnyOrigin]) -> Float32:
        if self.count == 0:
            return 0.0
        var k = NOVELTY_K
        if self.count < k:
            k = self.count
        var best = InlineArray[Float32, NOVELTY_K](fill=Float32(1e30))

        comptime remainder = BC_DIM % nelts
        comptime rem_start = BC_DIM - remainder
        for e in range(self.count):
            var entry = self.data + e * BC_DIM
            var acc = SIMD[DType.float32, nelts](0.0)
            for j in range(0, BC_DIM - nelts + 1, nelts):
                var diff = bc.load[width=nelts](j) - entry.load[width=nelts](j)
                acc = fma(diff, diff, acc)
            var d2 = acc.reduce_add()
            if remainder > 0:
                for j in range(rem_start, BC_DIM):
                    var diff = bc[j] - entry[j]
                    d2 = fma(diff, diff, d2)
            # Insert d2 into the ascending k-smallest array.
            if d2 < best[k - 1]:
                var pos = k - 1
                while pos > 0 and best[pos - 1] > d2:
                    best[pos] = best[pos - 1]
                    pos -= 1
                best[pos] = d2
        var total = Float32(0.0)
        for i in range(k):
            total += sqrt(best[i])
        return total / Float32(k)


# ==========================================
# The NS-ES meta-population driver
# ==========================================
# pop holds K policy weight vectors (K * POLICY_DIM), caller-initialized —
# per-agent diversity (e.g. randn * 0.5) is the caller's job, drawn from the
# test's seeded RNG stream. Every rollout anywhere in the run logs its per-tick
# cells into `coverage` and its FINAL tick's cell into `end_cov` (two
# uncheatable metric accumulators: raw visitation vs. distinct reachable
# END-STATES — the latter is what novelty over an end-state BC actually
# optimizes, and the repertoire currency of B-POC-2/4). Returns the total
# number of rollouts consumed, so the baseline can be granted the exact same
# budget. All scratch is allocated ONCE up front (the meta_fit_selfmod
# precedent) — nothing allocates inside the iteration loop.
def ns_es_run(
    pop: UnsafePointer[Float32, MutAnyOrigin],
    task: SandboxTask,
    mut archive: NoveltyArchive,
    mut coverage: CellSet,
    mut end_cov: CellSet,
    K: Int,
    iters: Int,
    N: Int,
    alpha: Float32,
    sigma: Float32,
) -> Int:
    if K <= 0 or iters <= 0 or N <= 0:
        return 0
    comptime pdim = POLICY_DIM
    comptime remainder = pdim % nelts
    comptime rem_start = pdim - remainder

    # ES stripes (param-sized) + rollout stripes (per-sample, x2 antithetic).
    var eps_all = alloc[Float32](N * pdim)
    var pert_all = alloc[Float32](N * pdim)
    var grad = alloc[Float32](pdim)
    var scale = alloc[Float32](pdim)
    var coeff = alloc[Float32](N)
    var grid_all = alloc[Float32](N * SB_CELLS)
    var obs_all = alloc[Float32](N * OBS_DIM)
    var logit_all = alloc[Float32](N * SB_ACTIONS)
    var bc_all = alloc[Float32](N * 2 * BC_DIM)
    var cells_all = alloc[Int64](N * 2 * SB_T)
    var nov = alloc[Float32](K)
    SandboxPolicyMemory.fill_scale(scale, pdim)

    var rollouts = 0

    # Seed the archive with the K initial centers (novelty is defined from
    # iteration 1). Serial; each center's BC is kept at its own offset in
    # bc_all (K <= 2N stripes available) so per-agent novelty is read against
    # the fully seeded archive with no extra rollouts. Each center's own BC is
    # in the archive when its novelty is read — one zero among its k nearest —
    # a uniform bias that cancels in the min-shifted selection rule below.
    for m in range(K):
        sandbox_rollout(
            pop + m * pdim,
            task,
            grid_all,
            obs_all,
            logit_all,
            bc_all + m * BC_DIM,
            cells_all,
            True,
        )
        rollouts += 1
        archive.add(bc_all + m * BC_DIM)
        for t in range(SB_T):
            _ = coverage.insert(cells_all[t])
        _ = end_cov.insert(cells_all[SB_T - 1])
    for m in range(K):
        nov[m] = archive.novelty(bc_all + m * BC_DIM)

    for _ in range(iters):
        # --- 1. Select the agent to step: probability proportional to the
        # min-shifted novelty (invariant to a common additive shrink as the
        # archive densifies; uniform when all agents are equal, e.g. at start).
        var nov_min = nov[0]
        for m in range(1, K):
            if nov[m] < nov_min:
                nov_min = nov[m]
        var total = Float32(0.0)
        for m in range(K):
            total += nov[m] - nov_min + Float32(1e-6)
        var u = Float32(random_float64(0.0, 1.0)) * total
        var msel = K - 1
        var acc_p = Float32(0.0)
        for m in range(K):
            acc_p += nov[m] - nov_min + Float32(1e-6)
            if u <= acc_p:
                msel = m
                break
        var w = pop + msel * pdim

        # --- 2. One antithetic ES step on agent msel; sample fitness = novelty
        # of the perturbed candidate's behaviour characterization.
        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                eps_s[j] = Float32(randn_float64(0.0, 1.0))

        @parameter
        def sample(s: Int):
            var eps_s = eps_all + s * pdim
            var pert = pert_all + s * pdim
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
                pert.store[width=nelts](j, fma(seps, pos, w_vec))
            if remainder > 0:
                for j in range(rem_start, pdim):
                    pert[j] = fma(eps_s[j] * scale[j], sigma, w[j])
            sandbox_rollout(
                pert, task, grid_s, obs_s, logit_s, bc_plus, cells_plus, True
            )
            var f_plus = archive.novelty(bc_plus)

            var neg = SIMD[DType.float32, nelts](-sigma)
            for j in range(0, pdim - nelts + 1, nelts):
                var w_vec = w.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert.store[width=nelts](j, fma(seps, neg, w_vec))
            if remainder > 0:
                for j in range(rem_start, pdim):
                    pert[j] = fma(eps_s[j] * scale[j], -sigma, w[j])
            sandbox_rollout(
                pert, task, grid_s, obs_s, logit_s, bc_minus, cells_minus, True
            )
            var f_minus = archive.novelty(bc_minus)

            coeff[s] = f_plus - f_minus

        parallelize[sample](N)
        rollouts += 2 * N

        # --- 3. Serial: merge every sample's cell log into the coverage sets
        # (the hash sets are never written concurrently), then reduce + step.
        for i in range(N * 2 * SB_T):
            _ = coverage.insert(cells_all[i])
        for s in range(2 * N):
            _ = end_cov.insert(cells_all[s * SB_T + SB_T - 1])

        # Fitness shaping: scale the antithetic coefficients to unit std so
        # the update magnitude is independent of the raw novelty scale (see
        # the module header). Serial, N floats — negligible.
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

        # With unit-std coefficients the 1/(2*sigma) finite-difference scale is
        # already absorbed; the step is a plain shaped average.
        var fac = alpha / Float32(N)
        var fac_vec = SIMD[DType.float32, nelts](fac)
        for j in range(0, pdim - nelts + 1, nelts):
            var w_vec = w.load[width=nelts](j)
            var sg = grad.load[width=nelts](j) * scale.load[width=nelts](j)
            w.store[width=nelts](j, fma(sg, fac_vec, w_vec))
        if remainder > 0:
            for j in range(rem_start, pdim):
                w[j] = fma(grad[j] * scale[j], fac, w[j])

        # --- 4. Evaluate the stepped center, archive its BC, refresh its
        # novelty (the per-generation add of Conti et al.).
        sandbox_rollout(
            w, task, grid_all, obs_all, logit_all, bc_all, cells_all, True
        )
        rollouts += 1
        for t in range(SB_T):
            _ = coverage.insert(cells_all[t])
        _ = end_cov.insert(cells_all[SB_T - 1])
        archive.add(bc_all)
        nov[msel] = archive.novelty(bc_all)

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
    nov.free()
    return rollouts
