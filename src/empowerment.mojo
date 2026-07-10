# ==========================================================================
# Exact empowerment (Vision B / B-POC-2.5): the second, learned-part-free
# intrinsic signal, compared against archive novelty (RESEARCH-NOTES rung 2's
# deferred half). Empowerment of a state = the capacity of the n-step action
# channel (Klyubin et al. 2005). The sandbox is FULLY deterministic, so the
# Blahut–Arimoto machinery collapses: a deterministic channel's capacity is
# exactly log2(#distinct states reachable in n steps). Exact empowerment is
# therefore an exhaustive DFS over all 6^n action sequences plus distinct-
# state counting — no iteration, no approximation, no learned parts, and (in
# deliberate contrast to novelty) NO archive: the signal is stateless and
# stationary.
#
# Cost honesty: one evaluation spends sum_{d=1..n} 6^d world ticks on top of
# the 64-tick rollout it scores (n=3: 258, n=4: 1554), while a novelty
# evaluation is nearly free. Per the B-POC-2.5 decision the comparison budget
# stays denominated in ROLLOUTS (as B-POC-1/2) and the enumeration ticks are
# COUNTED AND PRINTED as an uncharged-cost caveat, not charged.
#
# emp_es_run is the me_emitter_run skeleton with the scalar swapped — the
# only difference is the fitness (empowerment of the perturbed candidate's
# final state instead of archive novelty), so the head-to-head isolates the
# signal. Everything else is kept for comparability: fixed alpha/sigma,
# unit-std fitness shaping, uniform-elite re-seeding, full harvest into the
# EliteMap.
# ==========================================================================
from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.sys import simd_width_of
from std.math import fma, sqrt, log
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
    sandbox_step,
    sandbox_rollout,
    sandbox_rollout_state,
)
from map_elites import EliteMap, settle_tick

comptime nelts = simd_width_of[DType.float32]()

# Hard cap on the enumeration horizon (buffer sizing); the runtime horizon is
# a parameter <= this. 6^6 leaves would already be 46k evaluations of
# sandbox_step per fitness call — far past the useful cost range.
comptime EMP_MAX_H = 6
# Distinct-state set capacity per evaluation: open addressing, 50% max load.
# 6^4 = 1296 leaves fits; 6^5 would need the next power of two.
comptime EMP_SET_CAP = 4096
comptime EMP_SET_MASK = EMP_SET_CAP - 1
# Sentinel for the per-evaluation seen-set: state_hash never returns 0.
comptime EMP_EMPTY = Int64(0)
# Per-sample tick counters are strided one cache line apart so the N parallel
# samples don't false-share while hammering their counters inside the DFS.
comptime EMP_TICK_STRIDE = 8


# FNV-1a-style fold of the FULL world state (grid colours + avatar r/c +
# brush). Grid values are small non-negative ints stored as Float32. Never
# returns 0 (the seen-set sentinel).
def state_hash(
    grid: UnsafePointer[Float32, MutAnyOrigin], r: Int, c: Int, brush: Int
) -> Int64:
    var h = Int64(-3750763034362895579)  # 0xCBF29CE484222325
    comptime prime = Int64(0x100000001B3)
    for i in range(SB_CELLS):
        h = (h ^ Int64(Int(grid[i]))) * prime
    h = (h ^ Int64(r)) * prime
    h = (h ^ Int64(c)) * prime
    h = (h ^ Int64(brush)) * prime
    if h == 0:
        h = 1
    return h


# Exact n-step empowerment of one state: iterative DFS over all SB_ACTIONS^n
# action sequences (level d+1's grid is copied from level d, then stepped),
# each leaf's full-state hash inserted into the caller's seen-set stripe;
# returns log2(distinct leaves). `lvl_grids` must hold (horizon+1)*SB_CELLS
# floats and `seen` EMP_SET_CAP Int64s — both caller-provided, RE-ENTRANT
# (each parallel ES sample owns its stripes). `ticks` accumulates the number
# of sandbox_step calls spent (the uncharged-cost caveat).
def empowerment(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    r: Int,
    c: Int,
    brush: Int,
    task: SandboxTask,
    horizon: Int,
    lvl_grids: UnsafePointer[Float32, MutAnyOrigin],
    seen: UnsafePointer[Int64, MutAnyOrigin],
    ticks: UnsafePointer[Int, MutAnyOrigin],
) -> Float32:
    if horizon <= 0 or horizon > EMP_MAX_H:
        return 0.0
    for i in range(EMP_SET_CAP):
        seen[i] = EMP_EMPTY

    var rs = InlineArray[Int, EMP_MAX_H + 1](fill=0)
    var cs = InlineArray[Int, EMP_MAX_H + 1](fill=0)
    var bs = InlineArray[Int, EMP_MAX_H + 1](fill=0)
    var act = InlineArray[Int, EMP_MAX_H + 1](fill=0)
    memcpy(dest=lvl_grids, src=grid, count=SB_CELLS)
    rs[0] = r
    cs[0] = c
    bs[0] = brush

    var count = 0
    var depth = 0
    act[0] = 0
    while True:
        if depth == horizon:
            # Leaf: count the state if unseen (open addressing, sentinel 0;
            # max load 1296/4096 keeps probing short).
            var h_key = state_hash(
                lvl_grids + depth * SB_CELLS, rs[depth], cs[depth], bs[depth]
            )
            var h = Int((h_key * 0x9E3779B97F4A7C15) & Int64(EMP_SET_MASK))
            while True:
                if seen[h] == h_key:
                    break
                if seen[h] == EMP_EMPTY:
                    seen[h] = h_key
                    count += 1
                    break
                h = (h + 1) & EMP_SET_MASK
            # Ascend to the deepest level with an action left to try.
            var exhausted = False
            while True:
                if depth == 0:
                    exhausted = True
                    break
                depth -= 1
                act[depth] += 1
                if act[depth] < SB_ACTIONS:
                    break
            if exhausted:
                break
        else:
            # Descend: child state = parent state stepped by act[depth].
            var child = lvl_grids + (depth + 1) * SB_CELLS
            memcpy(dest=child, src=lvl_grids + depth * SB_CELLS, count=SB_CELLS)
            var nr = rs[depth]
            var nc = cs[depth]
            var nb = bs[depth]
            sandbox_step(
                child, nr, nc, nb, task.grav_dir, task.grav_rate, act[depth]
            )
            ticks[0] += 1
            depth += 1
            rs[depth] = nr
            cs[depth] = nc
            bs[depth] = nb
            act[depth] = 0

    # Capacity of a deterministic channel: log2 of the distinct-output count.
    return Float32(log(Float64(count)) / log(Float64(2.0)))


# ==========================================
# The empowerment-ES emitter driver
# ==========================================
# me_emitter_run (map_elites.mojo) with the scalar swapped: sample fitness =
# exact empowerment of the perturbed candidate's rollout's FINAL state. No
# NoveltyArchive — the signal needs no memory of the run. Same harvest-
# everything discipline (pert stripe N x 2 x pdim so both antithetic weight
# vectors are live at insert time), same leftover-budget filler, so the arm
# consumes `budget` rollouts EXACTLY. Total enumeration ticks are summed into
# `enum_ticks` for the printed uncharged-cost caveat. Returns rollouts.
def emp_es_run(
    mut emap: EliteMap,
    task: SandboxTask,
    mut coverage: CellSet,
    mut end_cov: CellSet,
    budget: Int,
    reseed_every: Int,
    N: Int,
    alpha: Float32,
    sigma: Float32,
    init_scale: Float32,
    horizon: Int,
    mut enum_ticks: Int,
) -> Int:
    if budget <= 0 or N <= 0 or reseed_every <= 0:
        return 0
    comptime pdim = POLICY_DIM
    comptime remainder = pdim % nelts
    comptime rem_start = pdim - remainder
    var lvl_size = (horizon + 1) * SB_CELLS

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
    var lvl_all = alloc[Float32](N * lvl_size)
    var seen_all = alloc[Int64](N * EMP_SET_CAP)
    var tick_all = alloc[Int](N * EMP_TICK_STRIDE)
    memset_zero(tick_all, N * EMP_TICK_STRIDE)
    SandboxPolicyMemory.fill_scale(scale, pdim)

    var rollouts = 0

    # --- First seed: a fresh random policy opens the map.
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

    var iter = 0
    while rollouts + 2 * N + 1 <= budget:
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
            var lvl_s = lvl_all + s * lvl_size
            var seen_s = seen_all + s * EMP_SET_CAP
            var tick_s = tick_all + s * EMP_TICK_STRIDE
            var fr = 0
            var fc = 0
            var fb = 0

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
            sandbox_rollout_state(
                pert_plus,
                task,
                grid_s,
                obs_s,
                logit_s,
                bc_plus,
                cells_plus,
                True,
                fr,
                fc,
                fb,
            )
            var f_plus = empowerment(
                grid_s, fr, fc, fb, task, horizon, lvl_s, seen_s, tick_s
            )

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
            sandbox_rollout_state(
                pert_minus,
                task,
                grid_s,
                obs_s,
                logit_s,
                bc_minus,
                cells_minus,
                True,
                fr,
                fc,
                fb,
            )
            var f_minus = empowerment(
                grid_s, fr, fc, fb, task, horizon, lvl_s, seen_s, tick_s
            )

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
        # ns_es_run/me_emitter_run skeleton verbatim).
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

        # --- Center eval + harvest (no archive to feed here).
        sandbox_rollout(
            w, task, grid_all, obs_all, logit_all, bc_all, cells_all, True
        )
        rollouts += 1
        for t in range(SB_T):
            _ = coverage.insert(cells_all[t])
        _ = end_cov.insert(cells_all[SB_T - 1])
        _ = emap.insert(cells_all[SB_T - 1], settle_tick(cells_all), w, bc_all)

    # --- Leftover budget (< 2N+1): fresh random-policy rollouts so the arm
    # consumes the budget exactly.
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

    for s in range(N):
        enum_ticks += tick_all[s * EMP_TICK_STRIDE]

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
    lvl_all.free()
    seen_all.free()
    tick_all.free()
    return rollouts
