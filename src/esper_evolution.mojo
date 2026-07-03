from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.sys import simd_width_of, size_of
from std.math import fma, exp, log
from std.random import randn_float64
from std.collections import List
from std.algorithm import parallelize

# The learning core is generic over a Memory (what the ES fits) whose associated
# Dom is a Domain (the Example type + metrics). It consumes generic ExamplePair/
# Task containers so the same fitness loop serves any domain. The concrete
# OperatorMemory + ArcGrid are used only by the grid convenience wrapper below.
from memory import Memory, SelfModMemory, ShapeMemory
from memory_es import OperatorMemory, AttnGatherMemory, ATTN_DIM
from memory_composed import (
    GeomColorComposedMemory,
    GeomCountComposedMemory,
    ShapeGeomComposedMemory,
    SHAPEGEOM_DIM,
)
from hope import ExamplePair, Task, ArcTaskPair, HopeNode, ArcGrid

# Default in-context fit schedule, shared by forward_with_learning and the solve
# driver. Tuned so the annealed ES reliably fits the expressible transforms.
comptime FIT_N = 128
comptime FIT_ALPHA0 = Float32(0.1)
comptime FIT_ALPHA1 = Float32(0.003)
comptime FIT_SIGMA0 = Float32(0.5)
comptime FIT_SIGMA1 = Float32(0.01)
comptime FIT_ITERS = 4000
comptime FIT_REG = Float32(0.0001)
# CONSTANT-COMPUTE demo normalization (the few-demo block): the FIT_* schedule
# was tuned at 8-demo synth tasks, and one ES iteration costs ~n_demos fitness
# passes while its SIGNAL also shrinks with fewer demos (weaker contrast on the
# demo average — measured: n=3 flip∘countmap solved 6/10 at the nominal budget,
# 9/10 at constant compute). The composed fit drivers therefore scale the
# iteration count by FIT_DEMO_REF / n_demos: every task gets the SAME total
# demo-evaluations regardless of demo count — a task-independent budgeting
# rule, not a per-task tune. At n_demos = 8 the factor is 1 (all synth proofs
# unchanged).
comptime FIT_DEMO_REF = 8

# Outer (slow-timescale) meta-learning schedule. The slow weights are nudged
# toward each task's fitted fast solution (Reptile averaging); the inner fit
# during meta-training keeps the WIDE FIT_SIGMA0 (it must DISCOVER the family's
# operator from a cold identity prior) but a SHORTER iteration budget — Reptile
# tolerates partial inner optimization.
comptime META_LR = Float32(0.3)
comptime META_ITERS = 12
comptime META_FIT_ITERS = 600

# Eval (fast in-context adaptation) schedule for the meta-prior test: a NARROW,
# short EXPLOIT fit (small sigma0, few iters). This is where a good prior pays
# off — from the meta-learned prior a cheap local fit lands the answer, whereas
# from a cold identity prior the same narrow fit cannot explore far enough to
# find the transform's basin and fails. Same eval schedule for both priors; only
# the prior differs. (Probed: at sigma0=0.12 / 300 iters, an exact-flip_h prior
# fits to 1.0 while a cold prior gets ~0; a wide sigma0>=0.2 lets cold solve too,
# erasing the gap — so the narrowness is the point.)
comptime EVAL_SIGMA0 = Float32(0.12)
comptime EVAL_ITERS = 300

comptime nelts = simd_width_of[DType.float32]()


# ==========================================
# Demonstration-driven operator fitness
# ==========================================
# Score a candidate operator (the fast-weight vector) on the train demos: run it
# on each demo INPUT and measure negative MSE against that demo's OUTPUT, then
# subtract an L2 anchor pulling the fast weights toward the slow prior. This is
# the real learning signal — it replaces the old identity surrogate that scored
# the weights directly against a known target grid (pure memorization).
def fitness[
    M: Memory
](
    weights: UnsafePointer[Float32, MutAnyOrigin],
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[M.Dom.Example]],
    op_output: UnsafePointer[Float32, MutAnyOrigin],
    reg_lambda: Float32,
) -> Float32:
    var num = len(demos)
    if num == 0:
        return 0.0

    var total = Float32(0.0)
    for d in range(num):
        var in_n = M.Dom.capacity(demos[d].input_grid)
        var out_n = M.Dom.capacity(demos[d].output_grid)
        # The memory is same-shape (output dims == input dims). If a demo's
        # output area differs, it is inexpressible here: assign a heavy penalty
        # instead of scoring (calling the metric on mismatched lengths would read
        # out of bounds), steering the ES away from such tasks.
        if in_n != out_n:
            total += Float32(-1.0e9)
            continue
        # Run the memory on the demo input, score against the demo output via the
        # Domain's continuous metric.
        M.apply(weights, demos[d].input_grid, op_output)
        total += M.Dom.distance(op_output, demos[d].output_grid, in_n)
    total = total / Float32(num)

    # L2 anchor: -reg_lambda * ||w - w_slow||^2 / param_dim. The slow weights are
    # the meta-learned prior; this folds HOPE's anchor into the ES objective.
    var pdim = M.param_dim()
    var anchor = Float32(0.0)
    for i in range(pdim):
        var diff = weights[i] - slow[i]
        anchor += diff * diff
    return total - reg_lambda * anchor / Float32(pdim)


# ==========================================
# ESWorkspace: Persistent Optimization State
# ==========================================
# Pre-allocated, reusable scratch so the ES hot loop performs zero per-iteration
# allocation. Two size notions resolve the pixels-vs-params seam: the ES vectors
# are PARAM-sized (the memory's `param_dim`), while the forward scratch is
# EXAMPLE-sized (worst-case flat capacity). Generic over the Memory `M`: the
# per-parameter ES preconditioner `scale` is filled by `M.fill_scale` (so each
# memory owns its own preconditioner — the operator's colour group, a uniform scale
# for an MLP, etc.).
#
# PARALLELISM: the ES inner loop's 2*N fitness evaluations are independent, so they
# run across cores (see evolve_fast_weights). For that, the per-sample scratch must
# be DISJOINT — `eps_all`, `perturbed_all`, `op_output_all` are sized `n_samples` ×
# (param or capacity), so sample s touches only its own stripe and threads never
# share mutable state. `coeff` holds each sample's (F+ - F-). `n_samples` defaults
# to FIT_N (every construction site fits with N = FIT_N), so callers are unchanged.
# All buffers alloc-once / free-once — no hot-loop allocation.
struct ESWorkspace[M: Memory](Movable):
    var grad_estimate: UnsafePointer[Float32, MutAnyOrigin]
    var eps_all: UnsafePointer[Float32, MutAnyOrigin]
    var perturbed_all: UnsafePointer[Float32, MutAnyOrigin]
    var scale: UnsafePointer[Float32, MutAnyOrigin]
    var op_output_all: UnsafePointer[Float32, MutAnyOrigin]
    var coeff: UnsafePointer[Float32, MutAnyOrigin]
    var size: Int
    var grid_capacity: Int
    var n_samples: Int

    def __init__(out self, grid_capacity: Int, n_samples: Int = FIT_N):
        var param_size = Self.M.param_dim()
        self.size = param_size
        self.grid_capacity = grid_capacity
        self.n_samples = n_samples
        self.grad_estimate = alloc[Float32](param_size)
        self.eps_all = alloc[Float32](n_samples * param_size)
        self.perturbed_all = alloc[Float32](n_samples * param_size)
        self.scale = alloc[Float32](param_size)
        self.op_output_all = alloc[Float32](n_samples * grid_capacity)
        self.coeff = alloc[Float32](n_samples)

        # The memory owns its per-parameter ES preconditioner.
        Self.M.fill_scale(self.scale, param_size)

    def __del__(deinit self):
        # Consuming moves suppress the source destructor, so each buffer is live
        # exactly once here — free unconditionally.
        self.grad_estimate.free()
        self.eps_all.free()
        self.perturbed_all.free()
        self.scale.free()
        self.op_output_all.free()
        self.coeff.free()


# ==========================================
# Zero-Allocation Evolution Strategy Update
# ==========================================
def evolve_fast_weights[
    M: Memory
](
    fast_weights: UnsafePointer[Float32, MutAnyOrigin],
    mut workspace: ESWorkspace[M],
    slow_weights: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[M.Dom.Example]],
    N: Int,
    alpha: Float32,
    sigma: Float32,
    reg_lambda: Float32,
):
    """Derivative-free Evolution Strategy update for a memory's fast weights.

    Uses real Gaussian noise with antithetic (mirrored) sampling, which both
    halves the estimator variance and centres the fitness (the +/- pair cancels
    any baseline), so no separate fitness normalisation is required:

        grad      = sum_i (F(w + sigma*eps_i) - F(w - sigma*eps_i)) * eps_i
        W_fast   += alpha / (2*N*sigma) * grad

    Here F is `fitness[M]` over the demonstrations — the candidate memory is run
    on each demo input and scored against its output via the Domain metric.

    The 2*N fitness evaluations are INDEPENDENT, so they run across cores: the N
    epsilons are drawn serially up front (preserving the exact RNG stream, so the
    result is reproducible and bit-identical to a sequential run), the per-sample
    evaluations run in parallel into disjoint workspace stripes, and the gradient is
    reduced serially in sample order (same float summation order). Reuses the
    pre-allocated ESWorkspace — no allocation in the loop.
    """
    var size = workspace.size

    # Guard degenerate configurations that would divide by zero or do nothing.
    # N must not exceed the workspace's per-sample buffers (sized to n_samples).
    if N <= 0 or size <= 0 or sigma == 0.0 or N > workspace.n_samples:
        return

    var remainder = size % nelts
    var rem_start = size - remainder
    var cap = workspace.grid_capacity

    # 1. Serial: draw all N epsilon vectors up front, in the same order a sequential
    #    loop would (sample 0's `size` draws, then sample 1's, ...). RNG is scalar
    #    and stays serial, so the stream — and the result — are reproducible.
    for s in range(N):
        var eps_s = workspace.eps_all + s * size
        for j in range(size):
            eps_s[j] = Float32(randn_float64(0.0, 1.0))

    # Local pointers captured by the parallel closure (the pointer is copied, not the
    # data; each sample indexes its own disjoint stripe, so no shared mutable state).
    var eps_all = workspace.eps_all
    var perturbed_all = workspace.perturbed_all
    var op_output_all = workspace.op_output_all
    var coeff = workspace.coeff
    var scale = workspace.scale

    # 2. Parallel: each sample builds w +/- sigma*(scale ⊙ eps) in its own perturbed
    #    stripe (SIMD/FMA), evaluates F+ and F- in its own forward scratch, and writes
    #    its antithetic coefficient. fitness/apply are pure reads of the (shared)
    #    weights and demos, so this is data-race free.
    @parameter
    def sample(s: Int):
        var eps_s = eps_all + s * size
        var pert_s = perturbed_all + s * size
        var out_s = op_output_all + s * cap

        var pos_sigma = SIMD[DType.float32, nelts](sigma)
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)
            var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](j)
            pert_s.store[width=nelts](j, fma(seps, pos_sigma, w_vec))
        if remainder > 0:
            for j in range(rem_start, size):
                pert_s[j] = fma(eps_s[j] * scale[j], sigma, fast_weights[j])
        var f_plus = fitness[M](pert_s, slow_weights, demos, out_s, reg_lambda)

        var neg_sigma = SIMD[DType.float32, nelts](-sigma)
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)
            var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](j)
            pert_s.store[width=nelts](j, fma(seps, neg_sigma, w_vec))
        if remainder > 0:
            for j in range(rem_start, size):
                pert_s[j] = fma(eps_s[j] * scale[j], -sigma, fast_weights[j])
        var f_minus = fitness[M](pert_s, slow_weights, demos, out_s, reg_lambda)

        coeff[s] = f_plus - f_minus

    parallelize[sample](N)

    # 3. Serial: reduce the antithetic gradient grad += sum_s (F+ - F-)_s * eps_s, in
    #    sample order (identical float summation to the sequential loop).
    memset_zero(workspace.grad_estimate, size)
    for s in range(N):
        var eps_s = workspace.eps_all + s * size
        var coeff_vec = SIMD[DType.float32, nelts](coeff[s])
        for j in range(0, size - nelts + 1, nelts):
            var grad_vec = workspace.grad_estimate.load[width=nelts](j)
            var eps_vec = eps_s.load[width=nelts](j)
            workspace.grad_estimate.store[width=nelts](
                j, fma(eps_vec, coeff_vec, grad_vec)
            )
        if remainder > 0:
            for j in range(rem_start, size):
                workspace.grad_estimate[j] = fma(
                    eps_s[j], coeff[s], workspace.grad_estimate[j]
                )

    # 4. Final step: W_fast += alpha / (2*N*sigma) * (scale ⊙ grad_estimate).
    #    The same per-parameter scale used in the perturbation makes this a
    #    consistent diagonal-preconditioned ES step.
    var update_factor = alpha / (2.0 * Float32(N) * sigma)
    var update_factor_vec = SIMD[DType.float32, nelts](update_factor)
    for j in range(0, size - nelts + 1, nelts):
        var w_vec = fast_weights.load[width=nelts](j)
        var sgrad = workspace.grad_estimate.load[width=nelts](
            j
        ) * workspace.scale.load[width=nelts](j)
        fast_weights.store[width=nelts](j, fma(sgrad, update_factor_vec, w_vec))
    if remainder > 0:
        for j in range(rem_start, size):
            fast_weights[j] = fma(
                workspace.grad_estimate[j] * workspace.scale[j],
                update_factor,
                fast_weights[j],
            )


# ==========================================
# In-context operator fit (the learning phase)
# ==========================================
# Fit the operator's fast weights to the demonstrations with ANNEALED Evolution
# Strategy: alpha and sigma decay geometrically from (alpha0, sigma0) to
# (alpha1, sigma1) over `iters` steps. The wide early sigma is essential — the
# operator's bilinear gather is locally smooth but globally has shallow basins,
# so the search must explore broadly before settling; the shrinking sigma then
# pins the parameters onto the integer values that reproduce a transform exactly.
# This is the shared recipe for every caller (forward_with_learning, the solve
# driver, the tests), so the schedule lives in one place.
def fit_operator[
    M: Memory
](
    fast_weights: UnsafePointer[Float32, MutAnyOrigin],
    mut workspace: ESWorkspace[M],
    slow_weights: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[M.Dom.Example]],
    N: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
):
    if iters <= 0:
        return
    var alpha_rate = log(alpha1 / alpha0) / Float32(iters)
    var sigma_rate = log(sigma1 / sigma0) / Float32(iters)
    for t in range(iters):
        var alpha = alpha0 * exp(alpha_rate * Float32(t))
        var sigma = sigma0 * exp(sigma_rate * Float32(t))
        evolve_fast_weights[M](
            fast_weights,
            workspace,
            slow_weights,
            demos,
            N,
            alpha,
            sigma,
            reg_lambda,
        )


# ==========================================
# Nested-learning forward pass
# ==========================================
# Two-timescale forward pass: the fast weights (the in-context memory) are fit to
# the demonstrations by the annealed ES, anchored to the slow weights (the
# meta-learned prior); then the learned operator is run on the held-out test
# input. This is the grid + OperatorMemory convenience wrapper (used by main and
# the forward-learning test); it constructs an ArcGrid result, so it stays
# concrete (a fully domain-generic forward needs a Domain-provided output
# constructor — deferred to when a second domain lands). The generic core it
# calls (ESWorkspace/fit_operator) is what carries Phase B.
def forward_with_learning(
    node: UnsafePointer[HopeNode, MutAnyOrigin],
    demonstrations: List[ArcTaskPair],
    test_input: ArcGrid,
) raises -> ArcGrid:
    # Worst-case grid area over the demos + test (apply produces same-shape
    # output) sizes the memory-output scratch once.
    var capacity = test_input.size()
    for d in range(len(demonstrations)):
        var in_area = demonstrations[d].input_grid.size()
        if in_area > capacity:
            capacity = in_area
        var out_area = demonstrations[d].output_grid.size()
        if out_area > capacity:
            capacity = out_area

    var workspace = ESWorkspace[OperatorMemory](capacity)
    fit_operator[OperatorMemory](
        node[].fast,
        workspace,
        node[].slow,
        demonstrations,
        FIT_N,
        FIT_ALPHA0,
        FIT_ALPHA1,
        FIT_SIGMA0,
        FIT_SIGMA1,
        FIT_ITERS,
        FIT_REG,
    )

    var result = ArcGrid(test_input.rows, test_input.cols)
    OperatorMemory.apply(node[].fast, test_input, result.data)
    return result^


# ==========================================
# Meta-learning the slow prior (second timescale)
# ==========================================
# Copy a weight slice (used to (re)init the fast weights from the slow prior
# before each inner fit).
def copy_weights(
    dst: UnsafePointer[Float32, MutAnyOrigin],
    src: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
):
    memcpy(dest=dst, src=src, count=size)


# Reptile outer step on the slow prior: slow += meta_lr * (fast - slow). The
# fitted fast weights are the inner loop's solution for one task; moving the
# prior a fraction toward it accumulates, over tasks, the shared structure of the
# task family. Same SIMD main-loop + scalar-remainder shape as the ES update.
def reptile_update(
    slow: UnsafePointer[Float32, MutAnyOrigin],
    fast: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
    meta_lr: Float32,
):
    var remainder = size % nelts
    var rem_start = size - remainder
    var lr_vec = SIMD[DType.float32, nelts](meta_lr)
    for j in range(0, size - nelts + 1, nelts):
        var s = slow.load[width=nelts](j)
        var f = fast.load[width=nelts](j)
        slow.store[width=nelts](j, fma(lr_vec, f - s, s))
    if remainder > 0:
        for j in range(rem_start, size):
            slow[j] = fma(meta_lr, fast[j] - slow[j], slow[j])


# Outer, low-frequency loop that turns `slow` from a fixed identity anchor into a
# META-LEARNED prior (HOPE's two-timescale structure). For each sampled task we
# fit a fresh operator IN-CONTEXT starting from the current prior (so both the
# init and the L2 anchor pull toward `slow`), then nudge `slow` toward that fitted
# solution via Reptile. The inner `fit_operator` is reused unchanged; the caller
# owns the (pre-sized) ESWorkspace, and the single `fast` buffer is allocated once
# here — no per-iteration heap allocation. After meta-training, a fresh task of
# the same family fits faster from `slow` than from a cold identity prior.
def reptile_meta_train[
    M: Memory
](
    slow: UnsafePointer[Float32, MutAnyOrigin],
    meta_tasks: List[Task[M.Dom.Example]],
    mut workspace: ESWorkspace[M],
    n_iters: Int,
    inner_iters: Int,
    meta_lr: Float32,
):
    var num = len(meta_tasks)
    if num == 0 or n_iters <= 0:
        return

    var pdim = M.param_dim()
    var fast = alloc[Float32](pdim)
    for t in range(n_iters):
        # Cycle deterministically through the meta-train tasks.
        ref task = meta_tasks[t % num]
        copy_weights(fast, slow, pdim)
        fit_operator[M](
            fast,
            workspace,
            slow,
            task.train,
            FIT_N,
            FIT_ALPHA0,
            FIT_ALPHA1,
            FIT_SIGMA0,
            FIT_SIGMA1,
            inner_iters,
            FIT_REG,
        )
        reptile_update(slow, fast, pdim, meta_lr)
    fast.free()


# ==========================================
# Self-modifying memory meta-fit (Phase B / B4)
# ==========================================
# Meta-fitness for a candidate SLOW vector of RecolorSelfModMemory: across the
# meta-task family, WRITE the memory's per-colour table from each task's train demos
# (the inner self-write — NOT searched by the ES), then score the learned read on
# that task's held-out test grids (continuous -MSE). Driving this up fits the slow
# read projections (embeddings + temperature) so the one-pass self-write generalises
# across recolor permutations. `state`/`op_output` are caller-provided per-sample
# scratch (disjoint across parallel samples).
def _selfmod_meta_fitness[
    M: SelfModMemory
](
    slow: UnsafePointer[Float32, MutAnyOrigin],
    meta_tasks: List[Task[M.Dom.Example]],
    state: UnsafePointer[Float32, MutAnyOrigin],
    op_output: UnsafePointer[Float32, MutAnyOrigin],
) -> Float32:
    var num = len(meta_tasks)
    if num == 0:
        return 0.0
    var total = Float32(0.0)
    var count = 0
    for t in range(num):
        ref task = meta_tasks[t]
        M.adapt(slow, task.train, state)
        for j in range(len(task.test)):
            ref tp = task.test[j]
            var n = M.Dom.capacity(tp.input_grid)
            M.apply(slow, state, tp.input_grid, op_output)
            total += M.Dom.distance(op_output, tp.output_grid, n)
            count += 1
    if count == 0:
        return 0.0
    return total / Float32(count)


# Meta-learn the slow projections of a SelfModMemory by the SAME antithetic, parallel,
# annealed ES used for the operator — but the ES dimension is just the small slow
# vector (the fast state is written by `adapt`, never searched), which is exactly why
# this is tractable where fitting fast weights was not. Bit-identical-style determinism
# (serial epsilons in fixed order, serial reduction); parallel only the independent
# 2*N meta-fitness evals (each in its own state/op scratch). Generic over the memory M.
def meta_fit_selfmod[
    M: SelfModMemory
](
    slow: UnsafePointer[Float32, MutAnyOrigin],
    meta_tasks: List[Task[M.Dom.Example]],
    grid_capacity: Int,
    N: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
):
    if iters <= 0 or N <= 0:
        return
    var sdim = M.slow_dim()
    var state_dim = M.state_dim()
    var remainder = sdim % nelts
    var rem_start = sdim - remainder

    var eps_all = alloc[Float32](N * sdim)
    var pert_all = alloc[Float32](N * sdim)
    var state_all = alloc[Float32](N * state_dim)
    var op_all = alloc[Float32](N * grid_capacity)
    var coeff = alloc[Float32](N)
    var grad = alloc[Float32](sdim)
    var scale = alloc[Float32](sdim)
    M.fill_scale(scale, sdim)

    var alpha_rate = log(alpha1 / alpha0) / Float32(iters)
    var sigma_rate = log(sigma1 / sigma0) / Float32(iters)

    for it in range(iters):
        var alpha = alpha0 * exp(alpha_rate * Float32(it))
        var sigma = sigma0 * exp(sigma_rate * Float32(it))

        # Serial: draw all N epsilon vectors (fixed RNG order -> reproducible).
        for s in range(N):
            var eps_s = eps_all + s * sdim
            for j in range(sdim):
                eps_s[j] = Float32(randn_float64(0.0, 1.0))

        # Parallel: antithetic meta-fitness in disjoint per-sample scratch.
        @parameter
        def sample(s: Int):
            var eps_s = eps_all + s * sdim
            var pert = pert_all + s * sdim
            var st = state_all + s * state_dim
            var op = op_all + s * grid_capacity

            var pos = SIMD[DType.float32, nelts](sigma)
            for j in range(0, sdim - nelts + 1, nelts):
                var w_vec = slow.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert.store[width=nelts](j, fma(seps, pos, w_vec))
            if remainder > 0:
                for j in range(rem_start, sdim):
                    pert[j] = fma(eps_s[j] * scale[j], sigma, slow[j])
            var f_plus = _selfmod_meta_fitness[M](pert, meta_tasks, st, op)

            var neg = SIMD[DType.float32, nelts](-sigma)
            for j in range(0, sdim - nelts + 1, nelts):
                var w_vec = slow.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert.store[width=nelts](j, fma(seps, neg, w_vec))
            if remainder > 0:
                for j in range(rem_start, sdim):
                    pert[j] = fma(eps_s[j] * scale[j], -sigma, slow[j])
            var f_minus = _selfmod_meta_fitness[M](pert, meta_tasks, st, op)

            coeff[s] = f_plus - f_minus

        parallelize[sample](N)

        # Serial reduce + step (same shape as evolve_fast_weights).
        memset_zero(grad, sdim)
        for s in range(N):
            var eps_s = eps_all + s * sdim
            var c_vec = SIMD[DType.float32, nelts](coeff[s])
            for j in range(0, sdim - nelts + 1, nelts):
                var g = grad.load[width=nelts](j)
                grad.store[width=nelts](
                    j, fma(eps_s.load[width=nelts](j), c_vec, g)
                )
            if remainder > 0:
                for j in range(rem_start, sdim):
                    grad[j] = fma(eps_s[j], coeff[s], grad[j])

        var fac = alpha / (2.0 * Float32(N) * sigma)
        var fac_vec = SIMD[DType.float32, nelts](fac)
        for j in range(0, sdim - nelts + 1, nelts):
            var w_vec = slow.load[width=nelts](j)
            var sg = grad.load[width=nelts](j) * scale.load[width=nelts](j)
            slow.store[width=nelts](j, fma(sg, fac_vec, w_vec))
        if remainder > 0:
            for j in range(rem_start, sdim):
                slow[j] = fma(grad[j] * scale[j], fac, slow[j])

    eps_all.free()
    pert_all.free()
    state_all.free()
    op_all.free()
    coeff.free()
    grad.free()
    scale.free()


# ==========================================
# Composed geometry × colour fit (ARC-AGI-2 block 5)
# ==========================================
# The per-task in-context fit for GeomColorComposedMemory — the energy-
# composition pipeline (see the struct comment in memory.mojo): each module is
# fit on a signal INVARIANT to the other's factor, then composed.
#
#   1. COLOUR: write the table V closed-form from the demos' count signatures
#      (geometry-invariant — one forward pass, no search).
#   2. PRE-MAP: rebuild the demo list with inputs mapped through V
#      (colour-then-gather; V is cellwise and the gather positional, so they
#      commute — this puts the geometry search on the exact, proven B3 fitness
#      landscape with no colour-table cliff).
#   3. GEOMETRY: the standard annealed fit_operator[AttnGatherMemory] over the
#      state's 7 attention slots (the state layout puts them first, so `state`
#      itself is the attention weight vector), anchored to the identity seed.
#
# The pre-map list is built once per task fit (not per ES iteration), so the
# hot loop stays allocation-free. `n_fit` is passed to the workspace so the
# parallel ES stripes are sized to the actual sample count.
def fit_geomcolor(
    state: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    n_fit: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
) raises:
    # 1. Colour self-write (fills state[GEOMCOLOR_V_OFF:]).
    GeomColorComposedMemory.write_color(state, demos)

    # 2. Pre-map demo inputs through the written V.
    var mapped = List[ArcTaskPair]()
    for d in range(len(demos)):
        var min_g = ArcGrid(demos[d].input_grid.rows, demos[d].input_grid.cols)
        for k in range(min_g.rows * min_g.cols):
            min_g.data[k] = GeomColorComposedMemory._v_lookup(
                state, demos[d].input_grid.data[k]
            )
        var mout = ArcGrid(demos[d].output_grid.rows, demos[d].output_grid.cols)
        memcpy(
            dest=mout.data,
            src=demos[d].output_grid.data,
            count=mout.rows * mout.cols,
        )
        mapped.append(ArcTaskPair(min_g^, mout^))

    # 3. Proven attention-gather fit on the geometry slots, identity anchor.
    # Constant-compute normalization: iters scaled by FIT_DEMO_REF / n_demos
    # (see the FIT_DEMO_REF comment) — same total demo-evaluations per task.
    var n_demos = len(demos)
    if n_demos < 1:
        n_demos = 1
    var iters_scaled = iters * FIT_DEMO_REF // n_demos
    var slow = alloc[Float32](ATTN_DIM)
    AttnGatherMemory.seed(slow)
    var ws = ESWorkspace[AttnGatherMemory](grid_capacity, n_fit)
    fit_operator[AttnGatherMemory](
        state,
        ws,
        slow,
        mapped,
        n_fit,
        alpha0,
        alpha1,
        sigma0,
        sigma1,
        iters_scaled,
        reg_lambda,
    )
    slow.free()


# ==========================================
# Composed geometry × count-rule fit (the content×geometry block)
# ==========================================
# The per-task in-context fit for GeomCountComposedMemory — fit_geomcolor's
# recipe one level up (see the struct comment in memory_composed.mojo):
#
#   1. CONTENT: write (P, V) closed-form from the demos' histogram signatures
#      (geometry-invariant — histograms ignore position; one pass, no search).
#   2. PRE-MAP: rebuild the demo list with inputs mapped through the inferred
#      content rule (content-then-gather; the toroidal count rule and the
#      permutation geometry COMMUTE, so the geometry search runs on the exact,
#      proven B3 fitness landscape of integer-valued grids).
#   3. GEOMETRY: the standard annealed fit_operator[AttnGatherMemory] over the
#      state's 7 attention slots, anchored to the identity seed.
#
# The pre-map list is built once per task fit (not per ES iteration), so the
# hot loop stays allocation-free.
def fit_geomcount(
    state: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    n_fit: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
) raises:
    # 1. Content self-write (fills the P-selection + V slots).
    GeomCountComposedMemory.write_content(state, demos)
    var p = GeomCountComposedMemory._selected_p(state)

    # 2. Pre-map demo inputs through the written content rule.
    var mapped = List[ArcTaskPair]()
    for d in range(len(demos)):
        var min_g = ArcGrid(demos[d].input_grid.rows, demos[d].input_grid.cols)
        for r in range(min_g.rows):
            for c in range(min_g.cols):
                min_g.data[
                    r * min_g.cols + c
                ] = GeomCountComposedMemory.content_at(
                    state, demos[d].input_grid, r, c, p
                )
        var mout = ArcGrid(demos[d].output_grid.rows, demos[d].output_grid.cols)
        memcpy(
            dest=mout.data,
            src=demos[d].output_grid.data,
            count=mout.rows * mout.cols,
        )
        mapped.append(ArcTaskPair(min_g^, mout^))

    # 3. Proven attention-gather fit on the geometry slots, identity anchor.
    # Constant-compute normalization: iters scaled by FIT_DEMO_REF / n_demos
    # (see the FIT_DEMO_REF comment) — same total demo-evaluations per task.
    var n_demos = len(demos)
    if n_demos < 1:
        n_demos = 1
    var iters_scaled = iters * FIT_DEMO_REF // n_demos
    var slow = alloc[Float32](ATTN_DIM)
    AttnGatherMemory.seed(slow)
    var ws = ESWorkspace[AttnGatherMemory](grid_capacity, n_fit)
    fit_operator[AttnGatherMemory](
        state,
        ws,
        slow,
        mapped,
        n_fit,
        alpha0,
        alpha1,
        sigma0,
        sigma1,
        iters_scaled,
        reg_lambda,
    )
    slow.free()


# ==========================================
# Shape-changing memory fit (Vision A / Next #1 — the output-size seam)
# ==========================================
# The shape-aware analogue of `fitness[M]` for a ShapeMemory. The key departure:
# the memory PREDICTS its own output shape from the input (`out_rows`/`out_cols`,
# read off the written shape rule) and writes that many cells, which are scored
# against the demo output at the OUTPUT area. If the predicted shape's area does
# not match the demo output's, the demo is inexpressible under the current shape
# rule — a heavy penalty (the honest analogue of the same-shape guard in
# `fitness`, and it prevents the OOB a mismatched compare would cause). The L2
# anchor is over the full param vector; the frozen shape-rule slots contribute a
# constant that cancels in the antithetic F+ - F- (they never move), so only the
# content (attention) slots feel it.
def fitness_shape[
    M: ShapeMemory
](
    state: UnsafePointer[Float32, MutAnyOrigin],
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[M.Dom.Example]],
    op_output: UnsafePointer[Float32, MutAnyOrigin],
    reg_lambda: Float32,
) -> Float32:
    var num = len(demos)
    if num == 0:
        return 0.0

    var total = Float32(0.0)
    for d in range(num):
        var pr = M.out_rows(state, demos[d].input_grid)
        var pc = M.out_cols(state, demos[d].input_grid)
        var out_n = M.Dom.capacity(demos[d].output_grid)
        # Predicted output shape wrong for this demo ⇒ inexpressible here.
        if pr * pc != out_n:
            total += Float32(-1.0e9)
            continue
        M.apply(state, demos[d].input_grid, pr, pc, op_output)
        total += M.Dom.distance(op_output, demos[d].output_grid, out_n)
    total = total / Float32(num)

    var pdim = M.param_dim()
    var anchor = Float32(0.0)
    for i in range(pdim):
        var diff = state[i] - slow[i]
        anchor += diff * diff
    return total - reg_lambda * anchor / Float32(pdim)


# Annealed antithetic ES over a ShapeMemory's params — the shape-aware sibling
# of fit_operator/evolve_fast_weights, self-contained (its own scratch, like
# meta_fit_selfmod) since it drives `fitness_shape` and sizes the forward
# scratch to the OUTPUT capacity. Only the scaled params move: the shape rule is
# written by `M.write` before the call and frozen (its `fill_scale` is 0), so
# this search touches only the content (attention) slots. Determinism mirrors
# evolve_fast_weights: serial epsilons in fixed order, parallel independent
# antithetic evals in disjoint per-sample scratch, serial reduction in sample
# order.
def fit_shape[
    M: ShapeMemory
](
    state: UnsafePointer[Float32, MutAnyOrigin],
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ExamplePair[M.Dom.Example]],
    grid_capacity: Int,
    N: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
):
    if iters <= 0 or N <= 0:
        return
    var pdim = M.param_dim()
    var remainder = pdim % nelts
    var rem_start = pdim - remainder

    var eps_all = alloc[Float32](N * pdim)
    var pert_all = alloc[Float32](N * pdim)
    var op_all = alloc[Float32](N * grid_capacity)
    var coeff = alloc[Float32](N)
    var grad = alloc[Float32](pdim)
    var scale = alloc[Float32](pdim)
    M.fill_scale(scale, pdim)

    var alpha_rate = log(alpha1 / alpha0) / Float32(iters)
    var sigma_rate = log(sigma1 / sigma0) / Float32(iters)

    for it in range(iters):
        var alpha = alpha0 * exp(alpha_rate * Float32(it))
        var sigma = sigma0 * exp(sigma_rate * Float32(it))

        # Serial: draw all N epsilon vectors (fixed RNG order -> reproducible).
        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                eps_s[j] = Float32(randn_float64(0.0, 1.0))

        # Parallel: antithetic fitness in disjoint per-sample scratch.
        @parameter
        def sample(s: Int):
            var eps_s = eps_all + s * pdim
            var pert = pert_all + s * pdim
            var op = op_all + s * grid_capacity

            var pos = SIMD[DType.float32, nelts](sigma)
            for j in range(0, pdim - nelts + 1, nelts):
                var w_vec = state.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert.store[width=nelts](j, fma(seps, pos, w_vec))
            if remainder > 0:
                for j in range(rem_start, pdim):
                    pert[j] = fma(eps_s[j] * scale[j], sigma, state[j])
            var f_plus = fitness_shape[M](pert, slow, demos, op, reg_lambda)

            var neg = SIMD[DType.float32, nelts](-sigma)
            for j in range(0, pdim - nelts + 1, nelts):
                var w_vec = state.load[width=nelts](j)
                var seps = eps_s.load[width=nelts](j) * scale.load[width=nelts](
                    j
                )
                pert.store[width=nelts](j, fma(seps, neg, w_vec))
            if remainder > 0:
                for j in range(rem_start, pdim):
                    pert[j] = fma(eps_s[j] * scale[j], -sigma, state[j])
            var f_minus = fitness_shape[M](pert, slow, demos, op, reg_lambda)

            coeff[s] = f_plus - f_minus

        parallelize[sample](N)

        # Serial reduce + step (same shape as evolve_fast_weights).
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

        var fac = alpha / (2.0 * Float32(N) * sigma)
        var fac_vec = SIMD[DType.float32, nelts](fac)
        for j in range(0, pdim - nelts + 1, nelts):
            var w_vec = state.load[width=nelts](j)
            var sg = grad.load[width=nelts](j) * scale.load[width=nelts](j)
            state.store[width=nelts](j, fma(sg, fac_vec, w_vec))
        if remainder > 0:
            for j in range(rem_start, pdim):
                state[j] = fma(grad[j] * scale[j], fac, state[j])

    eps_all.free()
    pert_all.free()
    op_all.free()
    coeff.free()
    grad.free()
    scale.free()


# The per-task in-context fit for ShapeGeomComposedMemory (the shape-change seam
# driver, parallel to fit_geomcolor): write the shape rule closed-form from the
# demos, then run the annealed shape-aware ES on the geometry slots. Constant-
# compute normalization (iters × FIT_DEMO_REF / n_demos) matches the other
# composed drivers — same total demo-evaluations per task. The written shape-
# rule slots are frozen by fill_scale, so the ES moves only the 7 attention
# params, on the proven B3 landscape but now reading the OUTPUT grid.
def fit_shape_geom(
    state: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    n_fit: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
) raises:
    ShapeGeomComposedMemory.write(state, demos)

    var n_demos = len(demos)
    if n_demos < 1:
        n_demos = 1
    var iters_scaled = iters * FIT_DEMO_REF // n_demos

    var slow = alloc[Float32](SHAPEGEOM_DIM)
    ShapeGeomComposedMemory.seed(slow)
    fit_shape[ShapeGeomComposedMemory](
        state,
        slow,
        demos,
        grid_capacity,
        n_fit,
        alpha0,
        alpha1,
        sigma0,
        sigma1,
        iters_scaled,
        reg_lambda,
    )
    slow.free()
