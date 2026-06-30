from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.sys import simd_width_of, size_of
from std.math import fma, exp, log
from std.random import randn_float64
from std.collections import List

# The learning core is generic over a Memory (what the ES fits) whose associated
# Dom is a Domain (the Example type + metrics). It consumes generic ExamplePair/
# Task containers so the same fitness loop serves any domain. The concrete
# OperatorMemory + ArcGrid are used only by the grid convenience wrapper below.
from memory import Memory, OperatorMemory
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
# are PARAM-sized (the memory's `param_dim`), while `op_output` is EXAMPLE-sized
# (worst-case flat capacity) to hold each memory forward pass. Generic over the
# Memory `M`: the per-parameter ES preconditioner `scale` is filled by
# `M.fill_scale` (so each memory owns its own preconditioner — the operator's
# colour group, a uniform scale for an MLP, etc.). All buffers freed once in
# __del__.
struct ESWorkspace[M: Memory](Movable):
    var grad_estimate: UnsafePointer[Float32, MutAnyOrigin]
    var epsilon: UnsafePointer[Float32, MutAnyOrigin]
    var perturbed_weights: UnsafePointer[Float32, MutAnyOrigin]
    var scale: UnsafePointer[Float32, MutAnyOrigin]
    var op_output: UnsafePointer[Float32, MutAnyOrigin]
    var size: Int
    var grid_capacity: Int

    def __init__(out self, grid_capacity: Int):
        var param_size = Self.M.param_dim()
        self.size = param_size
        self.grid_capacity = grid_capacity
        self.grad_estimate = alloc[Float32](param_size)
        self.epsilon = alloc[Float32](param_size)
        self.perturbed_weights = alloc[Float32](param_size)
        self.scale = alloc[Float32](param_size)
        self.op_output = alloc[Float32](grid_capacity)

        # The memory owns its per-parameter ES preconditioner.
        Self.M.fill_scale(self.scale, param_size)

    def __del__(deinit self):
        # Consuming moves suppress the source destructor, so each buffer is live
        # exactly once here — free unconditionally.
        self.grad_estimate.free()
        self.epsilon.free()
        self.perturbed_weights.free()
        self.scale.free()
        self.op_output.free()


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
    Reuses the pre-allocated ESWorkspace — no allocation in the loop.
    """
    var size = workspace.size

    # Guard degenerate configurations that would divide by zero or do nothing.
    if N <= 0 or size <= 0 or sigma == 0.0:
        return

    var remainder = size % nelts
    var rem_start = size - remainder

    # Zero the gradient accumulator.
    memset_zero(workspace.grad_estimate, size)

    for _ in range(N):
        # 1. Draw fresh Gaussian noise. RNG is inherently scalar; the arithmetic
        #    that follows stays vectorized + FMA.
        for j in range(size):
            workspace.epsilon[j] = Float32(randn_float64(0.0, 1.0))

        # 2a. perturbed = w + sigma * (scale ⊙ eps), then evaluate F+.
        var pos_sigma = SIMD[DType.float32, nelts](sigma)
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)
            var seps = workspace.epsilon.load[width=nelts](
                j
            ) * workspace.scale.load[width=nelts](j)
            workspace.perturbed_weights.store[width=nelts](
                j, fma(seps, pos_sigma, w_vec)
            )
        if remainder > 0:
            for j in range(rem_start, size):
                workspace.perturbed_weights[j] = fma(
                    workspace.epsilon[j] * workspace.scale[j],
                    sigma,
                    fast_weights[j],
                )
        var f_plus = fitness[M](
            workspace.perturbed_weights,
            slow_weights,
            demos,
            workspace.op_output,
            reg_lambda,
        )

        # 2b. perturbed = w - sigma * (scale ⊙ eps), then evaluate F-.
        var neg_sigma = SIMD[DType.float32, nelts](-sigma)
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)
            var seps = workspace.epsilon.load[width=nelts](
                j
            ) * workspace.scale.load[width=nelts](j)
            workspace.perturbed_weights.store[width=nelts](
                j, fma(seps, neg_sigma, w_vec)
            )
        if remainder > 0:
            for j in range(rem_start, size):
                workspace.perturbed_weights[j] = fma(
                    workspace.epsilon[j] * workspace.scale[j],
                    -sigma,
                    fast_weights[j],
                )
        var f_minus = fitness[M](
            workspace.perturbed_weights,
            slow_weights,
            demos,
            workspace.op_output,
            reg_lambda,
        )

        # 3. Accumulate the antithetic gradient: grad += (F+ - F-) * eps.
        var coeff = f_plus - f_minus
        var coeff_vec = SIMD[DType.float32, nelts](coeff)
        for j in range(0, size - nelts + 1, nelts):
            var grad_vec = workspace.grad_estimate.load[width=nelts](j)
            var eps_vec = workspace.epsilon.load[width=nelts](j)
            workspace.grad_estimate.store[width=nelts](
                j, fma(eps_vec, coeff_vec, grad_vec)
            )
        if remainder > 0:
            for j in range(rem_start, size):
                workspace.grad_estimate[j] = fma(
                    workspace.epsilon[j], coeff, workspace.grad_estimate[j]
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
