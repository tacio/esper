from std.memory import alloc, memset_zero, UnsafePointer
from std.sys import simd_width_of, size_of
from std.math import fma
from std.random import randn_float64

# The real fitness calculator (negative MSE).
from arc_io import calculate_fitness

comptime nelts = simd_width_of[DType.float32]()


# Map a candidate fast-weight vector to a fitness score. For this prototype the
# perturbed weights are treated as the raw output state of the primitive and
# scored directly against the target (an identity surrogate model). A full
# implementation would load them into a HopeNode and run apply_primitive first.
def evaluate_primitives(
    perturbed_weights: UnsafePointer[Float32, MutAnyOrigin],
    target: UnsafePointer[Float32, MutAnyOrigin],
    size: Int,
) -> Float32:
    return calculate_fitness(perturbed_weights, target, size)


# ==========================================
# ESWorkspace: Persistent Optimization State
# ==========================================
# Pre-allocated, reusable scratch so the ES hot loop performs zero per-iteration
# allocation. Owns three raw buffers; freed once in __del__.
struct ESWorkspace(Movable):
    var grad_estimate: UnsafePointer[Float32, MutAnyOrigin]
    var epsilon: UnsafePointer[Float32, MutAnyOrigin]
    var perturbed_weights: UnsafePointer[Float32, MutAnyOrigin]
    var size: Int

    def __init__(out self, size: Int):
        self.size = size
        self.grad_estimate = alloc[Float32](size)
        self.epsilon = alloc[Float32](size)
        self.perturbed_weights = alloc[Float32](size)

    def __del__(deinit self):
        # Consuming moves suppress the source destructor, so each buffer is live
        # exactly once here — free unconditionally.
        self.grad_estimate.free()
        self.epsilon.free()
        self.perturbed_weights.free()


# ==========================================
# Zero-Allocation Evolution Strategy Update
# ==========================================
def evolve_fast_weights(
    fast_weights: UnsafePointer[Float32, MutAnyOrigin],
    mut workspace: ESWorkspace,
    target: UnsafePointer[Float32, MutAnyOrigin],
    N: Int,
    alpha: Float32,
    sigma: Float32,
):
    """Derivative-free Evolution Strategy update for the fast weights.

    Uses real Gaussian noise with antithetic (mirrored) sampling, which both
    halves the estimator variance and centres the fitness (the +/- pair cancels
    any baseline), so no separate fitness normalisation is required:

        grad      = sum_i (F(w + sigma*eps_i) - F(w - sigma*eps_i)) * eps_i
        W_fast   += alpha / (2*N*sigma) * grad

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

        # 2a. perturbed = w + sigma * eps, then evaluate F+.
        var pos_sigma = SIMD[DType.float32, nelts](sigma)
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)
            var eps_vec = workspace.epsilon.load[width=nelts](j)
            workspace.perturbed_weights.store[width=nelts](
                j, fma(eps_vec, pos_sigma, w_vec)
            )
        if remainder > 0:
            for j in range(rem_start, size):
                workspace.perturbed_weights[j] = fma(
                    workspace.epsilon[j], sigma, fast_weights[j]
                )
        var f_plus = evaluate_primitives(
            workspace.perturbed_weights, target, size
        )

        # 2b. perturbed = w - sigma * eps, then evaluate F-.
        var neg_sigma = SIMD[DType.float32, nelts](-sigma)
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)
            var eps_vec = workspace.epsilon.load[width=nelts](j)
            workspace.perturbed_weights.store[width=nelts](
                j, fma(eps_vec, neg_sigma, w_vec)
            )
        if remainder > 0:
            for j in range(rem_start, size):
                workspace.perturbed_weights[j] = fma(
                    workspace.epsilon[j], -sigma, fast_weights[j]
                )
        var f_minus = evaluate_primitives(
            workspace.perturbed_weights, target, size
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

    # 4. Final step: W_fast += alpha / (2*N*sigma) * grad_estimate.
    var update_factor = alpha / (2.0 * Float32(N) * sigma)
    var update_factor_vec = SIMD[DType.float32, nelts](update_factor)
    for j in range(0, size - nelts + 1, nelts):
        var w_vec = fast_weights.load[width=nelts](j)
        var grad_vec = workspace.grad_estimate.load[width=nelts](j)
        fast_weights.store[width=nelts](
            j, fma(grad_vec, update_factor_vec, w_vec)
        )
    if remainder > 0:
        for j in range(rem_start, size):
            fast_weights[j] = fma(
                workspace.grad_estimate[j], update_factor, fast_weights[j]
            )
