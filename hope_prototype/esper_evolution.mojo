from memory import UnsafePointer, memset_zero
from sys import simdwidthof, sizeof
from math import fma
from tensor import Tensor, TensorShape

# Import the actual fitness calculator we built
from arc_io import calculate_fitness

# Define SIMD width based on the hardware target (e.g., AVX-512 = 16 float32s)
alias nelts = simdwidthof[DType.float32]()

# Replaces the mock placeholder to actually evaluate using the imported function
fn evaluate_primitives(perturbed_weights: UnsafePointer[Float32], target: Tensor[DType.float32], size: Int) -> Float32:
    # In a full model, this perturbed pointer would be loaded into the Expert and a full
    # forward pass executed to get the prediction grid. For the prototype, we treat the
    # weights themselves as the raw output state of the primitive to map the concept.

    # We wrap the UnsafePointer in a temporary Tensor view to feed to calculate_fitness
    # This avoids doing a deep copy of the array.
    # Note: In real Mojo, one must be careful about memory lifetimes, but since this
    # is synchronous within the loop, the Tensor view is safe.
    var pred_tensor = Tensor[DType.float32](
        perturbed_weights,
        target.shape()
    )

    return calculate_fitness(pred_tensor, target)

# ==========================================
# ESWorkspace: Persistent Optimization State
# ==========================================
struct ESWorkspace:
    var grad_estimate: UnsafePointer[Float32]
    var perturbed_weights: UnsafePointer[Float32]
    var epsilon: UnsafePointer[Float32]
    var size: Int

    fn __init__(inout self, size: Int):
        self.size = size
        self.grad_estimate = UnsafePointer[Float32].alloc(size)
        self.perturbed_weights = UnsafePointer[Float32].alloc(size)
        self.epsilon = UnsafePointer[Float32].alloc(size)

    # Move semantics to prevent double frees and safely transfer ownership
    fn __moveinit__(inout self, owned existing: Self):
        self.size = existing.size
        self.grad_estimate = existing.grad_estimate
        self.perturbed_weights = existing.perturbed_weights
        self.epsilon = existing.epsilon

        # Nullify old pointers
        existing.size = 0
        existing.grad_estimate = UnsafePointer[Float32]()
        existing.perturbed_weights = UnsafePointer[Float32]()
        existing.epsilon = UnsafePointer[Float32]()

    fn __del__(owned self):
        if self.grad_estimate:
            self.grad_estimate.free()
        if self.perturbed_weights:
            self.perturbed_weights.free()
        if self.epsilon:
            self.epsilon.free()

# ==========================================
# Zero-Allocation Execution Loop
# ==========================================
fn evolve_fast_weights(
    fast_weights: UnsafePointer[Float32],
    inout workspace: ESWorkspace,
    target: Tensor[DType.float32],
    N: Int,
    alpha: Float32,
    sigma: Float32
):
    """
    Derivative-free Evolution Strategy update for Fast Weights.
    W_fast = W_fast + alpha * (1 / (N * sigma)) * sum(F_i * epsilon_i)
    Reuses pre-allocated ESWorkspace to avoid dynamic allocation overhead.
    """
    var size = workspace.size

    # Rapidly zero out the grad_estimate buffer using native byte clearing
    memset_zero(workspace.grad_estimate.bitcast[UInt8](), size * sizeof[Float32]())

    # Generate N perturbed weight vectors
    for i in range(N):
        # 1. Generate Noise and Perturb Weights
        for j in range(0, size - nelts + 1, nelts):
            var w_vec = fast_weights.load[width=nelts](j)

            # Mock RNG for epsilon (in practice, use a proper random normal generator)
            var eps_vec = SIMD[DType.float32, nelts](0.1)
            workspace.epsilon.store[width=nelts](j, eps_vec)

            # Perturbed weights = w + sigma * epsilon
            var perturbed_vec = fma(eps_vec, SIMD[DType.float32, nelts](sigma), w_vec)
            workspace.perturbed_weights.store[width=nelts](j, perturbed_vec)

        # Handle remainder for perturbation
        var remainder = size % nelts
        if remainder > 0:
            var start_idx = size - remainder
            for j in range(start_idx, size):
                var w_val = fast_weights.load(j)
                var eps_val = Float32(0.1) # Mock noise
                workspace.epsilon.store(j, eps_val)
                workspace.perturbed_weights.store(j, fma(eps_val, sigma, w_val))

        # 2. Evaluate Fitness (F_i) using the actual MSE calculate_fitness function
        var F_i = evaluate_primitives(workspace.perturbed_weights, target, size)

        # 3. Accumulate: grad_estimate += F_i * epsilon_i
        for j in range(0, size - nelts + 1, nelts):
            var grad_vec = workspace.grad_estimate.load[width=nelts](j)
            var eps_vec = workspace.epsilon.load[width=nelts](j)

            # grad_vec + F_i * eps_vec
            var new_grad_vec = fma(eps_vec, SIMD[DType.float32, nelts](F_i), grad_vec)
            workspace.grad_estimate.store[width=nelts](j, new_grad_vec)

        # Handle remainder for accumulation
        if remainder > 0:
            var start_idx = size - remainder
            for j in range(start_idx, size):
                var grad_val = workspace.grad_estimate.load(j)
                var eps_val = workspace.epsilon.load(j)
                workspace.grad_estimate.store(j, fma(eps_val, F_i, grad_val))

    # 4. Final Update Step
    # update_factor = alpha / (N * sigma)
    var update_factor = alpha / (Float32(N) * sigma)
    var update_factor_simd = SIMD[DType.float32, nelts](update_factor)

    for j in range(0, size - nelts + 1, nelts):
        var w_vec = fast_weights.load[width=nelts](j)
        var grad_vec = workspace.grad_estimate.load[width=nelts](j)

        # W_fast = W_fast + update_factor * grad_estimate
        var new_w_vec = fma(grad_vec, update_factor_simd, w_vec)
        fast_weights.store[width=nelts](j, new_w_vec)

    # Handle remainder for final update
    var remainder = size % nelts
    if remainder > 0:
        var start_idx = size - remainder
        for j in range(start_idx, size):
            var w_val = fast_weights.load(j)
            var grad_val = workspace.grad_estimate.load(j)
            fast_weights.store(j, fma(grad_val, update_factor, w_val))
