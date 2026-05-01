from tensor import Tensor, TensorShape
from collections import List
from memory import UnsafePointer, memset_zero
from sys import sizeof

# ==========================================
# 1. ArcGrid
# ==========================================
struct ArcGrid:
    var data: Tensor[DType.int8]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        self.data = Tensor[DType.int8](TensorShape(rows, cols))

    @always_inline
    fn get(self, r: Int, c: Int) -> SIMD[DType.int8, 1]:
        return self.data[r, c]

    @always_inline
    fn set(inout self, r: Int, c: Int, val: SIMD[DType.int8, 1]):
        self.data[r, c] = val

# ==========================================
# 2. FastWeightBuffer
# ==========================================
struct FastWeightBuffer:
    var weights: Tensor[DType.float32]
    var capacity: Int

    fn __init__(inout self, capacity: Int):
        self.capacity = capacity
        self.weights = Tensor[DType.float32](TensorShape(capacity))

    @always_inline
    fn reset(inout self):
        memset_zero(self.weights.unsafe_ptr(), self.capacity * sizeof[Float32]())

# ==========================================
# 3. HopeNode
# ==========================================
struct HopeNode:
    var slow_weights: Tensor[DType.float32]
    var fast_weights: FastWeightBuffer
    var children: List[UnsafePointer[HopeNode]]

    fn __init__(inout self, slow_dim: Int, fast_dim: Int):
        self.slow_weights = Tensor[DType.float32](TensorShape(slow_dim))
        self.fast_weights = FastWeightBuffer(fast_dim)
        self.children = List[UnsafePointer[HopeNode]]()

    fn add_child(inout self, child_ptr: UnsafePointer[HopeNode]):
        self.children.append(child_ptr)

# ==========================================
# Phase 2.1: The HopeArena Allocator
# ==========================================
struct HopeArena:
    var data: UnsafePointer[UInt8]
    var capacity: Int
    var offset: Int

    fn __init__(inout self, capacity: Int):
        self.capacity = capacity
        self.offset = 0
        self.data = UnsafePointer[UInt8].alloc(self.capacity)
        # Initialize bare-metal memory to zero to avoid garbage state
        memset_zero(self.data, self.capacity)

    # Move semantics: Transfer ownership and nullify the source
    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data
        self.capacity = existing.capacity
        self.offset = existing.offset

        # Critical: Sever the old pointer's connection to the heap
        existing.data = UnsafePointer[UInt8]()
        existing.capacity = 0
        existing.offset = 0

    # Ensure memory is only freed if the pointer is active
    fn __del__(owned self):
        if self.data:
            self.data.free()

    # Fast bump-pointer allocation for HopeNodes
    fn alloc_node[T: AnyType](inout self) -> UnsafePointer[T]:
        var size = sizeof[T]()
        if self.offset + size > self.capacity:
            # Handle OOM gracefully in production, panic for now
            print("Arena Out of Memory!")
            return UnsafePointer[T]()

        var ptr = self.data.offset(self.offset).bitcast[T]()
        self.offset += size
        return ptr

# ==========================================
# Phase 2.2: Formalizing Logic Primitives
# ==========================================
# Define a standard signature for all primitives
alias PrimitiveFunc = fn(inout grid: Tensor[DType.float32], params: UnsafePointer[Float32]) -> None

# Example Primitive: Shift
fn prim_shift(inout grid: Tensor[DType.float32], params: UnsafePointer[Float32]):
    # params[0] = x_shift, params[1] = y_shift
    pass

# Example Primitive: FloodFill
fn prim_flood_fill(inout grid: Tensor[DType.float32], params: UnsafePointer[Float32]):
    pass

# ==========================================
# Phase 2.3: Vectorized Execution Loop
# ==========================================
from math import fma
from sys import simdwidthof

# Configure SIMD width based on target architecture (e.g., AVX-512 = 16 float32s)
alias nelts = simdwidthof[DType.float32]()

fn update_fast_weights(
    fast_weights: UnsafePointer[Float32],
    slow_weights: UnsafePointer[Float32],
    gradients: UnsafePointer[Float32],
    alpha: Float32,
    size: Int
):
    """
    Vectorized loop to update fast weights based on gradients and slow weights.
    """
    # Vectorized loop
    for i in range(0, size - nelts + 1, nelts):
        var fw_vec = fast_weights.load[width=nelts](i)
        var sw_vec = slow_weights.load[width=nelts](i)
        var grad_vec = gradients.load[width=nelts](i)

        # Example FMA update: W_fast = W_fast - alpha * (grad + regularizer * W_slow)
        # fma(a, b, c) -> a * b + c
        # Notice we are simply doing -alpha * grad_vec + fw_vec as the example
        # (ignoring the slow weights in this simple expression for now, but
        # showing how fma works).
        var step = fma(grad_vec, SIMD[DType.float32, nelts](-alpha), fw_vec)

        fast_weights.store[width=nelts](i, step)

    # Handle remainder if size is not a multiple of nelts
    var remainder = size % nelts
    if remainder > 0:
        var start_idx = size - remainder
        for i in range(start_idx, size):
            var fw_val = fast_weights.load(i)
            var sw_val = slow_weights.load(i)
            var grad_val = gradients.load(i)
            fast_weights.store(i, fma(grad_val, -alpha, fw_val))


# A mock struct to represent an ARC task pair (input/output demonstration)
struct ArcTaskPair:
    var input_grid: ArcGrid
    var output_grid: ArcGrid

    fn __init__(inout self, input_grid: ArcGrid, output_grid: ArcGrid):
        self.input_grid = input_grid
        self.output_grid = output_grid

fn forward_with_learning(inout root_node: HopeNode, demonstrations: List[ArcTaskPair], test_input: ArcGrid) -> ArcGrid:
    """
    Phase 2 execution loop: Nested Optimization.
    Routes tasks through the hierarchy and dynamically updates FastWeightBuffer.
    """
    var learning_rate: Float32 = 0.1

    # 1. Learning Phase (Nested Optimization on Demonstrations)
    for i in range(len(demonstrations)):
        var demo = demonstrations[i]

        # Simulate routing and error calculation
        var size = root_node.fast_weights.capacity

        # Allocate mock gradients for demonstration
        var mock_gradients = UnsafePointer[Float32].alloc(size)
        for g in range(size):
            mock_gradients.store(g, 0.5)

        # Update fast weights in-place using SIMD
        update_fast_weights(
            root_node.fast_weights.weights.unsafe_ptr(),
            root_node.slow_weights.unsafe_ptr(),
            mock_gradients,
            learning_rate,
            size
        )
        mock_gradients.free()

    # 2. Evaluation Phase (Inference on Test Input)
    var result_grid = ArcGrid(test_input.rows, test_input.cols)

    # Simulate an output prediction using the updated fast weights
    for r in range(test_input.rows):
        for c in range(test_input.cols):
            result_grid.set(r, c, test_input.get(r, c))

    return result_grid
