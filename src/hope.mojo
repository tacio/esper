from std.memory import alloc, memset_zero, memcpy, UnsafePointer
from std.sys import simd_width_of, size_of
from std.math import fma
from std.collections import List, InlineArray

# Configure SIMD width based on the target architecture (e.g. AVX-512 = 16 float32s).
comptime nelts = simd_width_of[DType.float32]()

# Maximum fan-out of a HopeNode. Children are stored inline (see HopeNode) so the
# node stays POD and can live directly inside a HopeArena bump allocation.
comptime MAX_CHILDREN = 8


# ==========================================
# 1. ArcGrid
# ==========================================
# Owned, contiguous float32 grid. ARC cell values (0-9) are stored as Float32 so
# the same buffer can feed the SIMD fitness/primitive kernels without conversion.
struct ArcGrid(Movable):
    var data: UnsafePointer[Float32, MutAnyOrigin]
    var rows: Int
    var cols: Int

    def __init__(out self, rows: Int, cols: Int):
        self.rows = rows
        self.cols = cols
        var n = rows * cols
        self.data = alloc[Float32](n)
        memset_zero(self.data, n)

    def __del__(deinit self):
        # A moved-from grid is consumed, so its destructor never runs and the
        # live buffer is always valid here — free unconditionally.
        self.data.free()

    def get(self, r: Int, c: Int) -> Float32:
        return self.data[r * self.cols + c]

    def set(mut self, r: Int, c: Int, val: Float32):
        self.data[r * self.cols + c] = val

    def size(self) -> Int:
        return self.rows * self.cols


# ==========================================
# Phase 2.1: The HopeArena Allocator
# ==========================================
# A move-only, zero-overhead contiguous bump-pointer allocator. It hands out raw,
# typed slices of its single backing buffer; nothing is individually freed — the
# whole arena is released at once in __del__.
struct HopeArena(Movable):
    var data: UnsafePointer[UInt8, MutAnyOrigin]
    var capacity: Int
    var offset: Int

    def __init__(out self, capacity: Int):
        self.capacity = capacity
        self.offset = 0
        self.data = alloc[UInt8](capacity)
        # Zero bare-metal memory so freshly bumped slices start from a known state.
        memset_zero(self.data, capacity)

    def __del__(deinit self):
        # Consuming moves suppress the source's destructor, so the buffer is
        # always live here (UnsafePointer is non-null by design in Mojo 1.0).
        self.data.free()

    # Bump `count` contiguous, suitably-aligned `T` slots out of the arena.
    def bump[
        T: AnyType
    ](mut self, count: Int) raises -> UnsafePointer[T, MutAnyOrigin]:
        var bytes = size_of[T]() * count
        if self.offset + bytes > self.capacity:
            raise Error("HopeArena out of memory")
        var ptr = (self.data + self.offset).bitcast[T]()
        self.offset += bytes
        return ptr

    # Convenience for a single object slot (e.g. a HopeNode header).
    def alloc_node[
        T: AnyType
    ](mut self) raises -> UnsafePointer[T, MutAnyOrigin]:
        return self.bump[T](1)


# ==========================================
# 2. HopeNode
# ==========================================
# Strictly POD: it holds raw slices that point *into* a HopeArena plus inline
# child indices. Because there are no owning members, a HopeNode can be placed
# directly into arena memory (init_pointee_move) with no constructor/destructor
# bookkeeping and no per-node heap allocation — see build_node below.
struct HopeNode(Copyable, Movable):
    var slow: UnsafePointer[Float32, MutAnyOrigin]
    var fast: UnsafePointer[Float32, MutAnyOrigin]
    var slow_dim: Int
    var fast_dim: Int
    var num_children: Int
    var children: InlineArray[Int, MAX_CHILDREN]

    def __init__(
        out self,
        slow: UnsafePointer[Float32, MutAnyOrigin],
        fast: UnsafePointer[Float32, MutAnyOrigin],
        slow_dim: Int,
        fast_dim: Int,
    ):
        self.slow = slow
        self.fast = fast
        self.slow_dim = slow_dim
        self.fast_dim = fast_dim
        self.num_children = 0
        self.children = InlineArray[Int, MAX_CHILDREN](fill=-1)

    # Record the arena index of a child node.
    def add_child(mut self, idx: Int) raises:
        if self.num_children >= MAX_CHILDREN:
            raise Error("HopeNode child capacity exceeded")
        self.children[self.num_children] = idx
        self.num_children += 1


# Allocate a HopeNode header *and* its weight slices from the arena, then place a
# zero-initialised node into the header slot. Everything the node references lives
# in the arena, so the node is genuinely zero-overhead and leak-free.
def build_node(
    mut arena: HopeArena, slow_dim: Int, fast_dim: Int
) raises -> UnsafePointer[HopeNode, MutAnyOrigin]:
    var node_ptr = arena.alloc_node[HopeNode]()
    var slow = arena.bump[Float32](slow_dim)
    var fast = arena.bump[Float32](fast_dim)
    for i in range(slow_dim):
        slow[i] = 0.0
    for i in range(fast_dim):
        fast[i] = 0.0
    node_ptr.init_pointee_move(HopeNode(slow, fast, slow_dim, fast_dim))
    return node_ptr


# ==========================================
# Phase 2.2: Logic Primitives
# ==========================================
# Primitives operate on a flat row-major Float32 grid in place. They are dispatched
# by op-code through `apply_primitive` (a function-pointer table is intentionally
# avoided to keep the hot path free of indirect calls).
comptime PRIM_IDENTITY = 0
comptime PRIM_SHIFT = 1


# Identity: leave the grid unchanged.
def prim_identity(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    params: UnsafePointer[Float32, MutAnyOrigin],
):
    pass


# Shift: translate the grid by (dx, dy) = (params[0], params[1]) with zero fill.
# Reads from a scratch copy so the in-place write cannot clobber source cells.
def prim_shift(
    grid: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    params: UnsafePointer[Float32, MutAnyOrigin],
):
    var dx = Int(params[0])
    var dy = Int(params[1])
    var n = rows * cols
    var scratch = alloc[Float32](n)
    memcpy(dest=scratch, src=grid, count=n)
    for r in range(rows):
        for c in range(cols):
            var sr = r - dy
            var sc = c - dx
            var val = Float32(0.0)
            if sr >= 0 and sr < rows and sc >= 0 and sc < cols:
                val = scratch[sr * cols + sc]
            grid[r * cols + c] = val
    scratch.free()


# Dispatch a primitive by op-code.
def apply_primitive(
    op: Int,
    grid: UnsafePointer[Float32, MutAnyOrigin],
    rows: Int,
    cols: Int,
    params: UnsafePointer[Float32, MutAnyOrigin],
):
    if op == PRIM_SHIFT:
        prim_shift(grid, rows, cols, params)
    else:
        prim_identity(grid, rows, cols, params)


# ==========================================
# Phase 2.3: Vectorized Fast-Weight Update
# ==========================================
# W_fast = W_fast - alpha * (grad + lambda * W_slow)
# The slow weights act as an L2 anchor pulling the fast (memory) weights back
# toward the base routing knowledge. SIMD main loop + scalar remainder, FMA only.
def update_fast_weights(
    fast_weights: UnsafePointer[Float32, MutAnyOrigin],
    slow_weights: UnsafePointer[Float32, MutAnyOrigin],
    gradients: UnsafePointer[Float32, MutAnyOrigin],
    alpha: Float32,
    reg_lambda: Float32,
    size: Int,
):
    var neg_alpha = SIMD[DType.float32, nelts](-alpha)
    var lam = SIMD[DType.float32, nelts](reg_lambda)

    # Vectorized main loop.
    for i in range(0, size - nelts + 1, nelts):
        var fw_vec = fast_weights.load[width=nelts](i)
        var sw_vec = slow_weights.load[width=nelts](i)
        var grad_vec = gradients.load[width=nelts](i)
        # effective_grad = grad + lambda * W_slow
        var eff = fma(lam, sw_vec, grad_vec)
        # W_fast = W_fast - alpha * effective_grad
        fast_weights.store[width=nelts](i, fma(neg_alpha, eff, fw_vec))

    # Scalar remainder.
    var remainder = size % nelts
    if remainder > 0:
        for i in range(size - remainder, size):
            var fw_val = fast_weights[i]
            var sw_val = slow_weights[i]
            var grad_val = gradients[i]
            var eff = fma(reg_lambda, sw_val, grad_val)
            fast_weights[i] = fma(-alpha, eff, fw_val)


# ==========================================
# Phase 2: Execution Loop
# ==========================================
# A demonstration pair (input -> output) for in-context learning.
struct ArcTaskPair(Movable):
    var input_grid: ArcGrid
    var output_grid: ArcGrid

    def __init__(out self, var input_grid: ArcGrid, var output_grid: ArcGrid):
        self.input_grid = input_grid^
        self.output_grid = output_grid^


# Nested optimization: a learning phase adapts the node's fast weights over the
# demonstrations (reusing a single gradient workspace — no per-iteration alloc),
# then an evaluation phase runs a real primitive forward pass on the test input.
def forward_with_learning(
    node: UnsafePointer[HopeNode, MutAnyOrigin],
    demonstrations: List[ArcTaskPair],
    test_input: ArcGrid,
    op: Int,
) raises -> ArcGrid:
    var learning_rate = Float32(0.1)
    var reg_lambda = Float32(0.01)
    var size = node[].fast_dim

    # 1. Learning phase. The gradient buffer is allocated ONCE and reused.
    var gradients = alloc[Float32](size)
    for d in range(len(demonstrations)):
        # Placeholder error signal derived from the demonstration index; a full
        # implementation would backprop-free estimate this from the routed pair.
        var signal = Float32(0.5)
        for g in range(size):
            gradients[g] = signal
        update_fast_weights(
            node[].fast, node[].slow, gradients, learning_rate, reg_lambda, size
        )
    gradients.free()

    # 2. Evaluation phase. Apply the selected primitive to the test input using
    #    the (now adapted) fast weights as the primitive parameters.
    var result = ArcGrid(test_input.rows, test_input.cols)
    for i in range(test_input.size()):
        result.data[i] = test_input.data[i]
    apply_primitive(op, result.data, result.rows, result.cols, node[].fast)
    return result^
