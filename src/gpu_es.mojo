# GPU-batched ES fitness (the GPU rungs, G1+). The compute map is stark: ~all
# engine FLOPs are the windowed attention-gather forward inside `fitness`,
# evaluated 2·N·n_demos times per ES iteration over thousands of iterations,
# while the searched parameter vector is only ATTN_DIM = 7 wide. So the GPU
# seam is exactly the fitness boundary: noise draw, antithetic coefficients,
# gradient reduction and the parameter update stay on the CPU (tiny,
# order-dependent, deterministic), and one kernel launch per iteration scores
# every (candidate × demo) pair in parallel — one thread block per pair, one
# thread per output-pixel stripe, a fixed-order shared-memory tree reduction
# for the squared error (no atomics ⇒ bit-identical GPU reruns).
#
# The per-pixel gather math is NOT duplicated here: the kernel calls the same
# `attn_pixel_*` free functions the CPU `Memory.apply` loops use (memory_es
# .mojo), so the reference path and the batched path cannot diverge. The CPU
# path stays the generic reference for every memory family; this module is
# deliberately specialized to the corpus-path forwards.
#
# Determinism contract (user decision, journal 2026-07-07): a GPU run is
# bit-identical to itself (fixed reduction order + the same serial CPU RNG
# stream as the CPU path), but NOT to a CPU run — the float summation order
# inside a demo's MSE differs, so ES trajectories diverge. Parity is checked
# at the fitness level (test_gpu_parity), quality at the usual held-out bars.
#
# Alloc discipline: device buffers + host staging are created ONCE per fit
# (like fit_shape's own-scratch pattern) and reused across iterations; the
# per-iteration device traffic is the perturbed candidates up (2N×7 floats)
# and the per-(candidate,demo) partials down (2N×n_demos floats) — a few KB.
from std.gpu.host import DeviceContext
from std.gpu import thread_idx, block_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer, alloc, memset_zero, stack_allocation
from std.math import fma, exp, log
from std.random import randn_float64
from hope import ArcTaskPair
from memory_es import (
    attn_pixel_plain,
    attn_pixel_toroidal,
    attn_pixel_reflect,
    ATTN_DIM,
    AttnGatherMemory,
)
from memory_composed import (
    ShapeGeomComposedMemory,
    ShapeGeomSettleMemory,
    SHAPEGEOM_DIM,
    SHAPEGEOM_TREL_OFF,
    SHAPEGEOM_SHAPE_OFF,
    SHAPEGEOM_MODE_OFF,
)

# Threads per block. Each block owns one (candidate, demo) pair and stripes
# its threads over the demo's output pixels; 256 covers a 16x16 grid in one
# pass and a 30x30 in four.
comptime GPU_BLOCK = 256


# One (candidate, demo) squared-error partial. Block b maps to
# cand = b / n_demos, demo = b % n_demos; thread t strides pixels t, t+256, ...
# computing the shared per-pixel gather minus the target, then a fixed-order
# tree reduction in shared memory writes partial[b] = SSE. Grids are flat
# stripes: demo d's input/output live at d*cap; dims in a small int32 buffer.
def _fitness_kernel_plain(
    pert: UnsafePointer[Float32, MutAnyOrigin],  # n_cand × ATTN_DIM
    demo_in: UnsafePointer[Float32, MutAnyOrigin],  # n_demos × cap
    demo_out: UnsafePointer[Float32, MutAnyOrigin],  # n_demos × cap
    dims: UnsafePointer[Int32, MutAnyOrigin],  # n_demos × 2 (rows, cols)
    partial: UnsafePointer[Float32, MutAnyOrigin],  # n_cand × n_demos
    n_demos: Int,
    cap: Int,
):
    var shared = stack_allocation[
        GPU_BLOCK, Float32, address_space=AddressSpace.SHARED
    ]()
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var cand = b // n_demos
    var demo = b % n_demos
    var rows = Int(dims[demo * 2 + 0])
    var cols = Int(dims[demo * 2 + 1])
    var n = rows * cols
    var w = pert + cand * ATTN_DIM
    var src = demo_in + demo * cap
    var tgt = demo_out + demo * cap

    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var r = i // cols
        var c = i % cols
        var got = attn_pixel_plain(w, src, rows, cols, rows, cols, r, c)
        var diff = got - tgt[i]
        acc = fma(diff, diff, acc)
        i += GPU_BLOCK

    # Fixed-order tree reduction (deterministic for a given block size).
    shared[tid] = acc
    barrier()
    var step = GPU_BLOCK // 2
    while step > 0:
        if tid < step:
            shared[tid] += shared[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        partial[b] = shared[0]


# CPU assembly of per-candidate fitness from the kernel's per-(cand,demo)
# SSE partials — the exact semantics of `fitness[M]` / `fitness_shape[M]`
# (esper_evolution.mojo): mean over demos of the Domain distance (negative
# MSE over that demo's SCORED pixel count) or the heavy shape penalty, minus
# the L2 anchor toward the slow prior over the memory's pdim. Shared by the
# fit loops and the single-candidate parity entry points so the semantics
# live once.
def _assemble_fitness(
    partial: UnsafePointer[Float32, MutAnyOrigin],  # n_cand × n_demos SSE
    mismatch: UnsafePointer[Int32, MutAnyOrigin],  # n_demos
    n_px: UnsafePointer[Int32, MutAnyOrigin],  # n_demos scored pixel counts
    pert_all: UnsafePointer[Float32, MutAnyOrigin],  # n_cand × pdim
    slow_weights: UnsafePointer[Float32, MutAnyOrigin],
    pdim: Int,
    n_cand: Int,
    n_demos: Int,
    reg_lambda: Float32,
    f_all: UnsafePointer[Float32, MutAnyOrigin],  # n_cand (out)
):
    for cand in range(n_cand):
        var total = Float32(0.0)
        for d in range(n_demos):
            if mismatch[d] == 1:
                total += Float32(-1.0e9)
                continue
            total += -(partial[cand * n_demos + d] / Float32(n_px[d]))
        total = total / Float32(n_demos)
        var anchor = Float32(0.0)
        var w = pert_all + cand * pdim
        for j in range(pdim):
            var diff = w[j] - slow_weights[j]
            anchor += diff * diff
        f_all[cand] = total - reg_lambda * anchor / Float32(pdim)


# The GPU sibling of `fit_operator[AttnGatherMemory]` (esper_evolution.mojo):
# the SAME annealed antithetic-ES schedule, the SAME serial RNG stream, the
# SAME gradient reduction and update arithmetic — only the 2N×n_demos fitness
# forwards are batched into one kernel launch per iteration. Self-contained
# (own scratch, no ESWorkspace) to keep the import DAG acyclic; specialized
# to AttnGatherMemory because that IS the corpus path's ES search (both
# fit_geomcolor/fit_local and fit_geomcount fit the 7 attention slots).
def fit_operator_gpu(
    fast_weights: UnsafePointer[Float32, MutAnyOrigin],
    slow_weights: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    N: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
) raises:
    if iters <= 0 or N <= 0:
        return
    var n_demos = len(demos)
    if n_demos == 0:
        return
    comptime pdim = ATTN_DIM
    var n_cand = 2 * N

    var ctx = DeviceContext()

    # --- Once-per-fit setup (device + host staging, reused every iteration).
    var d_pert = ctx.enqueue_create_buffer[DType.float32](n_cand * pdim)
    var d_in = ctx.enqueue_create_buffer[DType.float32](n_demos * grid_capacity)
    var d_out = ctx.enqueue_create_buffer[DType.float32](
        n_demos * grid_capacity
    )
    var d_dims = ctx.enqueue_create_buffer[DType.int32](n_demos * 2)
    var d_partial = ctx.enqueue_create_buffer[DType.float32](n_cand * n_demos)

    var h_in = alloc[Float32](n_demos * grid_capacity)
    var h_out = alloc[Float32](n_demos * grid_capacity)
    var h_dims = alloc[Int32](n_demos * 2)
    # Same-shape guard, decided once (dims are fixed): a demo whose output
    # area differs from its input's is inexpressible for this memory — the
    # kernel skips nothing, but its partial is REPLACED by the same heavy
    # penalty `fitness[M]` uses, keeping semantics identical.
    var mismatch = alloc[Int32](n_demos)
    var n_px = alloc[Int32](n_demos)
    for d in range(n_demos):
        var in_n = demos[d].input_grid.rows * demos[d].input_grid.cols
        var out_n = demos[d].output_grid.rows * demos[d].output_grid.cols
        mismatch[d] = 1 if in_n != out_n else 0
        n_px[d] = Int32(in_n)
        h_dims[d * 2 + 0] = Int32(demos[d].input_grid.rows)
        h_dims[d * 2 + 1] = Int32(demos[d].input_grid.cols)
        for k in range(in_n):
            h_in[d * grid_capacity + k] = demos[d].input_grid.data[k]
        for k in range(out_n):
            h_out[d * grid_capacity + k] = demos[d].output_grid.data[k]
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=h_in)
    ctx.enqueue_copy(dst_buf=d_out, src_ptr=h_out)
    ctx.enqueue_copy(dst_buf=d_dims, src_ptr=h_dims)

    var eps_all = alloc[Float32](N * pdim)
    var pert_all = alloc[Float32](n_cand * pdim)
    var f_all = alloc[Float32](n_cand)
    var partial = alloc[Float32](n_cand * n_demos)
    var coeff = alloc[Float32](N)
    var grad = alloc[Float32](pdim)
    var scale = alloc[Float32](pdim)
    AttnGatherMemory.fill_scale(scale, pdim)

    var alpha_rate = log(alpha1 / alpha0) / Float32(iters)
    var sigma_rate = log(sigma1 / sigma0) / Float32(iters)

    for t in range(iters):
        var alpha = alpha0 * exp(alpha_rate * Float32(t))
        var sigma = sigma0 * exp(sigma_rate * Float32(t))

        # 1. Serial CPU: the SAME RNG stream (and order) as the CPU path.
        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                eps_s[j] = Float32(randn_float64(0.0, 1.0))

        # 2. CPU build of every candidate: stripe s = w + sigma*(scale⊙eps_s),
        #    stripe N+s = the antithetic mirror (pdim = 7 — trivial work).
        for s in range(N):
            var eps_s = eps_all + s * pdim
            var pos = pert_all + s * pdim
            var neg = pert_all + (N + s) * pdim
            for j in range(pdim):
                var seps = eps_s[j] * scale[j]
                pos[j] = fma(seps, sigma, fast_weights[j])
                neg[j] = fma(seps, -sigma, fast_weights[j])

        # 3. One launch scores all (candidate × demo) pairs.
        ctx.enqueue_copy(dst_buf=d_pert, src_ptr=pert_all)
        ctx.enqueue_function[_fitness_kernel_plain](
            d_pert,
            d_in,
            d_out,
            d_dims,
            d_partial,
            n_demos,
            grid_capacity,
            grid_dim=n_cand * n_demos,
            block_dim=GPU_BLOCK,
        )
        ctx.enqueue_copy(dst_ptr=partial, src_buf=d_partial)
        ctx.synchronize()

        # 4. CPU: assemble each candidate's fitness (fitness[M] semantics).
        _assemble_fitness(
            partial,
            mismatch,
            n_px,
            pert_all,
            slow_weights,
            pdim,
            n_cand,
            n_demos,
            reg_lambda,
            f_all,
        )
        for s in range(N):
            coeff[s] = f_all[s] - f_all[N + s]

        # 5. Serial gradient reduction + preconditioned step — the identical
        #    arithmetic to evolve_fast_weights' steps 3-4 (pdim = 7, scalar).
        memset_zero(grad, pdim)
        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                grad[j] = fma(eps_s[j], coeff[s], grad[j])
        var update_factor = alpha / (2.0 * Float32(N) * sigma)
        for j in range(pdim):
            fast_weights[j] = fma(
                grad[j] * scale[j], update_factor, fast_weights[j]
            )

    h_in.free()
    h_out.free()
    h_dims.free()
    mismatch.free()
    n_px.free()
    eps_all.free()
    pert_all.free()
    f_all.free()
    partial.free()
    coeff.free()
    grad.free()
    scale.free()


# Single-candidate GPU fitness — the parity-test entry point (mirrors one
# fitness[AttnGatherMemory] call through the kernel + _assemble_fitness, so
# test_gpu_parity can compare the device path against the CPU reference at
# the fitness level, where CPU/GPU agreement is meaningful — trajectories
# are allowed to diverge, fitness values are not, beyond reduction-order
# float noise).
def gpu_fitness_plain(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    slow_weights: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    reg_lambda: Float32,
) raises -> Float32:
    var n_demos = len(demos)
    if n_demos == 0:
        return 0.0
    comptime pdim = ATTN_DIM

    var ctx = DeviceContext()
    var d_pert = ctx.enqueue_create_buffer[DType.float32](pdim)
    var d_in = ctx.enqueue_create_buffer[DType.float32](n_demos * grid_capacity)
    var d_out = ctx.enqueue_create_buffer[DType.float32](
        n_demos * grid_capacity
    )
    var d_dims = ctx.enqueue_create_buffer[DType.int32](n_demos * 2)
    var d_partial = ctx.enqueue_create_buffer[DType.float32](n_demos)

    var h_in = alloc[Float32](n_demos * grid_capacity)
    var h_out = alloc[Float32](n_demos * grid_capacity)
    var h_dims = alloc[Int32](n_demos * 2)
    var mismatch = alloc[Int32](n_demos)
    var n_px = alloc[Int32](n_demos)
    for d in range(n_demos):
        var in_n = demos[d].input_grid.rows * demos[d].input_grid.cols
        var out_n = demos[d].output_grid.rows * demos[d].output_grid.cols
        mismatch[d] = 1 if in_n != out_n else 0
        n_px[d] = Int32(in_n)
        h_dims[d * 2 + 0] = Int32(demos[d].input_grid.rows)
        h_dims[d * 2 + 1] = Int32(demos[d].input_grid.cols)
        for k in range(in_n):
            h_in[d * grid_capacity + k] = demos[d].input_grid.data[k]
        for k in range(out_n):
            h_out[d * grid_capacity + k] = demos[d].output_grid.data[k]
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=h_in)
    ctx.enqueue_copy(dst_buf=d_out, src_ptr=h_out)
    ctx.enqueue_copy(dst_buf=d_dims, src_ptr=h_dims)
    ctx.enqueue_copy(dst_buf=d_pert, src_ptr=weights)

    var partial = alloc[Float32](n_demos)
    var f = alloc[Float32](1)
    ctx.enqueue_function[_fitness_kernel_plain](
        d_pert,
        d_in,
        d_out,
        d_dims,
        d_partial,
        n_demos,
        grid_capacity,
        grid_dim=n_demos,
        block_dim=GPU_BLOCK,
    )
    ctx.enqueue_copy(dst_ptr=partial, src_buf=d_partial)
    ctx.synchronize()
    _assemble_fitness(
        partial,
        mismatch,
        n_px,
        weights,
        slow_weights,
        pdim,
        1,
        n_demos,
        reg_lambda,
        f,
    )
    var result = f[0]
    h_in.free()
    h_out.free()
    h_dims.free()
    mismatch.free()
    n_px.free()
    partial.free()
    f.free()
    return result


# ==========================================
# Rung G2: the shape path (toroidal / reflect gathers)
# ==========================================
# The shape kernel's block layout mirrors _fitness_kernel_plain; the two
# differences come straight from fitness_shape's semantics: the query grid is
# the PREDICTED output extent (pred_rows × pred_cols — constant during a fit
# because the written shape-rule slots are frozen by fill_scale, so the host
# computes them once per demo), and the per-pixel read is mode-dispatched to
# the toroidal or reflect gather exactly like ShapeGeomComposedMemory.apply
# (the mode slot is frozen too, but reading it per candidate keeps the kernel
# a pure function of the state vector — single source of truth). A demo whose
# predicted area mismatches its true output area gets pred dims ZEROED by the
# host (the kernel then does no work, no OOB reads) and the heavy penalty is
# applied in _assemble_fitness, same as the CPU path.
def _fitness_kernel_shape(
    pert: UnsafePointer[Float32, MutAnyOrigin],  # n_cand × SHAPEGEOM_DIM
    demo_in: UnsafePointer[Float32, MutAnyOrigin],  # n_demos × cap
    demo_out: UnsafePointer[Float32, MutAnyOrigin],  # n_demos × cap
    dims: UnsafePointer[Int32, MutAnyOrigin],  # n_demos × 4 (in r,c; pred r,c)
    partial: UnsafePointer[Float32, MutAnyOrigin],  # n_cand × n_demos
    n_demos: Int,
    cap: Int,
):
    var shared = stack_allocation[
        GPU_BLOCK, Float32, address_space=AddressSpace.SHARED
    ]()
    var b = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var cand = b // n_demos
    var demo = b % n_demos
    var in_rows = Int(dims[demo * 4 + 0])
    var in_cols = Int(dims[demo * 4 + 1])
    var pr = Int(dims[demo * 4 + 2])
    var pc = Int(dims[demo * 4 + 3])
    var n = pr * pc
    var w = pert + cand * SHAPEGEOM_DIM
    var src = demo_in + demo * cap
    var tgt = demo_out + demo * cap
    var trel_r = w[SHAPEGEOM_TREL_OFF + 0]
    var trel_c = w[SHAPEGEOM_TREL_OFF + 1]
    var kr = w[SHAPEGEOM_SHAPE_OFF + 0]
    var kc = w[SHAPEGEOM_SHAPE_OFF + 2]
    var reflect = w[SHAPEGEOM_MODE_OFF] >= Float32(0.5)

    var acc = Float32(0.0)
    var i = tid
    while i < n:
        var r = i // pc
        var c = i % pc
        var got = Float32(0.0)
        if reflect:
            got = attn_pixel_reflect(
                w, trel_r, trel_c, kr, kc, src, in_rows, in_cols, pr, pc, r, c
            )
        else:
            got = attn_pixel_toroidal(
                w, trel_r, trel_c, kr, kc, src, in_rows, in_cols, pr, pc, r, c
            )
        var diff = got - tgt[i]
        acc = fma(diff, diff, acc)
        i += GPU_BLOCK

    shared[tid] = acc
    barrier()
    var step = GPU_BLOCK // 2
    while step > 0:
        if tid < step:
            shared[tid] += shared[tid + step]
        barrier()
        step //= 2
    if tid == 0:
        partial[b] = shared[0]


# Host staging for the shape kernel, shared by the fit loop and the parity
# entry point: flatten the demos, predict each demo's output dims from the
# (frozen) written shape rule, and set the mismatch flag / zeroed pred dims
# for inexpressible demos. Returns nothing; fills the caller's buffers.
def _stage_shape_demos(
    state: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    h_in: UnsafePointer[Float32, MutAnyOrigin],
    h_out: UnsafePointer[Float32, MutAnyOrigin],
    h_dims: UnsafePointer[Int32, MutAnyOrigin],
    mismatch: UnsafePointer[Int32, MutAnyOrigin],
    n_px: UnsafePointer[Int32, MutAnyOrigin],
):
    for d in range(len(demos)):
        var in_n = demos[d].input_grid.rows * demos[d].input_grid.cols
        var out_n = demos[d].output_grid.rows * demos[d].output_grid.cols
        var pr = ShapeGeomComposedMemory.out_rows(state, demos[d].input_grid)
        var pc = ShapeGeomComposedMemory.out_cols(state, demos[d].input_grid)
        var bad = pr * pc != out_n
        mismatch[d] = 1 if bad else 0
        n_px[d] = Int32(out_n)
        h_dims[d * 4 + 0] = Int32(demos[d].input_grid.rows)
        h_dims[d * 4 + 1] = Int32(demos[d].input_grid.cols)
        # Zeroed pred dims make the kernel a no-op for this demo (no OOB).
        h_dims[d * 4 + 2] = 0 if bad else Int32(pr)
        h_dims[d * 4 + 3] = 0 if bad else Int32(pc)
        for k in range(in_n):
            h_in[d * grid_capacity + k] = demos[d].input_grid.data[k]
        for k in range(out_n):
            h_out[d * grid_capacity + k] = demos[d].output_grid.data[k]


# The GPU sibling of `fit_shape[M]` (esper_evolution.mojo): same annealed
# antithetic ES, same serial RNG stream, same reduction/update arithmetic;
# the 2N×n_demos fitness_shape forwards are batched into one launch per
# iteration. Non-generic: the kernel assumes the SHAPEGEOM state layout,
# shared by the ShapeGeomComposed/Settle pair, whose ONLY difference is the
# fill_scale (the settle type freezes the temperature slot) — carried here
# by the `settle` flag. This is deliberately the shape-seam specialization,
# not a generic ShapeMemory backend. Predicted demo dims are computed ONCE
# (the shape slots are frozen), matching fitness_shape's per-call values.
def fit_shape_gpu(
    state: UnsafePointer[Float32, MutAnyOrigin],
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    N: Int,
    alpha0: Float32,
    alpha1: Float32,
    sigma0: Float32,
    sigma1: Float32,
    iters: Int,
    reg_lambda: Float32,
    settle: Bool,
) raises:
    if iters <= 0 or N <= 0:
        return
    var n_demos = len(demos)
    if n_demos == 0:
        return
    var pdim = ShapeGeomComposedMemory.param_dim()
    var n_cand = 2 * N

    var ctx = DeviceContext()

    var d_pert = ctx.enqueue_create_buffer[DType.float32](n_cand * pdim)
    var d_in = ctx.enqueue_create_buffer[DType.float32](n_demos * grid_capacity)
    var d_out = ctx.enqueue_create_buffer[DType.float32](
        n_demos * grid_capacity
    )
    var d_dims = ctx.enqueue_create_buffer[DType.int32](n_demos * 4)
    var d_partial = ctx.enqueue_create_buffer[DType.float32](n_cand * n_demos)

    var h_in = alloc[Float32](n_demos * grid_capacity)
    var h_out = alloc[Float32](n_demos * grid_capacity)
    var h_dims = alloc[Int32](n_demos * 4)
    var mismatch = alloc[Int32](n_demos)
    var n_px = alloc[Int32](n_demos)
    _stage_shape_demos(
        state, demos, grid_capacity, h_in, h_out, h_dims, mismatch, n_px
    )
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=h_in)
    ctx.enqueue_copy(dst_buf=d_out, src_ptr=h_out)
    ctx.enqueue_copy(dst_buf=d_dims, src_ptr=h_dims)

    var eps_all = alloc[Float32](N * pdim)
    var pert_all = alloc[Float32](n_cand * pdim)
    var f_all = alloc[Float32](n_cand)
    var partial = alloc[Float32](n_cand * n_demos)
    var coeff = alloc[Float32](N)
    var grad = alloc[Float32](pdim)
    var scale = alloc[Float32](pdim)
    # The DISCOVER/SETTLE phase difference is exactly the fill_scale (the
    # settle type freezes the temperature slot) — a flag here, a type on CPU.
    if settle:
        ShapeGeomSettleMemory.fill_scale(scale, pdim)
    else:
        ShapeGeomComposedMemory.fill_scale(scale, pdim)

    var alpha_rate = log(alpha1 / alpha0) / Float32(iters)
    var sigma_rate = log(sigma1 / sigma0) / Float32(iters)

    for t in range(iters):
        var alpha = alpha0 * exp(alpha_rate * Float32(t))
        var sigma = sigma0 * exp(sigma_rate * Float32(t))

        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                eps_s[j] = Float32(randn_float64(0.0, 1.0))

        for s in range(N):
            var eps_s = eps_all + s * pdim
            var pos = pert_all + s * pdim
            var neg = pert_all + (N + s) * pdim
            for j in range(pdim):
                var seps = eps_s[j] * scale[j]
                pos[j] = fma(seps, sigma, state[j])
                neg[j] = fma(seps, -sigma, state[j])

        ctx.enqueue_copy(dst_buf=d_pert, src_ptr=pert_all)
        ctx.enqueue_function[_fitness_kernel_shape](
            d_pert,
            d_in,
            d_out,
            d_dims,
            d_partial,
            n_demos,
            grid_capacity,
            grid_dim=n_cand * n_demos,
            block_dim=GPU_BLOCK,
        )
        ctx.enqueue_copy(dst_ptr=partial, src_buf=d_partial)
        ctx.synchronize()

        _assemble_fitness(
            partial,
            mismatch,
            n_px,
            pert_all,
            slow,
            pdim,
            n_cand,
            n_demos,
            reg_lambda,
            f_all,
        )
        for s in range(N):
            coeff[s] = f_all[s] - f_all[N + s]

        memset_zero(grad, pdim)
        for s in range(N):
            var eps_s = eps_all + s * pdim
            for j in range(pdim):
                grad[j] = fma(eps_s[j], coeff[s], grad[j])
        var update_factor = alpha / (2.0 * Float32(N) * sigma)
        for j in range(pdim):
            state[j] = fma(grad[j] * scale[j], update_factor, state[j])

    h_in.free()
    h_out.free()
    h_dims.free()
    mismatch.free()
    n_px.free()
    eps_all.free()
    pert_all.free()
    f_all.free()
    partial.free()
    coeff.free()
    grad.free()
    scale.free()


# Single-candidate GPU shape fitness — the parity-test entry point for the
# shape kernel (mirrors one fitness_shape[M] call; see gpu_fitness_plain).
def gpu_fitness_shape(
    state: UnsafePointer[Float32, MutAnyOrigin],
    slow: UnsafePointer[Float32, MutAnyOrigin],
    demos: List[ArcTaskPair],
    grid_capacity: Int,
    reg_lambda: Float32,
) raises -> Float32:
    var n_demos = len(demos)
    if n_demos == 0:
        return 0.0
    var pdim = ShapeGeomComposedMemory.param_dim()

    var ctx = DeviceContext()
    var d_pert = ctx.enqueue_create_buffer[DType.float32](pdim)
    var d_in = ctx.enqueue_create_buffer[DType.float32](n_demos * grid_capacity)
    var d_out = ctx.enqueue_create_buffer[DType.float32](
        n_demos * grid_capacity
    )
    var d_dims = ctx.enqueue_create_buffer[DType.int32](n_demos * 4)
    var d_partial = ctx.enqueue_create_buffer[DType.float32](n_demos)

    var h_in = alloc[Float32](n_demos * grid_capacity)
    var h_out = alloc[Float32](n_demos * grid_capacity)
    var h_dims = alloc[Int32](n_demos * 4)
    var mismatch = alloc[Int32](n_demos)
    var n_px = alloc[Int32](n_demos)
    _stage_shape_demos(
        state, demos, grid_capacity, h_in, h_out, h_dims, mismatch, n_px
    )
    ctx.enqueue_copy(dst_buf=d_in, src_ptr=h_in)
    ctx.enqueue_copy(dst_buf=d_out, src_ptr=h_out)
    ctx.enqueue_copy(dst_buf=d_dims, src_ptr=h_dims)
    ctx.enqueue_copy(dst_buf=d_pert, src_ptr=state)

    var partial = alloc[Float32](n_demos)
    var f = alloc[Float32](1)
    ctx.enqueue_function[_fitness_kernel_shape](
        d_pert,
        d_in,
        d_out,
        d_dims,
        d_partial,
        n_demos,
        grid_capacity,
        grid_dim=n_demos,
        block_dim=GPU_BLOCK,
    )
    ctx.enqueue_copy(dst_ptr=partial, src_buf=d_partial)
    ctx.synchronize()
    _assemble_fitness(
        partial, mismatch, n_px, state, slow, pdim, 1, n_demos, reg_lambda, f
    )
    var result = f[0]
    h_in.free()
    h_out.free()
    h_dims.free()
    mismatch.free()
    n_px.free()
    partial.free()
    f.free()
    return result
