# The ES-fit forward memory family: parameter vectors fit in-context by the
# annealed Evolution Strategy through the generic `fitness[M]`/`fit_operator[M]`
# core. Grown per roadmap block: the structured OperatorMemory (Phase A,
# dormant), the per-cell MLP (B1), the sequence-domain pair (B2), and the
# attention gather (B3). The trait seam lives in memory.mojo; the self-write
# family in memory_selfmod*.mojo; the composed memory in memory_composed.mojo.
from std.memory import UnsafePointer
from std.math import tanh, fma, floor, round, exp
from hope import (
    ArcGrid,
    Sequence,
    OP_DIM,
    COLOR_OFF,
    COLOR_DIM,
    seed_identity_operator,
    apply_operator,
)
from arc_io import GridDomain, SeqDomain
from memory import Memory

# Per-group ES step scale for the structured operator (a diagonal preconditioner).
# The colour-LUT parameters are normalized to ~unit scale but with tight 1/9
# spacing, so they need a smaller step than the affine. This used to be a
# hardcoded branch in ESWorkspace; it now belongs to the operator memory that
# owns the colour group (a different memory has a different/uniform scale).
comptime COLOR_SCALE = Float32(0.6)


# The structured affine + colour-LUT operator, wrapped as the first Memory.
# Pure delegation to the existing hope.mojo entry points — this is the
# zero-behavior-change instance that keeps the whole suite green through the new
# generic path. (The operator stays the Phase-A baseline; emergent memories grow
# alongside it and are measured on the subsets they can express.)
#
# DORMANT (block 5): its whole expressible subset {flip_h, flip_v, transpose,
# recolor} — and compositions the structured operator was never proven on — is
# now subsumed EMERGENTLY by `GeomColorComposedMemory` (attention-gather
# geometry × a count-signature colour self-write; no hand-coded affine/LUT),
# proven in test_composed_generalization. OperatorMemory (and hope.mojo's
# `apply_operator`) is kept only so the real-ARC M8 eval (arc_solve/main) keeps
# its baseline number; it is no longer the grid path's frontier. Full removal
# is a later cleanup.
struct OperatorMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return OP_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        seed_identity_operator(weights)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # The colour group gets the smaller step (diagonal preconditioner).
        for c in range(COLOR_DIM):
            scale[COLOR_OFF + c] = COLOR_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        apply_operator(weights, inp.data, dst, inp.rows, inp.cols)


# ==========================================
# First emergent memory: a per-cell MLP (Phase B / B1)
# ==========================================
# The first training-wheel removal: a generic per-cell function approximator with
# NO hand-coded colour structure. Over a K=1 receptive field (the centre cell) it
# is a 1->H->1 MLP: x = in[r,c]/9; z = W2 . tanh(W1*x + b1) + b2; and the output is
# SQUASHED to the valid colour range, out[r,c] = 9 * (tanh(z)+1)/2. The squash is
# essential: unlike the operator (a bounded gather of input values), a raw linear
# MLP output is unbounded, so a large W2 explodes the MSE and the ES diverges;
# bounding the output to [0,9] keeps the fitness landscape sane under the shared
# annealed schedule (the operator gets the same [0,9]-bounded property for free).
# The colour mapping (e.g. recolor) is *learned* as the output layer over a fixed
# tanh basis tiling [0,1] — emergent, not a LUT. Same-shape, continuous (smooth ES
# landscape; exact_match rounds only at scoring), scalar-per-cell like the operator
# gather. Cannot express global geometry (flip/transpose) — that needs a global
# addressing memory (B3); the MLP targets the local/colour subset.
comptime MLP_HIDDEN = 16
comptime MLP_W1_OFF = 0
comptime MLP_B1_OFF = MLP_HIDDEN
comptime MLP_W2_OFF = 2 * MLP_HIDDEN
comptime MLP_B2_OFF = 3 * MLP_HIDDEN
comptime MLP_DIM = 3 * MLP_HIDDEN + 1
# The output layer (W2, b2) moves freely; the basis (W1, b1) is nudged gently so
# the seeded tiling of [0,1] stays a stable feature map (a diagonal preconditioner,
# the MLP analogue of the operator's COLOR_SCALE).
comptime MLP_BASIS_SCALE = Float32(0.2)
# Output squash to EXACTLY the valid colour range [0, 9]. Bounding the output
# keeps the MSE (and the ES) stable; squashing to [0,9] specifically means a
# SATURATED output lands on a valid extreme colour (0 or 9), which rounds
# correctly — a wider range would saturate to invalid colours (e.g. -2/11) that
# never match (diagnosed via scratch/probe_mlp). Extremes need only out<=0.5 or
# >=8.5 (well inside tanh saturation), so the exact-match tolerance covers them.
comptime MLP_OUT_LO = Float32(0.0)
comptime MLP_OUT_HI = Float32(9.0)


# The per-element MLP forward (shared by MLPMemory over a grid and SeqMLPMemory
# over a sequence — both are purely per-cell, so the math lives once here): map a
# raw token value through x = v/9; z = W2 . tanh(W1*x + b1) + b2; squash to [0,9].
def _mlp_cell(
    weights: UnsafePointer[Float32, MutAnyOrigin], x_raw: Float32
) -> Float32:
    var x = x_raw / 9.0
    var z = weights[MLP_B2_OFF]
    for h in range(MLP_HIDDEN):
        var a = tanh(fma(weights[MLP_W1_OFF + h], x, weights[MLP_B1_OFF + h]))
        z = fma(weights[MLP_W2_OFF + h], a, z)
    # Squash to [MLP_OUT_LO, MLP_OUT_HI] (bounded output -> bounded MSE -> stable
    # ES; the [0,9] range means a saturated output lands on a valid extreme
    # colour, which rounds correctly under exact_match's tolerance).
    return MLP_OUT_LO + (MLP_OUT_HI - MLP_OUT_LO) * 0.5 * (tanh(z) + 1.0)


struct MLPMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return MLP_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Seed a fixed tanh basis that tiles the colour range. Each hidden unit
        # is a STEEP ramp (W1=20): it transitions over ~1/W1 in x, which must be
        # narrower than the 1/9 spacing between adjacent colours, else the basis
        # can't separate them — so the basis is near piecewise-constant, a soft
        # lookup table. Unit h transitions (W1*x + b1 = 0) at x = center_h, with
        # the centers tiling a range slightly WIDER than [0,1] (the valid
        # x = colour/9 span) so the edge colours (0 and 9) sit INSIDE the basis
        # with ramps on both sides (else they fall at the tiling boundary with no
        # features to shape them). The output layer starts at zero (W2 = b2 = 0);
        # the ES fits W2/b2 to interpolate the colour map over this basis (a
        # robust, ~linear-in-W2 fit). The 9->0 recolor wrap fits for free: colour
        # 9 saturates the squash low, which is exactly 0.
        comptime LO = Float32(-0.2)
        comptime HI = Float32(1.2)
        var denom = Float32(MLP_HIDDEN - 1)
        for h in range(MLP_HIDDEN):
            var center = LO + (HI - LO) * (Float32(h) / denom)
            weights[MLP_W1_OFF + h] = 20.0
            weights[MLP_B1_OFF + h] = -20.0 * center
            weights[MLP_W2_OFF + h] = 0.0
        weights[MLP_B2_OFF] = 0.0

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # Keep the basis (W1, b1) gentle; let the output layer (W2, b2) move.
        for h in range(MLP_HIDDEN):
            scale[MLP_W1_OFF + h] = MLP_BASIS_SCALE
            scale[MLP_B1_OFF + h] = MLP_BASIS_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(inp.rows * inp.cols):
            dst[i] = _mlp_cell(weights, inp.data[i])


# ==========================================
# Second domain (Phase B / B2): sequence memories
# ==========================================
# These two memories prove the Domain seam carries cross-domain: both conform to
# the SAME Memory trait with Dom = SeqDomain (a non-grid domain) and are fit by the
# UNCHANGED ES core. SeqOperatorMemory is the structured 1-D analog of
# OperatorMemory (a position affine + value LUT); SeqMLPMemory is the emergent
# per-element analog of MLPMemory (it reuses the exact same MLP layout + _mlp_cell).

# 1-D structured operator layout (grid-size-independent, the analog of OP_DIM):
# a 2-param centered affine on position (scale a, translation t) + a 10-entry value
# LUT over tokens 0..9, normalized /9 to keep it at the affine's ~unit scale.
comptime SEQ_COORD_DIM = 2
comptime SEQ_VALUE_DIM = 10
comptime SEQ_DIM = SEQ_COORD_DIM + SEQ_VALUE_DIM
comptime SEQ_A_OFF = 0
comptime SEQ_T_OFF = 1
comptime SEQ_VALUE_OFF = SEQ_COORD_DIM
# The value group needs a smaller ES step than the affine (tight 1/9 spacing) — a
# diagonal preconditioner, exactly the 1-D analog of the operator's COLOR_SCALE.
comptime SEQ_VALUE_SCALE = Float32(0.6)


# Value-LUT lookup (the 1-D analog of _color_of): round to the nearest token, clamp
# to [0,9], read the (normalized) LUT entry and rescale to the 0..9 range.
def _seq_value_of(
    weights: UnsafePointer[Float32, MutAnyOrigin], x: Float32
) -> Float32:
    var idx = Int(round(x))
    if idx < 0:
        idx = 0
    elif idx > SEQ_VALUE_DIM - 1:
        idx = SEQ_VALUE_DIM - 1
    return Float32(SEQ_VALUE_DIM - 1) * weights[SEQ_VALUE_OFF + idx]


# Read a sequence cell with zero-fill out of bounds (the 1-D analog of
# _cell_or_zero) — the search can sample OOB but the optima never do.
def _seq_cell_or_zero(
    data: UnsafePointer[Float32, MutAnyOrigin], length: Int, i: Int
) -> Float32:
    if i < 0 or i >= length:
        return 0.0
    return data[i]


# The structured sequence operator. Per output position p (over `length`), centered
# cp = p - (L-1)/2; source sp = a*cp + (L-1)/2 + t. VALUE-then-gather: the two
# integer source cells are mapped through the value LUT BEFORE the 1-D linear
# interpolation (the 1-D case of the grid's bilinear colour-then-gather — it
# decouples the value fit from the affine's precision, and the linear blend keeps a
# smooth ES landscape). Output stays continuous; exact_match rounds only at scoring.
# Integer params reproduce reverse (a=-1, t=0), shift-k (a=1, t=k) and increment
# (value LUT (v+1)%10) exactly.
struct SeqOperatorMemory(Memory):
    comptime Dom = SeqDomain

    @staticmethod
    def param_dim() -> Int:
        return SEQ_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        weights[SEQ_A_OFF] = 1.0  # identity position map
        weights[SEQ_T_OFF] = 0.0
        for v in range(SEQ_VALUE_DIM):
            weights[SEQ_VALUE_OFF + v] = Float32(v) / Float32(SEQ_VALUE_DIM - 1)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # The value group gets the smaller step (diagonal preconditioner).
        for v in range(SEQ_VALUE_DIM):
            scale[SEQ_VALUE_OFF + v] = SEQ_VALUE_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: Sequence,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var a = weights[SEQ_A_OFF]
        var t = weights[SEQ_T_OFF]
        var center = Float32(inp.length - 1) * Float32(0.5)
        for p in range(inp.length):
            var cp = Float32(p) - center
            var sp = fma(a, cp, center + t)
            var i0 = Int(floor(sp))
            var f = sp - Float32(i0)
            var v0 = _seq_value_of(
                weights, _seq_cell_or_zero(inp.data, inp.length, i0)
            )
            var v1 = _seq_value_of(
                weights, _seq_cell_or_zero(inp.data, inp.length, i0 + 1)
            )
            dst[p] = fma(f, v1 - v0, v0)


# The emergent per-element sequence memory: the SAME 1->H->1 tanh MLP as MLPMemory
# (it reuses MLP_DIM/seed/fill_scale and the shared _mlp_cell forward verbatim),
# applied over the sequence instead of the grid. Expresses element-wise maps only
# (e.g. increment) — NO hand-coded value structure; cannot do reverse/shift (a
# position permutation; that is B3's emergent global addressing). This shows the
# emergent path transfers to the new domain unchanged.
struct SeqMLPMemory(Memory):
    comptime Dom = SeqDomain

    @staticmethod
    def param_dim() -> Int:
        return MLP_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        MLPMemory.seed(weights)

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        MLPMemory.fill_scale(scale, n)

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: Sequence,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        for i in range(inp.length):
            dst[i] = _mlp_cell(weights, inp.data[i])


# ==========================================
# Emergent global addressing (Phase B / B3): attention gather (geometry)
# ==========================================
# The per-cell MLP (B1) is LOCAL — it cannot express geometry, because flip/
# transpose are GLOBAL coordinate permutations (out[i] = in[perm(i)]). OperatorMemory
# does geometry with a HAND-CODED affine that computes one source address and gathers
# exactly there. B3 re-earns geometry through an emergent GLOBAL READ instead: a
# learned position attention. Each output cell i (centered coord v_i) reads from ALL
# input cells j (centered coord v_j) weighted by a softmax over the coordinate
# similarity -beta*||M*v_i + t - v_j||^2, where M is a learned 2x2 projection, t a
# learned translation, and beta a learned temperature. As beta grows the softmax ->
# one-hot at the input cell nearest M*v_i + t, so an integer M reproduces a
# permutation EXACTLY (flip_h = [[1,0],[0,-1]], flip_v = [[-1,0],[0,1]], transpose =
# [[0,1],[1,0]]); a soft beta keeps a smooth ES gradient in M (the bilinear analog).
# The mechanism — a global similarity read, not a single hand-coded address — is the
# substrate B4's self-modifying memory builds on; the residual linear coord
# projection is what later milestones dissolve. param_dim is fixed (7), grid-size-
# independent. Like apply_operator this is an inherently gather/reduction kernel, so
# it is scalar-per-cell; the SIMD/FMA hot path (the weight-space ES update) is
# untouched and generic. Geometry-only — colour stays MLPMemory's job; one emergent
# memory covering BOTH geometry and colour is a future step (a single ES fit over the
# coupled 56-D memory was not honestly learnable without hand-staging the fit — see
# the journal's B3 entry; that retirement of OperatorMemory routes through B4).
comptime ATTN_M_OFF = 0  # 2x2 coordinate projection [[m00,m01],[m10,m11]]
comptime ATTN_T_OFF = 4  # translation (t_r, t_c)
comptime ATTN_BETA_OFF = 6  # temperature (raw; beta = raw^2 keeps it >= 0 w/o overflow)
comptime ATTN_DIM = 7
# beta = ATTN_BETA_SEED^2 ~ 2: a soft ~1-cell peak so identity reads mostly self and
# the gradient in M is non-zero (beta -> 0 gives a flat, M-independent mean; beta ->
# inf gives a flat plateau — moderate beta is where moving M moves the soft peak).
comptime ATTN_BETA_SEED = Float32(1.4)
comptime ATTN_BETA_SCALE = Float32(1.0)
# Softmax window half-width (see apply): the gather scans a (2W+1)^2 window
# centred on q instead of the whole grid. Grids <= W+1 wide are covered for any
# centre => synth results bit-identical; real-ARC grids get the ~3-5x cut.
comptime ATTN_WINDOW = 6


struct AttnGatherMemory(Memory):
    comptime Dom = GridDomain

    @staticmethod
    def param_dim() -> Int:
        return ATTN_DIM

    @staticmethod
    def seed(weights: UnsafePointer[Float32, MutAnyOrigin]):
        # Identity addressing: M = I, t = 0 -> q_i = v_i -> each cell reads (mostly)
        # itself, so an unfit memory is the identity and the ES departs from it.
        weights[ATTN_M_OFF + 0] = 1.0
        weights[ATTN_M_OFF + 1] = 0.0
        weights[ATTN_M_OFF + 2] = 0.0
        weights[ATTN_M_OFF + 3] = 1.0
        weights[ATTN_T_OFF + 0] = 0.0
        weights[ATTN_T_OFF + 1] = 0.0
        weights[ATTN_BETA_OFF] = ATTN_BETA_SEED

    @staticmethod
    def fill_scale(scale: UnsafePointer[Float32, MutAnyOrigin], n: Int):
        for i in range(n):
            scale[i] = 1.0
        # Temperature gets its own step (diagonal preconditioner) so it can sharpen
        # at a different rate than the geometry params.
        scale[ATTN_BETA_OFF] = ATTN_BETA_SCALE

    @staticmethod
    def apply(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        # Same-shape path: query grid == input grid. Delegates to the
        # output-shape-aware gather with out dims = in dims, so this is
        # BIT-IDENTICAL to the pre-shape-seam apply (the query centre, source
        # centre, projection and windowed softmax all reduce to the old code).
        Self.apply_shaped(weights, inp, inp.rows, inp.cols, dst)

    # Output-shape-aware gather (the shape-change seam, Vision A / Next #1). The
    # QUERY grid is (out_rows, out_cols) centred on the OUTPUT extent, while the
    # gather still reads the INPUT grid (inp.rows/cols) centred on the INPUT
    # extent — the learned projection q = M*v_out + t maps an output coordinate
    # into the input's centred frame. Decoupling query size from source size is
    # the whole change: M = I, t = 0 reads the centred input (a centred crop),
    # M = sI a subsample by s, M = ±perm a flip/transpose within the resize.
    # For out == in this is exactly the same-shape gather above.
    @staticmethod
    def apply_shaped(
        weights: UnsafePointer[Float32, MutAnyOrigin],
        inp: ArcGrid,
        out_rows: Int,
        out_cols: Int,
        dst: UnsafePointer[Float32, MutAnyOrigin],
    ):
        var m00 = weights[ATTN_M_OFF + 0]
        var m01 = weights[ATTN_M_OFF + 1]
        var m10 = weights[ATTN_M_OFF + 2]
        var m11 = weights[ATTN_M_OFF + 3]
        var t_r = weights[ATTN_T_OFF + 0]
        var t_c = weights[ATTN_T_OFF + 1]
        var beta_raw = weights[ATTN_BETA_OFF]
        var beta = beta_raw * beta_raw  # >= 0, no exp overflow risk
        # Source (gather) grid: the input's extent and centre.
        var rows = inp.rows
        var cols = inp.cols
        var cr = Float32(rows - 1) * Float32(0.5)
        var cc = Float32(cols - 1) * Float32(0.5)
        # Query grid: the OUTPUT extent and centre (== input's when out == in).
        var cr_out = Float32(out_rows - 1) * Float32(0.5)
        var cc_out = Float32(out_cols - 1) * Float32(0.5)

        for r in range(out_rows):
            var vr = Float32(r) - cr_out
            for c in range(out_cols):
                var vc = Float32(c) - cc_out
                # Read target q = M*v + t (where in coord space to gather from).
                var qr = fma(m00, vr, fma(m01, vc, t_r))
                var qc = fma(m10, vr, fma(m11, vc, t_c))

                # WINDOWED softmax (the at-scale enabler): restrict both passes
                # to a (2W+1)^2 window centred on q. The peak is always inside
                # (the window follows q, so projection/translation are
                # unaffected); only the far tail is truncated, and even at the
                # softest beta the wide-sigma ES visits (raw ~0.4 => beta~0.16)
                # a distance-7 cell weighs e^(-0.16*49) ~ 4e-4 — negligible.
                # For grids <= ATTN_WINDOW+1 wide the window spans the whole
                # grid for ANY centre, so every synth-scale result is
                # BIT-IDENTICAL to the full scan; on real ARC grids (up to
                # 30x30) this cuts the O(cells^2) apply by ~3-5x. The centre is
                # clamped into bounds FIRST, so the window always overlaps the
                # grid (z can never be 0).
                var ctr_r = Int(round(qr + cr))
                var ctr_c = Int(round(qc + cc))
                if ctr_r < 0:
                    ctr_r = 0
                if ctr_r > rows - 1:
                    ctr_r = rows - 1
                if ctr_c < 0:
                    ctr_c = 0
                if ctr_c > cols - 1:
                    ctr_c = cols - 1
                var r_lo = ctr_r - ATTN_WINDOW
                if r_lo < 0:
                    r_lo = 0
                var r_hi = ctr_r + ATTN_WINDOW
                if r_hi > rows - 1:
                    r_hi = rows - 1
                var c_lo = ctr_c - ATTN_WINDOW
                if c_lo < 0:
                    c_lo = 0
                var c_hi = ctr_c + ATTN_WINDOW
                if c_hi > cols - 1:
                    c_hi = cols - 1

                # Pass 1: max score over the window (numerical stability — the
                # subtracted max makes the largest exponent 0, so exp never overflows).
                var max_score = Float32(-1.0e30)
                for rj in range(r_lo, r_hi + 1):
                    var dvr = qr - (Float32(rj) - cr)
                    var dvr2 = dvr * dvr
                    for cj in range(c_lo, c_hi + 1):
                        var dvc = qc - (Float32(cj) - cc)
                        var score = -beta * (dvr2 + dvc * dvc)
                        if score > max_score:
                            max_score = score

                # Pass 2: softmax-weighted gather (streaming, no per-cell buffer).
                var z = Float32(0.0)
                var s = Float32(0.0)
                for rj in range(r_lo, r_hi + 1):
                    var dvr = qr - (Float32(rj) - cr)
                    var dvr2 = dvr * dvr
                    for cj in range(c_lo, c_hi + 1):
                        var dvc = qc - (Float32(cj) - cc)
                        var score = -beta * (dvr2 + dvc * dvc)
                        var w = exp(score - max_score)
                        z += w
                        s = fma(w, inp.data[rj * cols + cj], s)
                dst[r * out_cols + c] = s / z


# TOROIDAL output-shaped gather (the upscale/tiling family, Vision A / Next #1
# rung a). Same learned attention read as `apply_shaped` (`weights` is the same
# 7-slot attention block) with two changes that make MODULAR content rules
# expressible:
#
# - The source grid is a TORUS: the coordinate displacement per axis is wrapped
#   into (-extent/2, extent/2] (`d -= extent*round(d/extent)`), so a query past
#   the input's edge reads the input's periodic image — tiling's sawtooth
#   `out[r] = in[r mod rows]`, provably outside any single affine (M, t), is
#   the nearest wrapped cell of an affine map. Toroidal topology is a substrate
#   choice (precedent: the selfmod-grid memories' toroidal neighbourhoods), not
#   a task primitive: which modular rule (if any) is read is entirely in the
#   learned (M, t, trel, beta). For queries that stay in range (crop / flip /
#   subsample / upscale) the wrap touches only the far softmax tail — at any
#   sharp temperature the nearest source cell is unchanged.
# - `trel_r`/`trel_c` are EXTENT-RELATIVE translations: q += trel*extent. The
#   centred query/source frames leave tiling with a size-dependent phase
#   (n/2 for tile-2) that a constant t cannot cancel across the varying demo
#   sizes; a translation in units of the source extent absorbs it with one
#   size-independent parameter (trel = 1/2 for tile-2), fit by the ES exactly
#   like t.
# - `kr`/`kc` NORMALIZE the query by the (written, frozen) shape-rule slope:
#   q = M*(v_out/k) + ... — resize-as-identity. The content search must not
#   re-learn the scale the shape rule already knows: without this, a resize
#   family's M is 1/k and its exactness tolerance shrinks with the output
#   extent (measured: upscale-2's m11 needs ±0.045, far below the ES's
#   settling noise at the staircase-viable sigma — the fit reliably parked
#   just off the plateau). Normalized, M = I (the seed) IS the pure-resize
#   solution for every k, and all tolerances are size-free. k <= 0 (a written
#   constant-output rule) falls back to 1.
#
# The window scan is the wrapped analogue of `apply_shaped`'s: the span is
# capped at the torus period (scanning more would visit a source cell twice
# through its periodic images), the centre is wrapped into bounds, and indices
# wrap modularly. For extents <= 2*ATTN_WINDOW+1 every source cell is scanned,
# so synth-scale results have no window truncation at all.
def attn_gather_toroidal(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    trel_r: Float32,
    trel_c: Float32,
    kr: Float32,
    kc: Float32,
    inp: ArcGrid,
    out_rows: Int,
    out_cols: Int,
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    var m00 = weights[ATTN_M_OFF + 0]
    var m01 = weights[ATTN_M_OFF + 1]
    var m10 = weights[ATTN_M_OFF + 2]
    var m11 = weights[ATTN_M_OFF + 3]
    var t_r = weights[ATTN_T_OFF + 0]
    var t_c = weights[ATTN_T_OFF + 1]
    var beta_raw = weights[ATTN_BETA_OFF]
    var beta = beta_raw * beta_raw
    var rows = inp.rows
    var cols = inp.cols
    var frows = Float32(rows)
    var fcols = Float32(cols)
    var cr = Float32(rows - 1) * Float32(0.5)
    var cc = Float32(cols - 1) * Float32(0.5)
    var cr_out = Float32(out_rows - 1) * Float32(0.5)
    var cc_out = Float32(out_cols - 1) * Float32(0.5)

    # Query normalization by the shape-rule slope (see the header).
    var inv_kr = Float32(1.0)
    if kr > Float32(1.0e-3):
        inv_kr = 1.0 / kr
    var inv_kc = Float32(1.0)
    if kc > Float32(1.0e-3):
        inv_kc = 1.0 / kc

    # Window span, capped at the torus period per axis (see the header).
    var span_r = 2 * ATTN_WINDOW + 1
    if span_r > rows:
        span_r = rows
    var span_c = 2 * ATTN_WINDOW + 1
    if span_c > cols:
        span_c = cols

    for r in range(out_rows):
        var vr = (Float32(r) - cr_out) * inv_kr
        for c in range(out_cols):
            var vc = (Float32(c) - cc_out) * inv_kc
            # q = M*v + t + trel*extent, in the input's centred frame.
            var qr = fma(m00, vr, fma(m01, vc, t_r)) + trel_r * frows
            var qc = fma(m10, vr, fma(m11, vc, t_c)) + trel_c * fcols

            # Window start: centre the span on q's wrapped source index.
            var ctr_r = Int(round(qr + cr)) % rows
            if ctr_r < 0:
                ctr_r += rows
            var ctr_c = Int(round(qc + cc)) % cols
            if ctr_c < 0:
                ctr_c += cols
            var r0 = ctr_r - ATTN_WINDOW
            var c0 = ctr_c - ATTN_WINDOW

            # Pass 1: max score over the wrapped window (numerical stability).
            var max_score = Float32(-1.0e30)
            for kr in range(span_r):
                var rj = (r0 + kr) % rows
                if rj < 0:
                    rj += rows
                var dvr = qr - (Float32(rj) - cr)
                dvr -= frows * round(dvr / frows)
                var dvr2 = dvr * dvr
                for kc in range(span_c):
                    var cj = (c0 + kc) % cols
                    if cj < 0:
                        cj += cols
                    var dvc = qc - (Float32(cj) - cc)
                    dvc -= fcols * round(dvc / fcols)
                    var score = -beta * (dvr2 + dvc * dvc)
                    if score > max_score:
                        max_score = score

            # Pass 2: softmax-weighted gather (streaming, no per-cell buffer).
            var z = Float32(0.0)
            var s = Float32(0.0)
            for kr in range(span_r):
                var rj = (r0 + kr) % rows
                if rj < 0:
                    rj += rows
                var dvr = qr - (Float32(rj) - cr)
                dvr -= frows * round(dvr / frows)
                var dvr2 = dvr * dvr
                for kc in range(span_c):
                    var cj = (c0 + kc) % cols
                    if cj < 0:
                        cj += cols
                    var dvc = qc - (Float32(cj) - cc)
                    dvc -= fcols * round(dvc / fcols)
                    var score = -beta * (dvr2 + dvc * dvc)
                    var w = exp(score - max_score)
                    z += w
                    s = fma(w, inp.data[rj * cols + cj], s)
            dst[r * out_cols + c] = s / z


# Symmetric triangle fold of a CENTERED source coordinate `x` into the base
# grid's centered range [-extent/2, extent/2], via reflection at the grid's two
# EDGES (period 2*extent). This is the reflect analogue of the toroidal gather's
# `x -= extent*round(x/extent)` wrap: where the torus repeats the grid
# (sawtooth), reflection MIRRORS every other copy (triangle). Unlike the
# sawtooth the triangle is CONTINUOUS (slope +-1, no jumps), so the ES sees a
# clean gradient everywhere. Reads the kaleidoscope tiling `out[R+i]=in[R-1-i]`:
# the alternate-flipped mirror tiling the plain torus provably cannot express.
def _reflect_fold(x: Float32, extent: Float32) -> Float32:
    var half = extent * Float32(0.5)
    var period = extent + extent
    var u = x + half
    u -= period * floor(u / period)  # into [0, 2*extent)
    if u >= extent:
        u = period - u  # reflect the far half back
    return u - half


# REFLECT (mirror-tiling) output-shaped gather (Vision A / Next #1 Rung D). The
# twin of `attn_gather_toroidal`: the SAME learned attention read and the SAME
# centered query `q = M*v + t + trel*extent` (so M/t/trel/beta and the
# resize-as-identity query normalization all carry over unchanged, and the same
# `seed_periodic` seed — M=kI, trel=(k-1)/2 — is the pure mirror-tiling
# solution, verified: for a k-tiling that seed makes q the centered source index
# of the output cell, which `_reflect_fold` then folds into the base tile). The
# ONE change from the torus: the source is read through the symmetric reflection
# above instead of the periodic wrap, so odd tiles are mirrored. The window is a
# plain bounded scan over the base cells around the FOLDED query (no period
# doubling — the fold already brought q into the base range), same cost as
# `apply_shaped`.
def attn_gather_reflect(
    weights: UnsafePointer[Float32, MutAnyOrigin],
    trel_r: Float32,
    trel_c: Float32,
    kr: Float32,
    kc: Float32,
    inp: ArcGrid,
    out_rows: Int,
    out_cols: Int,
    dst: UnsafePointer[Float32, MutAnyOrigin],
):
    var m00 = weights[ATTN_M_OFF + 0]
    var m01 = weights[ATTN_M_OFF + 1]
    var m10 = weights[ATTN_M_OFF + 2]
    var m11 = weights[ATTN_M_OFF + 3]
    var t_r = weights[ATTN_T_OFF + 0]
    var t_c = weights[ATTN_T_OFF + 1]
    var beta_raw = weights[ATTN_BETA_OFF]
    var beta = beta_raw * beta_raw
    var rows = inp.rows
    var cols = inp.cols
    var frows = Float32(rows)
    var fcols = Float32(cols)
    var cr = Float32(rows - 1) * Float32(0.5)
    var cc = Float32(cols - 1) * Float32(0.5)
    var cr_out = Float32(out_rows - 1) * Float32(0.5)
    var cc_out = Float32(out_cols - 1) * Float32(0.5)

    # Query normalization by the shape-rule slope (see attn_gather_toroidal).
    var inv_kr = Float32(1.0)
    if kr > Float32(1.0e-3):
        inv_kr = 1.0 / kr
    var inv_kc = Float32(1.0)
    if kc > Float32(1.0e-3):
        inv_kc = 1.0 / kc

    for r in range(out_rows):
        var vr = (Float32(r) - cr_out) * inv_kr
        for c in range(out_cols):
            var vc = (Float32(c) - cc_out) * inv_kc
            # Same centered query as the torus, then fold (not wrap).
            var qr = fma(m00, vr, fma(m01, vc, t_r)) + trel_r * frows
            var qc = fma(m10, vr, fma(m11, vc, t_c)) + trel_c * fcols
            var fr = _reflect_fold(qr, frows)
            var fc = _reflect_fold(qc, fcols)

            # Bounded window centered on the folded query's base cell.
            var ctr_r = Int(round(fr + cr))
            if ctr_r < 0:
                ctr_r = 0
            if ctr_r > rows - 1:
                ctr_r = rows - 1
            var ctr_c = Int(round(fc + cc))
            if ctr_c < 0:
                ctr_c = 0
            if ctr_c > cols - 1:
                ctr_c = cols - 1
            var r_lo = ctr_r - ATTN_WINDOW
            if r_lo < 0:
                r_lo = 0
            var r_hi = ctr_r + ATTN_WINDOW
            if r_hi > rows - 1:
                r_hi = rows - 1
            var c_lo = ctr_c - ATTN_WINDOW
            if c_lo < 0:
                c_lo = 0
            var c_hi = ctr_c + ATTN_WINDOW
            if c_hi > cols - 1:
                c_hi = cols - 1

            # Pass 1: max score over the window (numerical stability).
            var max_score = Float32(-1.0e30)
            for rj in range(r_lo, r_hi + 1):
                var dvr = fr - (Float32(rj) - cr)
                var dvr2 = dvr * dvr
                for cj in range(c_lo, c_hi + 1):
                    var dvc = fc - (Float32(cj) - cc)
                    var score = -beta * (dvr2 + dvc * dvc)
                    if score > max_score:
                        max_score = score

            # Pass 2: softmax-weighted gather.
            var z = Float32(0.0)
            var s = Float32(0.0)
            for rj in range(r_lo, r_hi + 1):
                var dvr = fr - (Float32(rj) - cr)
                var dvr2 = dvr * dvr
                for cj in range(c_lo, c_hi + 1):
                    var dvc = fc - (Float32(cj) - cc)
                    var score = -beta * (dvr2 + dvc * dvc)
                    var w = exp(score - max_score)
                    z += w
                    s = fma(w, inp.data[rj * cols + cj], s)
            dst[r * out_cols + c] = s / z
